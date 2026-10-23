"""T-VR2 for F1 (kalman_jax) on stock inputs: stock LineFinder seed, the stock fit's hit list and final hit states
(active, left/right), and the stock straw crossings (KalSeed::straws, kst_*), so only the fit differs. Per-track
comparison with stock KKLine (FitOK tracks of single-seed events), in units of the stock errors; plus the material
check: our wall/gas paths and X0 fraction at the final F1 parameters vs kst_wall/kst_gas/kst_radlen.

Hit states are fixed in every stage (kalman_jax.fit_track(fixed=...)); null hits use the NullDriftVar variance (the
final CHDRIFTEXT flag). The schedule (temperatures, crossing settings, status tests) is the stock one.
"""
import json
import os
import time

import awkward as ak
import numpy as np
import uproot

from . import hitreco, linefit

FITOK = 1 << 15
KL = ("d0", "phi0", "z0", "theta", "t0")


def _pad(rows_trk, vals, ntrk, fill):
    """Flat values with a track index -> [ntrk, L] padded (L = max count, at least 1)."""
    cnt = np.bincount(rows_trk, minlength=ntrk)
    L = max(int(cnt.max()) if len(cnt) else 1, 1)
    pos = np.arange(len(rows_trk)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    out = np.full((ntrk, L) + vals.shape[1:], fill, dtype=vals.dtype)
    out[rows_trk, pos] = vals
    return out, cnt


def read_inputs(path, cond, settings, max_tracks=None):
    """One dump file -> per-track seed, padded hits (linefit.build_hits) with stock states, padded crossings, stock
    reference values. Only events with one seed and one KKLine fit, and only FitOK fits."""
    d = hitreco.read_digis([path])
    sh = hitreco.straw_hits(d, cond, settings)
    from .cli import _ref, _drift_table
    rs = _ref([path], "s_")
    for c in ("time", "et0", "et1", "edep", "flag"):
        sh[c] = rs[c].astype(sh[c].dtype)
    drift = _drift_table(path)
    cols = ["sd_a0", "sd_b0", "sd_a1", "sd_b1", "sd_t0", "kl_status", "kl_chi2", "kl_ndof", "kl_nactive",
            "kh_ind", "kh_ambig", "kh_active", "kst_sid", "kst_active", "kst_wall", "kst_gas", "kst_radlen"]
    cols += ["kl_" + k for k in KL] + ["kl_" + k + "err" for k in KL]
    a = uproot.open(path)["trkdump/events"].arrays(cols)
    ev0 = np.searchsorted(sh["ev"], np.arange(d["evkey"].shape[0] + 1), side="left")
    one = (ak.num(a.kl_d0) == 1) & (ak.num(a.sd_a0) == 1)
    st = ak.fill_none(ak.firsts(a.kl_status), 0)
    sel = ak.to_numpy(one & ((np.asarray(st, dtype=np.int64) & FITOK) != 0))
    ie = np.flatnonzero(sel)
    if max_tracks:
        ie = ie[:max_tracks]
    b = a[ie]
    ntrk = len(b)
    flat = lambda k: ak.to_numpy(ak.flatten(b[k]))
    ref = {k: flat(k) for k in cols if k.startswith(("kl_", "sd_"))}
    nh = ak.to_numpy(ak.num(b.kh_ind))
    trk = np.repeat(np.arange(ntrk), nh)
    shi = flat("kh_ind") + np.repeat(ev0[ie], nh)
    amb = flat("kh_ambig")
    lr = np.where(np.abs(amb) == 1, amb, 0).astype(np.float64)
    act = flat("kh_active").astype(bool)
    H = linefit.build_hits(trk, shi, lr, act, sh, cond, ntrk)
    H["valid"] = H["sh"] >= 0
    # crossings: straw geometry from the conditions by unique straw index
    ns = ak.to_numpy(ak.num(b.kst_sid))
    ctrk = np.repeat(np.arange(ntrk), ns)
    us = hitreco.unique_straw(flat("kst_sid").astype(np.int64))
    stc = cond["straw"]
    mid = np.stack([stc["midx"][us], stc["midy"][us], stc["midz"][us]], 1)
    wdir = np.stack([stc["dirx"][us], stc["diry"][us], stc["dirz"][us]], 1)
    X = {}
    X["mid"], cnt = _pad(ctrk, mid, ntrk, 0.0)
    X["wdir"], _ = _pad(ctrk, wdir, ntrk, 0.0)
    X["wdir"][..., 0] = np.where(np.arange(X["wdir"].shape[1])[None] < cnt[:, None], X["wdir"][..., 0], 1.0)
    X["valid"], _ = _pad(ctrk, np.ones(len(ctrk), bool), ntrk, False)
    X["stock_active"], _ = _pad(ctrk, flat("kst_active").astype(bool), ntrk, False)
    for k in ("wall", "gas", "radlen"):
        X["kst_" + k], _ = _pad(ctrk, flat("kst_" + k).astype(np.float64), ntrk, 0.0)
    # crossing -> hit index on the same straw (or -1)
    hsid = np.where(H["valid"], hitreco.unique_straw(sh["sid"][np.maximum(H["sh"], 0)].astype(np.int64)), -1)
    xs = np.where(X["valid"], 0, -2)
    xs_us, _ = _pad(ctrk, us, ntrk, -2)
    eq = xs_us[:, :, None] == hsid[:, None, :]
    X["hit"] = np.where(eq.any(-1), eq.argmax(-1), -1)
    x0 = linefit.seed_params(*(ref[k] for k in ("sd_a0", "sd_b0", "sd_a1", "sd_b1", "sd_t0")))
    return x0, H, X, ref, drift, int(len(ie))


def _jnp_tree(dct, keys=None):
    import jax.numpy as jnp
    return {k: jnp.asarray(v) for k, v in dct.items() if keys is None or k in keys}


def run(path, outdir, conditions, settings="mc", max_tracks=None, chunk=512, material=True):
    import jax
    import jax.numpy as jnp
    from . import kalman_jax as K, kkmaterial
    from .linefit_jax import DRIFT_KEYS, HIT_KEYS
    os.makedirs(outdir, exist_ok=True)
    cond = hitreco.load_conditions(conditions)
    t0 = time.perf_counter()
    x0, H, X, ref, drift, ntrk = read_inputs(path, cond, settings, max_tracks)
    t_read = time.perf_counter() - t0
    dj = {k: jnp.asarray(np.asarray(drift[k], np.float64)) for k in DRIFT_KEYS}
    ann = K.load_anns()
    stages = K.stock_stages()
    if not material:
        stages = dict(stages, sx_maxdoca=jnp.zeros_like(stages["sx_maxdoca"]))
    hkeys = HIT_KEYS + ("valid",)
    Hn = {k: np.asarray(H[k]) for k in hkeys}
    Xn = {k: X[k] for k in ("mid", "wdir", "valid", "hit")}
    if not material:
        Xn["valid"] = np.zeros_like(Xn["valid"])
    fixed = {"active": H["active"], "lr": H["lr"], "nulldv": np.ones_like(H["active"])}

    def one(x, h, c, fx):
        cf, hist, good = K.fit_track(x, h, c, dj, stages, ann, fixed=fx)
        xing = K.straw_poca_batch(cf["xe"][h["valid"].shape[0]:], c)
        return cf["xe"][0], cf["Ce"][0], hist[0], hist[1], hist[2], hist[3], good, cf["xe"], cf["Ce"]

    f = jax.jit(jax.vmap(one))
    outs, t_fit, first = [], 0.0, None
    for s0 in range(0, ntrk, chunk):
        sl = slice(s0, min(s0 + chunk, ntrk))
        n = sl.stop - sl.start
        padn = chunk - n if ntrk > chunk else 0
        pad = lambda a: np.concatenate([a, np.repeat(a[-1:], padn, 0)]) if padn else a
        args = (jnp.asarray(pad(x0[sl])), {k: jnp.asarray(pad(v[sl])) for k, v in Hn.items()},
                {k: jnp.asarray(pad(v[sl])) for k, v in Xn.items()}, {k: jnp.asarray(pad(v[sl])) for k, v in fixed.items()})
        tt = time.perf_counter()
        o = f(*args)
        jax.block_until_ready(o)
        dt = time.perf_counter() - tt
        if first is None:
            first = dt
        else:
            t_fit += dt
        outs.append([np.asarray(a)[:n] for a in o])
        print(f"vr2f1: {sl.stop}/{ntrk} tracks, chunk {dt:.1f} s", flush=True)
    xf, Cf, stat, chi2, ndof, niter, good, xe, Ce = (np.concatenate([o[i] for o in outs]) for i in range(9))
    # KinematicLine parameters on the piece holding the t0 point (KalSeed::t0Segment / PiecewiseTrajectory::t0): the
    # POCA to the z axis, s1 along the track; the piece is the smoothed state at the last effect before s1
    N = H["valid"].shape[1]
    xf, Cf, ipiece = _t0_piece(xe, Ce, X, H, N)
    res_piece = {"t0_piece_is_first_effect_frac": float(np.mean(ipiece == 0))}
    kl = linefit.to_kinematic_line(xf)
    Jk = np.zeros((ntrk, 5, 5))
    eps = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-3])
    klv = lambda xx: np.stack([linefit.to_kinematic_line(xx)[k] for k in KL], 1)
    for k in range(5):
        dx = np.zeros(5); dx[k] = eps[k]
        Jk[:, :, k] = (klv(xf + dx) - klv(xf - dx)) / (2 * eps[k])
    Ck = np.einsum("tij,tjk,tlk->til", Jk, Cf, Jk)
    res = {"input": path, "n_tracks": ntrk, "material": material, "read_s": t_read, "first_chunk_s": first,
           "warm_tracks_per_s": (ntrk - min(chunk, ntrk)) / t_fit if t_fit > 0 else None, "backend": jax.default_backend(),
           **res_piece, "final_status_counts": np.bincount(stat[:, -1], minlength=8).tolist(), "good_frac": float(good.mean()),
           "iterations_per_stage_q50": np.median(np.diff(np.concatenate([np.zeros((ntrk, 1)), niter], 1), axis=1), 0).tolist()}
    ok = good
    for i, k in enumerate(KL):
        dv = kl[k] - ref["kl_" + k]
        if k == "phi0":
            dv = (dv + np.pi) % (2 * np.pi) - np.pi
        pull = dv / np.where(ref["kl_" + k + "err"] > 0, ref["kl_" + k + "err"], np.nan)
        m = ok & np.isfinite(pull)
        res[f"{k}_dsigma"] = {"q50": float(np.median(pull[m])),
                              "q16_84": [float(np.quantile(pull[m], .16)), float(np.quantile(pull[m], .84))],
                              "frac_within_0p1": float(np.mean(np.abs(pull[m]) < 0.1)),
                              "frac_within_1": float(np.mean(np.abs(pull[m]) < 1.0))}
        res[f"{k}_err_ratio_q50"] = float(np.median(np.sqrt(Ck[ok, i, i]) / ref["kl_" + k + "err"][ok]))
    cpn = chi2[:, -1] / np.maximum(ndof[:, -1], 1)
    res["chi2ndf_q50"] = {"stock": float(np.median(ref["kl_chi2"] / np.maximum(ref["kl_ndof"], 1))),
                          "f1": float(np.median(cpn[ok]))}
    res["ndof_agree_frac"] = float(np.mean(ndof[ok, -1] == ref["kl_ndof"][ok]))
    # material check at the final per-crossing pieces: DOCA and DOCA variance -> paths, X0 fraction
    xi, xv = X["valid"], X["valid"] & ok[:, None]
    import jax.numpy as jnp2
    xc, Cc = xe[:, N:], Ce[:, N:]
    dd = []
    for s0 in range(0, ntrk, 2048):
        sl = slice(s0, min(s0 + 2048, ntrk))
        o = jax.vmap(lambda x, C, m, w: jax.vmap(lambda xx, CC, mm, ww: tuple(
            v[0] for v in _xing_geom(K, xx, CC, mm, ww)))(x, C, m, w))(
            jnp2.asarray(xc[sl]), jnp2.asarray(Cc[sl]), jnp2.asarray(X["mid"][sl]), jnp2.asarray(X["wdir"][sl]))
        dd.append([np.asarray(a) for a in o])
    doca, dvar, dirdot = (np.concatenate([o[i] for o in dd]) for i in range(3))
    nsig_final = 1.0
    wall, gas = (np.asarray(a) for a in kkmaterial.path_lengths(doca, dvar, dirdot, nsig_final))
    m = xv & X["stock_active"]
    rw = (wall[m] - X["kst_wall"][m])
    rg = (gas[m] - X["kst_gas"][m])
    res["material_check"] = {"n_crossings": int(m.sum()),
                             "wall_diff_mm_q16_50_84": np.quantile(rw, [.16, .5, .84]).tolist(),
                             "gas_diff_mm_q16_50_84": np.quantile(rg, [.16, .5, .84]).tolist(),
                             "kst_wall_q50": float(np.median(X["kst_wall"][m])), "kst_gas_q50": float(np.median(X["kst_gas"][m])),
                             "radlen_over_paths_q50": _x0_ratio(X, m)}
    with open(os.path.join(outdir, "vr2f1.json"), "w") as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps(res, indent=1))
    return res


def _xing_geom(K, x, C, mid, wdir):
    import jax
    st, doca, c = K.straw_poca(x, mid[None], wdir[None])
    J = jax.jacfwd(lambda xx: K.straw_poca(xx, mid[None], wdir[None])[1])(x)
    return doca, (J @ C @ J.T)[0], c


def _x0_ratio(X, m):
    """kst_radlen / (wall/X0_wall + gas/X0_gas) using the stock paths: checks the X0 values behind kst_radlen
    (X0 from MatEnv; reported only as the ratio to Mylar 28.54 cm and Ar/CO2 ~ 125 m nominal)."""
    nominal = X["kst_wall"][m] / 285.4 + X["kst_gas"][m] / 125000.0
    r = X["kst_radlen"][m] / np.where(nominal > 0, nominal, np.nan)
    return float(np.nanmedian(r))


def _t0_piece(xe, Ce, X, H, N):
    """Per track: the piece (effect index) holding the POCA to the z axis, found self-consistently as
    PiecewiseTrajectory::t0 (start from the middle effect, move to the piece holding that piece's z-axis POCA)."""
    def s_of(x, P):
        a0, b0, a1, b1, _ = x.T
        p = np.stack([a0, np.zeros_like(a0), b0], -1)
        d = np.stack([a1, -np.ones_like(a1), b1], -1)
        u = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return np.einsum("...i,...i->...", P - p, u), p, u
    T, E = xe.shape[:2]
    # effect positions: hits at the wire POCA of their own piece, crossings likewise (+ wire radius is negligible here)
    mids = np.concatenate([H["mid"], X["mid"]], 1)
    wd = np.concatenate([H["wdir"], X["wdir"]], 1)
    valid = np.concatenate([H["valid"], X["valid"]], 1)
    a0, b0, a1, b1 = (xe[..., i] for i in range(4))
    p = np.stack([a0, np.zeros_like(a0), b0], -1)
    d = np.stack([a1, -np.ones_like(a1), b1], -1)
    u = d / np.linalg.norm(d, axis=-1, keepdims=True)
    c = np.einsum("tei,tei->te", u, wd)
    dl = p - mids
    du, dw = np.einsum("tei,tei->te", dl, u), np.einsum("tei,tei->te", dl, wd)
    s_eff = np.where(valid, (dw * c - du) / np.maximum(1 - c * c, 1e-12), np.inf)
    order = np.argsort(s_eff, 1)
    s_sorted = np.take_along_axis(s_eff, order, 1)
    nval = valid.sum(1)
    cur = order[np.arange(T), nval // 2]
    for _ in range(10):
        x = xe[np.arange(T), cur]
        uz = (x[:, 3] / np.sqrt(x[:, 2] ** 2 + 1 + x[:, 3] ** 2))
        pp = np.stack([x[:, 0], np.zeros(T), x[:, 1]], 1)
        uu = np.stack([x[:, 2], -np.ones(T), x[:, 3]], 1); uu /= np.linalg.norm(uu, axis=1, keepdims=True)
        s1 = (pp[:, 2] * uz - np.sum(pp * uu, 1)) / np.maximum(1 - uz * uz, 1e-12)
        k = np.clip(np.sum(s_sorted <= s1[:, None], 1) - 1, 0, None)
        cur = order[np.arange(T), k]
    return xe[np.arange(T), cur], Ce[np.arange(T), cur], cur
