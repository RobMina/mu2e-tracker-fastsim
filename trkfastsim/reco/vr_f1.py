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


def _straw_axes(path, cond, table):
    """Straw axes for the crossings (strawaxis.straw_axes) from the dump's aligned wire ends + TrkAlignStraw."""
    from . import strawaxis
    c = uproot.open(path)["trkdump/conditions"].arrays(["sid", "wcalx", "wcaly", "wcalz", "whvx", "whvy", "whvz"],
                                                       library="np")
    sid = c["sid"][0].astype(np.int64)
    us = hitreco.unique_straw(sid)
    S = len(cond["straw"]["midx"])
    wc, wh = np.zeros((S, 3)), np.zeros((S, 3))
    wc[us] = np.stack([c["wcalx"][0], c["wcaly"][0], c["wcalz"][0]], 1)
    wh[us] = np.stack([c["whvx"][0], c["whvy"][0], c["whvz"][0]], 1)
    return strawaxis.straw_axes(cond["straw"], wc, wh, strawaxis.read_trkalignstraw(table))


def read_inputs(path, cond, settings, max_tracks=None, straw_axis=None):
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
            "kh_ind", "kh_ambig", "kh_active", "kh_rdoca", "kst_sid", "kst_active", "kst_wall", "kst_gas", "kst_radlen"]
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
    # stock reference DOCA per hit (TrkStrawHitSeed stores refDOCA * endSign; sign flipped as in reco-vr2)
    H["stock_rdoca"], _ = _pad(trk, flat("kh_rdoca").astype(np.float64), ntrk, np.nan)
    H["stock_rdoca"] = H["stock_rdoca"] * H["esign"] * -1.0
    # crossings: straw geometry from the conditions by unique straw index
    ns = ak.to_numpy(ak.num(b.kst_sid))
    ctrk = np.repeat(np.arange(ntrk), ns)
    us = hitreco.unique_straw(flat("kst_sid").astype(np.int64))
    stc = cond["straw"]
    if straw_axis:
        smid, sdir = _straw_axes(path, cond, straw_axis)
        mid, wdir = smid[us], sdir[us]
    else:
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


def run(path, outdir, conditions, settings="mc", max_tracks=None, chunk=512, material=True, straw_axis=None):
    import jax
    import jax.numpy as jnp
    cache = os.environ.get("F1_JAX_CACHE")
    if cache:
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    from . import kalman_jax as K, kkmaterial
    from .linefit_jax import DRIFT_KEYS, HIT_KEYS
    os.makedirs(outdir, exist_ok=True)
    cond = hitreco.load_conditions(conditions)
    t0 = time.perf_counter()
    x0, H, X, ref, drift, ntrk = read_inputs(path, cond, settings, max_tracks, straw_axis or os.environ.get("F1_STRAWAXIS"))
    t_read = time.perf_counter() - t0
    dj = {k: jnp.asarray(np.asarray(drift[k], np.float64)) for k in DRIFT_KEYS}
    ann = K.load_anns()
    stages = K.stock_stages()
    if not material:
        stages = dict(stages, sx_maxdoca=jnp.zeros_like(stages["sx_maxdoca"]))
    hkeys = HIT_KEYS + ("valid",)
    Hn = {k: np.asarray(H[k]) for k in hkeys}
    Xn = {k: X[k] for k in ("mid", "wdir", "valid", "hit")}
    if os.environ.get("F1_PATHS") == "stock":          # diagnostic: scattering from the stock KalSeed path lengths
        Xn["kst_wall"], Xn["kst_gas"] = X["kst_wall"], X["kst_gas"]
    if os.environ.get("F1_XACT") == "stock":           # diagnostic: crossing activity from the stock KalSeed
        Xn["stock_active"] = X["stock_active"] & X["valid"]
    if not material:
        Xn["valid"] = np.zeros_like(Xn["valid"])
    fixed = {"active": H["active"], "lr": H["lr"], "nulldv": np.ones_like(H["active"])}

    def one(x, h, c, fx, maxniter):
        cf, hist, good = K.fit_track(x, h, c, dj, stages, ann, fixed=fx, maxniter=maxniter)
        xt0, Ct0, _ = K.t0_piece(cf)
        return xt0, Ct0, hist[0], hist[1], hist[2], hist[3], good, cf["xe"], cf["Ce"]

    # two passes, exact: a vmapped while_loop runs every lane until the slowest converges, so one track at MaxNIter
    # makes the whole batch pay 11 iterations. Pass 1 caps at FAST_NITER; tracks that reached the cap in any stage
    # are refitted with the stock MaxNIter (10); every other track already stopped before the cap, so its result is
    # the same as with MaxNIter 10.
    FAST_NITER = int(os.environ.get("F1_FAST_NITER", "3"))
    f1 = jax.jit(jax.vmap(lambda x, h, c, fx: one(x, h, c, fx, FAST_NITER)))
    MAXN = int(os.environ.get("F1_MAXNITER", "10"))     # stock MaxNIter 10; larger only as a diagnostic
    f10 = jax.jit(jax.vmap(lambda x, h, c, fx: one(x, h, c, fx, MAXN)))

    def f(*args):
        """Pass 1 only; returns outputs and the chunk-local indices that reached the cap (refitted later, all
        chunks of a bucket shape together)."""
        t1 = time.perf_counter()
        o = f1(*args)
        jax.block_until_ready(o)
        stats["pass1_s"].append(time.perf_counter() - t1)
        it = np.diff(np.concatenate([np.zeros((o[5].shape[0], 1)), np.asarray(o[5])], 1), axis=1)
        return o, np.flatnonzero((it >= FAST_NITER).any(1))

    stats = {"redo": 0, "pass1_s": [], "pass2_s": []}
    # size buckets: tracks sorted by (hits, crossings) and padded per chunk to the next power of two (>= 16), so a
    # few long tracks do not pad every chunk to 94 hits / 212 crossings; one compile per bucket shape
    nh = Hn["valid"].sum(1)
    nx = Xn["valid"].sum(1)
    p2 = lambda n, lo: np.maximum(lo, 2 ** np.ceil(np.log2(np.maximum(n, 1)))).astype(int)
    Nf, Mf = Hn["valid"].shape[1], Xn["valid"].shape[1]
    NMAIN, MMAIN = int(os.environ.get("F1_MAIN_HITS", "32")), int(os.environ.get("F1_MAIN_XINGS", "64"))
    # three size classes: main, middle (2x), overflow (the file maximum)
    CLS = [(NMAIN, MMAIN), (min(Nf, 2 * NMAIN), min(Mf, 2 * MMAIN)), (Nf, Mf)]
    key = np.where((nh <= CLS[0][0]) & (nx <= CLS[0][1]), 0, np.where((nh <= CLS[1][0]) & (nx <= CLS[1][1]), 1, 2))
    order = np.lexsort((nx, nh, key))
    res_x = [None] * 9
    outs_idx, outs, t_fit, first, shapes, btr, pend = [], [], 0.0, None, {}, {}, {}
    # chunks never straddle two buckets; the last chunk of a bucket is padded to a power of two (>= 256) of tracks
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    bounds = [(b0, min(b0 + chunk, b1)) for b0, b1 in zip(starts, np.r_[starts[1:], ntrk]) for b0 in range(b0, b1, chunk)]
    done = 0
    for s0, s1 in bounds:
        idx = order[s0:s1]
        Nb, Mb = CLS[key[idx[0]]]
        n = len(idx)
        padn = (chunk if n == chunk else int(256 * np.ceil(n / 256))) - n          # multiples of 256
        pad = lambda a: np.concatenate([a, np.repeat(a[-1:], padn, 0)]) if padn else a
        args = (jnp.asarray(pad(x0[idx])), {k: jnp.asarray(pad(v[idx][:, :Nb])) for k, v in Hn.items()},
                {k: jnp.asarray(pad(v[idx][:, :Mb])) for k, v in Xn.items()},
                {k: jnp.asarray(pad(v[idx][:, :Nb])) for k, v in fixed.items()})
        tt = time.perf_counter()
        o, redo = f(*args)
        dt = time.perf_counter() - tt
        redo = redo[redo < n]
        if len(redo):
            pend.setdefault(int(key[idx[0]]), []).append((len(outs), redo, idx[redo]))
        new_shape = (Nb, Mb, n + padn) not in shapes
        shapes[(Nb, Mb, n + padn)] = 1
        btr[(Nb, Mb)] = btr.get((Nb, Mb), 0) + n
        if first is None:
            first = dt
        elif not new_shape:
            t_fit += dt
        o = [np.asarray(a)[:n].copy() for a in o]
        # per-effect pieces back to the full padded width (hits N, crossings M) for the analysis below
        for j in (7, 8):
            full = np.zeros((n, Nf + Mf) + o[j].shape[2:])
            full[:, :Nb] = o[j][:, :Nb]
            full[:, Nf:Nf + Mb] = o[j][:, Nb:]
            o[j] = full
        outs.append(o)
        outs_idx.append(idx)
        done += n
        print(f"vr2f1: {done}/{ntrk} tracks, bucket N={Nb} M={Mb} ({n} + {padn} pad), {dt:.2f} s{' (compile)' if new_shape else ''}",
              flush=True)
    # pass 2: every capped track refitted at MaxNIter 10 in one call at the widest shape needed (padded tracks repeat
    # the last one); results written back into each chunk's outputs
    stats["pass2_shape"] = []
    for cls, items in sorted(pend.items()):
        gidx = np.concatenate([it_[2] for it_ in items])
        nr = len(gidx)
        stats["redo"] += nr
        Nb, Mb = CLS[cls]
        npad = int(256 * np.ceil(nr / 256)) - nr
        sel = np.concatenate([gidx, np.repeat(gidx[-1:], npad)])
        args = (jnp.asarray(x0[sel]), {k: jnp.asarray(v[sel][:, :Nb]) for k, v in Hn.items()},
                {k: jnp.asarray(v[sel][:, :Mb]) for k, v in Xn.items()}, {k: jnp.asarray(v[sel][:, :Nb]) for k, v in fixed.items()})
        t2 = time.perf_counter()
        o2 = f10(*args)
        jax.block_until_ready(o2)
        stats["pass2_s"].append(round(time.perf_counter() - t2, 3))
        stats["pass2_shape"].append([Nb, Mb, nr, nr + npad])
        o2 = [np.asarray(a)[:nr] for a in o2]
        k = 0
        for io, redo, _ in items:
            for j in range(9):
                v = o2[j][k:k + len(redo)]
                if j in (7, 8):
                    full = np.zeros((len(redo), Nf + Mf) + v.shape[2:])
                    full[:, :Nb] = v[:, :Nb]
                    full[:, Nf:Nf + Mb] = v[:, Nb:]
                    v = full
                outs[io][j][redo] = v
            k += len(redo)
    inv = np.argsort(np.concatenate(outs_idx))
    timing = {"pass1_s": [round(x, 3) for x in stats["pass1_s"]], "pass2_s": stats["pass2_s"],
              "pass2_shape": stats.get("pass2_shape"), "total_fit_s": round(sum(stats["pass1_s"]) + sum(stats["pass2_s"]), 3),
              "refit_with_maxniter10": stats["redo"], "fast_niter": FAST_NITER, "bucket_tracks": {f"{a}x{b}": c for (a, b), c in btr.items()}, "n_compiles": len(shapes),
              "warm_fit_s": t_fit}
    xf, Cf, stat, chi2, ndof, niter, good, xe, Ce = (np.concatenate([o[i] for o in outs])[inv] for i in range(9))
    # KinematicLine parameters on the piece holding the t0 point (KalSeed::t0Segment / PiecewiseTrajectory::t0): the
    # POCA to the z axis, s1 along the track; the piece is the smoothed state at the last effect before s1
    N = H["valid"].shape[1]
    res_piece = {}
    kl = linefit.to_kinematic_line(xf)
    Jk = np.zeros((ntrk, 5, 5))
    eps = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-3])
    klv = lambda xx: np.stack([linefit.to_kinematic_line(xx)[k] for k in KL], 1)
    for k in range(5):
        dx = np.zeros(5); dx[k] = eps[k]
        Jk[:, :, k] = (klv(xf + dx) - klv(xf - dx)) / (2 * eps[k])
    Ck = np.einsum("tij,tjk,tlk->til", Jk, Cf, Jk)
    res = {"input": path, "n_tracks": ntrk, "material": material, "straw_axis": straw_axis or os.environ.get("F1_STRAWAXIS"), "read_s": t_read, "first_chunk_s": first,
           "timing": timing, "backend": jax.default_backend(),
           **res_piece, "final_status_counts": np.bincount(stat[:, -1], minlength=8).tolist(), "good_frac": float(good.mean()),
           "iterations_per_stage_q50": np.median(np.diff(np.concatenate([np.zeros((ntrk, 1)), niter], 1), axis=1), 0).tolist(),
           "iterations_per_stage_q99": np.quantile(np.diff(np.concatenate([np.zeros((ntrk, 1)), niter], 1), axis=1), .99, 0).tolist(),
           "iterations_per_stage_max": np.diff(np.concatenate([np.zeros((ntrk, 1)), niter], 1), axis=1).max(0).tolist()}
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
    pulls = {}
    for k in KL:
        dv = kl[k] - ref["kl_" + k]
        if k == "phi0":
            dv = (dv + np.pi) % (2 * np.pi) - np.pi
        pulls[k] = dv / np.where(ref["kl_" + k + "err"] > 0, ref["kl_" + k + "err"], np.nan)
    np.savez(os.path.join(outdir, "pulls.npz"), good=good, cpn_f1=chi2[:, -1] / np.maximum(ndof[:, -1], 1),
             cpn_stock=ref["kl_chi2"] / np.maximum(ref["kl_ndof"], 1), nh=H["valid"].sum(1), nact=H["active"].sum(1),
             nx=X["valid"].sum(1), nxact=X["stock_active"].sum(1), **pulls,
             **{"ref_" + k: ref["kl_" + k] for k in KL}, **{"f1_" + k: kl[k] for k in KL})
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
    doca_all = doca
    nsig_final = 1.0
    wall, gas = (np.asarray(a) for a in kkmaterial.path_lengths(doca, dvar, dirdot, nsig_final))
    m = xv & X["stock_active"]
    # hit DOCA at the final F1 pieces vs the stock reference DOCA (geometry and trajectory consistency, no material)
    from .linefit_jax import _resid_one
    dj_np = dj
    def hdoca(x, h):
        return jax.vmap(lambda xx, hi: _resid_one(xx, {k: v[None] for k, v in hi.items()}, dj_np, None, True)[3][0])(x, h)
    hk = [k for k in HIT_KEYS if k in Hn]
    dd = []
    for s0 in range(0, ntrk, 2048):
        sl = slice(s0, min(s0 + 2048, ntrk))
        dd.append(np.asarray(jax.vmap(hdoca)(jnp2.asarray(xe[sl, :N]), {k: jnp2.asarray(Hn[k][sl]) for k in hk})))
    hd = np.concatenate(dd)
    hm = H["valid"] & H["active"] & ok[:, None] & np.isfinite(H["stock_rdoca"])
    ddoca = (hd - H["stock_rdoca"])[hm]
    res["hit_doca_vs_stock_ref"] = {"n": int(hm.sum()), "q16_50_84_mm": np.quantile(ddoca, [.16, .5, .84]).tolist(),
                                    "abs_q50_q99_mm": np.quantile(np.abs(ddoca), [.5, .99]).tolist()}
    # crossing activity: ours (KKStrawXing rule at the final pieces: |DOCA| < 2.5 mm or the hit active) vs stock flag
    hact_x = np.zeros_like(X["valid"])
    hhx = (X["hit"] >= 0) & X["valid"]
    hact_x[hhx] = np.take_along_axis(H["active"], np.maximum(X["hit"], 0), 1)[hhx]
    allx = X["valid"] & ok[:, None]
    ours = (np.abs(doca_all) < 2.5) | hact_x
    sa = X["stock_active"]
    # in bounds (between the first and last active hit along the track), positions on the t0 piece line
    def s_along(P):
        a0, b0, a1, b1 = (xf[:, i] for i in range(4))
        p = np.stack([a0, np.zeros_like(a0), b0], 1)
        u = np.stack([a1, -np.ones_like(a1), b1], 1)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        return np.einsum("tni,ti->tn", P - p[:, None], u)
    sh_ = np.where(H["valid"] & H["active"], s_along(H["mid"]), np.nan)
    sx_ = s_along(X["mid"])
    inb_x = (sx_ > np.nanmin(sh_, 1)[:, None]) & (sx_ < np.nanmax(sh_, 1)[:, None])
    mis = (ours != sa) & allx
    res["crossing_activity_mismatch_inbounds"] = {"mismatches": int(mis.sum()), "in_bounds": int((mis & inb_x).sum()),
                                                  "in_bounds_doca_q16_50_84": np.quantile(np.abs(doca_all)[mis & inb_x], [.16, .5, .84]).tolist()
                                                  if (mis & inb_x).any() else None}
    so = ~ours & sa & allx
    res["crossing_activity_stock_only_split"] = {"with_hit": int((so & hhx).sum()), "no_hit": int((so & ~hhx).sum()),
                                                 "with_inactive_hit": int((so & hhx & ~hact_x).sum())}
    res["crossing_activity"] = {"n": int(allx.sum()), "agree": float(np.mean((ours == sa)[allx])),
                                "ours_only": int((ours & ~sa & allx).sum()), "stock_only": int((~ours & sa & allx).sum()),
                                "stock_only_doca_q16_50_84": np.quantile(np.abs(doca_all)[~ours & sa & allx], [.16, .5, .84]).tolist()
                                if (~ours & sa & allx).any() else None,
                                "ours_only_doca_q16_50_84": np.quantile(np.abs(doca_all)[ours & ~sa & allx], [.16, .5, .84]).tolist()
                                if (ours & ~sa & allx).any() else None}
    np.savez(os.path.join(outdir, "xings.npz"), doca=doca[m], dvar=dvar[m], dirdot=dirdot[m], wall=wall[m], gas=gas[m],
             kst_wall=X["kst_wall"][m], kst_gas=X["kst_gas"][m], kst_radlen=X["kst_radlen"][m],
             hashit=(X["hit"] >= 0)[m], trk=np.nonzero(m)[0])
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
    # diagnostic: the neighbouring pieces (one crossing earlier / later along the track)
    isx = (order >= N) & np.isfinite(s_sorted)
    pos = np.arange(order.shape[1])[None]
    before = np.where(isx & (pos <= k[:, None]), pos, -1).max(1)          # last crossing at or before k
    after = np.where(isx & (pos > k[:, None]), pos, 10 ** 9).min(1)       # first crossing after k
    prev_e = order[np.arange(T), np.clip(before - 1, 0, None)]
    next_e = order[np.arange(T), np.clip(np.where(after < 10 ** 9, after, k), 0, order.shape[1] - 1)]
    _t0_piece.neighbours = (np.where(before > 0, prev_e, cur), np.where(after < 10 ** 9, next_e, cur))
    return xe[np.arange(T), cur], Ce[np.arange(T), cur], cur


def _put(a, b, idx):
    a = a.copy()
    a[idx] = b
    return a
