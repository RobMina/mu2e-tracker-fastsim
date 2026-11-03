"""Fast track table: the per-track contract between a tracker fit (stock KKLine from the dump, F2) and the
EventNtuple writer (`ntuple.py`).

One table per StrawHitTableDumper file. Arrays (numpy, saved with `save`/`load` as .npz):

  events  E   run, subrun, event (int64), pbt (float32): every event of the input file, in file order
  tracks  T   ev (event row), x [T, 6] = (a0, b0, a1, b1, t0, mom) (line through (a0, 0, b0), direction (a1, -1, b1),
              t0 at that point, mom MeV/c), cov [T, 6, 6] (NaN where the fit has no such term), chi2, ndof,
              status (EventNtuple trk.status: 1 converged, 2 OK not converged, -1 failed), goodfit (0/1),
              momfit (1 when mom is a fitted parameter, 0 when fixed)
  hits    [T, N] padded, `hvalid` marks real slots: the linefit.build_hits geometry (mid, wdir, hlen, esign, vsig,
              tmeas, totdt, uvar, wdist, edep, sh) plus sid, flag, et0, et1, tot0, tot1, eend and the final state
              (EventNtuple trkhits.state: -2 inactive, -1/+1 drift left/right, 0 null; stock tables keep -3)
  xings   [T, M] padded, `xvalid` marks real slots (absent when the fit has no material model): sid, active,
              wall, gas, wire, radlen, doca, docavar, dirdot (NaN where the source does not provide it)
  drift       the dump's drift table (dt, rdrift, dvel, sderr, uderr) for rdrift/dvel/sderr at the fit dT
  meta        JSON: fit type, stock conditioning, settings, counts, source files

Producers: `from_stock` (the dump's own KKLine fits: the writer-closure reference), `from_f2` (the reco-vr3 chain
from digis: hit reco, SimpleTimeCluster, LineFinder, F2 with its own hit states).
"""
import json

import awkward as ak
import numpy as np
import uproot

from . import hitreco, linefit

VERSION = "fasttracks-v1"
KALMAN_OK, KALMAN_CONVERGED, FITOK = 1 << 5, 1 << 12, 1 << 15      # TrkFitFlag bits
STATE_INACTIVE, STATE_NULL = -2, 0
HIT_KEYS = ("mid", "wdir", "hlen", "esign", "vsig", "tmeas", "totdt", "uvar", "wdist", "edep", "sh")
SH_EXTRA = ("sid", "flag", "et0", "et1", "tot0", "tot1", "eend")
XING_KEYS = ("sid", "active", "wall", "gas", "wire", "radlen", "doca", "docavar", "dirdot")


def _events(path):
    a = uproot.open(path)["trkdump/events"].arrays(["run", "subrun", "event", "pbt"], library="np")
    return {"run": a["run"].astype(np.int64), "subrun": a["subrun"].astype(np.int64),
            "event": a["event"].astype(np.int64), "pbt": a["pbt"].astype(np.float32)}


def _state(active, lr):
    return np.where(active, np.where(lr > 0, 1, np.where(lr < 0, -1, STATE_NULL)), STATE_INACTIVE).astype(np.int32)


def _hits(trk, shi, lr, active, sh, cond, ntrk, state=None):
    """Padded hit block for flat (track, global makeSH index, lr, active[, state]) lists, in list order per track."""
    H = linefit.build_hits(trk, shi, lr, active, sh, cond, ntrk)
    out = {k: H[k] for k in HIT_KEYS}
    out["hvalid"] = H["sh"] >= 0
    idx = np.maximum(H["sh"], 0)
    for k in SH_EXTRA:
        out[k] = np.where(out["hvalid"], np.asarray(sh[k])[idx], 0)
    out["active"] = H["active"] & out["hvalid"]
    out["lr"] = H["lr"]
    if state is None:
        out["state"] = np.where(out["hvalid"], _state(out["active"], H["lr"]), STATE_INACTIVE).astype(np.int32)
    else:
        o = np.lexsort((np.arange(len(trk)), trk))
        cnt = np.bincount(trk, minlength=ntrk)
        pos = np.arange(len(trk)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        st = np.full(H["sh"].shape, STATE_INACTIVE, np.int32)
        st[trk[o], pos] = np.asarray(state)[o]
        out["state"] = st
    return out


def _pad(trk, vals, ntrk, fill):
    cnt = np.bincount(trk, minlength=ntrk)
    L = max(int(cnt.max()) if len(cnt) else 1, 1)
    pos = np.arange(len(trk)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    out = np.full((ntrk, L) + np.asarray(vals).shape[1:], fill, dtype=np.asarray(vals).dtype)
    out[trk, pos] = vals
    return out


def _cov_from_kl_errors(kl, x):
    """Diagonal KinematicLine errors -> covariance in (a0, b0, a1, b1, t0, mom) through the numerical Jacobian of
    from_kinematic_line (correlations are not in the dump, so this is an approximation)."""
    names = ("d0", "phi0", "z0", "theta", "t0")
    p = np.stack([kl[k] for k in names], 1).astype(np.float64)
    J = np.zeros((len(p), 5, 5))
    eps = np.array([1e-3, 1e-6, 1e-3, 1e-6, 1e-3])
    for i in range(5):
        dp = np.zeros(5)
        dp[i] = eps[i]
        hi = linefit.from_kinematic_line(*(p + dp).T)
        lo = linefit.from_kinematic_line(*(p - dp).T)
        J[:, :, i] = (hi - lo) / (2 * eps[i])
    V = np.zeros((len(p), 5, 5))
    for i, k in enumerate(names):
        V[:, i, i] = np.asarray(kl[k + "err"], np.float64) ** 2
    C = np.full((len(p), 6, 6), np.nan)
    C[:, :5, :5] = np.einsum("tij,tjk,tlk->til", J, V, J)
    C[:, 5, :5] = C[:, :5, 5] = 0.0
    C[:, 5, 5] = np.asarray(kl["momerr"], np.float64) ** 2
    return C


def _flat(a, k):
    return ak.to_numpy(ak.flatten(a[k]))


def from_stock(path, cond, settings="mc"):
    """The dump's own KKLine fits (every saved KalSeed of every event): stock hits and states, stock crossings."""
    from .cli import _ref, _drift_table
    d = hitreco.read_digis([path])
    sh = hitreco.straw_hits(d, cond, settings)
    rs = _ref([path], "s_")
    for c in ("time", "et0", "et1", "edep", "flag"):
        sh[c] = rs[c].astype(sh[c].dtype)
    cols = ["kl_" + k for k in ("d0", "phi0", "z0", "theta", "t0", "mom")]
    cols += [c + "err" for c in cols] + ["kl_status", "kl_chi2", "kl_ndof", "kh_trk", "kh_ind", "kh_ambig", "kh_active"]
    tree = uproot.open(path)["trkdump/events"]
    xcols = ["kst_trk", "kst_sid", "kst_active", "kst_wall", "kst_gas", "kst_wire", "kst_radlen"]
    has_x = all(c in tree.keys() for c in xcols)                   # older dumps have no straw crossings
    a = tree.arrays(cols + (xcols if has_x else []))
    nfit = ak.to_numpy(ak.num(a.kl_d0))
    T = int(nfit.sum())
    ev = np.repeat(np.arange(len(a)), nfit)
    tr0 = np.concatenate([[0], np.cumsum(nfit)])
    kl = {k[3:]: _flat(a, k) for k in cols if k.startswith("kl_")}
    x = np.zeros((T, 6))
    x[:, :5] = linefit.from_kinematic_line(kl["d0"], kl["phi0"], kl["z0"], kl["theta"], kl["t0"])
    x[:, 5] = kl["mom"]
    st = kl["status"].astype(np.int64)
    status = np.where(st & KALMAN_CONVERGED, 1, np.where(st & KALMAN_OK, 2, -1)).astype(np.int32)
    ev0 = np.searchsorted(sh["ev"], np.arange(len(a) + 1), side="left")
    nh = ak.to_numpy(ak.num(a.kh_trk))
    htrk = _flat(a, "kh_trk").astype(np.int64) + np.repeat(tr0[:-1], nh)
    shi = _flat(a, "kh_ind").astype(np.int64) + np.repeat(ev0[:-1], nh)
    amb = _flat(a, "kh_ambig").astype(np.int64)
    act = _flat(a, "kh_active").astype(bool)
    lr = np.where(np.abs(amb) == 1, amb, 0).astype(np.float64)
    hits = _hits(htrk, shi, lr, act, sh, cond, T, state=amb.astype(np.int32))
    X = None
    if has_x:
        ns = ak.to_numpy(ak.num(a.kst_trk))
        xtrk = _flat(a, "kst_trk").astype(np.int64) + np.repeat(tr0[:-1], ns)
        X = {"sid": _pad(xtrk, _flat(a, "kst_sid").astype(np.int64), T, 0),
             "active": _pad(xtrk, _flat(a, "kst_active").astype(bool), T, False),
             "xvalid": _pad(xtrk, np.ones(len(xtrk), bool), T, False)}
        for k in ("wall", "gas", "wire", "radlen"):
            X[k] = _pad(xtrk, _flat(a, "kst_" + k).astype(np.float64), T, np.nan)
        for k in ("doca", "docavar", "dirdot"):
            X[k] = np.full(X["sid"].shape, np.nan)
    meta = {"version": VERSION, "fit": "KKLine (stock, from the StrawHitTableDumper KalSeeds)", "fit_type": "stock",
            "stock_conditioned": True, "source": path, "settings": settings, "n_events": int(len(a)), "n_tracks": T,
            "cov": "diagonal KinematicLine errors propagated to the line parameters (no correlations in the dump)"}
    return {"events": _events(path), "ev": ev, "x": x, "cov": _cov_from_kl_errors(kl, x), "chi2": kl["chi2"].astype(np.float64),
            "ndof": kl["ndof"].astype(np.int64), "status": status, "goodfit": ((st & FITOK) != 0).astype(np.int32),
            "momfit": np.ones(T, np.int32), **hits, "xings": X, "drift": _drift_table(path), "meta": meta}


def from_f2(path, cond, settings="mc", niter=4, all_fits=False, drift=None):
    """reco-vr3 chain on the dump's digis (fast hit reco, SimpleTimeCluster, LineFinder, F2 with its own states).
    Every seed is fitted; `all_fits` keeps the failed ones (goodfit 0), else only the fits passing the KKFit gates.
    `drift` supplies an explicit calibration when the digi-only input has no conditions tree."""
    from .cli import _drift_table
    explicit_drift = drift is not None
    drift = _drift_table(path) if drift is None else drift
    return from_f2_digis(hitreco.read_digis([path]), _events(path), cond, settings, niter, all_fits, drift,
                         source=path, explicit_drift=explicit_drift)


def from_f2_digis(d, events, cond, settings="mc", niter=4, all_fits=False, drift=None, source=None,
                  explicit_drift=True, sh=None, stages=None):
    """from_f2 on in-memory digis (hitreco.read_digis layout) and events (_events layout). `sh`: straw hits already
    made from d (e.g. by the JAX twin); `stages`: dict that receives the intermediate products and stage timings."""
    import time
    from . import linefind, linefind_jax, linefit_jax
    t = time.perf_counter()
    sh = hitreco.straw_hits(d, cond, settings) if sh is None else sh
    t_hits = time.perf_counter()
    tc = linefind.time_clusters(sh)
    t_tc = time.perf_counter()
    sd = linefind_jax.line_seeds(sh, tc, cond)
    t_sd = time.perf_counter()
    ns = sd["n"]
    trk = np.repeat(np.arange(len(ns)), ns)
    H = linefit.build_hits(trk, sd["member"], np.zeros(len(trk)), np.ones(len(trk), bool), sh, cond, len(ns))
    x0 = linefit.seed_params(sd["a0"], sd["b0"], sd["a1"], sd["b1"], sd["t0"])
    f = linefit_jax.fit_own_states(x0, H, drift, niter=niter) if len(ns) else None
    t_fit = time.perf_counter()
    T = len(ns)
    if T:
        nact = f["active"].sum(1)
        good = (nact >= 5) & np.isfinite(f["chi2"]) & (f["ndof"] >= 1)          # KKFIT MinNStrawHits 5, MinNDOF 1
    else:
        good = np.zeros(0, bool)
    keep = np.ones(T, bool) if all_fits else good
    k = np.flatnonzero(keep)
    x = np.zeros((len(k), 6))
    cov = np.full((len(k), 6, 6), np.nan)
    if len(k):
        x[:, :5] = f["x"][k]
        cov[:, :5, :5] = f["cov"][k]
    x[:, 5] = linefit.MOM
    # hit block in seed-member order; the final F2 states (active, lr) replace the all-active start
    hk = np.isin(trk, k)
    remap = np.full(T, -1)
    remap[k] = np.arange(len(k))
    act_f = np.zeros(len(trk), bool)
    lr_f = np.zeros(len(trk))
    if T:
        cnt = np.bincount(trk, minlength=T)
        pos = np.arange(len(trk)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        act_f = np.asarray(f["active"])[trk, pos].astype(bool)
        lr_f = np.asarray(f["lr"])[trk, pos].astype(np.float64)
    hits = _hits(remap[trk[hk]], sd["member"][hk], lr_f[hk], act_f[hk], sh, cond, len(k))
    meta = {"version": VERSION, "fit": "F2 (linefit_jax.fit_own_states, fixed momentum, no material)", "fit_type": "F2",
            "stock_conditioned": False, "chain": "reco-vr3: fast makeSH, SimpleTimeCluster, LineFinder (JAX), F2 own states",
            "source": source, "settings": settings, "niter": niter, "all_fits": all_fits, "n_events": int(len(d["evkey"])),
            "n_hits": int(len(sh["ev"])), "n_seeds": int(T), "n_good": int(good.sum()), "n_written": int(len(k)),
            "drift_source": "explicit calibration" if explicit_drift else "input conditions tree",
            "status_rule": "goodfit = (>= 5 active hits, finite chi2, ndof >= 1); status 1 when goodfit (fixed iteration "
                           "count, no convergence test), -1 otherwise"}
    g = good[k] if T else np.zeros(0, bool)
    if stages is not None:
        stages.update(hits=sh, clusters=tc, seeds=sd, fit=f, timing={"hits_s": t_hits - t, "clusters_s": t_tc - t_hits,
                      "seeds_s": t_sd - t_tc, "fit_s": t_fit - t_sd, "table_s": time.perf_counter() - t_fit})
    return {"events": events, "ev": sd["ev"][k].astype(np.int64), "x": x, "cov": cov,
            "chi2": np.asarray(f["chi2"])[k].astype(np.float64) if T else np.zeros(0),
            "ndof": np.asarray(f["ndof"])[k].astype(np.int64) if T else np.zeros(0, np.int64),
            "status": np.where(g, 1, -1).astype(np.int32), "goodfit": g.astype(np.int32), "momfit": np.zeros(len(k), np.int32),
            **hits, "xings": None, "drift": drift, "meta": meta}


def save(table, path):
    flat = {k: v for k, v in table.items() if k not in ("events", "xings", "drift", "meta")}
    flat.update({"ev_" + k: v for k, v in table["events"].items()})
    flat.update({"drift_" + k: np.asarray(v) for k, v in table["drift"].items()})
    if table["xings"] is not None:
        flat.update({"x_" + k: v for k, v in table["xings"].items()})
    np.savez_compressed(path, meta=json.dumps(table["meta"], sort_keys=True), **flat)


def load(path):
    z = np.load(path, allow_pickle=False)
    t = {"meta": json.loads(str(z["meta"])), "events": {}, "drift": {}, "xings": {}}
    for k in z.files:
        if k == "meta":
            continue
        if k.startswith("ev_"):
            t["events"][k[3:]] = z[k]
        elif k.startswith("drift_"):
            t["drift"][k[6:]] = z[k]
        elif k.startswith("x_"):
            t["xings"][k[2:]] = z[k]
        else:
            t[k] = z[k]
    if not t["xings"]:
        t["xings"] = None
    return t
