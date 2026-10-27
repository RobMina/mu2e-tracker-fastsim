"""Fast track table: the per-track contract between a tracker fit (stock KKLine from the dump, F1, F2) and the
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
from digis: hit reco, SimpleTimeCluster, LineFinder, F2 with its own hit states), `from_f1` (reco-vr2f1 outputs saved
with --save-tracks: stock-conditioned seeds, hits and crossings, F1 states and parameters).
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


def from_f2(path, cond, settings="mc", niter=4, all_fits=False):
    """reco-vr3 chain on the dump's digis (fast hit reco, SimpleTimeCluster, LineFinder, F2 with its own states).
    Every seed is fitted; `all_fits` keeps the failed ones (goodfit 0), else only the fits passing the KKFit gates."""
    from . import linefind, linefind_jax, linefit_jax
    from .cli import _drift_table
    d = hitreco.read_digis([path])
    sh = hitreco.straw_hits(d, cond, settings)
    tc = linefind.time_clusters(sh)
    sd = linefind_jax.line_seeds(sh, tc, cond)
    drift = _drift_table(path)
    ns = sd["n"]
    trk = np.repeat(np.arange(len(ns)), ns)
    H = linefit.build_hits(trk, sd["member"], np.zeros(len(trk)), np.ones(len(trk), bool), sh, cond, len(ns))
    x0 = linefit.seed_params(sd["a0"], sd["b0"], sd["a1"], sd["b1"], sd["t0"])
    f = linefit_jax.fit_own_states(x0, H, drift, niter=niter) if len(ns) else None
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
            "source": path, "settings": settings, "niter": niter, "all_fits": all_fits, "n_events": int(len(d["evkey"])),
            "n_seeds": int(T), "n_good": int(good.sum()), "n_written": int(len(k)),
            "status_rule": "goodfit = (>= 5 active hits, finite chi2, ndof >= 1); status 1 when goodfit (fixed iteration "
                           "count, no convergence test), -1 otherwise"}
    g = good[k] if T else np.zeros(0, bool)
    return {"events": _events(path), "ev": sd["ev"][k].astype(np.int64), "x": x, "cov": cov,
            "chi2": np.asarray(f["chi2"])[k].astype(np.float64) if T else np.zeros(0),
            "ndof": np.asarray(f["ndof"])[k].astype(np.int64) if T else np.zeros(0, np.int64),
            "status": np.where(g, 1, -1).astype(np.int32), "goodfit": g.astype(np.int32), "momfit": np.zeros(len(k), np.int32),
            **hits, "xings": None, "drift": drift, "meta": meta}


F1_STATUS = {0: 1, 1: 2}            # kalman_jax CONVERGED -> 1, UNCONVERGED (MaxNIter reached) -> 2, else -1


def from_f1(path, cond, f1_tracks, settings="mc"):
    """reco-vr2f1 --save-tracks output on the same dump: rebuilds the stock-conditioned inputs exactly as the run did
    (read_inputs with the saved max_tracks / straw-axis / FitOK selection) and attaches the F1 results."""
    from . import vr_f1
    from .cli import _ref
    z = np.load(f1_tracks, allow_pickle=False)
    run = json.loads(str(z["meta"]))
    x0, H, X, ref, drift, T = vr_f1.read_inputs(path, cond, settings, run["max_tracks"], run["straw_axis"],
                                                 fitok_only=not run["own_states"])
    if not np.array_equal(ref["event_index"], z["event_index"]):
        raise ValueError(f"{f1_tracks}: event list differs from the dump selection; not the same run")
    d = hitreco.read_digis([path])
    sh = hitreco.straw_hits(d, cond, settings)
    rs = _ref([path], "s_")
    for c in ("time", "et0", "et1", "edep", "flag"):
        sh[c] = rs[c].astype(sh[c].dtype)
    hv = H["sh"] >= 0
    tr, pos = np.nonzero(hv)
    fact, flr = z["fact"].astype(bool), z["flr"].astype(np.float64)
    hits = _hits(tr, H["sh"][hv], np.where(fact, flr, 0.0)[hv], fact[hv], sh, cond, T)
    a = uproot.open(path)["trkdump/events"].arrays(["kst_sid", "kst_wire"])[ref["event_index"]]
    ns = ak.to_numpy(ak.num(a.kst_sid))
    xt = np.repeat(np.arange(T), ns)
    Xo = {"xvalid": X["valid"], "sid": _pad(xt, _flat(a, "kst_sid").astype(np.int64), T, 0)[:, :X["valid"].shape[1]],
          "active": z["xact"].astype(bool) & X["valid"], "wall": z["xwall"], "gas": z["xgas"],
          "wire": np.full(X["valid"].shape, np.nan), "doca": z["xdoca"], "docavar": z["xdvar"], "dirdot": z["xdirdot"]}
    Xo["radlen"] = np.where(X["valid"], z["xwall"] / X0_WALL + z["xgas"] / X0_GAS, np.nan)
    stat = z["stat"].astype(np.int64)
    events = _events(path)
    if run["max_tracks"]:
        # a --max-tracks run covers the events up to its last selected one: later events are not written
        ncover = int(ref["event_index"].max()) + 1 if T else 0
        events = {k: v[:ncover] for k, v in events.items()}
    meta = {"version": VERSION, "fit": "F1 (kalman_jax, six parameters, stock schedule)", "fit_type": "F1",
            "stock_conditioned": True,
            "conditioning": "stock LineFinder seed, stock KalSeed hit list (kh_*), stock straw-crossing list (kst_*); "
                            "single-seed single-fit events only" + ("" if run["own_states"] else "; stock final hit states, FitOK only"),
            "source": path, "f1_tracks": f1_tracks, "settings": settings, "f1_run": run, "n_events": int(len(d["evkey"])),
            "n_tracks": int(T), "n_good": int(z["good"].sum()), "n_events_written": int(len(events["run"])),
            "radlen": "wall/X0 + gas/X0 with nominal X0 (Mylar 285.4 mm, Ar/CO2 125 m): approximation"}
    return {"events": events, "ev": ref["event_index"].astype(np.int64), "x": z["xf"].astype(np.float64),
            "cov": z["Cf"].astype(np.float64), "chi2": z["chi2"].astype(np.float64), "ndof": z["ndof"].astype(np.int64),
            "status": np.vectorize(lambda s: F1_STATUS.get(int(s), -1), otypes=[np.int32])(stat) if T else np.zeros(0, np.int32),
            "goodfit": z["good"].astype(np.int32), "momfit": np.ones(T, np.int32), **hits, "xings": Xo, "drift": drift,
            "meta": meta}


X0_WALL, X0_GAS = 285.4, 125000.0     # mm, nominal (vr_f1._x0_ratio)


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
