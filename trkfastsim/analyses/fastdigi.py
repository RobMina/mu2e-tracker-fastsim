"""digi-cond / digi -- fast tracker digitisation (trkfastsim.digi) of StrawGasStep tables into StrawDigi tables.

digi-cond: conditions file (.npz) from the stock job's sources: the fhicl-dump of the digitisation job (input), the
DbService CSVs of the run (dbTool print-run ... --content) and the GDML export.

digi: StrawGasStep tables (dump/sgsteps: fast tables from `steps`, or G4 tables from StrawGasStepTableDumper; detector
frame) -> one table per input, <outdir>/fastdigi.<input stem>.root, tree trkdump/events with the StrawHitTableDumper
digi columns (run, subrun, event, pbt, spill, d_sid, d_tdc0, d_tdc1, d_tot0, d_tot1, d_pmp, d_nadc, d_adc), so
trkfastsim.reco.hitreco.read_digis reads it as it reads stock digis, plus fast-only truth columns d_step0/d_step1
(row of the input sgsteps tree whose cluster the crossing at the cal/hv end refers to, as StrawDigiMC). Events: those
of --events (StrawHitTableDumper files: the stock digitiser's events, for paired comparisons), else the production
PrimaryFilter tracker term on the steps (the path A filter with no CRV table). Off spill: pbt = 0, spill = 0.
"""
import os
import time

import awkward as ak
import numpy as np
import uproot

from .. import digi as D
from .. import digicond, tables

STEP_COLS = ["run", "subrun", "event", "plane", "panel", "straw", "shape", "ion", "eion", "len", "width", "t",
             "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz", "sp"]


def check_against_dump(c, ref):
    """Straws of the conditions against the (reco-aligned) tracker in a StrawHitTableDumper conditions tree."""
    t = uproot.open(ref)["trkdump/conditions"].arrays(library="np")
    g = {k: np.asarray(v[0]) for k, v in t.items()}
    us = D.unique_straw(g["sid"].astype(np.int64))
    mid = np.stack([g["midx"], g["midy"], g["midz"]], 1)
    dr = np.stack([g["dirx"], g["diry"], g["dirz"]], 1)
    return {"straws": int(len(us)), "dir_dot_min": float((dr * c.straw_dir[us]).sum(1).min()),
            "mid_dist_max_mm": float(np.linalg.norm(mid - c.straw_mid[us], axis=1).max()),
            "mid_dist_median_mm": float(np.median(np.linalg.norm(mid - c.straw_mid[us], axis=1))),
            "halflen_diff_max_mm": float(np.abs(g["hlen"] - c.wire_halflen[us]).max()),
            "strawOuterRadius": [float(g["strawOuterRadius"]), c.strawOuterRadius]}


def run_cond(inputs, outdir, preamp=None, align=None, gdml=None, global_constants=None, out=None, check_ref=None, **kw):
    if not preamp:
        raise SystemExit("digi-cond FCL_DUMP --preamp TrkPreampStraw.csv [--align TrkAlignStrawSim.csv] --gdml GDML")
    from .steps import default_gdml
    c = digicond.build(inputs[0], preamp, gdml or default_gdml(), align, None, global_constants)
    out = out or os.path.join(outdir, "digicond.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    c.save(out)
    print(f"wrote {out}: EAverage {c.EAverage:.6g} MeV, threshold mean {c.vthresh.mean():.3f} mV, "
          f"drift time at 2.5 mm {c.drift_times[-1, 0]:.2f} ns, window {c.eventWindowLength:.0f} ns")
    if check_ref:
        print("straws vs the reco tracker of", check_ref, check_against_dump(c, check_ref))


def event_keys(paths):
    """(run, subrun, event) of every event of StrawHitTableDumper files, in file order."""
    ks = []
    for p in tables.expand(paths):
        a = uproot.open(p)["trkdump/events"].arrays(["run", "subrun", "event"], library="np")
        ks.append(np.stack([a["run"], a["subrun"], a["event"]], 1).astype(np.int64))
    return np.concatenate(ks) if ks else np.zeros((0, 3), np.int64)


def _keyid(k):
    return (k[:, 0] << 40) | (k[:, 1] << 20) | k[:, 2]


def load_steps(path):
    s = tables.read(path, "sgsteps", STEP_COLS)
    return s


def write_digis(path, keys, d, nsamp):
    """Digis of the events keys [E, 3] (d['ev'] indexes keys) -> trkdump/events in the StrawHitTableDumper layout."""
    E = len(keys)
    cnt = np.bincount(d["ev"], minlength=E) if len(d["ev"]) else np.zeros(E, np.int64)

    def jag(a):
        return ak.unflatten(np.asarray(a, np.int32), cnt)
    cols = {"run": keys[:, 0].astype(np.uint32), "subrun": keys[:, 1].astype(np.uint32),
            "event": keys[:, 2].astype(np.uint32), "pbt": np.zeros(E, np.float32), "spill": np.zeros(E, np.int32),
            "d_sid": jag(d["sid"]), "d_tdc0": jag(d["tdc"][:, 0]), "d_tdc1": jag(d["tdc"][:, 1]),
            "d_tot0": jag(d["tot"][:, 0]), "d_tot1": jag(d["tot"][:, 1]), "d_pmp": jag(d["pmp"]),
            "d_nadc": jag(np.full(len(d["sid"]), nsamp)), "d_adc": ak.unflatten(np.asarray(d["adc"], np.int32).ravel(), cnt * nsamp),
            "d_step0": jag(d["step"][:, 0]), "d_step1": jag(d["step"][:, 1])}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with uproot.recreate(path) as f:
        f.mktree("trkdump/events", {k: (v.type.content if isinstance(v, ak.Array) else v.dtype) for k, v in cols.items()})
        f["trkdump/events"].extend(cols)


def validate_backend(backend):
    if backend not in ("numpy", "jax"):
        raise ValueError(f"unsupported digitizer backend {backend!r}; expected 'numpy' or 'jax'")


def digitize_jax(st, nev, cond, key, stats=None):
    """Load the optional JAX digitizer only when selected."""
    from ..digi_jax import digitize
    return digitize(st, nev, cond, key, stats)


def digitize_table(path, cond, rng, keys=None, chunk=1000, stats=None, backend="numpy"):
    """Return (event keys [E,3], digis indexing keys, step count).

    rng is a NumPy Generator by default, or an explicit JAX root key for backend='jax'.
    JAX folds in each chunk's event offset; reproducibility requires the same chunk size.
    """
    validate_backend(backend)
    if backend == "jax":
        from jax import random as jr
    s = load_steps(path)
    sk = s[["run", "subrun", "event"]].to_numpy(np.int64)
    if keys is None:
        f = tables.trk_filter(s)
        keys = np.array(f.index[f.pass_trk].tolist(), np.int64).reshape(-1, 3)
        o = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        keys = keys[o]
    kid, sid_ = _keyid(keys), _keyid(sk)
    order = np.argsort(kid, kind="stable")
    pos = np.searchsorted(kid[order], sid_)
    pos = np.minimum(pos, len(kid) - 1) if len(kid) else pos
    hit = (kid[order][pos] == sid_) if len(kid) else np.zeros(len(sid_), bool)
    ev = np.where(hit, order[pos] if len(kid) else 0, -1)
    s = s[hit].reset_index(drop=True)
    ev = ev[hit]
    row_in_table = np.nonzero(hit)[0]
    st = {k: s[k].to_numpy() for k in ("shape", "ion", "eion", "len", "width", "t", "x0", "y0", "z0", "x1", "y1", "z1")}
    st["sid"] = geometry_sid(s)
    st["ev"] = ev
    outs = []
    for e0 in range(0, len(keys), chunk):
        m = (ev >= e0) & (ev < e0 + chunk)
        idx = np.nonzero(m)[0]
        sub = {k: v[idx] for k, v in st.items()}
        sub["ev"] = sub["ev"] - e0
        t0 = time.perf_counter()
        if backend == "numpy":
            d = D.digitize(sub, min(chunk, len(keys) - e0), cond, rng, stats)
        else:
            d = digitize_jax(sub, min(chunk, len(keys) - e0), cond, jr.fold_in(rng, e0), stats)
        if stats is not None:
            stats["digitize_s"] = stats.get("digitize_s", 0.0) + time.perf_counter() - t0
        d["ev"] = d["ev"] + e0
        d["step"] = row_in_table[idx[d["step"]]] if len(d["step"]) else d["step"]
        outs.append(d)
    out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]} if outs else None
    return keys, out, len(st["sid"])


def geometry_sid(s):
    return (s["plane"].to_numpy(np.int64) << 10) | (s["panel"].to_numpy(np.int64) << 7) | s["straw"].to_numpy(np.int64)


def run_digi(inputs, outdir, cond=None, events=None, seed=1, chunk=1000, backend="numpy", **kw):
    validate_backend(backend)
    if not cond:
        raise SystemExit("digi STEP_TABLES --cond digicond.npz [--events STOCK_STRAWHITS ...] [--seed N]")
    if backend == "jax":
        from jax import random as jr
        root_key = jr.key(seed)
    c = digicond.Conditions.load(cond)
    keys_all = event_keys(events) if events else None
    os.makedirs(outdir, exist_ok=True)
    for i, p in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([seed, i]) if backend == "numpy" else jr.fold_in(root_key, i)
        keys = None
        if keys_all is not None:
            sk = tables.read(p, "sgsteps", ["run", "subrun"]).drop_duplicates().to_numpy(np.int64)
            m = np.isin(_keyid(np.c_[keys_all[:, :2], np.zeros(len(keys_all), np.int64)]),
                        _keyid(np.c_[sk, np.zeros(len(sk), np.int64)]))
            keys = keys_all[m]
        stats = {}
        t0 = time.perf_counter()
        keys, d, nstep = digitize_table(p, c, rng, keys, chunk, stats, backend=backend)
        dt = time.perf_counter() - t0
        stem = os.path.basename(p).rsplit(".root", 1)[0]
        out = os.path.join(outdir, f"fastdigi.{stem}.root")
        write_digis(out, keys, d, c.nADCSamples)
        nd = len(d["sid"]) if d else 0
        dd = stats.get("digitize_s", 0.0)
        print(f"{p}: {len(keys)} events, {nstep} steps, {nd} digis, {dt:.2f} s with I/O, digitisation {dd:.2f} s "
              f"({1e3 * dd / max(len(keys), 1):.3f} ms/event)"
              f" -> {out}\n  {stats}")
