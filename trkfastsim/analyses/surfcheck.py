"""surfcheck -- tracker VD crossings (SurfaceSteps, sids 0-4) of the primary muon, fast sim against Geant4.

Selection in both arms: events whose primary muon passes the PrimaryFilter tracker term (>= 12 good steps), as v1compare;
rates per second of live time (unpaired). Only crossings of the event's MC primary muon (by particle identity) are
compared; the G4 share of crossings from other particles is reported separately (the fast sim writes only the muon).
Per sid: crossings per selected event, rate per live time, the surface coordinate residual (z on planes, r on
cylinders; should be ~0), binned ratios of the free coordinates, |p| and cos(angle to the surface normal), and the
time relative to the muon's first straw.

    surfcheck FAST.root [...] --compare G4.root [...] --outdir DIR
"""
import json
import os

import numpy as np

from mu2efastsim import stats
from .. import surfaces, tables as T

NAMES = {0: "TT_Front", 1: "TT_Mid", 2: "TT_Back", 3: "TT_Inner", 4: "TT_Outer"}


def _load(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    prim = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)][T.KEY + ["sp"]]
    sm = s.merge(prim, on=T.KEY + ["sp"])
    f = T.trk_filter(sm)
    sel = f[f.pass_trk].reset_index()[T.KEY]
    t1 = sm.merge(sel, on=T.KEY).groupby(T.KEY).t.min().rename("t_first").reset_index()
    ss = T.read_many(paths, "surfsteps")
    ss = ss[ss.surf.isin(list(NAMES))].merge(sel, on=T.KEY)
    is_prim = ss.merge(prim.assign(_p=1), on=T.KEY + ["sp"], how="left")._p.fillna(0).to_numpy() == 1
    return sel, ss[is_prim].merge(t1, on=T.KEY), ss[~is_prim]


def _obs(g):
    x, y, z = [(g[f"{a}0"].to_numpy() + g[f"{a}1"].to_numpy()) / 2 for a in "xyz"]
    r = np.hypot(x, y)
    p = np.sqrt(g.px ** 2 + g.py ** 2 + g.pz ** 2).to_numpy()
    sid = g.surf.to_numpy()
    plane = sid <= 2
    zsurf = np.select([sid == 0, sid == 1, sid == 2], [surfaces.Z_FRONT, surfaces.Z_MID, surfaces.Z_BACK], np.nan)
    rsurf = np.where(sid == 3, surfaces.R_IN, surfaces.R_OUT)
    cosn = np.where(plane, np.abs(g.pz.to_numpy()) / p, np.abs(g.px.to_numpy() * x + g.py.to_numpy() * y) / (p * r))
    return {"resid": np.where(plane, z - zsurf, r - rsurf), "free1": np.where(plane, r, z), "phi": np.arctan2(y, x),
            "p": p, "cosn": cosn, "dt": (g.t - g.t_first).to_numpy()}


def run(inputs, outdir, compare=None, **kw):
    if not compare:
        raise SystemExit("surfcheck FAST --compare G4REF")
    outdir = outdir or "plots/surfcheck"
    os.makedirs(outdir, exist_ok=True)
    fp, rp = T.expand(inputs), T.expand(compare)
    lt_f, lt_r = T.livetime(fp), T.livetime(rp)
    sel_f, pf, _ = _load(fp)
    sel_r, pr, other_r = _load(rp)
    out = {"livetime_s": [lt_f, lt_r], "selected": [int(len(sel_f)), int(len(sel_r))], "by_sid": {}}
    print(f"live time fast {lt_f:.2f} s, G4 {lt_r:.2f} s; selected muons fast {len(sel_f):,}, G4 {len(sel_r):,}")
    print(f"{'sid':>3} {'name':9} {'n fast':>7} {'n G4':>7} {'per ev fast':>11} {'per ev G4':>10} {'rate ratio-1':>14} "
          f"{'|resid| p99 f/G4 [mm]':>22} {'free':>6} {'phi':>6} {'|p|':>6} {'cosn':>6} {'dt':>6} {'G4 non-mu':>9}")
    for sid, name in NAMES.items():
        a, b = pf[pf.surf == sid], pr[pr.surf == sid]
        nf, nr = len(a), len(b)
        rr = (nf / lt_f) / max(nr / lt_r, 1e-30) - 1 if nr else float("nan")
        err = np.sqrt(1 / max(nf, 1) + 1 / max(nr, 1)) * (rr + 1) if nr else float("nan")
        d = {"n": [nf, nr], "per_selected_event": [nf / max(len(sel_f), 1), nr / max(len(sel_r), 1)],
             "rate_ratio_minus_1": rr, "rate_ratio_err": err,
             "g4_nonprimary_share": float(len(other_r[other_r.surf == sid]) / max(nr + len(other_r[other_r.surf == sid]), 1))}
        if nf and nr:
            of, orf = _obs(a), _obs(b)
            d["resid_p99_mm"] = [float(np.quantile(np.abs(of["resid"]), 0.99)), float(np.quantile(np.abs(orf["resid"]), 0.99))]
            for k in ("free1", "phi", "p", "cosn", "dt"):
                rb = stats.ratio_band(of[k], orf[k], min_ref=100)
                d[f"{k}_band_m2s"] = rb["max_abs_dev_minus_2sigma"]
                d[f"{k}_median"] = [float(np.median(of[k])), float(np.median(orf[k]))]
        out["by_sid"][name] = d
        f = lambda k: ("  n/a" if d.get(k) is None else f"{d[k]:6.3f}")
        print(f"{sid:>3} {name:9} {nf:>7} {nr:>7} {d['per_selected_event'][0]:>11.3f} {d['per_selected_event'][1]:>10.3f} "
              f"{rr:+8.3f}+-{err:.3f} {str([round(v, 3) for v in d.get('resid_p99_mm', [])]):>22} {f('free1_band_m2s')} "
              f"{f('phi_band_m2s')} {f('p_band_m2s')} {f('cosn_band_m2s')} {f('dt_band_m2s')} {d['g4_nonprimary_share']:>9.3f}")
    with open(os.path.join(outdir, "surfcheck.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {os.path.join(outdir, 'surfcheck.json')}")
