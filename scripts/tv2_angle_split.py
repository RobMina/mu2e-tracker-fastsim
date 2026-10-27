"""TT_Outer reco - truth residuals split by crossing: entry (upper half, y > 0 in the detector frame) and exit (lower
half), fast vs G4 EventNtuples, among the tracker-term-selected events (as tv2compare). Space-angle residual: median
and q84; projected angle residual in the two transverse directions (as dx/dy): half the 16-84 % width.

    python3 tracker/scripts/tv2_angle_split.py FAST_NTS G4_NTS FAST_TABLES G4_SGSTEPS
"""
import sys

import awkward as ak
import numpy as np
import uproot

from trkfastsim import tables as T
from trkfastsim.analyses import tv2compare as TV


def residuals(nts, sel):
    rows = {"y": [], "dang": [], "ah": [], "av": [], "dperp_h": [], "dperp_v": []}
    for path in T.expand([nts]):
        t = uproot.open(path)[TV.TREE]
        keys = np.stack([t[f"evtinfo/{k}"].array(library="np").astype(np.int64) for k in ("run", "subrun", "event")], 1)
        m = np.isin(TV._keyid_rows(keys), sel)
        rp, rm, rt, tp, tm, tt, _ = TV._pairs_at(t["trksegs"].array()[m], t["trksegsmc"].array()[m], TV.SID_OUTER)
        if not len(rp):
            continue
        u = tm / np.linalg.norm(tm, axis=1)[:, None]
        ur = rm / np.linalg.norm(rm, axis=1)[:, None]
        h = np.cross(u, np.array([0.0, 1.0, 0.0]))
        h /= np.maximum(np.linalg.norm(h, axis=1), 1e-12)[:, None]
        v = np.cross(h, u)
        d = rp - tp
        dp = d - (d * u).sum(1)[:, None] * u
        rows["y"].append(tp[:, 1])
        rows["dang"].append(1e3 * np.arccos(np.clip((ur * u).sum(1), -1, 1)))
        rows["ah"].append(1e3 * (ur * h).sum(1))
        rows["av"].append(1e3 * (ur * v).sum(1))
        rows["dperp_h"].append((dp * h).sum(1))
        rows["dperp_v"].append((dp * v).sum(1))
    return {k: np.concatenate(v) for k, v in rows.items()}


def w(x):
    q16, q84 = np.quantile(x, [0.16, 0.84])
    return 0.5 * (q84 - q16)


fast_nts, g4_nts, fast_tab, g4_tab = sys.argv[1:5]
sf, _ = TV._selected([fast_tab])
sg, _ = TV._selected([g4_tab])
F, G = residuals(fast_nts, sf), residuals(g4_nts, sg)
print(f"{'crossing':>8s} {'arm':>4s} {'n':>7s} | space angle q50/q84 mrad | proj angle width h/v mrad | position width h/v mm")
for lab, cut in (("entry", lambda r: r["y"] > 0), ("exit", lambda r: r["y"] <= 0), ("all", lambda r: np.ones(len(r["y"]), bool))):
    for arm, r in (("fast", F), ("G4", G)):
        c = cut(r)
        print(f"{lab:>8s} {arm:>4s} {c.sum():7d} | {np.median(r['dang'][c]):6.2f} / {np.quantile(r['dang'][c], 0.84):6.2f} | "
              f"{w(r['ah'][c]):6.2f} / {w(r['av'][c]):6.2f} | {w(r['dperp_h'][c]):5.2f} / {w(r['dperp_v'][c]):5.2f}")
