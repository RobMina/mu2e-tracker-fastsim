"""Merge arrivalcheck event tables of several batches and print the per-margin summary (as arrivalcheck), optionally
for the G4 CRV-term subset.

    python3 tracker/scripts/arrival_merge.py "arr_v2_*/arrivalcheck_events.csv.gz" [--crv "crvsteps glob"]
"""
import sys

import numpy as np
import pandas as pd

from trkfastsim import tables as T
from trkfastsim.analyses.arrivalcheck import MARGIN_EDGES, _q

ev = pd.concat([pd.read_csv(f) for f in T.expand([sys.argv[1]])], ignore_index=True)
if "--crv" in sys.argv:
    from trkfastsim.analyses.seedcheck import crv_pass_from
    ev = ev.merge(crv_pass_from(T.read_many(T.expand([sys.argv[sys.argv.index("--crv") + 1]]), "steps")), on=T.KEY)
ev = ev[ev.f_near]
print(f"G4 muons crossing the plane with the generated line near the footprint: {len(ev):,}")
b = np.clip(np.digitize(ev.margin, MARGIN_EDGES) - 1, 0, len(MARGIN_EDGES) - 2)
print(f"{'margin':>12s} {'n':>7s} | arrive | |p| plane q10/q50/q90 G4 | fast | aimed G4 | fast | aimed ratio")
for k in range(len(MARGIN_EDGES) - 1):
    g = ev[b == k]
    if len(g) < 20:
        continue
    fa = g[g.f_alive]
    ag, af = np.mean(g.g_miss <= 0), np.mean(g.f_alive & (g.f_miss <= 0))
    err = np.sqrt(ag * (1 - ag) / len(g) * 2) / max(ag, 1e-9)
    print(f"{str([MARGIN_EDGES[k], MARGIN_EDGES[k + 1]]):>12s} {len(g):7d} | {g.f_alive.mean():.4f} | {_q(g.g_p)} | {_q(fa.f_p)} | "
          f"{ag:.4f} | {af:.4f} | {af / max(ag, 1e-9) - 1:+.4f} +- {err:.4f}")
