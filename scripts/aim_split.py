"""Among G4 muons arriving at the plane (arrivalcheck events, generated line near the footprint), by energy margin: the
share aimed at the straw footprint in G4 and in the fast arm, split into the two pieces of the fast transport:
the generated line itself aimed at the footprint (no scattering), and the share the scattering moves in vs out
(fast aimed but line not, line aimed but fast not), with G4's corresponding numbers relative to the same line.

    python3 tracker/scripts/aim_split.py "arr_v2_*/arrivalcheck_events.csv.gz"
"""
import sys

import numpy as np
import pandas as pd

from trkfastsim import geometry, steps as st, tables as T
from trkfastsim.analyses.arrivalcheck import MARGIN_EDGES
from trkfastsim.analyses.steps import default_gdml
from trkfastsim.analyses.worldcheck import _miss_distance

ev = pd.concat([pd.read_csv(f) for f in T.expand([sys.argv[1]])], ignore_index=True)
ev = ev[ev.f_near].reset_index(drop=True)
geom = geometry.from_gdml(default_gdml())
y_pl = st.plane_y(geom)
p0 = np.sqrt(ev.px ** 2 + ev.py ** 2 + ev.pz ** 2).to_numpy()
line_aim = _miss_distance(ev[["x", "y", "z"]].to_numpy(), ev[["px", "py", "pz"]].to_numpy() / p0[:, None], geom, y_pl) <= 0
g_aim = (ev.g_miss <= 0).to_numpy()
f_aim = (ev.f_alive & (ev.f_miss <= 0)).to_numpy()
b = np.clip(np.digitize(ev.margin, MARGIN_EDGES) - 1, 0, len(MARGIN_EDGES) - 2)
print(f"{'margin':>12s} {'n':>7s} | line aimed | G4: in / out (vs line) | fast: in / out (vs line) | aimed G4 / fast")
for k in range(len(MARGIN_EDGES) - 1):
    m = b == k
    if m.sum() < 50:
        continue
    la, ga, fa = line_aim[m], g_aim[m], f_aim[m]
    print(f"[{MARGIN_EDGES[k]:g}, {MARGIN_EDGES[k + 1]:g}) {m.sum():7d} | {la.mean():.4f} | "
          f"{np.mean(ga & ~la):.4f} / {np.mean(~ga & la):.4f} | {np.mean(fa & ~la):.4f} / {np.mean(~fa & la):.4f} | "
          f"{ga.mean():.4f} / {fa.mean():.4f}")
