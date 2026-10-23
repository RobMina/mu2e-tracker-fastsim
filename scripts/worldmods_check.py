"""Check a worldmods modification: pieces added/changed, and the material column along probe lines before and after.

    python3 tracker/scripts/worldmods_check.py issue2017
"""
import sys

import numpy as np

from mu2efastsim import world as W
from trkfastsim import geometry, worldmods
from trkfastsim.analyses.steps import default_gdml

name = sys.argv[1]
nm = []
pcs = W.load(default_gdml(), names=nm)
p2, n2 = worldmods.apply(name, list(pcs), list(nm))
print(f"{name}: pieces {len(pcs)} -> {len(p2)}; new names {sorted(set(n2) - set(nm))}; "
      f"changed dsAreaUpper concrete pieces {sum(1 for (m, *_), n in zip(p2, n2) if n.endswith('_soil2017'))}")
x0, z0 = geometry.DETECTOR_ORIGIN_EXTRACTED[0], geometry.DETECTOR_ORIGIN_EXTRACTED[2]
A, B = W.PieceSet(pcs).build_grid(500.0), W.PieceSet(p2).build_grid(500.0)
for y in (3200.0, 4300.0):
    for z in (z0 - 15000, z0, z0 + 1000):
        o = np.array([[x0 - 6000, y, z]]); d = np.array([[1.0, 0, 0]])
        ca, cb = A.column(o, d, np.array([12000.0])), B.column(o, d, np.array([12000.0]))
        keys = sorted(set(ca) | set(cb))
        diff = {k: round(float(cb.get(k, [0])[0] - ca.get(k, [0])[0]), 1) for k in keys
                if abs(cb.get(k, [0])[0] - ca.get(k, [0])[0]) > 0.05}
        print(f"  horizontal line y {y:.0f} dz {z - z0:+.0f}: column change (mm) {diff}")
rng = np.random.default_rng(1)
n = 200000
o = np.c_[rng.uniform(x0 - 5000, x0 + 5000, n), np.full(n, 9000.0), rng.uniform(z0 - 20000, z0 + 5000, n)]
th = np.arccos(rng.uniform(0.5, 1.0, n)); ph = rng.uniform(0, 2 * np.pi, n)
d = np.c_[np.sin(th) * np.cos(ph), -np.cos(th), np.sin(th) * np.sin(ph)]
L = (o[:, 1] - 800.0) / -d[:, 1]
ca, cb = A.column(o, d, L), B.column(o, d, L)
for k in sorted(set(ca) | set(cb)):
    dk = cb.get(k, np.zeros(n)) - ca.get(k, np.zeros(n))
    if np.any(np.abs(dk) > 0.05):
        print(f"  random downward lines (cos zenith 0.5-1, to y 800): {k}: lines changed {np.mean(np.abs(dk) > 0.05):.4f}, "
              f"mean change on changed {dk[np.abs(dk) > 0.05].mean():+.1f} mm")
