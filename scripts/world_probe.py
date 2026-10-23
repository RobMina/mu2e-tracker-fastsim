"""Material intervals of the stage C world (world.load, every non-air piece) along straight lines, in 5 mm steps.

    python3 tracker/scripts/world_probe.py "x0,y0,z0:x1,y1,z1" [...]     (Mu2e frame, mm)
"""
import sys

import numpy as np

from mu2efastsim import world as W
from trkfastsim.analyses.steps import default_gdml

ps = W.PieceSet(W.load(default_gdml())).build_grid(500.0)
STEP = 5.0
for arg in sys.argv[1:]:
    a, b = (np.array([float(v) for v in s.split(",")]) for s in arg.split(":"))
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    s = np.arange(0.0, L, STEP)
    col = ps.column(a + s[:, None] * u, np.tile(u, (len(s), 1)), np.full(len(s), STEP))
    mats = np.array(["air"] * len(s), dtype=object)
    for m, c in col.items():
        mats[c > 0.5 * STEP] = m
    print(f"== {a} -> {b}")
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and mats[j + 1] == mats[i]:
            j += 1
        p0, p1 = a + s[i] * u, a + (s[j] + STEP) * u
        print(f"   {mats[i]:16s} from {np.round(p0).astype(int)} to {np.round(p1).astype(int)}  ({(j - i + 1) * STEP:.0f} mm)")
        i = j + 1
