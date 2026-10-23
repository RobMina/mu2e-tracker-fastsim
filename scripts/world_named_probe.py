"""World pieces (world.load with names) crossing given lines: per line, each named volume's material and the interval
it occupies along the line, plus the pieces' bounding boxes relative to the tracker axis point (-3904, 0, 24171).

    python3 tracker/scripts/world_named_probe.py "x0,y0,z0:x1,y1,z1" [...]  [--match REGEX]
"""
import re
import sys

import numpy as np

from mu2efastsim import world as W
from trkfastsim.analyses.steps import default_gdml

args = [a for a in sys.argv[1:] if not a.startswith("--")]
rx = re.compile(sys.argv[sys.argv.index("--match") + 1]) if "--match" in sys.argv else None
if rx:
    args = [a for a in args if a != rx.pattern]
names = []
pcs = W.load(default_gdml(), names=names)
TRK = np.array([-3904.0, 0.0, 24171.0])
for arg in args:
    a, b = (np.array([float(v) for v in s.split(",")]) for s in arg.split(":"))
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    print(f"== {a - TRK} -> {b - TRK} (relative to the tracker axis point)")
    hits = []
    for (m, P, lo, hi), n in zip(pcs, names):
        if rx and not rx.search(n):
            continue
        den, num = P[:, :3] @ u, P[:, 3] - P[:, :3] @ a
        with np.errstate(divide="ignore", invalid="ignore"):
            s = num / den
        enter = np.max(np.where(den < 0, s, -np.inf))
        leave = np.min(np.where(den > 0, s, np.inf))
        if np.any((den == 0) & (num < 0)):
            continue
        e0, e1 = max(enter, 0.0), min(leave, L)
        if e1 > e0:
            hits.append((e0, e1, n, m, lo - TRK, hi - TRK))
    for e0, e1, n, m, lo, hi in sorted(hits):
        print(f"   {n:34s} {m:14s} s {e0:7.0f}..{e1:7.0f}  box x {lo[0]:6.0f}..{hi[0]:6.0f} y {lo[1]:6.0f}..{hi[1]:6.0f} "
              f"z {lo[2]:7.0f}..{hi[2]:7.0f}")
