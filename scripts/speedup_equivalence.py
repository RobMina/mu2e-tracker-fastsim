"""Equivalence of the stage C speed-ups on random lines through the tracker region: the bisection split length against
the linear 1/16 scan, and the calorimeter column with the box cull against without it (exact equality reported).

    python3 tracker/scripts/speedup_equivalence.py [N]
"""
import sys

import numpy as np

from trkfastsim import crossings, geometry, steps as st
from trkfastsim.analyses.steps import default_gdml

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
gdml = default_gdml()
geom = geometry.from_gdml(gdml)
it = st.TrackerInterior(gdml, geom)
lo, hi = crossings.panel_boxes(geom)
lo, hi = lo.min(0) - 300, hi.max(0) + 300
rng = np.random.default_rng(5)
o = np.c_[rng.uniform(lo[0], hi[0], n), np.full(n, hi[1]), rng.uniform(lo[2], hi[2], n)]
d = np.c_[rng.normal(0, 0.4, n), -np.ones(n), rng.normal(0, 0.4, n)]
d /= np.linalg.norm(d, axis=1)[:, None]
seg = rng.uniform(50, 2000, n)
x = it._support_sums(it._support(o, d, seg, False)[0], {}, n)[1]
m = x > st.KINK_SUP_XX0
o, d, seg, x = o[m], d[m], seg[m], x[m]
target = x / np.ceil(x / st.KINK_SUP_XX0)
bis = it._split_length(o, d, seg, target)
lin = seg.copy()
found = np.zeros(len(seg), bool)
xs = []
for k in range(1, it.SPLIT_PROBES):
    L = seg * k / it.SPLIT_PROBES
    xk = it._support_sums(it._support(o, d, L, False)[0], {}, len(seg))[1]
    xs.append(xk)
    h = ~found & (xk >= target)
    lin[h] = L[h]
    found |= h
xs = np.array(xs)
nonmono = np.any(np.diff(xs, axis=0) < -1e-12, axis=0)
diff = bis != lin
print(f"split lines {len(seg)}: bisection != linear scan for {diff.sum()}; non-monotone x(L) on {nonmono.sum()} lines; "
      f"differences on non-monotone lines {np.sum(diff & nonmono)}")
cc = it.shells.calo
full = cc.column(o, d, seg)
part = it.shells.column(o, d, seg, nsub=it.nsub)
coax = it.shells.coax.column(o, d, seg, nsub=it.nsub)
bad = 0
for k, v in full.items():
    ref = v + coax.get(k, 0.0)
    bad += int(np.sum(part.get(k, np.zeros(len(o))) != ref))
print(f"calorimeter cull: {bad} (line, material) entries differ from the uncull column (exact comparison)")
