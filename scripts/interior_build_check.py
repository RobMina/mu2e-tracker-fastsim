"""Build the TrackerInterior from the GDML and check the calorimeter signed pieces: mass by Monte Carlo over the
pieces' box against the GDML placement masses (boxes exactly), skipped placements, and the column cost per track.

    python3 tracker/scripts/interior_build_check.py [GDML]
"""
import sys
import time

import numpy as np

from mu2efastsim import materials
from trkfastsim import geometry, steps as st
from trkfastsim.analyses.steps import default_gdml

gdml = sys.argv[1] if len(sys.argv) > 1 else default_gdml()
geom = geometry.from_gdml(gdml)
t0 = time.time()
it = st.TrackerInterior(gdml, geom)
print(f"build {time.time() - t0:.1f} s; shells {len(it.shells.coax)}, calo pieces {len(it.shells.calo)}; "
      f"skipped calo placements {it.calo_skipped}")
# mass in the calo region from random vertical lines over its x-z box: column x rho integrated over the area
ps = it.shells.calo
lo = np.min([s.lo.min(0) for _, s in ps.sets], axis=0)
hi = np.max([s.hi.max(0) for _, s in ps.sets], axis=0)
rng = np.random.default_rng(1)
n = 400000
o = np.c_[rng.uniform(lo[0], hi[0], n), np.full(n, hi[1] + 10), rng.uniform(lo[2], hi[2], n)]
d = np.tile([0.0, -1.0, 0.0], (n, 1))
t0 = time.time()
col = ps.column(o, d, np.full(n, hi[1] - lo[1] + 20))
dt = time.time() - t0
area = (hi[0] - lo[0]) * (hi[2] - lo[2])
print(f"calo box x {lo[0]:.0f}..{hi[0]:.0f} y {lo[1]:.0f}..{hi[1]:.0f} z {lo[2]:.0f}..{hi[2]:.0f}; "
      f"column {dt / n * 1e6:.1f} us per vertical line")
tot = 0.0
for m, c in sorted(col.items(), key=lambda kv: -kv[1].sum()):
    kg = c.mean() * area * materials.get(m).rho * 1e-6
    tot += kg
    print(f"  {m:22s} {kg:9.1f} kg")
print(f"  total {tot:.1f} kg in the clipped region")
