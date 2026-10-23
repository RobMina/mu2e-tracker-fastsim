"""Boolean (subtraction/union) volumes the stage C world keeps as their first constituent: per volume, the material
column along vertical lines over the tracker footprint (+ margin) first-only (world.load) vs exact (world.signed_local),
and the mass difference over the sampled area. Answers whether the first-only approximation matters above the tracker.

    python3 tracker/scripts/world_boolean_check.py [GDML] [--margin MM] [--n N]
"""
import argparse
import re
import xml.etree.ElementTree as ET

import numpy as np

from mu2efastsim import materials, world as W
from trkfastsim import crossings, geometry, steps as st
from trkfastsim.analyses.steps import default_gdml

ap = argparse.ArgumentParser()
ap.add_argument("gdml", nargs="?", default=None)
ap.add_argument("--margin", type=float, default=8000.0)
ap.add_argument("--n", type=int, default=200000)
a = ap.parse_args()
gdml = a.gdml or default_gdml()
geom = geometry.from_gdml(gdml)
y_pl = st.plane_y(geom)
lo, hi = crossings.panel_boxes(geom)
x0, x1 = lo[:, 0].min() - a.margin, hi[:, 0].max() + a.margin
z0, z1 = lo[:, 2].min() - a.margin, hi[:, 2].max() + a.margin
root = ET.parse(gdml).getroot()
vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
solids = {s.get("name"): s for s in root.find("solids")}
vac = re.compile(r"AIR|Vacuum|VACUUM|Galactic")
rows = []


def walk(lv, R, t):
    for pv in vols[lv].findall("physvol"):
        ch = pv.find("volumeref").get("ref")
        p = pv.find("position")
        pos = np.array([float(p.get(k, 0)) for k in "xyz"]) if p is not None else np.zeros(3)
        R2, t2 = R @ W._rot(pv.find("rotation")), R @ pos + t
        mat = W._strip(vols[ch].find("materialref").get("ref"))
        s = solids[vols[ch].find("solidref").get("ref")]
        if s.tag in ("subtraction", "union") and not vac.search(mat):
            first = W._pieces_of(s, solids, R2, t2, mat, y_pl)
            loc = W.signed_local(s, solids)
            if first and loc is not None:
                exact = [(sg, mat, P, l, h) for sg, P, l, h in W._move(loc, R2, t2) if h[1] >= y_pl]
                rows.append((W._strip(ch), mat, s.tag, first, exact))
            elif first:
                rows.append((W._strip(ch), mat, s.tag, first, None))
        walk(ch, R2, t2)


walk(root.find("setup/world").get("ref"), np.eye(3), np.zeros(3))
rng = np.random.default_rng(1)
o = np.c_[rng.uniform(x0, x1, a.n), np.full(a.n, 20000.0), rng.uniform(z0, z1, a.n)]
d = np.tile([0.0, -1.0, 0.0], (a.n, 1))
L = np.full(a.n, 20000.0 - y_pl)
area = (x1 - x0) * (z1 - z0)
print(f"{len(rows)} boolean volumes above the plane; vertical lines over x {x0:.0f}..{x1:.0f} z {z0:.0f}..{z1:.0f}")
for name, mat, tag, first, exact in rows:
    f = W.PieceSet(first).column(o, d, L).get(mat, np.zeros(a.n))
    if not f.any():
        continue
    rho = materials.BY_NAME[mat].rho if mat in materials.BY_NAME else np.nan
    if exact is None:
        print(f"  {name:36s} {mat:18s} {tag:11s} first-only {f.mean() * area * rho * 1e-6:10.1f} kg  exact: unsupported")
        continue
    e = W.SignedPieceSet(exact).column(o, d, L).get(mat, np.zeros(a.n))
    hit = f > 0
    print(f"  {name:36s} {mat:18s} {tag:11s} first-only {f.mean() * area * rho * 1e-6:10.1f} kg  exact "
          f"{e.mean() * area * rho * 1e-6:10.1f} kg  lines hit {hit.mean():.3f}  mean column on hit {f[hit].mean():.0f} -> {e[hit].mean():.0f} mm")
