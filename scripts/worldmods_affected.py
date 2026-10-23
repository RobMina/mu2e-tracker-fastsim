"""Pass rate of the muons whose generated-line column a world modification changes, baseline vs modified run (same
primaries and seeds), split by the size of the column change. Muons selected in either arm.

    python3 tracker/scripts/worldmods_affected.py MODNAME "A/fasttrkC.*.root" "B/fasttrkC.*.root"
"""
import sys

import numpy as np

from mu2efastsim import materials, world as W
from mu2efastsim.tables import KEY, primary_muons
from trkfastsim import geometry, steps as st, tables as T, worldmods
from trkfastsim.analyses.steps import default_gdml

mod, pa, pb = sys.argv[1:4]


def passed(pat):
    s = T.read_many(T.expand([pat]), "sgsteps")
    f = T.trk_filter(s[s.sp == 1])
    return f[f.pass_trk].reset_index()[KEY]


A, B = passed(pa), passed(pb)
g = primary_muons(T.read_many(T.expand([pa]), "gen"))
g = g.merge(A.assign(ok_a=True), on=KEY, how="left").merge(B.assign(ok_b=True), on=KEY, how="left")
g = g[g.ok_a.notna() | g.ok_b.notna()].fillna({"ok_a": False, "ok_b": False}).reset_index(drop=True)
p = np.sqrt(g.px ** 2 + g.py ** 2 + g.pz ** 2).to_numpy()
o, d = g[["x", "y", "z"]].to_numpy(), g[["px", "py", "pz"]].to_numpy() / p[:, None]
gdml = default_gdml()
geom = geometry.from_gdml(gdml)
y_pl = st.plane_y(geom)
L = (o[:, 1] - y_pl) / -d[:, 1]
nm = []
pcs = W.load(gdml, y_min=y_pl, names=nm)
p2, _ = worldmods.apply(mod, list(pcs), list(nm))


def gcm2(pieces):
    col = W.PieceSet(pieces).build_grid(500.0).column(o, d, L)
    return sum(c * materials.BY_NAME.get(m, materials.CONCRETE_MARS).rho / 10.0 for m, c in col.items() if m != "StrawGas")


dd = gcm2(p2) - gcm2(pcs)
for lab, sel in (("unchanged", np.abs(dd) <= 0.01), ("-15..-0.01", (dd < -0.01)), ("+0.01..+50", (dd > 0.01) & (dd <= 50)),
                 ("+50..+150", (dd > 50) & (dd <= 150)), ("> +150", dd > 150)):
    x = g[sel]
    na, nb = int(x.ok_a.sum()), int(x.ok_b.sum())
    ao, bo = int((x.ok_a & ~x.ok_b).sum()), int((x.ok_b & ~x.ok_a).sum())
    print(f"  column change {lab:>11s}: muons {len(x):6d}  A {na:6d}  B {nb:6d}  B/A-1 {nb / max(na, 1) - 1:+.4f} "
          f"+- {np.sqrt(ao + bo) / max(na, 1):.4f}  (A-only {ao}, B-only {bo})")
