"""Deterministic impact of a world modification on selected muons: the material column (g/cm2) along each selected
muon's generated line down to the plane above the tracker, before and after the modification (no random numbers).

    python3 tracker/scripts/worldmods_column_impact.py MODNAME "fast_as_v16/fasttrkC.*.root"
"""
import sys

import numpy as np

from mu2efastsim import materials, world as W
from mu2efastsim.tables import KEY, primary_muons
from trkfastsim import geometry, steps as st, tables as T, worldmods
from trkfastsim.analyses.steps import default_gdml

mod, pat = sys.argv[1], sys.argv[2]
paths = T.expand([pat])
s = T.read_many(paths, "sgsteps")
f = T.trk_filter(s[s.sp == 1])
g = primary_muons(T.read_many(paths, "gen")).merge(f[f.pass_trk].reset_index()[KEY], on=KEY)
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
    # as world_loss: materials without constants count as concrete in the stage C world column
    return sum(c * materials.BY_NAME.get(m, materials.CONCRETE_MARS).rho / 10.0 for m, c in col.items() if m != "StrawGas")


a, b = gcm2(pcs), gcm2(p2)
dd = b - a
ch = np.abs(dd) > 0.01
margin = p / (2.1 * np.maximum(a, 1e-3))
print(f"{mod}: selected muons {len(g)}; generated line column changed for {ch.sum()} ({ch.mean():.4f}); "
      f"change on those: mean {dd[ch].mean() if ch.any() else 0:+.2f} g/cm2, range {dd.min():+.2f}..{dd.max():+.2f}; "
      f"mean over all {dd.mean():+.4f} g/cm2 ({dd.mean() / a.mean():+.5f} relative)")
near = ch & (margin < 1.5)
print(f"  changed and near range-out (margin < 1.5): {near.sum()} muons; column change on them {dd[near].mean() if near.any() else 0:+.2f} g/cm2")
