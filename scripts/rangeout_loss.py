"""Energy-loss distribution near range-out: the stage C loss model (transport.slab_loss, the code world_loss calls per
material) against G4 slabs. Muons crossing one G4_CONCRETE slab with no scattering on the fast side: the total
kinetic-energy loss of every muon (stoppers count as losing their whole kinetic energy) against G4's (stopped muons:
end inside the slab or kinetic energy < 1 MeV), quantiles of loss / initial kinetic energy, transmission, and the
mean range. Separates straggling (the loss distribution) from path effects (G4 muons scatter, so their path is longer
than the slab).

    python3 tracker/scripts/rangeout_loss.py slab/rangeout/slab.t300_p*.root [...]
"""
import os
import re
import sys

import numpy as np

from mu2efastsim import eloss
from mu2efastsim import transport as tr
from trkfastsim import tables as T
from trkfastsim.analyses.slabcheck import G4_CONCRETE, RHO

MMU = eloss.MMU
print(f"{'g/cm2':>5s} {'p':>5s} | trans G4 / fast | G4 path/slab q50/q90 (transmitted) | "
      f"loss/KE0 q10/q50/q90 G4 | fast (no scattering)")
for path in sys.argv[1:]:
    m = re.search(r"t(\d+(?:\.\d+)?)_p(\d+)", os.path.basename(path))
    tg, pm = float(m.group(1)), int(m.group(2))
    thick = tg / RHO * 10.0
    pa = T.read(path, "particles")
    mu = pa[(pa.isprim == 1) & (pa.pdg.abs() == 13)]
    ke0 = np.sqrt(pm ** 2 + MMU ** 2) - MMU
    pe = np.sqrt(mu.epx ** 2 + mu.epy ** 2 + mu.epz ** 2).to_numpy()
    out = (mu.epz.to_numpy() > 0) & (mu.ez.to_numpy() > 0.5 * thick - 1.0)
    ke_end = np.where(out, np.sqrt(pe ** 2 + MMU ** 2) - MMU, 0.0)
    g_loss = (ke0 - ke_end) / ke0
    # G4 path length through the slab for transmitted muons: from the end point, straight-line lower bound is thick
    dz = mu.ez.to_numpy() - mu.sz.to_numpy()
    seg = np.sqrt((mu.ex - mu.sx).to_numpy() ** 2 + (mu.ey - mu.sy).to_numpy() ** 2 + dz ** 2)
    rng = np.random.default_rng(1)
    n = 30000
    lost = tr.slab_loss(rng, G4_CONCRETE, np.full(n, float(pm)), np.full(n, thick))
    f_ok = ke0 - lost > tr.TK_STOP
    f_loss = np.where(f_ok, lost, ke0) / ke0
    q = lambda x: "/".join(f"{v:.3f}" for v in np.quantile(x, (0.1, 0.5, 0.9)))
    pr = (seg[out] / thick) if out.any() else np.array([np.nan])
    print(f"{tg:5.0f} {pm:5d} | {out.mean():.4f} / {f_ok.mean():.4f} | {np.nanquantile(pr, 0.5):.3f}/{np.nanquantile(pr, 0.9):.3f} | "
          f"{q(g_loss)} | {q(f_loss)}")
