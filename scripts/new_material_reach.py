"""Which world-transport pieces (world.load, as the CRV and tracker stage C worlds) use a material that had no constants
before 2026-10-23 (then the CONCRETE_MARS default of the world column): count and mm of pieces per material.

    python3 tracker/scripts/new_material_reach.py [GDML]
"""
import sys

from mu2efastsim import world
from trkfastsim.analyses.steps import default_gdml

NEW = {"AluminumHoneycomb", "CarbonFiber", "CalCableRunOuter", "CalCableRunFiber", "TrkCableRunFiber", "TrkCableRunOuter",
       "G4_CESIUM_IODIDE", "Polyethylene092", "Polyetheretherketone", "Lyso_01", "StainlessSteel"}
gdml = sys.argv[1] if len(sys.argv) > 1 else default_gdml()
pcs = world.load(gdml)
hit = {}
for m, P, lo, hi in pcs:
    if m in NEW:
        e = hit.setdefault(m, [0, 1e9, -1e9])
        e[0] += 1
        e[1], e[2] = min(e[1], lo[1]), max(e[2], hi[1])
print(f"{len(pcs)} world pieces; with a newly defined material: {hit or 'none'}")
