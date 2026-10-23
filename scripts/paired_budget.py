"""Paired fast vs G4 tracker-term rate on the same CRY primaries (fatecheck events: every generated primary muon, fast
rerun, G4 pass from the PrimaryFilter-kept reference), total and per energy-margin bin, with the paired error
sqrt(G4-only + fast-only) / G4. Compare with the unpaired 10x deficit against prod_as20 (survivalcheck).

    python3 tracker/scripts/paired_budget.py FATECHECK_DIR/fatecheck_events.csv.gz "gen.001400_*.root"
    python3 tracker/scripts/paired_budget.py "fate_v2_*/fatecheck_events.csv.gz,fate_v16/fatecheck_events.csv.gz" "GENGLOB1,GENGLOB2"
(comma-separated lists combine several fatecheck runs and gen-table sets)
"""
import sys

import numpy as np
import pandas as pd

from mu2efastsim.tables import KEY, primary_muons
from trkfastsim import geometry, steps as st, tables as T
from trkfastsim.analyses.steps import default_gdml
from trkfastsim.analyses.worldcheck import _column

EDGES = np.array([0, 1, 1.25, 1.5, 2, 3, 5, 1e9])
ev = pd.concat([pd.read_csv(f) for pat in sys.argv[1].split(",") for f in T.expand([pat])], ignore_index=True)
ev = ev[ev.fast_pass | ev.g4_pass].copy()
gen = primary_muons(T.read_many(T.expand(sys.argv[2].split(",")), "gen")).merge(ev[KEY], on=KEY)
gdml = default_gdml()
geom = geometry.from_gdml(gdml)
gen["gcm2"], _ = _column(gen, st.load_world(gdml, geom), st.plane_y(geom))
ev = ev.merge(gen[KEY + ["gcm2"]], on=KEY)
ev["margin"] = ev.p_gen / (2.1 * np.maximum(ev.gcm2, 1e-3))


def line(g, lab):
    a, b = int(g.g4_pass.sum()), int(g.fast_pass.sum())
    go, fo = int((g.g4_pass & ~g.fast_pass).sum()), int((g.fast_pass & ~g.g4_pass).sum())
    print(f"{lab:>16s}  G4 {a:5d} fast {b:5d}  fast/G4-1 {b / max(a, 1) - 1:+.4f} +- {np.sqrt(go + fo) / max(a, 1):.4f}"
          f"  (G4-only {go}, fast-only {fo})  diff {b - a:+5d}")


print(f"events with a pass in either arm: {len(ev):,}")
line(ev, "all")
b = np.clip(np.digitize(ev.margin, EDGES) - 1, 0, len(EDGES) - 2)
for k in range(len(EDGES) - 1):
    line(ev[b == k], f"[{EDGES[k]:g}, {EDGES[k + 1]:g})")
