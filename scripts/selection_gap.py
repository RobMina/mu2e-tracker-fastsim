"""Where the paired tracker-term deficit at a given energy-margin range comes from (fatecheck events, 11 M generated).

For events in [MLO, MHI) of energy margin that pass in exactly one arm: the fast fate (missed / few / stopped_above)
and good-step counts of both arms, how far the other arm's count is from the 12-straw threshold, G4's own good-step
count distribution near threshold (11-13) against fast's, and the primary muon's |p| at G4's first straw vs the fast
|p| at the plane. A deficit that sits in G4 counts of 12-13 against fast 9-11 points at the straw count per muon near
threshold (geometry/path in the tracker); 'missed' with G4 >> 12 at an aim or scattering difference.

    python3 tracker/scripts/selection_gap.py "fate_v2_*/fatecheck_events.csv.gz,fate_v16/..." "GENGLOB,..." "SGSTEPGLOB,..." [MLO MHI]
"""
import sys

import numpy as np
import pandas as pd

from mu2efastsim.tables import KEY, primary_muons
from trkfastsim import geometry, steps as st, tables as T
from trkfastsim.analyses.steps import default_gdml
from trkfastsim.analyses.worldcheck import _column

fate_pats, gen_pats, g4_pats = (a.split(",") for a in sys.argv[1:4])
mlo, mhi = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (2.0, 5.0)
ev = pd.concat([pd.read_csv(f) for p in fate_pats for f in T.expand([p])], ignore_index=True)
ev = ev[ev.fast_pass | ev.g4_pass].copy()
gen = primary_muons(T.read_many(T.expand(gen_pats), "gen")).merge(ev[KEY], on=KEY)
gdml = default_gdml()
geom = geometry.from_gdml(gdml)
gen["gcm2"], _ = _column(gen, st.load_world(gdml, geom), st.plane_y(geom))
ev = ev.merge(gen[KEY + ["gcm2"]], on=KEY)
ev["margin"] = ev.p_gen / (2.1 * np.maximum(ev.gcm2, 1e-3))
sel = ev[(ev.margin >= mlo) & (ev.margin < mhi)].copy()
# G4 per-muon good-step counts and |p| at the first straw, for the selected events
paths = T.expand(g4_pats)
keys = sel[KEY]
s = pd.concat([T.read(p, "sgsteps").merge(keys, on=KEY) for p in paths], ignore_index=True)
pa = pd.concat([T.read(p, "particles", KEY + ["key", "pdg", "ismcprim"]).merge(keys, on=KEY) for p in paths], ignore_index=True)
pm = pa[(pa.ismcprim == 1) & (pa.pdg.abs() == 13)].rename(columns={"key": "sp"})[KEY + ["sp"]]
s = s.merge(pm, on=KEY + ["sp"])
g = s[T.good_steps(s)].groupby(KEY).size().rename("ngood_g4").reset_index()
f1 = s.sort_values("t").groupby(KEY).first().reset_index()
f1["p_first_g4"] = np.sqrt(f1.px ** 2 + f1.py ** 2 + f1.pz ** 2)
sel = sel.merge(g, on=KEY, how="left").merge(f1[KEY + ["p_first_g4"]], on=KEY, how="left")
sel["ngood_g4"] = sel.ngood_g4.fillna(0).astype(int)
go, fo = sel[sel.g4_pass & ~sel.fast_pass], sel[sel.fast_pass & ~sel.g4_pass]
both = sel[sel.g4_pass & sel.fast_pass]
print(f"margin [{mlo}, {mhi}): events {len(sel)}; G4 {int(sel.g4_pass.sum())}, fast {int(sel.fast_pass.sum())}, "
      f"G4-only {len(go)}, fast-only {len(fo)}, both {len(both)}")
print("G4-only by fast fate:", go.fate.value_counts().to_dict())
print("fast-only: G4 kept the event" , int(fo.in_g4.sum()), "of", len(fo), "; G4 muon stopped above the plane",
      int((fo.g4_stopped.fillna(False) & (fo.ey_mu2e > st.plane_y(geom))).sum()))
edges = [0, 1, 6, 9, 11, 12, 13, 15, 20, 40, 200]
h = lambda x: np.histogram(x, edges)[0].tolist()
print(f"good-step bins {edges}")
print(f"  G4-only: G4 count {h(go.ngood_g4)}  fast count {h(go.ngood)}")
print(f"  fast-only: G4 count {h(fo.ngood_g4)}  fast count {h(fo.ngood)}")
print(f"  both: G4 count {h(both.ngood_g4)}  fast count {h(both.ngood)}")
print(f"  mean straws in both-pass events: G4 {both.ngood_g4.mean():.3f}, fast {both.ngood.mean():.3f}")
near = lambda d, c: int(((d[c] >= 12) & (d[c] <= 13)).sum())
print(f"  passes at 12-13 straws: G4-only events {near(go, 'ngood_g4')} of {len(go)}, fast-only {near(fo, 'ngood')} of {len(fo)}")
miss = go[go.fate == "missed"]
print(f"  G4-only 'missed': fast line miss distance at the plane q50/q90 {miss.miss_plane_mm.quantile(0.5):.0f}/"
      f"{miss.miss_plane_mm.quantile(0.9):.0f} mm (0 = aimed inside the footprint); share aimed inside {np.mean(miss.miss_plane_mm <= 0):.3f}")
