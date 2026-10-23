"""Near range-out through the world above the plane: does G4 select muons whose true path crosses less material than
their generated line? From geomcheck's paired event table (G4 side only, so the fast version it was made with does not
matter): per energy-margin bin (margin = p_gen / (2.1 MeV cm2/g x generated-line column), as survivalcheck), for G4
tracker-term events, the ratio of the fast-world column along the G4 trajectory to the one along the generated line,
and the G4 loss to the plane against the Bethe mean of the fast-world column along the generated line and along G4's
own path (the material model is right if the latter is ~1), the deflection and the lateral offset at the plane.

    python3 tracker/scripts/rangeout_paths.py GEOMCHECK_DIR/geomcheck_events.csv.gz
"""
import sys

import numpy as np
import pandas as pd

EDGES = [0, 1, 1.25, 1.5, 2, 3, 5, 1e9]
ev = pd.read_csv(sys.argv[1])
ev = ev[ev.g4_pass & ev.g4_crossed & (ev.gen_gcm2 > 1)].copy()
ev["margin"] = ev.p_gen / (2.1 * ev.gen_gcm2)
ev["path_over_line"] = ev.g4path_gcm2 / ev.gen_gcm2
ev["loss_over_bethe"] = ev.g4_loss_plane / ev.gen_bethe.clip(lower=1e-3)
ev["loss_over_path_bethe"] = ev.g4_loss_plane / ev.g4path_bethe.clip(lower=1e-3)
ev["b"] = pd.cut(ev.margin, EDGES, right=False)
q = lambda s: f"{s.quantile(0.1):.3f}/{s.median():.3f}/{s.quantile(0.9):.3f}"
print(f"G4 tracker-term events crossing the plane: {len(ev)}")
print(f"{'margin':>12s} {'n':>5s} | path/line col q10/q50/q90 | share path < 0.9 line | G4 loss / line Bethe q10/q50/q90 | "
      f"G4 loss / path Bethe q10/q50/q90 | defl q50 mrad | offset at plane q50/q90 mm")
for b, g in ev.groupby("b", observed=True):
    print(f"{str(b):>12s} {len(g):5d} | {q(g.path_over_line):>24s} | {np.mean(g.path_over_line < 0.9):8.3f} | "
          f"{q(g.loss_over_bethe):>30s} | {q(g.loss_over_path_bethe):>30s} | {1e3 * g.g4_theta_plane.median():6.0f} | "
          f"{g.g4_offset_plane.median():5.0f}/{g.g4_offset_plane.quantile(0.9):5.0f}")
