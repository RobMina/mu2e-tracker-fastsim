"""Seeded deficit (seedcheck events tables, plane seed, CRV-term subset) by |p| at the seed and by G4 straw count:
G4 passes, seeded passes, straight passes, G4-only and seed-only events, and for G4-only events the seeded fast count
and the straight-line count (how far below 12 they fall).

    python3 tracker/scripts/seed_breakdown.py "seed_v2_*/seedcheck_events.csv.gz"
"""
import sys

import numpy as np
import pandas as pd

from trkfastsim import tables as T

ev = pd.concat([pd.read_csv(f) for p in sys.argv[1:] for f in T.expand([p])], ignore_index=True)
ev = ev[(ev.variant == "plane") & ev.crv_pass]
print(f"plane seed, CRV subset: {len(ev)} seeds; G4 {int(ev.g4_pass.sum())}, seeded {int(ev.seed_pass.sum())}, "
      f"straight {int(ev.straight_pass.sum())}")
edges = [0, 300, 500, 1000, 2000, 5000, 1e9]
b = np.clip(np.digitize(ev.p_seed, edges) - 1, 0, len(edges) - 2)
for k in range(len(edges) - 1):
    x = ev[b == k]
    g, s, l = int(x.g4_pass.sum()), int(x.seed_pass.sum()), int(x.straight_pass.sum())
    go, so = int((x.g4_pass & ~x.seed_pass).sum()), int((x.seed_pass & ~x.g4_pass).sum())
    print(f"  p {edges[k]:5.0f}-{min(edges[k + 1], 1e5):6.0f}: G4 {g:5d} seeded {s:5d} ({s / max(g, 1) - 1:+.3f} +- "
          f"{np.sqrt(go + so) / max(g, 1):.3f}) straight {l:5d} ({l / max(g, 1) - 1:+.3f})")
go = ev[ev.g4_pass & ~ev.seed_pass]
so = ev[ev.seed_pass & ~ev.g4_pass]
h = lambda c: np.bincount(np.clip(c, 0, 30).astype(int), minlength=31)
print("G4-only: seeded count 0..11:", h(go.n_seed)[:12].tolist(), "| straight count 0..11/12+:", h(go.n_straight)[:12].tolist(),
      int((go.n_straight >= 12).sum()))
print("seed-only: G4 count 0..11:", h(so.ngood_g4)[:12].tolist())
print("G4-only with straight >= 12 (fast transport lost it):", int((go.n_straight >= 12).sum()), "of", len(go),
      "; seed-only with straight < 12:", int((so.n_straight < 12).sum()), "of", len(so))
both = ev[ev.g4_pass & ev.seed_pass]
print(f"both: mean straws G4 {both.ngood_g4.mean():.3f}, seeded {both.n_seed.mean():.3f}, straight {both.n_straight.mean():.3f}")
# below 500 MeV/c: G4-only events with seeded count 0 (the fast muon never reached a straw) vs a short count
lo = go[go.p_seed < 500]
print(f"G4-only below 500 MeV/c: {len(lo)}; seeded count 0: {int((lo.n_seed == 0).sum())}, 1-11: {int(((lo.n_seed > 0) & (lo.n_seed < 12)).sum())}; "
      f"fast min |p| < 75 MeV/c: {int((lo.fast_pmin < 75).sum())}; G4 straws q50 {lo.ngood_g4.median():.0f}")
