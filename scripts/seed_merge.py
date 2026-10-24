"""Sum seedcheck counts over batches (seedcheck.json files): per variant and subset, G4 passes, seeded fast passes,
straight-line passes, and the paired fast/G4 - 1 with error sqrt(G4-only + fast-only) / G4.

    python3 tracker/scripts/seed_merge.py "seed_v2_*/seedcheck.json" [...]
"""
import json
import sys

import numpy as np

from trkfastsim import tables as T

runs = [json.load(open(f)) for p in sys.argv[1:] for f in T.expand([p])]
print(f"{len(runs)} batches")
for v in ("plane", "envelope"):
    for sub in ("all", "crv"):
        t = [r["variants"][v][sub]["counts"]["total"] for r in runs]
        g, s, l, go, so = (sum(x[k] for x in t) for k in ("g4_pass", "seed_pass", "straight_pass", "g4_only", "seed_only"))
        print(f"  {v:8s} {sub:3s}  G4 {g:6d}  seeded {s:6d} ({s / g - 1:+.4f} +- {np.sqrt(go + so) / g:.4f})  "
              f"straight {l:6d} ({l / g - 1:+.4f})")
