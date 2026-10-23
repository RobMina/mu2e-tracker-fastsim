"""Seeded pass counts of several seedcheck runs per configuration, averaged over rng seeds (mean and seed-to-seed rms).

    python3 tracker/scripts/seedcheck_seeds.py CONFIG=DIR1,DIR2,... [CONFIG=...]
"""
import json
import sys

import numpy as np

for arg in sys.argv[1:]:
    name, dirs = arg.split("=", 1)
    runs = [json.load(open(f"{d}/seedcheck.json")) for d in dirs.split(",")]
    for v in ("plane", "envelope"):
        for sub in ("all", "crv"):
            t = [r["variants"][v][sub]["counts"] for r in runs]
            g4 = t[0]["total"]["g4_pass"]
            sp = np.array([x["total"]["seed_pass"] for x in t], float)
            lo = np.array([sum(b["seed_pass"] for b in x["by_p_seed"] if b["p"][1] <= 1000) for x in t], float)
            g4lo = sum(b["g4_pass"] for b in t[0]["by_p_seed"] if b["p"][1] <= 1000)
            st = np.array([x["total"]["mean_straws_g4_seed_straight"][1] for x in t])
            print(f"{name:10s} {v:8s} {sub:3s} seed/g4 {sp.mean() / g4 - 1:+.4f} +- {sp.std(ddof=1) / g4 if len(sp) > 1 else 0:.4f} "
                  f"(n={len(sp)})  <1 GeV/c {lo.mean() / g4lo - 1:+.4f}  straws {st.mean():.3f} (G4 "
                  f"{t[0]['total']['mean_straws_g4_seed_straight'][0]})")
