"""Side-by-side summary of two seedcheck.json outputs (counts and G4/fast width ratios per variant and subset).

    python3 tracker/scripts/seedcheck_compare.py BEFORE/seedcheck.json AFTER/seedcheck.json [--variants plane envelope]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("before")
ap.add_argument("after")
ap.add_argument("--variants", nargs="+", default=["plane", "envelope"])
a = ap.parse_args()
B, A = json.load(open(a.before)), json.load(open(a.after))
for v in a.variants:
    for sub in ("all", "crv"):
        cb, ca = B["variants"][v][sub]["counts"], A["variants"][v][sub]["counts"]
        tb, ta = cb["total"], ca["total"]
        print(f"== {v}/{sub}: g4 {ta['g4_pass']}  seed {tb['seed_pass']} -> {ta['seed_pass']}  "
              f"seed/g4 {tb['seed_over_g4']} -> {ta['seed_over_g4']}  straight/g4 {tb['straight_over_g4']} -> "
              f"{ta['straight_over_g4']}  straws g4/seed/straight {tb['mean_straws_g4_seed_straight']} -> "
              f"{ta['mean_straws_g4_seed_straight']}")
        for rb, ra in zip(cb["by_p_seed"], ca["by_p_seed"]):
            print(f"   p {rb['p'][0]:6.0f}-{min(rb['p'][1], 1e5):6.0f}  g4 {ra['g4_pass']:5d}  seed {rb['seed_pass']:5d} -> "
                  f"{ra['seed_pass']:5d}  straight {ra['straight_pass']:5d}")
        wb = {(tuple(r["xx0"]), tuple(r["p"])): r for r in B["variants"][v][sub]["deflection_over_theta0"]}
        for r in A["variants"][v][sub]["deflection_over_theta0"]:
            k = (tuple(r["xx0"]), tuple(r["p"]))
            old = wb.get(k, {}).get("g4_over_fast_sigma")
            print(f"   xx0 {k[0]} p {k[1][0]:.0f}-{min(k[1][1], 1e5):.0f} n {r['n']:5d}  G4/fast {old} -> {r['g4_over_fast_sigma']}")
