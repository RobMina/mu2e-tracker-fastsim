"""Where the selected-rate deficit sits: per survivalcheck bin (energy margin; generated |p| x column), the fast-G4
difference in selected events per live time scaled to the G4 live time, as a share of the total difference.

    python3 tracker/scripts/deficit_budget.py SURVIVALCHECK_DIR/survivalcheck.json
"""
import json
import sys

d = json.load(open(sys.argv[1]))
lt_f, lt_g = d["livetime_s"]
nf, ng = d["selected"]
tot = nf * lt_g / lt_f - ng
print(f"selected fast {nf} G4 {ng}; difference {tot:+.0f} events ({tot / ng:+.2%})")
for blk in d["by_margin"]:
    if blk["sample"] != "all":
        continue
    print("by energy margin:")
    for r in blk["rows"]:
        a, b = r["n"]
        dd = a * lt_g / lt_f - b
        print(f"  {str(r['margin']):>16s}  fast {a:6d} G4 {b:6d}  diff {dd:+7.0f} ({r['ratio_minus_1']:+.3f} +- {r['err']:.3f})  "
              f"share of total {dd / tot:6.1%}  of G4 rate {dd / ng:+.2%}")
print("by generated |p| x column (cells with |diff| > 50):")
rows = d["by_p_gen_column"]
for r in rows:
    for c in r.get("cells", [r]):
        if "n" not in c:
            continue
        a, b = c["n"]
        dd = a * lt_g / lt_f - b
        if abs(dd) > 50:
            print(f"  p {r.get('p_gen', c.get('p_gen'))} col {c.get('gcm2', c.get('column'))}: fast {a} G4 {b} diff {dd:+.0f} "
                  f"share {dd / tot:.1%}")
