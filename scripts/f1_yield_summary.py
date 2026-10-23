"""Summarise reco-vr2f1 --own-states yield.npz: F1 vs stock FitOK confusion and what the disagreeing seeds look like.
python3 scripts/f1_yield_summary.py DIR"""
import os
import sys

import numpy as np

z = np.load(os.path.join(sys.argv[1], "yield.npz"))
g, s = z["good"], z["stockok"]
print(f"seeds {len(g)}: both {np.sum(g & s)}, F1 only {np.sum(g & ~s)}, stock only {np.sum(~g & s)}, neither {np.sum(~g & ~s)}"
      f"  -> F1/stock {g.sum() / max(s.sum(), 1):.4f}")
q = lambda a: np.round(np.quantile(a, [.16, .5, .84]), 2).tolist() if len(a) else None
for name, m in (("both", g & s), ("F1 only (stock failed)", g & ~s), ("stock only", ~g & s), ("neither", ~g & ~s)):
    if m.sum() == 0:
        continue
    print(f"{name:24s} n {m.sum():5d}  seed hits {q(z['ninseed'][m])}  all hits {q(z['nh'][m])}  active F1 {q(z['nact_f1'][m])} "
          f"stock {q(z['nact_stock'][m])}  drift F1 {q(z['ndrift_f1'][m])} stock {q(z['ndrift_stock'][m])}  chi2/ndf F1 {q(z['cpn_f1'][m])}"
          f" stock {q(z['cpn_stock'][m])}")
m = ~s
print("stock failures: stock drift hits == 0:", f"{np.mean(z['ndrift_stock'][m] == 0) * 100:.1f} %;",
      "F1 drift hits == 0:", f"{np.mean(z['ndrift_f1'][m] == 0) * 100:.1f} %;",
      "stock active < 5:", f"{np.mean(z['nact_stock'][m] < 5) * 100:.1f} %")
st = z["stat_all"]
print("F1 status per stage on stock failures (fraction unusable, >= 2):", np.round((st[m] >= 2).mean(0), 3).tolist())
print("F1 status per stage on stock FitOK   (fraction unusable, >= 2):", np.round((st[s] >= 2).mean(0), 3).tolist())
