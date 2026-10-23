"""Drill into the F1-vs-stock yield disagreements from reco-vr2f1 --own-states: status per stage, the stage where F1
became unusable or where stock's fit likely died, and the stock status word bits (TrkFitFlag) of the seeds.
python3 scripts/f1_yield_drill.py DIR"""
import os
import sys

import numpy as np

z = np.load(os.path.join(sys.argv[1], "yield.npz"))
g, s = z["good"], z["stockok"]
st = z["stat_all"]
names = {0: "conv", 1: "unconv", 2: "lowNDOF", 3: "gap", 4: "params", 5: "chisq", 7: "failed"}
BITS = {5: "kalmanOK", 15: "FitOK", 11: "KalmanConverged"}
for name, m in (("F1 only", g & ~s), ("stock only", ~g & s), ("neither", ~g & ~s)):
    if not m.any():
        continue
    print(f"== {name}: {m.sum()}")
    last = st[m, -1]
    print("   F1 final status:", {names.get(k, k): int(v) for k, v in zip(*np.unique(last, return_counts=True))})
    firstbad = np.where((st[m] >= 2).any(1), (st[m] >= 2).argmax(1), -1)
    print("   F1 first unusable stage (-1 none; 0-3 seed fit, 4-12 extension):",
          {int(k): int(v) for k, v in zip(*np.unique(firstbad, return_counts=True))})
    kb = z["kl_status"][m]
    print("   stock bits:", {nm: int(((kb >> b) & 1).sum()) for b, nm in BITS.items()})
    print("   gseed false (F1 extension gate):", int((~z["gseed"][m]).sum()))

# join with the stock fit history (scripts/stock_fit_history.py OUT.npz) by event index, if given
if len(sys.argv) > 2 and "event_index" in z.files:
    h = np.load(sys.argv[2])
    hev = h["ev"]
    pos = np.searchsorted(hev, z["event_index"])
    have = (pos < len(hev)) & (hev[np.minimum(pos, len(hev) - 1)] == z["event_index"])
    fin = np.where(have, h["final"][np.minimum(pos, len(hev) - 1)], "none")
    met = np.where(have, h["meta"][np.minimum(pos, len(hev) - 1)], -1)
    for name, m in (("F1 only", g & ~s & have), ("neither", ~g & ~s & have), ("stock only", ~g & s & have)):
        if m.any():
            k, c = np.unique(np.char.add(np.char.add(fin[m].astype(str), " meta "), met[m].astype(str)), return_counts=True)
            print(f"   {name} by stock final status: " + ", ".join(f"{kk} {cc}" for kk, cc in sorted(zip(k, c), key=lambda x: -x[1])))
