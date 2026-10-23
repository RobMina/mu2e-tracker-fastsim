"""Parameter-change test diagnostic from reco-vr2f1 --own-states (yield.npz): per stage, the largest KinKal
'DivergenceDeltaParams' chi2 (parameter change vs the previous fit at front and back, on the previous covariance) and
the momentum change at that iteration, split by stock FitOK / failed. Joined with the stock fit history if given.
python3 scripts/f1_pardiv.py DIR [stock_history.npz]"""
import os
import sys

import numpy as np

z = np.load(os.path.join(sys.argv[1], "yield.npz"))
s = z["stockok"]
dp, dm = z["dpmax"], z["dmomf"]
q = lambda a: " ".join(f"{x:9.3g}" for x in np.quantile(a, [.5, .9, .99]))
print("stage  | dpchi2 q50/q90/q99 stock OK      | stock failed                   | dmom at max (MeV/c) failed q50/q90")
for k in range(dp.shape[1]):
    print(f"{k:5d}  | {q(dp[s, k])} | {q(dp[~s, k])} | {np.quantile(np.abs(dm[~s, k]), .5):.3g} {np.quantile(np.abs(dm[~s, k]), .9):.3g}")
thr = np.r_[np.full(4, 1e8), np.full(9, 1e6)]
over = dp > thr[None, :]
print("tracks with dpchi2 over threshold in any stage: stock OK", int(over[s].any(1).sum()), "/", int(s.sum()),
      " stock failed", int(over[~s].any(1).sum()), "/", int((~s).sum()))
