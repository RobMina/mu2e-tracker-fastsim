"""Momentum error and momentum change per stage, stock (trace, full stages) vs F1 (reco-vr2f1 --own-states yield.npz with
smomf/smref/mterm/dmomf/dpmax), split into stock-good and stock-ParamsDiverged seeds.
python3 scripts/mom_compare.py TRACE.root F1_YIELD_DIR"""
import os
import sys

import awkward as ak
import numpy as np
import uproot

t = uproot.open(sys.argv[1])["kktrace/trace"].arrays(["subrun", "event", "stage", "status", "maxit", "cf", "pf", "pb"])
z = np.load(os.path.join(sys.argv[2], "yield.npz"))
full = ak.to_numpy(t.maxit) < 0
key = np.stack([ak.to_numpy(t.subrun), ak.to_numpy(t.event)], 1)[full]
st, status = ak.to_numpy(t.stage)[full], ak.to_numpy(t.status)[full]
cf = ak.to_numpy(t.cf).reshape(-1, 6, 6)[full]
pf, pb = ak.to_numpy(t.pf).reshape(-1, 6)[full], ak.to_numpy(t.pb).reshape(-1, 6)[full]
uk, inv = np.unique(key, axis=0, return_inverse=True)
S = np.full((len(uk), 13), -9); SM = np.full((len(uk), 13), np.nan); FB = np.full((len(uk), 13), np.nan)
S[inv, st] = status; SM[inv, st] = np.sqrt(cf[:, 5, 5]); FB[inv, st] = pf[:, 5] - pb[:, 5]
good = S[:, 12] == 0
pd = (S == 4).any(1)
f1 = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(z["subrun"], z["event"]))}
fi = np.array([f1.get((int(a), int(b)), -1) for a, b in uk])
q = lambda a: f"{np.nanquantile(a, .5):.2e}" if np.isfinite(a).any() else "  -  "
print("stage | stock sigma(mom_front) good / PD-seeds | F1 sigma(mom_front) good / PD | stock front-back mom good | F1")
for k in range(13):
    m1 = fi[good & (fi >= 0)]; m2 = fi[pd & (fi >= 0)]
    print(f"{k:5d} | {q(SM[good, k])} / {q(SM[pd, k])} | {q(z['smomf'][m1, k])} / {q(z['smomf'][m2, k])} | "
          f"{np.nanmedian(FB[good, k]):.3f} | {np.nanmedian(z['dmfb'][m1, k]):.3f}")
# F1's largest parameter-change chi2 and its momentum term per stage on the stock-PD seeds
m2 = fi[pd & (fi >= 0)]
print("F1 on stock-PD seeds, per stage: max dp chi2 q50 / mom term q50 / sigma(mom) of the reference q50 / dmom q50")
for k in range(4, 13):
    print(f"  {k:2d}: {np.median(z['dpmax'][m2, k]):.3g} / {np.median(z['mterm'][m2, k]):.3g} / {np.median(z['smref'][m2, k]):.2e}"
          f" / {np.median(np.abs(z['dmomf'][m2, k])):.3g}")
