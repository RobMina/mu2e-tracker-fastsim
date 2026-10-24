"""Front momentum iteration by iteration at stock's ParamsDiverged stage: stock without the parameter test
(trace_kkline_mc_iter_nopd.fcl rows with maxit 1..n, then the full stage) and with it (the normal trace), vs F1
(yield.npz mhist/shist: front momentum and status after each iteration of every stage), for a few seeds.
python3 scripts/mom_iter.py TRACE_ITER.root TRACE_NOPD.root F1_YIELD_DIR [nseeds]"""
import os
import sys

import awkward as ak
import numpy as np
import uproot


def load(p):
    t = uproot.open(p)["kktrace/trace"].arrays(["subrun", "event", "stage", "status", "maxit", "pf", "hstatus", "hmiter"])
    rows = {}
    for r in range(len(t)):
        rows[(int(t.subrun[r]), int(t.event[r]), int(t.stage[r]), int(t.maxit[r]))] = r
    return t, rows


a, ra = load(sys.argv[1])
b, rb = load(sys.argv[2])
z = np.load(os.path.join(sys.argv[3], "yield.npz"))
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
f1 = {(int(s), int(e)): i for i, (s, e) in enumerate(zip(z["subrun"], z["event"]))}
seeds = sorted({(k[0], k[1]) for k in ra})
shown = 0
summary = []
for s in seeds:
    full = {k[2]: r for k, r in ra.items() if (k[0], k[1]) == s and k[3] < 0}
    dead = [k for k in sorted(full) if int(a.status[full[k]]) == 4]
    if not dead or s not in f1:
        continue
    d = dead[0]
    its = sorted((k[3], r) for k, r in rb.items() if (k[0], k[1]) == s and k[2] == d and k[3] >= 1)
    fullb = rb.get((s[0], s[1], d, -1))
    mb = [float(b.pf[r][5]) for _, r in its] + ([float(b.pf[fullb][5])] if fullb is not None else [])
    prev = rb.get((s[0], s[1], d - 1, -1))
    m_in = float(b.pf[prev][5]) if prev is not None else np.nan
    i = f1[s]
    sh = z["shist"][i, d]; mh = z["mhist"][i, d]
    mf = [float(m) for m, q in zip(mh, sh) if q >= 0]
    f1prev = [float(m) for m, q in zip(z["mhist"][i, d - 1], z["shist"][i, d - 1]) if q >= 0]
    summary.append((m_in, mb, f1prev[-1] if f1prev else np.nan, mf))
    if shown < n:
        shown += 1
        print(f"seed subrun {s[0]} event {s[1]}: stock ParamsDiverged at stage {d}; F1 good {bool(z['good'][i])}")
        print(f"   entering stage {d}: stock mom {m_in:.4f}, F1 mom {summary[-1][2]:.4f}")
        print("   stock (no test) after each iteration: " + " ".join(f"{m:.4f}" for m in mb))
        print("   F1 after each iteration:            " + " ".join(f"{m:.4f}" for m in mf))
# aggregate: first-iteration momentum step at the dying stage, stock vs F1
st1 = np.array([s_[1][0] - s_[0] for s_ in summary if s_[1] and np.isfinite(s_[0])])
f11 = np.array([s_[3][0] - s_[2] for s_ in summary if s_[3] and np.isfinite(s_[2])])
print(f"seeds {len(summary)}: first-iteration momentum step at stock's dying stage: stock |q50| {np.median(np.abs(st1)):.4f}"
      f" q90 {np.quantile(np.abs(st1), .9):.4f}; F1 |q50| {np.median(np.abs(f11)):.4f} q90 {np.quantile(np.abs(f11), .9):.4f} MeV/c")
ent = np.array([s_[0] - s_[2] for s_ in summary if np.isfinite(s_[0]) and np.isfinite(s_[2])])
print(f"momentum entering that stage, stock - F1: q16/50/84 {np.round(np.quantile(ent, [.16, .5, .84]), 4)} MeV/c")
