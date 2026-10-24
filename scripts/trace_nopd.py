"""Stock with and without the parameter-change test (trace_kkline_mc_iter.fcl vs trace_kkline_mc_iter_nopd.fcl):
for every stock ParamsDiverged seed, the diverged iteration accepted in the no-test run gives the trajectory jump the
test rejected. Reports KinKal's front/back dp chi2 (against the previous iteration's piece, with and without the
momentum), the parameters that move, and the final outcome of the no-test fit.
python3 scripts/trace_nopd.py TRACE_ITER.root TRACE_NOPD.root"""
import sys

import awkward as ak
import numpy as np
import uproot

P = ("d0", "phi0", "z0", "theta", "t0", "mom")
cols = ["subrun", "event", "stage", "status", "maxit", "pf", "cf", "pb", "cb", "chi2", "ndof"]


def load(p):
    t = uproot.open(p)["kktrace/trace"].arrays(cols)
    d = {c: ak.to_numpy(t[c]) for c in cols if c not in ("pf", "cf", "pb", "cb")}
    d["pf"] = ak.to_numpy(t.pf).reshape(-1, 6); d["cf"] = ak.to_numpy(t.cf).reshape(-1, 6, 6)
    d["pb"] = ak.to_numpy(t.pb).reshape(-1, 6); d["cb"] = ak.to_numpy(t.cb).reshape(-1, 6, 6)
    idx = {}
    for r in range(len(d["stage"])):
        idx[(int(d["subrun"][r]), int(d["event"][r]), int(d["stage"][r]), int(d["maxit"][r]))] = r
    return d, idx


a, ia = load(sys.argv[1])
b, ib = load(sys.argv[2])
seeds = {(k[0], k[1]) for k in ia}
res, finals = [], []
for s in seeds:
    full = {k[2]: r for k, r in ia.items() if (k[0], k[1]) == s and k[3] < 0}
    dead = [k for k in sorted(full) if a["status"][full[k]] == 4]
    if not dead:
        continue
    d = dead[0]
    # reference: stock's last accepted iteration of stage d (largest maxit with a usable status), else stage d-1
    its = sorted((k[3], r) for k, r in ia.items() if (k[0], k[1]) == s and k[2] == d and k[3] >= 1)
    ok = [r for m, r in its if a["status"][r] < 2]
    r0 = ok[-1] if ok else full.get(d - 1)
    if r0 is None:
        continue
    m0 = a["maxit"][r0] if ok else 0
    # the no-test run with the same truncation one iteration further: the iteration stock rejected
    r1 = ib.get((s[0], s[1], d, m0 + 1)) if ok else ib.get((s[0], s[1], d, 1))
    if r1 is None:
        r1 = ib.get((s[0], s[1], d, -1))
    if r1 is None:
        continue
    out = {}
    for tag in ("f", "b"):
        dp = b["p" + tag][r1] - a["p" + tag][r0]
        dp[1] = (dp[1] + np.pi) % (2 * np.pi) - np.pi
        C = a["c" + tag][r0]
        out[tag] = (float(dp @ np.linalg.solve(C, dp)), float(dp[:5] @ np.linalg.solve(C[:5, :5], dp[:5])),
                    dp / np.sqrt(np.diag(C)), dp[5], np.sqrt(C[5, 5]))
    res.append((s, d, out))
    fb = ib.get((s[0], s[1], 12, -1))
    finals.append(int(b["status"][fb]) if fb is not None else -1)
print(f"stock ParamsDiverged seeds: {len(res)}")
for tag, name in (("f", "front"), ("b", "back")):
    tot = np.array([r[2][tag][0] for r in res]); nm = np.array([r[2][tag][1] for r in res])
    pul = np.array([r[2][tag][2] for r in res])
    print(f"{name}: dp chi2 q16/50/84 {np.round(np.quantile(tot, [.16, .5, .84]), 1)}; over 1e6 {np.mean(tot > 1e6) * 100:.0f} %;"
          f" without mom q50 {np.quantile(nm, .5):.3g}, over 1e6 {np.mean(nm > 1e6) * 100:.0f} %")
    print("   |dp|/sigma q50: " + ", ".join(f"{P[i]} {np.median(np.abs(pul[:, i])):.3g}" for i in range(6)))
    big = tot > 1e6
    dmab = np.array([r[2][tag][3] for r in res]); sm = np.array([r[2][tag][4] for r in res])
    print(f"   dmom q16/50/84 {np.round(np.quantile(dmab, [.16, .5, .84]), 3)} MeV/c; sigma(mom) ref q50 {np.median(sm):.2e}"
          + (f"; on chi2 > 1e6: |dmom| q50 {np.median(np.abs(dmab[big])):.3g} MeV/c, sigma(mom) q50 {np.median(sm[big]):.2e}" if big.any() else ""))
    if big.any():
        print("   on dp chi2 > 1e6: |dp|/sigma q50: " + ", ".join(f"{P[i]} {np.median(np.abs(pul[big, i])):.3g}" for i in range(6)))
names = {-1: "none", 0: "conv", 1: "unconv", 2: "lowNDOF", 3: "gap", 4: "params", 5: "chisq", 7: "failed"}
print("final status of these seeds without the test:", {names.get(k, k): int(v) for k, v in zip(*np.unique(finals, return_counts=True))})
