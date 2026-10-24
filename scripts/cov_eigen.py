"""Conditioning of stock's front-piece covariance (trace cf, KinematicLine parameters d0 phi0 z0 theta t0 mom): smallest
eigenvalue of the correlation matrix and its direction, at the state entering stock's dying stage (stage d-1, full) and
after the first iteration of stage d (maxit 1, normal trace), for ParamsDiverged seeds vs stock-good seeds (stage 6/10).
python3 scripts/cov_eigen.py TRACE_ITER.root"""
import sys

import awkward as ak
import numpy as np
import uproot

P = ("d0", "phi0", "z0", "theta", "t0", "mom")
t = uproot.open(sys.argv[1])["kktrace/trace"].arrays(["subrun", "event", "stage", "status", "maxit", "cf"])
cf = ak.to_numpy(t.cf).reshape(-1, 6, 6)
rows = {}
for r in range(len(t)):
    rows[(int(t.subrun[r]), int(t.event[r]), int(t.stage[r]), int(t.maxit[r]))] = r


def cond(C):
    s = np.sqrt(np.diag(C))
    R = C / np.outer(s, s)
    w, v = np.linalg.eigh(R)
    return w[0], np.abs(v[:, 0])


seeds = {(k[0], k[1]) for k in rows}
pd, good = [], []
for s in seeds:
    st = {k[2]: rows[k] for k in rows if (k[0], k[1]) == s and k[3] < 0}
    dead = [k for k in sorted(st) if int(t.status[st[k]]) == 4]
    if dead:
        d = dead[0]
        for tag, r in (("entering", st.get(d - 1)), ("iter0", rows.get((s[0], s[1], d, 1)))):
            if r is not None:
                pd.append((tag, d) + cond(cf[r]))
    elif 12 in st and int(t.status[st[12]]) == 0:
        for d in (7, 11):
            for tag, r in (("entering", st.get(d - 1)), ("iter0", rows.get((s[0], s[1], d, 1)))):
                if r is not None:
                    good.append((tag, d) + cond(cf[r]))
for name, L in (("stock ParamsDiverged", pd), ("stock good (stages 7, 11)", good)):
    for tag in ("entering", "iter0"):
        w = np.array([x[2] for x in L if x[0] == tag])
        v = np.array([x[3] for x in L if x[0] == tag])
        if len(w) == 0:
            continue
        print(f"{name:26s} {tag:8s} n {len(w):4d}: min corr eigenvalue q10/50/90 "
              f"{np.quantile(w, .1):.2e} {np.quantile(w, .5):.2e} {np.quantile(w, .9):.2e}; "
              f"direction |v| q50: " + " ".join(f"{P[i]} {np.median(v[:, i]):.2f}" for i in range(6)))

# F1: final t0-piece covariance in KinematicLine parameters (yield.npz Ck_front), stock-PD vs stock-good seeds
if len(sys.argv) > 2:
    import os
    z = np.load(os.path.join(sys.argv[2], "yield.npz"))
    if "Ck_front" in z.files:
        w = np.array([cond(C)[0] for C in z["Ck_front"]])
        for name, m in (("F1 on stock-failed seeds", ~z["stockok"]), ("F1 on stock-good seeds", z["stockok"])):
            print(f"{name:26s} final    n {m.sum():4d}: min corr eigenvalue q10/50/90 "
                  f"{np.quantile(w[m], .1):.2e} {np.quantile(w[m], .5):.2e} {np.quantile(w[m], .9):.2e}")
    full = {(int(a), int(b)): r for (a, b, s_, m_), r in rows.items() if s_ == 12 and m_ < 0}
    f1 = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(z["subrun"], z["event"]))}
    pairs = [(cond(cf[r])[0], cond(z["Ck_front"][f1[k]])[0]) for k, r in full.items()
             if k in f1 and int(t.status[r]) == 0]
    if pairs:
        p = np.array(pairs)
        print(f"same seeds, stage 12 (stock good): min corr eigenvalue stock q50 {np.median(p[:, 0]):.2e}, F1 q50 "
              f"{np.median(p[:, 1]):.2e}; ratio F1/stock q16/50/84 {np.round(np.quantile(p[:, 1] / p[:, 0], [.16, .5, .84]), 2)}")
