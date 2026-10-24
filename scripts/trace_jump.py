"""Where stock's ParamsDiverged comes from (trace_kkline_mc_iter.fcl output): for each seed that dies with
ParamsDiverged at stage d, take the state after the last good iteration (the reference) and the diverged state, and
compute KinKal's parameter-change chi2 at the front, dp^T C_ref^-1 dp, with the contribution of each parameter
(d0, phi0, z0, theta, t0, mom) alone and of the momentum-free 5x5 block. Optional F1 yield dir: restrict to the seeds
whose F1 per-stage hit states equal stock's up to d (the 'identical states' set).
python3 scripts/trace_jump.py TRACE_ITER.root [F1_YIELD_DIR]"""
import sys

import awkward as ak
import numpy as np
import uproot

P = ("d0", "phi0", "z0", "theta", "t0", "mom")
t = uproot.open(sys.argv[1])["kktrace/trace"].arrays(["subrun", "event", "stage", "status", "maxit", "pf", "cf", "pb", "cb"])
key = np.stack([ak.to_numpy(t.subrun), ak.to_numpy(t.event)], 1)
stage, status, maxit = ak.to_numpy(t.stage), ak.to_numpy(t.status), ak.to_numpy(t.maxit)
pf = ak.to_numpy(t.pf).reshape(-1, 6); cf = ak.to_numpy(t.cf).reshape(-1, 6, 6)
pb = ak.to_numpy(t.pb).reshape(-1, 6); cb = ak.to_numpy(t.cb).reshape(-1, 6, 6)
uk, inv = np.unique(key, axis=0, return_inverse=True)
rows = {}
for r in range(len(t)):
    rows.setdefault(inv[r], []).append(r)
out = []
for s, rr in rows.items():
    rr = np.array(rr)
    full = rr[maxit[rr] < 0]
    if len(full) == 0:
        continue
    st = {int(stage[r]): int(status[r]) for r in full}
    dead = [k for k in sorted(st) if st[k] == 4]
    if not dead:
        continue
    d = dead[0]
    its = rr[(stage[rr] == d) & (maxit[rr] >= 1)]
    ok = [r for r in its if status[r] < 2]
    if not ok:
        ref = [r for r in full if stage[r] == d - 1]
        if not ref:
            continue
        ok = ref
    r0 = max(ok, key=lambda r: maxit[r])
    r1 = full[stage[full] == d][0]
    terms = {}
    for tag, p, c in (("front", pf, cf), ("back", pb, cb)):
        dp = p[r1] - p[r0]
        dp[1] = (dp[1] + np.pi) % (2 * np.pi) - np.pi
        W = np.linalg.inv(c[r0])
        tot = float(dp @ W @ dp)
        single = [float(dp[i] ** 2 / c[r0][i, i]) for i in range(6)]
        W5 = np.linalg.inv(c[r0][:5, :5])
        nomom = float(dp[:5] @ W5 @ dp[:5])
        terms[tag] = (tot, single, nomom, dp)
    out.append((tuple(uk[s]), d, terms))
sel = None
if len(sys.argv) > 2:
    import os
    z = np.load(os.path.join(sys.argv[2], "yield.npz"))
    sel = {(int(a), int(b)) for a, b, g, o in zip(z["subrun"], z["event"], z["good"], z["stockok"]) if g and not o}
print(f"stock ParamsDiverged seeds traced: {len(out)}" + (f"; of them F1-good: {sum(1 for o in out if o[0] in sel)}" if sel else ""))
for tag in ("front", "back"):
    tot = np.array([o[2][tag][0] for o in out])
    nm = np.array([o[2][tag][2] for o in out])
    sg = np.array([o[2][tag][1] for o in out])
    print(f"{tag}: chi2 q50/q90 {np.quantile(tot, .5):.3g} {np.quantile(tot, .9):.3g}; without mom (5x5) q50/q90 "
          f"{np.quantile(nm, .5):.3g} {np.quantile(nm, .9):.3g}; over 1e6: all {np.mean(tot > 1e6) * 100:.0f} %, "
          f"without mom {np.mean(nm > 1e6) * 100:.0f} %")
    print("   single-parameter terms q50: " + ", ".join(f"{P[i]} {np.median(sg[:, i]):.3g}" for i in range(6)))
    dm = np.array([o[2][tag][3][5] for o in out])
    print(f"   dmom q16/50/84 {np.round(np.quantile(dm, [.16, .5, .84]), 4)} MeV/c")
