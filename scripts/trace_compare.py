"""Stage-by-stage hit states: stock (KinematicLineFitTrace, trace_kkline_mc.fcl) vs F1 (reco-vr2f1 --own-states,
yield.npz with stage_active/stage_lr). Seeds matched by (subrun, event) (one seed per event), hits by makeSH index.
Per stage: hit-state agreement (inactive / null / left / right) on seeds where both are still fitting; and for the
seeds where stock fails and F1 does not (or vice versa), the first stage where the states differ and what changed.
python3 scripts/trace_compare.py TRACE.root YIELD_DIR"""
import os
import sys

import awkward as ak
import numpy as np
import uproot

t = uproot.open(sys.argv[1])["kktrace/trace"].arrays()
z = np.load(os.path.join(sys.argv[2], "yield.npz"))
key_f1 = {(int(s), int(e)): i for i, (s, e) in enumerate(zip(z["subrun"], z["event"]))}
NS = 13
INACT, NULL = -2, 0


def f1_state(i, k):
    """F1 state per hit at stage k: -2 inactive, 0 null, +-1 left/right (F1 lr = sign of DOCA, stock right = DOCA > 0)."""
    a, l = z["stage_active"][i, k], z["stage_lr"][i, k]
    return np.where(a, np.sign(l).astype(int), INACT)


rows = {}
for r in range(len(t)):
    kk = (int(t.subrun[r]), int(t.event[r]))
    rows.setdefault(kk, {})[int(t.stage[r])] = r
common = [k for k in rows if k in key_f1]
print(f"seeds in trace {len(rows)}, in F1 {len(key_f1)}, common {len(common)}")
agree = np.zeros(NS); tot = np.zeros(NS); opp = np.zeros(NS)
conf = np.zeros((NS, 4, 4), int)               # stock state (inact, null, left, right) x F1 state
idx = {INACT: 0, NULL: 1, -1: 2, 1: 3}
first_diff = {}
for kk in common:
    i = key_f1[kk]
    shl = z["shloc"][i]
    for k, r in sorted(rows[kk].items()):
        if int(t.status[r]) >= 2:                 # stock unusable at this stage: stop comparing
            break
        sind = ak.to_numpy(t.shind[r]); sst = ak.to_numpy(t.state[r])
        pos = {int(h): j for j, h in enumerate(shl) if h >= 0}
        fs = f1_state(i, k)
        ss, ff = [], []
        for h, s in zip(sind, sst):
            if int(h) in pos and s > -3:
                ss.append(int(s)); ff.append(int(fs[pos[int(h)]]))
        ss, ff = np.array(ss), np.array(ff)
        if len(ss) == 0:
            continue
        tot[k] += len(ss); agree[k] += np.sum(ss == ff); opp[k] += np.sum((ss * ff) == -1)
        for a_, b_ in zip(ss, ff):
            conf[k, idx[a_], idx[b_]] += 1
        if kk not in first_diff and np.any(ss != ff):
            first_diff[kk] = (k, int(np.sum(ss != ff)), len(ss))
print("stage  hits    state agree   opposite side")
for k in range(NS):
    if tot[k]:
        print(f"{k:5d} {int(tot[k]):6d}   {agree[k] / tot[k] * 100:6.2f} %    {opp[k] / tot[k] * 100:5.2f} %")
names = ["inact", "null", "left", "right"]
for k in (3, 4, 7, 12):
    print(f"stage {k} stock (rows) x F1 (cols) " + " ".join(f"{n:>6s}" for n in names))
    for a_ in range(4):
        print(f"   {names[a_]:>6s} " + " ".join(f"{conf[k, a_, b_]:6d}" for b_ in range(4)))
# seeds where the outcomes differ
for name, sel in (("F1 good, stock failed", lambda i: z["good"][i] and not z["stockok"][i]),
                  ("stock good, F1 failed", lambda i: not z["good"][i] and z["stockok"][i])):
    ks = [kk for kk in common if sel(key_f1[kk])]
    fd = [first_diff.get(kk, (-1, 0, 0))[0] for kk in ks]
    death = []
    for kk in ks:
        st_ = [(k, int(t.status[r])) for k, r in sorted(rows[kk].items())]
        d = [k for k, s in st_ if s >= 2]
        death.append(d[0] if d else -1)
    print(f"== {name}: {len(ks)} seeds; first stage with a hit-state difference: "
          f"{dict(zip(*np.unique(fd, return_counts=True)))}; stock death stage: {dict(zip(*np.unique(death, return_counts=True)))}")
    same_before = sum(1 for a_, b_ in zip(fd, death) if b_ >= 0 and (a_ < 0 or a_ >= b_))
    print(f"   seeds whose states were identical up to stock's death stage: {same_before}")
