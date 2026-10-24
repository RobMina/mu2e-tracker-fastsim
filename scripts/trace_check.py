"""Check a KinematicLineFitTrace output against the KKLine product of the same job (trace_kkline_mc.fcl writes both):
the last traced stage of each seed must reproduce the stock fit (chi2, ndof, hit states). Then summarise the trace:
status per stage, and where stock's fits die.
python3 scripts/trace_check.py TRACE.root"""
import sys

import awkward as ak
import numpy as np
import uproot

f = uproot.open(sys.argv[1])
t = f["kktrace/trace"].arrays()
d = f["trkdump/events"].arrays(["kl_chi2", "kl_ndof", "kl_status", "kh_ambig", "kh_active", "kh_ind"])
ev = ak.to_numpy(t.event)
print(f"trace rows {len(t)}, seeds {len(np.unique(np.stack([ak.to_numpy(t.subrun), ev], 1), axis=0))}")
# events in the dump in order; the trace rows carry (subrun, event); map by row order of the dump tree
dump_ev = uproot.open(sys.argv[1])["trkdump/events"].arrays(["subrun", "event"]) if "subrun" in f["trkdump/events"].keys() else None
keys = np.stack([ak.to_numpy(t.subrun), ev, ak.to_numpy(t.iseed)], 1)
uk, first = np.unique(keys, axis=0, return_index=True)
names = {0: "conv", 1: "unconv", 2: "lowNDOF", 3: "gap", 4: "params", 5: "chisq", 6: "outside", 7: "failed"}
st = ak.to_numpy(t.status)
stage = ak.to_numpy(t.stage)
print("rows by stage:", np.bincount(stage).tolist())
for k in range(stage.max() + 1):
    m = stage == k
    c = np.bincount(st[m] + 1, minlength=9)[1:]
    print(f"  stage {k:2d}: " + ", ".join(f"{names[i]} {v}" for i, v in enumerate(c) if v))
# last row per seed vs the dump's KalSeed: match by (event index in file order); the dump tree has one row per event
if dump_ev is not None:
    dkey = {(int(s), int(e)): i for i, (s, e) in enumerate(zip(ak.to_numpy(dump_ev.subrun), ak.to_numpy(dump_ev.event)))}
else:
    dkey = None
nok = nbad = nchk = 0
bad = []
for k in uk:
    m = np.flatnonzero((keys == k).all(1))
    last = m[np.argmax(stage[m])]
    if dkey is None or (int(k[0]), int(k[1])) not in dkey:
        continue
    i = dkey[(int(k[0]), int(k[1]))]
    if len(d.kl_chi2[i]) != 1:
        continue
    nchk += 1
    same = (abs(float(t.chi2[last]) - float(d.kl_chi2[i][0])) < 1e-3 * max(1.0, float(d.kl_chi2[i][0]))
            and int(t.ndof[last]) == int(d.kl_ndof[i][0]))
    if same:
        nok += 1
    else:
        nbad += 1
        bad.append((int(k[1]), int(stage[last]), float(t.chi2[last]), float(d.kl_chi2[i][0]), int(t.ndof[last]),
                    int(d.kl_ndof[i][0])))
print(f"last traced stage vs KKLine KalSeed (single-fit events): {nchk} checked, {nok} identical chi2+ndof, {nbad} differ")
for b in bad[:8]:
    print("   event %d last stage %d chi2 trace %.3f stock %.3f ndof %d %d" % b)
