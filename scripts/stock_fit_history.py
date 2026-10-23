"""Stock KKLine fit outcome per seed from a PrintLevel-1 log (dump_tracks_mc_saveall_print.fcl): each fit prints its
final status at the end of the seed fit (CHSEEDFIT, meta-iterations 1-4) and, if usable, at the end of the extension
(CHDRIFTEXT, 5-13). Joined in order with the dump's KalSeeds (SaveAllFits: one KalSeed per fitted seed).
python3 scripts/stock_fit_history.py LOG DUMP.root [OUT.npz]"""
import re
import sys

import awkward as ak
import numpy as np
import uproot

pat = re.compile(r"Fit Status (\w+)\s.*?Meta-iteration (\d+) iteration (\d+)")
fits = []
for line in open(sys.argv[1]):
    m = pat.search(line)
    if not m:
        continue
    st, meta, it = m.group(1), int(m.group(2)), int(m.group(3))
    if meta <= 4 or not fits or fits[-1]["meta"] > 4 and False:
        if meta <= 4:
            fits.append({"seed": st, "seedmeta": meta, "final": st, "meta": meta, "iter": it})
            continue
    fits[-1].update(final=st, meta=meta, iter=it)
a = uproot.open(sys.argv[2])["trkdump/events"].arrays(["kl_status", "sd_n"])
st = np.asarray(ak.flatten(a.kl_status), dtype=np.int64)
ev = np.repeat(np.arange(len(a)), ak.to_numpy(ak.num(a.kl_status)))
print(f"fits in log {len(fits)}, KalSeeds in dump {len(st)}")
n = min(len(fits), len(st))
fitok = (st[:n] >> 15) & 1 == 1
final = np.array([f["final"] for f in fits[:n]])
meta = np.array([f["meta"] for f in fits[:n]])
ok_by_log = (final == "Converged") | (final == "Unconverged")
print(f"usable by log {ok_by_log.sum()}, FitOK {fitok.sum()}; usable but not FitOK (goodFit rho/z test) {(ok_by_log & ~fitok).sum()},"
      f" FitOK but unusable {(fitok & ~ok_by_log).sum()}")
fails = ~fitok
print("stock failures by final status and meta-iteration:")
for (s_, m_), c in sorted(zip(*np.unique(np.stack([final[fails], meta[fails].astype(str)], 1), axis=0, return_counts=True)),
                          key=lambda x: -x[1])[:20] if fails.any() else []:
    pass
keys, cnt = np.unique(np.char.add(np.char.add(final[fails], " meta "), meta[fails].astype(str)), return_counts=True)
for k, c in sorted(zip(keys, cnt), key=lambda x: -x[1]):
    print(f"   {c:5d}  {k}")
print("by status:", {k: int(v) for k, v in zip(*np.unique(final[fails], return_counts=True))})
if len(sys.argv) > 3:
    np.savez(sys.argv[3], ev=ev[:n], fitok=fitok, final=final, meta=meta)
