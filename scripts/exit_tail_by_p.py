"""Exit deflection (last straw -> TT_Outer exit crossing) by |p| at the last straw, fast vs G4, normalised to the
Highland theta0 of the material between them is not available on the G4 side, so the comparison is per |p| bin:
q50/q90/q99 [mrad] and the share above 300 mrad.

    python3 tracker/scripts/exit_tail_by_p.py "FAST/fasttrkC*.root" "G4/sgsteps.*.root"
"""
import sys

import numpy as np
import pandas as pd

from trkfastsim import tables as T


def table(pat):
    rows = []
    for f in T.expand([pat]):
        s = T.read(f, "sgsteps")
        p = T.read(f, "particles")
        pm = (p[(p.ismcprim == 1) & (p.pdg.abs() == 13)] if "ismcprim" in p else p[(p.key == 1) & (p.pdg.abs() == 13)])
        pm = pm.rename(columns={"key": "sp"})[T.KEY + ["sp"]]
        s = s.merge(pm, on=T.KEY + ["sp"])
        fl = T.trk_filter(s)
        sel = fl[fl.pass_trk].reset_index()[T.KEY]
        s = s.merge(sel, on=T.KEY).sort_values(T.KEY + ["t"])
        last = s.groupby(T.KEY).last().reset_index()
        ss = T.read(f, "surfsteps").merge(pm, on=T.KEY + ["sp"]).merge(sel, on=T.KEY)
        ss = ss[ss.surf == 4]
        ss = ss[0.5 * (ss.y0 + ss.y1) <= 0].sort_values("t").groupby(T.KEY).last().reset_index()
        m = ss.merge(last[T.KEY + ["px", "py", "pz"]], on=T.KEY, suffixes=("", "_s"))
        a, b = m[["px", "py", "pz"]].to_numpy(float), m[["px_s", "py_s", "pz_s"]].to_numpy(float)
        m["ang"] = 1e3 * np.arccos(np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1))
        m["pl"] = np.linalg.norm(b, axis=1)
        m["pe"] = np.linalg.norm(a, axis=1)
        rows.append(m[["ang", "pl", "pe"]])
    return pd.concat(rows, ignore_index=True)


F, G = table(sys.argv[1]), table(sys.argv[2])
for lo, hi in ((0, 150), (150, 200), (200, 300), (300, 500), (500, 1000)):
    a, b = F[(F.pl >= lo) & (F.pl < hi)], G[(G.pl >= lo) & (G.pl < hi)]
    q = lambda x: "/".join(f"{v:.0f}" for v in np.quantile(x.ang, (0.5, 0.9, 0.99))) if len(x) > 20 else "-"
    print(f"p last {lo:4d}-{hi:4d}: fast n {len(a):5d} share {len(a) / len(F):.4f} {q(a):>14s} >300 {np.mean(a.ang > 300):.3f} "
          f"p_exit/p q50 {np.median(a.pe / a.pl) if len(a) else 0:.2f} | G4 n {len(b):5d} share {len(b) / len(G):.4f} {q(b):>14s} "
          f">300 {np.mean(b.ang > 300):.3f} p_exit/p q50 {np.median(b.pe / b.pl) if len(b) else 0:.2f}")
