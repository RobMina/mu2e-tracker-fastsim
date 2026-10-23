"""Tracker-term pass rate of two fast runs on the same primaries and seeds (e.g. baseline vs a world modification):
total and by generated |p|, with the paired error sqrt(A-only + B-only) / A.

    python3 tracker/scripts/paired_fast_arms.py "A/fasttrkC.*.root" "B/fasttrkC.*.root"
"""
import sys

import numpy as np

from mu2efastsim.tables import KEY, primary_muons
from trkfastsim import tables as T

P_EDGES = np.array([0.0, 500.0, 750.0, 1000.0, 1500.0, 3000.0, 1e9])


def passes(pat):
    paths = T.expand([pat])
    s = T.read_many(paths, "sgsteps")
    f = T.trk_filter(s[s.sp == 1])
    sel = f[f.pass_trk].reset_index()[KEY]
    g = primary_muons(T.read_many(paths, "gen"))
    g["p_gen"] = np.sqrt(g.px ** 2 + g.py ** 2 + g.pz ** 2)
    return g.merge(sel.assign(ok=True), on=KEY, how="left").fillna({"ok": False})


a, b = passes(sys.argv[1]), passes(sys.argv[2])
m = a[KEY + ["p_gen", "ok"]].merge(b[KEY + ["ok"]], on=KEY, suffixes=("_a", "_b"))
bins = np.clip(np.digitize(m.p_gen, P_EDGES) - 1, 0, len(P_EDGES) - 2)
for lab, sel in [("all", np.ones(len(m), bool))] + [(f"p {P_EDGES[k]:.0f}-{min(P_EDGES[k + 1], 1e5):.0f}", bins == k)
                                                     for k in range(len(P_EDGES) - 1)]:
    x = m[sel]
    na, nb = int(x.ok_a.sum()), int(x.ok_b.sum())
    ao, bo = int((x.ok_a & ~x.ok_b).sum()), int((x.ok_b & ~x.ok_a).sum())
    print(f"{lab:>14s}  A {na:6d}  B {nb:6d}  B/A-1 {nb / max(na, 1) - 1:+.4f} +- {np.sqrt(ao + bo) / max(na, 1):.4f}  "
          f"(A-only {ao}, B-only {bo})")
