"""CRV-term rates per sector of two fast CRV transport runs on the same primaries and seeds (e.g. baseline vs a world
modification): events with >= 1 primary-muon crossing of a sector (any visible energy; the production CRV term needs a
CrvStep > 0.1 MeV, crossings of a bar deposit ~2 MeV/cm so the two coincide for muons), and the paired difference.

    python3 tracker/scripts/crv_sector_rates.py "A/fastC.*.root" "B/fastC.*.root"
"""
import sys

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables
from crvfastsim.analyses.transport import DEFAULT_COUNTERS

sec = geometry.from_offline_config(geometry.find_counters(DEFAULT_COUNTERS)).sector


def per_event(pat):
    s = tables.read_many(tables.expand([pat]), "steps", ["run", "subrun", "event", "bar", "sp", "edep"])
    s = s[(s.sp == 1) & (s.edep > 0.1)]
    s["sector"] = np.asarray(sec)[s.bar.to_numpy()]
    return s[["run", "subrun", "event", "sector"]].drop_duplicates()


a, b = per_event(sys.argv[1]), per_event(sys.argv[2])
print(f"{'sector':>10s} {'A':>8s} {'B':>8s}  B/A-1   paired err  (A-only, B-only)")
for k in sorted(set(a.sector) | set(b.sector), key=str) + ["any"]:
    ea = set(map(tuple, (a if k == "any" else a[a.sector == k])[["run", "subrun", "event"]].to_numpy()))
    eb = set(map(tuple, (b if k == "any" else b[b.sector == k])[["run", "subrun", "event"]].to_numpy()))
    na, nb, ao, bo = len(ea), len(eb), len(ea - eb), len(eb - ea)
    print(f"{str(k):>10s} {na:8d} {nb:8d}  {nb / max(na, 1) - 1:+.4f}  +- {np.sqrt(ao + bo) / max(na, 1):.4f}  ({ao}, {bo})")
