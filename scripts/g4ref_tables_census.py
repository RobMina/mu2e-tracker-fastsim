"""Integrity census of a G4 reference's tables: per shard, generated events and live time (CRV gen table), accepted
events (tracker table gen tree), and tracker-term passes; flags shards whose numbers are off (truncated dumps).

    python3 tracker/scripts/g4ref_tables_census.py CRV_TABLES_DIR TRK_TABLES_DIR
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from trkfastsim import tables as T

cdir, tdir = sys.argv[1:3]
rows = []
for g in sorted(glob.glob(os.path.join(cdir, "gen.001400_*.root"))):
    seq = os.path.basename(g)[4:-5]
    t = glob.glob(os.path.join(tdir, f"sgsteps.*{seq}.root"))
    s = T.read(g, "subruns")
    ng = T.read(g, "gen", ["run", "subrun", "event"]).drop_duplicates().shape[0]
    na = T.read(t[0], "gen", ["run", "subrun", "event"]).drop_duplicates().shape[0] if t else -1
    rows.append({"seq": seq, "generated": ng, "livetime": float(s.livetime.sum()), "accepted": na})
df = pd.DataFrame(rows)
print(f"shards {len(df)}; generated {df.generated.sum():,}; live time {df.livetime.sum():.3f} s; accepted {df.accepted.sum():,} "
      f"({df.accepted.sum() / df.generated.sum():.4f})")
bad = df[(df.generated != 50000) | (df.accepted < 0) | (np.abs(df.accepted / df.generated - 0.05) > 0.006)]
print(f"suspicious shards: {len(bad)}")
print(bad.to_string(index=False) if len(bad) else "")
