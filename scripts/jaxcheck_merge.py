"""Combined jaxcheck comparison over several shards from the tables `jaxcheck --save` wrote (numpy vs jax, same seeds).

    python3 tracker/scripts/jaxcheck_merge.py OUT.json DIR [DIR ...]   (each DIR holds fasttrkC{np,jx}.<shard>.root)
"""
import glob
import json
import os
import sys

import pandas as pd

from trkfastsim import tables
from trkfastsim.analyses import jaxcheck


def main(out, dirs):
    a, b, g, names = [], [], [], []
    for d in dirs:
        for fn in sorted(glob.glob(os.path.join(d, "fasttrkCnp.*.root"))):
            fj = fn.replace("fasttrkCnp.", "fasttrkCjx.")
            a.append(tables.read(fn, "sgsteps"))
            b.append(tables.read(fj, "sgsteps"))
            g.append(tables.read(fn, "gen"))
            names.append(os.path.basename(fn))
    cat = lambda x: pd.concat(x, ignore_index=True)
    rec = {"file": f"{len(names)} shards", "shards": names, "compare": jaxcheck.compare(cat(a), cat(b), cat(g)),
           "reproducible": "see the per-shard jaxcheck.json"}
    print("\n".join([f"# combined over {', '.join(names)}"] + jaxcheck._summary(rec)))
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
