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
    reference = os.environ.get("JAXCHECK_NUMPY_REFERENCE")
    for d in dirs:
        jax_files = sorted(glob.glob(os.path.join(d, "fasttrkCjx.*.root")))
        for fj in jax_files:
            fn = fj.replace("fasttrkCjx.", "fasttrkCnp.")
            if reference:
                fn = os.path.join(reference, os.path.basename(os.path.normpath(d)), os.path.basename(fn))
            if not os.path.isfile(fn):
                raise FileNotFoundError(f"missing numpy reference for {fj}: {fn}")
            a.append(tables.read(fn, "sgsteps"))
            b.append(tables.read(fj, "sgsteps"))
            g.append(tables.read(fn, "gen"))
            names.append(os.path.basename(fn))
    if not names:
        raise ValueError("no matched numpy/JAX shard tables")
    cat = lambda x: pd.concat(x, ignore_index=True)
    rec = {"file": f"{len(names)} shards", "shards": names, "compare": jaxcheck.compare(cat(a), cat(b), cat(g)),
           "reproducible": "see the per-shard jaxcheck.json"}
    print("\n".join([f"# combined over {', '.join(names)}"] + jaxcheck._summary(rec)))
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
