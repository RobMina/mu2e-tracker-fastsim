"""Concatenate fast step tables (steps output: sgsteps, particles, gen, subruns, surfsteps) into one file per group, so
one path A grid job (StrawGasStepsFromTable, events matched by run/subrun/event) can read several gen-stream files.

    python3 tracker/scripts/merge_tables.py OUT.root IN1.root [IN2.root ...]
"""
import sys

import pandas as pd
import uproot

from mu2efastsim import tables as T

TREES = ("sgsteps", "particles", "gen", "subruns", "surfsteps")


def main(out, inputs):
    trees = {}
    for t in TREES:
        parts = []
        for f in inputs:
            with uproot.open(f) as fh:
                if f"{T.DIR}/{t}" in fh:
                    parts.append(fh[f"{T.DIR}/{t}"].arrays(library="pd"))
        if parts:
            trees[t] = pd.concat(parts, ignore_index=True)
    n = {t: len(v) for t, v in trees.items()}
    T.write(out, trees, as_ttree=True)
    print(f"{out}: {len(inputs)} inputs, rows {n}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
