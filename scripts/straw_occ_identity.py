"""Audit an off-axis placement A/B: identical generators, primary steps and surface steps.

Usage: straw_occ_identity.py BASE_DIR CAND_DIR --out NEW_JSON
Counts unique event identities from finished generator trees, not subrun metadata.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trkfastsim import tables as T


def primary_steps(path):
    s = T.read(path, "sgsteps")
    p = T.read(path, "particles")
    keys = p.loc[(p.isprim == 1) & (p.pdg.abs() == 13), T.KEY + ["key"]].rename(columns={"key": "sp"})
    out = s.merge(keys, on=T.KEY + ["sp"], validate="many_to_one")
    return out.drop(columns=["nsgs"], errors="ignore")


def equal(a, b):
    return list(a.columns) == list(b.columns) and a.shape == b.shape and all(
        np.array_equal(a[k].to_numpy(), b[k].to_numpy(), equal_nan=True) for k in a)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("candidate")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    af = sorted(Path(args.base).glob("fasttrk*.root"))
    bf = sorted(Path(args.candidate).glob("fasttrk*.root"))
    if not af or [f.name for f in af] != [f.name for f in bf]:
        raise ValueError("both arms require identical, nonempty file-name coverage")
    rows, exposure = [], set()
    for a, b in zip(af, bf):
        ga, gb = T.read(a, "gen"), T.read(b, "gen")
        ev = ga[T.KEY].drop_duplicates()
        subruns = set(ev[["run", "subrun"]].itertuples(index=False, name=None))
        if subruns & exposure:
            raise ValueError("overlapping subrun coverage")
        exposure |= subruns
        pa, pb = primary_steps(a), primary_steps(b)
        rows.append({"base": str(a), "candidate": str(b), "generator_events": len(ev),
                     "primary_steps": len(pa), "generator_identical": equal(ga, gb),
                     "primary_steps_identical_excluding_event_nsgs": equal(pa, pb),
                     "surface_steps_identical": equal(T.read(a, "surfsteps"), T.read(b, "surfsteps"))})
    result = {"files": rows, "generator_events": sum(r["generator_events"] for r in rows),
              "primary_steps": sum(r["primary_steps"] for r in rows), "subruns": len(exposure),
              "all_identical": all(r[k] for r in rows for k in
                                   ("generator_identical", "primary_steps_identical_excluding_event_nsgs", "surface_steps_identical"))}
    with open(args.out, "x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    if not result["all_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
