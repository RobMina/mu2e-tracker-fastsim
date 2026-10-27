"""Check finished path-A ROOT row coverage against generator events and selected steps.

Run with --steps FAST_TABLES --ntuple NTUPLE --hits HITS --out NEW_JSON.
Reads the highest ROOT cycles and validates identities, not metadata event counts.
"""
import argparse
import json

import numpy as np
import uproot

from trkfastsim import tables as T
from trkfastsim.analyses import tv2compare as TV


def _events(path, tree, prefix=""):
    with uproot.open(path) as f:
        t = f[tree]
        a = t.arrays([prefix + c for c in T.KEY], library="np")
        rows = np.column_stack([a[prefix + c] for c in T.KEY])
        key = TV._keyid_rows(rows)
        if len(key) != len(np.unique(key)):
            raise ValueError(f"duplicate event identities in {path}:{tree}")
        return key, {"path": path, "tree": tree, "entries": t.num_entries,
                     "cycles_present": [k for k in f.keys() if k.split(";")[0] == tree]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", nargs="+", required=True)
    p.add_argument("--ntuple", required=True)
    p.add_argument("--hits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    gen = T.read_many(args.steps, "gen", T.KEY).drop_duplicates()
    generated = TV._keyid_rows(gen.to_numpy())
    selected, live = TV._selected(args.steps)
    nk, nt = _events(args.ntuple, TV.TREE, "evtinfo/")
    hk, ht = _events(args.hits, "trkdump/events")
    TV._validate_coverage(nk, selected, "ntuple")
    TV._validate_coverage(hk, selected, "hits")
    if not np.array_equal(np.sort(nk), np.sort(hk)):
        raise ValueError("ntuple and hit-table event identities differ")
    if len(np.setdiff1d(nk, generated)):
        raise ValueError("output contains events absent from generator table")
    report = {"generated_events_from_tree": len(generated), "selected_events_from_steps": len(selected),
              "livetime_s": live, "ntuple": nt, "hits": ht,
              "missing_selected_events": 0, "duplicate_event_rows": 0,
              "unselected_output_events": len(np.setdiff1d(nk, selected)),
              "ntuple_hits_identity_equal": True,
              "selection_convention": "any primary muon; tracker PrimaryFilter term"}
    with open(args.out, "x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
