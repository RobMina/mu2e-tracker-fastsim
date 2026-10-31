"""Exact highest-cycle event coverage for a bounded stock consumer; no subrun exposure inference."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from scripts.transport_capture_audit import digest
from trkfastsim import tables


def keys(frame):
    value = pd.MultiIndex.from_frame(frame[tables.KEY])
    if value.duplicated().any():
        raise ValueError("duplicate event identity")
    return value


def run(table, ntuple, hits, out):
    paths = list(map(Path, (table, ntuple, hits)))
    hashes = {str(p): digest(p) for p in paths}
    sgs = tables.read(table, "sgsteps")
    generated = tables.read(table, "gen")[tables.KEY].drop_duplicates()
    decisions = tables.trk_filter(sgs).reset_index()
    selected = decisions.loc[decisions.pass_trk, tables.KEY]
    expected = keys(selected)
    counts = {}
    for label, path, tree, prefix in (("ntuple", ntuple, "EventNtuple/ntuple", "evtinfo/"),
                                       ("hits", hits, "trkdump/events", "")):
        with uproot.open(path) as root:
            t = root[tree]
            frame = pd.DataFrame({name: t[prefix + name].array(library="np") for name in tables.KEY})
            actual = keys(frame)
            if len(expected.difference(actual)) or len(actual.difference(expected)):
                raise ValueError(f"{label} event coverage differs from unchanged merged PrimaryFilter")
            counts[label] = dict(entries=t.num_entries, cycles=[k for k in root.keys() if k.split(";")[0] == tree],
                                 identities=frame.to_dict("records"))
            if label == "ntuple":
                status = t["trk/trk.status"].array()
                counts[label]["tracks"] = int(ak.sum(ak.num(status, axis=1)))
    result = dict(scope="bounded stock input/output consumer compatibility; not G4/rate/secondary closure",
        generated_events=len(generated), selected_events=len(selected), output=counts,
        highest_cycle_exact_event_coverage=True, exposure="no livetime inferred for a partial subrun",
        input_sha256=hashes, source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_production=False)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("stock consumer product changed")
    with Path(out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table")
    for name in ("ntuple", "hits", "out"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    run(args.table, args.ntuple, args.hits, args.out)
