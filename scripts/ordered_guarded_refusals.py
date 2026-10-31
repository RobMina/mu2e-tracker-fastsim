"""Join actual guarded refusal cells/parents to all primary identities without acceptance extrapolation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest

KEY = ["run", "subrun", "event", "sp"]


def run(directory, outdir):
    directory, outdir = Path(directory), Path(outdir)
    receipt = json.loads((directory / "complete.json").read_text())
    if receipt["status"] != "refused":
        raise ValueError("explicitly refused guarded producer required")
    paths = [directory / "complete.json", directory / "handoff_primaries.csv"]
    paths += sorted(directory.glob("navigation_*/*.csv")) + sorted(directory.glob("navigation_*/rays.npz"))
    hashes = {str(p): digest(p) for p in paths}
    outdir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for query in sorted(directory.glob("navigation_*")):
        if not (query / "result.refusals.csv").is_file():
            continue
        refusals = pd.read_csv(query / "result.refusals.csv")
        if refusals.empty:
            continue
        with np.load(query / "rays.npz", allow_pickle=False) as rays:
            keys = pd.DataFrame(rays["primary_keys"], columns=KEY).assign(lane=np.arange(len(rays["primary_keys"])))
            origin, direction = rays["origin"], rays["direction"]
        full = refusals.merge(keys, on="lane", validate="many_to_one")
        full.to_csv(outdir / (query.name + ".refusals.csv"), index=False)
        cells = pd.read_csv(query / "cells.csv")
        cells = cells[cells.lane.isin(refusals.lane)].copy()
        cells = cells.merge(keys, on="lane", validate="many_to_one")
        cells["length_mm"] = cells.s1_mm - cells.s0_mm
        cells["y_min_mm"] = np.minimum(origin[cells.lane, 1] + direction[cells.lane, 1] * cells.s0_mm,
                                        origin[cells.lane, 1] + direction[cells.lane, 1] * cells.s1_mm)
        cells["y_max_mm"] = np.maximum(origin[cells.lane, 1] + direction[cells.lane, 1] * cells.s0_mm,
                                        origin[cells.lane, 1] + direction[cells.lane, 1] * cells.s1_mm)
        cells.to_csv(outdir / (query.name + ".cells.csv"), index=False)
        grouped = cells.groupby(KEY + ["source", "material", "replaces_material", "state", "reason"], as_index=False).agg(
            length_mm=("length_mm", "sum"), y_min_mm=("y_min_mm", "min"), y_max_mm=("y_max_mm", "max"), cells=("lane", "size"))
        grouped.to_csv(outdir / (query.name + ".cell_totals.csv"), index=False)
        parents = pd.read_csv(query / "result.parent_comparisons.csv")
        bad = parents[~parents.expected_parent].merge(keys, on="lane", validate="many_to_one")
        bad.to_csv(outdir / (query.name + ".parent_mismatch.csv"), index=False)
        parent_totals = []
        if len(bad):
            bad["length_mm"] = bad.s1_mm - bad.s0_mm
            parent_totals = bad.groupby(KEY + ["material", "replaces_material", "old_material"], as_index=False).length_mm.sum().to_dict("records")
        summaries.append(dict(query=query.name, refused_primary_keys=full[KEY].drop_duplicates().to_dict("records"),
            reasons={str(k): int(v) for k, v in full.reason.value_counts().items()},
            cells=grouped.to_dict("records"), parent_mismatch=parent_totals))
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("guarded refusal inputs changed")
    result = dict(scope="actual refused-query identity/material/ownership census, not missing-debit or acceptance attribution",
        generated_events=receipt["generated_events"], input_primary_muons=receipt["input_primary_muons"],
        world_delivered=receipt["world_delivered"], queries=summaries, input_sha256=hashes,
        source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.outdir)
