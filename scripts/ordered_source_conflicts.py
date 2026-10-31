"""Saved signed-source sweep census exposing added raw cross-material conflicts; no priority or repair."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest


def conflicts(raw):
    keys = ["lane", "material"]
    edges = pd.concat([raw[keys].assign(s=raw.s0_mm, delta=raw.sign), raw[keys].assign(s=raw.s1_mm, delta=-raw.sign)])
    edges = edges.groupby(keys + ["s"], as_index=False).delta.sum().sort_values(keys + ["s"])
    edges["count"] = edges.groupby(keys).delta.cumsum()
    edges["s1_mm"] = edges.groupby(keys).s.shift(-1)
    occupied = edges[(edges["count"] != 0) & (edges.s1_mm > edges.s)].rename(columns={"s": "s0_mm"})
    merged = occupied.merge(occupied, on="lane", suffixes=("_a", "_b"))
    merged["s0_mm"] = np.maximum(merged.s0_mm_a, merged.s0_mm_b)
    merged["s1_mm"] = np.minimum(merged.s1_mm_a, merged.s1_mm_b)
    bad = merged[(merged.material_a < merged.material_b) & (merged.s1_mm > merged.s0_mm)].copy()
    bad["length_mm"] = bad.s1_mm - bad.s0_mm
    return bad


def run(directory, rays, outdir):
    directory, rays, outdir = map(Path, (directory, rays, outdir))
    paths = [directory / "adjusted_hits.csv", rays]
    hashes = {str(p): digest(p) for p in paths}
    raw = pd.read_csv(paths[0])
    conflict = conflicts(raw)
    with np.load(rays, allow_pickle=False) as saved:
        for j, key in enumerate(("run", "subrun", "event", "sp")):
            conflict[key] = saved["primary_keys"][conflict.lane.to_numpy(np.int64), j]
    outdir.mkdir(parents=True, exist_ok=False)
    conflict.to_csv(outdir / "conflicts.csv", index=False)
    totals = conflict.groupby(["run", "subrun", "event", "sp", "material_a", "material_b"], as_index=False).length_mm.sum()
    totals.to_csv(outdir / "totals.csv", index=False)
    owners = conflict[["lane", "s0_mm", "s1_mm"]].drop_duplicates().merge(raw, on="lane", suffixes=("_conflict", ""))
    owners = owners[(owners.s0_mm < owners.s1_mm_conflict) & (owners.s1_mm > owners.s0_mm_conflict)]
    owners.to_csv(outdir / "raw_owners.csv", index=False)
    result = dict(scope="signed material overlap witnesses, not a precedence/geometry correction", totals=totals.to_dict("records"),
        raw_conflict_rows=len(conflict), original_source_terms=len(raw), input_sha256=hashes,
        source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--rays", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.rays, args.outdir)
