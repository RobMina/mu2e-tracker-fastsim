"""Saved possible-leaf overlap witnesses; no overlap cancellation, priority or transport admission."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest


def run(directory, outdir, *, max_pairs=2000000):
    directory, outdir = Path(directory), Path(outdir)
    paths = [directory / "cells.csv", directory / "rays.npz", directory / "placements.json", directory / "complete.json"]
    hashes = {str(p): digest(p) for p in paths}
    cells = pd.read_csv(paths[0]).reset_index(names="cell_id")
    sizes = cells.groupby("lane").size()
    if int((sizes * sizes).sum()) > max_pairs:
        raise ValueError("overlap witness pair cap")
    pairs = cells.merge(cells, on="lane", suffixes=("_a", "_b"))
    pairs = pairs[pairs.cell_id_a < pairs.cell_id_b].copy()
    pairs["s0_mm"] = np.maximum(pairs.s0_mm_a, pairs.s0_mm_b)
    pairs["s1_mm"] = np.minimum(pairs.s1_mm_a, pairs.s1_mm_b)
    pairs = pairs[pairs.s1_mm > pairs.s0_mm].copy()
    pairs["width_mm"] = pairs.s1_mm - pairs.s0_mm
    pairs["same_source"] = pairs.source_a == pairs.source_b
    with np.load(paths[1], allow_pickle=False) as saved:
        keys = saved["primary_keys"]
    for j, name in enumerate(("run", "subrun", "event", "sp")):
        pairs[name] = keys[pairs.lane.to_numpy(np.int64), j]
    outdir.mkdir(parents=True, exist_ok=False)
    pairs.to_csv(outdir / "overlaps.csv", index=False)
    totals = pairs.groupby(["source_a", "source_b", "material_a", "material_b", "state_a", "state_b"], as_index=False).agg(
        pairs=("lane", "size"), lanes=("lane", "nunique"), max_width_mm=("width_mm", "max"), sum_width_mm=("width_mm", "sum"))
    totals.to_csv(outdir / "source_pairs.csv", index=False)
    result = dict(scope="geometric overlap localization only, no cancellation or admission", pairs=len(pairs),
        lanes=int(pairs.lane.nunique()), same_source_pairs=int(pairs.same_source.sum()),
        source_pairs=totals.to_dict("records"), input_sha256=hashes,
        source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("overlap inputs changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.outdir)
