"""Actual newly exposed leaf Boolean occupancy gate without transport or hierarchy waiver."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_torus_pair_probe import load_pair_sources
from scripts.transport_capture_audit import digest
from trkfastsim import csgenvelope


def run(directory, gdml, source, outdir):
    directory, gdml, source, outdir = map(Path, (directory, gdml, source, outdir))
    receipt = json.loads((directory / "complete.json").read_text())
    chosen = [r for r in receipt["unresolved_placement_reports"] if r["leaf"] and r["solid_tag"] in ("subtraction", "union", "intersection")]
    modules = load_pair_sources(source)
    old = modules["omittedpath"]
    index = old.PlacementIndex(gdml, [r["source"] for r in chosen])
    with np.load(directory / "rays.npz", allow_pickle=False) as saved:
        origin, direction, lengths, keys = saved["origin"], saved["direction"], saved["lengths"], saved["primary_keys"]
    outdir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for i, report in enumerate(chosen):
        placement = index.placements[report["source"]]
        local_o, local_d = index.local_rays(placement, origin, direction)
        guard = (len(placement["ancestors"]) + 1) * old._roundoff(origin, origin + direction * lengths[:, None], placement["position"])
        cells, method = csgenvelope.probe(index.solids[placement["solid"]], index.solids, local_o, local_d, lengths,
                                         coordinate_guard_mm=guard)
        cells, ancestors = old.hierarchy_check(index, placement, cells, origin, direction)
        for j, key in enumerate(("run", "subrun", "event", "sp")):
            cells[key] = keys[cells.lane.to_numpy(np.int64), j]
        cells.to_csv(outdir / f"placement_{i}.cells.csv", index=False)
        cells["length_mm"] = cells.s1_mm - cells.s0_mm
        totals = cells.groupby(["run", "subrun", "event", "sp", "state", "reason"], as_index=False).length_mm.sum()
        summaries.append(dict(source=report["source"], material=report["material"], method=method,
            ancestry_unverified=ancestors, all_hierarchy_verified=bool(cells.ancestors_verified.all()),
            totals=totals.to_dict("records")))
    result = dict(scope="actual leaf Boolean guarded occupancy only, not mixed admission or transported debit", placements=summaries,
        input_sha256={str(gdml): digest(gdml), str(directory / "rays.npz"): digest(directory / "rays.npz")},
        source_sha256={str(Path(__file__)): digest(__file__), str(Path(csgenvelope.__file__)): digest(csgenvelope.__file__)}, admitted_for_physics=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("gdml", "source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    run(args.directory, args.gdml, args.source, args.outdir)
