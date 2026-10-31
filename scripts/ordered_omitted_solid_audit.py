"""Probe actual unsupported solids along saved slices using explicit reviewed geometry sources."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from scripts.transport_capture_audit import digest


def load_geometry_sources(directory):
    modules = {}
    for name in ("navigationcensus", "torusprobe", "omittedpath"):
        path = Path(directory) / f"{name}.py"
        qualified = f"trkfastsim.{name}"
        if qualified in sys.modules:
            raise ValueError(f"refusing to replace an imported module: {qualified}")
        spec = importlib.util.spec_from_file_location(qualified, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def run(replay, census, gdml, source, outdir):
    replay, census, gdml, source, outdir = map(Path, (replay, census, gdml, source, outdir))
    outdir.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((replay / "complete.json").read_text())
    payload = replay / "traversed.npz" if receipt["status"] == "bounded_g4_entry_complete" else census / "slices.npz"
    paths = [replay / "complete.json", payload, census / "loader_census.json", gdml,
             *[source / f"{name}.py" for name in ("navigationcensus", "torusprobe", "omittedpath")]]
    before = {str(path): digest(path) for path in paths}
    receipt = json.loads(paths[0].read_text())
    if receipt["status"] not in ("bounded_g4_entry_complete", "bounded_full_chain_complete"):
        raise ValueError("completed ordered replay required")
    modules = load_geometry_sources(source)
    inventory = json.loads(paths[2].read_text())
    with np.load(paths[1], allow_pickle=False) as saved:
        if receipt["status"] == "bounded_g4_entry_complete":
            origin, direction = saved["slice_pos0"], saved["slice_dir"]
            lengths = saved["slice_end"] - saved["slice_start"]
            lane = saved["slice_it"]
        else:
            origin, direction, lengths, lane = saved["origin"], saved["direction"], saved["length"], saved["lane"]
        cells, placements = modules["omittedpath"].probe(gdml, inventory["records"], origin, direction, lengths,
                                                        min_cell_mm=1e-7, max_cells=1000000)
        if len(cells):
            cells["interior_lane"] = lane[cells.lane.to_numpy(dtype=np.int64)]
    cells.to_csv(outdir / "omitted_cells.csv", index=False)
    (outdir / "placements.json").write_text(json.dumps(placements, indent=2) + "\n")
    resolved = [row for row in placements if row["resolved"]]
    unknown = [row for row in placements if not row["resolved"]]
    hit = [row for row in resolved if row["solid_upper_mm"] > 0]
    result = dict(scope="actual unsupported leaf-solid probes along traversed slices; no path/material correction or GDML admission",
        lanes=int(np.unique(lane).size), slices=len(lengths), unsupported_placements=len(placements), evaluated=len(resolved),
        unevaluated=len(unknown), cells=len(cells), possible_hit_placements=hit, unevaluated_placements=unknown,
        summed_placement_solid_lower_mm=float(sum(row["solid_lower_mm"] for row in resolved)),
        summed_placement_solid_upper_mm=float(sum(row["solid_upper_mm"] for row in resolved)),
        hierarchy_verified_placement_lower_mm=float(sum(row["hierarchy_verified_lower_mm"] for row in resolved)),
        warning="per-placement sums can overlap and are not net material/debit; caps/unknown ancestry remain explicit",
        input_sha256=before, admitted_for_production=False)
    if before != {str(path): digest(path) for path in paths}:
        raise ValueError("saved geometry audit inputs changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay")
    parser.add_argument("--census", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--geometry-source", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.replay, args.census, args.gdml, args.geometry_source, args.outdir)
