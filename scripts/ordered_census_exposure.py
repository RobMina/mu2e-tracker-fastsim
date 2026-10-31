"""Conservative actual-ray AABB exposure of unevaluated expanded-census placements; no shape admission."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest


def bounds_exposure(origin, direction, lengths, bounds):
    bounds = np.asarray(bounds, float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] < bounds[0]).any():
        raise ValueError("finite ordered known bounds required for exclusion")
    scale = max(1., float(np.abs(origin).max(initial=0.)), float(np.abs(bounds).max(initial=0.)), float(lengths.max(initial=0.)))
    guard = 1024 * np.finfo(float).eps * scale
    lo, hi = np.nextafter(bounds[0] - guard, -np.inf), np.nextafter(bounds[1] + guard, np.inf)
    parallel = direction == 0
    safe = np.where(parallel, 1., direction)
    left, right = (lo - origin) / safe, (hi - origin) / safe
    entry = np.maximum(0., np.where(parallel, -np.inf, np.nextafter(np.minimum(left, right), -np.inf)).max(axis=1))
    exit_ = np.minimum(lengths, np.where(parallel, np.inf, np.nextafter(np.maximum(left, right), np.inf)).min(axis=1))
    outside = (parallel & ((origin < lo) | (origin > hi))).any(axis=1)
    possible = ~outside & (exit_ >= entry) & (lengths > 0)
    return possible, entry, exit_, guard


def run(directory, outdir):
    directory, outdir = Path(directory), Path(outdir)
    paths = [directory / "rays.npz", directory / "census.json", directory / "complete.json"]
    hashes = {str(p): digest(p) for p in paths}
    census = json.loads(paths[1].read_text())
    receipt = json.loads(paths[2].read_text())
    unknown = {r["source"]: r for r in receipt["unresolved_placement_reports"]}
    with np.load(paths[0], allow_pickle=False) as saved:
        origin, direction, lengths, keys = saved["origin"], saved["direction"], saved["lengths"], saved["primary_keys"]
    summary, rows = [], []
    for record in census["records"]:
        if record["placement_path"] not in unknown:
            continue
        bounds = record.get("geometry_bounds_mm")
        if bounds is None:
            summary.append(dict(source=record["placement_path"], evaluable_bounds=False, possible_lanes=None))
            continue
        possible, entry, exit_, guard = bounds_exposure(origin, direction, lengths, bounds)
        summary.append(dict(source=record["placement_path"], material=record["material"], solid=record["solid_tag"],
            leaf=record["is_leaf"], evaluable_bounds=True, possible_lanes=int(possible.sum()), guard_mm=guard))
        selected = np.flatnonzero(possible)
        if len(selected):
            frame = pd.DataFrame(keys[selected], columns=["run", "subrun", "event", "sp"]).assign(lane=selected,
                source=record["placement_path"], material=record["material"], entry_mm=entry[selected], exit_mm=exit_[selected])
            rows.append(frame)
    outdir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summary).to_csv(outdir / "placements.csv", index=False)
    (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()).to_csv(outdir / "possible_ray_boxes.csv", index=False)
    result = dict(scope="outward padded conservative bounds only; possible hits are not material intersections or permission to drop lanes",
        placements=summary, input_sha256=hashes, source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("census exposure inputs changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.outdir)
