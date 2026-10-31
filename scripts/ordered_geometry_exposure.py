"""Audit completed ordered slices against preserved conservative GDML omission census."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest
from trkfastsim import crossings, geometry


def exposures(records, origin, direction, length, lane):
    rows = []
    unbounded = []
    for record in records:
        if record["classification"] not in ("unsupported", "approximate"):
            continue
        bounds = record.get("geometry_bounds_mm")
        if bounds is None:
            unbounded.append(record)
            continue
        bounds = np.asarray(bounds, float)
        parallel = direction == 0
        outside_parallel = np.any(parallel & ((origin < bounds[0]) | (origin > bounds[1])), axis=1)
        safe = np.where(parallel, 1., direction)
        a, b = (bounds[0] - origin) / safe, (bounds[1] - origin) / safe
        entry = np.maximum(0., np.max(np.where(parallel, -np.inf, np.minimum(a, b)), axis=1))
        end = np.minimum(length, np.min(np.where(parallel, np.inf, np.maximum(a, b)), axis=1))
        use = ~outside_parallel & (end > entry)
        if use.any():
            rows.append(pd.DataFrame(dict(lane=lane[use], slice=np.flatnonzero(use), s0_mm=entry[use],
                s1_mm=end[use], classification=record["classification"], path=record["path"],
                material=record["material"], solid=record["solid"], tag=record["solid_tag"])))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), unbounded


def saved_slices(replay, receipt):
    replay = Path(replay)
    if receipt["status"] == "bounded_g4_entry_complete":
        with np.load(replay / "traversed.npz", allow_pickle=False) as saved:
            return (saved["slice_pos0"], saved["slice_dir"], saved["slice_end"] - saved["slice_start"],
                    saved["slice_it"])
    if receipt["status"] != "bounded_full_chain_complete":
        raise ValueError("completed actual ordered output required")
    key = ["run", "subrun", "event", "sp"]
    vertices = pd.read_csv(replay / "vertices.csv").sort_values(key + ["path"], kind="stable")
    if vertices.duplicated(key + ["path"]).any():
        raise ValueError("duplicate ordered vertex path")
    length = vertices.groupby(key, sort=False).path.shift(-1) - vertices.path
    use = length > 0
    identities = pd.MultiIndex.from_frame(vertices[key])
    lane = pd.factorize(identities, sort=False)[0]
    origin = geometry.to_mu2e(vertices.loc[use, ["x", "y", "z"]].to_numpy(float))
    direction = vertices.loc[use, ["ux", "uy", "uz"]].to_numpy(float)
    next_position = vertices.groupby(key, sort=False)[["x", "y", "z"]].shift(-1).loc[use].to_numpy(float)
    expected = vertices.loc[use, ["x", "y", "z"]].to_numpy(float) + direction * length[use].to_numpy()[:, None]
    if not np.allclose(next_position, expected, rtol=0, atol=1e-7):
        raise ValueError("vertex endpoints disagree with recorded arc/direction")
    return origin, direction, length[use].to_numpy(), lane[use.to_numpy()]


def run(replay, gdml, census_source, outdir):
    replay, gdml, census_source, outdir = map(Path, (replay, gdml, census_source, outdir))
    outdir.mkdir(parents=True, exist_ok=False)
    report = json.loads((replay / "complete.json").read_text())
    payload = "traversed.npz" if report["status"] == "bounded_g4_entry_complete" else "vertices.csv"
    paths = [replay / "complete.json", replay / payload, gdml, census_source]
    before = {str(path): digest(path) for path in paths}
    spec = importlib.util.spec_from_file_location("trkfastsim.navigationcensus", census_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    geom = geometry.from_gdml(gdml)
    lo, hi = crossings.panel_boxes(geom)
    inventory = module.census(gdml, (lo.min(0) - 300., hi.max(0) + 300.))
    origins, dirs, length, lane = saved_slices(replay, report)
    np.savez_compressed(outdir / "slices.npz", origin=origins, direction=dirs, length=length, lane=lane)
    candidates, unbounded = exposures(inventory["records"], origins, dirs, length, lane)
    candidates.to_csv(outdir / "candidate_spans.csv", index=False)
    (outdir / "loader_census.json").write_text(json.dumps(inventory, indent=2) + "\n")
    result = dict(scope="conservative AABB exposure on actual traversed slices; never actual material occupancy/length",
        lanes=int(np.unique(lane).size), slices=len(length), candidate_pairs=len(candidates),
        unbounded_unsupported_or_approximate=len(unbounded),
        counts=inventory["counts"], admitted_for_production=False, input_sha256=before)
    if len(candidates):
        result["candidate_lanes"] = int(candidates.lane.nunique())
        result["placements"] = candidates.groupby(["path", "solid", "material", "tag", "classification"]).agg(
            slice_candidates=("lane", "size"), lanes=("lane", "nunique")).reset_index().to_dict("records")
    result["unknown_bound_records"] = unbounded
    if before != {str(path): digest(path) for path in paths}:
        raise ValueError("saved replay or source changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--census-source", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.replay, args.gdml, args.census_source, args.outdir)
