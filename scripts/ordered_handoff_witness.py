"""Saved actual omission coordinates versus the world/interior handoff; no energy repair."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest
from trkfastsim import geometry, steps, storednavigation, tables

KEY = tables.KEY + ["sp"]


def cell_geometry(cells):
    if cells.empty or not np.isfinite(cells[["path", "cell_start_path_mm", "cell_end_path_mm", "x", "y", "z", "ux", "uy", "uz"]].to_numpy()).all():
        raise ValueError("finite positive actual omission cells required")
    origin = geometry.to_mu2e(cells[["x", "y", "z"]].to_numpy(float))
    direction = cells[["ux", "uy", "uz"]].to_numpy(float)
    start = cells.cell_start_path_mm.to_numpy() - cells.path.to_numpy()
    end = cells.cell_end_path_mm.to_numpy() - cells.path.to_numpy()
    if (start < 0).any() or (end <= start).any() or not np.allclose(np.linalg.norm(direction, axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError("positive forward cells and unit directions required")
    return origin + direction * start[:, None], origin + direction * end[:, None], direction, end - start


def run(directory, cells_path, gdml, outdir):
    directory, cells_path, gdml, outdir = map(Path, (directory, cells_path, gdml, outdir))
    paths = [directory / "ordered_steps.root", cells_path, gdml]
    hashes = {str(p): digest(p) for p in paths}
    cells = pd.read_csv(cells_path)
    chosen = cells[(cells.material == "StainlessSteel") & cells.selected_event & cells.has_gas
                   & (cells.before_first_gas_mm > 0)].copy().reset_index(drop=True)
    if chosen.empty:
        raise ValueError("no selected incoming-steel omission witness")
    a, b, direction, length = cell_geometry(chosen)
    geom = geometry.from_gdml(gdml)
    plane = float(steps.plane_y(geom))
    ymin, ymax = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    chosen["cell_y_min_mm"], chosen["cell_y_max_mm"] = ymin, ymax
    chosen["entirely_below_world_plane"] = ymax < plane
    chosen["entirely_above_world_plane"] = ymin > plane
    interior = steps.TrackerInterior(gdml, geom)
    partition = storednavigation.support_intervals(interior.shells, a, direction, length,
        endpoint_budget_mm=1e-10, leading_resolution_mm=1e-9)
    partition["witness_cell_row"] = partition.lane
    partition["expected_parent"] = chosen.replaces_material.to_numpy()[partition.lane.to_numpy(np.int64)]
    partition["matches_declared_parent"] = partition.material == partition.expected_parent
    partition["span_mm"] = partition.s1_mm - partition.s0_mm
    outdir.mkdir(parents=True, exist_ok=False)
    chosen.to_csv(outdir / "cells.csv", index=False)
    partition.to_csv(outdir / "stored_parent_partition.csv", index=False)
    result = dict(scope="selected incoming actual omission geometry and stored-parent checks; upstream charged history unavailable",
        world_plane_y_mm=plane, selected_primaries=chosen[KEY].drop_duplicates().to_dict("records"), cells=len(chosen),
        cell_states=chosen.groupby("state").agg(length_mm=("cell_length_mm", "sum")).reset_index().to_dict("records"),
        y_min_mm=float(ymin.min()), y_max_mm=float(ymax.max()), all_cells_below_world_plane=bool((ymax < plane).all()),
        stored_material_spans=partition.groupby(["material", "expected_parent", "matches_declared_parent"]).span_mm.sum().reset_index().to_dict("records"),
        warnings=["world columns use generated ray, interior uses scattered/back-projected plane state",
            "below the endpoint plane on actual cells excludes direct upstream height-band occupancy there, not an equivalent generated-ray loss elsewhere",
            "parent agreement on these certified/possible spans is not complete raw-placement ownership or transport admission",
            "no posthoc debit, no arbitrary exterior clipping, no new material priority"], input_sha256=hashes,
        source_sha256={str(p): digest(p) for p in (Path(__file__), Path(storednavigation.__file__), Path(steps.__file__))}, admitted_for_production=False)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("handoff evidence changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--cells", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.cells, args.gdml, args.outdir)
