"""Recompute stored parent ownership on full actual refused-ray extents, without transport."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_torus_pair_probe import load_pair_sources
from scripts.transport_capture_audit import digest
from trkfastsim import boundarypolicy, placementownership


def run(directory, gdml, source, outdir, *, wide_intersections=False, local_planes=False):
    directory, gdml, source, outdir = map(Path, (directory, gdml, source, outdir))
    inputs = [directory / "rays.npz", directory / "cells.csv", directory.parent / "census.json", gdml]
    hashes = {str(p): digest(p) for p in inputs}
    census = json.loads(inputs[2].read_text())
    cells = pd.read_csv(inputs[1])
    modules = load_pair_sources(source)
    with np.load(inputs[0], allow_pickle=False) as saved:
        origin, direction, lengths = saved["origin"], saved["direction"], saved["lengths"]
    query = placementownership.RaySelectedSources(gdml, census["bbox_mm"], wide_intersections=wide_intersections, local_planes=local_planes)
    raw = query(origin, direction, lengths)
    adjusted, evidence = boundarypolicy.coalesce(raw, lengths, max_shift_mm=1e-10, leading_resolution_mm=1e-9)
    outdir.mkdir(parents=True, exist_ok=False)
    raw.to_csv(outdir / "raw_hits.csv", index=False)
    adjusted.to_csv(outdir / "adjusted_hits.csv", index=False)
    query.relationships.to_csv(outdir / "physical_terms.csv", index=False)
    try:
        normalized, comparison = query.normalize(adjusted, lengths)
    except ValueError as error:
        bad = modules["supportnavigation"]._bad_occupancy(adjusted)
        if len(bad):
            safe = lengths.copy()
            safe[bad] = 0.
            selected = adjusted[~adjusted.lane.isin(bad)]
            normalized, comparison = query.normalize(selected, safe)
            normalized.to_csv(outdir / "nonconflicting_normalized.csv", index=False)
            stored = boundarypolicy.partition(normalized, safe)
            possible = cells[~cells.lane.isin(bad)]
            parent, _, mismatch, ambiguous = modules["supportenvelope"]._parent_checks(possible, stored, normalized, bad, 2000000)
            parent.to_csv(outdir / "nonconflicting_parent_comparisons.csv", index=False)
            (outdir / "partial_geometry_only.json").write_text(json.dumps(dict(all_requested_rays=len(lengths),
                conflicting_lanes=bad.tolist(), parent_mismatch_on_nonconflicting_lanes=mismatch.tolist(),
                ambiguous_on_nonconflicting_lanes=ambiguous.tolist(),
                warning="geometric localization only, every conflicting lane retained/refused; no partial producer or rate output"), indent=2) + "\n")
        result = dict(status="refused", reason=str(error), actual_bbox_mm=query.bbox.tolist(), input_sha256=hashes,
            scope="source-selection expansion must not replace a parent mismatch with an unreported overlap", admitted_for_physics=False)
        (outdir / "refused.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return result
    normalized.to_csv(outdir / "normalized.csv", index=False)
    comparison.to_csv(outdir / "material_comparison.csv", index=False)
    stored = boundarypolicy.partition(normalized, lengths)
    parent, ownership, mismatch, ambiguous = modules["supportenvelope"]._parent_checks(cells, stored, normalized, np.array([], np.int64), 2000000)
    physical = placementownership.parent_owners(cells, normalized, max_pairs=2000000)
    parent.to_csv(outdir / "parent_comparisons.csv", index=False)
    physical.to_csv(outdir / "physical_parent_comparisons.csv", index=False)
    ownership.to_csv(outdir / "ownership.csv", index=False)
    result = dict(scope="actual source-selection parent/ownership refinement only; no loss or transport", rays=len(origin),
        old_bbox_mm=census["bbox_mm"], actual_bbox_mm=query.bbox.tolist(), parent_mismatch_lanes=mismatch.tolist(),
        ambiguous_lanes=ambiguous.tolist(), parent_comparisons=len(parent), physical_parent_comparisons=len(physical),
        strict_material_comparison_equal=bool((comparison.original == comparison.normalized).all()),
        input_sha256=hashes, source_sha256={str(Path(__file__)): digest(__file__), str(Path(placementownership.__file__)): digest(placementownership.__file__)},
        admitted_for_physics=False)
    if hashes != {str(p): digest(p) for p in inputs}:
        raise ValueError("source extent audit inputs changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("gdml", "source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--wide-intersections", action="store_true")
    parser.add_argument("--local-planes", action="store_true")
    args = parser.parse_args()
    run(args.directory, args.gdml, args.source, args.outdir, wide_intersections=args.wide_intersections, local_planes=args.local_planes)
