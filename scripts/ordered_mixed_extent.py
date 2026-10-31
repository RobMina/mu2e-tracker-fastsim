"""Saved complete-ray-domain census/source/mixed-envelope gate before any larger producer evolution."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_torus_pair_probe import load_pair_sources
from scripts.transport_capture_audit import digest
from trkfastsim import boundarypolicy, guardednavigation, mixedenvelope, placementownership, wideomission


def run(directory, gdml, source, outdir, *, inclusive_certificates=False, native_planar=False):
    directory, gdml, source, outdir = map(Path, (directory, gdml, source, outdir))
    names = ("navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope")
    paths = [directory / "rays.npz", directory.parent / "census.json", gdml, *[source / (n + ".py") for n in names]]
    hashes = {str(p): digest(p) for p in paths}
    import trkfastsim
    executed = [Path(__file__), *Path(trkfastsim.__file__).parent.glob("*.py")]
    source_hashes = {str(p): digest(p) for p in executed}
    modules = load_pair_sources(source)
    old_census = json.loads(paths[1].read_text())
    with np.load(paths[0], allow_pickle=False) as saved:
        origin, direction, lengths, keys = saved["origin"], saved["direction"], saved["lengths"], saved["primary_keys"]
    query = placementownership.RaySelectedSources(gdml, old_census["bbox_mm"], local_planes=True)
    raw = query(origin, direction, lengths)
    census = modules["navigationcensus"].census(gdml, query.bbox)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "census.json").write_text(json.dumps(census, indent=2) + "\n")
    raw.to_csv(outdir / "raw_hits.csv", index=False)
    np.savez_compressed(outdir / "rays.npz", origin=origin, direction=direction, lengths=lengths, primary_keys=keys,
                        source_selection_bbox_mm=query.bbox)
    probe = wideomission.WideOmission(modules["omittedpath"], authenticate_xtru=True)
    envelope = mixedenvelope.MixedEnvelope(modules["supportenvelope"], modules["supportnavigation"])
    if inclusive_certificates or native_planar:
        from trkfastsim.inclusiveenvelope import CertifiedOmission, InclusiveEnvelope
        probe = CertifiedOmission(probe, modules["omittedpath"])
        envelope = InclusiveEnvelope(modules["supportenvelope"], modules["supportnavigation"], probe)
    if native_planar:
        from trkfastsim.nativeenvelope import NativeOmission, NativeEnvelope
        probe = NativeOmission(probe, modules["omittedpath"])
        envelope = NativeEnvelope(modules["supportenvelope"], modules["supportnavigation"], probe)
    cells, reports = probe.probe(gdml, census["records"], origin, direction, lengths, min_cell_mm=1e-9, max_cells=1000000)
    cells.to_csv(outdir / "cells.csv", index=False)
    (outdir / "placements.json").write_text(json.dumps(reports, indent=2) + "\n")
    adjusted, endpoints = boundarypolicy.coalesce(raw, lengths, max_shift_mm=1e-10, leading_resolution_mm=1e-9)
    normalized, material = query.normalize(adjusted, lengths)
    physical = placementownership.parent_owners(cells, normalized, max_pairs=2000000)
    normalized.to_csv(outdir / "normalized.csv", index=False)
    material.to_csv(outdir / "material_comparison.csv", index=False)
    physical.to_csv(outdir / "physical_parent_comparisons.csv", index=False)
    query.relationships.to_csv(outdir / "physical_terms.csv", index=False)
    pair = envelope.compose(
        normalized, lengths, cells, reports, census["records"], endpoint_budget_mm=0., min_cell_mm=1e-9,
        max_component_mm=1e-6, max_ray_unknown_mm=1e-5)
    for name, value in pair.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(outdir / (name + ".csv"), index=False)
        elif isinstance(value, dict) and name.endswith("control"):
            control_dir = outdir / name
            control_dir.mkdir()
            for key, table in value.items():
                if isinstance(table, pd.DataFrame):
                    table.to_csv(control_dir / (key + ".csv"), index=False)
            (control_dir / "summary.json").write_text(json.dumps(value["summary"], indent=2) + "\n")
    result = dict(scope="entire saved ray-domain stored sources/census plus explicit mixed geometry pair; no transport",
        rays=len(origin), census_counts=census["counts"], summary=pair["summary"], evaluated_placements=len(reports),
        unresolved_placement_reports=[r for r in reports if not r["resolved"]], input_sha256=hashes,
        source_sha256=source_hashes, admitted_for_physics=False)
    if hashes != {str(p): digest(p) for p in paths} or source_hashes != {str(p): digest(p) for p in executed}:
        raise ValueError("mixed extent source inputs changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("gdml", "source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--inclusive-certificates", action="store_true")
    parser.add_argument("--native-planar", action="store_true")
    args = parser.parse_args()
    run(args.directory, args.gdml, args.source, args.outdir, inclusive_certificates=args.inclusive_certificates, native_planar=args.native_planar)
