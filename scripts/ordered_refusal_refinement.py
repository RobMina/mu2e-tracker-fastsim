"""Reprobe an actual refused query without transport; compare cell refinement under unchanged geometry budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_torus_pair_probe import load_pair_sources
from scripts.transport_capture_audit import digest


def run(directory, gdml, source, outdir, *, wide=False):
    directory, gdml, source, outdir = map(Path, (directory, gdml, source, outdir))
    inputs = [directory / "rays.npz", directory / "receipt.json", directory / "ownership_normalized_hits.csv",
              directory.parent / "census.json", gdml, *[source / f"{n}.py" for n in (
                  "navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope")]]
    hashes = {str(p): digest(p) for p in inputs}
    modules = load_pair_sources(source)
    probe = modules["omittedpath"]
    if wide:
        from trkfastsim.wideomission import WideOmission
        probe = WideOmission(probe)
    receipt = json.loads(inputs[1].read_text())
    if receipt["status"] != "refused":
        raise ValueError("actual refused query required")
    census = json.loads(inputs[3].read_text())
    raw = pd.read_csv(inputs[2])
    with np.load(inputs[0], allow_pickle=False) as saved:
        origin, direction, length = saved["origin"], saved["direction"], saved["lengths"]
        keys = saved["primary_keys"]
    outdir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for cell_mm in (receipt["min_cell_mm"], receipt["min_cell_mm"] / 10.):
        cells, reports = probe.probe(gdml, census["records"], origin, direction, length,
            min_cell_mm=cell_mm, max_cells=receipt["max_cells"])
        pair = modules["supportenvelope"].compose(raw, length, cells, reports, census["records"],
            endpoint_budget_mm=0., min_cell_mm=cell_mm, max_component_mm=receipt["max_component_mm"],
            max_ray_unknown_mm=receipt["max_ray_unknown_mm"], max_pairs=receipt["max_pairs"])
        target = outdir / f"cell_{cell_mm:.0e}"
        target.mkdir()
        cells.to_csv(target / "cells.csv", index=False)
        for name in ("unknown_components", "refusals", "decisions", "material_bounds"):
            pair[name].to_csv(target / (name + ".csv"), index=False)
        (target / "placements.json").write_text(json.dumps(reports, indent=2) + "\n")
        refused = pair["refusals"].copy()
        for j, key in enumerate(("run", "subrun", "event", "sp")):
            refused[key] = keys[refused.lane.to_numpy(np.int64), j]
        refused.to_csv(target / "refused_identities.csv", index=False)
        summaries.append(pair["summary"])
    if hashes != {str(p): digest(p) for p in inputs}:
        raise ValueError("refusal source data changed")
    result = dict(scope="actual saved query geometry-refinement sensitivity; no producer/rate output", summaries=summaries,
        arithmetic="stored-placement-wide-torus-v1" if wide else "reviewed-binary64-control",
        input_sha256=hashes, source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("gdml", "source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--wide", action="store_true")
    args = parser.parse_args()
    run(args.directory, args.gdml, args.source, args.outdir, wide=args.wide)
