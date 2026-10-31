"""Recompose actual saved initial rays with explicit authenticated xtru certificate policy; no evolution."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_torus_pair_probe import load_pair_sources
from scripts.transport_capture_audit import digest
from trkfastsim import mixedenvelope, wideomission


def run(directory, gdml, source, outdir):
    directory, gdml, source, outdir = map(Path, (directory, gdml, source, outdir))
    paths = [directory / "rays.npz", directory / "ownership_normalized_hits.csv", directory.parent / "census.json", gdml]
    hashes = {str(p): digest(p) for p in paths}
    modules = load_pair_sources(source)
    census = json.loads(paths[2].read_text())
    raw = pd.read_csv(paths[1])
    with np.load(paths[0], allow_pickle=False) as saved:
        origin, direction, lengths = saved["origin"], saved["direction"], saved["lengths"]
    probe = wideomission.WideOmission(modules["omittedpath"], authenticate_xtru=True)
    cells, reports = probe.probe(gdml, census["records"], origin, direction, lengths, min_cell_mm=1e-9, max_cells=1000000)
    pair = mixedenvelope.MixedEnvelope(modules["supportenvelope"], modules["supportnavigation"]).compose(
        raw, lengths, cells, reports, census["records"], endpoint_budget_mm=0., min_cell_mm=1e-9,
        max_component_mm=1e-6, max_ray_unknown_mm=1e-5)
    outdir.mkdir(parents=True, exist_ok=False)
    cells.to_csv(outdir / "cells.csv", index=False)
    (outdir / "placements.json").write_text(json.dumps(reports, indent=2) + "\n")
    for name, value in pair.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(outdir / (name + ".csv"), index=False)
    result = dict(scope="mixed-family geometric policy on original stored source selection; not full parent repair or transport", summary=pair["summary"],
        input_sha256=hashes, source_sha256={str(Path(__file__)): digest(__file__), str(Path(mixedenvelope.__file__)): digest(mixedenvelope.__file__),
            str(Path(wideomission.__file__)): digest(wideomission.__file__)}, admitted_for_physics=False)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("mixed saved geometry inputs changed")
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
