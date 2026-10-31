"""Recheck completed saved slices and source-prefix acquisition under stricter input gates."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts import ordered_response_probe as response
from scripts.ordered_stock_preflight import run as prefix
from scripts.transport_capture_audit import digest
from trkfastsim import geometry, tables


def run(directory, preflight, source_art, gdml, trusted_source_sha256, outdir):
    directory, preflight, outdir = map(Path, (directory, preflight, outdir))
    outdir.mkdir(parents=True, exist_ok=False)
    paths = [directory / "ordered_steps.root", directory / "traversed.npz", Path(gdml)]
    before = {str(p): digest(p) for p in paths}
    merged = tables.read(paths[0], "sgsteps")
    with np.load(paths[1], allow_pickle=False) as saved:
        _, _, audit = response.slice_rows(merged, saved, geometry.from_gdml(gdml))
    prefix(str(paths[0]), str(preflight / "source.root"), str(outdir / "prefix.json"), expected_events=1000,
        source_log=str(preflight / "source.log"), source_art=source_art, trusted_source_sha256=trusted_source_sha256)
    if before != {str(p): digest(p) for p in paths}:
        raise ValueError("saved physical slices changed")
    result = dict(status="saved_gates_pass", slice_audit=audit, input_sha256=before,
        scope="stricter saved-input gates only; transport/response/stock jobs not repeated",
        source_sha256={str(Path(__file__)): digest(__file__), str(Path(response.__file__)): digest(response.__file__)})
    (outdir / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "preflight", "source-art", "gdml", "trusted-source-sha256", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    run(args.directory, args.preflight, args.source_art, args.gdml, args.trusted_source_sha256, args.outdir)
