"""Locate actual omission cells on full producer identity/path relative to tracker steps and surfaces."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest
from trkfastsim import tables

KEY = tables.KEY + ["sp"]


def locate(vertices, segments, cells):
    vertices = vertices.sort_values(KEY + ["path"], kind="stable").reset_index(drop=True)
    length = vertices.groupby(KEY, sort=False).path.shift(-1) - vertices.path
    starts = vertices.loc[length > 0].reset_index(drop=True)
    if cells.empty:
        return pd.DataFrame()
    index = cells.lane.to_numpy(dtype=np.int64)
    if (index < 0).any() or (index >= len(starts)).any():
        raise ValueError("omission slice index outside ordered history")
    hit = starts.iloc[index][KEY + ["path", "x", "y", "z", "ux", "uy", "uz", "kinetic_MeV", "time_ns"]].reset_index(drop=True)
    hit["cell_start_path_mm"] = hit.path.to_numpy() + cells.s0_mm.to_numpy()
    hit["cell_end_path_mm"] = hit.path.to_numpy() + cells.s1_mm.to_numpy()
    for name in ("state", "material", "source", "ancestors_verified", "replaces_material"):
        hit[name] = cells[name].to_numpy()
    last = segments.groupby(KEY).path_mm.max().rename("last_emitted_gas_entry_path_mm").reset_index()
    hit = hit.merge(last, on=KEY, how="left", validate="many_to_one")
    hit["after_last_emitted_gas_entry"] = hit.cell_start_path_mm > hit.last_emitted_gas_entry_path_mm
    return hit


def run(directory, omission, out):
    directory, omission, out = Path(directory), Path(omission), Path(out)
    paths = [directory / "vertices.csv", directory / "segments.csv", omission / "omitted_cells.csv"]
    before = {str(path): digest(path) for path in paths}
    vertices, segments, cells = [pd.read_csv(path) for path in paths]
    witnesses = locate(vertices, segments, cells)
    if before != {str(path): digest(path) for path in paths}:
        raise ValueError("saved witness input changed")
    result = dict(scope="actual omission location on saved paths; no corrected energy/acceptance inference",
        input_sha256=before, witnesses=witnesses.astype(object).where(pd.notna(witnesses), None).to_dict("records"))
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--omission", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.omission, args.out)
