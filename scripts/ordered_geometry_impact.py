"""Locate actual omitted material relative to emitted tracker observations; never infer missing debit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest
from trkfastsim import tables

KEY = tables.KEY + ["sp"]


def summarize(muons, segments, cells, selected):
    if muons.duplicated(KEY).any():
        raise ValueError("duplicate muon identity")
    gas = segments.groupby(KEY).agg(first_gas_path_mm=("path_mm", "min"), last_gas_entry_path_mm=("path_mm", "max"),
                                    gas_rows=("path_mm", "size")).reset_index()
    joined = cells.merge(muons[KEY + ["stopped_interior", "path_interior_mm"]], on=KEY, how="left", validate="many_to_one")
    joined = joined.merge(gas, on=KEY, how="left", validate="many_to_one")
    if joined.path_interior_mm.isna().any():
        raise ValueError("omission cell lacks actual interior lane")
    joined["has_gas"] = joined.gas_rows.fillna(0) > 0
    joined["before_first_gas_mm"] = np.where(joined.has_gas,
        np.maximum(0., np.minimum(joined.cell_end_path_mm, joined.first_gas_path_mm) - joined.cell_start_path_mm), 0.)
    joined["after_last_gas_entry_mm"] = np.where(joined.has_gas,
        np.maximum(0., joined.cell_end_path_mm - np.maximum(joined.cell_start_path_mm, joined.last_gas_entry_path_mm)), 0.)
    selected_keys = pd.MultiIndex.from_frame(selected[tables.KEY])
    joined["selected_event"] = pd.MultiIndex.from_frame(joined[tables.KEY]).isin(selected_keys)
    grouped = joined.groupby(["state", "material", "has_gas", "selected_event", "stopped_interior"], dropna=False)
    totals = grouped.agg(cells=("path", "size"), total_mm=("cell_length_mm", "sum"),
        before_first_gas_mm=("before_first_gas_mm", "sum"), after_last_gas_entry_mm=("after_last_gas_entry_mm", "sum"),
        loader_inside_mm=("loader_aabb_inside_mm", "sum"), vd_inside_mm=("vd_cylinder_inside_mm", "sum")).reset_index()
    identities = joined[KEY + ["material", "has_gas", "selected_event", "stopped_interior"]].drop_duplicates()
    stops = muons.loc[muons.stopped_interior == 1].merge(gas, on=KEY, how="left", validate="one_to_one")
    affected = joined[KEY].drop_duplicates().assign(has_actual_omission=True)
    stops = stops.merge(affected, on=KEY, how="left", validate="one_to_one")
    stops["has_actual_omission"] = stops.has_actual_omission.fillna(False).astype(bool)
    return joined, totals, identities, stops


def run(directory, domain, outdir):
    directory, domain, outdir = map(Path, (directory, domain, outdir))
    paths = [directory / "muons.csv", directory / "segments.csv", domain / "omitted_cells.csv", directory / "ordered_steps.root"]
    before = {str(p): digest(p) for p in paths}
    sgs = tables.read(paths[3], "sgsteps")
    decisions = tables.trk_filter(sgs).reset_index()
    frames = summarize(pd.read_csv(paths[0]), pd.read_csv(paths[1]), pd.read_csv(paths[2]), decisions.loc[decisions.pass_trk])
    outdir.mkdir(parents=True, exist_ok=False)
    for name, frame in zip(("cells", "totals", "identities", "stops"), frames):
        frame.to_csv(outdir / f"{name}.csv", index=False)
    result = dict(scope="saved actual omission overlaps relative to gas entries and stopping identities; not causal rate or energy correction",
        omission_totals=frames[1].to_dict("records"), omission_primary_material_rows=len(frames[2]),
        stopped_primaries=len(frames[3]), stopped_with_actual_omission=int(frames[3].has_actual_omission.sum()),
        input_sha256=before, source_sha256={str(Path(__file__)): digest(__file__)},
        warnings=["per-placement lengths can overlap and are not net missing material", "after last gas entry is not after its exit or every VD observation",
                  "stopped means positive model cutoff, not validated G4 endpoint", "absence of unsupported leaf hits does not clear approximate polycones/ownership"], admitted_for_production=False)
    if before != {str(p): digest(p) for p in paths}:
        raise ValueError("saved geometry evidence changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.domain, args.outdir)
