"""Guarded inner/outer torus geometry on actual selected incoming slices; no transport correction."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from mu2efastsim import world
from scripts.ordered_omitted_solid_audit import load_geometry_sources
from scripts.transport_capture_audit import digest
from trkfastsim import convexpath, crossings, geometry, materialpath, steps, tables

KEY = tables.KEY + ["sp"]


class StoredHitQuery:
    """Load immutable labelled stored geometry once; evaluate new rays without re-reading GDML."""

    def __init__(self, gdml, bbox):
        tubes = world.load_tubes(gdml, bbox=bbox, skip_names=steps.INTERIOR_SKIP)
        labelled = [(f"tube:{i}|{t[0]}", *t[1:]) for i, t in enumerate(tubes) if t[0] != "StrawGas"]
        self.shell = world.CoaxialShells(labelled, geometry.DETECTOR_ORIGIN_EXTRACTED)
        pieces, _ = world.signed_subtree(gdml, steps.CALO_MOTHER, bbox=bbox, labels=True)
        pieces = [p for p in pieces if not p[1].endswith("|StrawGas")]
        self.pieces = world.SignedPieceSet(pieces, cell=steps.CALO_CELL_MM) if pieces else None

    def __call__(self, origin, direction, length):
        tube = materialpath.tube_hits(self.shell, origin, direction, length)
        labels = tube.material.str.rsplit("|", n=1, expand=True).reindex(columns=[0, 1])
        tube["source"], tube["material"], tube["sign"] = labels[0], labels[1], 1
        hits = [tube]
        if self.pieces is not None:
            for sign, packed in convexpath._sets(self.pieces):
                for lane, piece, start, end in convexpath._hits(packed, origin, direction, length):
                    labels = pd.Series(np.asarray(packed.mats, object)[packed.mat[piece]]).str.rsplit("|", n=1, expand=True).reindex(columns=[0, 1])
                    hits.append(pd.DataFrame(dict(lane=lane, s0_mm=start, s1_mm=end, material=labels[1], source=labels[0], sign=sign)))
        return pd.concat(hits, ignore_index=True)


def stored_hits(gdml, bbox, origin, direction, length):
    tubes = world.load_tubes(gdml, bbox=bbox, skip_names=steps.INTERIOR_SKIP)
    labelled = [(f"tube:{i}|{t[0]}", *t[1:]) for i, t in enumerate(tubes) if t[0] != "StrawGas"]
    shell = world.CoaxialShells(labelled, geometry.DETECTOR_ORIGIN_EXTRACTED)
    tube = materialpath.tube_hits(shell, origin, direction, length)
    labels = tube.material.str.rsplit("|", n=1, expand=True).reindex(columns=[0, 1])
    tube["source"], tube["material"], tube["sign"] = labels[0], labels[1], 1
    pieces, _ = world.signed_subtree(gdml, steps.CALO_MOTHER, bbox=bbox, labels=True)
    pieces = [p for p in pieces if not p[1].endswith("|StrawGas")]
    hits = [tube]
    if pieces:
        for sign, packed in convexpath._sets(world.SignedPieceSet(pieces, cell=steps.CALO_CELL_MM)):
            for lane, piece, start, end in convexpath._hits(packed, origin, direction, length):
                labels = pd.Series(np.asarray(packed.mats, object)[packed.mat[piece]]).str.rsplit("|", n=1, expand=True).reindex(columns=[0, 1])
                hits.append(pd.DataFrame(dict(lane=lane, s0_mm=start, s1_mm=end, material=labels[1], source=labels[0], sign=sign)))
    return pd.concat(hits, ignore_index=True)


def load_pair_sources(source):
    modules = load_geometry_sources(source)
    for name in ("supportnavigation", "supportenvelope"):
        qualified = f"trkfastsim.{name}"
        if qualified in sys.modules:
            raise ValueError("refusing imported geometry module replacement")
        spec = importlib.util.spec_from_file_location(qualified, Path(source) / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def run(directory, impact, gdml, source, outdir):
    directory, impact, gdml, source, outdir = map(Path, (directory, impact, gdml, source, outdir))
    names = ("navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope")
    paths = [directory / "traversed.npz", directory / "segments.csv", directory / "complete.json", impact / "cells.csv", gdml,
             *[source / f"{name}.py" for name in names]]
    hashes = {str(p): digest(p) for p in paths}
    cells = pd.read_csv(impact / "cells.csv")
    incoming = cells[(cells.material == "StainlessSteel") & cells.selected_event & cells.has_gas & (cells.before_first_gas_mm > 0)]
    identities = incoming[KEY].drop_duplicates()
    if identities.empty:
        raise ValueError("no actual selected incoming torus witness")
    with np.load(paths[0], allow_pickle=False) as saved:
        labels = pd.DataFrame(saved["muon_keys"], columns=KEY).assign(lane=np.arange(len(saved["muon_keys"])))
        chosen_lanes = labels.merge(identities, on=KEY, validate="one_to_one").lane.to_numpy(np.int64)
        first_gas = pd.read_csv(paths[1]).groupby(KEY).path_mm.min().reset_index()
        stop = labels.merge(first_gas, on=KEY, how="left", validate="one_to_one").path_mm.to_numpy()
        lane = saved["slice_it"]
        length = saved["slice_end"] - saved["slice_start"]
        use = np.isin(lane, chosen_lanes) & (saved["slice_start"] < stop[lane])
        origin, direction = saved["slice_pos0"][use], saved["slice_dir"][use]
        length = np.minimum(length[use], stop[lane[use]] - saved["slice_start"][use])
        selected = labels.iloc[lane[use]][KEY].reset_index(drop=True)
        selected["start_path_mm"], selected["requested_length_mm"] = saved["slice_start"][use], length
    modules = load_pair_sources(source)
    geom = geometry.from_gdml(gdml)
    lo, hi = crossings.panel_boxes(geom)
    bbox = (lo.min(0) - 300., hi.max(0) + 300.)
    census = modules["navigationcensus"].census(gdml, bbox)
    omission, reports = modules["omittedpath"].probe(gdml, census["records"], origin, direction, length,
                                                   min_cell_mm=1e-7, max_cells=1000000)
    raw = stored_hits(gdml, bbox, origin, direction, length)
    outdir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(outdir / "slices.csv", index=False)
    raw.to_csv(outdir / "raw_stored_hits.csv", index=False)
    omission.to_csv(outdir / "omitted_cells.csv", index=False)
    (outdir / "placements.json").write_text(json.dumps(reports, indent=2) + "\n")
    summaries = {}
    for tag, component_budget, ray_budget in (("zero_unknown", 0., 0.), ("reviewed_budgets", 1e-6, 1e-5)):
        result = modules["supportenvelope"].compose(raw, length, omission, reports, census["records"],
            endpoint_budget_mm=1e-10, min_cell_mm=1e-7, max_component_mm=component_budget, max_ray_unknown_mm=ray_budget)
        for name in ("inner_intervals", "outer_intervals", "material_bounds", "decisions", "refusals", "unknown_components",
                     "parent_comparisons", "ownership_comparisons", "replacement_terms"):
            result[name].to_csv(outdir / f"{tag}.{name}.csv", index=False)
        summaries[tag] = result["summary"]
    result = dict(scope="actual selected incoming-slice bounded geometric pair only; no loss/scattering/history correction",
        identities=identities.to_dict("records"), slices=len(length), summaries=summaries,
        domain="actual saved prefix through first gas; loader bbox selects stored sources only, no ray clipping",
        warnings=["opaque upstream state; no charged world query attribution", "subset probes recomputed, never reuse guards from another batch",
                  "paired geometric admission is not energy/time or full GDML admission", "engineering budgets unchanged, zero-unknown refusal control retained"],
        input_sha256=hashes, source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_production=False)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("actual torus-pair geometry input changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("impact", "gdml", "source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    run(args.directory, args.impact, args.gdml, args.source, args.outdir)
