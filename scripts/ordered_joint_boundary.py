"""Exact stored-planar boundary witnesses for every saved possible-leaf overlap; no transport repair."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_torus_pair_probe import load_pair_sources
from scripts.transport_capture_audit import digest
from trkfastsim import exactplanar


def run(directory, overlaps, gdml, source, outdir, *, max_work=2000000, native_model=False):
    directory, overlaps, gdml, source, outdir = map(Path, (directory, overlaps, gdml, source, outdir))
    paths = [directory / "cells.csv", directory / "rays.npz", directory / "placements.json", directory / "census.json",
             overlaps / "overlaps.csv", overlaps / "complete.json", gdml,
             *[source / (n + ".py") for n in ("navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope")]]
    hashes = {str(p): digest(p) for p in paths}
    from trkfastsim import nativeplanar
    executed = [Path(__file__), Path(exactplanar.__file__), Path(nativeplanar.__file__)]
    source_hashes = {str(p): digest(p) for p in executed}
    modules = load_pair_sources(source)
    old = modules["omittedpath"]
    pairs = pd.read_csv(overlaps / "overlaps.csv")
    membership = pd.concat([pairs[["lane", key]].rename(columns={key: "source"}) for key in ("source_a", "source_b")]).drop_duplicates()
    wanted = sorted(membership.source.unique())
    index = old.PlacementIndex(gdml, wanted)
    literal = nativeplanar.LiteralIndex(gdml, wanted) if native_model else None
    reports = {r["source"]: r for r in json.loads((directory / "placements.json").read_text())}
    with np.load(directory / "rays.npz", allow_pickle=False) as saved:
        origin, direction, lengths, keys = saved["origin"], saved["direction"], saved["lengths"], saved["primary_keys"]
    outdir.mkdir(parents=True, exist_ok=False)
    occupied, summaries, contacts = [], [], []
    for i, path in enumerate(wanted):
        lanes = membership.loc[membership.source == path, "lane"].to_numpy(np.int64)
        placement = index.placements[path]
        solid = index.solids[placement["solid"]]
        evaluator = nativeplanar.intervals if native_model else exactplanar.intervals
        model_placement = literal.placements[path] if native_model else placement
        rows, faults, result = evaluator(solid, index.solids, model_placement, origin[lanes], direction[lanes], lengths[lanes],
                                        reviewed=old, max_work=max_work)
        rows["lane"] = lanes[rows.lane.to_numpy(np.int64)]
        faults["lane"] = lanes[faults.lane.to_numpy(np.int64)]
        rows["source"], rows["material"] = path, reports[path]["material"]
        exactplanar.serialize_roots(rows).to_csv(outdir / f"placement_{i}.occupied.csv", index=False)
        exactplanar.serialize_roots(faults).to_csv(outdir / f"placement_{i}.faults.csv", index=False)
        contacts.extend(lanes[np.asarray(result["coplanar_lanes"], np.int64)].tolist())
        summaries.append(dict(source=path, selected_full_ray_lanes=lanes.tolist(), method=result))
        occupied.append(rows)
    occupied = pd.concat(occupied, ignore_index=True)
    sizes = occupied.groupby("lane").size()
    if int((sizes * sizes).sum()) > max_work:
        raise ValueError("exact joint occupied-pair cap")
    cross = occupied.merge(occupied, on="lane", suffixes=("_a", "_b"))
    cross = cross[cross.source_a < cross.source_b].copy()
    cross["q0"] = np.maximum(cross.q0_a, cross.q0_b)
    cross["q1"] = np.minimum(cross.q1_a, cross.q1_b)
    conflict = cross[cross.q1 > cross.q0].copy()
    contact = cross[cross.q1 == cross.q0].copy()
    for frame, name in ((conflict, "positive_overlap"), (contact, "exact_contacts")):
        frame["width_lower_mm"] = exactplanar.export(frame.q1.to_numpy(object) - frame.q0.to_numpy(object), False)
        frame["width_upper_mm"] = exactplanar.export(frame.q1.to_numpy(object) - frame.q0.to_numpy(object), True)
        for j, key in enumerate(("run", "subrun", "event", "sp")):
            frame[key] = keys[frame.lane.to_numpy(np.int64), j]
        clean = frame.drop(columns=[k for k in frame if k.startswith(("q0_", "q1_"))])
        exactplanar.serialize_roots(clean).to_csv(outdir / (name + ".csv"), index=False)
    totals = conflict.groupby(["source_a", "source_b", "material_a", "material_b"], as_index=False).agg(
        lanes=("lane", "nunique"), atoms=("lane", "size"), max_width_upper_mm=("width_upper_mm", "max"), sum_width_upper_mm=("width_upper_mm", "sum"))
    totals.to_csv(outdir / "conflict_sources.csv", index=False)
    result = dict(scope="all saved overlap lanes/sources, exact frozen planar model localization only; not hierarchy or transport admission",
        model=nativeplanar.POLICY if native_model else exactplanar.POLICY,
        saved_overlap_pairs=len(pairs), saved_overlap_lanes=int(pairs.lane.nunique()), evaluated_sources=len(wanted),
        source_lane_pairs=len(membership), positive_overlap_atoms=len(conflict), positive_overlap_lanes=int(conflict.lane.nunique()),
        exact_contact_atoms=len(contact), coplanar_lanes=sorted(set(contacts)),
        invalid_occupancy_atoms=sum(r["method"]["invalid_occupancy_atoms"] for r in summaries),
        conflict_sources=totals.to_dict("records"), placements=summaries,
        input_sha256=hashes, source_sha256=source_hashes, admitted_for_physics=False)
    if hashes != {str(p): digest(p) for p in paths} or source_hashes != {str(p): digest(p) for p in executed}:
        raise ValueError("joint predicate inputs or sources changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("overlaps", "gdml", "source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--native-model", action="store_true")
    args = parser.parse_args()
    run(args.directory, args.overlaps, args.gdml, args.source, args.outdir, native_model=args.native_model)
