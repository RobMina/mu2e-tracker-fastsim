"""Bounded geometry-only audit on retained G4 envelope seeds; no transport or G4 generation."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import world
from .. import convexpath, crossings, geometry, materialpath, steps
from ..interiorpath import interior_intervals
from .ledgercheck import sha256, source_manifest

KEY = ["run", "subrun", "event", "sp"]
SCOPE = ("Geometry-only stored-model audit on bounded retained-reference seeds; no energy, timing, "
         "scattering, acceptance denominator, independent GDML truth or default promotion.")


def read_reference(path, max_tree_rows=2000000):
    """Read explicit highest cycles and actual entries, refusing oversized or duplicate identities."""
    if not isinstance(max_tree_rows, int) or isinstance(max_tree_rows, bool) or max_tree_rows <= 0:
        raise ValueError("positive integer tree row bound required")
    frames, info = {}, {}
    columns = {"gen": KEY[:3], "particles": KEY[:3] + ["key", "ismcprim", "pdg"],
               "surfsteps": KEY + ["surf", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]}
    with uproot.open(path) as root:
        directory = root["dump"]
        for name in columns:
            cycles = [int(k.rsplit(";", 1)[1]) for k in directory.keys(cycle=True, recursive=False)
                      if k.rsplit(";", 1)[0] == name]
            if not cycles:
                raise ValueError(f"missing dump/{name}")
            obj = f"{name};{max(cycles)}"
            tree = directory[obj]
            if tree.num_entries > max_tree_rows:
                raise ValueError(f"{obj} exceeds explicit tree row bound")
            frames[name] = tree.arrays(columns[name], library="pd")
            info[name] = dict(object=f"dump/{obj}", entries=int(tree.num_entries), cycle=max(cycles))
    return frames, info


def reference_seeds(frames, bbox, max_rows, length_mm):
    """Select first envelope midpoint per full primary identity, then stable bounded identity order."""
    if (not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0
            or not np.isfinite(length_mm) or length_mm <= 0):
        raise ValueError("positive integer seed bound and finite path length required")
    particles = frames["particles"].rename(columns={"key": "sp"})
    prim = particles[(particles.ismcprim == 1) & (particles.pdg.abs() == 13)][KEY]
    if prim.duplicated(KEY).any():
        raise ValueError("duplicate primary identities")
    surface = frames["surfsteps"].copy()
    surface["reference_surface_row"] = np.arange(len(surface))
    envelope = surface[surface.surf.isin((0, 2, 4))].merge(prim, on=KEY, validate="many_to_one")
    if not np.isfinite(envelope.t.to_numpy(float)).all():
        raise ValueError("nonfinite primary-envelope time; first-crossing selection is undefined")
    selected = envelope.sort_values(KEY + ["t", "reference_surface_row"], kind="stable").drop_duplicates(KEY)
    if not np.isfinite(selected[KEY].to_numpy(float)).all() or (selected[KEY] != selected[KEY].astype("int64")).any().any():
        raise ValueError("identities must be finite integers")
    mid = .5 * (selected[["x0", "y0", "z0"]].to_numpy(float) + selected[["x1", "y1", "z1"]].to_numpy(float))
    o = geometry.to_mu2e(mid)
    pv = selected[["px", "py", "pz"]].to_numpy(float)
    p = np.linalg.norm(pv, axis=1)
    valid = np.isfinite(o).all(axis=1) & np.isfinite(p) & (p > 0)
    inside = ((o >= bbox[0]) & (o <= bbox[1])).all(axis=1)
    mask = valid & inside
    selected = selected.loc[mask].copy()
    o, pv, p = o[mask], pv[mask], p[mask]
    d = pv / p[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        exit_s = np.where(d > 0, (bbox[1] - o) / d, (bbox[0] - o) / d)
    exit_s = np.where(d == 0, np.inf, exit_s).min(axis=1)
    lengths = np.minimum(length_mm, exit_s)
    positive = lengths > 0
    selected = selected.loc[positive].copy()
    o, pv, lengths = o[positive], pv[positive], lengths[positive]
    eligible = len(selected)
    selected = selected.iloc[:max_rows].copy()
    out = selected[KEY + ["reference_surface_row", "surf", "t"]].reset_index(drop=True)
    out[["x", "y", "z"]] = o[:max_rows]
    out[["px", "py", "pz"]] = pv[:max_rows]
    out["length_mm"] = lengths[:max_rows]
    counts = dict(reference_gen_rows=len(frames["gen"]),
                  reference_gen_events=len(frames["gen"][KEY[:3]].drop_duplicates()),
                  reference_gen_domain="retained reference only; not all generated events",
                  primary_muons=len(prim), primary_events=len(prim[KEY[:3]].drop_duplicates()),
                  envelope_muons=int(len(mask)), invalid_state_muons=int((~valid).sum()),
                  outside_bbox_muons=int((valid & ~inside).sum()), zero_path_muons=int((~positive).sum()),
                  eligible_muons=eligible, selected_muons=len(out), bounded_out_muons=max(0, eligible - len(out)),
                  selected_events=len(out[KEY[:3]].drop_duplicates()))
    return out, counts


def occupancy(hits, by_material=True):
    """Exact-boundary signed sweep; retain invalid multiplicity and negative spans without clipping."""
    columns = ["lane", "material"] if by_material else ["lane"]
    if hits.empty:
        return pd.DataFrame(columns=columns + ["s0_mm", "s1_mm", "count"])
    starts = hits[columns].assign(s=hits.s0_mm, delta=hits.sign)
    ends = hits[columns].assign(s=hits.s1_mm, delta=-hits.sign)
    events = pd.concat([starts, ends], ignore_index=True).groupby(columns + ["s"], as_index=False).delta.sum()
    grouped = events.groupby(columns, sort=False)
    events["count"] = grouped.delta.cumsum()
    events["s1_mm"] = grouped.s.shift(-1)
    return events.loc[events.s1_mm > events.s, columns + ["s", "s1_mm", "count"]].rename(columns={"s": "s0_mm"})


def stored_hits(gdml, bbox, o, d, length):
    """Retain tube placement ordinals and calo group labels using the same stored-model clipping primitives."""
    names = []
    tubes = world.load_tubes(gdml, bbox=bbox, skip_names=steps.INTERIOR_SKIP, names=names)
    keep = [i for i, t in enumerate(tubes) if t[0] != "StrawGas"]
    labelled = [(f"tube:{i}|{tubes[i][0]}",) + tuple(tubes[i][1:]) for i in keep]
    shells = world.CoaxialShells(labelled, geometry.DETECTOR_ORIGIN_EXTRACTED)
    tube = materialpath.tube_hits(shells, o, d, length)
    labels = tube.material.str.split("|", n=1, expand=True).reindex(columns=[0, 1])
    tube["source"], tube["material"], tube["sign"], tube["component"] = labels[0], labels[1], 1, "tube"
    pcs, skipped = world.signed_subtree(gdml, steps.CALO_MOTHER, bbox=bbox, labels=True)
    pcs = [q for q in pcs if not q[1].endswith("|StrawGas")]
    rows = [tube]
    if pcs:
        packed = world.SignedPieceSet(pcs, cell=steps.CALO_CELL_MM)
        for bucket, (sign, ps) in enumerate(convexpath._sets(packed)):
            for lane, piece, enter, leave in convexpath._hits(ps, o, d, length):
                label = pd.Series(np.asarray(ps.mats, dtype=object)[ps.mat[piece]]).str.rsplit("|", n=1, expand=True)
                label = label.reindex(columns=[0, 1])
                rows.append(pd.DataFrame(dict(lane=lane, s0_mm=enter, s1_mm=leave, material=label[1],
                    source=label[0], sign=sign, component="calo", piece=piece, bucket=bucket)))
    catalog = [dict(source=f"tube:{i}", path=names[i], material=tubes[i][0]) for i in keep]
    return pd.concat(rows, ignore_index=True), dict(tube_catalog=catalog, tube_shells=len(keep),
        coaxial_shells=shells.n_coaxial, other_shells=len(shells.other), calo_signed_pieces=len(pcs),
        calo_skipped=[dict(key=list(k), count=v) for k, v in skipped.items()])


def summarize_occupancy(hits):
    """Keep both per-material and cross-material faults; tiny defects are evidence, not silently repaired."""
    material = occupancy(hits)
    total = occupancy(hits, by_material=False)
    faults = []
    for label, spans in (("material", material), ("total", total)):
        bad = spans[(spans["count"] < 0) | (spans["count"] > 1)].copy()
        bad["kind"] = label
        faults.append(bad)
    faults = pd.concat(faults, ignore_index=True)
    widths = faults.s1_mm - faults.s0_mm
    return material, faults, dict(fault_spans=len(faults), affected_lanes=int(faults.lane.nunique()),
        max_fault_length_mm=float(widths.max()) if len(widths) else 0.,
        sub_nanometre_fault_spans=int((widths < 1e-6).sum()),
        negative_spans=int((faults["count"] < 0).sum()))


def fault_witnesses(faults, hits, seeds):
    """Join bounded fault spans to contributing signed pieces, preserving identities and world positions."""
    faults = faults.copy()
    faults["fault_id"] = np.arange(len(faults))
    h = hits.rename(columns={"material": "hit_material", "s0_mm": "hit_s0_mm", "s1_mm": "hit_s1_mm"})
    joined = faults.merge(h, on="lane", suffixes=("", "_hit"))
    selected = ((joined.hit_s0_mm < joined.s1_mm) & (joined.hit_s1_mm > joined.s0_mm)
                & ((joined.kind == "total") | (joined.material == joined.hit_material))
                & ((joined.component == "combined") | (joined.component == joined.component_hit)))
    joined = joined.loc[selected].copy()
    lane = joined.lane.to_numpy(dtype=np.int64)
    origin = seeds[["x", "y", "z"]].to_numpy(float)
    momentum = seeds[["px", "py", "pz"]].to_numpy(float)
    direction = momentum / np.linalg.norm(momentum, axis=1)[:, None]
    midpoint = .5 * (joined.s0_mm.to_numpy(float) + joined.s1_mm.to_numpy(float))
    joined[["x_mm", "y_mm", "z_mm"]] = origin[lane] + direction[lane] * midpoint[:, None]
    return joined


def column_comparison(shells, hits, o, d, length):
    """Compare raw interval sums to legacy sampled columns, not a truth or overlap-admission test."""
    tube = hits[hits.component == "tube"].copy()
    tube["width"] = (tube.s1_mm - tube.s0_mm) * tube.sign
    exact = tube.groupby(["lane", "material"]).width.sum().unstack(fill_value=0.)
    summaries = {}
    for nsub in (8, 32, 128):
        sampled = shells.column(o, d, length, nsub=nsub)
        rows = []
        for material in sorted(set(sampled) | set(exact.columns)):
            analytic = exact.reindex(index=np.arange(len(o)), columns=[material], fill_value=0.).to_numpy().ravel()
            legacy = sampled.get(material, np.zeros(len(o)))
            diff = analytic - legacy
            rows.append(dict(material=material, analytic_total_mm=float(analytic.sum()),
                legacy_total_mm=float(legacy.sum()), max_abs_difference_mm=float(np.abs(diff).max()) if len(diff) else 0.,
                lanes_above_1e_6_mm=int((np.abs(diff) > 1e-6).sum())))
        summaries[str(nsub)] = rows
    return summaries


def omitted_candidates(result, o, d, length):
    """Conservative ray/AABB candidates for omitted solids; an AABB hit is not a material crossing."""
    rows, unknown = [], []
    parallel = d == 0
    safe = np.where(parallel, 1., d)
    for record in result["records"]:
        if record["classification"] != "unsupported":
            continue
        bounds = record["geometry_bounds_mm"]
        if bounds is None:
            unknown.append(record["placement_path"])
            continue
        lo, hi = np.asarray(bounds)
        a, b = (lo - o) / safe, (hi - o) / safe
        enter = np.maximum(0., np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=1))
        leave = np.minimum(length, np.where(parallel, np.inf, np.maximum(a, b)).min(axis=1))
        outside = (parallel & ((o < lo) | (o > hi))).any(axis=1)
        lanes = np.flatnonzero(~outside & (leave > enter))
        if len(lanes):
            rows.append(pd.DataFrame(dict(lane=lanes, s0_mm=enter[lanes], s1_mm=leave[lanes],
                source=record["placement_path"], material=record["material"], solid_tag=record["solid_tag"],
                scope=record["scope"])))
    columns = ["lane", "s0_mm", "s1_mm", "source", "material", "solid_tag", "scope"]
    candidates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    return candidates, dict(aabb_candidates=len(candidates), affected_lanes=int(candidates.lane.nunique()),
        unresolved_bounds=len(unknown), unresolved_placements=unknown,
        interpretation="Unsupported solids only: conservative AABB candidates, not proven solid intersections; "
                       "approximation discrepancies are separate; unknown bounds remain unresolved")


def census_summary(result):
    """Summarize possible omissions without interpreting unknown AABBs as absence."""
    records = pd.DataFrame(result["records"])
    columns = ["scope", "classification", "geometry_bbox_relation", "solid_tag", "material", "reason"]
    groups = records.groupby(columns).size().rename("placements").reset_index() if len(records) else pd.DataFrame()
    return dict(counts=result["counts"], placement_warnings=result["placement_warnings"],
                groups=groups.to_dict(orient="records"))


def run(reference, *, gdml, outdir, tracker_ref, core_ref, max_rows=256, length_mm=2000., max_tree_rows=2000000):
    """Write new audit evidence even on strict navigation refusal; never produce a transport history."""
    start = time.perf_counter()
    for ref in (tracker_ref, core_ref):
        if not re.fullmatch(r"[0-9a-f]{40}", ref):
            raise ValueError("full lowercase source commit hashes required")
    out = Path(outdir)
    if out.exists():
        raise FileExistsError(out)
    frames, trees = read_reference(reference, max_tree_rows)
    geom = geometry.from_gdml(str(gdml))
    lo, hi = crossings.panel_boxes(geom)
    bbox = (lo.min(0) - 300., hi.max(0) + 300.)
    seeds, counts = reference_seeds(frames, bbox, max_rows, length_mm)
    o, pv = seeds[["x", "y", "z"]].to_numpy(float), seeds[["px", "py", "pz"]].to_numpy(float)
    d = pv / np.linalg.norm(pv, axis=1)[:, None]
    length = seeds.length_mm.to_numpy(float)
    interior = steps.TrackerInterior(str(gdml), geom)
    hits, inventory = stored_hits(str(gdml), bbox, o, d, length)
    results, faults = {}, []
    for label in ("tube", "calo", "combined"):
        subset = hits if label == "combined" else hits[hits.component == label]
        _, bad, results[label] = summarize_occupancy(subset)
        bad["component"] = label
        faults.append(bad)
    strict = {}
    for name, action in (("tube", lambda: materialpath.tube_intervals(interior.shells.coax, o, d, length)),
                         ("calo", lambda: convexpath.convex_intervals(interior.shells.calo, o, d, length)
                          if interior.shells.calo is not None else None),
                         ("combined", lambda: interior_intervals(interior.shells, o, d, length))):
        try:
            table = action()
            strict[name] = dict(passed=True, rows=0 if table is None else len(table))
        except ValueError as error:
            strict[name] = dict(passed=False, error=str(error))
    from ..navigationcensus import census
    census_result = census(str(gdml), bbox)
    omitted, omitted_summary = omitted_candidates(census_result, o, d, length)
    bad = pd.concat(faults, ignore_index=True)
    identity = seeds[KEY].assign(lane=np.arange(len(seeds)))
    bad = bad.merge(identity, on="lane", how="left")
    witnesses = fault_witnesses(bad, hits, seeds)
    columns = column_comparison(interior.shells.coax, hits, o, d, length)
    manifest = dict(schema="trkfastsim.navigation-audit.v1", scope=SCOPE,
        tracker_ref=tracker_ref, core_ref=core_ref, refs_are_caller_supplied=True,
        sources_sha256=source_manifest(), reference=dict(path=str(Path(reference).resolve()), sha256=sha256(reference), trees=trees),
        gdml=dict(path=str(Path(gdml).resolve()), sha256=sha256(gdml)),
        settings=dict(max_rows=max_rows, max_tree_rows=max_tree_rows, requested_length_mm=length_mm,
                      bbox_mm=np.asarray(bbox).tolist(), seed="first tracker-envelope SurfaceStep midpoint per primary identity",
                      selection="stable full-identity order, finite nonzero momentum, seed inside bbox; length clipped at bbox exit",
                      units="mm, MeV/c, ns", frame="Mu2e world Cartesian", argv=sys.argv),
        counts=counts, inventory=inventory, occupancy=results, strict_navigation=strict,
        tube_column_comparison=columns, loader_census=census_summary(census_result),
        omitted_ray_candidates=omitted_summary,
        elapsed_seconds=time.perf_counter() - start, complete=True,
        admitted_for_transport=False, unresolved="loader census and requested-ray refusals require explicit resolution")
    out.mkdir(parents=True, exist_ok=False)
    seeds.to_csv(out / "seeds.csv", index=False)
    hits.to_csv(out / "raw_hits.csv", index=False)
    bad.to_csv(out / "faults.csv", index=False)
    witnesses.to_csv(out / "fault_witnesses.csv", index=False)
    omitted.merge(identity, on="lane", how="left").to_csv(out / "omitted_aabb_candidates.csv", index=False)
    (out / "loader_census.json").write_text(json.dumps(census_result, indent=2, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {key: manifest[key] for key in ("counts", "occupancy", "strict_navigation", "elapsed_seconds", "complete")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    for name in ("gdml", "outdir", "tracker-ref", "core-ref"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-tree-rows", type=int, default=2000000)
    parser.add_argument("--length-mm", type=float, default=2000.)
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2))


if __name__ == "__main__":
    main()
