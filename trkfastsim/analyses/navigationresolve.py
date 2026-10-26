"""Replay a saved bounded navigation audit with explicit endpoint and omitted-solid diagnostics."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import boundarypolicy, omittedpath
from . import navigationaudit
from .ledgercheck import sha256, source_manifest

SCOPE = ("Opt-in numerical endpoint and omitted-solid diagnostics on saved rays, not physical-history admission, "
         "energy/time transport, independent G4 truth or default promotion.")


def read_saved(directory, gdml, max_rows=512, max_hit_rows=1000000):
    """Read decimal CSV with round-trip float conversion, retain exact identities and immutable input hashes."""
    directory = Path(directory)
    if max_rows <= 0 or max_hit_rows <= 0:
        raise ValueError("positive saved-table bounds required")
    files = {name: directory / name for name in ("summary.json", "seeds.csv", "raw_hits.csv", "loader_census.json")}
    prior = json.loads(files["summary.json"].read_text())
    if prior.get("schema") != "trkfastsim.navigation-audit.v1" or not prior.get("complete"):
        raise ValueError("a completed bounded navigation audit is required")
    if sha256(gdml) != prior["gdml"]["sha256"]:
        raise ValueError("GDML hash differs from the saved audit")
    seeds = pd.read_csv(files["seeds.csv"], nrows=max_rows+1, float_precision="round_trip")
    hits = pd.read_csv(files["raw_hits.csv"], nrows=max_hit_rows+1, float_precision="round_trip")
    if len(seeds) > max_rows or len(hits) > max_hit_rows:
        raise ValueError("saved table exceeds explicit row bound; no truncation")
    key = seeds[navigationaudit.KEY].to_numpy(float)
    if not np.isfinite(key).all() or (key != np.floor(key)).any() or seeds.duplicated(navigationaudit.KEY).any():
        raise ValueError("saved particle identities must be unique finite integers")
    if len(seeds) != prior["counts"]["selected_muons"]:
        raise ValueError("saved seed count disagrees with its manifest")
    census = json.loads(files["loader_census.json"].read_text())
    inputs = {name: dict(path=str(path.resolve()), sha256=sha256(path)) for name, path in files.items()}
    return seeds, hits, census, prior, inputs


def boundary_invariance(hits, adjusted, lengths, budget):
    """Replay reversed source rows and two lane batches without changing ray coordinates."""
    columns = ["s0_mm", "s1_mm"]
    reversed_hits, _ = boundarypolicy.coalesce(hits.iloc[::-1], lengths, max_shift_mm=budget)
    permuted_equal = np.array_equal(reversed_hits.iloc[::-1][columns].to_numpy(), adjusted[columns].to_numpy())
    batched_equal = True
    for first, last in ((0, len(lengths)//2), (len(lengths)//2, len(lengths))):
        mask = (hits.lane >= first) & (hits.lane < last)
        batch = hits.loc[mask].copy()
        batch["lane"] -= first
        resolved, _ = boundarypolicy.coalesce(batch, lengths[first:last], max_shift_mm=budget)
        batched_equal &= np.array_equal(resolved[columns].to_numpy(), adjusted.loc[mask, columns].to_numpy())
    if not permuted_equal or not batched_equal:
        raise ValueError("endpoint policy is not invariant under source permutation / lane batching")
    return dict(reversed_source_order_identical=permuted_equal, two_lane_batches_identical=batched_equal)


def replay_boundaries(hits, lengths, budgets):
    """Evaluate distinct explicit absolute-mm budgets, retaining refusals and raw evidence separately."""
    if len(set(budgets)) != len(budgets) or not budgets:
        raise ValueError("unique nonempty budget list required")
    products, summary = {}, []
    for budget in budgets:
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("finite nonnegative endpoint budget required")
        name = f"boundary_{budget:.17g}mm"
        row = dict(budget_mm=budget)
        try:
            adjusted, evidence = boundarypolicy.coalesce(hits, lengths, max_shift_mm=budget)
        except ValueError as error:
            summary.append(dict(row, coalesced=False, partition_passed=False, reason=str(error)))
            continue
        products[name] = dict(adjusted_hits=adjusted, endpoints=evidence["endpoints"])
        _, _, occupancy = navigationaudit.summarize_occupancy(adjusted)
        row.update(coalesced=True, evidence=evidence["summary"], occupancy=occupancy,
                   invariance=boundary_invariance(hits, adjusted, np.asarray(lengths), budget))
        try:
            spans = boundarypolicy.partition(adjusted, lengths)
            products[name]["intervals"] = spans
            row.update(partition_passed=True, interval_rows=len(spans))
        except ValueError as error:
            row.update(partition_passed=False, reason=str(error))
        summary.append(row)
    return products, summary


def summarize_omissions(spans, reports):
    """Report geometric lower/upper lengths separately from parent-material replacement, not energy loss."""
    groups = []
    verified = ((spans.state == "inside") & spans.ancestors_verified & spans.leaf_verified
                if len(spans) else pd.Series(dtype=bool))
    for keys, rows in spans.groupby(["scope", "material", "replaces_material"]):
        known = (rows.state == "inside") & rows.ancestors_verified & rows.leaf_verified
        groups.append(dict(zip(["scope", "material", "replaces_material"], keys)) | dict(
            lower_mm=omittedpath.length_bound(rows[known], upper=False),
            upper_mm=omittedpath.length_bound(rows, upper=True), lanes=int(rows.lane.nunique())))
    return dict(placements=len(reports), unevaluated_placements=sum(not r.get("resolved", False) for r in reports),
        verified_hit_lanes=int(spans.loc[verified, "lane"].nunique()) if len(spans) else 0,
        possible_hit_lanes=int(spans.lane.nunique()), by_replacement=groups,
        note="Per-placement sums, not hierarchy-corrected aggregate columns; upper bounds include unresolved brackets")


def replacement_check(spans, stored, lengths, max_pairs=2000000):
    """Compare certified missing cells to stored occupancy; test signed parent replacement without transport."""
    known = spans[(spans.state == "inside") & spans.ancestors_verified & spans.leaf_verified].copy()
    if known.empty:
        return pd.DataFrame(), dict(known_cells=0, replacement_partition_passed=None)
    sizes = known.groupby("lane").size().to_frame("n").join(stored.groupby("lane").size().rename("m")).fillna(0)
    if (sizes.n * sizes.m).sum() > max_pairs:
        raise ValueError("replacement cross-product exceeds explicit pair bound")
    old = stored.rename(columns={"s0_mm": "old_s0_mm", "s1_mm": "old_s1_mm", "material": "old_material"})
    pairs = known.merge(old, on="lane")
    pairs["s0_mm"] = np.maximum(pairs.s0_mm, pairs.old_s0_mm)
    pairs["s1_mm"] = np.minimum(pairs.s1_mm, pairs.old_s1_mm)
    pairs = pairs[pairs.s1_mm > pairs.s0_mm].copy()
    pairs["expected_parent"] = pairs.old_material == pairs.replaces_material
    additions = known[["lane", "s0_mm", "s1_mm", "material"]].assign(sign=1)
    subtraction = known[known.replaces_material != "vacuum"][["lane", "s0_mm", "s1_mm", "replaces_material"]].rename(
        columns={"replaces_material": "material"}).assign(sign=-1)
    base = stored[stored.material != "vacuum"].assign(sign=1)
    trial = pd.concat([base, additions, subtraction], ignore_index=True)
    report = dict(known_cells=len(known), comparisons=len(pairs),
        unexpected_parent_cells=int((~pairs.expected_parent).sum()),
        unexpected_parent_length_mm=omittedpath.length_bound(pairs[~pairs.expected_parent], upper=True),
        claim="Certified inner cells only; unresolved torus boundaries remain excluded from this partial replacement trial")
    try:
        result = boundarypolicy.partition(trial, lengths)
        report.update(replacement_partition_passed=True, partial_interval_rows=len(result))
    except ValueError as error:
        report.update(replacement_partition_passed=False, reason=str(error))
    return pairs, report


def compare_resolution(reference_reports, coarse_reports):
    """Compare conservative per-placement sums at two cell resolutions on exactly the same rays."""
    first = {r["source"]: r for r in reference_reports}
    rows = []
    for coarser in coarse_reports:
        finer = first[coarser["source"]]
        if not finer["resolved"] or not coarser["resolved"]:
            rows.append(dict(source=coarser["source"], compared=False))
            continue
        low, high = finer["solid_lower_mm"], finer["solid_upper_mm"]
        clow, chigh = coarser["solid_lower_mm"], coarser["solid_upper_mm"]
        rows.append(dict(source=coarser["source"], compared=True, fine_lower_mm=low, fine_upper_mm=high,
            coarse_lower_mm=clow, coarse_upper_mm=chigh, bounds_overlap=max(low, clow) <= min(high, chigh),
            refined_width_no_larger=high-low <= chigh-clow,
            same_hit_lanes=finer["hit_lanes"] == coarser["hit_lanes"]))
    return rows


def run(audit_dir, *, gdml, outdir, budgets=(0., 1e-10, 1e-9), max_rows=512, max_hit_rows=1000000,
        min_cell_mm=1e-7, max_cells=1000000, skip_omitted=False, compare_cell_mm=None):
    """Preserve raw evidence and unique outputs; complete means diagnostics finished, never transport admitted."""
    start = time.perf_counter()
    output = Path(outdir)
    if output.exists():
        raise FileExistsError(output)
    seeds, hits, census, prior, inputs = read_saved(audit_dir, gdml, max_rows, max_hit_rows)
    o, pv = seeds[["x", "y", "z"]].to_numpy(float), seeds[["px", "py", "pz"]].to_numpy(float)
    momentum = np.linalg.norm(pv, axis=1)
    if not np.isfinite(o).all() or not np.isfinite(momentum).all() or (momentum <= 0).any():
        raise ValueError("invalid saved ray state")
    d, lengths = pv / momentum[:, None], seeds.length_mm.to_numpy(float)
    products, summaries = replay_boundaries(hits, lengths, budgets)
    spans, reports = None, None
    if not skip_omitted:
        spans, reports = omittedpath.probe(gdml, census["records"], o, d, lengths,
                                         min_cell_mm=min_cell_mm, max_cells=max_cells)
    resolution = None
    if compare_cell_mm is not None:
        if skip_omitted or not np.isfinite(compare_cell_mm) or compare_cell_mm < min_cell_mm:
            raise ValueError("comparison cell width must be finite, coarser, and omitted probes enabled")
        _, coarse_reports = omittedpath.probe(gdml, census["records"], o, d, lengths,
                                             min_cell_mm=compare_cell_mm, max_cells=max_cells)
        resolution = compare_resolution(reports, coarse_reports)
    replacements, replacement_summaries = {}, {}
    if spans is not None:
        for name, tables in products.items():
            if "intervals" in tables:
                replacements[name], replacement_summaries[name] = replacement_check(spans, tables["intervals"], lengths)
    summary = dict(schema="trkfastsim.navigation-resolution.v1", scope=SCOPE, inputs=inputs,
        gdml=dict(path=str(Path(gdml).resolve()), sha256=sha256(gdml)), sources_sha256=source_manifest(),
        tracker_ref=prior["tracker_ref"], core_ref=prior["core_ref"], seed_rows=len(seeds),
        seed_events=len(seeds[navigationaudit.KEY[:3]].drop_duplicates()), raw_hit_rows=len(hits),
        seed_domain=prior["counts"], settings=dict(budgets_mm=list(budgets), max_rows=max_rows,
            max_hit_rows=max_hit_rows, min_cell_mm=min_cell_mm, compare_cell_mm=compare_cell_mm,
            max_cells=max_cells, omitted_enabled=not skip_omitted),
        boundaries=summaries, omitted=None if spans is None else summarize_omissions(spans, reports),
        partial_parent_replacement=replacement_summaries, cell_resolution_comparison=resolution,
        complete=True, admitted_for_transport=False, elapsed_seconds=time.perf_counter()-start)
    output.mkdir(parents=True, exist_ok=False)
    identity = seeds[navigationaudit.KEY].assign(lane=np.arange(len(seeds)))
    for name, tables in products.items():
        for kind, table in tables.items():
            table.merge(identity, on="lane", how="left").to_csv(output / f"{name}_{kind}.csv", index=False)
    for name, table in replacements.items():
        if len(table):
            table.merge(identity, on="lane", how="left").to_csv(output / f"{name}_parent_replacement.csv", index=False)
    if spans is not None:
        spans.merge(identity, on="lane", how="left").to_csv(output / "omitted_solid_intervals.csv", index=False)
        (output / "omitted_placements.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return {key: summary[key] for key in ("seed_rows", "boundaries", "omitted", "elapsed_seconds", "complete")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0., 1e-10, 1e-9])
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--max-hit-rows", type=int, default=1000000)
    parser.add_argument("--min-cell-mm", type=float, default=1e-7)
    parser.add_argument("--max-cells", type=int, default=1000000)
    parser.add_argument("--compare-cell-mm", type=float)
    parser.add_argument("--skip-omitted", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2))


if __name__ == "__main__":
    main()
