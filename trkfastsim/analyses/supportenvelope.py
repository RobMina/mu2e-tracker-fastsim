"""Replay saved rays with a separately named quantified pair of torus occupancy envelopes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import omittedpath, supportenvelope as envelope, supportnavigation as strict
from . import navigationresolve as replay, supportnavigation as refusal_replay
from .navigationaudit import KEY
from .ledgercheck import sha256, source_manifest

TABLES = ("inner_intervals", "outer_intervals", "atoms", "material_bounds", "nonvacuum_bounds", "decisions", "refusals",
          "unknown_cells", "unknown_components", "parent_comparisons", "ownership_comparisons", "replacement_terms")
SCOPE = ("Paired inner/outer leaf-occupancy envelopes relative to the coalesced stored support model; "
         "explicit geometric uncertainty budgets, not exact crossings, energy/time bounds or transport admission.")


def read_control(directory, inputs, gdml, budgets):
    """Read unchanged refusal-first products and hash every file consumed by the envelope replay."""
    directory = Path(directory)
    files = {name: directory / name for name in ("summary.json", "omitted_cells.csv", "omitted_placements.json")}
    summary = json.loads(files["summary.json"].read_text())
    if summary.get("schema") != "trkfastsim.support-navigation-replay.v1" or not summary.get("complete"):
        raise ValueError("completed refusal-first support-navigation replay required")
    if summary.get("mode") != strict.MODE or summary.get("admitted_for_transport") is not False:
        raise ValueError("recognized non-transport refusal control required")
    if summary["gdml"]["sha256"] != sha256(gdml):
        raise ValueError("control GDML differs")
    if {k: v["sha256"] for k, v in inputs.items()} != {k: v["sha256"] for k, v in summary["inputs"].items()}:
        raise ValueError("control consumed different saved audit files")
    if not set(budgets) <= set(summary["settings"]["endpoint_budgets_mm"]):
        raise ValueError("every endpoint budget must have a saved refusal control")
    controls = {}
    for budget in budgets:
        name = f"candidate_{budget:.17g}mm"
        controls[budget] = {}
        for kind in ("intervals", "decisions", "refusals"):
            filename = f"{name}_{kind}.csv"
            files[filename] = directory / filename
            table = pd.read_csv(files[filename], float_precision="round_trip", nrows=1000001)
            if len(table) > 1000000:
                raise ValueError("saved control table exceeds explicit row bound")
            controls[budget][kind] = table
    cells = pd.read_csv(files["omitted_cells.csv"], float_precision="round_trip", nrows=1000001)
    if len(cells) > 1000000:
        raise ValueError("saved omission cell table exceeds explicit row bound")
    reports = json.loads(files["omitted_placements.json"].read_text())
    inputs = {k: dict(path=str(p.resolve()), sha256=sha256(p)) for k, p in files.items()}
    return cells, reports, summary, controls, inputs


def _check_control(result, previous):
    for kind, order in (("intervals", ["lane", "s0_mm"]), ("decisions", ["lane"]), ("refusals", ["lane", "reason"])):
        refusal_replay._equal_table(result["control"][kind], previous[kind], list(result["control"][kind]), order)


def _invariance(hits, lengths, cells, reports, records, baseline, settings):
    reverse = envelope.compose(hits.iloc[::-1], lengths, cells.iloc[::-1], reports[::-1], records[::-1], **settings)
    orders = dict(inner_intervals=["lane", "s0_mm"], outer_intervals=["lane", "s0_mm"], atoms=["lane", "s0_mm"],
        decisions=["lane"], refusals=["lane", "reason"], material_bounds=["lane", "material"],
        nonvacuum_bounds=["lane"], unknown_components=["lane", "s0_mm"])
    for key, order in orders.items():
        refusal_replay._equal_table(baseline[key], reverse[key], list(baseline[key]), order)
    return dict(reversed_evidence_identical=True, probe_batch_domain="full saved batch; split-guard invariance not claimed")


def _perturbations(hits, lengths, cells, reports, records, baseline, settings):
    rows = []
    for target, name in ((-np.inf, "minus-one-ulp"), (np.inf, "plus-one-ulp")):
        perturbed = hits.copy(deep=True)
        for key in ("s0_mm", "s1_mm"):
            original = hits[key].to_numpy(float)
            anchored = (original == 0.) | (original == lengths[hits.lane.to_numpy(np.int64)])
            perturbed[key] = np.where(anchored, original, np.nextafter(original, target))
        try:
            result = envelope.compose(perturbed, lengths, cells, reports, records, **settings)
            rows.append(dict(perturbation=name, completed=True, bounded_pair_rays=result["summary"]["bounded_pair_rays"],
                same_lane_decisions=bool(np.array_equal(result["decisions"].bounded_pair_admitted, baseline["decisions"].bounded_pair_admitted))))
        except ValueError as error:
            rows.append(dict(perturbation=name, completed=False, reason=str(error)))
    return rows


def _compare_bounds(fine, coarse):
    """Marginal bound consistency only; refusals and changed domains remain explicit."""
    a = fine["material_bounds"].rename(columns={"lower_mm": "fine_lower_mm", "upper_mm": "fine_upper_mm"})
    b = coarse["material_bounds"].rename(columns={"lower_mm": "coarse_lower_mm", "upper_mm": "coarse_upper_mm"})
    same = fine["decisions"].bounded_pair_admitted.to_numpy() & coarse["decisions"].bounded_pair_admitted.to_numpy()
    pairs = a[a.lane.isin(np.flatnonzero(same))].merge(b[b.lane.isin(np.flatnonzero(same))],
                                                    on=["lane", "material"], how="outer").fillna(0.)
    pairs["bounds_overlap"] = np.maximum(pairs.fine_lower_mm, pairs.coarse_lower_mm) <= np.minimum(pairs.fine_upper_mm, pairs.coarse_upper_mm)
    pairs["fine_nested_in_coarse"] = (pairs.fine_lower_mm >= pairs.coarse_lower_mm) & (pairs.fine_upper_mm <= pairs.coarse_upper_mm)
    return pairs, dict(common_pair_rays=int(same.sum()), compared_material_rows=len(pairs),
        all_bounds_overlap=bool(pairs.bounds_overlap.all()), fine_nested_rows=int(pairs.fine_nested_in_coarse.sum()),
        note="Arithmetic enclosure padding may prevent literal nesting for unchanged exact columns; no physics monotonicity claim")


def run(audit_dir, *, candidate_dir, gdml, outdir, max_component_mm, max_ray_unknown_mm,
        budgets=(0., 1e-10, 1e-9), min_cell_mm=1e-7, compare_cell_mm=1e-6,
        max_rows=512, max_hit_rows=1000000, max_cells=1000000, max_pairs=2000000):
    """No overwrites, truncation, new reference seeds or loss sampling; completion is written last."""
    start = time.perf_counter()
    output = Path(outdir)
    if output.exists():
        raise FileExistsError(output)
    if not budgets or len(set(budgets)) != len(budgets) or not np.isfinite(budgets).all() or np.any(np.asarray(budgets) < 0):
        raise ValueError("unique finite nonnegative endpoint budgets required")
    max_component_mm = envelope._budget(max_component_mm, "max_component_mm")
    max_ray_unknown_mm = envelope._budget(max_ray_unknown_mm, "max_ray_unknown_mm")
    if not np.isfinite(compare_cell_mm) or compare_cell_mm < min_cell_mm:
        raise ValueError("comparison width must be finite and no finer")
    seeds, hits, census, prior, inputs = replay.read_saved(audit_dir, gdml, max_rows, max_hit_rows)
    saved_cells, saved_reports, previous, controls, control_inputs = read_control(candidate_dir, inputs, gdml, budgets)
    if min_cell_mm != previous["settings"]["min_cell_mm"] or max_cells != previous["settings"]["max_cells"]:
        raise ValueError("primary probe settings must reproduce refusal control")
    origin = seeds[["x", "y", "z"]].to_numpy(float)
    pv = seeds[["px", "py", "pz"]].to_numpy(float)
    norm = np.linalg.norm(pv, axis=1)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("invalid saved ray momentum")
    direction, lengths = pv / norm[:, None], seeds.length_mm.to_numpy(float)
    identity = seeds[KEY].assign(lane=np.arange(len(seeds)))
    if set(KEY) & set(hits.columns) and not set(KEY) <= set(hits.columns):
        raise ValueError("partial raw-hit identity schema")
    if set(KEY) <= set(hits.columns):
        matched = hits.merge(identity, on="lane", suffixes=("", "_seed"), validate="many_to_one")
        if len(matched) != len(hits) or not np.array_equal(matched[KEY].to_numpy(), matched[[k+"_seed" for k in KEY]].to_numpy()):
            raise ValueError("raw-hit identities disagree with seed lanes")
    from ..navigationcensus import census as make_census
    if make_census(str(gdml), census["bbox_mm"]) != census:
        raise ValueError("current census differs from saved audit")
    current = strict.navigate(gdml, census, origin, direction, lengths, hits, endpoint_budget_mm=budgets[0],
                              min_cell_mm=min_cell_mm, max_cells=max_cells, max_pairs=max_pairs)
    cells, reports = current["omitted_cells"], current["placement_reports"]
    refusal_replay._equal_table(cells, saved_cells, list(cells), ["source", "lane", "s0_mm", "s1_mm", "state"])
    if reports != saved_reports:
        raise ValueError("omitted probe reports differ from saved refusal control")
    coarse_cells, coarse_reports = omittedpath.probe(gdml, census["records"], origin, direction, lengths,
                                                     min_cell_mm=compare_cell_mm, max_cells=max_cells)
    products, summaries = {}, []
    for budget in budgets:
        settings = dict(endpoint_budget_mm=budget, min_cell_mm=min_cell_mm, max_component_mm=max_component_mm,
                        max_ray_unknown_mm=max_ray_unknown_mm, max_pairs=max_pairs)
        fine = envelope.compose(hits, lengths, cells, reports, census["records"], **settings)
        _check_control(fine, controls[budget])
        coarse = envelope.compose(hits, lengths, coarse_cells, coarse_reports, census["records"],
                                  **(settings | dict(min_cell_mm=compare_cell_mm)))
        comparison, comparison_summary = _compare_bounds(fine, coarse)
        zero = envelope.compose(hits, lengths, cells, reports, census["records"],
                                **(settings | dict(max_component_mm=0., max_ray_unknown_mm=0.)))
        if not np.array_equal(zero["decisions"].bounded_pair_admitted, fine["control"]["decisions"].candidate_resolved):
            raise ValueError("zero uncertainty budgets disagree with strict refusal control")
        summary = dict(fine["summary"], saved_refusal_control_identical=True,
            zero_unknown_budget_control_identical=True, invariance=_invariance(hits, lengths, cells, reports, census["records"], fine, settings),
            stored_endpoint_perturbations=_perturbations(hits, lengths, cells, reports, census["records"], fine, settings),
            coarse_summary=coarse["summary"], resolution_comparison=comparison_summary)
        name = f"envelope_{budget:.17g}mm"
        products[name] = {key: fine[key] for key in TABLES}
        products[name]["resolution_comparison"] = comparison
        products[name + "_coarse"] = {key: coarse[key] for key in TABLES}
        summaries.append(summary)
    summary = dict(schema="trkfastsim.support-envelope-replay.v1", mode=envelope.MODE, scope=SCOPE,
        inputs=inputs, control_inputs=control_inputs, gdml=dict(path=str(Path(gdml).resolve()), sha256=sha256(gdml)),
        sources_sha256=source_manifest(), tracker_ref=prior["tracker_ref"], core_ref=prior["core_ref"],
        seed_rows=len(seeds), seed_events=len(seeds[KEY[:3]].drop_duplicates()), seed_domain=prior["counts"], raw_hit_rows=len(hits),
        settings=dict(endpoint_budgets_mm=list(budgets), min_cell_mm=min_cell_mm, compare_cell_mm=compare_cell_mm,
            max_component_mm=max_component_mm, max_ray_unknown_mm=max_ray_unknown_mm,
            max_rows=max_rows, max_hit_rows=max_hit_rows, max_cells=max_cells, max_pairs=max_pairs),
        candidates=summaries, omitted=replay.summarize_omissions(cells, reports),
        cell_resolution_comparison=replay.compare_resolution(reports, coarse_reports),
        complete=True, admitted_for_transport=False, elapsed_seconds=time.perf_counter()-start)
    output.mkdir(parents=True, exist_ok=False)
    seeds.to_csv(output / "seeds.csv", index=False)
    for name, tables in products.items():
        for kind, table in tables.items():
            table.drop(columns=KEY, errors="ignore").merge(identity, on="lane", how="left", validate="many_to_one").to_csv(
                output / f"{name}_{kind}.csv", index=False)
    for tag, spans, placement_reports in (("fine", cells, reports), ("coarse", coarse_cells, coarse_reports)):
        spans.merge(identity, on="lane", how="left", validate="many_to_one").to_csv(output / f"{tag}_omitted_cells.csv", index=False)
        (output / f"{tag}_omitted_placements.json").write_text(json.dumps(placement_reports, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return {key: summary[key] for key in ("mode", "seed_rows", "candidates", "omitted", "elapsed_seconds", "complete")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-component-mm", type=float, required=True)
    parser.add_argument("--max-ray-unknown-mm", type=float, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0., 1e-10, 1e-9])
    parser.add_argument("--min-cell-mm", type=float, default=1e-7)
    parser.add_argument("--compare-cell-mm", type=float, default=1e-6)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--max-hit-rows", type=int, default=1000000)
    parser.add_argument("--max-cells", type=int, default=1000000)
    parser.add_argument("--max-pairs", type=int, default=2000000)
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2))


if __name__ == "__main__":
    main()
