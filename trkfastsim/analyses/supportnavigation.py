"""Replay saved reference rays through the separate refusal-first support-navigation candidate."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import supportnavigation as candidate
from .. import omittedpath
from . import navigationresolve as replay
from .navigationaudit import KEY
from .ledgercheck import sha256, source_manifest

SCOPE = ("Bounded stored-model support navigation with actual omitted-leaf probes and whole-ray refusal; "
         "no energy/time transport, straw history, independent GDML/G4 truth or production promotion.")


def read_reviewed(directory, inputs, gdml):
    """Bind this replay to a completed resolution of exactly the same saved audit and geometry."""
    directory = Path(directory)
    files = {name: directory / name for name in ("summary.json", "omitted_solid_intervals.csv", "omitted_placements.json")}
    summary = json.loads(files["summary.json"].read_text())
    if summary.get("schema") != "trkfastsim.navigation-resolution.v1" or not summary.get("complete"):
        raise ValueError("completed navigation resolution required")
    if summary["gdml"]["sha256"] != sha256(gdml):
        raise ValueError("reviewed resolution GDML differs")
    if {k: v["sha256"] for k, v in inputs.items()} != {k: v["sha256"] for k, v in summary["inputs"].items()}:
        raise ValueError("reviewed resolution consumed different saved audit files")
    cells = pd.read_csv(files["omitted_solid_intervals.csv"], float_precision="round_trip", nrows=1000001)
    if len(cells) > 1000000:
        raise ValueError("reviewed omission cells exceed row bound")
    reports = json.loads(files["omitted_placements.json"].read_text())
    return cells, reports, summary, {k: dict(path=str(p.resolve()), sha256=sha256(p)) for k, p in files.items()}


def _equal_table(left, right, columns, order):
    left = left[columns].sort_values(order, kind="stable").reset_index(drop=True)
    right = right[columns].sort_values(order, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)


def _invariance(hits, lengths, cells, reports, records, result, budget, max_pairs):
    reverse = candidate.compose(hits.iloc[::-1], lengths, cells.iloc[::-1], reports[::-1], records[::-1],
                                endpoint_budget_mm=budget, max_pairs=max_pairs)
    for key, order in (("intervals", ["lane", "s0_mm"]), ("decisions", ["lane"]), ("refusals", ["lane", "reason"])):
        _equal_table(result[key], reverse[key], list(result[key].columns), order)
    return dict(reversed_source_and_cell_order_identical=True,
                note="Probe guard uses the full saved batch domain; lane-split guard invariance is not assumed")


def _endpoint_perturbation(hits, lengths, cells, reports, records, baseline, budget, max_pairs):
    """Stress saved stored endpoints only; never reinterpret perturbed endpoints as new geometric rays."""
    rows = []
    for target, name in ((-np.inf, "minus-one-ulp"), (np.inf, "plus-one-ulp")):
        perturbed = hits.copy(deep=True)
        for key in ("s0_mm", "s1_mm"):
            original = hits[key].to_numpy(float)
            changed = np.nextafter(original, target)
            anchors = (original == 0.) | (original == lengths[hits.lane.to_numpy(dtype=np.int64)])
            perturbed[key] = np.where(anchors, original, changed)
        try:
            result = candidate.compose(perturbed, lengths, cells, reports, records,
                                       endpoint_budget_mm=budget, max_pairs=max_pairs)
            rows.append(dict(perturbation=name, completed=True,
                resolved_rays=result["summary"]["resolved_rays"], refused_rays=result["summary"]["refused_rays"],
                same_lane_decisions=bool(np.array_equal(result["decisions"].candidate_resolved,
                                                       baseline["decisions"].candidate_resolved)),
                unresolved_cells=result["summary"]["unresolved_cells"],
                actual_max_shift_mm=result["summary"]["endpoint_policy"]["actual_max_shift_mm"]))
        except ValueError as error:
            rows.append(dict(perturbation=name, completed=False, reason=str(error)))
    return rows


def run(audit_dir, *, resolution_dir, gdml, outdir, budgets=(0., 1e-10, 1e-9), min_cell_mm=1e-7,
        compare_cell_mm=1e-6, max_rows=512, max_hit_rows=1000000, max_cells=1000000, max_pairs=2000000):
    """Write new evidence only, completion last; refused lanes have no complete interval rows."""
    start = time.perf_counter()
    output = Path(outdir)
    if output.exists():
        raise FileExistsError(output)
    if not budgets or len(set(budgets)) != len(budgets) or not np.isfinite(budgets).all() or np.any(np.asarray(budgets) < 0):
        raise ValueError("unique nonempty finite nonnegative budgets required")
    if not np.isfinite(compare_cell_mm) or compare_cell_mm < min_cell_mm:
        raise ValueError("comparison cell width must be finite and no finer")
    seeds, hits, census, prior, inputs = replay.read_saved(audit_dir, gdml, max_rows, max_hit_rows)
    reviewed_cells, reviewed_reports, reviewed, resolution_inputs = read_reviewed(resolution_dir, inputs, gdml)
    if min_cell_mm != reviewed["settings"]["min_cell_mm"] or max_cells != reviewed["settings"]["max_cells"]:
        raise ValueError("primary probe settings must reproduce reviewed resolution")
    origin = seeds[["x", "y", "z"]].to_numpy(float)
    momentum = seeds[["px", "py", "pz"]].to_numpy(float)
    norm = np.linalg.norm(momentum, axis=1)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("invalid saved momentum")
    direction, lengths = momentum / norm[:, None], seeds.length_mm.to_numpy(float)
    identity = seeds[KEY].assign(lane=np.arange(len(seeds)))
    if set(KEY) & set(hits.columns) and not set(KEY) <= set(hits.columns):
        raise ValueError("partial raw-hit identity schema")
    if set(KEY) <= set(hits.columns):
        matched = hits.merge(identity, on="lane", suffixes=("", "_seed"), validate="many_to_one")
        if not np.array_equal(matched[KEY].to_numpy(), matched[[k+"_seed" for k in KEY]].to_numpy()):
            raise ValueError("raw-hit identities disagree with seed lanes")
    from ..navigationcensus import census as make_census
    fresh_census = make_census(str(gdml), census["bbox_mm"])
    if fresh_census != census:
        raise ValueError("saved census differs from current GDML/loader census; rerun audit separately")
    result = candidate.navigate(gdml, census, origin, direction, lengths, hits,
        endpoint_budget_mm=budgets[0], min_cell_mm=min_cell_mm, max_cells=max_cells, max_pairs=max_pairs)
    cells, reports = result["omitted_cells"], result["placement_reports"]
    _equal_table(cells, reviewed_cells, list(cells.columns), ["source", "lane", "s0_mm", "s1_mm", "state"])
    if reports != reviewed_reports:
        raise ValueError("current omitted-solid reports differ from reviewed replay")
    _, coarse_reports = omittedpath.probe(gdml, census["records"], origin, direction, lengths,
                                          min_cell_mm=compare_cell_mm, max_cells=max_cells)
    comparison = replay.compare_resolution(reports, coarse_reports)
    products, summaries = {}, []
    for budget in budgets:
        if budget != budgets[0]:
            result = candidate.compose(hits, lengths, cells, reports, census["records"],
                                       endpoint_budget_mm=budget, max_pairs=max_pairs)
        name = f"candidate_{budget:.17g}mm"
        summary = dict(result["summary"],
            invariance=_invariance(hits, lengths, cells, reports, census["records"], result, budget, max_pairs),
            stored_endpoint_perturbations=_endpoint_perturbation(hits, lengths, cells, reports, census["records"], result, budget, max_pairs))
        summaries.append(summary)
        products[name] = {key: result[key] for key in ("intervals", "decisions", "refusals", "adjusted_hits",
                                                       "endpoints", "replacement_terms", "parent_comparisons", "ownership_comparisons")}
    summary = dict(schema="trkfastsim.support-navigation-replay.v1", mode=candidate.MODE, scope=SCOPE,
        inputs=inputs, reviewed_resolution_inputs=resolution_inputs, gdml=dict(path=str(Path(gdml).resolve()), sha256=sha256(gdml)),
        sources_sha256=source_manifest(), tracker_ref=prior["tracker_ref"], core_ref=prior["core_ref"],
        seed_rows=len(seeds), seed_events=len(seeds[KEY[:3]].drop_duplicates()), raw_hit_rows=len(hits), seed_domain=prior["counts"],
        settings=dict(endpoint_budgets_mm=list(budgets), unresolved_policy=candidate.POLICY, min_cell_mm=min_cell_mm,
            compare_cell_mm=compare_cell_mm, max_rows=max_rows, max_hit_rows=max_hit_rows, max_cells=max_cells, max_pairs=max_pairs),
        reviewed_cells_and_reports_identical=True, regenerated_census_identical=True,
        candidates=summaries, omitted=replay.summarize_omissions(cells, reports), cell_resolution_comparison=comparison,
        complete=True, admitted_for_transport=False, elapsed_seconds=time.perf_counter()-start)
    output.mkdir(parents=True, exist_ok=False)
    seeds.to_csv(output / "seeds.csv", index=False)
    for name, tables in products.items():
        for kind, table in tables.items():
            table = table.drop(columns=KEY, errors="ignore")
            table.merge(identity, on="lane", how="left", validate="many_to_one").to_csv(output / f"{name}_{kind}.csv", index=False)
    cells.merge(identity, on="lane", how="left", validate="many_to_one").to_csv(output / "omitted_cells.csv", index=False)
    (output / "omitted_placements.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return {key: summary[key] for key in ("mode", "seed_rows", "candidates", "omitted", "elapsed_seconds", "complete")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir")
    parser.add_argument("--resolution-dir", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
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
