"""Nested source-event bank replicas on saved primaries; step-level uncertainty only, no transport/reco."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts import secondary_replay as replay
from scripts.transport_capture_audit import digest
from trkfastsim import bankreplica, geometry, library, steps, tables


def metrics(products):
    events = pd.MultiIndex.from_frame(products["gen"][tables.KEY].drop_duplicates())
    sgs = products["sgsteps"]
    secondary = sgs[sgs.sp >= steps.SECONDARY_KEY0]
    group = secondary.groupby(tables.KEY)
    counts = group.size().reindex(events, fill_value=0).to_numpy(float)
    deposit = group.eion.sum().reindex(events, fill_value=0).to_numpy(float)
    passing = tables.trk_filter(sgs)
    flags = pd.MultiIndex.from_frame(passing[passing.pass_trk].reset_index()[tables.KEY])
    selection = events.isin(flags).astype(float)
    result = {}
    for name, values in (("secondary_steps_per_generated_event", counts),
                         ("secondary_eion_MeV_per_generated_event", deposit),
                         ("any_particle_tracker_term_fraction", selection)):
        result[name] = dict(mean=float(values.mean()), event_cluster_mean_se=float(values.std(ddof=1) / np.sqrt(len(values))),
                            events=len(values))
    return result


def variance_components(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("nested bank/draw matrix requires at least two finite replicas per level")
    repeats = values.shape[1]
    within = float(np.var(values, axis=1, ddof=1).mean())
    between_means = float(np.var(values.mean(axis=1), ddof=1))
    donor_raw = between_means - within / repeats
    bank_means = values.mean(axis=1)
    bank_variances = np.var(values, axis=1, ddof=1)
    nbank = len(values)
    variance_se, interval = None, None
    if nbank >= 3:
        remaining = nbank - 1
        leave_mean = (bank_means.sum() - bank_means) / remaining
        leave_second = ((np.square(bank_means).sum() - np.square(bank_means)) - remaining * leave_mean ** 2) / (remaining - 1)
        leave_within = (bank_variances.sum() - bank_variances) / remaining
        leave_variance = leave_second - leave_within / repeats
        variance_se = float(np.sqrt((nbank - 1) / nbank * np.square(leave_variance - leave_variance.mean()).sum()))
        interval = [float(donor_raw - 1.959963984540054 * variance_se), float(donor_raw + 1.959963984540054 * variance_se)]
    return dict(bank_replicas=len(values), donor_variance_jackknife_se=variance_se,
                donor_variance_normal95_raw=interval, draws_per_bank=repeats, within_bank_draw_variance=within,
                positive_point_estimate=bool(donor_raw > 0),
                positive_at_descriptive_normal95=bool(interval is not None and interval[0] > 0),
                interpretation="unresolved variance component; zero clipping is not evidence of zero uncertainty" if donor_raw <= 0 else
                               "positive pilot point estimate; bank-replica precision must be assessed, not established nonzero variance",
                variance_of_bank_draw_means=between_means, donor_variance_raw=donor_raw,
                donor_variance_nonnegative=max(donor_raw, 0.),
                donor_sigma_nonnegative=float(np.sqrt(max(donor_raw, 0.))),
                warning="method-of-moments MC-noise subtraction; negative raw retained; bank-jackknife normal interval is descriptive; fixed recipients")


def coordinate_only_reference(baseline, current):
    """Require exact primary/noncoordinate values and record recomputed secondary coordinates."""
    differences = {}
    for name in ("sgsteps", "particles"):
        baseline_frame, current_frame = baseline[name], current[name]
        protected = [c for c in baseline_frame if c not in ("x0", "y0", "z0", "x1", "y1", "z1")]
        replay.require_checks({name: replay.exact_check(baseline_frame[protected], current_frame[protected])},
                              "current-reference opt-in permits coordinate recomputation only")
        changed = {}
        for column in ("x0", "y0", "z0", "x1", "y1", "z1"):
            if column in baseline_frame:
                delta = current_frame[column].to_numpy(float) - baseline_frame[column].to_numpy(float)
                if np.any(delta != 0):
                    changed[column] = dict(count=int(np.count_nonzero(delta)), max_abs_mm=float(np.max(np.abs(delta))))
        differences[name] = changed
    original_primary, original_parts = replay.reconstruct_primary(baseline)
    new_primary, new_parts = replay.reconstruct_primary(current)
    replay.require_checks({"primary_steps": replay.exact_check(original_primary, new_primary),
                           "primary_particles": replay.exact_check(original_parts, new_parts)}, "current-reference primary changed")
    return differences


def run(baseline_path, bank_path, gdml, outdir, *, banks=8, draws=2, file_index=0, seed=1, replica_seed=701,
        current_reference=False):
    if not 2 <= banks <= 64 or not 2 <= draws <= 8:
        raise ValueError("bounded nested replay needs 2..64 banks and 2..8 draws")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    paths = list(map(Path, (baseline_path, bank_path, str(bank_path).replace(".root", ".meta.json"), gdml)))
    before = {str(path): digest(path) for path in paths}
    original = library.read(str(bank_path))
    bankreplica.validate(original)
    baseline = replay.read_baseline(baseline_path)
    geom = geometry.from_gdml(gdml)
    try:
        default = replay.validate_default(baseline, original, original, geom, seed, file_index)
    except replay.ReplayIdentityError as error:
        report = dict(status="refused", reason=str(error), checks=error.checks, input_sha256=before)
        (outdir / "refused.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not current_reference:
            raise
        replayed, _ = replay.repaste(baseline, original, geom, seed, file_index)
        report["coordinate_differences"] = coordinate_only_reference(baseline, replayed)
        (outdir / "current_reference.json").write_text(json.dumps(report, indent=2) + "\n")
        baseline = replayed
        default = replay.validate_default(baseline, original, original, geom, seed, file_index)
    selected, _ = bankreplica.replica(original, None, selection=np.arange(len(original["events"])))
    clone, _ = replay.repaste(baseline, selected, geom, seed, file_index)
    clone_checks = replay.require_checks({name: replay.exact_check(baseline[name], clone[name])
                                         for name in ("sgsteps", "particles")}, "identity bank remapping changed default paste")
    references = metrics(baseline)
    rows = []
    started = time.perf_counter()
    for bank_index in range(banks):
        replica, ancestry = bankreplica.replica(original, np.random.default_rng([replica_seed, bank_index]), stratify="pbin")
        ancestry["events"].to_parquet(outdir / f"events.{bank_index:03d}.parquet", index=False)
        ancestry["crossings"].to_parquet(outdir / f"crossings.{bank_index:03d}.parquet", index=False)
        for draw_index in range(draws):
            draw_seed = int(np.random.SeedSequence([replica_seed, bank_index, draw_index, 991]).generate_state(1)[0])
            products, placement = replay.repaste(baseline, replica, geom, draw_seed, file_index)
            for table in ("gen", "subruns", "surfsteps"):
                replay.require_checks({table: replay.exact_check(baseline[table], products[table])}, "protected table changed")
            original_primary, original_parts = replay.reconstruct_primary(baseline)
            candidate_primary, candidate_parts = replay.reconstruct_primary(products)
            replay.require_checks({"primary_steps": replay.exact_check(original_primary, candidate_primary),
                                   "primary_particles": replay.exact_check(original_parts, candidate_parts)}, "primary transport changed")
            row = dict(bank=bank_index, draw=draw_index, seed=draw_seed, metrics=metrics(products), placement=placement,
                       distinct_source_events=int(ancestry["events"].source_event_row.nunique()),
                       empty_offaxis_donors=int((replica["events"].noff == 0).sum()),
                       empty_instraw_crossings=int((replica["instraw"].nsec == 0).sum()))
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(scope="source-event finite-bank nested replay on fixed saved recipients; step-level only, not T-V3 track/reco errors",
        stratification="source-event pbin; common multiplicity for all in-straw crossings and off-axis content including empty donors",
        baseline_metrics=references, default_checks=default, identity_clone_checks=clone_checks, replicas=rows,
        current_reference=bool(current_reference), historical_payload_identical=not (outdir / "current_reference.json").exists(),
        variance={name: variance_components([[row["metrics"][name]["mean"] for row in rows if row["bank"] == index]
                                             for index in range(banks)]) for name in references},
        elapsed_s=time.perf_counter() - started, input_sha256=before, admitted_for_production=False,
        source_sha256={str(p): digest(p) for p in (Path(__file__), Path(bankreplica.__file__), Path(library.__file__), Path(replay.__file__))})
    if before != {str(path): digest(path) for path in paths}:
        raise ValueError("bank or saved recipient changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(variance=result["variance"], baseline=references), indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("--library", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--banks", type=int, default=8)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--replica-seed", type=int, default=701)
    parser.add_argument("--current-reference", action="store_true", help="retain historical refusal, explicitly recompute only secondary coordinates")
    args = parser.parse_args()
    run(args.baseline, args.library, args.gdml, args.outdir, banks=args.banks, draws=args.draws,
        file_index=args.file_index, seed=args.seed, current_reference=args.current_reference, replica_seed=args.replica_seed)
