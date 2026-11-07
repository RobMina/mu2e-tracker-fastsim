"""Independent fixed-support, fixed-LR numerical candidate from original seeds.

Both flight directions remain unassigned. Counts are numerical diagnostics, not
tracking efficiency or full-physics validation. The historical development1000
195 baseline convergence labels are not robust stationarity certificates:
only 13 passed the same threshold at the smallest diagnostic FD stencil. Neither
that stencil nor this candidate's local derivative checks prove optimality.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from trkfastsim.reco.stereoseed import _integer
from . import run1a_drift_diagnostic as diagnostic
from .run1a_timing import measurement_arrays, prepare_saved
from .run1a_wirefit import _quantiles, _sha256

# Explicit membership, never shape heuristics (a trace/parameter axis can equal H).
HIT_FIELDS = diagnostic.HIT_FIELDS | {"initial_LR", "poca_valid", "poca_iterations", "poca_converged",
    "poca_inside", "poca_boundary", "timing_valid", "drift_valid", "inside", "stationary", "parallel",
    "angle_valid", "sign_valid", "outside_oracle_mask", "final_poca_correction", "final_poca_g", "final_poca_h"}


def _counts(a):
    labels, counts = np.unique(a, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def original_seeds(wirefit, saved):
    """Require original wire/profile identities, including invalid/empty lanes."""
    expected = {"local_parameters": np.repeat(wirefit["fit_local_parameters"][:, None, :], 2, axis=1),
                "t0": np.asarray(saved["profile_t0"])}
    ng = len(wirefit["group_ev"])
    if expected["t0"].shape != (ng, 2) or expected["t0"].dtype.kind not in "fiu":
        raise ValueError("invalid original profile t0")
    seeds = {}
    for key, value in expected.items():
        actual = np.asarray(saved.get("drift_initial_"+key, value))
        if actual.dtype.kind not in "fiu" or not np.array_equal(actual, value, equal_nan=True):
            raise ValueError(f"original wire/profile seed differs: {key}")
        seeds[key] = actual.copy()
    for key, kind in (("initial_objective_valid", "b"), ("initial_objective", "fiu")):
        a = np.asarray(saved["drift_"+key])
        if a.shape != (ng, 2) or a.dtype.kind not in kind:
            raise ValueError(f"invalid saved initial field: {key}")
    initial_ok = saved["drift_initial_objective_valid"]
    if (np.any(initial_ok & ~saved["profile_profile_valid"])
            or not np.all(np.isfinite(saved["drift_initial_objective"][initial_ok]))):
        raise ValueError("invalid original objective/seed validity")
    return seeds


def _replay_initial(groups, valid, idx, padded, u, data, wirefit, saved, *, bz, mass):
    """Extend final-state replay guards to original-valid but failed-final lanes."""
    from trkfastsim.reco.helixtiming import prepare_timing
    ok = saved["drift_initial_objective_valid"][groups]
    poca_z = np.zeros(valid.shape)
    poca_z[valid] = wirefit["source_poca_z"][idx[valid]]
    timing = prepare_timing(wirefit["fit_local_parameters"][groups], wirefit["fit_z_ref"][groups],
        poca_z, **{k: a[::2] for k, a in padded.items()}, valid=valid, bz=bz, mass=mass,
        seed_valid=data["eligible"][groups])
    use = valid & np.any(ok, axis=1)[:, None]
    if not np.allclose(timing["signed_doca_positive"][use],
            saved["source_timing_signed_doca_positive"][idx[use]], rtol=1e-10, atol=1e-9):
        raise ValueError("original timing signed DOCA replay differs")
    lr = np.zeros(valid.shape+(2,))
    lr[valid] = saved["source_drift_initial_LR"][idx[valid]]
    check = valid[..., None] & ok[:, None, :]
    expected = np.sign(timing["signed_doca_positive"])[..., None]*np.array([-1, 1])
    if not np.array_equal(lr[check], expected[check]):
        raise ValueError("original initial LR replay differs")
    if valid.shape[1]:
        cal, hv = padded["wire_cal"][::2], padded["wire_hv"][::2]
        direction = cal-hv
        length = np.sqrt(np.sum(direction*direction, axis=-1))
        direction /= np.where(valid, length, 1.)[..., None]
        measured_z = (.5*cal+.5*hv)[..., 2]+u*direction[..., 2]
        with np.errstate(invalid="ignore"):
            span = np.max(np.where(valid, measured_z, -np.inf), axis=1)-np.min(
                np.where(valid, measured_z, np.inf), axis=1)
        extent = np.where(data["count"][groups] > 0, np.maximum(1., span), 1.)
        pars = wirefit["fit_local_parameters"][groups]
        slope = np.hypot(pars[:, 2], pars[:, 3])
        scale = np.column_stack((np.ones(len(groups)), np.ones(len(groups)), 1/extent, 1/extent,
                                 1/extent/extent/np.maximum(slope, 1/extent), np.full(len(groups), 10.)))
        if not np.allclose(saved["drift_parameter_scale"][groups][ok],
                np.repeat(scale[:, None, :], 2, axis=1)[ok], rtol=1e-12, atol=0):
            raise ValueError("original parameter scale replay differs")
    return np.moveaxis(lr, 2, 1).reshape(2*len(groups), valid.shape[1])


def summarize(saved, payload):
    old, new = saved["drift_status"], payload["candidate_status"]
    valid = saved["drift_objective_valid"] & payload["candidate_objective_valid"]
    delta = payload["candidate_final_objective"]-saved["drift_final_objective"]

    def objectives(rows):
        use = rows & valid
        return dict(hypotheses=int(rows.sum()), comparable_valid_hypotheses=int(use.sum()),
                    baseline_objective_valid=int(np.sum(rows & saved["drift_objective_valid"])),
                    candidate_objective_valid=int(np.sum(rows & payload["candidate_objective_valid"])),
                    no_comparable_objective=int(np.sum(rows & ~valid)),
                    candidate_final_minus_baseline_final=_quantiles(delta[use]),
                    objective_worse_than_baseline=int(np.sum(use & (delta > 0))),
                    objective_equal_baseline=int(np.sum(use & (delta == 0))),
                    objective_better_than_baseline=int(np.sum(use & (delta < 0))))

    labels = np.unique(new)
    out = dict(baseline_status_counts=_counts(old), candidate_status_counts=_counts(new),
        status_transition_table={str(label): {str(target): int(np.sum((old == label) & (new == target)))
            for target in labels} for label in np.unique(old)},
        termination_status_counts=_counts(payload["candidate_termination_status"]),
        certificate_status_counts=_counts(payload["candidate_certificate_status"]),
        objective_comparison_all=objectives(np.ones(old.shape, bool)),
        objective_comparison_by_baseline_status={str(label): objectives(old == label) for label in np.unique(old)},
        objective_comparison_scope="All rows valid in both fits, including failures; no success or residual recut. "
            "Worse final objectives are allowed for independent trajectories from the same seeds.",
        residual_comparison="No residual improvement or resolution claim; all original hit identities are retained.",
        assigned_directions=0, accepted_lr_updates=0, original_arrays_preserved=True,
        original_seed_equality_verified=True, initial_objective_equality_verified=True,
        initial_objective_replayed_hypotheses=int(saved["drift_initial_objective_valid"].sum()),
        initial_objective_replay_tolerance=dict(rtol=1e-10, atol=1e-9),
        candidate_missing_value_policy="Unattempted condition/gradient values are NaN regardless of batch peers; "
            "unselected hit floats are NaN and flags false. Baseline arrays are never normalized or changed.",
        direction_order=[-1, 1],
        fit_trial_evaluations=int(payload["candidate_fit_trial_evaluations"].sum()),
        certificate_evaluations=int(payload["candidate_certificate_evaluations"].sum()),
        evaluation_count_scope="Solver-reported per-hypothesis trial and certificate evaluations; "
            "not baseline diagnostic probes, analytic Jacobian work, or hardware throughput.")
    eligible_groups = np.any(saved["profile_profile_valid"], axis=1)
    out["eligible_groups"] = int(eligible_groups.sum())
    for prefix, success in (("baseline", saved["drift_success"]), ("candidate", payload["candidate_success"]),
                            ("baseline_same_check", payload["baseline_check_certified"])):
        out[prefix+"_success"] = dict(hypotheses=int(success.sum()), any_direction_groups=int(np.any(success, axis=1).sum()),
            both_direction_groups=int(np.all(success, axis=1).sum()), neither_direction_groups=int((~np.any(success, axis=1)).sum()),
            neither_eligible_groups=int(np.sum(eligible_groups & ~np.any(success, axis=1))),
            negative_z=int(success[:, 0].sum()), positive_z=int(success[:, 1].sum()))
    out["baseline_same_check_status_counts"] = _counts(payload["baseline_check_certificate_status"])
    out["baseline_same_check_pass_by_original_status"] = {str(label): int(np.sum((old == label) & payload["baseline_check_certified"]))
                                                         for label in np.unique(old)}
    out["baseline_same_check_scope"] = "Read-only application of identical analytic/adjacent-FD criterion to saved final states; original labels unchanged."
    out["analytic_stationary_hypotheses"] = int(payload["candidate_analytic_stationary"].sum())
    out["selected_certificate_pair_counts"] = _counts(payload["candidate_certificate_pair_index"][payload["candidate_success"]])
    out["candidate_gradient_over_threshold_checked"] = _quantiles((payload["candidate_gradient_norm"] /
        (1e-6*(1+np.sqrt(2*np.where(payload["candidate_objective_valid"], payload["candidate_final_objective"], 0.)))))[payload["candidate_success"]])
    use = payload["candidate_objective_valid"][saved["hit_group"][saved["source_selected"]]]
    out["candidate_final_abs_poca_correction_mm"] = _quantiles(np.abs(payload["source_candidate_final_poca_correction"][saved["source_selected"]][use]))
    out["linearization_checks"] = int(payload["candidate_linearization_checks"].sum())
    threshold = 1e-6*(1+np.sqrt(2*np.where(payload["candidate_objective_valid"], payload["candidate_final_objective"], 0.)))
    compatible = payload["candidate_certificate_same_branch"] & payload["candidate_certificate_numeric_good"]
    pairs = compatible[..., :-1] & compatible[..., 1:]
    out["candidate_certificate_details"] = dict(
        compatible_pairs_per_row=_counts(pairs.sum(axis=-1)[payload["candidate_objective_valid"]]),
        compatible_rows_by_scale=np.sum(compatible, axis=(0, 1)).tolist(),
        checked_rows_by_scale=np.sum(payload["candidate_certificate_scale_pass"], axis=(0, 1)).tolist(),
        compatible_gradient_disagreement_over_threshold=_quantiles((payload["candidate_certificate_gradient_difference"] /
            threshold[..., None])[compatible]),
        compatible_max_column_relative_error=_quantiles(np.max(payload["candidate_certificate_column_relative_error"], axis=-1)[compatible]))
    out["candidate_accepted_trial_count"] = int(np.sum(payload["candidate_trace_accepted_backtrack"] >= 0))
    out["candidate_invalid_trial_attempts"] = int(np.sum(payload["candidate_trace_trial_evaluated"] & ~payload["candidate_trace_trial_valid"]))
    out["candidate_unchanged_parameter_trials"] = int(payload["candidate_trace_trial_same_parameters"].sum())
    out["evaluation_count_scope"] = "Logical per-hypothesis trial, analytic-check and eligible certificate attempts; not total batched work or valid-probe counts."
    return out


def run(keys, sh, cond, association, wirefit, geom, response, model, saved, *, bz,
        mass=105.6583755, group_batch=32, max_iterations=40):
    """Replay the saved baseline once, then run an independent original-seed fit."""
    from trkfastsim.reco.stabledriftfit import fit_stable_drift_helix, certify
    from trkfastsim.reco.driftfitdiagnostic import _Evaluator, _empty_state
    from trkfastsim.reco.driftjacobian import linearize
    group_batch = _integer("group_batch", group_batch, 1)
    max_iterations = _integer("max_iterations", max_iterations, 0)
    if max_iterations > 200:
        raise ValueError("candidate iteration bound exceeded")
    if any(k.startswith(("candidate_", "source_candidate_")) for k in saved):
        raise ValueError("saved input must be the baseline, not a candidate output")
    prepared = time.perf_counter()
    data = prepare_saved(keys, sh, cond, association, wirefit)
    diagnostic.validate_saved(data, wirefit, saved)
    measurements = measurement_arrays(data, geom, response)
    seeds = original_seeds(wirefit, saved)
    ng, nh, ne = len(data["gev"]), len(data["hg"]), len(keys)
    count, selected = data["count"], data["selected"]
    width = int(count.max(initial=0))
    if width > 256:
        raise ValueError("candidate support bound exceeded")
    batch = min(group_batch, 512, max(1, 65536//(2*max(1, width))))
    replay_started = time.perf_counter()
    replay_report, replay_payload = diagnostic.run(keys, sh, cond, association, wirefit, geom, response, model, saved,
        bz=bz, mass=mass, group_batch=group_batch)
    del replay_payload
    candidate_started = time.perf_counter()
    starts = np.r_[0, np.cumsum(count)]
    rows, hits, baseline_checks = {}, {}, {}
    baseline_check_seconds = 0.
    for begin in range(0, max(1, ng), batch):
        groups = np.arange(begin, min(ng, begin+batch))
        valid = np.arange(width)[None, :] < count[groups, None]
        idx = np.zeros(valid.shape, np.int64)
        if len(selected):
            idx[valid] = selected[(starts[groups, None]+np.arange(width)[None, :])[valid]]
        padded = {}
        for key, value in measurements.items():
            a = np.zeros(valid.shape+value.shape[1:], value.dtype)
            a[valid] = value[idx[valid]]
            padded[key] = np.repeat(a, 2, axis=0)
        u, variance = np.zeros(valid.shape), np.ones(valid.shape)
        u[valid], variance[valid] = data["u"][idx[valid]], data["uvar"][idx[valid]]
        lr = _replay_initial(groups, valid, idx, padded, u, data, wirefit, saved, bz=bz, mass=mass)
        result = fit_stable_drift_helix(seeds["local_parameters"][groups].reshape(-1, 5),
            saved["drift_z_ref"][groups].reshape(-1), seeds["t0"][groups].reshape(-1),
            saved["drift_flight_direction"][groups].reshape(-1), **padded,
            measured_u=np.repeat(u, 2, axis=0), uvar=np.repeat(variance, 2, axis=0),
            valid=np.repeat(valid, 2, axis=0), model=model, bz=bz, mass=mass, initial_lr=lr,
            parameter_scale=saved["drift_parameter_scale"][groups].reshape(-1, 6),
            seed_valid=saved["profile_profile_valid"][groups].reshape(-1), max_iterations=max_iterations)
        check_started = time.perf_counter()
        old_evaluator = _Evaluator(saved["drift_local_parameters"][groups].reshape(-1, 5),
            saved["drift_z_ref"][groups].reshape(-1), saved["drift_t0"][groups].reshape(-1),
            saved["drift_flight_direction"][groups].reshape(-1), **padded,
            measured_u=np.repeat(u, 2, axis=0), uvar=np.repeat(variance, 2, axis=0),
            valid=np.repeat(valid, 2, axis=0), model=model, bz=bz, mass=mass, initial_lr=lr,
            parameter_scale=saved["drift_parameter_scale"][groups].reshape(-1, 6),
            seed_valid=saved["drift_objective_valid"][groups].reshape(-1))
        old_state = old_evaluator(old_evaluator.candidate) if len(groups) and width else _empty_state(old_evaluator)
        old_lin = linearize(old_evaluator, old_evaluator.candidate, old_state)
        old_check = certify(old_evaluator, old_evaluator.candidate, old_state, old_lin)
        old_check.update(analytic_gradient_norm=np.where(old_state["objective_valid"], old_lin["gradient_norm"], np.nan))
        for key, value in old_check.items():
            baseline_checks.setdefault(key, []).append(np.asarray(value).reshape((len(groups), 2)+np.asarray(value).shape[1:]))
        baseline_check_seconds += time.perf_counter()-check_started
        for key, value in result.items():
            a = np.asarray(value)
            if a.shape[:1] != (2*len(groups),):
                raise RuntimeError(f"unexpected candidate output {key} shape {a.shape}")
            a = a.reshape((len(groups), 2)+a.shape[1:])
            if key in HIT_FIELDS:
                if a.shape[2:3] != (width,):
                    raise RuntimeError(f"unexpected candidate hit output {key}")
                hits.setdefault(key, np.full((nh, 2)+a.shape[3:], np.nan if a.dtype.kind == "f" else 0, a.dtype))
                hits[key][idx[valid]] = np.moveaxis(a, 1, 2)[valid]
            else:
                rows.setdefault(key, []).append(a)
    payload = {key: value.copy() for key, value in saved.items()}
    payload.update({"candidate_"+key: np.concatenate(value) for key, value in rows.items()})
    payload.update({"source_candidate_"+key: value for key, value in hits.items()})
    payload.update({"baseline_check_"+key: np.concatenate(value) for key, value in baseline_checks.items()})
    initial_ok = saved["drift_initial_objective_valid"]
    if not np.array_equal(payload["candidate_initial_objective_valid"], initial_ok):
        raise ValueError("original initial objective validity replay differs")
    if not np.allclose(payload["candidate_initial_objective"][initial_ok], saved["drift_initial_objective"][initial_ok],
                       rtol=1e-10, atol=1e-9):
        raise ValueError("original initial objective replay differs")
    if not np.array_equal(payload["source_candidate_mask"], saved["source_drift_mask"]):
        raise ValueError("candidate changed fixed support")
    if not np.array_equal(payload["source_candidate_initial_lr"][selected], saved["source_drift_initial_LR"][selected]):
        raise ValueError("candidate changed original LR")
    objective_valid = payload["candidate_objective_valid"]
    if np.any(objective_valid & (payload["candidate_final_objective"] > payload["candidate_initial_objective"])):
        raise ValueError("candidate increased original fixed-support objective")
    report = summarize(saved, payload)
    report["candidate_objective_decreased_from_initial"] = int(np.sum(objective_valid &
        (payload["candidate_final_objective"] < payload["candidate_initial_objective"])))
    report.update(events=ne, groups=ng, hits=nh, fixed_selected_hits=len(selected),
        zero_hit_events=int(np.sum(np.bincount(data["sh"]["ev"], minlength=ne) == 0)),
        group_batch=group_batch, effective_group_batch=batch, fixed_full_width=width,
        candidate_max_iterations=max_iterations, baseline_max_iterations=40,
        numerical_settings=dict(condition_limit=1e12, gradient_tolerance=1e-6, initial_damping=1e-3,
            max_parameter_step=10., max_backtracks=8, finite_difference_steps=[1e-3, 1e-4, 1e-5, 1e-6],
            gradient_agreement_fraction=.25, jacobian_column_relative_tolerance=1e-3,
            certificate_scope="analytic stationary, adjacent-stencil checked; not a local-minimum or physics certificate"),
        budget_scope="Default candidate 40 steps versus saved baseline 40 steps is NOT equal compute. "
            "Trials, analytic derivatives and certificate evaluations have different costs.",
        preparation_seconds=replay_started-prepared, baseline_replay_seconds=candidate_started-replay_started,
        candidate_seconds=time.perf_counter()-candidate_started-baseline_check_seconds,
        baseline_adjacent_stencil_check_seconds=baseline_check_seconds,
        baseline_final_state_objective_default_gradient_replayed=True,
        objective_replayed_hypotheses=replay_report["objective_replayed_hypotheses"],
        nominal_axial_bz_T=float(bz), mass_hypothesis_MeV=float(mass), origin_drift_bz_T=float(model.bz),
        conditions_scope="Measured endpoints, resolved propagation calibration and stock drift response; nominal axial "
            "trajectory field and stated mass hypothesis. CLI repeats saved hashes, service configuration, oracle and IOV "
            "checks; not a complete database snapshot, field-map/material fit or stock-track validation.",
        input_provenance_caveat=replay_report["input_provenance_caveat"], caveat=__doc__)
    return report, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--group-batch", default=32, type=int)
    parser.add_argument("--max-iterations", default=40, type=int)
    parser.add_argument("--compare-saved", type=Path)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    started = time.perf_counter()
    inputs, original, paths = diagnostic.load_inputs(args.saved)
    loaded = time.perf_counter()
    if original["drift_fit"]["max_iterations"] != 40:
        raise ValueError("candidate comparison requires the saved default40 baseline")
    report, payload = run(*inputs, bz=original["nominal_axial_bz_T"], mass=original["mass_hypothesis_MeV"],
                          group_batch=args.group_batch, max_iterations=args.max_iterations)
    report.update(load_and_oracle_seconds=loaded-started,
        processing_scope="Original-seed candidate plus separately timed read-only baseline final-state diagnostic replay; "
            "no wire fit, association search or baseline optimizer rerun; timings exclude output writing.",
        historical_source_hash_equality_required=False,
        historical_implementation_hashes={k: v for k, v in original["inputs"].items() if k not in diagnostic.INPUTS},
        implementation_provenance="Current code hashes are separate from unchanged historical implementation hashes.")
    report["inputs"] = {key: dict(path=str(path.resolve()), sha256=_sha256(path)) for key, path in paths.items()}
    report["saved_baseline"] = {key: dict(path=str(path.resolve()), sha256=_sha256(path)) for key, path in
        (("report", args.saved/"report.json"), ("payload", args.saved/"diagnostics.npz"))}
    from trkfastsim.reco import stabledriftfit, driftjacobian, driftfitdiagnostic, helixdriftfit, helixtiming, fielddrift, helixfit
    from . import run1a_timing, run1a_wirefit, run1a_drift
    modules = (stabledriftfit, driftjacobian, driftfitdiagnostic, helixdriftfit, helixtiming, fielddrift, helixfit,
               diagnostic, run1a_timing, run1a_wirefit, run1a_drift)
    sources = {module.__name__: Path(module.__file__) for module in modules}
    sources[__name__] = Path(__file__)
    report["current_implementation"] = {key: dict(path=str(path.resolve()), sha256=_sha256(path))
                                        for key, path in sources.items()}
    if args.compare_saved:
        previous = diagnostic.read_npz(args.compare_saved)
        if set(previous) != set(payload):
            raise ValueError("candidate repeat schema differs")
        changed = [k for k, a in payload.items() if not np.array_equal(a, previous[k], equal_nan=a.dtype.kind in "fc")]
        if changed:
            raise ValueError(f"candidate repeat differs: {changed}")
        report["saved_comparison"] = dict(exact=True, different_arrays=[], path=str(args.compare_saved.resolve()),
                                          sha256=_sha256(args.compare_saved))
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir/"diagnostics.npz", **payload)
    (args.outdir/"report.json").write_text(text)
    print(text, end="")
    return report


if __name__ == "__main__":
    main()
