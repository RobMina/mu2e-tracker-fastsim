"""Read-only probes of saved joint-fit states, not a refit or ambiguity decision.

Every original event/group/hit and numerical status is retained. Probes distinguish
finite-difference scale dependence and calibration branches; crossing a bin alone
is not causal attribution. Pointwise LR improvements are conditional diagnostics,
not accepted sign changes. Input hashes identify this invocation, not missing
original association or saved joint-output provenance.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from trkfastsim.digi.digicond import parse_fhicl_dump
from trkfastsim.reco.fielddrift import build_model
from trkfastsim.reco.hitreco import load_conditions
from trkfastsim.reco.stereoseed import _integer
from .run1a_drift import audit, compare_model, read_oracle
from .run1a_timing import load_geometry, match_conditions_config, measurement_arrays, prepare_saved
from .run1a_wirefit import _quantiles, _sha256

IDENTITY = ("evkey", "source_hit_ev", "source_hit_sid", "hit_group", "group_ev", "source_selected", "group_support")
INPUTS = ("hits", "conditions", "association", "reference", "oracle", "config", "wirefit", "wirefit_report",
          "drift_check", "reference_config")
HIT_FIELDS = {"mask", "valid", "initial_lr", "residuals", "doca", "signed_doca", "phi", "delta_t", "r_drift",
              "signed_error", "c_drift", "drift_residual", "pull", "long_pull", "wire_u", "poca_z", "time_bin",
              "phi_bin_lower", "phi_bin_upper", "offset_bin", "signed_error_bin", "lr_preferred", "lr_improvement"}


def read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return dict(source)


def checked_paths(report):
    """Bind current input files to the existing report without rewriting its hashes."""
    paths = {key: Path(report["inputs"][key]["path"]) for key in INPUTS}
    for key, path in paths.items():
        if _sha256(path) != report["inputs"][key]["sha256"]:
            raise ValueError(f"recorded joint-fit input hash differs: {key}")
    return paths


def load_inputs(directory):
    report = json.loads((directory / "report.json").read_text())
    if report["drift_fit"]["numerical_settings"] != dict(condition_limit=1e12, gradient_tolerance=1e-6,
            fd_step=1e-3, initial_damping=1e-3, max_parameter_step=10., poca_iterations=20,
            poca_tolerance=1e-7, poca_max_step=50., poca_max_offset=500.):
        raise ValueError("diagnostic expects recorded default joint numerical settings")
    paths = checked_paths(report)
    hits, association, wirefit = (read_npz(paths[k]) for k in ("hits", "association", "wirefit"))
    keys = hits["evkey"]
    sh = {k[2:]: a for k, a in hits.items() if k.startswith("s_")}
    wire_report = json.loads(paths["wirefit_report"].read_text())
    for key in ("hits", "conditions", "association"):
        if wire_report["inputs"][key]["sha256"] != report["inputs"][key]["sha256"]:
            raise ValueError(f"wire-fit input hash differs: {key}")
    if wire_report["bz_tesla"] != report["nominal_axial_bz_T"]:
        raise ValueError("saved trajectory field differs")
    config = parse_fhicl_dump(paths["config"].read_text())
    match_conditions_config(parse_fhicl_dump(paths["reference_config"].read_text()), config)
    checked = json.loads(paths["drift_check"].read_text())
    if not checked.get("passed") or not checked.get("model", {}).get("passed"):
        raise ValueError("successful drift model audit required")
    for key, path in (("oracle", paths["oracle"]), ("config", paths["config"]),
                      ("export_log", paths["oracle"].parent / "export.log"),
                      ("export_done", paths["oracle"].parent / "export.done"),
                      ("source_manifest", paths["oracle"].parent / "source-sha256.txt")):
        if checked["inputs"][key]["sha256"] != _sha256(path):
            raise ValueError(f"drift audit hash differs: {key}")
    oracle = read_oracle(paths["oracle"])
    audit(oracle, config)
    if not compare_model(oracle, config)["passed"]:
        raise ValueError("stock drift oracle comparison failed")
    low = (oracle["iov_start_run"], oracle["iov_start_subrun"])
    high = (oracle["iov_end_run"], oracle["iov_end_subrun"])
    inside = ((keys[:, 0] > low[0]) | ((keys[:, 0] == low[0]) & (keys[:, 1] >= low[1]))) & (
              (keys[:, 0] < high[0]) | ((keys[:, 0] == high[0]) & (keys[:, 1] <= high[1])))
    if not np.all(inside):
        raise ValueError("input event outside oracle IOV")
    pro = config["services"]["ProditionsService"]
    model = build_model(pro["strawDrift"], pro["strawResponse"], wire_radius=oracle["wire_radius"],
                        outer_radius=oracle["outer_radius"], bz=oracle["origin_field"][2])
    args = (keys, sh, load_conditions(paths["conditions"]), association, wirefit,
            load_geometry(paths["reference"], keys, sh), pro["strawResponse"], model,
            read_npz(directory / "diagnostics.npz"))
    return args, report, paths


def validate_saved(data, wirefit, saved):
    ng, nh = len(data["gev"]), len(data["hg"])
    for key in IDENTITY:
        expected = wirefit[key]
        a = np.asarray(saved[key])
        if a.dtype.kind not in ("b" if key == "source_selected" else "iu") or not np.array_equal(a, expected):
            raise ValueError(f"joint-fit census differs: {key}")
    for key in ("drift_success", "drift_converged", "drift_objective_valid", "profile_profile_valid"):
        if np.asarray(saved[key]).shape != (ng, 2) or np.asarray(saved[key]).dtype.kind != "b":
            raise ValueError(f"invalid joint-fit flag: {key}")
    ok = saved["drift_objective_valid"]
    if np.asarray(saved["drift_status"]).shape != (ng, 2) or np.asarray(saved["drift_status"]).dtype.kind not in "US":
        raise ValueError("invalid saved joint status")
    if (not np.array_equal(saved["drift_success"], saved["drift_converged"])
            or not np.array_equal(saved["drift_status"] == "converged", saved["drift_success"])
            or np.any(saved["drift_success"] & ~ok)
            or np.any(ok & ~saved["profile_profile_valid"])
            or np.any(ok & ~data["eligible"][:, None])):
        raise ValueError("inconsistent saved joint flags/status")
    for key, shape in (("local_parameters", (ng, 2, 5)), ("t0", (ng, 2)), ("z_ref", (ng, 2)),
                       ("parameter_scale", (ng, 2, 6)), ("final_objective", (ng, 2)),
                       ("gradient_norm", (ng, 2)), ("flight_direction", (ng, 2))):
        a = np.asarray(saved["drift_"+key])
        if a.shape != shape or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a[ok])):
            raise ValueError(f"invalid saved joint field: {key}")
    if not np.array_equal(saved["drift_flight_direction"], np.tile([-1, 1], (ng, 1))):
        raise ValueError("joint direction order differs")
    if not np.array_equal(saved["drift_z_ref"], np.repeat(wirefit["fit_z_ref"][:, None], 2, axis=1), equal_nan=True):
        raise ValueError("joint reference z differs")
    selected = data["selected"]
    for key in ("initial_LR", "signed_doca", "phi", "delta_t", "r_drift", "signed_error", "pull", "long_pull"):
        a = np.asarray(saved["source_drift_"+key])
        if a.shape != (nh, 2) or a.dtype.kind not in "fiu":
            raise ValueError(f"invalid saved joint hit field: {key}")
        if not np.all(np.isfinite(a[selected][ok[data["hg"][selected]]])):
            raise ValueError(f"nonfinite saved joint hit field: {key}")
    for key in ("mask", "valid"):
        a = np.asarray(saved["source_drift_"+key])
        if a.shape != (nh, 2) or a.dtype.kind != "b":
            raise ValueError(f"invalid saved joint hit flag: {key}")
    if not np.array_equal(saved["source_drift_mask"], np.repeat(saved["source_selected"][:, None], 2, axis=1)):
        raise ValueError("joint hit mask differs from fixed support")
    if np.any(saved["source_drift_valid"] & ~saved["source_drift_mask"]):
        raise ValueError("saved valid hits outside fixed support")
    if not np.all(saved["source_drift_valid"][selected][ok[data["hg"][selected]]]):
        raise ValueError("saved objective-valid lane has invalid selected hit")
    timing_signed = np.asarray(saved["source_timing_signed_doca_positive"])
    if timing_signed.shape != (nh,) or timing_signed.dtype.kind not in "fiu":
        raise ValueError("invalid saved timing signed DOCA")
    expected_lr = np.sign(timing_signed[selected, None]) * np.array([-1, 1])
    if not np.array_equal(saved["source_drift_initial_LR"][selected][ok[data["hg"][selected]]],
                          expected_lr[ok[data["hg"][selected]]]):
        raise ValueError("saved initial LR differs from initial timing geometry")
    return ok


def summarize(saved, payload, model):
    ok = payload["diagnostic_objective_valid"]
    status = saved["drift_status"]
    out = dict(objective_replayed_hypotheses=int(ok.sum()), assigned_directions=0, accepted_lr_updates=0,
               refitted_hypotheses=0, status_unchanged=True, categories={})
    tolerance = 1e-6*(1+np.sqrt(2*np.where(ok, payload["diagnostic_objective"], 0.)))
    gradients = payload["diagnostic_gradient_norm"]
    small_stationary = payload["diagnostic_linearization_good"][..., -1] & (gradients[..., -1] <= tolerance)
    out["small_stencil_stationary_diagnostic_only"] = int(np.sum(ok & small_stationary))
    for label in np.unique(status):
        rows = status == label
        use = rows & ok
        entry = dict(hypotheses=int(rows.sum()), objective_valid=int(use.sum()),
                     final_objective=_quantiles(payload["diagnostic_objective"][use]),
                     default_gradient_over_tolerance=_quantiles(gradients[..., 0][use]/tolerance[use]),
                     smallest_gradient_over_tolerance=_quantiles(gradients[..., -1][use]/tolerance[use]),
                     smallest_stencil_stationary=int(np.sum(use & small_stationary)),
                     default_full_rank=int(np.sum(use & (payload["diagnostic_rank"][..., 0] == 6))),
                     default_condition=_quantiles(payload["diagnostic_condition"][..., 0][use]),
                     lr_improving_hypotheses=int(np.sum(use & (payload["diagnostic_lr_flip_count"] > 0))),
                     lr_improving_hits=int(payload["diagnostic_lr_flip_count"][use].sum()),
                     pointwise_lr_objective_gain=_quantiles(payload["diagnostic_lr_gain"][use]))
        delta = payload["diagnostic_descent_delta"]
        valid = payload["diagnostic_descent_valid"]
        margin = 128*np.finfo(float).eps*np.maximum(1., np.abs(payload["diagnostic_objective"]))
        descent = np.any(valid & (delta < -margin[..., None, None]), axis=-1)
        for i, name in enumerate(("default", "smallest")):
            entry[name+"_descent_any_strict"] = int(np.sum(use & np.any(valid[..., i, :] & (delta[..., i, :] < 0), axis=-1)))
            entry[name+"_descent_above_roundoff"] = int(np.sum(use & descent[..., i]))
            gain = np.max(np.where(valid[..., i, :], np.maximum(-delta[..., i, :], 0.), 0.), axis=-1)
            entry[name+"_best_probed_objective_gain"] = _quantiles(gain[use & descent[..., i]])
        entry["smallest_descent_without_default_descent"] = int(np.sum(use & descent[..., 1] & ~descent[..., 0]))
        branch_free = np.all(payload["diagnostic_perturbation_valid"][..., -1, :, :], axis=(-2, -1))
        for name in ("time", "phi", "offset", "signed_error", "signed_doca"):
            branch_free &= payload["diagnostic_"+name+"_branch_count"][..., -1] == 0
        entry["smallest_stencil_branch_free"] = int(np.sum(use & branch_free))
        entry["smallest_descent_and_branch_free"] = int(np.sum(use & descent[..., 1] & branch_free))
        entry["no_tested_descent_above_roundoff"] = int(np.sum(use & ~np.any(descent, axis=-1)))
        g = payload["diagnostic_equilibrated_gradient"]
        entry["gradient_change_last_two_scales_over_tolerance"] = _quantiles(
            np.max(np.abs(g[..., -1, :]-g[..., -2, :]), axis=-1)[use]/tolerance[use])
        for name, a in payload.items():
            if name.startswith("diagnostic_") and (name.endswith("_branch_count") or name.endswith("_regression_change_count")):
                entry[name.removeprefix("diagnostic_")+"_rows_by_scale"] = np.sum(a[use] > 0, axis=0).tolist()
        out["categories"][label] = entry
    cal = model.offset
    boundaries = cal.low + cal.width*np.arange(1, len(cal.slope))
    jumps = cal.intercept[1:] + cal.slope[1:]*boundaries - cal.intercept[:-1] - cal.slope[:-1]*boundaries
    out["calibration"] = dict(offset_bins=len(cal.slope), offset_bin_width_mm=float(cal.width),
                              absolute_offset_boundary_jumps_mm=_quantiles(np.abs(jumps)),
                              signed_error_constant=bool(np.all(model.signed.slope == 0) and
                                                        np.all(model.signed.intercept == model.signed.intercept[0])),
                              native_time_step_ns=float(model.delta_t), native_phi_step_rad=float(model.delta_phi))
    return out


def run(keys, sh, cond, association, wirefit, geom, response, model, saved, *, bz, mass=105.6583755, group_batch=32):
    from trkfastsim.reco import driftfitdiagnostic as diagnostic
    from trkfastsim.reco.helixtiming import prepare_timing
    data = prepare_saved(keys, sh, cond, association, wirefit)
    ok = validate_saved(data, wirefit, saved)
    measurements = measurement_arrays(data, geom, response)
    ng, nh, ne = len(data["gev"]), len(data["hg"]), len(keys)
    count, selected = data["count"], data["selected"]
    width = int(count.max(initial=0))
    if width > 256:
        raise ValueError("diagnostic support bound exceeded")
    group_batch = _integer("group_batch", group_batch, 1)
    batch = min(group_batch, 512, max(1, 65536//(2*max(1, width))))
    starts = np.r_[0, np.cumsum(count)]
    rows, hits = {}, {}
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
        poca_z = np.zeros(valid.shape)
        poca_z[valid] = wirefit["source_poca_z"][idx[valid]]
        timing = prepare_timing(wirefit["fit_local_parameters"][groups], wirefit["fit_z_ref"][groups],
            poca_z, **{k: a[::2] for k, a in padded.items()}, valid=valid, bz=bz, mass=mass,
            seed_valid=data["eligible"][groups])
        replay_hits = valid & np.any(ok[groups], axis=1)[:, None]
        if not np.allclose(timing["signed_doca_positive"][replay_hits],
                saved["source_timing_signed_doca_positive"][idx[replay_hits]], rtol=1e-10, atol=1e-9):
            raise ValueError("initial timing signed DOCA replay differs")
        if width:
            cal, hv = padded["wire_cal"][::2], padded["wire_hv"][::2]
            direction = cal-hv
            length = np.sqrt(np.sum(direction*direction, axis=-1))
            direction /= np.where(valid, length, 1.)[..., None]
            measured_z = (.5*cal+.5*hv)[..., 2]+u*direction[..., 2]
            with np.errstate(invalid="ignore"):
                span = np.max(np.where(valid, measured_z, -np.inf), axis=1)-np.min(
                    np.where(valid, measured_z, np.inf), axis=1)
            extent = np.where(count[groups] > 0, np.maximum(1., span), 1.)
            pars = wirefit["fit_local_parameters"][groups]
            slope = np.hypot(pars[:, 2], pars[:, 3])
            scale = np.column_stack((np.ones(len(groups)), np.ones(len(groups)), 1/extent, 1/extent,
                                     1/extent/extent/np.maximum(slope, 1/extent), np.full(len(groups), 10.)))
            if not np.allclose(saved["drift_parameter_scale"][groups][ok[groups]],
                    np.repeat(scale[:, None, :], 2, axis=1)[ok[groups]], rtol=1e-12, atol=0):
                raise ValueError("saved parameter scale differs from initial geometry")
        lr = np.zeros(valid.shape+(2,))
        lr[valid] = saved["source_drift_initial_LR"][idx[valid]]
        result = diagnostic.diagnose(saved["drift_local_parameters"][groups].reshape(-1, 5),
            saved["drift_z_ref"][groups].reshape(-1), saved["drift_t0"][groups].reshape(-1),
            saved["drift_flight_direction"][groups].reshape(-1), **padded,
            measured_u=np.repeat(u, 2, axis=0), uvar=np.repeat(variance, 2, axis=0),
            valid=np.repeat(valid, 2, axis=0), model=model, bz=bz, mass=mass,
            initial_lr=np.moveaxis(lr, 2, 1).reshape(2*len(groups), width),
            parameter_scale=saved["drift_parameter_scale"][groups].reshape(-1, 6),
            seed_valid=ok[groups].reshape(-1))
        for key, value in result.items():
            a = np.asarray(value).reshape((len(groups), 2)+np.asarray(value).shape[1:])
            if key in HIT_FIELDS:
                hits.setdefault(key, np.full((nh, 2)+a.shape[3:], np.nan if a.dtype.kind == "f" else 0, a.dtype))
                hits[key][idx[valid]] = np.moveaxis(a, 1, 2)[valid]
            else:
                rows.setdefault(key, []).append(a)
    payload = {key: value.copy() for key, value in saved.items()}
    payload.update({"diagnostic_"+key: np.concatenate(value) for key, value in rows.items()})
    payload.update({"source_diagnostic_"+key: value for key, value in hits.items()})
    if not np.array_equal(payload["diagnostic_objective_valid"], ok):
        raise ValueError("saved objective validity replay differs")
    for key, actual in (("final_objective", payload["diagnostic_objective"]),
                        ("gradient_norm", payload["diagnostic_gradient_norm"][..., 0])):
        if not np.allclose(saved["drift_"+key][ok], actual[ok], rtol=1e-10, atol=1e-9):
            raise ValueError(f"saved joint {key} replay differs")
    for key in ("signed_doca", "phi", "delta_t", "r_drift", "signed_error", "pull", "long_pull"):
        use = ok[data["hg"][selected]]
        if not np.allclose(saved["source_drift_"+key][selected][use],
                           payload["source_diagnostic_"+key][selected][use], rtol=1e-10, atol=1e-9):
            raise ValueError(f"saved joint hit {key} replay differs")
    report = summarize(saved, payload, model)
    report.update(events=ne, groups=ng, hits=nh, fixed_selected_hits=len(selected),
                  zero_hit_events=int(np.sum(np.bincount(data["sh"]["ev"], minlength=ne) == 0)),
                  group_batch=group_batch, original_arrays_preserved=True,
                  caveat=__doc__, mechanism_tags_are_overlapping_not_causal_partition=True,
                  input_provenance_caveat="Original association inputs and prior joint-output NPZ lack recorded hash binding; "
                                         "current hashes and replay do not repair those historical gaps.")
    return report, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--group-batch", default=32, type=int)
    parser.add_argument("--compare-saved", type=Path)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    started = time.perf_counter()
    inputs, original, paths = load_inputs(args.saved)
    loaded = time.perf_counter()
    report, payload = run(*inputs, bz=original["nominal_axial_bz_T"], mass=original["mass_hypothesis_MeV"],
                          group_batch=args.group_batch)
    report.update(load_and_oracle_seconds=loaded-started, processing_seconds=time.perf_counter()-loaded,
                  processing_scope="read-only final-state probes; no fit/search or baseline rerun")
    from trkfastsim.reco import driftfitdiagnostic, helixdriftfit, helixtiming, fielddrift, helixfit
    from trkfastsim.validation import run1a_timing, run1a_wirefit, run1a_drift
    paths.update(timing_driver=Path(run1a_timing.__file__), wirefit_driver=Path(run1a_wirefit.__file__),
                 oracle_driver=Path(run1a_drift.__file__))
    report["historical_source_hash_equality_required"] = False
    report["historical_implementation_hashes"] = {k: v for k, v in original["inputs"].items() if k not in INPUTS}
    report["implementation_provenance"] = "Current sources are hashed separately; historical sources are not asserted identical."
    report["finite_difference_steps"] = list(driftfitdiagnostic.FD_STEPS)
    report["descent_steps"] = list(driftfitdiagnostic.DESCENT_STEPS)
    report["descent_fd_steps"] = list(driftfitdiagnostic.DESCENT_FD_STEPS)
    report["coordinate_order"] = list(driftfitdiagnostic.COORDINATES)
    report["slope_side_order"] = list(driftfitdiagnostic.SIDES)
    report["branch_order"] = list(driftfitdiagnostic.BRANCH_LABELS)
    paths.update(saved_report=args.saved / "report.json", saved_payload=args.saved / "diagnostics.npz", driver=Path(__file__))
    paths.update({name: Path(module.__file__) for name, module in (("diagnostic_source", driftfitdiagnostic),
                  ("joint_source", helixdriftfit), ("timing_source", helixtiming), ("drift_source", fielddrift),
                  ("poca_source", helixfit))})
    report["inputs"] = {key: dict(path=str(path.resolve()), sha256=_sha256(path)) for key, path in paths.items()}
    if args.compare_saved:
        previous = read_npz(args.compare_saved)
        if set(previous) != set(payload):
            raise ValueError("diagnostic repeat schema differs")
        changed = [k for k, a in payload.items() if not np.array_equal(a, previous[k], equal_nan=a.dtype.kind in "fc")]
        if changed:
            raise ValueError(f"diagnostic repeat differs: {changed}")
        report["saved_comparison"] = dict(exact=True, different_arrays=[])
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir / "diagnostics.npz", **payload)
    (args.outdir / "report.json").write_text(text)
    print(text, end="")
    return report


if __name__ == "__main__":
    main()
