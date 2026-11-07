"""Measured-only synthetic census, original-seed replay and provenance guards."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_run1a_timing import fixture
from trkfastsim.validation import run1a_stabledrift as driver
from trkfastsim.validation.run1a_timing import run as timing_run


@pytest.fixture(scope="module")
def inputs():
    args = fixture()
    _, saved = timing_run(*args, bz=1.0117, fit_drift=True)
    return args+(saved,)


@pytest.fixture(scope="module")
def result(inputs):
    return driver.run(*inputs, bz=1.0117)


def test_full_census_every_baseline_array_preserved(inputs, result):
    report, payload = result
    assert report["events"] == report["groups"] == 3
    assert report["hits"] == report["fixed_selected_hits"] == 12
    assert report["zero_hit_events"] == 2
    assert report["assigned_directions"] == report["accepted_lr_updates"] == 0
    assert report["initial_objective_equality_verified"]
    assert report["initial_objective_replayed_hypotheses"] == 2
    assert report["baseline_final_state_objective_default_gradient_replayed"]
    for key, value in inputs[-1].items():
        np.testing.assert_array_equal(payload[key], value, err_msg=key)
        assert not np.shares_memory(payload[key], value)
    assert payload["candidate_status"].shape == (3, 2)
    assert payload["source_candidate_signed_doca"].shape == (12, 2)
    for name in ("correction", "g", "h"):
        assert payload["source_candidate_final_poca_"+name].shape == (12, 2)
        assert "candidate_final_poca_"+name not in payload
    assert payload["source_candidate_residuals"].shape == (12, 2, 2)
    assert payload["candidate_trace_parameters"].shape == (3, 2, 40, 6)
    np.testing.assert_array_equal(payload["candidate_flight_direction"], np.tile([-1, 1], (3, 1)))
    assert not any("momentum" in key or "covariance" in key or "winner" in key or key == "candidate_p_abs"
                   for key in payload if key.startswith("candidate_"))
    assert sum(sum(row.values()) for row in report["status_transition_table"].values()) == 6
    assert sum(report["termination_status_counts"].values()) == 6
    assert sum(report["certificate_status_counts"].values()) == 6
    assert report["candidate_max_iterations"] == report["baseline_max_iterations"] == 40
    assert "NOT equal compute" in report["budget_scope"]
    assert report["fit_trial_evaluations"] == int(payload["candidate_trace_trial_evaluated"].sum())
    assert report["certificate_evaluations"] == int(payload["candidate_certificate_evaluations"].sum())
    assert report["baseline_replay_seconds"] > 0 and report["candidate_seconds"] > 0
    json.dumps(report, allow_nan=False)


def test_exact_group_batches(inputs, result):
    _, whole = result
    _, split = driver.run(*inputs, bz=1.0117, group_batch=1)
    assert set(whole) == set(split)
    for key in whole:
        np.testing.assert_array_equal(whole[key], split[key], err_msg=key)


def test_original_seeds_not_saved_final_states(inputs, result):
    saved, payload = inputs[-1], result[1]
    assert not np.array_equal(saved["drift_local_parameters"][1], saved["drift_initial_local_parameters"][1])
    for key in ("local_parameters", "t0"):
        np.testing.assert_array_equal(payload["candidate_initial_"+key], saved["drift_initial_"+key])
    np.testing.assert_array_equal(payload["candidate_parameter_scale"], saved["drift_parameter_scale"])
    np.testing.assert_array_equal(payload["source_candidate_initial_lr"], saved["source_drift_initial_LR"])
    good = saved["drift_initial_objective_valid"]
    np.testing.assert_allclose(payload["candidate_initial_objective"][good], saved["drift_initial_objective"][good],
                               rtol=1e-10, atol=1e-9)


def test_missing_optional_seed_arrays_fall_back_only_to_wire_profile(inputs):
    args = copy.deepcopy(inputs)
    del args[-1]["drift_initial_local_parameters"]
    del args[-1]["drift_initial_t0"]
    _, payload = driver.run(*args, bz=1.0117, max_iterations=0)
    np.testing.assert_array_equal(payload["candidate_initial_local_parameters"],
                                  np.repeat(args[4]["fit_local_parameters"][:, None, :], 2, axis=1))
    np.testing.assert_array_equal(payload["candidate_initial_t0"], args[-1]["profile_t0"])


def test_failed_baseline_rows_are_attempted_from_profile_not_success(inputs, monkeypatch):
    from trkfastsim.reco import stabledriftfit
    args = copy.deepcopy(inputs)
    for key in ("drift_success", "drift_converged"):
        args[-1][key][:] = False
    args[-1]["drift_status"][1] = "max_iterations"
    original = stabledriftfit.fit_stable_drift_helix
    seen = []

    def check(*a, **kw):
        seen.append(kw["seed_valid"].copy())
        return original(*a, **kw)

    monkeypatch.setattr(stabledriftfit, "fit_stable_drift_helix", check)
    report, payload = driver.run(*args, bz=1.0117, max_iterations=0)
    np.testing.assert_array_equal(np.concatenate(seen), args[-1]["profile_profile_valid"].reshape(-1))
    assert payload["candidate_initial_objective_valid"].sum() == 2
    assert report["baseline_success"]["hypotheses"] == 0
    assert report["objective_comparison_by_baseline_status"]["max_iterations"]["comparable_valid_hypotheses"] == 2
    assert report["objective_comparison_all"]["objective_worse_than_baseline"] == 2


def test_baseline_diagnostic_invoked_exactly_once(inputs, monkeypatch):
    original, calls = driver.diagnostic.run, []

    def check(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(driver.diagnostic, "run", check)
    driver.run(*inputs, bz=1.0117, max_iterations=0)
    assert len(calls) == 1


@pytest.mark.parametrize("key", ["drift_initial_local_parameters", "drift_initial_t0", "profile_t0"])
def test_changed_original_seeds_rejected(inputs, key):
    args = copy.deepcopy(inputs)
    args[-1][key][1] += .1
    with pytest.raises(ValueError, match="original wire/profile seed"):
        driver.run(*args, bz=1.0117, max_iterations=0)


@pytest.mark.parametrize("key", ["drift_final_objective", "drift_gradient_norm", "drift_t0", "drift_parameter_scale",
                                 "source_drift_signed_doca", "source_drift_initial_LR"])
def test_baseline_replay_tamper_rejected(inputs, key):
    args = copy.deepcopy(inputs)
    args[-1][key] += .2
    with pytest.raises(ValueError, match="replay|scale|LR"):
        driver.run(*args, bz=1.0117, max_iterations=0)


def test_initial_objective_tamper_rejected(inputs):
    args = copy.deepcopy(inputs)
    args[-1]["drift_initial_objective"][1] += 1
    with pytest.raises(ValueError, match="initial objective replay"):
        driver.run(*args, bz=1.0117, max_iterations=0)


@pytest.mark.parametrize("key", ["drift_parameter_scale", "source_drift_initial_LR", "drift_initial_objective"])
def test_mandatory_original_fields_not_inferred(inputs, key):
    args = copy.deepcopy(inputs)
    del args[-1][key]
    with pytest.raises(KeyError):
        driver.run(*args, bz=1.0117, max_iterations=0)


@pytest.mark.parametrize("key", driver.diagnostic.IDENTITY)
def test_full_census_tamper_rejected(inputs, key):
    args = copy.deepcopy(inputs)
    a = args[-1][key]
    args[-1][key] = ~a if a.dtype.kind == "b" else a+1
    with pytest.raises(ValueError, match="census"):
        driver.run(*args, bz=1.0117, max_iterations=0)


def test_no_input_mutation_even_readonly_arrays(inputs):
    args = copy.deepcopy(inputs)
    before = copy.deepcopy(args)
    for mapping in args:
        if isinstance(mapping, dict):
            for value in mapping.values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
    driver.run(*args, bz=1.0117, max_iterations=0)
    for old, current in zip(before, args):
        if isinstance(old, dict):
            for key in old:
                if isinstance(old[key], np.ndarray):
                    np.testing.assert_array_equal(old[key], current[key], err_msg=key)
        elif isinstance(old, np.ndarray):
            np.testing.assert_array_equal(old, current)


def test_no_overwrite_precedes_loading(tmp_path, monkeypatch):
    def forbidden(*a):
        raise AssertionError("must not load")
    monkeypatch.setattr(driver.diagnostic, "load_inputs", forbidden)
    with pytest.raises(FileExistsError):
        driver.main(["--saved", str(tmp_path/"missing"), "--outdir", str(tmp_path)])


@pytest.mark.parametrize("missing", ["sha256", "wirefit", "inputs"])
def test_missing_input_hashes_rejected_before_outputs(tmp_path, missing):
    saved = tmp_path/"saved"
    saved.mkdir()
    paths = {key: saved/key for key in driver.diagnostic.INPUTS}
    for path in paths.values():
        path.write_text("synthetic")
    report = dict(drift_fit=dict(numerical_settings=dict(condition_limit=1e12, gradient_tolerance=1e-6,
        fd_step=1e-3, initial_damping=1e-3, max_parameter_step=10., poca_iterations=20,
        poca_tolerance=1e-7, poca_max_step=50., poca_max_offset=500.)),
        inputs={key: dict(path=str(path), sha256=driver._sha256(path)) for key, path in paths.items()})
    if missing == "sha256":
        del report["inputs"]["hits"]["sha256"]
    elif missing == "wirefit":
        del report["inputs"]["wirefit"]
    else:
        del report["inputs"]
    (saved/"report.json").write_text(json.dumps(report))
    out = tmp_path/"out"
    with pytest.raises(KeyError):
        driver.main(["--saved", str(saved), "--outdir", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("zero_groups", [False, True])
def test_all_empty_and_zero_group_census(zero_groups):
    args = list(fixture())
    args[1] = {k: v[:0] for k, v in args[1].items()}
    assoc, fit = args[3], args[4]
    for key in tuple(assoc):
        if key.startswith(("source_hit_", "association_source_")) or key == "hit_group":
            assoc[key] = assoc[key][:0]
    assoc["association_group_support"][:] = 0
    assoc["associated_seed_success"][:] = False
    for key in tuple(fit):
        if key.startswith("source_") or key == "hit_group":
            fit[key] = fit[key][:0]
    fit["group_support"][:] = 0
    for key in ("fit_success", "fit_converged", "fit_physical_identifiable"):
        fit[key] = np.zeros(3, bool)
    fit["fit_status"] = np.full(3, "empty")
    if zero_groups:
        for mapping in (assoc, fit):
            for key in tuple(mapping):
                if key != "evkey":
                    mapping[key] = mapping[key][:0]
    _, saved = timing_run(*args, bz=1.0117, fit_drift=True)
    report, payload = driver.run(*args, saved, bz=1.0117)
    assert report["events"] == report["zero_hit_events"] == 3
    assert report["groups"] == (0 if zero_groups else 3)
    assert report["hits"] == report["fixed_selected_hits"] == 0
    assert payload["source_candidate_signed_doca"].shape == (0, 2)
    for name in ("correction", "g", "h"):
        assert payload["source_candidate_final_poca_"+name].shape == (0, 2)
    assert payload["candidate_trace_parameters"].shape == (report["groups"], 2, 40, 6)
    for key, value in saved.items():
        np.testing.assert_array_equal(payload[key], value)
    json.dumps(report, allow_nan=False)


def test_cli_fresh_output_separate_provenance_and_exact_comparison(inputs, result, tmp_path, monkeypatch):
    saved = tmp_path/"baseline"
    saved.mkdir()
    original = dict(drift_fit=dict(max_iterations=40), nominal_axial_bz_T=1.0117,
                    mass_hypothesis_MeV=105.6583755, inputs=dict(historical=dict(path="old.py", sha256="historical")))
    (saved/"report.json").write_text(json.dumps(original))
    np.savez(saved/"diagnostics.npz", **inputs[-1])
    path = tmp_path/"measured"
    path.write_text("input")
    monkeypatch.setattr(driver.diagnostic, "load_inputs", lambda directory: (inputs, original, {"hits": path}))
    monkeypatch.setattr(driver, "run", lambda *a, **kw: (copy.deepcopy(result[0]), result[1]))
    out = tmp_path/"new"
    report = driver.main(["--saved", str(saved), "--outdir", str(out)])
    assert report["historical_implementation_hashes"] == original["inputs"]
    assert report["saved_baseline"]["payload"]["sha256"] == driver._sha256(saved/"diagnostics.npz")
    assert report["inputs"]["hits"]["sha256"] == driver._sha256(path)
    assert report["current_implementation"]
    assert report["load_and_oracle_seconds"] >= 0
    assert json.loads((saved/"report.json").read_text()) == original
    again = tmp_path/"again"
    report = driver.main(["--saved", str(saved), "--outdir", str(again), "--compare-saved", str(out/"diagnostics.npz")])
    assert report["saved_comparison"]["exact"]
    modified = {k: v.copy() for k, v in result[1].items()}
    modified["candidate_t0"][1] += 1
    np.savez(tmp_path/"changed.npz", **modified)
    with pytest.raises(ValueError, match="repeat differs"):
        driver.main(["--saved", str(saved), "--outdir", str(tmp_path/"bad"), "--compare-saved", str(tmp_path/"changed.npz")])
    assert not (tmp_path/"bad").exists()


@pytest.mark.parametrize("kwargs", [{"group_batch": 0}, {"group_batch": True}, {"max_iterations": -1}, {"max_iterations": 201}])
def test_invalid_bounds(inputs, kwargs):
    with pytest.raises(ValueError):
        driver.run(*inputs, bz=1.0117, **kwargs)


def test_unselected_hits_retained_without_candidate_residuals():
    args = list(fixture())
    args[1] = {k: np.concatenate((v, v[:1])) for k, v in args[1].items()}
    for mapping in (args[3], args[4]):
        for key in tuple(mapping):
            if key.startswith(("source_", "association_source_")) or key == "hit_group":
                mapping[key] = np.concatenate((mapping[key], mapping[key][:1]))
        mapping["hit_group"][-1] = -1
    args[3]["association_source_used"][-1] = False
    args[4]["source_selected"][-1] = False
    _, saved = timing_run(*args, bz=1.0117, fit_drift=True)
    report, payload = driver.run(*args, saved, bz=1.0117, max_iterations=0)
    assert report["hits"] == 13 and report["fixed_selected_hits"] == 12
    assert not payload["source_candidate_mask"][-1].any()
    assert not payload["source_candidate_valid"][-1].any()
    assert np.isnan(payload["source_candidate_signed_doca"][-1]).all()
    for name in ("correction", "g", "h"):
        assert np.isnan(payload["source_candidate_final_poca_"+name][-1]).all()


def test_width_bound_before_diagnostic_and_candidate(inputs, monkeypatch):
    original = driver.prepare_saved

    def oversized(*a):
        data = original(*a)
        data["count"] = np.full(3, 257)
        return data

    def forbidden(*a, **kw):
        raise AssertionError("must reject width before replay")

    monkeypatch.setattr(driver, "prepare_saved", oversized)
    monkeypatch.setattr(driver.diagnostic, "run", forbidden)
    with pytest.raises(ValueError, match="support bound"):
        driver.run(*inputs, bz=1.0117)


def test_huge_requested_batch_clamped(inputs, result):
    report, payload = driver.run(*inputs, bz=1.0117, group_batch=1000000)
    assert report["effective_group_batch"] <= 512
    assert report["effective_group_batch"]*2*report["fixed_full_width"] <= 65536
    for key, value in result[1].items():
        np.testing.assert_array_equal(payload[key], value, err_msg=key)


def test_objective_comparison_includes_every_old_status_and_success_stratum(inputs, result):
    saved, payload = copy.deepcopy(inputs[-1]), copy.deepcopy(result[1])
    saved["drift_status"] = np.array([["invalid_seed", "linearization_failed"], ["converged", "stalled"],
                                      ["max_iterations", "empty"]])
    saved["drift_objective_valid"] = np.ones((3, 2), bool)
    saved["drift_final_objective"] = np.full((3, 2), 10.)
    payload["candidate_objective_valid"] = np.ones((3, 2), bool)
    payload["candidate_final_objective"] = np.array([[11., 9.], [10., 12.], [13., 14.]])
    report = driver.summarize(saved, payload)
    assert report["objective_comparison_all"]["comparable_valid_hypotheses"] == 6
    assert report["objective_comparison_all"]["objective_worse_than_baseline"] == 4
    assert set(report["objective_comparison_by_baseline_status"]) == set(saved["drift_status"].ravel())
    assert all(row["comparable_valid_hypotheses"] == 1 for row in report["objective_comparison_by_baseline_status"].values())


def test_changed_hash_rejected():
    # Use this module itself as an available local file; the claimed hash is wrong.
    path = Path(__file__)
    report = {"inputs": {key: dict(path=str(path), sha256="not-the-current-hash") for key in driver.diagnostic.INPUTS}}
    with pytest.raises(ValueError, match="input hash differs"):
        driver.diagnostic.checked_paths(report)


def test_candidate_payload_not_accepted_as_baseline(inputs, result):
    with pytest.raises(ValueError, match="must be the baseline"):
        driver.run(*inputs[:-1], result[1], bz=1.0117)


def test_trace_axis_equal_hit_width_is_not_misclassified(inputs):
    _, payload = driver.run(*inputs, bz=1.0117, max_iterations=12)
    assert payload["candidate_trace_objective"].shape == (3, 2, 12)
    assert payload["source_candidate_signed_doca"].shape == (12, 2)
    assert "source_candidate_trace_objective" not in payload


def test_different_support_counts_keep_full_width_and_exact_batches():
    args = list(fixture())
    # Second nonempty event has only eight selected hits; global width stays 12.
    sh, assoc, fit = args[1], args[3], args[4]
    for key in tuple(sh):
        sh[key] = np.concatenate((sh[key], sh[key][:8]))
    sh["ev"][-8:] = 2
    for mapping in (assoc, fit):
        for key in tuple(mapping):
            if key.startswith(("source_", "association_source_")) or key == "hit_group":
                mapping[key] = np.concatenate((mapping[key], mapping[key][:8]))
        mapping["source_hit_ev"][-8:] = 2
        mapping["hit_group"][-8:] = 2
    assoc["association_group_support"][2] = fit["group_support"][2] = 8
    assoc["associated_seed_success"][2] = True
    for key in ("fit_success", "fit_converged", "fit_physical_identifiable"):
        fit[key][2] = True
    fit["fit_status"][2] = "converged"
    _, saved = timing_run(*args, bz=1.0117, fit_drift=True)
    report, whole = driver.run(*args, saved, bz=1.0117)
    other, split = driver.run(*args, saved, bz=1.0117, group_batch=1)
    assert report["hits"] == report["fixed_selected_hits"] == 20
    assert report["fixed_full_width"] == other["fixed_full_width"] == 12
    for key in whole:
        np.testing.assert_array_equal(whole[key], split[key], err_msg=key)


def test_original_valid_failed_final_rows_remain_candidate_seeds(inputs):
    args = copy.deepcopy(inputs)
    for key in ("drift_success", "drift_converged", "drift_objective_valid"):
        args[-1][key][:] = False
    args[-1]["drift_status"][1] = "linearization_failed"
    report, payload = driver.run(*args, bz=1.0117, max_iterations=0)
    assert report["objective_replayed_hypotheses"] == 0
    assert report["initial_objective_replayed_hypotheses"] == 2
    assert payload["candidate_initial_objective_valid"].sum() == 2
    assert report["objective_comparison_all"]["candidate_objective_valid"] == 2
    assert report["objective_comparison_all"]["comparable_valid_hypotheses"] == 0
    np.testing.assert_array_equal(payload["drift_status"], args[-1]["drift_status"])


@pytest.mark.parametrize("tamper", ["LR", "scale", "timing_and_LR"])
def test_original_seed_replay_guards_even_without_valid_baseline_final(inputs, tamper):
    args = copy.deepcopy(inputs)
    for key in ("drift_success", "drift_converged", "drift_objective_valid"):
        args[-1][key][:] = False
    args[-1]["drift_status"][1] = "linearization_failed"
    if tamper == "scale":
        args[-1]["drift_parameter_scale"][1] *= 2
    else:
        args[-1]["source_drift_initial_LR"] *= -1
        if tamper == "timing_and_LR":
            args[-1]["source_timing_signed_doca_positive"] *= -1
    with pytest.raises(ValueError, match="original .*replay differs"):
        driver.run(*args, bz=1.0117, max_iterations=0)


def test_identical_readonly_criterion_on_baseline_is_separate_from_original_labels(inputs, result):
    report, payload = result
    np.testing.assert_array_equal(payload['drift_success'], inputs[-1]['drift_success'])
    assert report['baseline_same_check_success']['hypotheses'] == payload['baseline_check_certified'].sum()
    assert sum(report['baseline_same_check_status_counts'].values()) == 6
    assert sum(report['baseline_same_check_pass_by_original_status'].values()) == payload['baseline_check_certified'].sum()
    assert payload['baseline_check_certificate_scale_pass'].shape == (3, 2, 4)
    assert report['baseline_adjacent_stencil_check_seconds'] > 0
    assert report['eligible_groups'] == 1
    assert report['candidate_success']['neither_eligible_groups'] <= report['eligible_groups']
