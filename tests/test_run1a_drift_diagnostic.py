"""Saved-state diagnostic census, replay and no-overwrite tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_run1a_timing import fixture
from trkfastsim.validation import run1a_drift_diagnostic as driver
from trkfastsim.validation.run1a_timing import run as timing_run


@pytest.fixture(scope="module")
def inputs():
    args = fixture()
    _, saved = timing_run(*args, bz=1.0117, fit_drift=True)
    return args + (saved,)


def test_diagnostic_preserves_every_original_array(inputs):
    report, payload = driver.run(*inputs, bz=1.0117)
    assert report["events"] == report["groups"] == 3
    assert report["hits"] == report["fixed_selected_hits"] == 12
    assert report["zero_hit_events"] == 2
    assert report["objective_replayed_hypotheses"] == 2
    assert report["accepted_lr_updates"] == report["refitted_hypotheses"] == report["assigned_directions"] == 0
    for key, value in inputs[-1].items():
        np.testing.assert_array_equal(payload[key], value, err_msg=key)
    assert payload["diagnostic_gradient_norm"].shape == (3, 2, 4)
    assert payload["source_diagnostic_signed_doca"].shape == (12, 2)
    json.dumps(report, allow_nan=False)


def test_exact_group_batches(inputs):
    _, whole = driver.run(*inputs, bz=1.0117)
    _, split = driver.run(*inputs, bz=1.0117, group_batch=1)
    for key in whole:
        np.testing.assert_array_equal(whole[key], split[key], err_msg=key)


@pytest.mark.parametrize("key", driver.IDENTITY)
def test_changed_census_rejected(inputs, key):
    args = copy.deepcopy(inputs)
    a = args[-1][key]
    args[-1][key] = ~a if a.dtype.kind == "b" else a+1
    with pytest.raises(ValueError, match="census"):
        driver.run(*args, bz=1.0117)


@pytest.mark.parametrize("key", ["source_selected", "hit_group", "group_support", "evkey"])
def test_value_preserving_wrong_census_dtype_rejected(inputs, key):
    args = copy.deepcopy(inputs)
    args[-1][key] = args[-1][key].astype(float)
    with pytest.raises(ValueError, match="census"):
        driver.run(*args, bz=1.0117)


@pytest.mark.parametrize("key", ["final_objective", "gradient_norm", "t0", "parameter_scale"])
def test_changed_joint_state_rejected_by_replay(inputs, key):
    args = copy.deepcopy(inputs)
    args[-1]["drift_"+key][1] += .3
    with pytest.raises(ValueError, match="replay|scale"):
        driver.run(*args, bz=1.0117)


@pytest.mark.parametrize("key", ["initial_LR", "signed_doca", "r_drift", "signed_error", "long_pull"])
def test_changed_hit_field_rejected(inputs, key):
    args = copy.deepcopy(inputs)
    args[-1]["source_drift_"+key][0] += .2
    with pytest.raises(ValueError, match="LR|replay"):
        driver.run(*args, bz=1.0117)


def test_changed_direction_rejected(inputs):
    args = copy.deepcopy(inputs)
    args[-1]["drift_flight_direction"][1] *= -1
    with pytest.raises(ValueError, match="direction"):
        driver.run(*args, bz=1.0117)


def test_failed_states_are_probed_not_promoted(inputs):
    args = copy.deepcopy(inputs)
    args[-1]["drift_success"][1] = False
    args[-1]["drift_converged"][1] = False
    args[-1]["drift_status"][1] = "max_iterations"
    report, payload = driver.run(*args, bz=1.0117)
    assert report["categories"]["max_iterations"]["objective_valid"] == 2
    assert not payload["drift_success"].any()
    np.testing.assert_array_equal(payload["drift_status"], args[-1]["drift_status"])


def test_no_overwrite_precedes_input_loading(tmp_path):
    with pytest.raises(FileExistsError):
        driver.main(["--saved", str(tmp_path/"missing"), "--outdir", str(tmp_path)])


def test_hash_binding_detects_changed_inputs(tmp_path):
    paths = {key: tmp_path/key for key in driver.INPUTS}
    for path in paths.values():
        path.write_text("fixture")
    report = {"inputs": {key: dict(path=str(path), sha256=driver._sha256(path)) for key, path in paths.items()}}
    assert driver.checked_paths(report) == paths
    paths["association"].write_text("changed")
    with pytest.raises(ValueError, match="hash differs: association"):
        driver.checked_paths(report)


@pytest.mark.parametrize('status', ['linearization_failed', 'ill_conditioned', 'poca_boundary'])
def test_every_failure_category_retained(inputs, status):
    args = copy.deepcopy(inputs)
    args[-1]['drift_success'][1] = False
    args[-1]['drift_converged'][1] = False
    args[-1]['drift_status'][1] = status
    report, _ = driver.run(*args, bz=1.0117)
    assert report['categories'][status]['hypotheses'] == 2
    assert sum(v['hypotheses'] for v in report['categories'].values()) == 6


def test_invalid_selected_hit_rejected(inputs):
    args = copy.deepcopy(inputs)
    args[-1]['source_drift_valid'][0, 0] = False
    with pytest.raises(ValueError, match='invalid selected hit'):
        driver.run(*args, bz=1.0117)


def test_bad_timing_shape_rejected(inputs):
    args = copy.deepcopy(inputs)
    args[-1]['source_timing_signed_doca_positive'] = np.r_[args[-1]['source_timing_signed_doca_positive'], 0.]
    with pytest.raises(ValueError, match='timing signed DOCA'):
        driver.run(*args, bz=1.0117)


@pytest.mark.parametrize('zero_groups', [False, True])
def test_all_empty_census(zero_groups):
    args = list(fixture())
    args[1] = {k: v[:0] for k, v in args[1].items()}
    assoc, fit = args[3], args[4]
    for key in tuple(assoc):
        if key.startswith('source_hit_') or key == 'hit_group' or key.startswith('association_source_'):
            assoc[key] = assoc[key][:0]
    assoc['association_group_support'][:] = 0
    assoc['associated_seed_success'][:] = False
    for key in tuple(fit):
        if key.startswith('source_') or key == 'hit_group':
            fit[key] = fit[key][:0]
    fit['group_support'][:] = 0
    for key in ('fit_success', 'fit_converged', 'fit_physical_identifiable'):
        fit[key] = np.zeros(3, bool)
    fit['fit_status'] = np.full(3, 'empty')
    if zero_groups:
        for mapping in (assoc, fit):
            for key in tuple(mapping):
                if key != 'evkey':
                    mapping[key] = mapping[key][:0]
    _, saved = timing_run(*args, bz=1.0117, fit_drift=True)
    report, payload = driver.run(*args, saved, bz=1.0117)
    assert report['events'] == report['zero_hit_events'] == 3
    assert report['groups'] == (0 if zero_groups else 3)
    assert report['hits'] == report['fixed_selected_hits'] == report['objective_replayed_hypotheses'] == 0
    assert payload['source_diagnostic_signed_doca'].shape == (0, 2)
    for key, value in saved.items():
        np.testing.assert_array_equal(payload[key], value)
    json.dumps(report, allow_nan=False)


def test_coordinated_initial_lr_and_timing_tamper_rejected(inputs):
    args = copy.deepcopy(inputs)
    args[-1]['source_timing_signed_doca_positive'] *= -1
    args[-1]['source_drift_initial_LR'] *= -1
    with pytest.raises(ValueError, match='initial timing signed DOCA replay'):
        driver.run(*args, bz=1.0117)


def test_unselected_hit_diagnostics_remain_missing(inputs):
    args = list(copy.deepcopy(inputs[:-1]))
    nh = len(args[1]['sid'])
    args[1] = {k: np.concatenate((v, v[:1])) for k, v in args[1].items()}
    for mapping in (args[3], args[4]):
        for key in tuple(mapping):
            if key.startswith('source_') or key.startswith('association_source_') or key == 'hit_group':
                mapping[key] = np.concatenate((mapping[key], mapping[key][:1]))
        mapping['hit_group'][-1] = -1
    args[3]['association_source_used'][-1] = False
    args[4]['source_selected'][-1] = False
    _, saved = timing_run(*args, bz=1.0117, fit_drift=True)
    report, payload = driver.run(*args, saved, bz=1.0117)
    assert report['hits'] == nh+1 and report['fixed_selected_hits'] == nh
    assert not payload['source_diagnostic_valid'][-1].any()
    assert np.isnan(payload['source_diagnostic_signed_doca'][-1]).all()
    saved['source_drift_valid'][-1] = True
    with pytest.raises(ValueError, match='outside fixed support'):
        driver.run(*args, saved, bz=1.0117)
