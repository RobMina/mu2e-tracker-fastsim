"""Synthetic saved-final replay, immutable census and CLI provenance regressions."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_run1a_timing import fixture
from trkfastsim.validation import run1a_drift_failure as driver
from trkfastsim.validation import run1a_stabledrift as stable
from trkfastsim.validation.run1a_timing import run as timing_run


def chain(args, **kwargs):
    _, saved = timing_run(*args, bz=1.0117, fit_drift=True)
    report, candidate = stable.run(*args, saved, bz=1.0117, **kwargs)
    return tuple(args)+(saved, candidate), report


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def inputs():
    return chain(fixture())


@pytest.fixture(scope="module")
def result(inputs):
    return driver.run(*inputs[0], bz=1.0117)


def test_every_array_immutable_and_exact_replay(inputs, result):
    args = copy.deepcopy(inputs[0])
    before = copy.deepcopy(args)
    for mapping in args:
        for value in mapping.values() if isinstance(mapping, dict) else ():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
    report, payload = driver.run(*args, bz=1.0117, group_batch=1)
    assert set(payload) == set(result[1])
    for key, value in payload.items():
        exact(value, result[1][key])
    for old, current in zip(before, args):
        if isinstance(old, dict):
            for key, value in old.items():
                if isinstance(value, np.ndarray):
                    exact(value, current[key])
        elif isinstance(old, np.ndarray):
            exact(old, current)
    for mapping in args[-2:]:
        for key, value in mapping.items():
            exact(value, payload[key])
            assert not np.shares_memory(value, payload[key])
    assert report['events'] == report['groups'] == 3
    assert report['hits'] == report['fixed_selected_hits'] == 12
    assert report['zero_hit_events'] == 2
    assert report['assigned_directions'] == report['accepted_lr_updates'] == 0
    assert report['diagnostic_only'] and report['saved_candidate_replayed_exactly']
    assert sum(report['saved_status_counts'].values()) == 6
    candidate = args[-1]
    for name in ('objective_valid', 'gradient_norm', 'certificate_status', 'certificate_same_branch',
                 'certificate_numeric_good', 'certificate_scale_pass', 'certificate_gradient_difference',
                 'certificate_column_relative_error'):
        a, b = payload['failure_original_'+name], candidate['candidate_'+name]
        use = candidate['candidate_objective_valid'] if a.dtype.kind == 'f' else slice(None)
        exact(a[use], b[use])
    valid = candidate['candidate_objective_valid']
    exact(payload['failure_original_objective'][valid], candidate['candidate_final_objective'][valid])
    exact(payload['failure_original_certified'], candidate['candidate_success'])
    assert not payload['failure_original_objective_valid'][[0, 2]].any()
    assert not payload['failure_polished_certified'][[0, 2]].any()
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('tamper', ['baseline', 'LR', 'scale', 'shape', 'dtype', 'census', 'validity',
                                    'gradient', 'certificate', 'branch'])
def test_tamper_rejected(inputs, tamper):
    args = copy.deepcopy(inputs[0])
    saved, candidate = args[-2:]
    if tamper == 'baseline':
        candidate['drift_t0'][1] += 1
    elif tamper == 'LR':
        candidate['source_candidate_initial_lr'] *= -1
    elif tamper == 'scale':
        candidate['candidate_parameter_scale'][1] *= 2
    elif tamper == 'shape':
        candidate['candidate_local_parameters'] = candidate['candidate_local_parameters'][:, :1]
    elif tamper == 'dtype':
        candidate['candidate_success'] = candidate['candidate_success'].astype(np.int64)
    elif tamper == 'census':
        for mapping in (saved, candidate):
            mapping['source_hit_sid'] += 1
    elif tamper == 'validity':
        candidate['candidate_objective_valid'][0, 0] = True
    elif tamper == 'gradient':
        candidate['candidate_gradient_norm'][1] += 1
    elif tamper == 'certificate':
        candidate['candidate_certificate_status'][1] = 'tampered'
    else:
        candidate['candidate_certificate_same_branch'][1] ^= True
    with pytest.raises(ValueError, match='differs|invalid candidate|inconsistent'):
        driver.run(*args, bz=1.0117)


@pytest.mark.parametrize('invalid_seed', [False, True])
def test_invalid_and_failed_finals_are_not_promoted(invalid_seed):
    measured = fixture()
    if invalid_seed:
        for key in ('fit_success', 'fit_converged', 'fit_physical_identifiable'):
            measured[4][key][1] = False
        measured[4]['fit_status'] = np.array(['empty', 'invalid_seed', 'empty'])
    args, _ = chain(measured, max_iterations=0)
    candidate = args[-1]
    assert candidate['candidate_objective_valid'].sum() == (0 if invalid_seed else 2)
    assert not candidate['candidate_success'].any()
    report, payload = driver.run(*args, bz=1.0117)
    exact(payload['candidate_status'], candidate['candidate_status'])
    exact(payload['candidate_success'], candidate['candidate_success'])
    exact(payload['failure_original_objective_valid'], candidate['candidate_objective_valid'])
    assert not payload['failure_polished_objective_valid'][~candidate['candidate_objective_valid']].any()
    assert sum(row['hypotheses'] for row in report['by_saved_candidate_status'].values()) == 6


@pytest.mark.parametrize('zero_groups', [False, True])
def test_empty_and_zero_group_census(zero_groups):
    args = list(fixture())
    args[1] = {k: v[:0] for k, v in args[1].items()}
    assoc, fit = args[3:5]
    for mapping in (assoc, fit):
        for key in tuple(mapping):
            if key.startswith(('source_', 'association_source_')) or key == 'hit_group':
                mapping[key] = mapping[key][:0]
    assoc['association_group_support'][:] = fit['group_support'][:] = 0
    assoc['associated_seed_success'][:] = False
    for key in ('fit_success', 'fit_converged', 'fit_physical_identifiable'):
        fit[key][:] = False
    fit['fit_status'] = np.full(3, 'empty')
    if zero_groups:
        for mapping in (assoc, fit):
            for key in tuple(mapping):
                if key != 'evkey':
                    mapping[key] = mapping[key][:0]
    inputs, _ = chain(args)
    report, payload = driver.run(*inputs, bz=1.0117)
    assert report['events'] == report['zero_hit_events'] == 3
    assert report['groups'] == (0 if zero_groups else 3)
    assert report['hits'] == report['fixed_selected_hits'] == 0
    assert payload['failure_original_objective_valid'].shape == (report['groups'], 2)
    assert payload['source_failure_poca_shift'].shape == (0, 2)
    for key, value in inputs[-1].items():
        exact(value, payload[key])
    json.dumps(report, allow_nan=False)


def test_unselected_hit_mapping(result):
    args = list(fixture())
    args[1] = {k: np.concatenate((v, v[:1])) for k, v in args[1].items()}
    for mapping in args[3:5]:
        for key in tuple(mapping):
            if key.startswith(('source_', 'association_source_')) or key == 'hit_group':
                mapping[key] = np.concatenate((mapping[key], mapping[key][:1]))
        mapping['hit_group'][-1] = -1
    args[3]['association_source_used'][-1] = args[4]['source_selected'][-1] = False
    inputs, _ = chain(args)
    report, payload = driver.run(*inputs, bz=1.0117)
    assert report['hits'] == 13 and report['fixed_selected_hits'] == 12
    for key, value in payload.items():
        if key.startswith('source_failure_'):
            exact(value[:-1], result[1][key])
            if value.dtype.kind == 'f':
                assert np.isnan(value[-1]).all()
            elif '_bin' in key:
                assert (value[-1] == -1).all()
            else:
                assert not value[-1].any()
    for key, value in inputs[-1].items():
        exact(value, payload[key])


@pytest.fixture
def disk(inputs, tmp_path, monkeypatch):
    saved, candidate = tmp_path/'baseline', tmp_path/'candidate'
    saved.mkdir()
    candidate.mkdir()
    original = dict(nominal_axial_bz_T=1.0117, mass_hypothesis_MeV=105.6583755)
    (saved/'report.json').write_text(json.dumps(original))
    np.savez(saved/'diagnostics.npz', **inputs[0][-2])
    measured = tmp_path/'measured'
    measured.write_text('synthetic measured inputs')
    paths = {k: measured for k in driver.prior.INPUTS}
    report = copy.deepcopy(inputs[1])
    report['saved_baseline'] = {k: dict(sha256=driver._sha256(saved/f)) for k, f in
                              (('report', 'report.json'), ('payload', 'diagnostics.npz'))}
    report['inputs'] = {k: dict(sha256=driver._sha256(p)) for k, p in paths.items()}
    report['current_implementation'] = {'historical.source': {'sha256': 'old-source-hash'}}
    (candidate/'report.json').write_text(json.dumps(report))
    np.savez(candidate/'diagnostics.npz', **inputs[0][-1])
    def load(directory):
        assert directory == saved
        return inputs[0][:-1], original, paths
    monkeypatch.setattr(driver.prior, 'load_inputs', load)
    return ['--saved', str(saved), '--candidate', str(candidate)], report


def test_cli_nooverwrite_before_loading(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError('must not load')
    monkeypatch.setattr(driver.prior, 'load_inputs', forbidden)
    with pytest.raises(FileExistsError):
        driver.main(['--saved', 'missing', '--candidate', 'missing', '--outdir', str(tmp_path)])


@pytest.mark.parametrize('tamper', ['report', 'payload', 'input', 'settings', 'missing'])
def test_cli_hash_and_settings_guards(disk, tmp_path, tamper):
    argv, report = disk
    if tamper in ('report', 'payload'):
        report['saved_baseline'][tamper]['sha256'] = 'wrong'
    elif tamper == 'input':
        report['inputs']['hits']['sha256'] = 'wrong'
    elif tamper == 'settings':
        report['numerical_settings']['gradient_tolerance'] *= 2
    else:
        del report['inputs']['hits']['sha256']
    (tmp_path/'candidate'/'report.json').write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        driver.main(argv+['--outdir', str(tmp_path/'bad')])
    assert not (tmp_path/'bad').exists()


def test_cli_sources_output_hash_and_exact_comparison(disk, result, tmp_path):
    argv, historical = disk
    out = tmp_path/'out'
    report = driver.main(argv+['--outdir', str(out), '--group-batch', '1'])
    assert report['historical_candidate_implementation'] == historical['current_implementation']
    assert report['output_payload_sha256'] == driver._sha256(out/'diagnostics.npz')
    assert json.loads((out/'report.json').read_text()) == report
    for source in report['current_implementation'].values():
        assert source['sha256'] == driver._sha256(Path(source['path']))
    for source in report['inputs'].values():
        assert source['sha256'] == driver._sha256(Path(source['path']))
    payload = driver.prior.read_npz(out/'diagnostics.npz')
    for key, value in result[1].items():
        exact(value, payload[key])
    again = driver.main(argv+['--outdir', str(tmp_path/'again'), '--compare-saved', str(out/'diagnostics.npz')])
    assert again['saved_comparison']['exact']
    payload['candidate_t0'][1] += 1
    np.savez(tmp_path/'changed.npz', **payload)
    with pytest.raises(ValueError, match='repeat differs'):
        driver.main(argv+['--outdir', str(tmp_path/'bad'), '--compare-saved', str(tmp_path/'changed.npz')])
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('key', ['source_candidate_poca_z', 'source_candidate_residuals',
    'source_candidate_time_bin', 'candidate_certificate_gradient_norm',
    'candidate_certificate_column_norm_ratio', 'candidate_certificate_pair_index'])
def test_reconstructible_final_field_tamper_rejected(inputs, key):
    args = copy.deepcopy(inputs[0])
    args[-1][key] += 1
    with pytest.raises(ValueError, match='replay differs'):
        driver.run(*args, bz=1.0117)


def test_partially_nonfinite_raw_parameters_retained(inputs):
    args = copy.deepcopy(inputs[0])
    args[-1]['candidate_local_parameters'][0, 0] = [23., -14., np.nan, .5, .002]
    args[-1]['candidate_t0'][0, 0] = 987.
    _, payload = driver.run(*args, bz=1.0117)
    np.testing.assert_array_equal(payload['failure_original_parameters'][0, 0], [23., -14., np.nan, .5, .002, 987.])
    assert not payload['failure_original_objective_valid'][0, 0]


def test_nonnumeric_candidate_linearization_raw_sentinel_replay(inputs, monkeypatch):
    args = copy.deepcopy(inputs[0][:-1])

    def fail(*a, **kw):
        raise np.linalg.LinAlgError('forced')

    from trkfastsim.reco import driftjacobian
    original = driftjacobian._normal_equations

    def broken(*a, **kw):
        with monkeypatch.context() as patch:
            patch.setattr(np.linalg, 'eigh', fail)
            return original(*a, **kw)

    monkeypatch.setattr(driftjacobian, '_normal_equations', broken)
    # Historical summary divides its max-float failure placeholder before success masking.
    with np.errstate(over='ignore'):
        _, candidate = stable.run(*args, bz=1.0117)
    assert (candidate['candidate_status'][1] == 'linearization_failed').all()
    report, payload = driver.run(*args, candidate, bz=1.0117)
    assert report['saved_candidate_replayed_exactly']
    assert np.isnan(payload['failure_original_gradient_norm'][1]).all()
    np.testing.assert_array_equal(payload['failure_original_raw_gradient_norm'][1], np.finfo(float).max)
