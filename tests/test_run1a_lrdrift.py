"""Measured-only saved-final continuation, immutable census and provenance guards."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_run1a_timing import fixture
from trkfastsim.reco import lrdriftfit as kernel
from trkfastsim.validation import run1a_lrdrift as driver
from trkfastsim.validation import run1a_stabledrift as stable
from trkfastsim.validation.run1a_timing import run as timing_run


def chain(measured, **kwargs):
    _, saved = timing_run(*measured, bz=1.0117, fit_drift=True)
    report, candidate = stable.run(*measured, saved, bz=1.0117, **kwargs)
    return tuple(measured)+(saved, candidate), report


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a, b)


@pytest.fixture(scope='module')
def inputs():
    return chain(fixture())


@pytest.fixture(scope='module')
def result(inputs):
    return driver.run(*inputs[0], bz=1.0117)


def test_all_saved_arrays_readonly_independent_and_exact_batches(inputs, result):
    args, before = copy.deepcopy(inputs[0]), copy.deepcopy(inputs[0])
    for mapping in args:
        for value in mapping.values() if isinstance(mapping, dict) else ():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
    report, payload = driver.run(*args, bz=1.0117, group_batch=1)
    assert set(payload) == set(result[1])
    for key in payload:
        exact(payload[key], result[1][key])
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
    assert report['zero_hit_events'] == 2 and report['assigned_directions'] == 0
    assert report['max_rounds'] == 3 and report['continuous_max_iterations'] == 40
    assert report['continuous_max_backtracks'] == 8
    assert sum(report['lr_status_counts'].values()) == 6
    assert not any(x in k for k in payload if k.startswith(('lr_', 'source_lr_'))
                   for x in ('momentum', 'covariance', 'winner', 'p_abs'))
    json.dumps(report, allow_nan=False)


def test_saved_finals_not_original_seeds_and_source_trace_mapping(inputs, monkeypatch):
    original, captured = kernel.fit_lr_drift_helix, []
    def spy(**kw):
        out = original(**kw)
        captured.append((kw, out))
        return out
    monkeypatch.setattr(kernel, 'fit_lr_drift_helix', spy)
    _, payload = driver.run(*inputs[0], bz=1.0117)
    kw, raw = captured[0]
    saved, candidate = inputs[0][-2:]
    assert not np.array_equal(candidate['candidate_local_parameters'][1], saved['drift_initial_local_parameters'][1])
    for name in ('local_parameters', 't0', 'parameter_scale'):
        exact(kw[name], candidate['candidate_'+name].reshape(kw[name].shape))
    exact(kw['seed_valid'], saved['profile_profile_valid'].reshape(-1))
    assert kw['max_rounds'] == 3 and kw['max_iterations'] == 40
    for key, value in raw.items():
        if key.startswith('hit_'):
            expected = np.moveaxis(value[2:4], 0, 1)
            if key.startswith('hit_trace_'):
                expected = value[2:4].transpose(2, 0, 1)
            exact(payload['source_lr_'+key[4:]], expected)
    assert payload['source_lr_trace_lr_before'].shape == (12, 2, 3)
    assert payload['lr_trace_before_objective'].shape == (3, 2, 3)


def test_actual_objective_monotone_refit_only_changed_and_same_combined_criterion(inputs, result):
    report, p = result
    accepted = p['lr_trace_flip_accepted']
    assert np.all(p['lr_trace_postflip_objective'][accepted] < p['lr_trace_before_objective'][accepted])
    exact(p['lr_trace_refit_attempted'], accepted)
    active = p['lr_trace_active']
    assert np.all(p['lr_trace_postrefit_objective'][active] <= p['lr_trace_postflip_objective'][active])
    valid = p['lr_objective_valid'] & inputs[0][-1]['candidate_objective_valid']
    assert np.all(p['lr_final_objective'][valid] <= inputs[0][-1]['candidate_final_objective'][valid])
    exact(p['lr_initial_combined_checked'], p['candidate_success'] & p['lr_initial_lr_stationary'])
    exact(p['lr_success'], p['lr_certified'] & p['lr_lr_stationary'] & ~p['lr_refit_error'] & ~p['lr_refit_numerical_failure'])
    assert report['saved_combined_checked']['hypotheses'] == int(p['lr_initial_combined_checked'].sum())
    assert report['final_combined_checked']['hypotheses'] == int(p['lr_success'].sum())
    assert 'no proposal/contract error or sticky refit numerical failure' in report['combined_criterion']
    assert report['continuous_refits'] == int(accepted.sum())
    assert np.all(p['lr_continuous_refits'] <= 3)


def test_measured_time_variation_really_flips_and_refits_failed_finals(monkeypatch):
    measured = fixture()
    for name in ('et0', 'et1'):
        measured[1][name][::3] -= 30.
    args, _ = chain(measured, max_iterations=0)
    original, seen = kernel.fit_stable_drift_helix, []
    def spy(**kw):
        seen.append(kw)
        return original(**kw)
    monkeypatch.setattr(kernel, 'fit_stable_drift_helix', spy)
    report, p = driver.run(*args, bz=1.0117)
    assert not args[-1]['candidate_success'].any()
    assert report['initial_pending_lr_hypotheses'] > 0
    assert seen and report['continuous_refits'] == sum(len(k['t0']) for k in seen)
    assert all(k['max_iterations'] == 40 and k['max_backtracks'] == 8 for k in seen)
    accepted = p['lr_trace_flip_accepted']
    assert np.all(p['lr_trace_postflip_objective'][accepted] < p['lr_trace_before_objective'][accepted])
    exact(p['lr_trace_refit_attempted'], accepted)


def test_zero_rounds_no_refit_and_identical_source_state(inputs, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('zero rounds cannot refit')
    monkeypatch.setattr(kernel, 'fit_stable_drift_helix', forbidden)
    report, p = driver.run(*inputs[0], bz=1.0117, max_rounds=0)
    candidate = inputs[0][-1]
    for name in ('local_parameters', 't0', 'final_objective', 'objective_valid', 'certificate_status', 'certified'):
        exact(p['lr_'+name], candidate['candidate_'+name])
    for name in ('initial_lr', 'mask', 'signed_doca', 'r_drift', 'residuals', 'final_poca_correction'):
        exact(p['source_lr_'+name], candidate['source_candidate_'+name])
    exact(p['source_lr_final_lr'], candidate['source_candidate_initial_lr'])
    assert p['source_lr_trace_lr_before'].shape == (12, 2, 0)
    assert report['continuous_refits'] == report['accepted_lr_flip_operations'] == 0


@pytest.mark.parametrize('invalid', [False, True])
def test_failures_included_without_success_selection(invalid, monkeypatch):
    measured = fixture()
    if invalid:
        for key in ('fit_success', 'fit_converged', 'fit_physical_identifiable'):
            measured[4][key][1] = False
        measured[4]['fit_status'] = np.array(['empty', 'invalid_seed', 'empty'])
    args, _ = chain(measured, max_iterations=1)
    assert not args[-1]['candidate_success'].any()
    original, seen = kernel.fit_lr_drift_helix, []
    def spy(**kw):
        seen.append(kw['seed_valid'])
        return original(**kw)
    monkeypatch.setattr(kernel, 'fit_lr_drift_helix', spy)
    report, p = driver.run(*args, bz=1.0117)
    exact(np.concatenate(seen), args[-2]['profile_profile_valid'].reshape(-1))
    assert p['lr_initial_objective_valid'].sum() == (0 if invalid else 2)
    assert not p['lr_success'][~args[-1]['candidate_objective_valid']].any()
    assert sum(v['hypotheses'] for v in report['by_saved_status'].values()) == 6


@pytest.mark.parametrize('zero_groups', [False, True])
def test_empty_and_zero_groups(zero_groups):
    measured = list(fixture())
    measured[1] = {k: v[:0] for k, v in measured[1].items()}
    assoc, fit = measured[3:5]
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
    args, _ = chain(measured)
    report, p = driver.run(*args, bz=1.0117)
    assert report['groups'] == (0 if zero_groups else 3)
    assert report['events'] == report['zero_hit_events'] == 3
    assert report['hits'] == report['fixed_selected_hits'] == 0
    assert p['source_lr_trace_lr_before'].shape == (0, 2, 3)
    assert p['lr_status'].shape == (report['groups'], 2)
    json.dumps(report, allow_nan=False)


def test_unselected_source_hit_kept_with_sentinels(result):
    measured = list(fixture())
    measured[1] = {k: np.concatenate((v, v[:1])) for k, v in measured[1].items()}
    for mapping in measured[3:5]:
        for key in tuple(mapping):
            if key.startswith(('source_', 'association_source_')) or key == 'hit_group':
                mapping[key] = np.concatenate((mapping[key], mapping[key][:1]))
        mapping['hit_group'][-1] = -1
    measured[3]['association_source_used'][-1] = measured[4]['source_selected'][-1] = False
    args, _ = chain(measured)
    report, p = driver.run(*args, bz=1.0117)
    assert report['hits'] == 13 and report['fixed_selected_hits'] == 12
    for key, value in p.items():
        if key.startswith('source_lr_'):
            exact(value[:-1], result[1][key])
            if value.dtype.kind == 'f':
                assert np.isnan(value[-1]).all()
            else:
                assert (value[-1] == (-1 if '_bin' in key else 0)).all()


@pytest.mark.parametrize('key', ['source_candidate_signed_doca', 'source_candidate_residuals',
    'source_candidate_time_bin', 'candidate_certificate_status', 'candidate_certificate_same_branch',
    'candidate_certificate_column_relative_error', 'candidate_success', 'candidate_status',
    'candidate_local_parameters', 'candidate_t0', 'candidate_final_objective', 'candidate_gradient_norm',
    'source_candidate_initial_lr', 'candidate_parameter_scale', 'candidate_z_ref', 'candidate_flight_direction',
    'drift_initial_local_parameters', 'drift_initial_t0', 'candidate_initial_local_parameters', 'candidate_initial_t0',
    'source_candidate_final_poca_correction', 'source_candidate_final_poca_g', 'source_candidate_final_poca_h',
    'candidate_equilibrated_gradient', 'candidate_certificate_column_norm_ratio'])
def test_replay_tamper_rejected(inputs, key):
    args = copy.deepcopy(inputs[0])
    a = args[-1][key]
    if a.dtype.kind == 'b':
        a[1] ^= True
    elif a.dtype.kind in 'US':
        a[1] = 'tampered' if key != 'candidate_status' else 'converged_certified'
        if key == 'candidate_status':
            a[1] = np.where(args[-1]['candidate_success'][1], 'tampered', 'converged_certified')
    else:
        a[1] = np.nan_to_num(a[1]) + 1
    if key.startswith('drift_initial_'):
        args[-2][key] = a.copy()
    with pytest.raises(ValueError):
        driver.run(*args, bz=1.0117, max_rounds=0)


@pytest.mark.parametrize('kw', [{'group_batch': 0}, {'group_batch': True}, {'max_rounds': -1},
    {'max_rounds': 9}, {'max_rounds': True}, {'max_iterations': 41}, {'max_iterations': -1}])
def test_budget_bounds(inputs, kw):
    with pytest.raises(ValueError):
        driver.run(*inputs[0], bz=1.0117, **kw)


def test_support_bound_and_continuation_not_input(inputs, result, monkeypatch):
    with pytest.raises(ValueError, match='fixed-LR'):
        driver.run(*inputs[0][:-1], result[1], bz=1.0117)
    original = driver.prepare_saved
    def wide(*a):
        data = original(*a)
        data['count'] = np.full(3, 257)
        return data
    monkeypatch.setattr(driver, 'prepare_saved', wide)
    with pytest.raises(ValueError, match='support bound'):
        driver.run(*inputs[0], bz=1.0117)


def test_cli_hash_guards_no_overwrite_and_exact_comparison(inputs, result, tmp_path, monkeypatch):
    saved, candidate = tmp_path/'saved', tmp_path/'candidate'
    original = dict(nominal_axial_bz_T=1.0117, mass_hypothesis_MeV=105.6583755)
    for directory, payload in ((saved, inputs[0][-2]), (candidate, inputs[0][-1])):
        directory.mkdir()
        np.savez(directory/'diagnostics.npz', **payload)
    (saved/'report.json').write_text(json.dumps(original))
    measured = tmp_path/'measured'
    measured.write_text('synthetic input')
    history = {'historical.py': {'path': 'historical.py', 'sha256': 'historical'}}
    cr = copy.deepcopy(inputs[1])
    cr.update(current_implementation=history, inputs={'hits': {'sha256': driver._sha256(measured)}},
              saved_baseline={k: {'sha256': driver._sha256(saved/n)} for k, n in
                              [('report', 'report.json'), ('payload', 'diagnostics.npz')]})
    (candidate/'report.json').write_text(json.dumps(cr))
    calls = []
    def loader(directory):
        calls.append(directory)
        return inputs[0][:-1], original, {'hits': measured}
    monkeypatch.setattr(driver.prior, 'load_inputs', loader)
    monkeypatch.setattr(driver, 'run', lambda *a, **kw: (copy.deepcopy(result[0]), result[1]))
    base = ['--saved', str(saved), '--candidate', str(candidate), '--outdir']
    with pytest.raises(FileExistsError):
        driver.main(base+[str(tmp_path)])
    assert not calls
    out = tmp_path/'out'
    report = driver.main(base+[str(out)])
    assert report['historical_candidate_implementation'] == history
    for record in report['current_implementation'].values():
        assert record['sha256'] == driver._sha256(Path(record['path']))
    assert report['output_payload_sha256'] == driver._sha256(out/'diagnostics.npz')
    assert report['inputs']['candidate_payload']['sha256'] == driver._sha256(candidate/'diagnostics.npz')
    assert json.loads((candidate/'report.json').read_text()) == json.loads(json.dumps(cr))
    again = driver.main(base+[str(tmp_path/'again'), '--compare-saved', str(out/'diagnostics.npz')])
    assert again['saved_comparison']['exact']
    modified = dict(result[1], lr_t0=result[1]['lr_t0'].copy())
    modified['lr_t0'][1] += 1
    np.savez(tmp_path/'tamper.npz', **modified)
    with pytest.raises(ValueError, match='repeat differs'):
        driver.main(base+[str(tmp_path/'bad'), '--compare-saved', str(tmp_path/'tamper.npz')])
    assert not (tmp_path/'bad').exists()
    for section, key in [('saved_baseline', 'report'), ('saved_baseline', 'payload'), ('inputs', 'hits')]:
        bad = copy.deepcopy(cr)
        bad[section][key]['sha256'] = 'wrong'
        (candidate/'report.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='hash differs'):
            driver.main(base+[str(tmp_path/'bad')])
        assert not (tmp_path/'bad').exists()


def test_multiple_unequal_groups_and_reordered_source_identity_exact_batches():
    measured = list(fixture())
    sh, assoc, fit = measured[1], measured[3], measured[4]
    for key in tuple(sh):
        sh[key] = np.concatenate((sh[key], sh[key][:8]))
    sh['ev'][-8:] = 2
    for name in ('et0', 'et1'):
        sh[name][::3] -= 30.
    for mapping in (assoc, fit):
        for key in tuple(mapping):
            if key.startswith(('source_', 'association_source_')) or key == 'hit_group':
                mapping[key] = np.concatenate((mapping[key], mapping[key][:8]))
        mapping['source_hit_ev'][-8:] = 2
        mapping['hit_group'][-8:] = 2
    assoc['association_group_support'][2] = fit['group_support'][2] = 8
    assoc['associated_seed_success'][2] = True
    for key in ('fit_success', 'fit_converged', 'fit_physical_identifiable'):
        fit[key][2] = True
    fit['fit_status'][2] = 'converged'
    order = np.r_[np.arange(11, -1, -1), np.arange(19, 11, -1)]
    for key in sh:
        sh[key] = sh[key][order]
    for mapping in (assoc, fit):
        for key in tuple(mapping):
            if key.startswith(('source_', 'association_source_')) or key == 'hit_group':
                mapping[key] = mapping[key][order]
    args, _ = chain(measured, max_iterations=0)
    report, whole = driver.run(*args, bz=1.0117)
    _, single = driver.run(*args, bz=1.0117, group_batch=1)
    assert report['fixed_full_width'] == 12 and report['fixed_selected_hits'] == 20
    assert report['continuous_refits'] > 0
    assert not args[-1]['candidate_success'].any()
    assert whole['source_lr_trace_lr_after'].shape == (20, 2, 3)
    for key in whole:
        exact(whole[key], single[key])
    exact(whole['source_hit_sid'], sh['sid'])
