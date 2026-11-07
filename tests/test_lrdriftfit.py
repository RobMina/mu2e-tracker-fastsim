"""Synthetic, fixed-support tests of bounded saved-final LR continuation."""
import copy
import inspect
import itertools
from dataclasses import replace

import numpy as np
import pytest

from test_helixdriftfit import model, synthetic, perturb  # noqa: F401
from test_driftfitdiagnostic import saved, affine_model
from trkfastsim.reco import lrdriftfit as lrfit
from trkfastsim.reco import stabledriftfit as stable
from trkfastsim.reco.driftfitdiagnostic import _Evaluator


def arguments(model, batch=2, wrong=False, noisy=False):
    inputs, _ = synthetic(model, batch=batch)
    if noisy:
        perturb(inputs)
    args, _ = saved(inputs)
    if wrong:
        args['initial_lr'][:, ::3] *= -1
    return args


def slice_args(args, selected):
    return {k: v[selected] if isinstance(v, np.ndarray) else v for k, v in args.items()}


def forbid(*args, **kwargs):
    raise AssertionError('no continuous refit allowed')


def test_signature_defaults_and_all_outputs_are_row_arrays(model):
    signature = inspect.signature(lrfit.fit_lr_drift_helix)
    other = inspect.signature(stable.fit_stable_drift_helix)
    assert list(signature.parameters) == list(other.parameters)+['max_rounds']
    for k in other.parameters:
        assert signature.parameters[k] == other.parameters[k]
    assert signature.parameters['max_rounds'].default == 3
    out = lrfit.fit_lr_drift_helix(**arguments(model), max_rounds=0)
    for value in out.values():
        assert isinstance(value, np.ndarray) and value.shape[0] == 2
    assert not {'p_abs', 'beta', 'covariance', 'charge', 'mask', 'initial_lr'} & out.keys()
    assert out['trace_active'].shape == (2, 0)


def test_enumerate_all_lr_choices_on_fixed_geometry(model):
    args = arguments(model, batch=1)
    args['valid'][:, 4:] = False
    args['initial_lr'][:, :4] *= -1
    ev = _Evaluator(**args)
    initial = ev(ev.candidate)
    preferred, gain = lrfit._preferred(initial, args['initial_lr'], ev.mask)
    costs = []
    for signs in itertools.product((-1., 1.), repeat=4):
        choice = args['initial_lr'].copy()
        choice[:, :4] = signs
        residual = (choice[:, :4]*initial['r_drift'][:, :4]
                    -initial['signed_doca'][:, :4])/initial['signed_error'][:, :4]
        costs.append(.5*np.sum(residual**2+initial['long_pull'][:, :4]**2))
    optimum = .5*np.sum(((preferred[:, :4]*initial['r_drift'][:, :4]
                         -initial['signed_doca'][:, :4])/initial['signed_error'][:, :4])**2
                       +initial['long_pull'][:, :4]**2)
    assert optimum == min(costs)
    np.testing.assert_allclose(gain[:, :4].sum(), initial['objective'][0]-optimum, atol=1e-12)
    out = lrfit.fit_lr_drift_helix(**args, max_iterations=0, max_rounds=1)
    np.testing.assert_array_equal(out['hit_final_lr'], preferred)
    np.testing.assert_allclose(out['final_objective'], optimum, atol=1e-12)


def test_negative_radii_not_absolute(model):
    args = arguments(model, batch=1)
    args['t0'] += 80.
    out = lrfit.fit_lr_drift_helix(**args, max_iterations=0, max_rounds=1)
    assert (out['hit_r_drift'] < 0).all()
    np.testing.assert_array_equal(out['hit_final_lr'], -np.sign(out['hit_signed_doca']))
    assert out['accepted_lr_flips'][0] == args['valid'].sum()
    assert out['final_objective'][0] < out['initial_objective'][0]
    assert not out['success'][0]


def test_zero_radius_tie_no_refit(model, monkeypatch):
    args = arguments(model)
    flat = affine_model(model)
    args['model'] = replace(flat, distances_tbins=np.zeros_like(flat.distances_tbins))
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', forbid)
    out = lrfit.fit_lr_drift_helix(**args)
    np.testing.assert_array_equal(out['hit_final_lr'], args['initial_lr'])
    np.testing.assert_array_equal(out['hit_lr_gain'], 0.)
    assert not out['continuous_refits'].any()
    assert not out['success'].any()


def test_sign_factors_exact_zero_and_overflow():
    r = np.array([[0., -0., 1., -1., 1e308, -1e308, 1e-300]])
    d = np.array([[-2., 2., 0., -0., -1e308, 1e308, -1e-300]])
    current = np.ones_like(r)
    state = dict(r_drift=r, signed_doca=d, signed_error=np.ones_like(r), objective_valid=np.array([True]))
    preferred, gain = lrfit._preferred(state, current, np.ones(r.shape, bool))
    np.testing.assert_array_equal(preferred, [[1, 1, 1, 1, -1, -1, -1]])
    np.testing.assert_array_equal(gain[:, :4], 0.)
    assert np.isinf(gain[:, 4:6]).all()


@pytest.mark.parametrize('rounds', [0, 3])
def test_no_flip_no_refit_and_zero_round_certificate(model, monkeypatch, rounds):
    args = arguments(model)
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', forbid)
    out = lrfit.fit_lr_drift_helix(**args, max_rounds=rounds)
    assert out['success'].all(), out['status']
    assert out['lr_stationary'].all() and out['certified'].all()
    assert not out['continuous_refits'].any()
    assert (out['last_refit_status'] == 'not_attempted').all()
    assert (out['last_refit_termination_status'] == 'not_attempted').all()
    for k in ('local_parameters', 't0'):
        np.testing.assert_array_equal(out[k], args[k])


def test_lr_stationarity_is_not_continuous_stationarity(model, monkeypatch):
    args = arguments(model, noisy=True)
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', forbid)
    out = lrfit.fit_lr_drift_helix(**args)
    assert out['lr_stationary'].all()
    assert not out['analytic_stationary'].any()
    assert not out['success'].any()


def test_all_certificate_fields_replayed_unmodified(model):
    args = arguments(model)
    reference = stable.fit_stable_drift_helix(**args, max_iterations=0)
    out = lrfit.fit_lr_drift_helix(**args, max_rounds=0)
    for key in reference:
        if key.startswith('certificate_') or key in ('certified', 'analytic_stationary',
                   'analytic_differentiable', 'gradient_norm', 'equilibrated_gradient', 'rank', 'condition'):
            np.testing.assert_array_equal(out[key], reference[key], err_msg=key)


def test_stationary_without_final_certificate_cannot_pass(model, monkeypatch):
    real = lrfit.certify
    def fail(*args, **kwargs):
        out = real(*args, **kwargs)
        out['certified'][:] = False
        return out
    monkeypatch.setattr(lrfit, 'certify', fail)
    out = lrfit.fit_lr_drift_helix(**arguments(model), max_rounds=0)
    assert out['analytic_stationary'].all() and out['lr_stationary'].all()
    assert not out['success'].any()
    assert (out['status'] == 'derivative_unstable').all()


def test_monotone_continuation_from_valid_failed_finals_and_formula(model):
    args = arguments(model, noisy=True)
    failed = stable.fit_stable_drift_helix(**args, max_iterations=1)
    assert failed['objective_valid'].all() and not failed['success'].any()
    args.update(local_parameters=failed['local_parameters'], t0=failed['t0'])
    args['initial_lr'][:, ::3] *= -1
    before = copy.deepcopy(args)
    out = lrfit.fit_lr_drift_helix(**args)
    assert out['success'].all(), (out['status'], out['last_refit_status'])
    assert (out['continuous_refits'] > 0).all()
    accepted = out['trace_flip_accepted']
    assert (out['trace_postflip_objective'][accepted] < out['trace_before_objective'][accepted]).all()
    refits = out['trace_refit_accepted']
    assert (out['trace_postrefit_objective'][refits] < out['trace_postflip_objective'][refits]).all()
    used = out['trace_active']
    assert (out['trace_postrefit_objective'][used] <= out['trace_postflip_objective'][used]).all()
    np.testing.assert_allclose(out['trace_predicted_lr_gain'][accepted],
                (out['trace_before_objective']-out['trace_postflip_objective'])[accepted], rtol=1e-13, atol=1e-11)
    objective = .5*np.sum(out['hit_residuals']**2, axis=(1, 2))
    np.testing.assert_array_equal(out['final_objective'], objective)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(args[key], value, err_msg=key)
    for key in ('parameter_scale', 'flight_direction', 'z_ref'):
        np.testing.assert_array_equal(out[key], args[key])
    np.testing.assert_array_equal(out['hit_initial_lr'], before['initial_lr'])
    np.testing.assert_array_equal(out['hit_mask'], args['valid'])


def test_only_changed_rows_refit_with_frozen_support(model, monkeypatch):
    args = arguments(model, batch=3, noisy=True)
    args['initial_lr'][1, ::3] *= -1
    args['valid'][1, -1] = False
    calls = []
    real = lrfit.fit_stable_drift_helix
    def spy(**kw):
        calls.append(kw)
        assert len(kw['t0']) == 1
        for k in ('valid', 'parameter_scale', 'flight_direction', 'z_ref', 'wire_cal', 'wire_hv', 'seed_valid'):
            np.testing.assert_array_equal(kw[k], args[k][1:2])
        return real(**kw)
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', spy)
    out = lrfit.fit_lr_drift_helix(**args, max_rounds=1)
    assert len(calls) == 1
    np.testing.assert_array_equal(out['continuous_refits'], [0, 1, 0])
    np.testing.assert_array_equal(out['local_parameters'][[0, 2]], args['local_parameters'][[0, 2]])


@pytest.mark.parametrize('limits', [dict(max_rounds=-1), dict(max_rounds=9), dict(max_rounds=1.5),
    dict(max_iterations=41), dict(max_iterations=-1), dict(max_backtracks=9), dict(max_backtracks=0),
    dict(max_rows=1), dict(max_cells=1), dict(max_rows=1025), dict(max_cells=65537),
    dict(gradient_tolerance=1e-5), dict(condition_limit=1e13), dict(initial_damping=1e13)])
def test_hard_bounds(model, limits):
    with pytest.raises(ValueError):
        lrfit.fit_lr_drift_helix(**arguments(model), **limits)


@pytest.mark.parametrize('shape', [(0, 17), (2, 0)])
def test_zero_shapes_all_fields(model, shape):
    args = arguments(model)
    b, h = shape
    hit = ('wire_cal', 'wire_hv', 'early_end', 'early_time', 'signal_speed', 'measured_u', 'uvar', 'valid', 'initial_lr')
    args = {k: (v[:b, :h] if k in hit else v[:b]) if isinstance(v, np.ndarray) else v for k, v in args.items()}
    out = lrfit.fit_lr_drift_helix(**args)
    reference = lrfit.fit_lr_drift_helix(**arguments(model))
    assert reference.keys() == out.keys()
    assert out['hit_residuals'].shape == (b, h, 2)
    assert out['hit_trace_lr_before'].shape == (b, 3, h)
    for value in out.values():
        assert value.shape[0] == b
    assert not out['success'].any()


def test_exact_batching_invalid_lanes_and_partial_nan(model):
    args = arguments(model, batch=6, wrong=True, noisy=True)
    args['seed_valid'][1] = False
    args['valid'][2] = False
    args['local_parameters'][3, 2] = np.nan
    args['early_time'][4, 0] = np.nan
    out = lrfit.fit_lr_drift_helix(**args)
    parts = [lrfit.fit_lr_drift_helix(**slice_args(args, slice(i, i+1))) for i in range(6)]
    for key in out:
        np.testing.assert_array_equal(out[key], np.concatenate([p[key] for p in parts]), err_msg=key)
    np.testing.assert_array_equal(out['local_parameters'][1:5], args['local_parameters'][1:5])
    np.testing.assert_array_equal(out['t0'][1:5], args['t0'][1:5])
    np.testing.assert_array_equal(out['hit_final_lr'][1:5], args['initial_lr'][1:5])
    assert not out['objective_valid'][1:5].any()
    assert not out['continuous_refits'][1:5].any()


@pytest.mark.parametrize('mode', ['flat', 'invalid', 'overflow'])
def test_gain_only_proposal_never_authorizes_flip(model, monkeypatch, mode):
    args = arguments(model, wrong=True)
    real = lrfit._Evaluator.__call__
    ev = _Evaluator(**args)
    old = real(ev, ev.candidate)['objective']
    def blocked(self, p):
        state = real(self, p)
        changed = np.any(self.initial_lr != args['initial_lr'], axis=1)
        if mode == 'flat':
            state['objective'][changed] = old[changed]
        elif mode == 'invalid':
            state['objective_valid'][changed] = False
        else:
            state['objective'][changed] = np.inf
        return state
    monkeypatch.setattr(lrfit._Evaluator, '__call__', blocked)
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', forbid)
    out = lrfit.fit_lr_drift_helix(**args)
    assert (out['trace_predicted_lr_gain'][:, 0] > 0).all()
    assert not out['accepted_lr_flips'].any() and not out['success'].any()
    assert out['refit_error'].all()
    np.testing.assert_array_equal(out['hit_final_lr'], args['initial_lr'])


@pytest.mark.parametrize('mode', ['increasing', 'invalid', 'mismatch', 'changed_equal', 'support', 'numerical'])
def test_refit_rejection_preserves_valid_flip_and_flags_failure(model, monkeypatch, mode):
    args = arguments(model, wrong=True)
    real = lrfit.fit_stable_drift_helix
    def broken(**kw):
        kw['max_iterations'] = 0
        out = real(**kw)
        if mode in ('increasing', 'invalid', 'changed_equal'):
            out['t0'] += 5.
            if mode == 'invalid':
                out['local_parameters'][:, 2] = np.nan
            replay = _Evaluator(**{k: v for k, v in kw.items() if k in args})
            p = np.column_stack((out['local_parameters'], out['t0']))
            out['final_objective'] = replay(p)['objective']
            if mode == 'changed_equal':
                out['final_objective'] = out['initial_objective'].copy()
        elif mode == 'mismatch':
            out['final_objective'] += 1.
        elif mode == 'support':
            out['mask'] = ~out['mask']
        else:
            out['termination_status'][:] = 'linearization_failed'
            out['status'][:] = 'linearization_failed'
            out['success'][:] = False
        return out
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', broken)
    out = lrfit.fit_lr_drift_helix(**args)
    assert not out['success'].any()
    assert (out['refit_failure_count'] > 0).all()
    assert (out['accepted_lr_flips'] > 0).all()
    assert (out['continuous_refits'] == 1).all()
    assert (out['final_objective'] < out['initial_objective']).all()
    np.testing.assert_array_equal(out['local_parameters'], args['local_parameters'])
    np.testing.assert_array_equal(out['t0'], args['t0'])
    assert not out['trace_refit_accepted'].any()
    if mode == 'numerical':
        assert out['refit_numerical_failure'].all()
        assert (out['status'] == 'refit_numerical_failure').all()
    else:
        assert out['refit_error'].all()


def test_zero_round_pending_lr_is_not_success(model):
    args = arguments(model, wrong=True)
    out = lrfit.fit_lr_drift_helix(**args, max_rounds=0)
    assert (out['pending_lr_flips'] > 0).all()
    assert not out['success'].any()
    assert out['lr_budget_exhausted'].all()
    assert not out['continuous_refits'].any()
    np.testing.assert_array_equal(out['hit_final_lr'], args['initial_lr'])


def test_identical_refit_preserves_flip_without_error(model):
    out = lrfit.fit_lr_drift_helix(**arguments(model, wrong=True), max_iterations=0, max_rounds=1)
    assert out['success'].all()
    assert not out['trace_refit_accepted'].any()
    assert not out['refit_error'].any()
    np.testing.assert_array_equal(out['trace_postflip_objective'], out['trace_postrefit_objective'])


@pytest.mark.parametrize('rounds', [1, 2, 3])
def test_later_rounds_continue_uncertified_refit_and_final_pending_budget(model, monkeypatch, rounds):
    args = arguments(model, batch=1)
    target_t0 = args['t0'].copy()
    args['t0'] += 500.
    real = lrfit.fit_stable_drift_helix
    calls = []
    def crossing(**kw):
        ev = _Evaluator(**{k: v for k, v in kw.items() if k in args})
        initial = ev(ev.candidate)
        out = real(**dict(kw, t0=target_t0.copy(), max_iterations=0))
        out['initial_t0'] = kw['t0'].copy()
        out['initial_objective'] = initial['objective'].copy()
        calls.append(out['success'].copy())
        return out
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', crossing)
    out = lrfit.fit_lr_drift_helix(**args, max_rounds=rounds)
    assert not calls[0].any()
    assert out['trace_refit_accepted'][0, 0]
    assert out['final_objective'][0] < out['initial_objective'][0]
    assert out['refit_failure_count'][0] == 1
    if rounds == 1:
        assert out['pending_lr_flips'][0] == args['valid'].sum()
        assert not out['success'][0]
        assert out['status'][0] == 'pending_lr_flips'
        assert out['lr_budget_exhausted'][0] and out['trace_lr_budget'][0, -1]
        assert (out['hit_lr_gain'] > 0).all()
    else:
        assert len(calls) == 2
        assert out['accepted_lr_rounds'][0] == 2
        assert out['accepted_lr_flips'][0] == 2*args['valid'].sum()
        assert out['pending_lr_flips'][0] == 0
        assert out['success'][0]
        assert not out['hit_lr_changed'].any()


def test_unselected_nan_lr_is_not_a_change_or_refit_contract_failure(model):
    args = arguments(model, batch=2, wrong=True)
    args['valid'][:, -1] = False
    args['initial_lr'][:, -1] = np.nan
    args['early_time'][:, -1] = np.nan
    out = lrfit.fit_lr_drift_helix(**args)
    assert out['success'].all(), out['status']
    assert not out['hit_lr_changed'][:, -1].any()
    assert np.isnan(out['hit_final_lr'][:, -1]).all()
    assert np.isnan(out['hit_lr_gain'][:, -1]).all()
    np.testing.assert_array_equal(out['hit_residuals'][:, -1], 0.)


def test_invalid_selected_lr_nan_does_not_claim_change(model):
    args = arguments(model)
    args['initial_lr'][0, 0] = np.nan
    out = lrfit.fit_lr_drift_helix(**args)
    assert not out['objective_valid'][0]
    assert not out['hit_lr_changed'][0].any()
    np.testing.assert_array_equal(out['hit_final_lr'], args['initial_lr'])


def test_gain_overflow_is_diagnostic_not_an_acceptance_veto(model, monkeypatch):
    args = arguments(model, wrong=True)
    expected = lrfit.fit_lr_drift_helix(**args)
    real = lrfit._preferred
    def overflow(*a):
        preferred, gain = real(*a)
        return preferred, np.where(preferred != a[1], np.inf, gain)
    monkeypatch.setattr(lrfit, '_preferred', overflow)
    out = lrfit.fit_lr_drift_helix(**args)
    assert out['trace_proposal_valid'][:, 0].all()
    assert np.isinf(out['trace_predicted_lr_gain'][:, 0]).all()
    assert (out['trace_proposed_objective'][:, 0] < out['initial_objective']).all()
    assert not out['trace_invalid'][:, 0].any()
    for key in ('local_parameters', 't0', 'final_objective', 'hit_final_lr', 'success', 'accepted_lr_flips'):
        np.testing.assert_array_equal(out[key], expected[key], err_msg=key)


def test_refit_equal_actual_objective_but_changed_parameters_is_failure(model, monkeypatch):
    args = arguments(model, wrong=True)
    real_fit, real_eval = lrfit.fit_stable_drift_helix, lrfit._Evaluator.__call__
    flipped = np.sign(_Evaluator(**args)(_Evaluator(**args).candidate)['signed_doca'])
    ev = _Evaluator(**dict(args, initial_lr=flipped))
    postflip = ev(ev.candidate)['objective']
    def flat(self, p):
        state = real_eval(self, p)
        matching = np.all(self.initial_lr == flipped, axis=1)
        state['objective'][matching] = postflip[matching]
        return state
    def shifted(**kw):
        out = real_fit(**dict(kw, max_iterations=0))
        out['t0'] += .1
        return out
    monkeypatch.setattr(lrfit._Evaluator, '__call__', flat)
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', shifted)
    out = lrfit.fit_lr_drift_helix(**args)
    assert out['refit_error'].all()
    assert (out['status'] == 'refit_no_strict_decrease').all()
    np.testing.assert_array_equal(out['t0'], args['t0'])


def test_evaluator_instances_never_mutate_lr(model, monkeypatch):
    original = lrfit._Evaluator
    instances = []
    class Tracked(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            instances.append((self, self.lr.copy(), self.initial_lr.copy()))
    monkeypatch.setattr(lrfit, '_Evaluator', Tracked)
    out = lrfit.fit_lr_drift_helix(**arguments(model, wrong=True))
    assert out['success'].all()
    assert len(instances) >= 3
    for ev, frozen_lr, frozen_initial in instances:
        np.testing.assert_array_equal(ev.lr, frozen_lr)
        np.testing.assert_array_equal(ev.initial_lr, frozen_initial)


def test_improved_refit_with_numerical_failure_is_retained_but_not_promoted(model, monkeypatch):
    args = arguments(model, wrong=True, noisy=True)
    real = lrfit.fit_stable_drift_helix
    def failed(**kw):
        out = real(**kw)
        assert (out['final_objective'] < out['initial_objective']).all()
        out['termination_status'][:] = 'linearization_failed'
        out['status'][:] = 'linearization_failed'
        out['success'][:] = False
        return out
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', failed)
    out = lrfit.fit_lr_drift_helix(**args)
    assert out['trace_refit_accepted'][:, 0].all()
    assert out['refit_numerical_failure'].all() and not out['success'].any()
    assert (out['trace_postrefit_objective'][:, 0] < out['trace_postflip_objective'][:, 0]).all()
    assert (out['status'] == 'refit_numerical_failure').all()


@pytest.mark.parametrize('rejected', [False, True])
def test_final_certificate_is_for_retained_parameters_and_current_lr(model, monkeypatch, rejected):
    args = arguments(model, wrong=True, noisy=True)
    if rejected:
        real = lrfit.fit_stable_drift_helix
        def bad(**kw):
            out = real(**kw)
            out['t0'] += 50.
            return out
        monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', bad)
    out = lrfit.fit_lr_drift_helix(**args)
    ev = _Evaluator(**dict(args, local_parameters=out['local_parameters'], t0=out['t0'], initial_lr=out['hit_final_lr']))
    state = ev(ev.candidate)
    check = stable.certify(ev, ev.candidate, state, lrfit.linearize(ev, ev.candidate, state))
    for key in check:
        np.testing.assert_array_equal(out[key], check[key], err_msg=key)
    np.testing.assert_array_equal(out['final_objective'], state['objective'])
    assert (out['accepted_lr_flips'] > 0).all()
    assert out['refit_error'].all() == rejected


def test_noncontiguous_changed_rows_distinct_parameters_scatter_exactly(model):
    args = arguments(model, batch=5, noisy=True)
    args['initial_lr'][[0, 3], ::3] *= -1
    args['seed_valid'][2] = False
    args['local_parameters'][:, 0] += np.arange(5)*.02
    whole = lrfit.fit_lr_drift_helix(**args)
    assert (whole['continuous_refits'][[0, 3]] > 0).all()
    assert not whole['continuous_refits'][[1, 2, 4]].any()
    for i in range(5):
        part = lrfit.fit_lr_drift_helix(**slice_args(args, slice(i, i+1)))
        for key in whole:
            np.testing.assert_array_equal(whole[key][i:i+1], part[key], err_msg=key)


def test_refit_between_postflip_and_preflip_is_rejected(model, monkeypatch):
    args = arguments(model, wrong=True)
    real = lrfit.fit_stable_drift_helix
    pre = _Evaluator(**args)
    before = pre(pre.candidate)['objective']
    def worse(**kw):
        out = real(**dict(kw, max_iterations=0))
        out['t0'] += .05
        ev = _Evaluator(**{k: v for k, v in kw.items() if k in args})
        p = np.column_stack((out['local_parameters'], out['t0']))
        out['final_objective'] = ev(p)['objective']
        assert (out['initial_objective'] < out['final_objective']).all()
        assert (out['final_objective'] < before).all()
        return out
    monkeypatch.setattr(lrfit, 'fit_stable_drift_helix', worse)
    out = lrfit.fit_lr_drift_helix(**args)
    assert out['refit_error'].all() and not out['trace_refit_accepted'].any()
    np.testing.assert_array_equal(out['t0'], args['t0'])
    assert (out['final_objective'] < out['initial_objective']).all()
