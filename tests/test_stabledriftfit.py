"""Analytic candidate tests without campaign truth or changed baseline defaults."""
import copy

import numpy as np
import pytest

from test_helixdriftfit import model, synthetic, perturb
from test_driftfitdiagnostic import saved, affine_model
from trkfastsim.reco import stabledriftfit as stable
from trkfastsim.reco import driftjacobian
from trkfastsim.reco.driftfitdiagnostic import _Evaluator


def args_from_original(inputs):
    args, baseline = saved(inputs, budget=0)
    return args, baseline


@pytest.mark.parametrize('tilted', [False, True])
def test_noiseless_recovery_original_seeds_fixed_lr(model, tilted):
    inputs, truth = synthetic(model, tilted=tilted)
    perturb(inputs)
    args, baseline = args_from_original(inputs)
    before = copy.deepcopy(args)
    actual = stable.fit_stable_drift_helix(**args)
    assert actual['success'].all(), (actual['status'], actual['certificate_status'], actual['gradient_norm'])
    assert actual['analytic_stationary'].all() and actual['certified'].all()
    assert (actual['status'] == 'converged_certified').all()
    np.testing.assert_allclose(actual['local_parameters'], truth['local_parameters'], atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual['t0'], truth['t0'], atol=2e-6, rtol=0)
    np.testing.assert_array_equal(actual['initial_objective'], baseline['initial_objective'])
    np.testing.assert_array_equal(actual['initial_lr'], args['initial_lr'])
    np.testing.assert_array_equal(actual['mask'], args['valid'])
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(args[key], value)
    used = actual['trace_accepted_backtrack'] >= 0
    indices = np.maximum(actual['trace_accepted_backtrack'], 0)[..., None]
    deltas = np.take_along_axis(actual['trace_trial_delta'], indices, axis=2)[..., 0]
    assert (deltas[used] < 0).all()
    assert not any(k in actual for k in ('p_abs', 'beta', 'covariance', 'charge'))


def test_zero_budget_nonstationary_is_failure(model):
    inputs, _ = synthetic(model)
    perturb(inputs)
    args, _ = args_from_original(inputs)
    actual = stable.fit_stable_drift_helix(**args, max_iterations=0)
    assert not actual['success'].any()
    assert (actual['status'] == 'max_iterations').all()
    assert not actual['iterations'].any()
    np.testing.assert_array_equal(actual['local_parameters'], args['local_parameters'])
    np.testing.assert_array_equal(actual['t0'], args['t0'])


def test_empty_invalid_and_failed_lanes_kept(model):
    inputs, _ = synthetic(model, batch=3)
    perturb(inputs)
    args, _ = args_from_original(inputs)
    args['valid'][0] = False
    args['seed_valid'][1] = False
    actual = stable.fit_stable_drift_helix(**args, max_iterations=1)
    np.testing.assert_array_equal(actual['status'], ['empty', 'invalid_seed', 'max_iterations'])
    assert not actual['success'].any()
    np.testing.assert_array_equal(actual['mask'], args['valid'])
    assert np.isnan(actual['final_objective'][:2]).all()
    assert actual['final_objective'][2] < actual['initial_objective'][2]


@pytest.mark.parametrize('shape', [(0, 17), (2, 0)])
def test_empty_shapes(model, shape):
    inputs, _ = synthetic(model)
    args, _ = args_from_original(inputs)
    b, h = shape
    for key, value in args.items():
        if isinstance(value, np.ndarray):
            args[key] = value[:b, :h] if key in ('wire_cal', 'wire_hv', 'early_end', 'early_time', 'signal_speed',
                                                'measured_u', 'uvar', 'valid', 'initial_lr') else value[:b]
    actual = stable.fit_stable_drift_helix(**args)
    assert actual['success'].shape == (b,)
    assert not actual['success'].any()
    assert actual['residuals'].shape == (b, h, 2)


def test_exact_batching_of_state_and_trace(model):
    inputs, _ = synthetic(model, batch=3)
    perturb(inputs)
    args, _ = args_from_original(inputs)
    args['seed_valid'][1] = False
    whole = stable.fit_stable_drift_helix(**args)
    parts = []
    for lo, hi in ((0, 1), (1, 2), (2, 3)):
        chunk = {k: a[lo:hi] if isinstance(a, np.ndarray) else a for k, a in args.items()}
        parts.append(stable.fit_stable_drift_helix(**chunk))
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate([p[key] for p in parts]), err_msg=key)


@pytest.mark.parametrize('limits', [dict(max_iterations=201), dict(max_backtracks=17), dict(max_backtracks=0),
                                    dict(max_rows=1), dict(max_cells=1), dict(gradient_tolerance=0),
                                    dict(initial_damping=1e13)])
def test_bounds_never_truncate(model, limits):
    inputs, _ = synthetic(model)
    args, _ = args_from_original(inputs)
    with pytest.raises(ValueError):
        stable.fit_stable_drift_helix(**args, **limits)


def test_common_norm_and_column_checks_catch_scaled_jacobian(model):
    inputs, _ = synthetic(model)
    args, _ = args_from_original(inputs)
    evaluator = _Evaluator(**args)
    p = evaluator.candidate
    center = evaluator(p)
    lin = driftjacobian.linearize(evaluator, p, center)
    bad = dict(lin, jacobian=2*lin['jacobian'], norm=2*lin['norm'])
    cert = stable.certify(evaluator, p, center, bad)
    assert not cert['certified'].any()
    good = cert['certificate_numeric_good'] & cert['certificate_same_branch']
    assert good.any()
    np.testing.assert_allclose(cert['certificate_column_norm_ratio'][good], .5, atol=1e-4)
    assert (cert['certificate_column_relative_error'][good] > .49).all()


def test_invalid_trial_never_accepted_and_trace_explains_stall(model, monkeypatch):
    inputs, _ = synthetic(model)
    perturb(inputs)
    args, _ = args_from_original(inputs)
    original = stable._Evaluator.__call__
    initial = np.column_stack((args['local_parameters'], args['t0']))

    def blocked(self, parameters):
        out = original(self, parameters)
        out['objective_valid'] &= np.all(parameters == initial, axis=1)
        return out

    monkeypatch.setattr(stable._Evaluator, '__call__', blocked)
    actual = stable.fit_stable_drift_helix(**args, initial_damping=1e12)
    assert not actual['success'].any()
    assert (actual['termination_status'] == 'stalled').all()
    assert (actual['trace_trial_evaluated'][:, 0].sum(axis=1) == 8).all()
    assert not actual['trace_trial_valid'][:, 0].any()
    assert not actual['accepted_steps'].any()
    np.testing.assert_array_equal(actual['local_parameters'], args['local_parameters'])


def test_eigensolve_failure_not_success(model, monkeypatch):
    inputs, _ = synthetic(model)
    args, _ = args_from_original(inputs)

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('synthetic failure')

    monkeypatch.setattr(np.linalg, 'eigh', fail)
    actual = stable.fit_stable_drift_helix(**args)
    assert not actual['success'].any()
    assert (actual['status'] == 'linearization_failed').all()
    assert not actual['accepted_steps'].any()


def test_nonfinite_seed_retains_other_original_parameters(model):
    inputs, _ = synthetic(model)
    args, _ = args_from_original(inputs)
    args['local_parameters'][0, 2] = np.nan
    actual = stable.fit_stable_drift_helix(**args)
    assert not actual['success'][0]
    np.testing.assert_array_equal(actual['local_parameters'][0], args['local_parameters'][0])
    assert actual['t0'][0] == args['t0'][0]


@pytest.mark.parametrize('allowed,expected', [([True, False, True, False], False),
                                             ([False, True, True, False], True)])
def test_certificate_requires_adjacent_passing_scales(model, monkeypatch, allowed, expected):
    inputs, _ = synthetic(model)
    args, _ = args_from_original(inputs)
    evaluator = _Evaluator(**args)
    p, center = evaluator.candidate, evaluator(evaluator.candidate)
    lin = driftjacobian.linearize(evaluator, p, center)
    calls = 0

    def force_branches(*args):
        nonlocal calls
        index = calls//12
        calls += 1
        return np.full(len(p), allowed[index])

    real = stable._linearize

    def controlled(parameters, scale, evaluate, residual, fd, condition):
        out = real(parameters, scale, evaluate, residual, fd, condition)
        out.update(good=np.ones(len(p), bool), g=lin['g'].copy(), norm=lin['norm'].copy(),
                   gradient_norm=lin['gradient_norm'].copy())
        return out

    monkeypatch.setattr(stable, '_same_branch', force_branches)
    monkeypatch.setattr(stable, '_linearize', controlled)
    cert = stable.certify(evaluator, p, center, lin)
    assert np.all(cert['certified'] == expected)


def test_valid_nondecreasing_trials_retained_as_failure(model, monkeypatch):
    inputs, _ = synthetic(model)
    perturb(inputs)
    args, _ = args_from_original(inputs)
    original = stable._Evaluator.__call__
    evaluator = _Evaluator(**args)
    objective = evaluator(evaluator.candidate)['objective']

    def flat(self, parameters):
        out = original(self, parameters)
        out['objective'] = objective.copy()
        return out

    monkeypatch.setattr(stable._Evaluator, '__call__', flat)
    actual = stable.fit_stable_drift_helix(**args, initial_damping=1e12)
    assert (actual['termination_status'] == 'stalled').all()
    assert not actual['success'].any()
    assert actual['trace_trial_valid'][:, 0].all()
    assert (actual['trace_trial_delta'][:, 0] == 0).all()
    assert not actual['accepted_steps'].any()


def test_angle_fold_signs_distinguish_equal_folded_cells(model):
    inputs, _ = synthetic(model, batch=1)
    args, _ = args_from_original(inputs)
    ev = _Evaluator(**args)
    p = ev.candidate
    state = ev(p)
    first = stable._angle_branch(ev, p, state)
    # Reflection about the field axis flips qr while leaving the folded angle equal.
    from trkfastsim.reco.helixfit import _trajectory
    point, _, _ = _trajectory(p[:, :5], ev.zref, state['poca_z'])
    s = np.where((ev.end == 0)[..., None], ev.direction, -ev.direction)
    b = np.array([0., 0., np.sign(ev.bz)])
    radial = np.cross(s, b)
    late = np.where((ev.end == 0)[..., None], ev.hv, ev.cal)
    sensor = late+np.sum((point-late)*s, axis=-1)[..., None]*s
    q = point-sensor
    reflected = q-2*np.sum(q*radial, axis=-1)[..., None]*radial/np.sum(radial*radial, axis=-1)[..., None]
    opposite = np.stack((np.sign(np.sum(reflected*radial, axis=-1)),
                         np.sign(np.sum(reflected*b, axis=-1))), axis=-1)
    assert np.all(first[..., 0] == -opposite[..., 0])
    np.testing.assert_allclose(np.arctan2(np.abs(np.sum(q*radial, axis=-1)), np.abs(np.sum(q*b, axis=-1))),
                               np.arctan2(np.abs(np.sum(reflected*radial, axis=-1)), np.abs(np.sum(reflected*b, axis=-1))), atol=1e-14)


def test_noisy_affine_fit_independent_scalar_least_squares(model):
    from scipy.optimize import least_squares
    from test_driftjacobian import independent_residual, nonconstant
    m = nonconstant(affine_model(model))
    inputs, _ = synthetic(m, batch=1)
    h = np.arange(inputs['valid'].shape[1])
    inputs['early_time'] += .02*np.sin(.7*h)
    inputs['measured_u'] += .03*np.cos(.9*h)
    perturb(inputs)
    args, _ = args_from_original(inputs)
    actual = stable.fit_stable_drift_helix(**args)
    assert actual['success'].all(), (actual['status'], actual['certificate_status'])
    assert actual['final_objective'][0] > 0
    ev = _Evaluator(**args)
    initial = ev.candidate.copy()

    def scalar_residual(scaled):
        return independent_residual(ev, initial+scaled[None, :]*ev.scale).ravel()

    reference = least_squares(scalar_residual, np.zeros(6), xtol=1e-11, ftol=1e-11, gtol=1e-9,
                              diff_step=1e-4, max_nfev=100)
    assert reference.success
    final = np.column_stack((actual['local_parameters'], actual['t0']))
    np.testing.assert_allclose(.5*np.sum(scalar_residual((final-initial)[0]/ev.scale[0])**2),
                               .5*np.sum(reference.fun**2), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(final, initial+reference.x[None, :]*ev.scale, rtol=0, atol=2e-5)


def test_exact_calibration_jump_is_not_certified(model):
    from dataclasses import replace
    m = affine_model(model)
    inputs, _ = synthetic(m, batch=1)
    args, _ = args_from_original(inputs)
    ev = _Evaluator(**args)
    center = ev(ev.candidate)
    edge = center['c_drift'][0, 0]
    cal = replace(m.offset, low=edge-1, width=1., intercept=np.array([0., .01]), slope=np.zeros(2))
    args['model'] = replace(m, offset=cal)
    actual = stable.fit_stable_drift_helix(**args, max_iterations=0)
    assert not actual['success'].any()
    assert not actual['analytic_differentiable'].any()
    assert (actual['status'] == 'nondifferentiable').all()
    assert (actual['certificate_status'] == 'analytic_nondifferentiable').all()
