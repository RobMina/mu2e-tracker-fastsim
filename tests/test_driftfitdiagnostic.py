"""Read-only diagnostic tests: independent synthetic geometry, no saved campaign data."""
from dataclasses import replace

import numpy as np
import pytest

from trkfastsim.reco import driftfitdiagnostic as diagnostic
from trkfastsim.reco.helixdriftfit import _linearize, fit_drift_helix
from test_helixdriftfit import model, perturb, synthetic  # noqa: F401


def saved(inputs, budget=0):
    fit = fit_drift_helix(**inputs, max_iterations=budget)
    args = dict(inputs, local_parameters=fit['local_parameters'], t0=fit['t0'],
                initial_lr=fit['initial_LR'], parameter_scale=fit['parameter_scale'])
    return args, fit


def assert_exact(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize('budget', [0, 1, 40])
def test_saved_objective_gradient_replay_including_failed_fits(model, budget):
    inputs, _ = synthetic(model)
    perturb(inputs)
    args, fit = saved(inputs, budget)
    actual = diagnostic.diagnose(**args)
    np.testing.assert_array_equal(actual['objective'], fit['final_objective'])
    np.testing.assert_array_equal(actual['objective_valid'], fit['objective_valid'])
    for name in ('residuals', 'doca', 'signed_doca', 'phi', 'delta_t', 'r_drift',
                 'signed_error', 'drift_residual', 'pull', 'long_pull', 'poca_z'):
        np.testing.assert_array_equal(actual[name], fit[name], err_msg=name)
    for name in ('gradient_norm', 'condition', 'rank'):
        np.testing.assert_array_equal(actual[name][:, 0], fit[name], err_msg=name)
    assert actual['linearization_good'].all()
    assert fit['success'].all() if budget == 40 else not fit['success'].any()
    assert 'success' not in actual and 'status' not in actual
    assert 'accepted_steps' not in actual and 'covariance' not in actual


def test_supplied_lr_is_frozen_and_negative_radius_preference(model):
    inputs, _ = synthetic(model, batch=1)
    args, fit = saved(inputs)
    args['t0'] = args['t0']+80.  # extrapolated, NEGATIVE radius is intentional
    args['initial_lr'] = -args['initial_lr']
    actual = diagnostic.diagnose(**args)
    assert actual['objective_valid'].all()
    assert (actual['r_drift'] < 0).all()
    np.testing.assert_array_equal(actual['initial_lr'], args['initial_lr'])
    np.testing.assert_array_equal(actual['lr_preferred'], -np.sign(actual['signed_doca']))
    assert np.max(np.abs(actual['residuals'][..., 0])) < 100
    flipped = diagnostic.evaluate(**dict(args, initial_lr=-args['initial_lr']))
    expected = .5*(actual['residuals'][..., 0]**2-flipped['residuals'][..., 0]**2)
    np.testing.assert_allclose(actual['lr_improvement'], expected, rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(actual['lr_gain'], np.maximum(expected, 0).sum(axis=1), atol=1e-11)
    # The original fit chose geometry signs; the diagnostic must not restore them.
    assert not np.array_equal(actual['initial_lr'], fit['initial_LR'])
    wrong = diagnostic.diagnose(**dict(args, initial_lr=-args['initial_lr']))
    assert (wrong['lr_improvement'] > 0).all()
    assert wrong['lr_flip_count'][0] == 17
    assert wrong['lr_gain'][0] > 0


def affine_model(model):
    """Exact linear time->radius and constant errors, independent of phi."""
    times = np.array([0., 100., 200.])
    inverse = np.broadcast_to((.04*times)[:, None], (3, model.times_dbins.shape[1])).copy()
    offset = replace(model.offset, low=0., width=.5,
                     intercept=np.zeros(8), slope=np.zeros(8))
    return replace(model, times=times, delta_t=100., distances_tbins=inverse,
                   offset=offset, ignore_phi=True)


def test_zero_radius_tie_retains_initial_lr(model):
    inputs, _ = synthetic(model, batch=1)
    args, _ = saved(inputs)
    # A zero-valued inverse grid and zero offset make every radius exactly zero.
    flat = affine_model(model)
    flat = replace(flat, distances_tbins=np.zeros_like(flat.distances_tbins))
    args['model'] = flat
    actual = diagnostic.diagnose(**args)
    assert actual['objective_valid'].all()
    assert (actual['r_drift'] == 0).all()
    np.testing.assert_array_equal(actual['lr_preferred'], args['initial_lr'])
    np.testing.assert_array_equal(actual['lr_improvement'], 0.)
    np.testing.assert_array_equal(actual['lr_gain'], 0.)
    np.testing.assert_array_equal(actual['lr_flip_count'], 0.)


def test_all_slopes_and_descent_replay_actual_normalized_objective(model):
    inputs, _ = synthetic(model, batch=1)
    perturb(inputs)
    args, _ = saved(inputs, 1)
    actual = diagnostic.diagnose(**args)
    p = np.column_stack((args['local_parameters'], args['t0']))

    def evaluate(candidate):
        return diagnostic.evaluate(**dict(args, local_parameters=candidate[:, :5], t0=candidate[:, 5]))

    center = evaluate(p)
    for s, fd in enumerate(diagnostic.FD_STEPS):
        lin = _linearize(p, args['parameter_scale'], evaluate, center['residuals'], fd, 1e12)
        for name in ('gradient_norm', 'condition', 'rank'):
            np.testing.assert_array_equal(actual[name][:, s], lin[name])
        np.testing.assert_array_equal(actual['equilibrated_gradient'][:, s], lin['g'])
        for coordinate in range(6):
            for side, sign in enumerate((-1, 1)):
                step = np.zeros_like(p)
                step[:, coordinate] = sign*fd*args['parameter_scale'][:, coordinate]
                trial = evaluate(p+step)
                np.testing.assert_array_equal(actual['perturbation_valid'][:, s, coordinate, side],
                                              trial['objective_valid'])
                np.testing.assert_array_equal(actual['objective_slopes'][:, s, coordinate, side],
                                              (trial['objective']-center['objective'])/(sign*fd))
    for s, fd_index in enumerate(diagnostic.DESCENT_FD_INDICES):
        direction = -actual['equilibrated_gradient'][:, fd_index]/actual['column_norm'][:, fd_index]
        direction /= np.max(np.abs(direction), axis=1)[:, None]
        np.testing.assert_array_equal(actual['descent_direction'][:, s], direction)
        for index, step in enumerate(diagnostic.DESCENT_STEPS):
            trial = evaluate(p+args['parameter_scale']*direction*step)
            np.testing.assert_array_equal(actual['descent_delta'][:, s, index],
                                          trial['objective']-center['objective'])
            np.testing.assert_array_equal(actual['descent_valid'][:, s, index], trial['objective_valid'])
    assert (actual['descent_delta'][0, :, -3:] < 0).all()


@pytest.mark.parametrize('field,value', [
    ('early_time', np.nan), ('signal_speed', 0.), ('uvar', -1.),
    ('initial_lr', 0.), ('initial_lr', np.nan), ('parameter_scale', 0.),
    ('parameter_scale', np.nan), ('local_parameters', np.nan),
    ('t0', np.nan), ('flight_direction', 0.), ('seed_valid', False)])
def test_invalid_rows_not_rescued_or_pruned(model, field, value):
    inputs, _ = synthetic(model)
    args, _ = saved(inputs)
    args[field] = args[field].copy()
    args[field][0] = value
    actual = diagnostic.diagnose(**args)
    np.testing.assert_array_equal(actual['objective_valid'], [False, True])
    assert np.isnan(actual['objective'][0])
    assert np.isnan(actual['residuals'][0]).all()
    assert np.isnan(actual['gradient_norm'][0]).all()
    assert not actual['numeric'][0].any()
    assert not actual['linearization_good'][0].any()
    assert not actual['perturbation_valid'][0].any()
    assert not actual['descent_valid'][0].any()
    assert np.isnan(actual['descent_delta'][0]).all()
    assert np.isnan(actual['lr_gain'][0]) and np.isnan(actual['lr_flip_count'][0])
    np.testing.assert_array_equal(actual['mask'], inputs['valid'])
    np.testing.assert_array_equal(actual['n_hits'], [17, 17])


def test_masked_payloads_are_not_observations_and_no_mutation(model):
    inputs, _ = synthetic(model)
    args, _ = saved(inputs)
    args['valid'][:, -2:] = False
    for name in ('wire_cal', 'wire_hv', 'early_time', 'signal_speed', 'measured_u', 'uvar', 'initial_lr'):
        args[name][:, -2:] = np.nan
    before = {key: value.copy() for key, value in args.items() if isinstance(value, np.ndarray)}
    for value in before.values():
        value.setflags(write=False)
    # Exercise read-only arrays, not just mutable input comparisons.
    args.update(before)
    actual = diagnostic.diagnose(**args)
    assert actual['objective_valid'].all()
    np.testing.assert_array_equal(actual['residuals'][:, -2:], 0.)
    assert np.isnan(actual['doca'][:, -2:]).all()
    assert np.isnan(actual['lr_improvement'][:, -2:]).all()
    np.testing.assert_array_equal(actual['objective'], .5*np.sum(actual['residuals']**2, axis=(1, 2)))
    for key, value in before.items():
        np.testing.assert_array_equal(args[key], value)
    actual['initial_lr'][0, 0] *= -1
    actual['local_parameters'][0, 0] += 3
    actual['mask'][0, 0] = False
    for key, value in before.items():
        np.testing.assert_array_equal(args[key], value)


@pytest.mark.parametrize('limits', [dict(max_rows=1), dict(max_cells=33),
                                   dict(max_rows=True), dict(max_cells=-1)])
def test_bounds_are_hard_errors(model, limits):
    inputs, _ = synthetic(model)
    args, _ = saved(inputs)
    with pytest.raises(ValueError):
        diagnostic.diagnose(**args, **limits)


@pytest.mark.parametrize('field,value', [('valid', 1), ('seed_valid', 1),
                                        ('parameter_scale', np.ones((2, 5))),
                                        ('initial_lr', np.ones((2, 16))), ('bz', 0.), ('mass', -1.)])
def test_shape_dtype_scalar_contract(model, field, value):
    inputs, _ = synthetic(model)
    args, _ = saved(inputs)
    args[field] = value
    with pytest.raises(ValueError):
        diagnostic.diagnose(**args)


def test_batching_array_exact_with_failed_and_invalid_rows(model):
    inputs, _ = synthetic(model, batch=3)
    perturb(inputs)
    args, fit = saved(inputs, 1)
    assert not fit['success'].any()
    args['seed_valid'][1] = False
    full = diagnostic.diagnose(**args)
    pieces = []
    for start, stop in ((0, 1), (1, 3)):
        part = {key: value[start:stop] if isinstance(value, np.ndarray) else value
                for key, value in args.items()}
        pieces.append(diagnostic.diagnose(**part, max_rows=stop-start, max_cells=(stop-start)*17))
    joined = {key: np.concatenate([piece[key] for piece in pieces]) for key in full}
    assert_exact(full, joined)


@pytest.mark.parametrize('batch,hits', [(0, 17), (2, 0), (0, 0)])
def test_empty_dimensions(model, batch, hits):
    args = dict(local_parameters=np.zeros((batch, 5)), z_ref=np.zeros(batch), t0=np.zeros(batch),
                flight_direction=np.ones(batch), wire_cal=np.zeros((batch, hits, 3)),
                wire_hv=np.zeros((batch, hits, 3)), early_end=np.zeros((batch, hits)),
                early_time=np.zeros((batch, hits)), signal_speed=np.ones((batch, hits)),
                measured_u=np.zeros((batch, hits)), uvar=np.ones((batch, hits)),
                valid=np.ones((batch, hits), bool), model=model, bz=1.,
                initial_lr=np.ones((batch, hits)), parameter_scale=np.ones((batch, 6)),
                seed_valid=np.ones(batch, bool))
    actual = diagnostic.diagnose(**args)
    assert not actual['objective_valid'].any()
    assert actual['residuals'].shape == (batch, hits, 2)
    assert actual['objective_slopes'].shape == (batch, 4, 6, 2)
    assert actual['descent_delta'].shape == (batch, 2, 7)
    assert np.isnan(actual['lr_gain']).all()


def test_calibration_switch_distinguished_from_smooth_derivative_discrepancy(model):
    inputs, _ = synthetic(model, batch=1)
    perturb(inputs)
    args, _ = saved(inputs)
    args['model'] = affine_model(model)
    center = diagnostic.evaluate(**args)
    # Place the first cdrift just above the offset regression edge at 1 mm.
    args['early_time'] = args['early_time'].copy()
    args['early_time'][0, 0] += (1.+1e-6-center['c_drift'][0, 0])/.04
    smooth = diagnostic.diagnose(**args)
    changed_offset = replace(args['model'].offset, intercept=np.array([0., 0., .2, .2, .2, .2, .2, .2]))
    switched = diagnostic.diagnose(**dict(args, model=replace(args['model'], offset=changed_offset)))
    assert smooth['objective_valid'].all() and switched['objective_valid'].all()
    # Identical-coefficient bins are not calibration regression changes.
    assert smooth['offset_branch_count'][0, 0] > 0
    assert not smooth['offset_regression_change_count'].any()
    assert switched['offset_regression_change_count'][0, 0] > 0
    assert switched['offset_regression_change_count'][0, -1] == 0
    assert not smooth['time_branch_count'].any() and not smooth['phi_branch_count'].any()
    # Even a smooth geometry/nonlinear normalized residual has FD-step dependence.
    discrepancy = np.max(np.abs(smooth['equilibrated_gradient'][0, 0]
                                -smooth['equilibrated_gradient'][0, -1]))
    assert discrepancy > 1e-8
    jumps = switched['objective_slopes'][0, 0, 5]
    regular = smooth['objective_slopes'][0, 0, 5]
    assert abs(jumps[1]-jumps[0]) > 100*abs(regular[1]-regular[0])
    assert np.max(np.abs(switched['equilibrated_gradient'][0, 0]
                         -switched['equilibrated_gradient'][0, -1])) > 10*discrepancy


def test_interpolation_knot_counts_are_separate_from_regression_changes(model):
    inputs, _ = synthetic(model, batch=1)
    args, _ = saved(inputs)
    center = diagnostic.evaluate(**args)
    # Put one time exactly at a native interpolation knot, away from calibration edges.
    target = round(center['delta_t'][0, 0]/model.delta_t)*model.delta_t
    args['early_time'] = args['early_time'].copy()
    args['early_time'][0, 0] += target-center['delta_t'][0, 0]
    actual = diagnostic.diagnose(**args)
    assert (actual['time_branch_count'][0] > 0).all()
    assert not actual['offset_regression_change_count'].any()
    assert not actual['signed_error_regression_change_count'].any()
    # At the smallest step, a continuous knot does not produce a finite objective jump.
    slopes = actual['objective_slopes'][0, -1, 5]
    assert np.isfinite(slopes).all()
    assert abs(slopes[0]-slopes[1]) < .01
    np.testing.assert_array_equal(actual['time_branch_count'], actual['branch_counts'][..., 0].sum(axis=(2, 3)))


def test_invalid_perturbation_is_not_a_zero_derivative_success(model):
    inputs, _ = synthetic(model, batch=1)
    args, _ = saved(inputs)
    args['parameter_scale'] = args['parameter_scale'].copy()
    args['parameter_scale'][0, 4] = args['local_parameters'][0, 4]/diagnostic.FD_STEPS[0]
    assert args['local_parameters'][0, 4]-diagnostic.FD_STEPS[0]*args['parameter_scale'][0, 4] == 0
    actual = diagnostic.diagnose(**args)
    assert actual['objective_valid'][0]
    assert not actual['perturbation_valid'][0, 0, 4, 0]
    assert np.isnan(actual['objective_slopes'][0, 0, 4, 0])
    assert not actual['numeric'][0, 0] and not actual['linearization_good'][0, 0]
    assert np.isnan(actual['gradient_norm'][0, 0])
    assert not actual['descent_valid'][0, 0].any()
    assert np.isnan(actual['descent_delta'][0, 0]).all()
    assert actual['numeric'][0, -1]
    np.testing.assert_array_equal(actual['branch_valid'], actual['perturbation_valid'])


def test_signed_doca_flip_recorded_without_lr_reinitialization(model):
    inputs, _ = synthetic(model, batch=1)
    args, _ = saved(inputs)
    args['parameter_scale'] = args['parameter_scale'].copy()
    args['parameter_scale'][0, 0] = 2000.
    actual = diagnostic.diagnose(**args)
    assert actual['objective_valid'][0]
    assert actual['signed_doca_branch_count'][0, 0] > 0
    np.testing.assert_array_equal(actual['initial_lr'], args['initial_lr'])
    # Replay a large x shift while preserving the original LR, not current side.
    shifted = args['local_parameters'].copy()
    shifted[:, 0] += 2.
    trial = diagnostic.evaluate(**dict(args, local_parameters=shifted))
    np.testing.assert_array_equal(trial['drift_residual'],
                                  args['initial_lr']*trial['r_drift']-trial['signed_doca'])
    np.testing.assert_array_equal(actual['objective_slopes'][:, 0, 0, 1],
                                  (trial['objective']-actual['objective'])/1e-3)


@pytest.mark.parametrize('ignore_phi', [False, True])
def test_native_branch_index_conventions_and_signed_error_switch(model, ignore_phi):
    inputs, _ = synthetic(model, batch=1)
    args, _ = saved(inputs)
    args['model'] = replace(model, ignore_phi=ignore_phi)
    center = diagnostic.evaluate(**args)
    signed = replace(model.signed, low=center['r_drift'][0, 0]-.5+1e-6, width=.5,
                     intercept=np.array([.11, .15, .15, .15, .15, .15, .15]),
                     slope=np.zeros(7))
    args['model'] = replace(args['model'], signed=signed)
    actual = diagnostic.diagnose(**args)
    assert actual['objective_valid'][0]
    assert actual['signed_error_branch_count'][0, 0] > 0
    assert actual['signed_error_regression_change_count'][0, 0] > 0
    for name, value, low, width, maximum in (
        ('time_bin', actual['delta_t'], 0., model.delta_t, model.times.size-2),
        ('offset_bin', actual['c_drift'], model.offset.low, model.offset.width, model.offset.slope.size-1),
        ('signed_error_bin', actual['r_drift'], signed.low, signed.width, signed.slope.size-1)):
        expected = np.minimum(np.maximum(np.floor((value-low)/width), 0), maximum)
        np.testing.assert_array_equal(actual[name], expected)
    angle = np.zeros_like(actual['phi']) if ignore_phi else actual['phi']
    np.testing.assert_array_equal(actual['phi_bin_lower'], np.floor(angle/model.delta_phi))
    np.testing.assert_array_equal(actual['phi_bin_upper'],
                                  np.minimum(np.ceil(angle/model.delta_phi), model.times_dbins.shape[1]-1))
    if ignore_phi:
        assert not actual['phi_branch_count'].any()


def test_native_drift_failure_and_exceptions_preserved(model, monkeypatch):
    inputs, _ = synthetic(model)
    args, _ = saved(inputs)
    broken = replace(model, signed=replace(model.signed,
                     intercept=-np.ones_like(model.signed.intercept), slope=np.zeros_like(model.signed.slope)))
    actual = diagnostic.diagnose(**dict(args, model=broken))
    assert not actual['objective_valid'].any()
    assert not actual['valid'].any()
    assert np.isnan(actual['objective']).all()
    np.testing.assert_array_equal(actual['n_hits'], [17, 17])

    def fail(*args):
        raise ValueError('native query failure')

    monkeypatch.setattr(diagnostic, 'drift_info', fail)
    with pytest.raises(ValueError, match='native query failure'):
        diagnostic.diagnose(**args)


@pytest.mark.parametrize('scale_index', range(4))
def test_affine_t0_slope_and_gradient_independent_analytic_oracle(model, scale_index):
    inputs, truth = synthetic(model, batch=1)
    args, _ = saved(inputs)
    args['model'] = affine_model(model)
    actual = diagnostic.diagnose(**args)
    lr = np.sign(truth['signed_doca'])
    sigma = .11
    pull = (lr*.04*truth['delta_t']-truth['signed_doca'])/sigma
    derivative = -lr*.04*args['parameter_scale'][:, 5, None]/sigma
    gradient = np.sum(pull*derivative, axis=1)
    squared_norm = np.sum(derivative**2, axis=1)
    h = diagnostic.FD_STEPS[scale_index]
    expected = gradient[:, None]+np.array([-1., 1.])*.5*h*squared_norm[:, None]
    np.testing.assert_allclose(actual['objective_slopes'][:, scale_index, 5], expected, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(actual['equilibrated_gradient'][:, scale_index, 5],
                               gradient/np.sqrt(squared_norm), rtol=2e-8, atol=2e-8)


def test_phi_incidence_and_continuous_coefficient_change_are_not_jump_claims(model):
    offset = replace(model.offset, low=0., width=1., intercept=np.array([0., -.2]), slope=np.array([0., .2]))
    joined = replace(model, offset=offset)
    center = dict(valid=np.array([[True, True]]), time_bin=np.array([[2, 2]]),
                  phi_bin_lower=np.array([[1, 1]]), phi_bin_upper=np.array([[1, 2]]),
                  offset_bin=np.array([[0, 0]]), signed_error_bin=np.array([[0, 0]]),
                  signed_doca=np.array([[1., 1.]]))
    trial = {k: a.copy() for k, a in center.items()}
    trial['phi_bin_upper'][0, 0] = 2
    trial['offset_bin'][0, 1] = 1
    counts, regression = diagnostic._branch_counts(center, trial, np.array([True]), joined)
    np.testing.assert_array_equal(counts, [[0, 1, 1, 0, 0]])
    np.testing.assert_array_equal(regression, [[1, 0]])
    assert offset.intercept[0]+offset.slope[0] == offset.intercept[1]+offset.slope[1]
