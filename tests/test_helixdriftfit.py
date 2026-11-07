"""Independent analytic fixtures for a CONDITIONAL joint engineering fit.

Scalar geometry/Brent inversions are test oracles, never hot-path kernels.
No direction identification, material, covariance or physics-resolution claim.
"""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from trkfastsim.reco import helixdriftfit as joint
from trkfastsim.reco.fielddrift import build_model, drift_info
from trkfastsim.reco.helixdriftfit import fit_drift_helix
from test_helixtiming import scalar_helix

BZ = 1.0117056369781494
MASS = 105.6583755
C = 299.792458


@pytest.fixture(scope='module')
def model():
    drift = dict(wireVoltage=1450., kVcm=[.4, .8, 1.5, 3., 6., 12., 24.],
                 cmus=[3.2, 4.5, 5.7, 6.5, 7., 7.1, 7.],
                 phiBins=9, deltaDistance=.037, deltaTime=.19)
    response = dict(driftIgnorePhi=False, dRdTScale=1.17,
                    driftOffBins=[0., 2.5], driftOffset=[.012]*8,
                    driftRMSBins=[-.1, 2.7], signedDriftRMS=[.11]*7,
                    unsignedDriftRMS=[.75]*7)
    return build_model(drift, response, wire_radius=.0125, outer_radius=2.5, bz=BZ)


def synthetic(model, batch=2, hits=17, tilted=True, omega=.0021):
    """Construct exact geometric POCAs and end times WITHOUT kernel helpers."""
    p = np.tile([12., -7., .9, -.4, omega], (batch, 1))
    p[:, 4] *= np.where(np.arange(batch) % 2 == 0, 1., -1.)
    zref = np.full(batch, 35.)
    t0 = 1100.+17.*np.arange(batch)
    flight = np.where(np.arange(batch) % 2 == 0, 1., -1.)
    cal, hv = np.zeros((batch, hits, 3)), np.zeros((batch, hits, 3))
    te, u, dt, phi, signed, z = (np.zeros((batch, hits)) for _ in range(6))
    end = np.broadcast_to(np.arange(hits) % 2, (batch, hits)).copy()
    for b in range(batch):
        path_scale = math.sqrt(1+p[b, 2]**2+p[b, 3]**2)
        momentum = .299792458*abs(BZ)*path_scale/abs(p[b, 4])
        beta = momentum/math.hypot(momentum, MASS)
        for h, dz in enumerate(np.linspace(-1300., 1300., hits)):
            z[b, h] = zref[b]+dz
            point, tangent = scalar_helix(p[b], zref[b], z[b, h])
            direction = np.array([math.cos(.71*h), math.sin(.71*h), .35*math.cos(.43*h) if tilted else 0.])
            direction /= np.linalg.norm(direction)
            q = np.cross(direction, tangent)
            q /= np.linalg.norm(q)
            q *= (.7+.8*(h+1)/(hits+1))*(-1 if h % 3 == 0 else 1)
            sensor = point-q
            cal[b, h], hv[b, h] = sensor+90*direction, sensor-130*direction
            u[b, h] = 20.
            s = direction if end[b, h] == 0 else -direction
            radial = np.cross(s, [0., 0., 1.])
            angle = math.atan2(abs(np.dot(q, radial)), q[2])
            phi[b, h] = math.pi-angle if angle > math.pi/2 else angle
            radius = np.linalg.norm(q)
            signed[b, h] = flight[b]*np.sign(np.dot(np.cross(s, tangent/path_scale), -q))*radius
            dt[b, h] = brentq(lambda t: float(drift_info(model, t, phi[b, h])['r_drift'])-radius,
                             -20., 150., xtol=1e-12)
            propagation = (90. if end[b, h] == 0 else 130.)/180.
            tof = dz*path_scale/(beta*C)
            te[b, h] = t0[b]+flight[b]*tof+dt[b, h]+propagation
    inputs = dict(local_parameters=p.copy(), z_ref=zref, t0=t0.copy(), flight_direction=flight,
                  wire_cal=cal, wire_hv=hv, early_end=end, early_time=te,
                  signal_speed=np.full((batch, hits), 180.), measured_u=u,
                  uvar=np.full((batch, hits), 9.), valid=np.ones((batch, hits), bool),
                  model=model, bz=BZ, seed_valid=np.ones(batch, bool))
    truth = dict(local_parameters=p, t0=t0, delta_t=dt, phi=phi, signed_doca=signed, poca_z=z)
    return inputs, truth


def perturb(inputs):
    inputs['local_parameters'] += [.12, -.08, .00008, -.00006, 1.5e-7]
    inputs['t0'] += .8
    return inputs


@pytest.mark.parametrize('tilted', [False, True])
def test_noiseless_joint_recovery_both_directions(model, tilted):
    inputs, truth = synthetic(model, tilted=tilted)
    perturb(inputs)
    actual = fit_drift_helix(**inputs)
    assert actual['success'].all(), (actual['status'], actual['gradient_norm'])
    assert np.all(actual['rank'] == 6)
    assert np.all(actual['condition'] < 1e12)
    assert np.all(actual['final_objective'] < 1e-10)
    assert np.all(actual['initial_objective'] > 1.)
    np.testing.assert_allclose(actual['local_parameters'], truth['local_parameters'], atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual['t0'], truth['t0'], atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual['delta_t'], truth['delta_t'], atol=3e-6, rtol=0)
    np.testing.assert_allclose(actual['phi'], truth['phi'], atol=1e-6, rtol=0)
    assert actual['physical_identifiable'].all()
    assert np.isfinite(actual['p_abs']).all() and np.all(actual['beta'] < 1.)
    expected = .5*np.sum(actual['residuals']**2, axis=(1, 2))
    np.testing.assert_array_equal(actual['final_objective'], expected)
    assert actual['objective_label'] == 'engineering_conditional_objective'
    assert 'covariance' not in actual and 'charge' not in actual
    assert actual['status_counts'] == {'converged': 2}


def test_exact_stationary_initial_state_zero_budget(model):
    inputs, truth = synthetic(model)
    actual = fit_drift_helix(**inputs, max_iterations=0)
    assert actual['success'].all(), actual['status']
    assert not actual['iterations'].any() and not actual['accepted_steps'].any()
    # Local POCA uses a fixed 1e-7 mm stationarity tolerance.
    assert np.all(actual['final_objective'] < 1e-16)
    assert np.all(actual['rank'] == 6)
    np.testing.assert_array_equal(actual['initial_LR'], np.sign(truth['signed_doca']))
    np.testing.assert_allclose(actual['wire_u'], inputs['measured_u'], atol=1e-7, rtol=0)
    for name in ('local_parameters', 't0', 'z_ref'):
        np.testing.assert_array_equal(actual[name], inputs[name])


@pytest.mark.parametrize('budget', [0, 1])
def test_iteration_exhaustion_is_not_objective_success(model, budget):
    inputs, _ = synthetic(model)
    perturb(inputs)
    actual = fit_drift_helix(**inputs, max_iterations=budget)
    assert actual['objective_valid'].all()
    assert not actual['success'].any()
    assert np.all(actual['status'] == 'max_iterations')
    assert np.all(actual['iterations'] == budget)
    assert np.isnan(actual['p_abs']).all() and np.isnan(actual['beta']).all()
    assert np.isfinite(actual['local_parameters']).all() and np.isfinite(actual['t0']).all()
    if budget:
        assert np.all(actual['final_objective'] < actual['initial_objective'])


@pytest.mark.parametrize('field,value', [
    ('signal_speed', 0.), ('signal_speed', -1.), ('signal_speed', np.nan),
    ('early_end', 2.), ('early_time', np.inf), ('wire_cal', np.nan),
    ('measured_u', np.nan), ('uvar', 0.), ('uvar', np.inf)])
def test_any_selected_bad_input_fails_entire_lane(model, field, value):
    inputs, _ = synthetic(model)
    inputs[field] = inputs[field].astype(float)
    inputs[field][0, 3] = value
    actual = fit_drift_helix(**inputs, max_iterations=0)
    np.testing.assert_array_equal(actual['success'], [False, True])
    assert actual['status'][0] == 'invalid_input'
    assert not actual['objective_valid'][0]
    assert actual['n_hits'][0] == 17
    np.testing.assert_array_equal(actual['mask'], inputs['valid'])
    assert np.isnan(actual['p_abs'][0])


@pytest.mark.parametrize('mode', ['none', 'false', 'nan_t0', 'nan_p', 'zero_omega', 'zero_slope', 'bad_direction'])
def test_invalid_seed_or_undefined_curvature_no_fallback(model, mode):
    inputs, _ = synthetic(model)
    if mode == 'none':
        del inputs['seed_valid']
    elif mode == 'false':
        inputs['seed_valid'][:] = False
    elif mode == 'nan_t0':
        inputs['t0'][:] = np.nan
    elif mode == 'nan_p':
        inputs['local_parameters'][:, 0] = np.nan
    elif mode == 'zero_omega':
        inputs['local_parameters'][:, 4] = 0.
    elif mode == 'zero_slope':
        inputs['local_parameters'][:, 2:4] = 0.
    else:
        inputs['flight_direction'][:] = 0.
    actual = fit_drift_helix(**inputs)
    assert not actual['success'].any() and not actual['objective_valid'].any()
    assert np.isnan(actual['beta']).all() and np.isnan(actual['p_abs']).all()
    np.testing.assert_array_equal(actual['local_parameters'], inputs['local_parameters'])
    np.testing.assert_array_equal(actual['t0'], inputs['t0'])


def test_masked_nans_and_empty_lane(model):
    inputs, _ = synthetic(model, batch=3)
    inputs['valid'][0, 3] = False
    inputs['valid'][1] = False
    for field in ('wire_cal', 'wire_hv', 'early_end', 'early_time', 'signal_speed', 'measured_u', 'uvar'):
        inputs[field] = inputs[field].astype(float)
        inputs[field][~inputs['valid']] = np.nan
    actual = fit_drift_helix(**inputs, max_iterations=0)
    np.testing.assert_array_equal(actual['success'], [True, False, True])
    assert actual['status'][1] == 'empty'
    for key in ('doca', 'phi', 'delta_t', 'r_drift', 'signed_error', 'pull', 'long_pull',
                'wire_u', 'poca_z', 'valid', 'initial_LR', 'residuals', 'poca_boundary'):
        assert not np.any(actual[key][~inputs['valid']]), key
    for batch, hits in ((0, 4), (3, 0), (0, 0)):
        inp, _ = synthetic(model, batch=batch, hits=hits)
        result = fit_drift_helix(**inp)
        assert result['success'].shape == (batch,)
        assert not result['success'].any()


def test_membership_lr_and_inputs_never_mutate(model):
    inputs, _ = synthetic(model)
    perturb(inputs)
    saved = {k: v.copy() for k, v in inputs.items() if isinstance(v, np.ndarray)}
    initial = fit_drift_helix(**inputs, max_iterations=0)
    actual = fit_drift_helix(**inputs)
    for key, value in saved.items():
        np.testing.assert_array_equal(inputs[key], value, err_msg=key)
    np.testing.assert_array_equal(actual['initial_LR'], initial['initial_LR'])
    np.testing.assert_array_equal(actual['mask'], inputs['valid'])
    np.testing.assert_array_equal(actual['drift_residual'],
                                  actual['initial_LR']*actual['r_drift']-actual['signed_doca'])


def test_exact_batch_chunk_and_budget_invariance(model):
    inputs, _ = synthetic(model, batch=3)
    perturb(inputs)
    both = fit_drift_helix(**inputs)
    bounded = fit_drift_helix(**inputs, max_rows=3, max_cells=51)
    for key in both:
        if isinstance(both[key], np.ndarray):
            np.testing.assert_array_equal(bounded[key], both[key], err_msg=key)
        else:
            assert bounded[key] == both[key]
    for b in range(3):
        one = {k: v[b:b+1] if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
        actual = fit_drift_helix(**one, max_rows=1, max_cells=17)
        for key, values in both.items():
            if isinstance(values, np.ndarray):
                np.testing.assert_array_equal(actual[key], values[b:b+1], err_msg=key)


@pytest.mark.parametrize('sigma', [0., -.1, np.nan, np.inf])
def test_single_bad_drift_hit_fails_whole_candidate(model, monkeypatch, sigma):
    inputs, _ = synthetic(model, batch=1)
    real = joint.drift_info
    def bad(model, dt, phi):
        info = real(model, dt, phi)
        info['signed_error'][0] = sigma
        return info
    monkeypatch.setattr(joint, 'drift_info', bad)
    actual = fit_drift_helix(**inputs)
    assert actual['status'][0] == 'drift_failed'
    assert not actual['objective_valid'][0] and not actual['success'][0]
    assert actual['n_hits'][0] == 17
    assert actual['mask'].all()


def test_native_nonpositive_signed_error(model):
    inputs, _ = synthetic(model)
    inputs['model'] = replace(model, signed=replace(model.signed,
        intercept=-np.ones_like(model.signed.intercept), slope=np.zeros_like(model.signed.slope)))
    actual = fit_drift_helix(**inputs)
    assert np.all(actual['status'] == 'drift_failed')
    assert not actual['success'].any()


def test_unclipped_dt_radius_and_oracle_coverage(model):
    inputs, _ = synthetic(model, batch=1)
    inputs['early_time'][0, 0] -= 500.
    inputs['early_time'][0, -1] += 500.
    actual = fit_drift_helix(**inputs, max_iterations=0)
    assert actual['objective_valid'][0]
    assert actual['outside_oracle'][0] == 2
    assert actual['delta_t'][0, 0] < -50. and actual['delta_t'][0, -1] > 200.
    assert actual['r_drift'][0, 0] < 0. and actual['r_drift'][0, -1] > 2.5
    info = drift_info(model, actual['delta_t'], actual['phi'])
    np.testing.assert_array_equal(actual['r_drift'], info['r_drift'])
    np.testing.assert_array_equal(actual['pull'],
        (actual['initial_LR']*info['r_drift']-actual['signed_doca'])/info['signed_error'])


def test_finite_wire_endpoints_outside_and_zero_length(model):
    for mode in ('endpoint', 'outside', 'zero'):
        inputs, _ = synthetic(model, batch=1)
        direction = inputs['wire_cal'][0, 0]-inputs['wire_hv'][0, 0]
        direction /= np.linalg.norm(direction)
        sensor = inputs['wire_cal'][0, 0]-90*direction
        if mode == 'zero':
            inputs['wire_cal'][0, 0] = inputs['wire_hv'][0, 0]
        elif mode == 'endpoint':
            # Use an exactly represented endpoint, not a tilted POCA rounded just inside.
            inputs['local_parameters'][0, 3] = 0.
            inputs['wire_hv'][0, 0] = [12., -8., 35.]
            inputs['wire_cal'][0, 0] = [112., -8., 35.]
            inputs['measured_u'][0, 0] = -50.
        else:
            inputs['wire_hv'][0, 0] = sensor+direction
            # Preserve the independently known measured point, just change origin.
            midpoint = .5*(inputs['wire_cal'][0, 0]+inputs['wire_hv'][0, 0])
            inputs['measured_u'][0, 0] = np.dot(sensor-midpoint, direction)
        actual = fit_drift_helix(**inputs)
        assert not actual['success'][0] and not actual['objective_valid'][0]
        assert actual['status'][0] in ('outside_wire', 'poca_failed', 'invalid_input')


def test_zero_doca_and_near_parallel_fail_not_reassign_or_drop(model):
    for parallel in (False, True):
        inputs, truth = synthetic(model, batch=1)
        h = 8
        point, tangent = scalar_helix(inputs['local_parameters'][0], inputs['z_ref'][0], truth['poca_z'][0, h])
        if parallel:
            s = tangent/np.linalg.norm(tangent)
            q = np.cross(s, [0., 1., 0.])
            q /= np.linalg.norm(q)
        else:
            s = np.array([1., 0., 0.])
            q = np.zeros(3)
        inputs['wire_cal'][0, h], inputs['wire_hv'][0, h] = point-q+100*s, point-q-100*s
        inputs['measured_u'][0, h] = 0.
        actual = fit_drift_helix(**inputs)
        assert not actual['success'][0] and not actual['objective_valid'][0]
        assert actual['n_hits'][0] == 17
        assert actual['status'][0] in ('parallel', 'poca_failed', 'undefined_angle')


def test_invalid_fd_perturbation_is_linearization_failed(model, monkeypatch):
    inputs, _ = synthetic(model, batch=1)
    real = joint.prepare_timing
    initial_x = inputs['local_parameters'][0, 0]
    def invalidate_plus(*args, **kwargs):
        result = real(*args, **kwargs)
        result['row_valid'] &= args[0][:, 0] <= initial_x+.0005
        return result
    monkeypatch.setattr(joint, 'prepare_timing', invalidate_plus)
    actual = fit_drift_helix(**inputs, max_iterations=0)
    assert actual['objective_valid'][0] and actual['final_objective'][0] < 1e-18
    assert actual['status'][0] == 'linearization_failed'
    assert not actual['success'][0] and actual['rank'][0] == 0
    assert np.isnan(actual['p_abs'][0])


def test_poca_boundary_is_explicit_failure(model, monkeypatch):
    inputs, _ = synthetic(model, batch=1)
    real = joint._poca
    def boundary(*args, **kwargs):
        result = real(*args, **kwargs)
        result['boundary'][0, 0] = True
        result['converged'][0, 0] = False
        return result
    monkeypatch.setattr(joint, '_poca', boundary)
    actual = fit_drift_helix(**inputs)
    assert actual['status'][0] == 'poca_boundary'
    assert actual['poca_boundary'][0, 0]
    assert not actual['success'][0]


def test_actual_residual_fd_recomputes_geometry_sigma_and_beta(model, monkeypatch):
    inputs, _ = synthetic(model, batch=1)
    # Nonconstant sigma requires normalization derivatives in the independent secant check.
    inputs['model'] = replace(model, signed=replace(model.signed,
        intercept=np.full_like(model.signed.intercept, .08), slope=np.full_like(model.signed.slope, .03)))
    perturb(inputs)
    calls, evaluations = [], []
    real = joint._linearize
    def capture(parameters, scale, evaluate, residual, fd_step, condition_limit):
        def record(candidate):
            result = evaluate(candidate)
            calls.append(candidate.copy())
            evaluations.append(result)
            return result
        return real(parameters, scale, record, residual, fd_step, condition_limit)
    monkeypatch.setattr(joint, '_linearize', capture)
    actual = fit_drift_helix(**inputs, max_iterations=0)
    assert len(calls) == 12
    assert all(e['objective_valid'][0] for e in evaluations)
    p = np.column_stack((inputs['local_parameters'], inputs['t0']))
    jac = np.empty((17, 2, 6))
    for j in range(6):
        assert calls[2*j][0, j] == p[0, j]+1e-3*actual['parameter_scale'][0, j]
        plus, minus = evaluations[2*j:2*j+2]
        jac[..., j] = (plus['residuals'][0]-minus['residuals'][0])/.002
    flat = jac.reshape(-1, 6)
    r = actual['residuals'][0].reshape(-1)
    expected = np.max(np.abs(flat.T@r)/np.sqrt(np.sum(flat**2, axis=0)))
    assert actual['gradient_norm'][0] == pytest.approx(expected, rel=1e-14)
    # Perturbations refind POCA and vary measured beta, radius and sigma.
    assert np.max(np.abs(evaluations[0]['poca_z']-evaluations[1]['poca_z'])) > 1e-5
    assert evaluations[8]['beta'][0] != evaluations[9]['beta'][0]
    assert np.max(np.abs(evaluations[10]['signed_error']-evaluations[11]['signed_error'])) > 1e-6


@pytest.mark.parametrize('options', [dict(max_rows=1), dict(max_cells=33), dict(max_rows=True),
    dict(max_iterations=-1), dict(fd_step=0.), dict(gradient_tolerance=np.inf), dict(mass=0.),
    dict(bz=0.), dict(condition_limit=1.), dict(initial_damping=1e13), dict(max_parameter_step=-1.)])
def test_strict_bounds_and_options(model, options):
    inputs, _ = synthetic(model)
    inputs.update(options)
    with pytest.raises(ValueError):
        fit_drift_helix(**inputs)


@pytest.mark.parametrize('field,value', [('valid', np.ones((2, 17))), ('seed_valid', [1, 1]),
    ('flight_direction', 1), ('local_parameters', np.zeros((2, 6))), ('early_end', True)])
def test_strict_shapes_and_dtypes(model, field, value):
    inputs, _ = synthetic(model)
    inputs[field] = value
    with pytest.raises(ValueError):
        fit_drift_helix(**inputs)


@pytest.mark.parametrize('field', ['local_parameters', 'wire_cal', 'measured_u', 't0'])
def test_extreme_finite_values_do_not_warn_or_succeed(model, field):
    inputs, _ = synthetic(model)
    inputs[field][:] = 1e308
    if field == 't0':
        # Unsafe extrapolation raises rather than fabricating a successful radius.
        with pytest.raises(ValueError, match='unsafe|nonfinite'):
            fit_drift_helix(**inputs)
    else:
        actual = fit_drift_helix(**inputs)
        assert not actual['success'].any()
        assert np.isnan(actual['p_abs']).all()


def test_current_spatial_sign_can_flip_but_initial_lr_is_fixed(model, monkeypatch):
    inputs, truth = synthetic(model, batch=1)
    real = joint._linearize
    trials = []
    def probe(parameters, scale, evaluate, residual, fd_step, condition_limit):
        crossed = parameters.copy()
        crossed[:, 0] += 3.
        trials.append(evaluate(crossed))
        return real(parameters, scale, evaluate, residual, fd_step, condition_limit)
    monkeypatch.setattr(joint, '_linearize', probe)
    actual = fit_drift_helix(**inputs, max_iterations=0)
    assert actual['success'][0]
    crossed = trials[0]
    assert crossed['objective_valid'][0] and crossed['valid'].all()
    assert np.any(np.sign(crossed['signed_doca']) != np.sign(truth['signed_doca']))
    np.testing.assert_array_equal(crossed['drift_residual'],
        actual['initial_LR']*crossed['r_drift']-crossed['signed_doca'])
    reassigned = np.sign(crossed['signed_doca'])*crossed['r_drift']-crossed['signed_doca']
    assert np.max(np.abs(reassigned-crossed['drift_residual'])) > 1.


def test_rank_deficiency_cannot_pass_even_perfect_objective(model):
    inputs, _ = synthetic(model, hits=2)
    actual = fit_drift_helix(**inputs, max_iterations=0)
    assert actual['objective_valid'].all()
    assert np.all(actual['final_objective'] < 1e-15)
    assert np.all(actual['rank'] < 6)
    assert np.all(actual['status'] == 'ill_conditioned')
    assert not actual['success'].any()
    assert np.isnan(actual['p_abs']).all()


def test_clamped_step_damping_stall_is_failure(model):
    inputs, _ = synthetic(model, batch=1)
    perturb(inputs)
    actual = fit_drift_helix(**inputs, max_parameter_step=1e-25, initial_damping=1e12)
    assert actual['status'][0] == 'stalled'
    assert not actual['success'][0] and actual['objective_valid'][0]
    assert actual['damping'][0] > 1e12
    assert actual['iterations'][0] == 1 and actual['accepted_steps'][0] == 0
    np.testing.assert_array_equal(actual['local_parameters'], inputs['local_parameters'])
    np.testing.assert_array_equal(actual['t0'], inputs['t0'])


def test_eigensolve_failure_cannot_pass_stationary_seed(model, monkeypatch):
    inputs, _ = synthetic(model, batch=1)
    def failed(*args, **kwargs):
        raise np.linalg.LinAlgError('forced failure')
    monkeypatch.setattr(np.linalg, 'eigh', failed)
    actual = fit_drift_helix(**inputs, max_iterations=0)
    assert actual['status'][0] == 'linearization_failed'
    assert actual['objective_valid'][0] and not actual['success'][0]
    assert actual['rank'][0] == 0 and np.isnan(actual['p_abs'][0])


def test_noisy_independent_scipy_scalar_geometry_oracle(model):
    from scipy.optimize import least_squares
    inputs, truth = synthetic(model)
    # This smooth negative-flight lane avoids a calibration or phi switch.
    inputs = {k: v[1:2].copy() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
    inputs['early_time'] += .25*np.sin(np.arange(17))
    inputs['measured_u'] += .1*np.cos(np.arange(17))
    perturb(inputs)
    actual = fit_drift_helix(**inputs)
    assert actual['success'][0], (actual['status'], actual['gradient_norm'])
    base = np.r_[inputs['local_parameters'][0], inputs['t0'][0]]
    scale = actual['parameter_scale'][0]
    fixed_lr = actual['initial_LR'][0].copy()

    def independent_residual(q):
        candidate = base+scale*q
        p, t0 = candidate[:5], candidate[5]
        path = math.sqrt(1+p[2]**2+p[3]**2)
        momentum = .299792458*abs(BZ)*path/abs(p[4])
        beta = momentum/math.hypot(momentum, MASS)
        residual = []
        for h in range(17):
            cal, hv = inputs['wire_cal'][0, h], inputs['wire_hv'][0, h]
            mid = .5*(cal+hv)
            direction = (cal-hv)/np.linalg.norm(cal-hv)
            def at(z):
                point, tangent = scalar_helix(p, inputs['z_ref'][0], z)
                u = np.dot(point-mid, direction)
                separation = point-mid-u*direction
                return u, separation, tangent
            def gradient(z):
                _, separation, tangent = at(z)
                return np.dot(separation, tangent)
            z = brentq(gradient, truth['poca_z'][1, h]-5., truth['poca_z'][1, h]+5., xtol=1e-11)
            u, separation, tangent = at(z)
            early = cal if inputs['early_end'][0, h] == 0 else hv
            s = direction if inputs['early_end'][0, h] == 0 else -direction
            sensor = mid+u*direction
            phi = math.atan2(abs(np.dot(separation, np.cross(s, [0., 0., 1.]))), separation[2])
            phi = math.pi-phi if phi > math.pi/2 else phi
            ts = inputs['early_time'][0, h]-np.dot(early-sensor, s)/inputs['signal_speed'][0, h]
            tof = (z-inputs['z_ref'][0])*path/(beta*C)
            flight = inputs['flight_direction'][0]
            dt = ts-t0-flight*tof
            signed = flight*np.sign(np.dot(np.cross(s, tangent/path), -separation))*np.linalg.norm(separation)
            info = drift_info(model, dt, phi)
            residual.extend([(fixed_lr[h]*info['r_drift']-signed)/info['signed_error'],
                             (u-inputs['measured_u'][0, h])/math.sqrt(inputs['uvar'][0, h])])
        return np.array(residual)
    oracle = least_squares(independent_residual, np.zeros(6), jac='3-point',
                           xtol=1e-10, ftol=1e-10, gtol=1e-9, max_nfev=100)
    assert oracle.success
    assert actual['final_objective'][0] == pytest.approx(oracle.cost, abs=1e-9)
    np.testing.assert_allclose(np.r_[actual['local_parameters'][0], actual['t0'][0]],
                               base+scale*oracle.x, atol=3e-6, rtol=0)


def test_noisy_calibration_stall_does_not_get_budget_success(model):
    inputs, _ = synthetic(model, batch=1)
    perturb(inputs)
    inputs['early_time'] += .25*np.sin(np.arange(17))
    inputs['measured_u'] += .1*np.cos(np.arange(17))
    actual = fit_drift_helix(**inputs)
    # A piecewise-model stall must fail with finite last-iterate diagnostics.
    assert actual['objective_valid'][0]
    assert actual['final_objective'][0] < actual['initial_objective'][0]
    if not actual['success'][0]:
        assert actual['status'][0] in ('stalled', 'max_iterations', 'linearization_failed')
        assert np.isnan(actual['p_abs'][0])
    else:
        assert actual['rank'][0] == 6
        assert actual['gradient_norm'][0] <= 1e-6*(1+math.sqrt(2*actual['final_objective'][0]))


def test_converged_near_straight_fit_does_not_release_physical_momentum(model, monkeypatch):
    inputs, _ = synthetic(model, batch=1, omega=1e-12)
    original = joint._linearize

    def stationary_certificate(parameters, scale, evaluate, residual, fd_step, condition_limit):
        result = original(parameters, scale, evaluate, residual, fd_step, condition_limit)
        result['good'][:] = True
        result['numeric'][:] = True
        result['rank'][:] = 6
        result['gradient_norm'][:] = 0.
        return result

    monkeypatch.setattr(joint, '_linearize', stationary_certificate)
    actual = fit_drift_helix(**inputs, max_iterations=0)
    assert actual['objective_valid'][0] and actual['success'][0]
    assert not actual['physical_identifiable'][0]
    assert np.isnan(actual['p_abs'][0]) and np.isnan(actual['beta'][0])
    assert np.isfinite(actual['local_parameters']).all()
