"""Independent derivatives and branch contracts; no saved campaign inputs."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from trkfastsim.reco import driftjacobian as analytic
from trkfastsim.reco.driftfitdiagnostic import _Evaluator, _empty_state
from trkfastsim.reco.fielddrift import drift_info
from test_driftfitdiagnostic import affine_model
from test_helixdriftfit import model, synthetic, perturb  # noqa: F401
from test_helixtiming import scalar_helix


def setup(model, batch=2, tilted=True, change=True):
    inputs, truth = synthetic(model, batch=batch, tilted=tilted)
    inputs['initial_lr'] = np.sign(truth['signed_doca'])
    # Saved scales, deliberately not inferred anew at each candidate.
    inputs['parameter_scale'] = np.tile([1., 1.3, 1/2600., 1/2400., 1/6e6, 10.], (batch, 1))
    if change:
        perturb(inputs)
    ev = _Evaluator(**inputs)
    p = ev.candidate.copy()
    return inputs, ev, p, ev(p)


def nonconstant(model):
    off = replace(model.offset, intercept=np.full_like(model.offset.intercept, .012),
                  slope=np.full_like(model.offset.slope, .07))
    err = replace(model.signed, intercept=np.full_like(model.signed.intercept, .13),
                  slope=np.full_like(model.signed.slope, .025))
    return replace(model, offset=off, signed=err)


def same_branch(center, trial):
    for name in ('time_bin', 'phi_bin_lower', 'phi_bin_upper', 'offset_bin', 'signed_error_bin'):
        np.testing.assert_array_equal(center[name], trial[name], err_msg=name)
    np.testing.assert_array_equal(np.sign(center['signed_doca']), np.sign(trial['signed_doca']))
    assert trial['objective_valid'].all()


@pytest.mark.parametrize('tilted', [False, True])
@pytest.mark.parametrize('bzsign', [-1., 1.])
@pytest.mark.parametrize('step', [1e-4, 1e-5, 2e-6])
def test_full_chain_multiple_scales_same_branch(model, tilted, bzsign, step):
    inputs, _, _, _ = setup(nonconstant(model), tilted=tilted)
    inputs['bz'] *= bzsign
    ev = _Evaluator(**inputs)
    p, state = ev.candidate, ev(ev.candidate)
    out = analytic.linearize(ev, p, state)
    assert out['numeric'].all() and out['good'].all() and out['differentiable'].all()
    assert (out['rank'] == 6).all()
    numerical = np.empty_like(out['jacobian'])
    for j in range(6):
        shift = np.zeros_like(p)
        shift[:, j] = ev.scale[:, j]*step
        plus, minus = ev(p+shift), ev(p-shift)
        same_branch(state, plus)
        same_branch(state, minus)
        numerical[..., j] = (plus['residuals']-minus['residuals'])/(2*step)
        for name in ('poca_z', 'wire_u', 'signed_doca', 'phi', 'delta_t',
                     'c_drift', 'r_drift', 'signed_error'):
            reference = (plus[name]-minus[name])/(2*step)
            np.testing.assert_allclose(out[name+'_jacobian'][..., j], reference,
                                       rtol=4e-5, atol=3e-6, err_msg=name)
    np.testing.assert_allclose(out['jacobian'], numerical, rtol=4e-5, atol=1e-5)
    expected_gradient = np.sum(numerical*state['residuals'][..., None], axis=(1, 2))
    np.testing.assert_allclose(out['g']*out['norm'], expected_gradient, rtol=4e-6, atol=1e-5)


def test_independent_affine_t0_normalization_and_gradient(model):
    inputs, ev, p, state = setup(nonconstant(affine_model(model)))
    out = analytic.linearize(ev, p, state)
    # Affine r=.93*.04*dt-.012 and sigma=.13+.025*r give an independent t0 slope.
    dr = -.93*.04
    sigma = .13+.025*state['r_drift']
    numerator = ev.lr*state['r_drift']-state['signed_doca']
    derivative = (ev.lr*dr/sigma-numerator*.025*dr/sigma**2)*ev.scale[:, None, 5]
    np.testing.assert_allclose(out['jacobian'][..., 0, 5], derivative, rtol=3e-15, atol=1e-14)
    np.testing.assert_array_equal(out['jacobian'][..., 1, 5], 0.)
    expected = np.sum(derivative*state['residuals'][..., 0], axis=1)
    np.testing.assert_allclose(out['g'][:, 5]*out['norm'][:, 5], expected, rtol=3e-15)
    assert not np.allclose(derivative, ev.lr*dr/sigma*ev.scale[:, None, 5])
    stock = drift_info(inputs['model'], state['delta_t'], state['phi'])
    assert not np.allclose(stock['drift_velocity'], .93*.04)


def scalar_calibration(model, dt, phi):
    """Independent scalar composition including extrapolation and folded phi."""
    folded = abs(math.fmod(0. if model.ignore_phi else phi, math.pi))
    if folded > math.pi/2:
        folded = math.pi-folded
    j = min(max(math.floor(dt/model.delta_t), 0), len(model.times)-2)
    k = min(math.floor(folded/model.delta_phi), model.distances_tbins.shape[1]-2)
    t, f = (dt-model.times[j])/model.delta_t, (folded-k*model.delta_phi)/model.delta_phi
    q = model.distances_tbins
    lo = q[j, k]+t*(q[j+1, k]-q[j, k])
    hi = q[j, k+1]+t*(q[j+1, k+1]-q[j, k+1])
    c = lo+f*(hi-lo)
    def line(cal, x):
        i = min(max(math.floor((x-cal.low)/cal.width), 0), len(cal.slope)-1)
        return cal.intercept[i]+cal.slope[i]*x
    r = c-line(model.offset, c)
    return np.array([c, r, line(model.signed, r)])


@pytest.mark.parametrize('ignore_phi', [False, True])
def test_dense_calibration_partials_independent_scalar_and_tails(model, ignore_phi):
    m = replace(nonconstant(model), ignore_phi=ignore_phi)
    ts = np.r_[[-10000., -33.173], (np.arange(0, len(m.times)-1, 13)+.317)*m.delta_t,
               [m.times[-1]+17.43, 10000.123]]
    phis = np.array([-.43, .0713, .417, 1.49, 2.19, 4.13])
    dt, phi = np.meshgrid(ts, phis)
    out = analytic.calibration_partials(m, dt, phi)
    assert out['differentiable'].all()
    stock = drift_info(m, dt, phi)
    for name in stock:
        np.testing.assert_array_equal(out[name], stock[name])
    scalar = np.array([scalar_calibration(m, t, p) for t, p in zip(dt.flat, phi.flat)])
    scalar = scalar.reshape(dt.shape+(3,))
    for k, name in enumerate(('c_drift', 'r_drift', 'signed_error')):
        np.testing.assert_allclose(out[name], scalar[..., k], rtol=5e-14, atol=1e-12)
    for h in (1e-4, 1e-5):
        for axis, names in ((0, ('dc_dt', 'dr_dt', 'dsigma_dt')),
                            (1, ('dc_dphi', 'dr_dphi', 'dsigma_dphi'))):
            derivative = np.array([(scalar_calibration(m, t+h*(axis == 0), p+h*(axis == 1))
                                    -scalar_calibration(m, t-h*(axis == 0), p-h*(axis == 1)))/(2*h)
                                   for t, p in zip(dt.flat, phi.flat)]).reshape(dt.shape+(3,))
            for k, name in enumerate(names):
                np.testing.assert_allclose(out[name], derivative[..., k], rtol=2e-6, atol=2e-8)
    assert analytic.calibration_partials(m, np.array([]), np.array([]))['dc_dt'].shape == (0,)
    assert analytic.calibration_partials(m, 3.123, .43)['dc_dt'].shape == ()


@pytest.mark.parametrize('omega', [0., 1e-18, -1e-12, 3e-8, -.0021])
def test_fixed_z_partials_independent_integral_near_straight(omega):
    p = np.array([[12., -7., .9, -.4, omega]])
    zref, z = np.array([35.]), np.array([[-1265., 35., 1335.]])
    pt, vt = analytic.trajectory_partials(p, zref, z)
    for h, zz in enumerate(z[0]):
        length = zz-zref[0]
        tx, ty = p[0, 2:4]
        # Differentiate the defining tangent integral independently of sinc/cosc.
        funs = (lambda t: math.cos(omega*t), lambda t: -math.sin(omega*t),
                lambda t: -t*(tx*math.sin(omega*t)+ty*math.cos(omega*t)))
        yfun = (lambda t: math.sin(omega*t), lambda t: math.cos(omega*t),
                lambda t: t*(tx*math.cos(omega*t)-ty*math.sin(omega*t)))
        reference = np.zeros((3, 5))
        reference[0, 0], reference[1, 1] = 1., 1.
        for j in range(3):
            reference[0, j+2] = quad(funs[j], 0., length, epsabs=1e-8)[0]
            reference[1, j+2] = quad(yfun[j], 0., length, epsabs=1e-8)[0]
        np.testing.assert_allclose(pt[0, h], reference, rtol=3e-14, atol=2e-10)
        expected_v = np.zeros((3, 5))
        for j in range(3):
            expected_v[0, j+2], expected_v[1, j+2] = funs[j](length), yfun[j](length)
        np.testing.assert_allclose(vt[0, h], expected_v, rtol=3e-15, atol=1e-12)


def independent_residual(ev, candidate):
    """Scalar SciPy stationarity roots and geometry independent of kernel POCA."""
    result = np.zeros((ev.hits, 2))
    for h in range(ev.hits):
        p = candidate[0, :5]
        w, mid = ev.direction[0, h], ev.mid[0, h]
        def stationarity(z):
            point, tangent = scalar_helix(p, ev.zref[0], z)
            delta = point-mid
            q = delta-np.dot(delta, w)*w
            return np.dot(q, tangent)
        anchor = mid[2]+ev.u[0, h]*w[2]
        z = brentq(stationarity, anchor-10., anchor+10., xtol=5e-13)
        point, tangent = scalar_helix(p, ev.zref[0], z)
        u = np.dot(point-mid, w)
        q = point-(mid+u*w)
        s = w if ev.end[0, h] == 0 else -w
        early = ev.cal[0, h] if ev.end[0, h] == 0 else ev.hv[0, h]
        b = np.array([0., 0., np.sign(ev.bz)])
        phi = math.atan2(abs(np.dot(q, np.cross(s, b))), abs(np.dot(q, b)))
        signed = ev.flight[0]*np.sign(np.dot(np.cross(s, tangent), -q))*np.linalg.norm(q)
        sensor_time = ev.te[0, h]-np.dot(early-(mid+u*w), s)/ev.speed[0, h]
        momentum = .299792458*abs(ev.bz)*math.sqrt(1+p[2]**2+p[3]**2)/abs(p[4])
        beta = momentum/math.hypot(momentum, ev.mass)
        tof = (z-ev.zref[0])*math.sqrt(1+p[2]**2+p[3]**2)/(beta*299.792458)
        dt = sensor_time-candidate[0, 5]-ev.flight[0]*tof
        _, radius, sigma = scalar_calibration(ev.model, dt, phi)
        result[h] = [(ev.lr[0, h]*radius-signed)/sigma, (u-ev.u[0, h])/ev.sigma_u[0, h]]
    return result


def test_independent_scipy_geometry_oracle(model):
    _, ev, p, state = setup(nonconstant(model), batch=1)
    out = analytic.linearize(ev, p, state)
    np.testing.assert_allclose(state['residuals'][0], independent_residual(ev, p), atol=1e-8, rtol=1e-7)
    for h in (2e-4, 2e-5):
        for j in range(6):
            shift = np.zeros_like(p)
            shift[:, j] = h*ev.scale[:, j]
            reference = (independent_residual(ev, p+shift)-independent_residual(ev, p-shift))/(2*h)
            np.testing.assert_allclose(out['jacobian'][0, ..., j], reference, atol=4e-7, rtol=2e-5)


@pytest.mark.parametrize('kind', ['identical', 'jump', 'kink'])
@pytest.mark.parametrize('calibration', ['offset', 'signed'])
def test_regression_boundaries_values_and_slopes(model, kind, calibration):
    m = affine_model(model)
    # c = .04*dt; offset zero, r=c, boundary at 1.0.
    slopes, intercept = np.zeros(2), np.zeros(2)
    if calibration == 'signed':
        intercept[:] = .2
    if kind == 'jump':
        intercept[1] += .01
    elif kind == 'kink':
        slopes[1] = .03
        intercept[1] -= .03  # equal value at join, unequal derivative
    cal = replace(getattr(m, calibration), low=0., width=1., slope=slopes, intercept=intercept)
    m = replace(m, **{calibration: cal})
    out = analytic.calibration_partials(m, 25., .431)
    key = 'offset_boundary' if calibration == 'offset' else 'signed_error_boundary'
    assert bool(out[key]) == (kind != 'identical')
    assert bool(out['differentiable']) == (kind == 'identical')
    for sign in (-1., 1.):
        # Arithmetic slack catches nextafter but is not a physical bin buffer.
        near = analytic.calibration_partials(m, np.nextafter(25., 25.+sign), .431)
        far = analytic.calibration_partials(m, 25.+sign*1e-8, .431)
        assert bool(near[key]) == (kind != 'identical')
        assert far['differentiable']
    for dt in (-1000., 1000.):
        assert analytic.calibration_partials(m, dt, .431)['differentiable']


@pytest.mark.parametrize('which', ['time', 'phi', 'zero_phi', 'fold_phi'])
def test_exact_native_knots_not_smooth_certificates(model, which):
    dt, phi = 2.371*model.delta_t, .371*model.delta_phi
    if which == 'time':
        dt = 5*model.delta_t
    elif which == 'phi':
        phi = 3*model.delta_phi
    elif which == 'zero_phi':
        phi = 0.
    else:
        phi = np.pi/2
    out = analytic.calibration_partials(model, dt, phi)
    assert not out['differentiable']
    key = 'time_boundary' if which == 'time' else 'phi_boundary'
    assert out[key]
    assert np.isfinite(out['dc_dt']) and np.isfinite(out['dc_dphi'])
    assert analytic.calibration_partials(model, dt+1e-8, phi+1e-8)['differentiable']


def test_native_tail_endpoints_are_not_false_time_branches(model):
    for dt in (0., model.times[-1], -1e5, 1e5):
        assert not analytic.calibration_partials(model, dt, .431)['time_boundary']


def test_numeric_rank_separate_from_differentiability(model):
    inputs, ev, p, state = setup(model, batch=1)
    # Re-time a selected hit to a native knot using actual center delta_t.
    inputs['early_time'][0, 2] += 30*model.delta_t-state['delta_t'][0, 2]
    ev = _Evaluator(**inputs)
    state = ev(p)
    out = analytic.linearize(ev, p, state)
    assert out['numeric'][0] and out['good'][0] and out['rank'][0] == 6
    assert not out['differentiable'][0] and not out['differentiable_hit'][0, 2]
    assert np.any(out['jacobian'][0, 2])


@pytest.mark.parametrize('field,value', [('early_time', np.nan), ('seed_valid', False),
                                        ('initial_lr', 0.), ('parameter_scale', np.nan),
                                        ('local_parameters', np.nan), ('uvar', -1.)])
def test_invalid_rows_no_fake_certificate(model, field, value):
    inputs, _, _, _ = setup(model)
    inputs[field][0] = value
    ev = _Evaluator(**inputs)
    out = analytic.linearize(ev, ev.candidate, ev(ev.candidate))
    for name in ('numeric', 'good', 'differentiable'):
        np.testing.assert_array_equal(out[name], [False, True])
    assert out['rank'][0] == 0
    assert not out['differentiable_hit'][0].any()
    np.testing.assert_array_equal(out['jacobian'][0], 0.)


def test_masks_readonly_no_mutation_no_evaluator_calls(model, monkeypatch):
    inputs, _, _, _ = setup(model)
    inputs['valid'][:, -2:] = False
    for name in ('wire_cal', 'wire_hv', 'early_time', 'signal_speed', 'measured_u', 'uvar', 'initial_lr'):
        inputs[name][:, -2:] = np.nan
    ev = _Evaluator(**inputs)
    p, state = ev.candidate, ev(ev.candidate)
    before = {name: value.copy() for name, value in state.items()}
    for value in state.values():
        value.setflags(write=False)
    p.setflags(write=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('analytic Jacobian must not evaluate perturbations')
    monkeypatch.setattr(_Evaluator, '__call__', forbidden)
    out = analytic.linearize(ev, p, state)
    assert out['good'].all() and out['differentiable'].all()
    np.testing.assert_array_equal(out['jacobian'][:, -2:], 0.)
    assert not out['differentiable_hit'][:, -2:].any()
    for name, value in before.items():
        np.testing.assert_array_equal(state[name], value)


def test_exact_batching_with_invalid_and_masked_rows(model):
    inputs, _, _, _ = setup(model, batch=4)
    inputs['seed_valid'][1] = False
    inputs['valid'][2, -2:] = False
    ev = _Evaluator(**inputs)
    full = analytic.linearize(ev, ev.candidate, ev(ev.candidate))
    parts = []
    for start, stop in ((0, 1), (1, 3), (3, 4)):
        args = {name: value[start:stop] if isinstance(value, np.ndarray) else value
                for name, value in inputs.items()}
        e = _Evaluator(**args)
        parts.append(analytic.linearize(e, e.candidate, e(e.candidate)))
    for name in full:
        np.testing.assert_array_equal(full[name], np.concatenate([part[name] for part in parts]), err_msg=name)


@pytest.mark.parametrize('batch,hits', [(0, 17), (2, 0), (0, 0)])
def test_empty_shapes(model, batch, hits):
    inputs, _, _, _ = setup(model, batch=2)
    args = {name: (np.zeros((batch,)+value.shape[1:], dtype=value.dtype)
                   if isinstance(value, np.ndarray) else value) for name, value in inputs.items()}
    for name, value in args.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[1] == 17:
            args[name] = value[:, :hits]
    args['parameter_scale'][:] = 1.
    ev = _Evaluator(**args)
    out = analytic.linearize(ev, ev.candidate, _empty_state(ev))
    assert out['jacobian'].shape == (batch, hits, 2, 6)
    assert not out['numeric'].any() and not out['differentiable'].any()
    assert not out['good'].any()


def test_eigh_failure_explicit(model, monkeypatch):
    _, ev, p, state = setup(model)
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('injected')
    monkeypatch.setattr(np.linalg, 'eigh', fail)
    out = analytic.linearize(ev, p, state)
    assert not out['numeric'].any() and not out['good'].any() and not out['differentiable'].any()
    assert (out['rank'] == 0).all()
    np.testing.assert_array_equal(out['jacobian'], 0.)


def test_condition_limit_and_rank_failure_are_not_nondifferentiability(model):
    _, ev, p, state = setup(model)
    out = analytic.linearize(ev, p, state, condition_limit=1.)
    assert out['numeric'].all() and out['differentiable'].all() and not out['good'].any()
    assert (out['rank'] == 6).all()
    for bad in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError):
            analytic.linearize(ev, p, state, condition_limit=bad)


def test_rank_deficient_not_numeric_failure(model):
    inputs, _, _, _ = setup(affine_model(model))
    inputs['model'] = replace(inputs['model'], distances_tbins=np.zeros_like(inputs['model'].distances_tbins))
    ev = _Evaluator(**inputs)
    out = analytic.linearize(ev, ev.candidate, ev(ev.candidate))
    assert out['numeric'].all() and out['differentiable'].all()
    assert not out['good'].any() and (out['rank'] < 6).all()
    np.testing.assert_array_equal(out['jacobian'][..., 5], 0.)


def test_frozen_lr_does_not_require_current_doca_sign(model):
    inputs, _, _, _ = setup(nonconstant(model), batch=1)
    # A current side different from frozen LR is valid, not a branch boundary.
    inputs['initial_lr'] *= -1
    ev = _Evaluator(**inputs)
    p, state = ev.candidate, ev(ev.candidate)
    out = analytic.linearize(ev, p, state)
    assert out['differentiable'].all() and out['good'].all()
    for j in (0, 4, 5):
        step = np.zeros_like(p)
        step[:, j] = 1e-5*ev.scale[:, j]
        plus, minus = ev(p+step), ev(p-step)
        same_branch(state, plus)
        same_branch(state, minus)
        np.testing.assert_allclose(out['jacobian'][..., j],
                                   (plus['residuals']-minus['residuals'])/2e-5,
                                   atol=1e-5, rtol=4e-5)
    np.testing.assert_array_equal(ev.lr, inputs['initial_lr'])


def test_valid_geometry_at_absolute_angle_branch_not_certified(model):
    inputs, truth = synthetic(model, batch=1, hits=1)
    # A horizontal wire along the transverse tangent gives q.b=0 and phi=pi/2.
    p = inputs['local_parameters'][0]
    z = inputs['z_ref'][0]
    point, tangent = scalar_helix(p, z, z)
    w = np.array([p[2], p[3], 0.])
    w /= np.linalg.norm(w)
    q = np.cross(w, tangent)
    q /= np.linalg.norm(q)
    inputs['wire_cal'][0, 0] = point-q+90*w
    inputs['wire_hv'][0, 0] = point-q-130*w
    inputs['initial_lr'] = np.sign(truth['signed_doca'])
    inputs['parameter_scale'] = np.ones((1, 6))
    # Geometric angle branches remain conservatively flagged when phi is ignored.
    inputs['model'] = affine_model(model)
    ev = _Evaluator(**inputs)
    state = ev(ev.candidate)
    assert state['objective_valid'][0]
    out = analytic.linearize(ev, ev.candidate, state)
    assert out['numeric'][0]
    assert not out['differentiable'][0] and not out['differentiable_hit'][0, 0]


@pytest.mark.parametrize('factor', [0., 1e-10, 1e-8])
def test_zero_and_nearzero_doca_guards(model, factor):
    inputs, truth = synthetic(model, batch=1)
    # Move the endpoints normally towards the exact known trajectory POCA.
    for h in range(17):
        point, _ = scalar_helix(inputs['local_parameters'][0], inputs['z_ref'][0], truth['poca_z'][0, h])
        w = inputs['wire_cal'][0, h]-inputs['wire_hv'][0, h]
        w /= np.linalg.norm(w)
        sensor = inputs['wire_cal'][0, h]-90*w
        q = point-sensor
        inputs['wire_cal'][0, h] += (1-factor)*q
        inputs['wire_hv'][0, h] += (1-factor)*q
    inputs['initial_lr'] = np.sign(truth['signed_doca'])
    inputs['parameter_scale'] = np.ones((1, 6))
    ev = _Evaluator(**inputs)
    state = ev(ev.candidate)
    out = analytic.linearize(ev, ev.candidate, state)
    if factor <= 1e-10:
        assert not state['objective_valid'][0]
        assert not out['numeric'][0] and not out['differentiable'][0]
    else:
        # No additional physical DOCA buffer replaces the existing 1e-9 mm guard.
        assert state['objective_valid'][0]
        assert out['numeric'][0]


def test_arithmetic_guard_scales_without_changing_values(model):
    dt, phi = 5*model.delta_t+1e-10, 3*model.delta_phi+1e-10
    plain = analytic.calibration_partials(model, dt, phi)
    guarded = analytic.calibration_partials(model, dt, phi, time_scale=1e5, phi_scale=1e5)
    assert plain['differentiable']
    assert not guarded['differentiable']
    assert guarded['time_boundary'] and guarded['phi_boundary']
    for name in ('c_drift', 'r_drift', 'signed_error', 'dc_dt', 'dc_dphi', 'dr_dt', 'dr_dphi'):
        np.testing.assert_array_equal(plain[name], guarded[name])


@pytest.mark.parametrize('calibration', ['offset', 'signed'])
@pytest.mark.parametrize('different', [False, True])
def test_actual_evaluator_regression_boundary_not_stationarity(model, calibration, different):
    inputs, _, p, state = setup(affine_model(model), batch=1)
    m = inputs['model']
    value = state['c_drift'][0, 3] if calibration == 'offset' else state['r_drift'][0, 3]
    intercept = np.array([0., .001 if different else 0.])
    if calibration == 'signed':
        intercept += .11
    cal = replace(getattr(m, calibration), low=value-1., width=1.,
                  intercept=intercept, slope=np.zeros(2))
    inputs['model'] = replace(m, **{calibration: cal})
    ev = _Evaluator(**inputs)
    center = ev(p)
    out = analytic.linearize(ev, p, center)
    assert center['objective_valid'][0] and out['numeric'][0] and out['good'][0]
    assert bool(out['differentiable_hit'][0, 3]) == (not different)
    assert bool(out['differentiable'][0]) == (not different)


@pytest.mark.parametrize('bad', [np.nan, np.inf, 1e300])
def test_invalid_candidate_preserves_other_lane(model, bad):
    _, ev, p, state = setup(model)
    p[0, 0] = bad
    state = ev(p)
    out = analytic.linearize(ev, p, state)
    for name in ('numeric', 'good', 'differentiable'):
        np.testing.assert_array_equal(out[name], [False, True])
    np.testing.assert_array_equal(out['jacobian'][0], 0.)


@pytest.mark.parametrize('dt,phi', [(np.nan, .3), (3., np.inf), (1e308, .3)])
def test_unsafe_calibration_queries_raise_stock_contract(model, dt, phi):
    with pytest.raises(ValueError):
        analytic.calibration_partials(model, dt, phi)
