"""Analytic geometry/oracle tests; no truth enters the production refit API."""
import copy

import numpy as np
import pytest
from scipy.optimize import least_squares, root_scalar

from trkfastsim.reco.helixfit import fit_wire_helix
from trkfastsim.reco.helixseed import fit_helix_seed


def curve(p, z, z_ref=0.):
    x, y, tx, ty, w = p
    a = w*(z-z_ref)
    sinc = np.sinc(a/np.pi)
    cosc = .5*a*np.sinc(a/(2*np.pi))**2
    return np.stack((x+(z-z_ref)*(tx*sinc-ty*cosc),
                     y+(z-z_ref)*(ty*sinc+tx*cosc), z), axis=-1)


def fixture(omega=.002, batch=1, noise=0., perturb=True):
    rng = np.random.default_rng(314159)
    n = 18
    z = np.linspace(-250., 250., n)
    true = np.array([30., -15., .7, -.35, omega])
    point = curve(true, z)
    angle = np.linspace(.1, 4*np.pi+.1, n)
    d = np.stack((np.cos(angle), np.sin(angle), .12*np.sin(1.3*angle)+.04), axis=-1)
    d /= np.linalg.norm(d, axis=-1)[:, None]
    along = np.linspace(-45., 65., n)
    # Common-normal wire shifts preserve known POCA z before longitudinal noise.
    tangent = np.stack((true[2]*np.cos(omega*z)-true[3]*np.sin(omega*z),
                        true[2]*np.sin(omega*z)+true[3]*np.cos(omega*z), np.ones(n)), axis=-1)
    normal = np.cross(tangent, d)
    normal /= np.linalg.norm(normal, axis=-1)[:, None]
    mid = point-along[:, None]*d + noise*rng.normal(size=n)[:, None]*normal
    u = along+noise*3*rng.normal(size=n)
    p = true+np.array([3., -2., .02, -.015, .00015]) if perturb else true.copy()
    seed = dict(local_parameters=np.tile(p, (batch, 1)), z_ref=np.zeros(batch),
                success=np.ones(batch, bool))
    args = (np.tile(mid, (batch, 1, 1)), np.tile(d, (batch, 1, 1)),
            np.tile(u, (batch, 1)), np.full((batch, n), 9.),
            np.full((batch, n), 1000.), np.ones((batch, n), bool))
    return true, seed, args


def fit(seed, args, **kwargs):
    return fit_wire_helix(seed, *args, bz=1.0117, **kwargs)


@pytest.mark.parametrize('omega', [.002, -.003, 0., 1e-13, -1e-9])
def test_analytic_recovery_and_straight_limit(omega):
    truth, seed, args = fixture(omega)
    out = fit(seed, args)
    assert out['success'][0], (out['status'], out['gradient_norm'], out['final_objective'])
    assert out['final_objective'][0] < out['initial_objective'][0]*1e-10
    np.testing.assert_allclose(out['local_parameters'][0], truth, atol=2e-6, rtol=1e-5)
    assert out['iterations'][0] > 0
    assert out['accepted_steps'][0] > 0
    assert out['rank'][0] == 5
    assert out['condition'][0] < 1e12
    for key in ('poca_valid', 'poca_converged', 'poca_inside'):
        assert out[key].all()
    assert not out['poca_boundary'].any()
    np.testing.assert_allclose(out['doca'], 0., atol=1e-6)
    np.testing.assert_allclose(out['wire_pull'], 0., atol=1e-6)
    if abs(omega)*500 < 1e-8:
        assert not out['physical_identifiable'][0]
        assert np.isnan(out['p_abs'][0])
        assert np.isnan(out['radius'][0])
    else:
        assert out['physical_identifiable'][0]
        radius = np.hypot(truth[2], truth[3])/abs(omega)
        np.testing.assert_allclose(out['radius'][0], radius, rtol=2e-4)
        np.testing.assert_allclose(out['pitch'][0], 1/omega, rtol=2e-4)
        np.testing.assert_allclose(out['pt_abs'][0], .299792458*1.0117*radius, rtol=2e-4)
        radial = out['radius'][0]*np.array([np.cos(out['phase_ref'][0]), np.sin(out['phase_ref'][0])])
        np.testing.assert_allclose(out['center_xy'][0]+radial, truth[:2], atol=1e-6)


@pytest.mark.parametrize('sign', [-1, 1])
def test_actual_fit_helix_seed_api(sign):
    truth, _, args = fixture(sign*.002)
    positions = curve(truth, np.linspace(-250., 250., 18))[None]
    seed = fit_helix_seed(positions, args[-1], bz=1.0117)
    assert seed['success'].all()
    seed['center_xy'] += [3., -2.]
    seed['radius'] *= 1.01
    out = fit(seed, args)
    assert out['success'].all(), out['status']
    z_ref = seed['z_ref'][0]
    np.testing.assert_allclose(out['local_parameters'][0, :2], curve(truth, np.array(z_ref))[:2], atol=1e-6)
    np.testing.assert_allclose(out['local_parameters'][0, 4], truth[4], atol=1e-9)


def oracle_residual(q, initial, scale, args):
    """Independent scalar root solve per wire; NOT the production POCA/Jacobian."""
    p = initial+scale*q
    mid, directions, measured, uvar, _, valid = args
    residual = []
    for m, d, u, var, keep in zip(mid[0], directions[0], measured[0], uvar[0], valid[0]):
        if not keep:
            continue
        d = d/np.linalg.norm(d)
        z0 = m[2]+u*d[2]

        def point_tangent(z):
            # Independent complex-integral representation of local trajectory.
            w = p[4]
            t = complex(p[2], p[3])
            integral = z if w == 0 else np.expm1(1j*w*z)/(1j*w)
            xy = complex(p[0], p[1])+t*integral
            v = t*np.exp(1j*w*z)
            return np.array([xy.real, xy.imag, z]), np.array([v.real, v.imag, 1.])

        def gradient(z):
            point, tangent = point_tangent(z)
            delta = point-m
            r = delta-np.dot(delta, d)*d
            return np.dot(r, tangent)

        root = root_scalar(gradient, bracket=(z0-100., z0+100.), xtol=1e-11).root
        point, _ = point_tangent(root)
        delta = point-m
        predicted = np.dot(delta, d)
        residual.extend((delta-predicted*d)/2.5)
        residual.append((predicted-u)/np.sqrt(var))
    return np.array(residual)


@pytest.mark.parametrize('omega', [.002, -.002, 0.])
def test_noisy_scalar_scipy_oracle(omega):
    _, seed, args = fixture(omega, noise=.8)
    out = fit(seed, args)
    assert out['success'].all(), (out['status'], out['gradient_norm'])
    p0 = seed['local_parameters'][0]
    scale = out['parameter_scale'][0]
    oracle = least_squares(oracle_residual, np.zeros(5), args=(p0, scale, args),
                           xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=150,
                           jac='3-point', diff_step=1e-4)
    assert oracle.success
    np.testing.assert_allclose(out['final_objective'][0], oracle.cost, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose((out['local_parameters'][0]-p0)/scale, oracle.x, atol=2e-4, rtol=2e-5)
    np.testing.assert_allclose(out['residuals'][0].ravel(), oracle.fun, atol=2e-5)
    assert out['final_objective'][0] < out['initial_objective'][0]


def test_no_input_mutation_and_exact_batch_invariance():
    _, seed, args = fixture(batch=4, noise=.4)
    seed['local_parameters'][1, 0] += 1.
    seed['success'][2] = False
    args[-1][3] = False
    args[0][3] = np.nan
    before = copy.deepcopy((seed, args))
    out = fit(seed, args)
    for i in range(4):
        separate = fit({k: v[i:i+1] for k, v in seed.items()}, tuple(a[i:i+1] for a in args))
        for key in out:
            np.testing.assert_array_equal(out[key][i:i+1], separate[key], err_msg=key)
    for key in seed:
        np.testing.assert_array_equal(seed[key], before[0][key])
    for a, b in zip(args, before[1]):
        np.testing.assert_array_equal(a, b)
    assert out['status'].tolist()[2:] == ['invalid_seed', 'empty']
    assert not out['success'][2:].any()


@pytest.mark.parametrize('index', range(5))
def test_selected_nonfinite_fails_whole_lane(index):
    _, seed, args = fixture(batch=2)
    args[index][0, 2] = np.nan
    out = fit(seed, args)
    assert out['status'][0] == 'invalid_input'
    assert not out['success'][0]
    assert out['success'][1]
    assert out['n_hits'][0] == 18
    assert not out['objective_valid'][0]
    assert not out['poca_valid'][0].any()
    assert np.isnan(out['p_abs'][0])


@pytest.mark.parametrize('index,value', [(1, 0.), (3, 0.), (3, -1.), (4, -1.)])
def test_nonphysical_hit_fails_lane(index, value):
    _, seed, args = fixture()
    args[index][0, 0] = value
    out = fit(seed, args)
    assert out['status'][0] == 'invalid_input'
    assert not out['success'][0]


def test_masked_payloads_do_not_change_fit():
    _, seed, args = fixture()
    args[-1][:, -3:] = False
    baseline = fit(seed, args)
    for a in args[:-1]:
        a[:, -3:] = np.nan
    changed = fit(seed, args)
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], changed[key], err_msg=key)
    assert changed['success'][0]
    assert changed['n_hits'][0] == 15
    assert not changed['poca_valid'][:, -3:].any()


@pytest.mark.parametrize('batch,hits', [(0, 18), (0, 0), (3, 0), (2, 3)])
def test_empty_and_insufficient_shapes(batch, hits):
    seed = dict(local_parameters=np.zeros((batch, 5)), z_ref=np.zeros(batch))
    args = (np.zeros((batch, hits, 3)), np.ones((batch, hits, 3)),
            np.zeros((batch, hits)), np.ones((batch, hits)),
            np.ones((batch, hits)), np.ones((batch, hits), bool))
    out = fit(seed, args)
    assert out['success'].shape == (batch,)
    assert out['residuals'].shape == (batch, hits, 4)
    assert not out['success'].any()
    assert np.all(out['n_hits'] == hits)
    if batch:
        assert np.all(out['status'] == ('empty' if hits == 0 else 'insufficient_hits'))


def test_repeated_wires_rank_failure_not_seed_success():
    _, seed, args = fixture(perturb=False)
    args = tuple(np.repeat(a[:, :1], 18, axis=1) for a in args)
    out = fit(seed, args)
    assert not out['success'][0]
    assert out['status'][0] == 'ill_conditioned'
    assert out['rank'][0] < 5
    assert np.isfinite(out['condition'][0])
    assert np.isnan(out['p_abs'][0])


def test_parallel_track_wire_has_no_strict_poca():
    _, seed, args = fixture(omega=0., perturb=False)
    args[1][:] = [0.7, -.35, 1.]
    out = fit(seed, args)
    assert not out['success'][0]
    assert out['status'][0] == 'initial_poca_failed'
    assert not out['poca_converged'].all()


def test_finite_lengths_required_for_every_selected_hit():
    _, seed, args = fixture(perturb=False)
    args[4][0, 0] = 1.
    out = fit(seed, args)
    assert not out['success'][0]
    assert out['status'][0] == 'initial_poca_failed'
    assert out['n_hits'][0] == 18
    assert not out['poca_inside'][0, 0]
    args[-1][0, 0] = False
    assert fit(seed, args)['success'][0]


@pytest.mark.parametrize('budget', [0, 1])
def test_iteration_exhaustion_not_success(budget):
    _, seed, args = fixture()
    out = fit(seed, args, max_iterations=budget)
    assert not out['success'][0]
    assert out['status'][0] == 'max_iterations'
    assert out['iterations'][0] == budget
    assert out['final_objective'][0] <= out['initial_objective'][0]
    assert np.isnan(out['p_abs'][0])
    assert np.isfinite(out['local_parameters']).all()


def test_stationary_seed_and_zero_iteration_check():
    _, seed, args = fixture(perturb=False)
    out = fit(seed, args, max_iterations=0)
    assert out['success'][0]
    assert out['iterations'][0] == 0
    assert out['final_objective'][0] < 1e-24


@pytest.mark.parametrize('options', [dict(poca_max_iterations=0), dict(poca_max_offset=1e-10),
                                     dict(poca_max_iterations=1, poca_max_step=1e-10)])
def test_poca_exhaustion_and_boundary_stall(options):
    _, seed, args = fixture()
    out = fit(seed, args, **options)
    assert not out['success'][0]
    assert out['status'][0] == 'initial_poca_failed'
    assert not out['poca_converged'].all()


def test_direction_scale_and_reversal():
    _, seed, args = fixture(noise=.3)
    baseline = fit(seed, args)
    args[1][:] *= -1e200
    args[2][:] *= -1
    out = fit(seed, args)
    assert out['success'][0]
    np.testing.assert_allclose(out['local_parameters'], baseline['local_parameters'], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out['final_objective'], baseline['final_objective'], rtol=1e-12)
    np.testing.assert_allclose(out['wire_u'], -baseline['wire_u'], atol=1e-9)


@pytest.mark.parametrize('where', ['local', 'z', 'success'])
def test_invalid_seed_preserved(where):
    _, seed, args = fixture(batch=2)
    if where == 'local':
        seed['local_parameters'][0, 0] = np.inf
    elif where == 'z':
        seed['z_ref'][0] = np.nan
    else:
        seed['success'][0] = False
    out = fit(seed, args)
    assert out['status'][0] == 'invalid_seed'
    assert not out['success'][0]
    assert out['success'][1]
    assert np.isnan(out['local_parameters'][0]).all()


@pytest.mark.parametrize('kwargs', [dict(max_iterations=-1), dict(max_iterations=True),
                                    dict(max_iterations=1.2), dict(transverse_scale=0),
                                    dict(transverse_scale=np.inf), dict(fd_step=0),
                                    dict(condition_limit=1), dict(max_rows=0),
                                    dict(max_cells=1), dict(poca_tolerance=0),
                                    dict(poca_max_iterations=-1), dict(initial_damping=1e13)])
def test_configuration_errors(kwargs):
    _, seed, args = fixture()
    with pytest.raises(ValueError):
        fit(seed, args, **kwargs)


@pytest.mark.parametrize('bz', [0., np.nan, np.inf, True, [1.]])
def test_bad_field(bz):
    _, seed, args = fixture()
    with pytest.raises(ValueError):
        fit_wire_helix(seed, *args, bz=bz)


def test_shape_and_dtype_errors():
    _, seed, args = fixture()
    for i in range(len(args)):
        malformed = list(args)
        malformed[i] = args[i][0]
        with pytest.raises(ValueError):
            fit(seed, malformed)
    malformed = list(args)
    malformed[-1] = args[-1].astype(int)
    with pytest.raises(ValueError):
        fit(seed, malformed)
    seed['success'] = np.ones(1)
    with pytest.raises(ValueError):
        fit(seed, args)


def test_huge_finite_geometry_fails_without_nonfinite_diagnostics():
    _, seed, args = fixture()
    args[0][:] = 1e308
    out = fit(seed, args)
    assert not out['success'][0]
    for key in ('condition', 'eigenvalues', 'gradient_norm', 'damping', 'initial_objective',
                'final_objective', 'residuals', 'doca', 'wire_u', 'wire_pull', 'poca_z'):
        assert np.isfinite(out[key]).all(), key


def test_lm_cannot_succeed_by_stalling_at_a_finite_wire_endpoint():
    _, seed, args = fixture(noise=.8)
    initial = fit(seed, args, max_iterations=0)
    unconstrained = fit(seed, args)
    assert unconstrained['success'][0]
    growth = np.abs(unconstrained['wire_u'][0])-np.abs(initial['wire_u'][0])
    j = np.argmax(growth)
    assert growth[j] > 1.
    args[4][0, j] = .5*(abs(unconstrained['wire_u'][0, j])+abs(initial['wire_u'][0, j]))
    out = fit(seed, args)
    assert not out['success'][0]
    assert out['status'][0] in ('max_iterations', 'stalled')
    assert out['objective_valid'][0]
    assert out['poca_inside'].all()  # Last ACCEPTED iterate remains feasible.
    assert out['n_hits'][0] == 18
    assert out['gradient_norm'][0] > 1e-4
    assert np.isnan(out['p_abs'][0])


def test_tilt_is_not_flattened_and_longitudinal_measurements_are_used():
    _, seed, args = fixture(noise=.2)
    baseline = fit(seed, args)
    changed = copy.deepcopy(args)
    changed[1][..., 2] = 0.
    flat = fit(seed, changed)
    assert np.linalg.norm(flat['local_parameters']-baseline['local_parameters']) > .01
    changed = copy.deepcopy(args)
    changed[2][:] += np.linspace(-2., 5., 18)
    longitudinal = fit(seed, changed)
    assert longitudinal['success'][0]
    assert np.linalg.norm(longitudinal['local_parameters']-baseline['local_parameters']) > .01


def test_translated_reference_parameterization():
    truth, seed, args = fixture()
    shift = np.array([500., -800., 1e6])
    args[0][:] += shift
    seed['local_parameters'][:, :2] += shift[:2]
    seed['z_ref'][:] += shift[2]
    out = fit(seed, args)
    assert out['success'][0], out['status']
    truth[:2] += shift[:2]
    np.testing.assert_allclose(out['local_parameters'][0], truth, atol=2e-6, rtol=1e-7)
    np.testing.assert_array_equal(out['z_ref'], seed['z_ref'])


def test_zero_transverse_slope_cannot_identify_curvature():
    _, seed, args = fixture(omega=0., perturb=False)
    seed['local_parameters'][0, 2:] = 0.
    args[0][0, :, :2] = seed['local_parameters'][0, :2]-args[2][0, :, None]*args[1][0, :, :2]
    out = fit(seed, args)
    assert not out['success'][0]
    assert out['status'][0] == 'ill_conditioned'
    assert out['rank'][0] < 5
    assert not out['physical_identifiable'][0]


@pytest.mark.parametrize('component', [0, 2, 4])
def test_huge_finite_parameters_do_not_contaminate_other_lanes(component):
    _, seed, args = fixture(batch=2)
    seed['local_parameters'][0, component] = 1e308
    out = fit(seed, args)
    assert not out['success'][0]
    assert out['success'][1]
    assert out['status'][0] == 'initial_poca_failed'
    for key in ('condition', 'eigenvalues', 'gradient_norm', 'damping', 'initial_objective',
                'final_objective', 'residuals', 'doca', 'wire_u', 'wire_pull', 'poca_z'):
        assert np.isfinite(out[key]).all(), key
    separate = fit({k: v[1:] for k, v in seed.items()}, tuple(a[1:] for a in args))
    for key in out:
        np.testing.assert_array_equal(out[key][1:], separate[key], err_msg=key)


def test_selected_invalid_geometry_with_no_fit_budget_is_not_accepted():
    _, seed, args = fixture(perturb=False)
    args[2][0, 0] = np.inf
    out = fit(seed, args, max_iterations=0)
    assert not out['success'][0]
    assert out['status'][0] == 'invalid_input'
    assert out['n_hits'][0] == 18


def test_condition_limit_is_a_failure_not_a_ridge_solution():
    _, seed, args = fixture()
    out = fit(seed, args, condition_limit=1.001)
    assert not out['success'][0]
    assert out['status'][0] == 'ill_conditioned'
    assert out['iterations'][0] == 0
    np.testing.assert_array_equal(out['local_parameters'], seed['local_parameters'])
