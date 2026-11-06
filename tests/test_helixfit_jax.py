"""Independent synthetic geometry checks; no CPU fitter needed except optional parity."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
from trkfastsim.reco import helixfit_jax as fit


def geometry(batch=1, hits=12, omega=.002):
    p = np.tile([35., -18., .55, -.25, omega], (batch, 1))
    z_ref = np.full(batch, 80.)
    z = np.linspace(-240., 400., hits)
    dz = z-z_ref[0]
    if omega:
        x = p[0, 0]+(p[0, 2]*np.sin(omega*dz)-p[0, 3]*(1-np.cos(omega*dz)))/omega
        y = p[0, 1]+(p[0, 3]*np.sin(omega*dz)+p[0, 2]*(1-np.cos(omega*dz)))/omega
    else:
        x, y = p[0, 0]+p[0, 2]*dz, p[0, 1]+p[0, 3]*dz
    point = np.stack((x, y, z), -1)
    angle = np.arange(hits)*1.23
    direction = np.stack((np.cos(angle), np.sin(angle), .25*np.sin(.7+angle)), -1)
    direction /= np.linalg.norm(direction, axis=-1)[:, None]
    u = 25.*np.cos(angle*.3)
    mid = point-u[:, None]*direction
    return [p, z_ref, np.tile(mid, (batch, 1, 1)), np.tile(direction, (batch, 1, 1)),
            np.tile(u, (batch, 1)), np.full((batch, hits), 9.), np.full((batch, hits), 500.),
            np.ones((batch, hits), bool)]


def perturbed(batch=1):
    args = geometry(batch)
    args[0] += np.array([1.8, -1.2, .008, -.006, 2e-5])
    return args


compiled = jax.jit(fit.fit_wire_helix)
no_budget = jax.jit(lambda *a: fit.fit_wire_helix(*a, max_iterations=0))


@pytest.mark.parametrize('omega', [.002, -.002, 0.])
def test_exact_geometry_and_straight_limit(omega):
    args = geometry(omega=omega)
    out = no_budget(*args)
    assert bool(out['success'][0])
    assert int(out['status'][0]) == fit.CONVERGED
    assert int(out['rank'][0]) == 5
    assert int(out['iterations'][0]) == int(out['accepted_steps'][0]) == 0
    assert np.all(out['poca_valid'] & out['poca_converged'] & out['poca_inside'])
    assert not np.any(out['poca_boundary'])
    assert float(out['final_objective'][0]) < 1e-22
    np.testing.assert_allclose(out['local_parameters'], args[0], atol=0, rtol=0)
    np.testing.assert_allclose(out['wire_u'], args[4], atol=2e-12, rtol=0)
    np.testing.assert_allclose(out['poca_z'][0], np.linspace(-240., 400., 12), atol=2e-12, rtol=0)
    assert out['status'].dtype == np.dtype('int32')
    assert all(isinstance(value, jax.Array) for value in out.values())
    assert not {'momentum', 'p_abs', 'pt_abs', 'charge', 'covariance', 'flight_direction'} & out.keys()


def test_perturbed_fit_strict_improvement_and_recovery():
    args = perturbed()
    out = compiled(*args)
    assert bool(out['success'][0])
    assert int(out['accepted_steps'][0]) > 0
    assert 0 <= float(out['final_objective'][0]) < float(out['initial_objective'][0])*1e-10
    np.testing.assert_allclose(out['local_parameters'], geometry()[0], rtol=2e-7, atol=3e-7)
    assert float(out['gradient_norm'][0]) <= 1e-7*(1+np.sqrt(2*float(out['final_objective'][0])))
    assert np.all(out['poca_converged'] & out['poca_inside'])


def test_no_budget_and_last_budget_gradient_not_update_proxy():
    args = perturbed()
    out = no_budget(*args)
    assert bool(out['objective_valid'][0]) and not bool(out['success'][0])
    assert int(out['status'][0]) == fit.MAX_ITERATIONS
    np.testing.assert_array_equal(out['local_parameters'], args[0])
    np.testing.assert_array_equal(out['initial_objective'], out['final_objective'])
    one = fit.fit_wire_helix(*args, max_iterations=1)
    assert int(one['accepted_steps'][0]) == 1
    assert float(one['final_objective'][0]) < float(one['initial_objective'][0])
    assert not bool(one['success'][0])
    assert float(one['gradient_norm'][0]) > 1e-7*(1+np.sqrt(2*float(one['final_objective'][0])))
    clipped = fit.fit_wire_helix(*args, max_iterations=2, max_parameter_step=1e-9)
    assert not bool(clipped['success'][0])


def test_final_budget_convergence_uses_recomputed_gradient():
    args = perturbed()
    out = fit.fit_wire_helix(*args, max_iterations=3)
    assert bool(out['success'][0])
    assert int(out['accepted_steps'][0]) == int(out['iterations'][0]) == 3
    replay_args = [np.asarray(out['local_parameters'])]+args[1:]
    replay = no_budget(*replay_args)
    assert bool(replay['success'][0])
    np.testing.assert_allclose(replay['gradient_norm'], out['gradient_norm'], atol=1e-11)


def test_rejected_finite_wire_trial_preserves_last_accepted_state():
    args = perturbed()
    initial = no_budget(*args)
    wire_u = np.asarray(initial['wire_u'][0])
    index = int(np.argmax(np.abs(args[4][0])-np.abs(wire_u)))
    args[6][0, index] = abs(wire_u[index])+1e-9
    out = fit.fit_wire_helix(*args, max_iterations=5)
    assert bool(out['objective_valid'][0]) and not bool(out['success'][0])
    assert int(out['iterations'][0]) == 5 and int(out['accepted_steps'][0]) == 0
    np.testing.assert_array_equal(out['local_parameters'], args[0])
    np.testing.assert_array_equal(out['initial_objective'], out['final_objective'])
    np.testing.assert_allclose(out['residuals'], initial['residuals'], atol=1e-12)
    assert np.all(out['poca_inside'] & out['poca_converged'])
    assert int(out['n_hits'][0]) == 12


def test_bad_lanes_never_poison_peer_and_membership_is_fixed():
    args = geometry(batch=9)
    args[0][1, 0] = np.nan
    args[1][2] = np.inf
    args[2][3, 0, 0] = np.nan
    args[3][4, 0] = 0.
    args[5][5, 0] = 0.
    args[6][6, 0] = -1.
    args[0][7, 2] = 1e200
    args[7][8] = False
    out = compiled(*args)
    np.testing.assert_array_equal(out['success'], [True]+[False]*8)
    np.testing.assert_array_equal(out['status'], [fit.CONVERGED, fit.INVALID_SEED, fit.INVALID_SEED,
                                               fit.INVALID_INPUT, fit.INVALID_INPUT, fit.INVALID_INPUT,
                                               fit.INVALID_INPUT, fit.INITIAL_POCA_FAILED, fit.EMPTY])
    np.testing.assert_array_equal(out['n_hits'], [12]*8+[0])
    for key in ('condition', 'gradient_norm', 'initial_objective', 'final_objective', 'residuals', 'doca', 'wire_u'):
        assert np.all(np.isfinite(out[key])), key
    single = no_budget(*geometry())
    np.testing.assert_allclose(out['local_parameters'][0], single['local_parameters'][0], atol=1e-12)
    assert not np.any(out['poca_valid'][1:7])


def test_masked_nan_payloads_seed_valid_and_insufficient_hits():
    args = geometry(batch=3)
    args[7][:, -1] = False
    for a in args[2:7]:
        a[:, -1] = np.nan
    args[7][2, 3:] = False
    out = jax.jit(lambda *a: fit.fit_wire_helix(*a, seed_valid=jax.numpy.array([True, False, True])))(*args)
    np.testing.assert_array_equal(out['status'], [fit.CONVERGED, fit.INVALID_SEED, fit.INSUFFICIENT_HITS])
    np.testing.assert_array_equal(out['n_hits'], [11, 11, 3])
    assert not np.any(out['poca_valid'][:, -1])
    np.testing.assert_array_equal(out['residuals'][:, -1], 0.)
    assert not np.any(out['objective_valid'][1:])


def test_finite_wire_rejection_does_not_drop_hit():
    args = geometry()
    args[6][0, 0] = 1.
    out = compiled(*args)
    assert int(out['status'][0]) == fit.INITIAL_POCA_FAILED
    assert not bool(out['success'][0]) and not bool(out['objective_valid'][0])
    assert int(out['n_hits'][0]) == 12
    assert not bool(out['poca_inside'][0, 0])
    np.testing.assert_array_equal(out['local_parameters'], args[0])


def test_poca_zero_budget_and_bounded_basin_failure():
    args = perturbed()
    for kw in ({'poca_max_iterations': 0}, {'poca_max_offset': 1e-12}):
        out = fit.fit_wire_helix(*args, **kw)
        assert not bool(out['success'][0])
        assert int(out['status'][0]) == fit.INITIAL_POCA_FAILED
        np.testing.assert_array_equal(out['local_parameters'], args[0])


def test_singular_and_derived_normal_overflow_are_failures():
    args = geometry(batch=3)
    args[0][1:] = [0., 0., .5, .25, 0.]
    args[1][1:] = 0.
    args[2][1:] = [0., 0., 0.]
    args[3][1:] = [1., 0., 0.]
    args[4][1:] = 0.
    args[5][2] = 3e-308
    out = no_budget(*args)
    assert bool(out['success'][0])
    np.testing.assert_array_equal(out['status'][1:], fit.ILL_CONDITIONED)
    assert np.all(np.isfinite(out['condition']))
    assert np.all(np.isfinite(out['eigenvalues']))
    assert np.all(np.asarray(out['rank'][1:]) < 5)
    assert not np.any(out['success'][1:])


def test_jit_and_eager_equivalence():
    args = perturbed()
    eager, jit = fit.fit_wire_helix(*args), compiled(*args)
    assert eager.keys() == jit.keys()
    for key in eager:
        if eager[key].dtype.kind in 'biu':
            np.testing.assert_array_equal(eager[key], jit[key], err_msg=key)
        else:
            np.testing.assert_allclose(eager[key], jit[key], atol=2e-9, rtol=2e-8, equal_nan=True, err_msg=key)
    jaxpr = str(jax.make_jaxpr(fit.fit_wire_helix)(*args))
    assert 'callback' not in jaxpr
    assert 'while[' in jaxpr


@pytest.mark.parametrize('batch,hits', [(0, 12), (2, 0), (0, 0)])
def test_empty_axes(batch, hits):
    args = [np.zeros((batch, 5)), np.zeros(batch), np.zeros((batch, hits, 3)), np.zeros((batch, hits, 3)),
            np.zeros((batch, hits)), np.ones((batch, hits)), np.ones((batch, hits)), np.zeros((batch, hits), bool)]
    out = compiled(*args)
    assert out['local_parameters'].shape == (batch, 5)
    assert out['residuals'].shape == (batch, hits, 4)
    assert out['status'].shape == (batch,)
    assert np.all(np.asarray(out['status']) == fit.EMPTY)
    assert not np.any(out['success'])


@pytest.mark.parametrize('kw', [{'max_rows': 0}, {'max_cells': 11}, {'max_iterations': -1},
                               {'max_iterations': True}, {'poca_max_iterations': 1.2},
                               {'transverse_scale': 0}, {'fd_step': np.nan}, {'condition_limit': 1},
                               {'initial_damping': 1e13}])
def test_static_configuration_checks(kw):
    with pytest.raises(ValueError):
        fit.fit_wire_helix(*geometry(), **kw)


@pytest.mark.parametrize('index,value', [(0, np.zeros((1, 4))), (1, np.zeros((1, 1))),
                                        (3, np.ones((1, 12, 2))), (7, np.ones((1, 12)))])
def test_shape_and_dtype_checks(index, value):
    args = geometry()
    args[index] = value
    with pytest.raises(ValueError):
        fit.fit_wire_helix(*args)
    with pytest.raises(ValueError):
        fit.fit_wire_helix(*geometry(), seed_valid=np.ones(1))


@pytest.mark.parametrize('omega', [.002, 0.])
def test_optional_numpy_reference_parity(omega):
    cpu = pytest.importorskip('trkfastsim.reco.helixfit')
    args = geometry(omega=omega)
    args[0] += np.array([1.8, -1.2, .008, -.006, 2e-5])
    ref = cpu.fit_wire_helix({'local_parameters': args[0], 'z_ref': args[1]}, *args[2:], bz=1.)
    out = compiled(*args)
    assert bool(out['success'][0]) == bool(ref['success'][0])
    assert fit.STATUS_NAMES[int(out['status'][0])] == ref['status'][0]
    for key in ('local_parameters', 'initial_objective', 'final_objective', 'condition',
                'gradient_norm', 'parameter_scale', 'residuals', 'doca', 'wire_u', 'poca_z'):
        np.testing.assert_allclose(out[key], ref[key], rtol=2e-7, atol=3e-8, err_msg=key)
    for key in ('n_hits', 'iterations', 'accepted_steps', 'rank', 'poca_valid', 'poca_converged', 'poca_inside'):
        np.testing.assert_array_equal(out[key], ref[key], err_msg=key)
