"""Independent scalar/analytic checks of comparison-only coordinate operands."""
import math

import jax.numpy as jnp
import numpy as np
import pytest

from trkfastsim.reco import nativecompare as N


BASE = np.array([125., -730., .35, .8, 42., 380.])
LIGHT = 299.792458
MASS = 105.6583755


def analytic(x):
    """Scalar signed-impact geometry and derivatives, without fitter/JAX calls."""
    a, b, c, d, t, p = map(float, x)
    q = 1 + c * c
    h = math.sqrt(q)
    n = math.sqrt(q + d * d)
    energy = math.hypot(p, MASS)
    inv_speed = energy / (LIGHT * p)
    s = -a * c * n / q
    y = np.array([a / h, math.atan2(-1., c), b - a * c * d / q,
                  math.atan2(h, d), t + s * inv_speed, p])
    J = np.zeros((6, 6))
    J[0, 0], J[0, 2] = 1 / h, -a * c / (q * h)
    J[1, 2] = 1 / q
    J[2, :4] = [-c * d / q, 1., -a * d * (1 - c*c) / q**2, -a * c / q]
    J[3, 2], J[3, 3] = c * d / (h * n*n), -h / (n*n)
    J[4, 0] = -c * n / q * inv_speed
    J[4, 2] = -a * (n/q + c*c/(n*q) - 2*c*c*n/q**2) * inv_speed
    J[4, 3] = -a * c * d / (n*q) * inv_speed
    J[4, 4] = 1.
    J[4, 5] = -s * MASS**2 / (LIGHT * energy * p*p)
    J[5, 5] = 1.
    return y, J


def inverse_parameters(y):
    """Independent native point/direction intersection with the y=0 plane."""
    d0, phi, z0, theta, t0, p = y
    ux, uy, uz = math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)
    px, py = -d0 * math.sin(phi), d0 * math.cos(phi)
    distance = -py / uy
    return np.array([px + distance*ux, z0 + distance*uz, -ux/uy, -uz/uy,
                     t0 + distance*math.hypot(p, MASS)/(LIGHT*p), p])


def covariance():
    # Full off-diagonal structure spanning geometry, flight time, and momentum.
    a = np.arange(1., 37.).reshape(6, 6) / 37.
    a += np.diag([2., 3., .5, .8, 1.5, 4.])
    return a @ a.T


def material_D():
    return np.array([[2., -1., .2], [3., 4., -.3], [.1, -.2, .002],
                     [-.1, .3, -.001], [.4, -.5, .01], [0., 0., 1.]])


@pytest.mark.parametrize("x", [BASE, [-70., 300., -.3, -.4, 95., 800.],
                               [2., -10., 0., 0., 8., 100.],
                               [17., 32., 1.2, -2., -9., 40.]])
def test_parameters_and_full_jacobian_against_analytic_fixture(x):
    x = np.array(x)
    expected, jac = analytic(x)
    actual, cov = N.transform_parameters(jnp.asarray(x), jnp.asarray(covariance()))
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(cov, jac @ covariance() @ jac.T, rtol=3e-12, atol=3e-11)
    # D's columns expose every column of the full derivative independently.
    for start in (0, 3):
        D = np.eye(6)[:, start:start+3]
        shift, noise = N.transform_material(x, D, 1., 1., 1.)
        np.testing.assert_allclose(shift, jac[:, start+2], rtol=3e-12, atol=1e-12)
        np.testing.assert_allclose(noise, jac[:, start:start+3] @ jac[:, start:start+3].T,
                                   rtol=3e-12, atol=1e-12)
    assert isinstance(actual, np.ndarray) and isinstance(cov, np.ndarray)


def test_covariance_roundtrip_and_momentum_dependent_time():
    y, cov = N.transform_parameters(BASE, covariance())
    inv = np.linalg.inv(analytic(BASE)[1])
    np.testing.assert_allclose(inverse_parameters(y), BASE, rtol=1e-13, atol=1e-11)
    np.testing.assert_allclose(inv @ cov @ inv.T, covariance(), rtol=1e-11, atol=1e-10)
    only_momentum = np.zeros((6, 6))
    only_momentum[5, 5] = 9.
    _, transformed = N.transform_parameters(BASE, only_momentum)
    time_dp = analytic(BASE)[1][4, 5]
    assert time_dp != 0
    np.testing.assert_allclose(transformed[4, 5], 9*time_dp, rtol=2e-13)
    np.testing.assert_allclose(transformed[4, 4], 9*time_dp**2, rtol=2e-13)


def test_measurement_scalar_fixture_and_affine_origin_regression():
    H = np.array([[.2, -.4, 2., 3., -.6, .8], [-1., 2., .5, -.2, 4., -.7],
                  [.3, .2, -1., 1., .2, .9]])
    r, w = np.array([.7, -2., 4.]), np.array([2., .3, 0.])
    yr, J = analytic(BASE)
    A = np.linalg.inv(J)
    Hy = H @ A
    W, b = N.transform_measurement(BASE, H, r, w)
    expected_W = np.zeros((6, 6))
    expected_b = np.zeros(6)
    for row, residual, weight in zip(Hy, r, w):
        expected_W += weight * np.outer(row, row)
        expected_b += weight * row * (sum(row[i]*yr[i] for i in range(6)) - residual)
    np.testing.assert_allclose(W, expected_W, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(b, expected_b, rtol=2e-12, atol=2e-9)
    local_b = H.T @ (w * (H @ BASE - r))
    wrong_b = A.T @ local_b
    assert np.linalg.norm(b - wrong_b) > 100.
    np.testing.assert_allclose(b - wrong_b, W @ (yr - J @ BASE), rtol=1e-11, atol=1e-8)
    # Residuals and quadratic changes agree under the affine linearized map.
    dy = np.array([.02, -.003, .04, .002, -.01, .03])
    dx = A @ dy
    np.testing.assert_allclose(r + H @ dx, r + Hy @ dy, rtol=2e-13, atol=1e-13)
    native_change = 2*dy @ (W @ yr - b) + dy @ W @ dy
    residual_change = np.sum(w * ((r + H @ dx)**2 - r**2))
    np.testing.assert_allclose(native_change, residual_change, rtol=1e-9, atol=1e-8)


def test_material_scalar_fixture_first_order_and_separate_exact_shift():
    D = material_D()
    sv, dm, dmv = .02, -35., 4.
    yr, J = analytic(BASE)
    expected_shift = J @ (D[:, 2]*dm)
    expected_cov = np.zeros((6, 6))
    for col, variance in zip(D.T, (sv, sv, dmv)):
        expected_cov += variance * np.outer(J @ col, J @ col)
    shift, cov = N.transform_material(BASE, D, sv, dm, dmv)
    exact = N.exact_material_shift(BASE, D, dm)
    np.testing.assert_allclose(shift, expected_shift, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(cov, expected_cov, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(exact, analytic(BASE + D[:, 2]*dm)[0] - yr, rtol=2e-12, atol=1e-12)
    assert np.linalg.norm(exact - shift) > .01
    # First-order discrepancy decreases quadratically with update magnitude.
    smaller = N.exact_material_shift(BASE, D, dm/100)
    assert np.linalg.norm(smaller - shift/100) < np.linalg.norm(exact - shift)/5000


@pytest.mark.parametrize("batch", [(), (2,), (2, 3), (0,), (2, 0), (0, 3)])
def test_batch_empty_shapes_and_lane_independence(batch):
    x = np.broadcast_to(BASE, batch + (6,)).copy()
    if x.size:
        x[..., 0] += np.arange(math.prod(batch)).reshape(batch)
    C = np.broadcast_to(covariance(), batch + (6, 6)).copy()
    D = np.broadcast_to(material_D(), batch + (6, 3)).copy()
    H = np.broadcast_to(np.eye(6)[:3], batch + (3, 6)).copy()
    r, w = np.ones(batch + (3,)), np.full(batch + (3,), 2.)
    dm = np.full(batch, -2.)
    before = [a.copy() for a in (x, C, D, H, r, w, dm)]
    y, cov = N.transform_parameters(x, C)
    W, b = N.transform_measurement(x, H, r, w)
    shift, noise = N.transform_material(x, D, .01, dm, np.full(batch, 1.))
    exact = N.exact_material_shift(x, D, dm)
    assert y.shape == b.shape == shift.shape == exact.shape == batch + (6,)
    assert cov.shape == W.shape == noise.shape == batch + (6, 6)
    for index in np.ndindex(batch):
        expected_y, J = analytic(x[index])
        np.testing.assert_allclose(y[index], expected_y, rtol=1e-12, atol=1e-11)
        np.testing.assert_allclose(cov[index], J @ C[index] @ J.T, rtol=1e-12, atol=1e-10)
        single_W, single_b = N.transform_measurement(x[index], H[index], r[index], w[index])
        np.testing.assert_allclose(W[index], single_W, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(b[index], single_b, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(shift[index], J @ D[index][:, 2] * dm[index], rtol=1e-12, atol=1e-12)
    for actual, original in zip((x, C, D, H, r, w, dm), before):
        np.testing.assert_array_equal(actual, original)


def test_empty_rows_zero_operands_and_tiny_covariance_not_regularized():
    W, b = N.transform_measurement(BASE, np.empty((0, 6)), np.empty(0), np.empty(0))
    np.testing.assert_array_equal(W, np.zeros((6, 6)))
    np.testing.assert_array_equal(b, np.zeros(6))
    W, b = N.transform_measurement(BASE, np.ones((2, 6)), np.ones(2), np.zeros(2))
    np.testing.assert_array_equal(W, np.zeros((6, 6)))
    np.testing.assert_array_equal(b, np.zeros(6))
    shift, cov = N.transform_material(BASE, material_D(), 0., 0., 0.)
    np.testing.assert_array_equal(shift, np.zeros(6))
    np.testing.assert_array_equal(cov, np.zeros((6, 6)))
    tiny = np.eye(6)*1e-30
    J = analytic(BASE)[1]
    np.testing.assert_allclose(N.transform_parameters(BASE, tiny)[1], J @ tiny @ J.T, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(N.transform_parameters(BASE, np.zeros((6, 6)))[1], np.zeros((6, 6)))


@pytest.mark.parametrize("index,value,error", [(5, 0., "momentum"), (5, -1., "momentum"),
    (5, np.nan, "nonfinite"), (0, np.inf, "nonfinite"), (3, -np.inf, "nonfinite"),
    (0, 0., "singular"), (3, 2e6, "singular"), (2, 1e200, "singular")])
def test_reference_guards_apply_to_all_conversions(index, value, error):
    x = np.broadcast_to(BASE, (2, 6)).copy()
    x[1, index] = value
    calls = [lambda: N.transform_parameters(x, np.broadcast_to(np.eye(6), (2, 6, 6))),
             lambda: N.transform_measurement(x, np.ones((2, 1, 6)), np.ones((2, 1)), np.ones((2, 1))),
             lambda: N.transform_material(x, np.ones((2, 6, 3)), 1., 1., 1.),
             lambda: N.exact_material_shift(x, np.ones((2, 6, 3)), 1.)]
    for call in calls:
        with pytest.raises(ValueError, match=error):
            call()


@pytest.mark.parametrize("operand", ["C", "H", "r", "w", "D", "sv", "dm", "dmv"])
def test_nonfinite_operand_rejection(operand):
    values = dict(C=np.eye(6), H=np.ones((2, 6)), r=np.ones(2), w=np.ones(2),
                  D=material_D(), sv=np.array(.1), dm=np.array(-1.), dmv=np.array(1.))
    values[operand].flat[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        if operand == "C":
            N.transform_parameters(BASE, values["C"])
        elif operand in ("H", "r", "w"):
            N.transform_measurement(BASE, values["H"], values["r"], values["w"])
        else:
            N.transform_material(BASE, values["D"], values["sv"], values["dm"], values["dmv"])


@pytest.mark.parametrize("call", [
    lambda: N.transform_parameters(np.ones(5), np.eye(6)),
    lambda: N.transform_parameters(1., np.eye(6)),
    lambda: N.transform_parameters(BASE, np.ones((1, 6, 6))),
    lambda: N.transform_measurement(BASE, np.ones((2, 5)), np.ones(2), np.ones(2)),
    lambda: N.transform_measurement(BASE, np.ones((2, 6)), np.ones(3), np.ones(2)),
    lambda: N.transform_measurement(BASE, np.ones((2, 6)), np.ones(2), np.ones((1, 2))),
    lambda: N.transform_material(BASE, np.ones((3, 6)), 1., 1., 1.),
    lambda: N.transform_material(BASE, material_D(), np.ones(2), 1., 1.),
    lambda: N.exact_material_shift(BASE, material_D(), np.ones(2)),
])
def test_shape_mismatch_rejected_without_implicit_batch_broadcast(call):
    with pytest.raises(ValueError, match="shape"):
        call()


def test_negative_weights_variances_and_invalid_exact_endpoint():
    with pytest.raises(ValueError, match="nonnegative"):
        N.transform_measurement(BASE, np.ones((1, 6)), np.zeros(1), [-1.])
    for sv, dmv in [(-1., 0.), (0., -1.)]:
        with pytest.raises(ValueError, match="nonnegative"):
            N.transform_material(BASE, material_D(), sv, 1., dmv)
    with pytest.raises(ValueError, match="momentum"):
        N.exact_material_shift(BASE, material_D(), -BASE[5])


def test_nonfinite_computed_jacobian_and_singular_inverse_rejected(monkeypatch):
    monkeypatch.setattr(N, "_jacobian", lambda x: np.zeros((len(x), 6, 6)))
    with pytest.raises(ValueError, match="singular coordinate Jacobian"):
        N.transform_parameters(BASE, np.eye(6))
    monkeypatch.setattr(N, "_jacobian", lambda x: np.full((len(x), 6, 6), np.nan))
    with pytest.raises(ValueError, match="nonfinite.*Jacobian"):
        N.transform_parameters(BASE, np.eye(6))
