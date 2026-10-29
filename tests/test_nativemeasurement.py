"""Independent analytic-line/autodiff checks of the diagnostic native row policy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trkfastsim.reco import nativemeasurement as N

jax.config.update("jax_enable_x64", True)
Q = np.array([45., -1.1, 280., 1.4, 35., 450.])
MID = np.array([250., 180., 390.])
WIRE = np.array([.8, .6, 0.])


def position(q, time):
    d, f, z, th, t, p = q
    u = jnp.array([jnp.sin(th)*jnp.cos(f), jnp.sin(th)*jnp.sin(f), jnp.cos(th)])
    return jnp.array([-d*jnp.sin(f), d*jnp.cos(f), z]) + (time-t)*299.792458*p/jnp.hypot(p, 105.6583755)*u


def momentum(q):
    return q[5]*jnp.array([jnp.sin(q[3])*jnp.cos(q[1]), jnp.sin(q[3])*jnp.sin(q[1]), jnp.cos(q[3])])


def residual(q, mid, wire, sign, vs):
    p = jnp.array([-q[0]*jnp.sin(q[1]), q[0]*jnp.cos(q[1]), q[2]])
    u = momentum(q)/q[5]
    # Solve orthogonality as a linear system, independently of the closed CA expressions.
    axes = jnp.stack((u, -wire), axis=-1)
    st, sw = jnp.linalg.solve(axes.T@axes, axes.T@(mid-p))
    delta = mid+sw*wire-(p+st*u)
    sense = jnp.where(jnp.dot(jnp.cross(sign*wire, u), delta) < 0, -1., 1.)
    doca = sense*jnp.linalg.norm(delta)
    speed = 299.792458*q[5]/jnp.hypot(q[5], 105.6583755)
    return jnp.array([-doca, 70.-(500.-sign*sw)/vs-q[4]-st/speed, sign*sw])


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("mom", [40., 450., 2000.])
def test_fixed_time_kinematics_and_exact_omitted_direction_term(sign, mom):
    q = Q.copy(); q[5] = mom
    r = N.rows(q, MID, WIRE, np.array(sign), np.array(180.))
    t = r["particle_time"]
    pos, _, _, dx, dm = N.kinematics(q, t)
    np.testing.assert_allclose(pos, position(q, t), rtol=2e-13, atol=2e-11)
    np.testing.assert_allclose(dx, jax.jacfwd(position)(q, t), rtol=2e-13, atol=2e-11)
    np.testing.assert_allclose(dm, jax.jacfwd(momentum)(q)/mom, rtol=2e-13, atol=2e-13)
    full = np.asarray(jax.jacfwd(residual)(q, MID, WIRE, sign, 180.))
    np.testing.assert_allclose(r["H"][0], full[0], rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(r["H"][2]+r["full_minus_native_long"], full[2], rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(r["H"][1]+r["omitted_time_direction"], full[1], rtol=2e-12, atol=2e-11)
    assert np.max(np.abs(r["omitted_time_direction"][[1, 3]])) > .01
    np.testing.assert_array_equal(r["omitted_time_direction"][[0, 2, 4, 5]], 0.)
    np.testing.assert_allclose(r["doca"], -residual(q, MID, WIRE, sign, 180.)[0], atol=1e-11)


@pytest.mark.parametrize("wire", [np.array([.3, -.4, np.sqrt(.75)]), np.array([0., 1., 0.]), WIRE])
@pytest.mark.parametrize("q", [Q, np.array([-80., -.6, -500., 2.2, -30., 120.])])
def test_source_rows_and_corrections_independently(wire, q):
    mid, sign, vs = MID + np.array([40., -10., 90.]), -1., 160.
    u = np.asarray(momentum(q))/q[5]
    origin = np.asarray(position(q, q[4]))
    A = np.column_stack((u, -wire))
    st, sw = np.linalg.lstsq(A, mid-origin, rcond=None)[0]
    speed = 299.792458*q[5]/np.hypot(q[5], 105.6583755)
    time = q[4]+st/speed
    X = np.asarray(jax.jacfwd(position)(q, time))
    dm = np.asarray(jax.jacfwd(momentum)(q))/q[5]
    du = np.asarray(jax.jacfwd(lambda v: momentum(v)/v[5])(q))
    delta = mid+sw*wire-(origin+st*u)
    s = sign*wire
    c = s@u
    den = 1-c*c
    native_time = -((c*s-u)/(speed*den)-(s-c*u)/(vs*den))@X
    native_long = (s@X + c*(c*s@X-u@X-delta@dm)/den)
    dtime = (c/vs-1/speed)*(delta@du)/den
    dlong = 2*c*(delta@du)/den
    actual = N.rows(q, mid, wire, np.array(sign), np.array(vs))
    np.testing.assert_allclose(actual["H"][1], native_time, rtol=1e-12, atol=2e-11)
    np.testing.assert_allclose(actual["H"][2], native_long, rtol=1e-12, atol=2e-11)
    np.testing.assert_allclose(actual["omitted_time_direction"], dtime, rtol=1e-12, atol=2e-11)
    np.testing.assert_allclose(actual["full_minus_native_long"], dlong, rtol=1e-12, atol=2e-11)
    np.testing.assert_allclose([delta@s, delta@u], 0., atol=1e-11)


def test_information_matches_native_weight_sign_and_null_row_flip():
    H = N.rows(Q, MID, WIRE, np.array(1.), np.array(180.))["H"]
    r, w = np.array([3., 8., -12.]), np.array([.2, .03, .001])
    actual = N.information(Q, H, r, w)
    dRdP = -H
    W = sum(wi*np.outer(di, di) for wi, di in zip(w, dRdP))
    b = W@Q + sum(wi*di*ri for wi, di, ri in zip(w, dRdP, r))
    np.testing.assert_allclose(actual[0], W, rtol=1e-13, atol=1e-11)
    np.testing.assert_allclose(actual[1], b, rtol=1e-13, atol=1e-10)
    # Native null residual is +doca; the F1 row is -doca with both derivative and value flipped.
    H2, r2 = H.copy(), r.copy()
    H2[0] *= -1; r2[0] *= -1
    for a, b in zip(actual, N.information(Q, H2, r2, w)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("shape", [(0,), (2,), (2, 3)])
def test_empty_and_multidimensional_batches(shape):
    result = N.rows(np.broadcast_to(Q, shape+(6,)), np.broadcast_to(MID, shape+(3,)),
                    np.broadcast_to(WIRE, shape+(3,)), np.ones(shape), np.full(shape, 180.))
    assert result["H"].shape == shape+(3, 6)
    if np.prod(shape):
        expected = N.rows(Q, MID, WIRE, np.array(1.), np.array(180.))["H"]
        np.testing.assert_allclose(result["H"], np.broadcast_to(expected, result["H"].shape))
    W, b = N.information(np.broadcast_to(Q, shape+(6,)), result["H"], np.zeros(shape+(3,)), np.ones(shape+(3,)))
    assert W.shape == shape+(6, 6) and b.shape == shape+(6,)


@pytest.mark.parametrize("field,value,error", [
    ("parameters", np.zeros(5), "shape"),
    ("parameters", np.array([45., -1.1, 280., 1.4, 35., 0.]), "positive"),
    ("parameters", Q*np.nan, "nonfinite"),
    ("midpoint", np.array([np.inf, 0., 0.]), "nonfinite"),
    ("wire_direction", 2*WIRE, "unit"),
    ("end_sign", np.array(0.), "end sign"),
    ("signal_speed", np.array(-1.), "signal speed"),
    ("signal_speed", np.array([180.]), "shape"),
])
def test_guards(field, value, error):
    args = dict(parameters=Q, midpoint=MID, wire_direction=WIRE, end_sign=np.array(1.), signal_speed=np.array(180.))
    args[field] = value
    with pytest.raises(ValueError, match=error):
        N.rows(**args)


def test_parallel_zero_doca_and_information_guards():
    _, direction, _, _, _ = N.kinematics(Q, Q[4])
    with pytest.raises(ValueError, match="parallel"):
        N.rows(Q, MID, direction, np.array(1.), np.array(180.))
    q = np.array([0., 0., 0., np.pi/2, 0., 100.])
    with pytest.raises(ValueError, match="zero-distance"):
        N.rows(q, np.zeros(3), np.array([0., 1., 0.]), np.array(1.), np.array(180.))
    with pytest.raises(ValueError, match="negative"):
        N.information(Q, np.zeros((3, 6)), np.zeros(3), -np.ones(3))
    with pytest.raises(ValueError, match="nonfinite"):
        N.information(Q, np.full((3, 6), np.nan), np.zeros(3), np.ones(3))


def test_orthogonal_normal_stable_at_small_nonzero_doca():
    q = Q.copy()
    origin, u, _, _, _ = N.kinematics(q, q[4])
    normal = np.cross(WIRE, u); normal /= np.linalg.norm(normal)
    mid = origin+100*u+1e-7*normal
    r = N.rows(q, mid, WIRE, np.array(1.), np.array(180.))
    dx = np.asarray(jax.jacfwd(position)(q, r["particle_time"]))
    np.testing.assert_allclose(r["orthogonal_doca_H"], normal@dx, rtol=1e-12, atol=1e-11)
    assert abs(r["orthogonal_doca_H"][4]) < 1e-11
    assert 0 < abs(r["doca"]) < 2e-7


def test_perpendicular_and_time_cancellation_geometries():
    q = Q.copy()
    _, u, speed, _, _ = N.kinematics(q, q[4])
    normal = np.cross(u, WIRE); normal /= np.linalg.norm(normal)
    perpendicular = N.rows(q, MID, normal, np.array(1.), np.array(180.))
    np.testing.assert_allclose(perpendicular["full_minus_native_long"], 0., atol=1e-11)
    wire = .5*u+np.sqrt(.75)*normal
    cancellation = N.rows(q, MID, wire, np.array(1.), np.array(.5*speed))
    np.testing.assert_allclose(cancellation["omitted_time_direction"], 0., atol=1e-11)
    assert np.max(np.abs(cancellation["full_minus_native_long"])) > 1.


def test_inputs_unmodified():
    q, mid, wire, sign, vs = Q.copy(), MID.copy(), WIRE.copy(), np.array(-1.), np.array(180.)
    before = [v.copy() for v in (q, mid, wire, sign, vs)]
    N.rows(q, mid, wire, sign, vs)
    for a, b in zip(before, (q, mid, wire, sign, vs)):
        np.testing.assert_array_equal(a, b)
