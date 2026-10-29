"""Independent scalar controls for the fixed arithmetic diagnostic (not ROOT)."""
import numpy as np
import pytest

from trkfastsim.reco import nativearith as A, nativemeasurement as N

Q = np.array([45., -1.1, 280., 1.4, 73000., 2000.])


def scalar_kinematics(q, t, mass):
    # Deliberately scalar components, no call to either production implementation.
    d, f, z, th, t0, p = q
    sf, cf, st, ct = np.sin(f), np.cos(f), np.sin(th), np.cos(th)
    u, axis, ta = [st*cf, st*sf, ct], [-sf, cf, 0.], [ct*cf, ct*sf, -st]
    E = np.sqrt(p*p+mass*mass)
    v, g, dt = (p/E)*299.792458, E/mass, t-t0
    origin, radial = [-d*sf, d*cf, z], [cf, sf, 0.]
    pos, X, M = np.empty(3), np.zeros((3, 6)), np.zeros((3, 6))
    for i in range(3):
        pos[i] = origin[i]+(dt*v)*u[i]
        X[i, 0] = axis[i]
        X[i, 1] = ((st*v)*dt)*axis[i]-d*radial[i]
        X[i, 2] = float(i == 2)
        X[i, 3] = (v*dt)*ta[i]
        X[i, 4] = -v*u[i]
        X[i, 5] = -(dt/((g*g)*p))*X[i, 4]
        M[i, 1] = ((p*st)*axis[i])/p
        M[i, 3] = (p*ta[i])/p
        M[i, 5] = u[i]/p
    return pos, np.array(u), v, X, M


def dot(a, b):
    return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2]


def scalar_rows(q, line, state, mass):
    _, u, v, X, M = scalar_kinematics(q, state["particle_time"], mass)
    s, delta = line["direction"], state["delta"]
    cross = [s[1]*u[2]-s[2]*u[1], s[2]*u[0]-s[0]*u[2], s[0]*u[1]-s[1]*u[0]]
    sign = np.copysign(1., dot(cross, delta))
    norm = np.sqrt(dot(delta, delta))
    c = dot(s, u)
    den = 1.-c*c
    pf, sf = 1./(v*den), 1./(line["speed"]*den)
    signed = [sign*(delta[i]*(1./norm)) for i in range(3)]
    bg = [pf*(s[i]*c-u[i])-sf*(s[i]-c*u[i]) for i in range(3)]
    H = np.empty((3, 6))
    for j in range(6):
        H[0, j], H[1, j] = dot(signed, X[:, j]), -dot(bg, X[:, j])
        ds, dp, ddm = dot(s, X[:, j]), dot(u, X[:, j]), dot(delta, M[:, j])
        H[2, j] = ds+((c*ds-dp-ddm)/den)*c
    return {"H": H, "doca": sign*norm, "delta_dot_particle": dot(delta, u),
            "delta_dot_sensor": dot(delta, s),
            "longitudinal": dot(state["sensor_poca"]-line["mid"], s)}


def scalar_information(q, H, r, w):
    matrices, vectors = [], []
    for k in range(3):
        W, b = np.empty((6, 6)), np.empty(6)
        for i in range(6):
            for j in range(6):
                W[i, j] = (H[k, i]*w[k])*H[k, j]
            value = W[i, 0]*q[0]
            for j in range(1, 6):
                value = value+W[i, j]*q[j]
            b[i] = value-(H[k, i]*r[k])*w[k]
        matrices.append(W); vectors.append(b)
    return ((matrices[0]+matrices[1])+matrices[2], (vectors[0]+vectors[1])+vectors[2])


def operands(shape=()):
    def batch(x):
        x = np.asarray(x)
        return np.broadcast_to(x, shape+x.shape).copy()
    q = batch(Q)
    line = {"direction": batch([.3, -.4, np.sqrt(.75)]), "speed": batch(180.),
            "mid": batch([250., 180., 390.])}
    state = {"usable": batch(True), "particle_time": batch(73001.125),
             "delta": batch([1e-9, -2e-9, 3e-9]), "sensor_poca": batch([260., 170., 395.])}
    return q, line, state


@pytest.mark.parametrize("mass", [105.6583755, 105.658367, 200.])
@pytest.mark.parametrize("p,dt", [(40., -1.125), (450., 0.), (2000., 1.125), (1e7, 1e-6)])
def test_kinematics_scalar_expression_control(mass, p, dt):
    q = Q.copy(); q[5] = p
    for actual, expected in zip(A.kinematics(q, q[4]+dt, mass), scalar_kinematics(q, q[4]+dt, mass)):
        np.testing.assert_array_equal(actual, expected)
    # Algebraic control, not a floating-point equivalence claim.
    for actual, expected in zip(A.kinematics(q, q[4]+dt, mass), N.kinematics(q, q[4]+dt, mass)):
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-13)


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("scale", [1., 1e9, 1e-100])
def test_rows_scalar_control_retains_tiny_nonorthogonal_delta(sign, scale):
    q, line, state = operands()
    line["direction"] *= sign
    state["delta"] *= scale
    actual = A.rows_at_state(q, line, state, 105.658367)
    expected = scalar_rows(q, line, state, 105.658367)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_fixed_contraction_and_information_cancellations():
    # Left association yields 1, right association yields 0. Never choose by target.
    a, b = np.array([1e16, -1e16, 1.]), np.ones(3)
    assert A._dot3(a, b) == 1.
    np.testing.assert_array_equal(A._contract3(a, np.ones((3, 6))), np.ones(6))
    assert a[0]+(a[1]+a[2]) == 0.
    q = np.array([1e16, -1e16, 1., 0., 0., 1.])
    H = np.array([[1., 1., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 1., 0., 0.]])
    r, w = np.array([1., 1e16, -1e16]), np.array([1., 1., 1.])
    for actual, expected in zip(A.information(q, H, r, w), scalar_information(q, H, r, w)):
        np.testing.assert_array_equal(actual, expected)
    # Information row sum is (row0+row1)+row2, not a factored b calculation.
    H[:] = 0.; H[:, 0] = 1.; q[:] = 0.
    r[:] = [-1e16, 1e16, -1.]
    assert A.information(q, H, r, w)[1][0] == 1.


def test_unfactored_information_is_observably_different_under_cancellation():
    q = np.array([1e16, -3e16+4., 0., 0., 0., 1.])
    H = np.zeros((3, 6)); H[0, :2] = [3., 1.]
    r, w = np.array([4., 0., 0.]), np.array([.1, 0., 0.])
    W, b = A.information(q, H, r, w)
    expectedW, expectedb = scalar_information(q, H, r, w)
    np.testing.assert_array_equal(W, expectedW)
    np.testing.assert_array_equal(b, expectedb)
    factored = (H*w[:, None]).T @ ((H@q)-r)
    np.testing.assert_array_equal(factored, 0.)
    assert np.any(b != factored)


def test_phi_derivative_cancellation_keeps_source_grouping():
    q = Q.copy(); q[1] = .7
    t = q[4]+1.125
    _, _, speed, _, _ = scalar_kinematics(q, t, 105.6583755)
    coefficient = (np.sin(q[3])*speed)*(t-q[4])
    q[0] = (coefficient*(-np.sin(q[1])))/np.cos(q[1])
    # Both large terms cancel; no tolerance can hide a regrouped expression.
    for actual, expected in zip(A.kinematics(q, t), scalar_kinematics(q, t, 105.6583755)):
        np.testing.assert_array_equal(actual, expected)
    assert A.kinematics(q, t)[3][0, 1] == 0.


def test_information_general_scalar_reference_and_algebra():
    rng = np.random.default_rng(971)
    q = rng.normal(size=(7, 6))*1e4
    H, r, w = rng.normal(size=(7, 3, 6)), rng.normal(size=(7, 3)), rng.uniform(.01, 2., (7, 3))
    W, b = A.information(q, H, r, w)
    for k in range(7):
        expected = scalar_information(q[k], H[k], r[k], w[k])
        np.testing.assert_array_equal(W[k], expected[0])
        np.testing.assert_array_equal(b[k], expected[1])
    oldW, oldb = N.information(q, H, r, w)
    np.testing.assert_allclose(W, oldW, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(b, oldb, rtol=1e-13, atol=1e-9)


@pytest.mark.parametrize("shape", [(), (0,), (2,), (2, 3)])
def test_shapes_per_lane_mass_and_immutable_inputs(shape):
    q, line, state = operands(shape)
    mass = np.full(shape, 105.658367)
    if shape and mass.size:
        mass.flat[-1] = 200.
    arrays = [q, mass, *line.values(), *state.values()]
    before = [x.copy() for x in arrays]
    for x in arrays:
        x.flags.writeable = False
    pos, u, v, dx, dm = A.kinematics(q, state["particle_time"], mass)
    assert pos.shape == u.shape == shape+(3,)
    assert v.shape == shape and dx.shape == dm.shape == shape+(3, 6)
    H = A.rows_at_state(q, line, state, mass)["H"]
    assert H.shape == shape+(3, 6)
    r, w = np.zeros(shape+(3,)), np.ones(shape+(3,))
    H.flags.writeable = r.flags.writeable = w.flags.writeable = False
    Hbefore = H.copy()
    W, b = A.information(q, H, r, w)
    assert W.shape == shape+(6, 6) and b.shape == shape+(6,)
    for index in np.ndindex(shape):
        scalar = scalar_rows(q[index], {k: x[index] for k, x in line.items()},
                             {k: x[index] for k, x in state.items()}, mass[index])
        np.testing.assert_array_equal(H[index], scalar["H"])
    for actual, expected in zip(arrays, before):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(H, Hbefore)
    np.testing.assert_array_equal(r, 0.); np.testing.assert_array_equal(w, 1.)


@pytest.mark.parametrize("mass", [0., -1., np.nan, np.inf, np.ones(2)])
def test_mass_guard(mass):
    q, line, state = operands()
    with pytest.raises(ValueError, match="mass"):
        A.kinematics(q, state["particle_time"], mass)
    with pytest.raises(ValueError, match="mass"):
        A.rows_at_state(q, line, state, mass)


@pytest.mark.parametrize("q,t,error", [(np.ones(5), 0., "shape"), (np.ones((2, 6)), 0., "shape"),
    (Q*np.nan, 0., "nonfinite"), (Q, np.inf, "nonfinite"),
    (np.array([0., 0., 0., 1., 0., 0.]), 0., "positive"),
    (np.array([0., 0., 0., 1., 0., -1.]), 0., "positive")])
def test_kinematic_input_guards(q, t, error):
    with pytest.raises(ValueError, match=error):
        A.kinematics(q, t)


@pytest.mark.parametrize("container,key,value,error", [
    ("line", "direction", [1., 1., 0.], "unit"), ("line", "speed", -1., "positive"),
    ("line", "speed", [180.], "shape"), ("line", "mid", [np.inf, 0., 0.], "nonfinite"),
    ("state", "usable", False, "unusable"), ("state", "usable", np.nan, "nonfinite"),
    ("state", "delta", [0., 0., 0.], "zero-distance"),
    ("state", "delta", [0., 0.], "shape"),
    ("state", "sensor_poca", [0., np.nan, 0.], "nonfinite")])
def test_row_guards(container, key, value, error):
    q, line, state = operands()
    (line if container == "line" else state)[key] = value
    with pytest.raises(ValueError, match=error):
        A.rows_at_state(q, line, state)


def test_parallel_overflow_and_information_guards():
    q, line, state = operands()
    line["direction"] = scalar_kinematics(q, q[4], 105.6583755)[1]
    with pytest.raises(ValueError, match="parallel"):
        A.rows_at_state(q, line, state)
    q[5] = 1e308
    with pytest.raises(ValueError, match="nonfinite"):
        A.kinematics(q, q[4])
    with pytest.raises(ValueError, match="nonfinite"):
        A.kinematics(Q, Q[4], 1e308)
    for H, r, w, error in [(np.zeros((3, 6)), np.zeros(3), -np.ones(3), "negative"),
                           (np.zeros((2, 6)), np.zeros(3), np.ones(3), "shape"),
                           (np.zeros((3, 6)), np.zeros(2), np.ones(3), "shape"),
                           (np.zeros((3, 6)), np.zeros(3), np.array([1., np.inf, 1.]), "nonfinite"),
                           (np.full((3, 6), np.nan), np.zeros(3), np.ones(3), "nonfinite"),
                           (np.full((3, 6), 1e308), np.zeros(3), np.ones(3), "nonfinite")]:
        with pytest.raises(ValueError, match=error):
            A.information(Q, H, r, w)


@pytest.mark.parametrize("sign", [-1., 1.])
def test_sensor_unit_uses_header_reciprocal_multiplication(sign):
    early = sign*np.array([[13., 7., -2.], [131.25, -89.125, .0023]])
    late = np.array([[1., -2., 3.], [-3.25, 8., 1.]])
    mt, vs = np.array([73000., 8100.]), np.array([180., 210.])
    before = [a.copy() for a in (early, late, mt, vs)]
    line = A.sensor(early, late, mt, vs)
    delta = early-late
    norm = np.sqrt((delta[:, 0]**2+delta[:, 1]**2)+delta[:, 2]**2)
    np.testing.assert_array_equal(line["direction"], delta*(1./norm)[:, None])
    np.testing.assert_array_equal(line["length"], norm)
    np.testing.assert_array_equal(line["end"], early)
    for value, saved in zip((early, late, mt, vs), before):
        np.testing.assert_array_equal(value, saved)
