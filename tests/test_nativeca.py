"""Independent scalar control-flow checks for local CA reconstruction diagnostics."""
import numpy as np
import pytest

from trkfastsim.reco import nativeca as C, nativemeasurement as N


Q = np.array([45., -1.1, 280., 1.4, 73000., 2000.])
WIRE = np.array([.8, .6, 0.])
MID = np.array([250., 180., 390.])


def scalar_oracle(q, early, late, mtime, vs, ph, sh, precision, cap):
    d, phi, z, theta, t0, p = q
    u = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    origin = np.array([-d*np.sin(phi), d*np.cos(phi), z])
    v = p/np.sqrt(p*p+105.6583755**2)*299.792458
    s = (early-late)/np.linalg.norm(early-late)
    c = s@u
    if 1-c*c < 1e-5:
        return ph, sh, 4, 0
    dp = ds = np.inf
    for n in range(cap):
        pp, sp = origin+((ph-t0)*v)*u, early-((mtime-sh)*vs)*s
        delta = sp-pp
        pu, ss = sum(delta*u), sum(delta*s)
        dp, ds = (pu-ss*c)/((1-c*c)*v), (pu*c-ss)/((1-c*c)*vs)
        ph, sh = ph+dp, sh+ds
        if abs(dp) <= precision and abs(ds) <= precision:
            break
    return ph, sh, int(not (abs(dp) < precision and abs(ds) < precision)), n+1


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("cap", [1, 2, 100])
def test_masked_updates_match_scalar_control_flow(sign, cap):
    q = np.stack([Q, Q+np.array([12., .1, -80., -.3, 5., 30.])])
    early = np.broadcast_to(MID+sign*500*WIRE, (2, 3)).copy()
    late = np.broadcast_to(MID-sign*500*WIRE, (2, 3)).copy()
    mt, vs = np.array([73003., 73010.]), np.array([180., 220.])
    line = C.sensor(early, late, mt, vs)
    ph, sh = np.array([72999., 73004.]), np.array([73000., 73006.])
    result = C.solve(q, line, ph, sh, max_updates=cap)
    expected = np.array([scalar_oracle(q[i], early[i], late[i], mt[i], vs[i], ph[i], sh[i], 1e-4, cap) for i in range(2)])
    np.testing.assert_allclose(result["particle_time"], expected[:, 0], rtol=0, atol=2e-11)
    np.testing.assert_allclose(result["sensor_time"], expected[:, 1], rtol=0, atol=2e-11)
    np.testing.assert_array_equal(result["status"], expected[:, 2])
    np.testing.assert_array_equal(result["updates"], expected[:, 3])
    assert np.all(result["usable"])
    if cap == 1:
        assert np.all(result["status"] == 1)
    else:
        assert np.all(result["status"] == 0)


def test_constructor_hint_and_second_update_geometry():
    line = C.sensor(MID+500*WIRE, MID-500*WIRE, np.array(73003.), np.array(180.))
    corrected = np.array(73000.)
    ph, sh = C.constructor_hint(Q, line, corrected)
    pp, _, _ = C.particle(Q, ph)
    sp = C.sensor_position(line, sh)
    np.testing.assert_allclose(pp[2], sp[2], atol=1e-8)
    first = C.solve(Q, line, ph, sh)
    second = C.solve(Q, line, first["particle_time"], first["sensor_time"])
    analytic = N.rows(Q, MID, WIRE, np.array(1.), np.array(180.))
    np.testing.assert_allclose(second["particle_time"], analytic["particle_time"], atol=2e-11, rtol=0)
    r = C.rows_at_state(Q, line, second)
    np.testing.assert_allclose(r["H"], analytic["H"], rtol=1e-8, atol=1e-6)
    assert int(second["updates"]) == 1


def test_mixed_parallel_lanes_and_empty_batch():
    q = np.stack([Q, Q])
    _, u, _ = C.particle(q, q[:, 4])
    early = MID+500*np.stack([u[0], WIRE])
    late = MID-500*np.stack([u[0], WIRE])
    line = C.sensor(early, late, np.full(2, 73003.), np.full(2, 180.))
    result = C.solve(q, line, np.full(2, 73000.), np.full(2, 73000.))
    np.testing.assert_array_equal(result["status"], [4, 0])
    assert result["updates"][0] == 0
    with pytest.raises(ValueError, match="unusable"):
        C.rows_at_state(q, line, result)
    line = C.sensor(np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0))
    result = C.solve(np.empty((0, 6)), line, np.empty(0), np.empty(0))
    assert result["delta"].shape == (0, 3)


@pytest.mark.parametrize("precision,cap", [(0., 100), (np.nan, 100), (1e-4, 0), (1e-4, 101), (1e-4, True)])
def test_update_guards(precision, cap):
    line = C.sensor(MID+500*WIRE, MID-500*WIRE, np.array(73003.), np.array(180.))
    with pytest.raises(ValueError, match="precision/update"):
        C.solve(Q, line, np.array(73000.), np.array(73000.), precision, cap)


def test_endpoint_and_hint_guards():
    with pytest.raises(ValueError, match="positive"):
        C.sensor(MID, MID, np.array(1.), np.array(180.))
    line = C.sensor(MID+500*WIRE, MID-500*WIRE, np.array(73003.), np.array(180.))
    with pytest.raises(ValueError, match="nonfinite"):
        C.solve(Q, line, np.array(np.nan), np.array(73000.))


def test_different_update_counts_and_overflow_fail_closed():
    q = np.stack([Q, Q])
    line = C.sensor(np.broadcast_to(MID+500*WIRE, (2, 3)), np.broadcast_to(MID-500*WIRE, (2, 3)),
                    np.full(2, 73003.), np.full(2, 180.))
    solved = C.solve(q, line, np.full(2, 73000.), np.full(2, 73000.))
    ph, sh = solved["particle_time"].copy(), solved["sensor_time"].copy()
    ph[1] -= 3.; sh[1] -= 2.
    mixed = C.solve(q, line, ph, sh)
    assert mixed["updates"][0] == 1 and mixed["updates"][1] > 1
    np.testing.assert_array_equal(mixed["particle_time"], solved["particle_time"])
    np.testing.assert_array_equal(mixed["sensor_time"], solved["sensor_time"])
    with pytest.raises(ValueError, match="failure-state propagation unsupported"):
        C.solve(q, line, np.array([73000., 1e308]), np.full(2, 73000.))


@pytest.mark.parametrize("mass", [105.658367, 200.])
def test_explicit_mass_in_position_and_derivative(mass):
    t = np.array(73001.)
    pp, u, v = C.particle(Q, t, mass)
    expected_v = Q[5]/np.sqrt(Q[5]**2+mass**2)*299.792458
    assert v == expected_v
    np.testing.assert_allclose(N.kinematics(Q, t, mass)[0], pp, rtol=1e-13, atol=1e-11)
    step = 1e-3
    plus, minus = Q.copy(), Q.copy()
    plus[5] += step; minus[5] -= step
    fd = (C.particle(plus, t, mass)[0]-C.particle(minus, t, mass)[0])/(2*step)
    np.testing.assert_allclose(N.kinematics(Q, t, mass)[3][:, 5], fd, rtol=1e-5, atol=1e-9)


def test_invalid_mass_refused():
    for mass in (0., np.nan, -1., np.ones(2)):
        with pytest.raises(ValueError, match="mass"):
            C.particle(Q, np.array(73001.), mass)
        with pytest.raises(ValueError, match="mass"):
            N.kinematics(Q, np.array(73001.), mass)


def test_sensor_position_uses_end_not_midpoint_and_no_input_changes():
    early, late = MID+500*WIRE, MID-500*WIRE
    line = C.sensor(early, late, np.array(73003.), np.array(180.))
    np.testing.assert_array_equal(C.sensor_position(line, line["measurement_time"]), early)
    np.testing.assert_allclose(C.sensor_position(line, line["measurement_time"]-1000/180), late, atol=1e-8)
    before = {k: v.copy() for k, v in line.items()}
    C.solve(Q, line, np.array(73000.), np.array(73000.))
    for k, value in before.items():
        np.testing.assert_array_equal(line[k], value)
