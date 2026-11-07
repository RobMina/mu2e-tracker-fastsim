"""Independent scalar pinned-source semantics, not recorded stock-oracle outputs.

The scalar reference deliberately uses maker loops, math/libm and separate
GSL-style fits per query. It does not call any implementation helper. Native
array/interpolation checks use absolute roundoff budgets, while query chunking
must be bit-identical. No physical-fit or remote Offline acceptance is implied.
"""
import copy
import math
import struct

import numpy as np
import pytest

from trkfastsim.reco.fielddrift import MAX_CELLS, MAX_NODES, build_model, drift_info


WIRE = 0.0125
OUTER = 2.5
BZ = 1.23456789012345


def configs():
    drift = dict(wireVoltage=1450.0, kVcm=[0.4, 0.8, 1.5, 3.0, 6.0, 12.0, 24.0],
                 cmus=[3.2, 4.5, 5.7, 6.5, 7.0, 7.1, 7.0],
                 phiBins=9, deltaDistance=0.037, deltaTime=0.19)
    response = dict(driftIgnorePhi=False, dRdTScale=1.17, driftOffBins=[0.0, 2.5],
                    driftOffset=[0.017, 0.023, 0.002, -0.014, -0.011, 0.014, 0.051, 0.019],
                    driftRMSBins=[-0.1, 2.7],
                    signedDriftRMS=[0.1, 0.11, 0.09, 0.14, 0.18, 0.12, 0.17],
                    unsignedDriftRMS=[0.12, 0.09, 0.11, 0.1, 0.07, 0.18, 0.19],
                    useNonLinearDrift=False, linearDriftVelocity=123.0)
    return drift, response


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def scalar_maker(drift, wire=WIRE, outer=OUTER, bz=BZ):
    """Literal scalar maker recurrence and inversion with its unusual upper index."""
    log_radius = math.log(outer / wire)
    raw_cc = f32(f32(bz) * log_radius / drift["wireVoltage"])
    edata = [v * 100.0 for v in drift["kVcm"]]
    sdata = [v * 0.01 for v in drift["cmus"]]
    ddata = [drift["wireVoltage"] / (e * log_radius) for e in edata]
    np_ = drift["phiBins"]
    dd, dt = drift["deltaDistance"], drift["deltaTime"]
    dp = (math.pi / 2.0) / (np_ - 1)
    radii, speeds, native = [0.0], [0.0], [[0.0] * np_]
    distance, t, maximum = dd, 0.0, 0.0
    index = len(ddata) - 2
    while distance < outer:
        while ddata[index] < distance:
            if index == 0:
                break
            index -= 1
        d0, d1 = ddata[index + 1], ddata[index]
        s0, s1 = sdata[index + 1], sdata[index]
        speed = s0 + (s1 - s0) * (distance - d0) / (d1 - d0)
        radii.append(distance)
        speeds.append(speed)
        t += dd / speed
        c = raw_cc * (distance / t) * 1_000_000.0
        row = []
        for p in range(np_):
            zetta = c * distance * 0.001
            gamma = (1.0 + math.pow(zetta, 2) / 3.0) / (
                1.0 + math.pow(zetta * math.cos(dp * p), 2) / 3.0)
            row.append(t * gamma)
            maximum = max(maximum, t * gamma)
        native.append(row)
        distance += dd
    speeds[0] = speeds[1]
    nt = math.ceil(maximum / dt)
    times = [dt * j for j in range(nt)]
    inverse = [[0.0] * np_ for _ in range(nt)]
    for p in range(np_):
        index = 0
        for j in range(1, nt):
            while index < len(radii) - 2:
                if times[j] < native[index][p]:
                    break
                index += 1
            inverse[j][p] = radii[index] + (times[j] - native[index][p]) / (
                native[index + 1][p] - native[index][p]) * dd
    return dict(cc=raw_cc * 1000.0, dd=dd, dt=dt, dp=dp, radii=radii, speeds=speeds,
                native=native, times=times, inverse=inverse, maximum=maximum)


def scalar_calibration(bins, values, x):
    width = (bins[1] - bins[0]) / len(values)
    ibin = min(len(values) - 1, max(0, math.floor((x - bins[0]) / width)))
    start, stop = max(0, ibin - 2), min(len(values) - 1, ibin + 2)
    xs = [bins[0] + width * (j + 0.5) for j in range(start, stop + 1)]
    ys = values[start:stop + 1]
    mx = my = 0.0
    for i in range(len(xs)):
        mx += (xs[i] - mx) / (i + 1)
        my += (ys[i] - my) / (i + 1)
    mxx = mxy = 0.0
    for i in range(len(xs)):
        dx, dy = xs[i] - mx, ys[i] - my
        mxx += (dx * dx - mxx) / (i + 1)
        mxy += (dx * dy - mxy) / (i + 1)
    slope = mxy / mxx
    intercept = my - mx * slope
    return intercept + slope * x, slope


def scalar_query(native, response, dt, phi):
    used = 0.0 if response["driftIgnorePhi"] else float(phi)
    folded = abs(math.fmod(used, math.pi))
    if folded > math.pi / 2:
        folded = math.pi - folded
    p0 = max(0, math.floor(folded / native["dp"]))
    p1 = min(len(native["inverse"][0]) - 1, math.ceil(folded / native["dp"]))
    ti = min(len(native["times"]) - 2, max(0, math.floor(dt / native["dt"])))
    frac = (dt - native["times"][ti]) / native["dt"]
    lower = native["inverse"][ti][p0] + frac * (
        native["inverse"][ti + 1][p0] - native["inverse"][ti][p0])
    if folded == 0:
        cdrift = lower
    else:
        upper = native["inverse"][ti][p1] + frac * (
            native["inverse"][ti + 1][p1] - native["inverse"][ti][p1])
        cdrift = lower + (folded - p0 * native["dp"]) / native["dp"] * (upper - lower)
    correction, slope = scalar_calibration(response["driftOffBins"], response["driftOffset"], cdrift)
    rdrift = cdrift - correction
    di = min(len(native["radii"]) - 2, max(0, math.floor(cdrift / native["dd"])))
    speed = native["speeds"][di] + (cdrift - native["radii"][di]) / native["dd"] * (
        native["speeds"][di + 1] - native["speeds"][di])
    signed, _ = scalar_calibration(response["driftRMSBins"], response["signedDriftRMS"], rdrift)
    unsigned, _ = scalar_calibration(response["driftRMSBins"], response["unsignedDriftRMS"], rdrift)
    return dict(used_phi=used, c_drift=cdrift, r_drift=rdrift,
                drift_velocity=speed * (1.0 - slope) * response["dRdTScale"],
                signed_error=signed, unsigned_error=unsigned)


@pytest.fixture(scope="module")
def setup():
    drift, response = configs()
    model = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    return model, scalar_maker(drift), drift, response


def check_queries(setup, dt, phi):
    model, native, _, response = setup
    dt, phi = np.broadcast_arrays(dt, phi)
    actual = drift_info(model, dt, phi)
    expected = [scalar_query(native, response, float(t), float(p))
                for t, p in zip(dt.ravel(), phi.ravel())]
    assert set(actual) == {"used_phi", "c_drift", "r_drift", "drift_velocity", "signed_error", "unsigned_error"}
    for key in actual:
        assert actual[key].shape == dt.shape
        values = np.array([row[key] for row in expected]).reshape(dt.shape)
        np.testing.assert_allclose(actual[key], values, rtol=0, atol=3e-12, err_msg=key)
    return actual


def test_maker_native_grids_and_cc(setup):
    model, native, drift, _ = setup
    assert model.cc == native["cc"]
    assert model.bz == BZ and model.bz_float == f32(BZ)
    unrounded = BZ * math.log(OUTER / WIRE) / drift["wireVoltage"] * 1000.0
    rounded_bz_only = f32(BZ) * math.log(OUTER / WIRE) / drift["wireVoltage"] * 1000.0
    assert model.cc != unrounded and model.cc != rounded_bz_only
    np.testing.assert_array_equal(model.distances, native["radii"])
    assert np.any(model.distances != np.arange(model.distances.size) * drift["deltaDistance"])
    np.testing.assert_array_equal(model.instant_speeds, native["speeds"])
    np.testing.assert_allclose(model.times_dbins, native["native"], rtol=0, atol=3e-14)
    np.testing.assert_array_equal(model.times, native["times"])
    np.testing.assert_allclose(model.distances_tbins, native["inverse"], rtol=0, atol=2e-15)
    assert model.times.size == math.ceil(native["maximum"] / model.delta_t)
    assert model.times[-1] < native["maximum"] <= model.times[-1] + model.delta_t
    ordinary = np.interp(model.times[1:-1], model.times_dbins[:, 0], model.distances)
    assert np.max(np.abs(ordinary - model.distances_tbins[1:-1, 0])) > 1e-5
    for array in (model.distances, model.times, model.instant_speeds, model.times_dbins,
                  model.distances_tbins, model.offset.slope, model.signed.intercept):
        assert not array.flags.writeable


def test_repeated_addition_grid_can_include_extra_outer_near_node():
    drift, response = configs()
    drift["deltaDistance"] = 0.001
    model = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    native = scalar_maker(drift)
    assert model.distances.size == 2501
    assert model.distances[-1] < OUTER
    np.testing.assert_array_equal(model.distances, native["radii"])
    np.testing.assert_allclose(model.times_dbins, native["native"], rtol=0, atol=3e-14)
    np.testing.assert_allclose(model.distances_tbins, native["inverse"], rtol=0, atol=3e-15)


def test_maker_strict_upper_index_on_exact_time_equality():
    drift, response = configs()
    first = scalar_maker(drift, bz=0)
    drift["deltaTime"] = first["native"][9][0]
    native = scalar_maker(drift, bz=0)
    model = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=0)
    assert model.times[1] == model.times_dbins[9, 0]
    assert model.distances_tbins[1, 0] == native["inverse"][1][0]
    assert abs(model.distances_tbins[1, 0] - model.distances[9]) > 1e-6
    np.testing.assert_allclose(model.distances_tbins, native["inverse"], rtol=0, atol=2e-15)


def test_scalar_broadcast_empty(setup):
    for dt, phi in ((-3.0, -0.22), (np.arange(12).reshape(3, 4), 0.2),
                    (np.arange(3)[:, None], np.array([-4.0, 0.0, 1.0, 3.0])[None, :]),
                    ([], []), (np.empty((0, 3)), np.ones((1, 3))), (2.0, [])):
        result = check_queries(setup, dt, phi)
        assert all(value.dtype == np.float64 for value in result.values())


def test_bin_boundaries_wrapped_negative_tail_queries(setup):
    model = setup[0]
    ts = np.r_[model.times[::7], -50.0, -model.delta_t, -0.0, 0.0,
               model.times[-1], model.times[-1] + model.delta_t, 200.0, 500.0]
    ts = np.unique(np.r_[ts, np.nextafter(ts, -np.inf), np.nextafter(ts, np.inf)])
    ph = model.delta_phi * np.arange(model.times_dbins.shape[1])
    ph = np.r_[ph, -ph, math.pi - ph, ph + 6 * math.pi, ph - 7 * math.pi,
               1.23456789012345, -0.0, 1e100]
    ph = np.unique(np.r_[ph, np.nextafter(ph, -np.inf), np.nextafter(ph, np.inf)])
    actual = check_queries(setup, ts[:, None], ph[None, :])
    assert np.min(actual["c_drift"]) < 0 and np.max(actual["r_drift"]) > OUTER
    np.testing.assert_array_equal(actual["used_phi"], np.broadcast_to(ph, actual["used_phi"].shape))
    t, p = 35.678, 1.23456789012345
    a = drift_info(model, t, p)["c_drift"]
    b = drift_info(model, t, f32(p))["c_drift"]
    assert abs(a - b) > 1e-12


def test_calibration_bin_edges_and_nonconstant_errors(setup):
    model, native, drift, response = setup
    edges = np.linspace(*response["driftOffBins"], len(response["driftOffset"]) + 1)
    edges = np.r_[edges, -1, 4]
    times = []
    for radius in edges:
        i = max(0, min(model.times.size - 2, np.searchsorted(model.distances_tbins[:, 0], radius) - 1))
        time = model.times[i] + (radius - model.distances_tbins[i, 0]) / (
            model.distances_tbins[i + 1, 0] - model.distances_tbins[i, 0]) * model.delta_t
        times.extend((time - 1e-8, time, time + 1e-8))
    check_queries(setup, times, 0.0)
    result = drift_info(model, [-30.0, 10.0, 30.0, 90.0, 120.0], 0.4)
    assert np.ptp(result["signed_error"]) > 0.01
    assert np.ptp(result["unsigned_error"]) > 0.01
    assert np.ptp(result["c_drift"] - result["r_drift"]) > 0.01
    assert result["signed_error"][-1] != result["signed_error"][-2]
    altered = dict(response, useNonLinearDrift=True, linearDriftVelocity=-1000.0)
    other = build_model(drift, altered, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    for key, value in result.items():
        np.testing.assert_array_equal(value, drift_info(other, [-30, 10, 30, 90, 120], 0.4)[key])


@pytest.mark.parametrize("length", [2, 3, 4, 5, 6, 11])
def test_up_to_five_bin_local_regression(length):
    drift, response = configs()
    j = np.arange(length)
    response["driftOffset"] = (0.005 * j * j - 0.01 * j).tolist()
    response["signedDriftRMS"] = (0.1 + 0.001 * j**3).tolist()
    response["unsignedDriftRMS"] = (0.2 - 0.005 * j + 0.001 * j**2).tolist()
    model = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    check_queries((model, scalar_maker(drift), drift, response), np.linspace(-40, 170, 233), 0.71)


def test_ignore_phi_and_signed_field(setup):
    model, native, drift, response = setup
    negative = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=-BZ)
    assert negative.cc == -model.cc
    np.testing.assert_array_equal(negative.times_dbins, model.times_dbins)
    np.testing.assert_array_equal(negative.distances_tbins, model.distances_tbins)
    response = dict(response, driftIgnorePhi=True)
    ignored = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    result = check_queries((ignored, native, drift, response), np.linspace(-10, 140, 61),
                           np.linspace(-19, 19, 61))
    baseline = drift_info(model, np.linspace(-10, 140, 61), 0.0)
    for key in result:
        np.testing.assert_array_equal(result[key], baseline[key])
    zero = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=0)
    assert zero.cc == 0
    np.testing.assert_array_equal(zero.times_dbins, np.broadcast_to(zero.times_dbins[:, :1],
                                                                  zero.times_dbins.shape))


def test_exact_chunk_invariance(setup):
    model = setup[0]
    rng = np.random.default_rng(183)
    dt, phi = rng.uniform(-200, 400, 1231), rng.uniform(-30, 30, 1231)
    whole = drift_info(model, dt, phi)
    for chunk in (1, 7, 64, 257):
        parts = [drift_info(model, dt[i:i + chunk], phi[i:i + chunk])
                 for i in range(0, dt.size, chunk)]
        for key in whole:
            np.testing.assert_array_equal(whole[key], np.concatenate([p[key] for p in parts]))
    scalar = drift_info(model, dt[13], phi[13])
    for key in whole:
        assert scalar[key] == whole[key][13]


@pytest.mark.parametrize("mapping,key,value", [
    (0, "wireVoltage", 0), (0, "wireVoltage", "1450"), (0, "wireVoltage", True),
    (0, "wireVoltage", np.inf), (0, "wireVoltage", 1e-308),
    (0, "deltaDistance", 0), (0, "deltaDistance", -0.1), (0, "deltaDistance", OUTER),
    (0, "deltaDistance", 1e-300), (0, "deltaDistance", np.nan),
    (0, "deltaTime", 0), (0, "deltaTime", 1e-300), (0, "deltaTime", 1e10),
    (0, "phiBins", 1), (0, "phiBins", 2.5), (0, "phiBins", True),
    (0, "phiBins", MAX_NODES + 1), (0, "phiBins", MAX_NODES),
    (0, "kVcm", [1]), (0, "kVcm", [2, 1, 3, 4, 5, 6, 7]),
    (0, "kVcm", [1, 1, 2, 3, 4, 5, 6]), (0, "kVcm", [0, 1, 2, 3, 4, 5, 6]),
    (0, "kVcm", [[1, 2]]), (0, "kVcm", [np.nan] * 7),
    (0, "kVcm", [1.0] * (MAX_NODES + 1)),
    (0, "cmus", [1]), (0, "cmus", [1, 1, 1, 1, 1, 1, 0]),
    (0, "cmus", [1, 1, 1, 1, 1, 100, 1]),
    (1, "driftIgnorePhi", 0), (1, "driftIgnorePhi", "false"),
    (1, "dRdTScale", np.nan), (1, "dRdTScale", [1.0]),
    (1, "driftOffBins", [0]), (1, "driftOffBins", [0, 0]), (1, "driftOffBins", [2, 1]),
    (1, "driftOffset", []), (1, "driftOffset", [1]), (1, "driftOffset", [0, np.inf]),
    (1, "driftRMSBins", [2, 1]), (1, "driftRMSBins", [0, 1e-300]),
    (1, "signedDriftRMS", [1, 2]), (1, "unsignedDriftRMS", [1]),
])
def test_invalid_config(mapping, key, value):
    cfg = configs()
    cfg[mapping][key] = value
    with pytest.raises(ValueError):
        build_model(*cfg, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)


@pytest.mark.parametrize("kwargs", [dict(wire_radius=0), dict(wire_radius=OUTER),
                                    dict(wire_radius=np.nan), dict(outer_radius=-1),
                                    dict(outer_radius=np.inf), dict(bz=np.inf),
                                    dict(bz=1e100), dict(bz="1.0"), dict(bz=None)])
def test_invalid_geometry_field(kwargs):
    args = dict(wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    args.update(kwargs)
    with pytest.raises(ValueError):
        build_model(*configs(), **args)


def test_mapping_missing_keys_cell_bound_and_config_ownership(setup):
    with pytest.raises(ValueError):
        build_model([], {}, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    for which, key in ((0, "deltaTime"), (1, "driftIgnorePhi")):
        cfg = configs()
        del cfg[which][key]
        with pytest.raises(ValueError, match="missing"):
            build_model(*cfg, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    drift, response = configs()
    drift["deltaTime"] = 0.001
    drift["phiBins"] = 100
    with pytest.raises(ValueError, match="cell bound"):
        build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    drift, response = configs()
    untouched = copy.deepcopy((drift, response))
    model = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    assert (drift, response) == untouched
    original = drift_info(model, [1, 20], 0.3)
    response["driftOffset"][:] = [99] * 8
    drift["cmus"][:] = [99] * 7
    for key, value in original.items():
        np.testing.assert_array_equal(value, drift_info(model, [1, 20], 0.3)[key])
    assert model.times_dbins.size + model.distances_tbins.size <= MAX_CELLS


@pytest.mark.parametrize("dt,phi", [(np.nan, 0), (0, np.inf), ([1, np.inf], 0),
                                    (1, [0, np.nan]), ([1, 2], [0, 1, 2]),
                                    (1 + 2j, 0), ("12", 0), (True, 0),
                                    (None, 0), ([], np.nan), (1e308, 0)])
def test_invalid_queries(setup, dt, phi):
    with pytest.raises(ValueError):
        drift_info(setup[0], dt, phi)


def test_ignored_phi_still_requires_finite():
    drift, response = configs()
    response["driftIgnorePhi"] = True
    model = build_model(drift, response, wire_radius=WIRE, outer_radius=OUTER, bz=BZ)
    with pytest.raises(ValueError):
        drift_info(model, [], np.inf)
    with pytest.raises(ValueError):
        drift_info(None, 0, 0)
