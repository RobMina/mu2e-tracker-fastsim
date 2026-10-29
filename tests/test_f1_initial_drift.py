"""Synthetic zero-B drift tests; no native jobs, frozen captures, or fitter runs."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_initial_drift as D


@pytest.fixture
def config():
    # dataDist=[8,4,2] for outer=8, wire=1; converted speeds=[1,2,4].
    return dict(useDb=False, wireVoltage=8. * np.log(8.), kVcm=[.01, .02, .04],
                cmus=[100., 200., 400.], deltaDistance=1., deltaTime=.2, phiBins=3,
                driftOffBins=[0., 8.], driftOffset=[.2, .4, .6, .8],
                driftRMSBins=[0., 8.], signedDriftRMS=[.5, 1., 1.5, 2.],
                unsignedDriftRMS=[1., 2., 3., 4.], dRdTScale=1.25, driftIgnorePhi=False)


@pytest.fixture
def fcl(tmp_path, config):
    path = tmp_path / "expanded.fcl"
    def block(keys):
        return "\n".join(k + ": " + json.dumps(config[k]) for k in keys)
    path.write_text("services: { conditions: { strawDrift: {\n" + block(D.DRIFT_FIELDS) +
                    '\nignored: "literal"\n} strawResponse: {\n' + block(D.RESPONSE_FIELDS) + "\n} } }\n")
    return path


def scalar_table(c, outer, wire):
    """Small test oracle uses source scalar loops, independently of vector search."""
    data_d = [c["wireVoltage"] / (e * 100. * np.log(outer / wire)) for e in c["kVcm"]]
    data_v = [v * .01 for v in c["cmus"]]
    ds, ts, vs = [0.], [0.], [0.]
    dd, dt = c["deltaDistance"], c["deltaTime"]
    distance, time, index = dd, 0., len(data_d) - 2
    while distance < outer:
        while data_d[index] < distance and index > 0:
            index -= 1
        d0, d1, v0, v1 = data_d[index + 1], data_d[index], data_v[index + 1], data_v[index]
        speed = v0 + (v1 - v0) * (distance - d0) / (d1 - d0)
        time += dd / speed
        ds.append(distance)
        ts.append(time)
        vs.append(speed)
        distance += dd
    vs[0] = vs[1]
    tg = dt * np.arange(int(np.ceil(ts[-1] / dt)))
    td = [0.]
    for time in tg[1:]:
        index = next((i for i, t in enumerate(ts) if t > time), len(ts) - 1)
        index = min(index, len(ds) - 2)
        td.append(ds[index] + (time - ts[index]) / (ts[index + 1] - ts[index]) * dd)
    return dict(distance_grid=np.array(ds), instant_speed=np.array(vs), distance_times=np.array(ts),
                time_grid=tg, time_distances=np.array(td), deltaD=dd, deltaT=dt)


def scalar_regression(x, bins, values):
    width = (bins[1] - bins[0]) / len(values)
    i = min(len(values) - 1, max(0, int(np.floor((x - bins[0]) / width))))
    indices = list(range(max(0, i - 2), min(len(values) - 1, i + 2) + 1))
    meanx = meany = dx2 = dxdy = 0.
    for count, j in enumerate(indices, 1):
        meanx += (bins[0] + width * (j + .5) - meanx) / count
        meany += (values[j] - meany) / count
    for count, j in enumerate(indices, 1):
        dx, dy = bins[0] + width * (j + .5) - meanx, values[j] - meany
        dx2 += (dx * dx - dx2) / count
        dxdy += (dx * dy - dxdy) / count
    b = dxdy / dx2
    a = meany - meanx * b
    return a + b * x, b


def scalar_info(times, t, c):
    result = {k: [] for k in ("cdrift", "rdrift", "velocity", "sderr", "uderr")}
    for time in times:
        i = max(0, min(len(t["time_grid"]) - 2, int(np.floor(time / t["deltaT"]))))
        cd = t["time_distances"][i] + (time - t["time_grid"][i]) / t["deltaT"] * (
            t["time_distances"][i + 1] - t["time_distances"][i])
        correction, slope = scalar_regression(cd, c["driftOffBins"], c["driftOffset"])
        rd = cd - correction
        i = max(0, min(len(t["distance_grid"]) - 2, int(np.floor(cd / t["deltaD"]))))
        velocity = t["instant_speed"][i] + (cd - t["distance_grid"][i]) / t["deltaD"] * (
            t["instant_speed"][i + 1] - t["instant_speed"][i])
        velocity = (velocity * (1. - slope)) * c["dRdTScale"]
        sd, _ = scalar_regression(rd, c["driftRMSBins"], c["signedDriftRMS"])
        ud, _ = scalar_regression(rd, c["driftRMSBins"], c["unsignedDriftRMS"])
        for k, v in zip(result, (cd, rd, velocity, sd, ud)):
            result[k].append(v)
    return result


def test_parser_jsonable_and_fields_only(fcl, config):
    parsed = D.drift_configuration(fcl)
    assert parsed == config
    assert json.loads(json.dumps(parsed, allow_nan=False)) == parsed
    assert "ignored" not in parsed


@pytest.mark.parametrize("before,after", [
    ("strawDrift:", "strawDrift: {} strawDrift:"),
    ("strawResponse:", "strawResponse: {} strawResponse:"),
    ("strawDrift: {", "strawDrift: @local::drift {"),
    ("strawDrift: {", "strawDrift: { nested: {x: 1}"),
    ('ignored: "literal"', "ignored: @local::x"),
    ("useDb: false", "useDb: true"), ("useDb: false", "useDb: 0"),
    ("useDb: false", "useDb: false\nuseDb: false"),
    ("useDb: false", ""),
    ("phiBins: 3", "phiBins: 3.0"), ("phiBins: 3", "phiBins: true"),
    ("phiBins: 3", "phiBins: 1"), ("phiBins: 3", "phiBins: 1000001"),
    ("deltaDistance: 1.0", "deltaDistance: 0"),
    ("deltaTime: 0.2", "deltaTime: 1e309"),
    ("cmus: [100.0, 200.0, 400.0]", "cmus: [1,2]"),
    ("cmus: [100.0, 200.0, 400.0]", "cmus: [1,0,3]"),
    ("cmus: [100.0, 200.0, 400.0]", "cmus: [1,nan,3]"),
    ("cmus: [100.0, 200.0, 400.0]", "cmus: [1,2,3,]"),
    ("cmus: [100.0, 200.0, 400.0]", "cmus: [[1,2,3]]"),
    ("kVcm: [0.01, 0.02, 0.04]", "kVcm: [1,1,2]"),
    ("driftOffBins: [0.0, 8.0]", "driftOffBins: [0,8,9]"),
    ("driftOffBins: [0.0, 8.0]", "driftOffBins: [8,0]"),
    ("driftOffset: [0.2, 0.4, 0.6, 0.8]", "driftOffset: []"),
    ("driftOffset: [0.2, 0.4, 0.6, 0.8]", "driftOffset: [1]"),
    ("driftIgnorePhi: false", "driftIgnorePhi: 0"),
])
def test_parser_rejects_bad_fields(fcl, before, after):
    text = fcl.read_text()
    assert before in text
    fcl.write_text(text.replace(before, after))
    with pytest.raises(ValueError):
        D.drift_configuration(fcl)


@pytest.mark.parametrize("extra", ["#include \"other.fcl\"", "BEGIN_PROLOG", "END_PROLOG",
                                   "services.conditions.strawDrift.deltaTime: 1",
                                   "services.conditions.strawResponse.dRdTScale: 2"])
def test_parser_rejects_directives_and_overrides(fcl, extra):
    fcl.write_text(fcl.read_text() + "\n" + extra)
    with pytest.raises(ValueError):
        D.drift_configuration(fcl)


def test_parser_scientific_multiline_literals_and_quoted_table_decoys(fcl):
    fcl.write_text(fcl.read_text().replace("cmus: [100.0, 200.0, 400.0]", "cmus: [\n+1e2, 2.E2, .4e3\n]") +
                   '\n# strawDrift: {}\ndecoy: "strawDrift: {} strawResponse: {}"\n')
    assert D.drift_configuration(fcl)["cmus"] == [100., 200., 400.]


def test_handcomputed_distance_speed_time_and_first_greater(config):
    t = D.build_zero_field(config, 8., 1.)
    np.testing.assert_array_equal(t["distance_grid"], np.arange(8.))
    np.testing.assert_array_equal(t["instant_speed"], [5., 5., 4., 3., 2., 1.75, 1.5, 1.25])
    expected_time = [0., .2, .45, .45 + 1. / 3.]
    np.testing.assert_array_equal(t["distance_times"][:4], expected_time)
    # time=.2 equals distance_times[1]. FIRST GREATER selects index=2,
    # giving 2+(.2-.45)/(1/3)=1.25, rather than the usual inversion 1.
    assert t["time_distances"][1] == pytest.approx(1.25, abs=1e-15)
    assert t["time_distances"][1] != 1.
    assert t["time_distances"][0] == 0.
    assert len(t["time_grid"]) == int(np.ceil(t["distance_times"][-1] / config["deltaTime"]))
    expected = scalar_table(config, 8., 1.)
    for key in t:
        np.testing.assert_array_equal(t[key], expected[key], err_msg=key)


@pytest.mark.parametrize("dd,dt", [(1., .2), (.1, .07), (.03, .11), (.001, .05), (3., .03)])
def test_vectorized_grid_matches_sequential_oracle(config, dd, dt):
    config.update(deltaDistance=dd, deltaTime=dt)
    actual = D.build_zero_field(config, 8., 1.)
    expected = scalar_table(config, 8., 1.)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_sequential_distance_rounding_not_multiplication_and_strict_outer(config):
    config.update(deltaDistance=.1)
    t = D.build_zero_field(config, 8., 1.)
    assert not np.array_equal(t["distance_grid"], .1 * np.arange(len(t["distance_grid"])))
    assert np.all(t["distance_grid"] < 8.)
    assert t["distance_grid"][-1] + .1 >= 8.
    config["deltaTime"] = .01
    t = D.build_zero_field(config, .8, .1)
    # 8 additions of .1 are below .8; this endpoint is deliberately retained.
    assert len(t["distance_grid"]) == 9
    assert t["distance_grid"][-1] < .8


def test_zero_b_phi_flag_and_phi_count_do_not_change_tables(config):
    t = D.build_zero_field(config, 8., 1.)
    config.update(phiBins=20, driftIgnorePhi=True)
    other = D.build_zero_field(config, 8., 1.)
    for key in t:
        np.testing.assert_array_equal(t[key], other[key])


def test_regression_linear_handcomputed_extrapolation():
    x = np.array([-10., 0., 1., 2., 4., 8., 20.])
    y, slope = D._interpolate_calib(x, [0., 8.], [3., 7., 11., 15.])
    np.testing.assert_allclose(y, 1. + 2. * x, rtol=0, atol=1e-14)
    np.testing.assert_allclose(slope, np.full(len(x), 2.), rtol=0, atol=1e-15)


def test_regression_nonlinear_window_ties_and_online_source_oracle():
    bins, values = [-1., 6.], [.1, 4., 2., 8., 3., 1., -2.]
    edges = np.arange(-1., 7.)
    x = np.r_[-20., np.nextafter(edges, -np.inf), edges, np.nextafter(edges, np.inf), 30.]
    value, slope = D._interpolate_calib(x, bins, values)
    expected = np.array([scalar_regression(v, bins, values) for v in x])
    np.testing.assert_array_equal(value, expected[:, 0])
    np.testing.assert_array_equal(slope, expected[:, 1])
    assert D.CALIBRATION_METHOD == "NumPy online regression (not certified native GSL)"


def test_info_vectorized_scalar_oracle_negative_ties_endpoints_and_extrapolation(config):
    table = D.build_zero_field(config, 8., 1.)
    times = np.r_[-3., -0., table["time_grid"], table["time_grid"][-1] + 1., 10.]
    actual = D.drift_info(times, table, config)
    expected = scalar_info(times, table, config)
    assert set(actual) == {"cdrift", "rdrift", "velocity", "sderr", "uderr"}
    for name in actual:
        assert actual[name].shape == times.shape
        assert actual[name].dtype == np.float64
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert actual["cdrift"][0] < 0
    assert actual["sderr"][0] < 0 and actual["uderr"][0] < 0
    assert actual["cdrift"][-1] > table["distance_grid"][-1]
    assert actual["velocity"][-1] < 0  # extrapolation is permitted, not clipped


def test_handcomputed_info_correction_and_instant_speed_not_time_derivative(config):
    table = D.build_zero_field(config, 8., 1.)
    info = D.drift_info(np.array([0., .2]), table, config)
    # Offset=.1+.1*r, signed error=.25+.25*r, unsigned error=.5+.5*r.
    np.testing.assert_allclose(info["cdrift"], [0., 1.25], rtol=0, atol=1e-15)
    np.testing.assert_allclose(info["rdrift"], [-.1, 1.025], rtol=0, atol=1e-15)
    np.testing.assert_allclose(info["velocity"], [5. * .9 * 1.25, 4.75 * .9 * 1.25], rtol=0, atol=2e-15)
    np.testing.assert_allclose(info["sderr"], .25 + .25 * info["rdrift"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(info["uderr"], .5 + .5 * info["rdrift"], rtol=0, atol=1e-15)
    assert info["velocity"][0] != 1.25 / .2 * .9 * 1.25


def test_empty_times_and_input_immutability(config):
    table = D.build_zero_field(config, 8., 1.)
    old = deepcopy((config, table))
    empty = D.drift_info(np.array([]), table, config)
    assert all(x.shape == (0,) for x in empty.values())
    D.drift_info(np.array([-1., 0., 1.]), table, config)
    for before, after in zip(old, (config, table)):
        for k in before:
            np.testing.assert_array_equal(before[k], after[k])


@pytest.mark.parametrize("times", [0., [[1.]], [True], ["1"], [1j], [None], [np.nan], [np.inf], [-np.inf],
                                    np.ma.array([1.], mask=[True]), [10**1000], [1e308]])
def test_info_bad_times_fail_closed(config, times):
    table = D.build_zero_field(config, 8., 1.)
    with pytest.raises(ValueError):
        D.drift_info(times, table, config)


@pytest.mark.parametrize("name", list(D.DRIFT_FIELDS + D.RESPONSE_FIELDS))
def test_missing_config_fields_rejected(config, name):
    del config[name]
    with pytest.raises(ValueError):
        D.build_zero_field(config, 8., 1.)


@pytest.mark.parametrize("name", ["wireVoltage", "deltaDistance", "deltaTime", "dRdTScale", "kVcm", "cmus",
                                   "driftOffBins", "driftOffset", "driftRMSBins", "signedDriftRMS", "unsignedDriftRMS"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, True, "1", 1j])
def test_bad_config_numerics(config, name, bad):
    config[name] = [bad] * len(config[name]) if isinstance(config[name], list) else bad
    with pytest.raises(ValueError):
        D.build_zero_field(config, 8., 1.)


@pytest.mark.parametrize("name", ["wireVoltage", "deltaDistance", "deltaTime", "dRdTScale", "kVcm", "cmus",
                                   "driftOffBins", "driftOffset", "driftRMSBins", "signedDriftRMS", "unsignedDriftRMS"])
def test_true_mask_rejected_allfalse_mask_permitted(config, name):
    config[name] = np.ma.array(config[name], mask=False)
    D.build_zero_field(config, 8., 1.)
    config[name].mask = True
    with pytest.raises(ValueError, match="masked"):
        D.build_zero_field(config, 8., 1.)


@pytest.mark.parametrize("outer,wire", [(0, 1), (1, 1), (1, 2), (8, 0), (-1, 1), (8, -1),
                                        (np.inf, 1), (8, np.nan), (True, 1), (8, "1"), ([8], 1),
                                        (np.ma.array(8., mask=True), 1), (1e308, 1e-308)])
def test_bad_geometry(config, outer, wire):
    with pytest.raises(ValueError):
        D.build_zero_field(config, outer, wire)


@pytest.mark.parametrize("changes", [dict(deltaDistance=1e-12), dict(deltaTime=1e-12),
                                     dict(deltaDistance=8.), dict(deltaTime=1e100),
                                     dict(wireVoltage=1e-300), dict(kVcm=[1e306, 1e307, 1e308]),
                                     dict(cmus=[1e-300, 2e-300, 4e-300]),
                                     dict(driftOffBins=[-1e308, 1e308])])
def test_grid_and_arithmetic_guards(config, changes):
    config.update(changes)
    with pytest.raises(ValueError):
        D.build_zero_field(config, 8., 1.)


@pytest.mark.parametrize("name", ["distance_grid", "instant_speed", "distance_times", "time_grid", "time_distances",
                                   "deltaD", "deltaT"])
def test_table_missing_mask_nonfinite_and_allfalse(config, name):
    table = D.build_zero_field(config, 8., 1.)
    original = table.pop(name)
    with pytest.raises(ValueError):
        D.drift_info([1.], table, config)
    table[name] = np.ma.array(original, mask=False)
    D.drift_info(np.ma.array([1.], mask=False), table, config)
    table[name].mask = True
    with pytest.raises(ValueError, match="masked"):
        D.drift_info([1.], table, config)
    table[name] = np.full(np.shape(original), np.nan)
    with pytest.raises(ValueError):
        D.drift_info([1.], table, config)


@pytest.mark.parametrize("name,value", [("distance_grid", [0., 1., .5]), ("distance_grid", [0., 2.]),
                                         ("instant_speed", [1.]), ("distance_times", [1., 2.]),
                                         ("time_grid", [1., 2.]), ("time_distances", [1., 2.]),
                                         ("deltaD", 2.), ("deltaT", .3)])
def test_invalid_table_structure(config, name, value):
    table = D.build_zero_field(config, 8., 1.)
    table[name] = value
    with pytest.raises(ValueError):
        D.drift_info([1.], table, config)


@pytest.mark.parametrize("which", ["offset", "error", "scale"])
def test_info_arithmetic_overflow_rejected(config, which):
    table = D.build_zero_field(config, 8., 1.)
    if which == "offset":
        config["driftOffset"] = [1e308, -1e308, 1e308, -1e308]
    elif which == "error":
        config["signedDriftRMS"] = [1e308, -1e308, 1e308, -1e308]
    else:
        config["dRdTScale"] = 1e308
    with pytest.raises(ValueError):
        D.drift_info([1.], table, config)


def test_phi_bins_true_mask_rejected_allfalse_permitted(config):
    config["phiBins"] = np.ma.array(3, mask=False)
    D.build_zero_field(config, 8., 1.)
    config["phiBins"].mask = True
    with pytest.raises(ValueError, match="masked"):
        D.build_zero_field(config, 8., 1.)


def test_time_lookup_first_greater_between_samples_and_end_cap(config):
    config["deltaTime"] = .1
    table = D.build_zero_field(config, 8., 1.)
    # t=.1 lies below the first nonzero time .2: use index=1, not index=0.
    assert table["time_distances"][1] == pytest.approx(1. + (.1 - .2) / (.45 - .2))
    last_time = table["time_grid"][-1]
    assert table["distance_times"][-2] < last_time < table["distance_times"][-1]
    i = len(table["distance_grid"]) - 2
    expected = table["distance_grid"][i] + (last_time - table["distance_times"][i]) / (
        table["distance_times"][i + 1] - table["distance_times"][i]) * table["deltaD"]
    assert table["time_distances"][-1] == expected


def test_two_point_table_and_two_point_regression(config):
    config.update(deltaDistance=5., deltaTime=.1, driftOffset=[.1, .2],
                  signedDriftRMS=[.3, .4], unsignedDriftRMS=[.5, .6])
    table = D.build_zero_field(config, 8., 1.)
    assert len(table["distance_grid"]) == 2
    times = np.array([-1., 0., .1, 3.])
    actual = D.drift_info(times, table, config)
    expected = scalar_info(times, table, config)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_degenerate_regression_coordinates_rejected():
    with pytest.raises(ValueError):
        D._interpolate_calib(np.array([1e16]), [1e16, 1e16 + 2.], [1., 2., 3., 4., 5., 6.])


def test_bulk_lanes_and_no_pulse_dependent_loops(config):
    table = D.build_zero_field(config, 8., 1.)
    times = np.linspace(-1., 5., 20_001)
    actual = D.drift_info(times, table, config)
    assert all(v.shape == times.shape and np.all(np.isfinite(v)) for v in actual.values())
    import ast
    module = ast.parse(Path(D.__file__).read_text())
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("build_zero_field", "drift_info", "_interpolate_calib"):
            assert not any(isinstance(n, ast.While) for n in ast.walk(node))
            for loop in (n for n in ast.walk(node) if isinstance(n, ast.For)):
                assert not any(isinstance(n, ast.Name) and n.id in ("times", "x", "distance") for n in ast.walk(loop.iter))


def test_velocity_preserves_native_source_divide_multiply_order(config):
    config.update(deltaDistance=.13, deltaTime=.037, dRdTScale=1.23456789)
    table = D.build_zero_field(config, 8., 1.)
    times = np.linspace(-.7, 8., 301)
    actual = D.drift_info(times, table, config)
    expected = scalar_info(times, table, config)
    np.testing.assert_array_equal(actual["velocity"], expected["velocity"])
