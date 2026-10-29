"""Synthetic source-calibration operands; no frozen evidence, remote/native jobs, or fit runs."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_initial_measurements as M


@pytest.fixture
def response_path(tmp_path):
    path = tmp_path / "expanded.fcl"
    path.write_text('''services: { conditions: { strawResponse: {
 eBins: 3
 eBinWidth: 0.5
 eHalfPVScale: [0.75, 1, 1.125]
 defaultHalfPropVelocity: 80
 strawHalfPropVelocity: []
 useDb: false
 tdCentralRes: [20, 40, 35]
 tdResSlope: [0.1, 0.2, 0.05]
 centralWirePos: 65
 rmsLongErrors: true
 truncateLongitudinal: true
 wireLengthBuffer: 2
 strawLengthFactor: 0.9
 errorFactor: 1.25
 totTBins: 3
 totTBinWidth: 5
 totEBins: 2
 totEBinWidth: 0.0005
 totDriftTime: [0.13, 2.23, 4.37, 6.43, 8.59, 10.61]
 totDriftError: [1.11, 2.22, 3.33, 4.44, 5.55, 6.66]
 ignoredLiteral: "safe literal"
 ignoredSequence: [1, 2, 3]
} } }
''')
    return path


@pytest.fixture
def config(response_path):
    return M.response_configuration(response_path)


def inputs(n=1):
    measured = dict(energy=np.zeros(n), et0=np.zeros(n), et1=np.zeros(n),
                    tot0=np.zeros(n), tot1=np.zeros(n))
    geometry = dict(hlen=np.full(n, 500.), direction=np.tile([0.123456789, .987654321, .015], (n, 1)))
    return measured, geometry


def scalar_piece(c, name, energy):
    grid = [i * c["eBinWidth"] for i in range(c["eBins"])]
    width = (grid[-1] - grid[0]) / (len(grid) - 1)
    x = 1000. * energy
    i = min(len(grid) - 2, max(0, int(np.floor((x - grid[0]) / width))))
    slope = (c[name][i + 1] - c[name][i]) / width
    return c[name][i] + (x - grid[i]) * slope


def oracle(measured, geometry, c):
    out = {k: [] for k in ("dtime", "timevar", "wdist", "uvar", "udir", "early_end", "signal_speed", "truncated")}
    for i in range(len(measured["energy"])):
        energy, et0, et1, tot0, tot1 = (np.float32(measured[k][i]) for k in ("energy", "et0", "et1", "tot0", "tot1"))
        energy = float(energy)
        halfpv = scalar_piece(c, "eHalfPVScale", energy) * c["defaultHalfPropVelocity"]
        wdist = halfpv * float(np.float32(et1 - et0))
        wderr = scalar_piece(c, "tdCentralRes", energy)
        if c["rmsLongErrors"]:
            if abs(wdist) > c["centralWirePos"]:
                wderr += (abs(wdist) - c["centralWirePos"]) * scalar_piece(c, "tdResSlope", energy)
        else:
            wderr += scalar_piece(c, "tdResSlope", energy) * abs(wdist) * abs(wdist)
        wderr = max(30., wderr)
        truncated = abs(wdist) > geometry["hlen"][i] + c["wireLengthBuffer"] * wderr and c["truncateLongitudinal"]
        if truncated:
            wdist = np.copysign(geometry["hlen"][i] * c["strawLengthFactor"], wdist)
            wderr *= c["errorFactor"]
        early = int(et1 < et0)
        tb = min(c["totTBins"] - 1, int(float(tot1 if early else tot0) / c["totTBinWidth"]))
        eb = min(c["totEBins"] - 1, int(energy / c["totEBinWidth"]))
        index = tb * c["totEBins"] + eb
        out["dtime"].append(float(np.float32(c["totDriftTime"][index] - .1)))
        out["timevar"].append(float(np.float32(c["totDriftError"][index] ** 2)))
        out["wdist"].append(float(np.float32(wdist)))
        out["uvar"].append(float(np.float32(wderr * wderr)))
        out["udir"].append(geometry["direction"][i].astype(np.float32).astype(float))
        out["early_end"].append(early)
        out["signal_speed"].append(2. * halfpv)
        out["truncated"].append(truncated)
    return out


def test_response_configuration_is_jsonable_and_reuses_speed_provenance(response_path, config):
    saved = M.source_speed_configuration(response_path)
    assert all(config[k] == v for k, v in saved.items())
    assert config["eBins"] == 3
    assert config["tdCentralRes"] == [20., 40., 35.]
    assert type(config["rmsLongErrors"]) is bool
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    assert "ignoredLiteral" not in config


@pytest.mark.parametrize("before,after", [
    ("useDb: false", "useDb: true"),
    ("strawHalfPropVelocity: []", "strawHalfPropVelocity: [80]"),
    ("eBins: 3", "eBins: 2"), ("eBins: 3", "eBins: 3.0"),
    ("eBins: 3", "eBins: 1"), ("eBins: 3", "eBins: true"),
    ("eBinWidth: 0.5", "eBinWidth: 0"), ("eBinWidth: 0.5", "eBinWidth: 1e309"),
    ("defaultHalfPropVelocity: 80", "defaultHalfPropVelocity: -1"),
    ("tdCentralRes: [20, 40, 35]", "tdCentralRes: [20, 40]"),
    ("tdCentralRes: [20, 40, 35]", "tdCentralRes: [20, 0, 35]"),
    ("tdResSlope: [0.1, 0.2, 0.05]", "tdResSlope: [0.1, nan, 0.05]"),
    ("eHalfPVScale: [0.75, 1, 1.125]", "eHalfPVScale: [0.75, -1, 1.125]"),
    ("totTBins: 3", "totTBins: 0"), ("totEBins: 2", "totEBins: -1"),
    ("totEBins: 2", "totEBins: 3"), ("totTBinWidth: 5", "totTBinWidth: 0"),
    ("totEBinWidth: 0.0005", "totEBinWidth: -1"),
    ("totDriftError: [1.11, 2.22, 3.33, 4.44, 5.55, 6.66]", "totDriftError: [0, 1, 1, 1, 1, 1]"),
    ("totDriftTime: [0.13, 2.23, 4.37, 6.43, 8.59, 10.61]", "totDriftTime: [0.13, 2.23, 4.37, 6.43, 8.59]"),
    ("rmsLongErrors: true", "rmsLongErrors: 1"),
    ("truncateLongitudinal: true", "truncateLongitudinal: \"true\""),
    ("wireLengthBuffer: 2", "wireLengthBuffer: -1"),
    ("errorFactor: 1.25", "errorFactor: 0"),
    ("strawLengthFactor: 0.9", "strawLengthFactor: false"),
    ("centralWirePos: 65", "centralWirePos: -1"),
    ("centralWirePos: 65", "centralWirePos: 65\n centralWirePos: 60"),
    ("centralWirePos: 65", ""),
    ("tdResSlope: [0.1, 0.2, 0.05]", "tdResSlope: [0.1,,0.2,0.05]"),
    ("tdResSlope: [0.1, 0.2, 0.05]", "tdResSlope: [0.1,0.2,0.05,]"),
    ("tdResSlope: [0.1, 0.2, 0.05]", "tdResSlope: [[0.1,0.2,0.05]]"),
    ("tdResSlope: [0.1, 0.2, 0.05]", "tdResSlope: @local::slope"),
    ("ignoredLiteral: \"safe literal\"", "ignoredLiteral: {x: 1}"),
    ("} } }", ""), ("} } }", "} } }\nstrawResponse: {}"),
    ("} } }", "} } }\nservices.conditions.strawResponse.useDb: true"),
])
def test_response_parser_rejects_malformed_or_unsupported(response_path, before, after):
    response_path.write_text(response_path.read_text().replace(before, after))
    with pytest.raises(ValueError):
        M.response_configuration(response_path)


def test_response_parser_multiline_sequences_and_scientific_literals(response_path):
    response_path.write_text(response_path.read_text().replace("[20, 40, 35]", "[\n +2e1,\n 4.0E1, 35.\n]"))
    assert M.response_configuration(response_path)["tdCentralRes"] == [20., 40., 35.]


@pytest.mark.parametrize("rms,truncate", [(True, True), (True, False), (False, True), (False, False)])
def test_vectorized_reconstruction_matches_scalar_source_oracle(config, rms, truncate):
    n = 64
    measured, geometry = inputs(n)
    rng = np.random.default_rng(42)
    measured.update(energy=rng.uniform(0, .003, n), et0=rng.uniform(-3., 4., n),
                    et1=rng.uniform(-8., 12., n), tot0=rng.uniform(0, 30, n), tot1=rng.uniform(0, 30, n))
    geometry["hlen"] = rng.uniform(20, 150, n)
    config.update(rmsLongErrors=rms, truncateLongitudinal=truncate)
    before = deepcopy((measured, geometry, config))
    expected = oracle(measured, geometry, config)
    actual = M.reconstruct(measured, geometry, config)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    for old, new in zip(before, (measured, geometry, config)):
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])
    assert all(v.dtype == np.float64 for k, v in actual.items() if k not in ("early_end", "truncated"))


def test_float_subtraction_not_double_and_tie_selects_cal(config):
    measured, geometry = inputs(3)
    measured.update(et0=np.array([2**25, 1., 1. + 1e-9]), et1=np.array([1., 1., 1.]),
                    tot0=np.zeros(3), tot1=np.full(3, 10.))
    config["truncateLongitudinal"] = False
    actual = M.reconstruct(measured, geometry, config)
    np.testing.assert_array_equal(actual["early_end"], [1, 0, 0])
    np.testing.assert_array_equal(actual["dtime"], np.float32([8.49, .03, .03]).astype(float))
    assert actual["wdist"][0] == float(np.float32(-60. * 2**25))
    expected_error = 20. + (60. * 2**25 - 65.) * .1
    assert actual["uvar"][0] == float(np.float32(expected_error * expected_error))
    assert float(np.float32(1.) - np.float32(2**25)) != 1. - 2**25


def test_bins_double_division_float_inputs_edges_and_large_clipping(config):
    measured, geometry = inputs(8)
    tinybelow = np.nextafter(np.float32(.0005), np.float32(0))
    measured.update(energy=np.array([0, tinybelow, .0005, .001, .0005, 0, 1e20, 0]),
                    tot0=np.array([0, 5, np.nextafter(np.float32(5), np.float32(0)), 5, 10, 1e20, 1e20, 0]))
    actual = M.reconstruct(measured, geometry, config)
    index = np.array([0, 2, 1, 3, 5, 4, 5, 0])
    np.testing.assert_array_equal(actual["dtime"], np.float32(np.array(config["totDriftTime"])[index] - .1).astype(float))
    np.testing.assert_array_equal(actual["timevar"], np.float32(np.array(config["totDriftError"])[index] ** 2).astype(float))
    assert np.isfinite(actual["signal_speed"]).all()


def test_tot_energy_division_is_double_not_float(config):
    measured, geometry = inputs()
    value = np.float32(.0005)
    config["totEBinWidth"] = float(value) + 1e-12
    config["totTBinWidth"] = 5. + 1e-8
    measured["energy"][:] = value
    measured["tot0"][:] = 5.
    actual = M.reconstruct(measured, geometry, config)
    assert actual["dtime"][0] == float(np.float32(.13 - .1))


def test_central_threshold_floor_and_quadratic_mode(config):
    measured, geometry = inputs(5)
    config.update(eHalfPVScale=[1., 1., 1.], scale=[1., 1., 1.], defaultHalfPropVelocity=1.,
                  mean_half_velocity=1., tdCentralRes=[20., 20., 20.], tdResSlope=[2., 2., 2.])
    measured["et1"][:] = [65., -65., 70., -71., 0.]
    result = M.reconstruct(measured, geometry, config)
    np.testing.assert_array_equal(result["uvar"], np.array([30., 30., 30., 32., 30.]) ** 2)
    config["rmsLongErrors"] = False
    result = M.reconstruct(measured, geometry, config)
    expected = np.maximum(30., 20. + 2. * measured["et1"] ** 2)
    np.testing.assert_array_equal(result["uvar"], np.float32(expected ** 2).astype(float))


def test_truncation_strict_threshold_sign_error_and_uncast_position(config):
    measured, geometry = inputs(4)
    config.update(eHalfPVScale=[1., 1., 1.], scale=[1., 1., 1.], defaultHalfPropVelocity=1.,
                  mean_half_velocity=1., tdCentralRes=[30., 30., 30.], tdResSlope=[0., 0., 0.])
    geometry["hlen"][:] = 100.
    measured["et1"][:] = [160., np.nextafter(np.float32(160), np.float32(np.inf)), -161., 161.]
    actual = M.reconstruct(measured, geometry, config)
    np.testing.assert_array_equal(actual["truncated"], [False, True, True, True])
    np.testing.assert_array_equal(actual["wdist"], [160., 90., -90., 90.])
    np.testing.assert_array_equal(actual["uvar"], [900., 1406.25, 1406.25, 1406.25])
    config["truncateLongitudinal"] = False
    assert not M.reconstruct(measured, geometry, config)["truncated"].any()


def test_direction_is_float_cast_without_normalization(config):
    measured, geometry = inputs()
    actual = M.reconstruct(measured, geometry, config)["udir"]
    np.testing.assert_array_equal(actual, geometry["direction"].astype(np.float32).astype(float))
    assert not np.isclose(np.linalg.norm(actual), 1., rtol=1e-6)


def test_piece_line_slope_first_and_extrapolation():
    grid = np.array([0., .3, .6])
    values = np.array([2.234567891, 8.345678912, 3.456789123])
    x = np.array([0., .123456789, .3, .5, .6, 1.4, 1e20])
    expected = []
    for value in x:
        i = min(1, max(0, int(np.floor(value / .3))))
        slope = (values[i + 1] - values[i]) / .3
        expected.append(values[i] + (value - grid[i]) * slope)
    actual = M._piece_line(grid, values, x)
    np.testing.assert_array_equal(actual, expected)
    index = np.clip(np.floor(x / .3), 0, 1).astype(np.int64)
    multiply_first = values[index] + (x - grid[index]) * (values[index + 1] - values[index]) / .3
    assert np.any(actual != multiply_first)


def test_variances_square_double_errors_before_combohit_cast(config):
    measured, geometry = inputs()
    error = 35.123456789
    config.update(tdCentralRes=[error] * 3, totDriftError=[error] * 6)
    actual = M.reconstruct(measured, geometry, config)
    expected = float(np.float32(error * error))
    assert expected != float(np.float32(float(np.float32(error)) ** 2))
    assert actual["uvar"][0] == actual["timevar"][0] == expected


@pytest.mark.parametrize("directive", ['#include "overrides.fcl"', 'BEGIN_PROLOG', 'END_PROLOG'])
def test_parser_rejects_unexpanded_directives(response_path, directive):
    response_path.write_text(response_path.read_text() + directive + "\n")
    with pytest.raises(ValueError, match="unexpanded"):
        M.response_configuration(response_path)


@pytest.mark.parametrize("field,value", [
    ("energy", [-1e-100]), ("energy", [np.nan]), ("energy", [np.inf]), ("energy", [1e40]),
    ("energy", [[0.]]), ("energy", []), ("energy", ["0"]), ("energy", [True]),
    ("tot0", [-1]), ("tot1", [-1]), ("tot1", [np.inf]), ("et0", [np.nan]),
    ("et0", [1e40]), ("et1", [0., 1.]), ("et1", [1j]),
])
def test_rejects_bad_measurements(config, field, value):
    measured, geometry = inputs()
    measured[field] = value
    with pytest.raises(ValueError):
        M.reconstruct(measured, geometry, config)


@pytest.mark.parametrize("field,value", [
    ("hlen", [0.]), ("hlen", [-1.]), ("hlen", [np.inf]), ("hlen", 500.),
    ("direction", [[0., 0., 0.]]), ("direction", [[1., np.nan, 0.]]),
    ("direction", [[1e40, 0., 0.]]), ("direction", [[1., 0.]]),
])
def test_rejects_bad_geometry(config, field, value):
    measured, geometry = inputs()
    geometry[field] = value
    with pytest.raises(ValueError):
        M.reconstruct(measured, geometry, config)


@pytest.mark.parametrize("field,value", [
    ("rmsLongErrors", 1), ("truncateLongitudinal", "false"), ("eBins", True),
    ("totTBins", 1.5), ("totDriftError", [1e-100] * 6), ("tdCentralRes", [1e308] * 3),
    ("totDriftTime", [1e40] * 6), ("totDriftError", [1e30] * 6),
    ("tdResSlope", [[1., 2., 3.]]), ("eBinWidth", .1),
])
def test_revalidates_caller_config_and_rejects_arithmetic_overflow(config, field, value):
    measured, geometry = inputs()
    config[field] = value
    with pytest.raises(ValueError):
        M.reconstruct(measured, geometry, config)


@pytest.mark.parametrize("group,field", [
    ("measured", name) for name in ("energy", "et0", "et1", "tot0", "tot1")
] + [
    ("geometry", name) for name in ("hlen", "direction")
] + [
    ("config", name) for name in M.POSITIVE + M.NONNEGATIVE + M.SEQUENCES
])
@pytest.mark.parametrize("has_mask", [True, False], ids=["masked", "all_false_mask"])
def test_reconstruct_checks_masks_before_array_conversion(config, group, field, has_mask):
    measured, geometry = inputs(2)
    expected = M.reconstruct(measured, geometry, config)
    target = {"measured": measured, "geometry": geometry, "config": config}[group]
    mask = np.zeros(np.shape(target[field]), dtype=bool)
    mask.reshape(-1)[-1] = has_mask
    target[field] = np.ma.array(target[field], mask=mask)
    if has_mask:
        label = field if group == "config" else group + " " + field
        with pytest.raises(ValueError, match="masked values: " + label):
            M.reconstruct(measured, geometry, config)
    else:
        actual = M.reconstruct(measured, geometry, config)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def test_rejects_float_time_subtraction_overflow(config):
    measured, geometry = inputs()
    measured.update(et0=[-3e38], et1=[3e38])
    with pytest.raises(ValueError, match="nonfinite"):
        M.reconstruct(measured, geometry, config)


def test_empty_batch_has_consistent_shapes(config):
    result = M.reconstruct(*inputs(0), config)
    for field, value in result.items():
        assert value.shape == ((0, 3) if field == "udir" else (0,))
