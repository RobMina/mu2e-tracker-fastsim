"""Conditional LineFinder -> native seed construction, not full-fit equivalence.

The frozen LineFinder producer assigns FitParams == MinuitParams and _t0.
Consequently its saved sd_* values are sufficient here; this is not a claim
about arbitrary producers. All operands are synthetic, with no receipt edits,
remote jobs, captured constructor outputs, or fitter execution.
"""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_seed_contract as S


@pytest.fixture
def config():
    # Independent literals: changing the production constants must not change
    # the expected frozen contract along with them.
    return dict(momentum=2000., errors=[15., .05, 10., .02, 2., 1.],
                constraints=[0., 0., 0., 0., 0., .0001])


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "expanded.fcl"
    block = """{
 SeedMomentum: 2000
 FitParticle: 13
 SeedErrors: [15, 0.05, 10, 0.02, 2, 1]
 ParameterConstraints: [0, 0, 0, 0, 0, 0.0001]
}
"""
    path.write_text('source: { module_label: "LineFinder" module_type: "LineFinder" }\n'
                    + "first: " + block + "second: " + block)
    return path


def source_momentum(raw):
    """ROOT XYZVectorF: each product/sum, sqrt, reciprocal and scale is float."""
    v = np.column_stack((raw[:, 2], -np.ones(len(raw)), raw[:, 3])).astype(np.float32)
    square = np.float32(np.float32(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])
                        + v[:, 2] * v[:, 2])
    norm = np.sqrt(square)
    reciprocal = np.float32(1.) / norm
    normalized = np.float32(v * reciprocal[:, None])
    return np.float32(normalized * np.float32(2000.)).astype(np.float64)


def test_generate_float_unit_reciprocal_then_float_scale_before_double_norm(config):
    raw = np.array([[17., -31., .2, .7, 19.],
                    [-12., 40., .123456789, -.987654321, -8.]])
    mass = np.array([105.6583755, 511.])
    before = raw.copy(), mass.copy(), deepcopy(config)
    actual = S.generate(raw, mass, config)
    expected = source_momentum(raw)
    np.testing.assert_array_equal(actual["momentum3"], expected)
    p2 = (expected[:, 0] ** 2 + expected[:, 1] ** 2) + expected[:, 2] ** 2
    np.testing.assert_array_equal(actual["parameters"][:, 5], np.sqrt(p2))
    np.testing.assert_array_equal(actual["speed"], np.sqrt(p2) / np.sqrt(p2 + mass * mass) * 299.792458)
    assert actual["momentum3"].dtype == np.dtype("float64")
    # These operands actually distinguish all tempting reassociations.
    v = np.column_stack((raw[:, 2], -np.ones(len(raw)), raw[:, 3])).astype(np.float32)
    norm = np.sqrt((v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) + v[:, 2] * v[:, 2])
    division = ((v / norm[:, None]) * np.float32(2000.)).astype(float)
    assert np.any(expected != division)
    promoted_scale = (v * (np.float32(1.) / norm)[:, None]).astype(float) * 2000.
    assert np.any(expected != promoted_scale)
    double_v = v.astype(float)
    double_normalized = double_v / np.linalg.norm(double_v, axis=1)[:, None] * 2000.
    assert np.any(expected != double_normalized)
    assert np.any(actual["parameters"][:, 5] != 2000.)
    np.testing.assert_array_equal(raw, before[0])
    np.testing.assert_array_equal(mass, before[1])
    assert config == before[2]


def test_dot_preserves_cartesian_three_term_grouping_and_float32():
    v = np.array([[1e10, -1e10, 3.]], dtype=np.float32)
    actual = S.dot(v, np.ones_like(v))
    assert actual.dtype == np.dtype("float32")
    np.testing.assert_array_equal(actual, [3.])


@pytest.mark.parametrize("x,z,t", [(12., 31., 7.), (-12., -31., -7.), (0., 0., 0.)])
def test_simple_down_seed_has_analytic_source_geometry(config, x, z, t):
    actual = S.generate(np.array([[x, z, 0., 0., t]]), np.array([105.658]), config)
    # Position is (A0, 0, B0), direction is (0, -1, 0); already at POCA.
    np.testing.assert_array_equal(actual["momentum3"], [[0., -2000., 0.]])
    np.testing.assert_allclose(actual["parameters"],
                               [[x, -np.pi / 2, z, np.pi / 2, t, 2000.]], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(actual["parallel"], [False])


@pytest.mark.parametrize("a0,b0,a1,b1,t0", [
    (13., 27., .2, .7, 80.), (-13., -27., -.3, -.8, -80.),
    (0., -9., -.7, .2, 0.), (5., -7., 0., 5000., -4.),
    (-5., 7., 0., -5000., 4.),
])
def test_nonparallel_geometry_agrees_with_xy_minimization(config, a0, b0, a1, b1, t0):
    raw = np.array([[a0, b0, a1, b1, t0]])
    actual = S.generate(raw, np.array([105.658]), config)
    mom = source_momentum(raw)[0]
    momentum = np.sqrt((mom[0] ** 2 + mom[1] ** 2) + mom[2] ** 2)
    direction = mom / momentum
    speed = momentum / np.sqrt(momentum ** 2 + 105.658 ** 2) * 299.792458
    # Independent geometric minimization of transverse distance to the Z axis.
    step = -a0 * direction[0] / (direction[0] ** 2 + direction[1] ** 2)
    point = np.array([a0, 0., b0]) + step * direction
    d0 = np.copysign(np.hypot(point[0], point[1]), a0)
    expected = [d0, np.arctan2(direction[1], direction[0]), point[2],
                np.arccos(direction[2]), t0 + step / speed, momentum]
    np.testing.assert_allclose(actual["parameters"][0], expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(actual["parallel"], [False])
    assert np.all(np.isfinite(actual["parameters"]))


@pytest.mark.parametrize("b1", [20000., -20000., 1e12, -1e12])
def test_parallel_source_fallback_is_finite_and_not_transverse_doca(config, b1):
    actual = S.generate(np.array([[3., 4., 0., b1, -17.]]), np.array([105.658]), config)
    # POCAUtil's sinsq < 1e-8 fallback sets both line displacements to zero.
    # The distance is then 3D to the origin, not just abs(A0).
    np.testing.assert_array_equal(actual["parallel"], [True])
    np.testing.assert_allclose(actual["parameters"][0, [0, 2, 4]], [5., 4., -17.], rtol=0, atol=0)
    assert np.all(np.isfinite(actual["parameters"]))
    assert np.all(np.isfinite(actual["speed"]))


def test_covariance_constraint_mask_copies_and_range_midpoints(config):
    raw = np.array([[3., 4., .2, .7, -17.], [-3., -4., -.2, -.7, 17.]])
    generated = S.generate(raw, np.full(2, 105.658), config)
    expected = np.diag(np.square([15., .05, 10., .02, 2., 1.]))
    np.testing.assert_array_equal(generated["covariance"], np.broadcast_to(expected, (2, 6, 6)))
    q = generated["parameters"]
    ranges = np.array([[-11., 9.], [4., 4.]])
    old_q, old_ranges = q.copy(), ranges.copy()
    result = S.constraint(q, ranges, config)
    np.testing.assert_array_equal(result["parameters"], q)
    np.testing.assert_array_equal(result["mask"], [[0, 0, 0, 0, 0, 1]] * 2)
    assert result["mask"].dtype == np.dtype("int64")
    constraint_covariance = np.diag([1., 1., 1., 1., 1., .0001 ** 2])
    np.testing.assert_array_equal(result["covariance"], np.broadcast_to(constraint_covariance, (2, 6, 6)))
    np.testing.assert_array_equal(result["time"], [-1., 4.])
    shifted = S.constraint(q, ranges + 20., config)
    np.testing.assert_array_equal(shifted["parameters"], result["parameters"])
    np.testing.assert_array_equal(shifted["time"], result["time"] + 20.)
    result["parameters"][0, 0] = 999.
    result["covariance"][0, 0, 0] = 999.
    result["mask"][0, 0] = 1
    assert result["covariance"][1, 0, 0] == 1.
    assert result["mask"][1, 0] == 0
    generated["covariance"][0, 0, 0] = 999.
    assert generated["covariance"][1, 0, 0] == 225.
    np.testing.assert_array_equal(q, old_q)
    np.testing.assert_array_equal(ranges, old_ranges)


@pytest.mark.parametrize("ranges", [np.zeros((2, 3)), np.array([[2., 1.], [0., 1.]]),
                                    np.array([[np.nan, 1.], [0., 1.]]),
                                    np.array([[0., np.inf], [0., 1.]])])
def test_constraint_rejects_invalid_ranges(config, ranges):
    with pytest.raises(ValueError, match="constraint range"):
        S.constraint(np.zeros((2, 6)), ranges, config)


@pytest.mark.parametrize("q", [np.zeros((2, 5)), np.full((2, 6), np.nan), np.full((2, 6), np.inf)])
def test_constraint_rejects_invalid_parameters(config, q):
    with pytest.raises(ValueError, match="constraint parameters"):
        S.constraint(q, np.zeros((2, 2)), config)


def test_constraint_rejects_other_sigmas(config):
    config["constraints"][0] = .1
    with pytest.raises(ValueError, match="constraint sigmas"):
        S.constraint(np.zeros((1, 6)), np.zeros((1, 2)), config)


@pytest.mark.parametrize("raw", [np.zeros((1, 4)), np.zeros((1, 5), dtype=np.int64),
                                  np.full((1, 5), np.nan), np.full((1, 5), np.inf)])
def test_generate_rejects_invalid_raw(config, raw):
    with pytest.raises(ValueError, match="raw seed"):
        S.generate(raw, np.ones(1), config)


@pytest.mark.parametrize("mass", [np.array([0.]), np.array([-1.]), np.array([np.nan]),
                                   np.array([np.inf]), np.ones((1, 1)), np.ones(2)])
def test_generate_rejects_invalid_mass(config, mass):
    with pytest.raises(ValueError, match="seed mass"):
        S.generate(np.zeros((1, 5)), mass, config)


def test_generate_rejects_empty_population(config):
    with pytest.raises(ValueError, match="seed mass"):
        S.generate(np.zeros((0, 5)), np.zeros(0), config)


@pytest.mark.parametrize("field,value", [("momentum", 1999.), ("errors", [1.] * 6),
                                         ("constraints", [0.] * 6)])
def test_generate_rejects_unfrozen_config(config, field, value):
    config[field] = value
    with pytest.raises(ValueError, match="seed configuration"):
        S.generate(np.zeros((1, 5)), np.ones(1), config)


def test_bounds_jagged_shuffled_parents_keep_all_hits_and_singletons():
    parent = np.array([2, 0, 2, 1, 0, 2, 0])
    times = np.array([-9., 20., 4., 7., -3., -2., 10000.], dtype=np.float32)
    expected = [[-3., 10000.], [7., 7.], [-9., 4.]]
    np.testing.assert_array_equal(S.bounds(parent, times, 3), expected)
    order = np.array([5, 0, 3, 6, 1, 4, 2])
    np.testing.assert_array_equal(S.bounds(parent[order], times[order], 3), expected)


@pytest.mark.parametrize("parent,times,n,error", [
    ([0, 2], [1., 2.], 3, "empty range"),
    (np.array([], dtype=np.int64), np.array([], dtype=float), 1, "empty range"),
    ([-1, 0], [1., 2.], 1, "range parent"), ([0, 2], [1., 2.], 2, "outside"),
    ([0], [1.], 0, "outside"), ([0.], [1.], 1, "range parent"),
    ([True], [1.], 1, "range parent"),
    (np.array([2 ** 63], dtype=np.uint64), [1.], 1, "overflow"),
    ([[0], [0]], [1., 2.], 1, "range parent"),
    ([0, 0], [1.], 1, "range times"), ([0], [1], 1, "range times"),
    ([0], [np.nan], 1, "range times"), ([0], [np.inf], 1, "range times"),
    ([0], [-np.inf], 1, "range times"),
])
def test_bounds_rejects_invalid_population(parent, times, n, error):
    with pytest.raises(ValueError, match=error):
        S.bounds(parent, times, n)


def jagged(rows, dtype):
    return ak.values_astype(ak.Array(rows), dtype)


@pytest.fixture
def events():
    # Same event number in different runs/subruns, two seeds in event zero,
    # shared hits across seeds, nonmonotone memberships, and an unselected event.
    data = dict(run=np.array([1, 2, 1, 1]), subrun=np.array([2, 2, 3, 2]),
                event=np.array([9, 9, 9, 10]))
    raw = np.array([[11., 21., .2, .7, 31.], [12., 22., -.3, -.8, 32.],
                    [13., 23., .4, .9, 33.], [14., 24., -.5, -.6, 34.]])
    for i, name in enumerate(("a0", "b0", "a1", "b1", "t0")):
        data["sd_" + name] = ak.unflatten(raw[:, i], [2, 1, 1, 0])
    data.update(sd_n=jagged([[2, 2], [2], [1], []], np.int32),
                sd_ind=jagged([[2, 0, 2, 1], [1, 0], [2], []], np.int32),
                s_time=jagged([[9., -3., 9., 10000.], [7., 1.], [0., 50., -2.], [123.]], np.float32),
                s_sid=jagged([[10, 11, 65535, 13], [20, 21], [30, 31, 0], [40]], np.uint16))
    return ak.Array(data)


@pytest.fixture
def target_keys():
    return np.array([[1, 2, 9, 1], [2, 2, 9, 0], [1, 2, 9, 0], [1, 3, 9, 0]])


def test_inputs_join_full_keys_seed_order_and_ordered_membership(events, target_keys):
    actual = S.inputs(events, target_keys)
    np.testing.assert_array_equal(actual["raw"],
        [[12., 22., -.3, -.8, 32.], [13., 23., .4, .9, 33.],
         [11., 21., .2, .7, 31.], [14., 24., -.5, -.6, 34.]])
    parent = np.array([0, 0, 1, 1, 2, 2, 3])
    np.testing.assert_array_equal(actual["parent"], parent)
    np.testing.assert_array_equal(actual["hit_keys"],
                                  np.column_stack((target_keys[parent], [2, 1, 1, 0, 2, 0, 2])))
    np.testing.assert_array_equal(actual["strawid"], [65535, 11, 21, 20, 65535, 10, 0])
    # The unselected late hit at 10000 is excluded; all source seed hits count.
    np.testing.assert_array_equal(actual["construction_range"], [[-3., 9.], [1., 7.], [9., 9.], [-2., -2.]])
    assert actual["raw"].dtype == np.dtype("float64")


def test_inputs_event_and_target_shuffles_are_equivariant(events, target_keys):
    before = S.inputs(events, target_keys)
    event_shuffle = events[[3, 2, 0, 1]]
    shuffled = S.inputs(event_shuffle, target_keys)
    for name in before:
        np.testing.assert_array_equal(shuffled[name], before[name])
    order = np.array([2, 0, 3, 1])
    shuffled = S.inputs(event_shuffle, target_keys[order])
    for name in ("raw", "construction_range"):
        np.testing.assert_array_equal(shuffled[name], before[name][order])
    hit_order = np.array([4, 5, 0, 1, 6, 2, 3])
    np.testing.assert_array_equal(shuffled["hit_keys"], before["hit_keys"][hit_order])
    np.testing.assert_array_equal(shuffled["strawid"], before["strawid"][hit_order])
    np.testing.assert_array_equal(shuffled["parent"], [0, 0, 1, 1, 2, 3, 3])


@pytest.mark.parametrize("change,error", [
    (lambda k: k[[0, 0]], "duplicate"), (lambda k: k.astype(float), "integer"),
    (lambda k: k.astype(bool), "integer"), (lambda k: k[:, :3], "integer"),
    (lambda k: k + [0, 0, 0, -2], "negative"),
    (lambda k: k + [0, 0, 0, 2], "seed index"),
    (lambda k: k + [10, 0, 0, 0], "missing full identity"),
    (lambda k: k + [0, 10, 0, 0], "missing full identity"),
    (lambda k: k + [0, 0, 10, 0], "missing full identity"),
    (lambda k: np.full(k.shape, 2 ** 63, dtype=np.uint64), "overflow"),
])
def test_inputs_rejects_invalid_target_keys(events, target_keys, change, error):
    with pytest.raises(ValueError, match=error):
        S.inputs(events, change(target_keys))


@pytest.mark.parametrize("field,rows,dtype,error", [
    ("sd_a0", [[11., 12.], [13.], [14.], []], np.float32, "raw seed field"),
    ("sd_b0", [[21.], [23.], [24.], []], np.float64, "raw seed lengths"),
    ("sd_t0", [[31., np.nan], [33.], [34.], []], np.float64, "raw seed field"),
    ("sd_b1", [[.7, -.8], [np.inf], [-.6], []], np.float64, "raw seed field"),
    ("sd_n", [[2], [2], [1], []], np.int64, "membership lengths"),
    ("sd_n", [[2, 2], [2], [1], []], np.float64, "hit counts"),
    ("sd_n", [[-1, 2], [2], [1], []], np.int64, "hit counts"),
    ("sd_n", [[1, 2], [2], [1], []], np.int64, "flattening"),
    ("sd_ind", [[2, 0, 2, 1], [1, 0], [2], []], np.float64, "hit indices"),
    ("sd_ind", [[2, -1, 2, 1], [1, 0], [2], []], np.int64, "hit indices"),
    ("sd_ind", [[4, 0, 2, 1], [1, 0], [2], []], np.int64, "outside event"),
    ("sd_ind", [[2, 2, 2, 1], [1, 0], [2], []], np.int64, "duplicate"),
    ("s_time", [[9., -3., 9., 10000.], [7., 1.], [0., 50., -2.], [123.]], np.float64, "corrected time"),
    ("s_time", [[9., -3., np.nan, 10000.], [7., 1.], [0., 50., -2.], [123.]], np.float32, "corrected time"),
    ("s_time", [[9., -3., 9., 10000.], [7., 1.], [0., 50., -2.], [np.inf]], np.float32, "corrected time"),
    ("s_sid", [[10, 11, 12], [20, 21], [30, 31, 32], [40]], np.int64, "straw lengths"),
    ("s_sid", [[10, 11, 12, 13], [20, 21], [30, 31, 32], [40]], np.float64, "straw id"),
    ("s_sid", [[10, 11, -1, 13], [20, 21], [30, 31, 32], [40]], np.int64, "straw id"),
    ("s_sid", [[10, 11, 65536, 13], [20, 21], [30, 31, 32], [40]], np.int64, "straw id"),
])
def test_inputs_rejects_malformed_jagged_fields(events, target_keys, field, rows, dtype, error):
    events = ak.with_field(events, jagged(rows, dtype), field)
    with pytest.raises(ValueError, match=error):
        S.inputs(events, target_keys)


@pytest.mark.parametrize("field,rows", [
    ("sd_a0", [[11., None], [13.], [14.], []]),
    ("sd_n", [[2, None], [2], [1], []]),
    ("sd_ind", [[2, None, 2, 1], [1, 0], [2], []]),
    ("s_time", [[9., None, 9., 10000.], [7., 1.], [0., 50., -2.], [123.]]),
    ("s_sid", [[10, None, 12, 13], [20, 21], [30, 31, 32], [40]]),
])
def test_inputs_rejects_missing_jagged_values(events, target_keys, field, rows):
    dtype = np.float32 if field == "s_time" else np.float64 if field == "sd_a0" else np.int64
    events = ak.with_field(events, jagged(rows, dtype), field)
    with pytest.raises(ValueError, match="missing"):
        S.inputs(events, target_keys)


def test_inputs_rejects_empty_selected_seed(events, target_keys):
    events = ak.with_field(events, jagged([[4, 0], [2], [1], []], np.int64), "sd_n")
    with pytest.raises(ValueError, match="empty source seed"):
        S.inputs(events, target_keys[:1])


def test_inputs_rejects_duplicate_event_identity(events, target_keys):
    with pytest.raises(ValueError, match="duplicate"):
        S.inputs(events[[0, 0, 1, 2]], target_keys)


@pytest.mark.parametrize("field", ["run", "subrun", "event"])
@pytest.mark.parametrize("dtype,value,error", [(float, 1., "dtype"), (np.int64, -1, "negative")])
def test_inputs_rejects_invalid_event_identity(events, target_keys, field, dtype, value, error):
    events = ak.with_field(events, np.full(len(events), value, dtype=dtype), field)
    with pytest.raises(ValueError, match=error):
        S.inputs(events, target_keys)


def test_configuration_accepts_only_frozen_literal_values(config_path, config):
    assert S.configuration(config_path) == config
    config_path.write_text(config_path.read_text().replace("2000", "+2e3").replace(
        "[15, 0.05, 10, 0.02, 2, 1]", "[\n15., 5e-2, 1e1, 2e-2, 2., 1.\n]"))
    assert S.configuration(config_path) == config


@pytest.mark.parametrize("before,after", [
    ("SeedMomentum: 2000", "SeedMomentum: 1999"),
    ("FitParticle: 13", "FitParticle: -13"),
    ("SeedErrors: [15, 0.05, 10, 0.02, 2, 1]", "SeedErrors: [15, 0.05, 10, 0.02, 2, 2]"),
    ("ParameterConstraints: [0, 0, 0, 0, 0, 0.0001]", "ParameterConstraints: [0, 0, 0, 0, 0, 0]"),
    ("SeedErrors: [15, 0.05, 10, 0.02, 2, 1]", "SeedErrors: [15, 0.05, 10, 0.02, 2]"),
    ("SeedErrors: [15, 0.05, 10, 0.02, 2, 1]", "SeedErrors: @local::errors"),
    ("SeedMomentum: 2000", "SeedMomentum: @local::momentum"),
    ("SeedMomentum: 2000", "SeedMomentum: nan"),
    ("SeedMomentum: 2000", "SeedMomentum: inf"),
    ("0.0001]", "nan]"), ("0.0001]", "inf]"),
    ('module_label: "LineFinder"', 'module_label: "OtherFinder"'),
    ('module_type: "LineFinder"', 'module_type: "OtherFinder"'),
])
def test_configuration_rejects_unsupported_literals(config_path, before, after):
    config_path.write_text(config_path.read_text().replace(before, after, 1))
    with pytest.raises(ValueError):
        S.configuration(config_path)


@pytest.mark.parametrize("name,line", [
    ("SeedMomentum", " SeedMomentum: 2000\n"), ("FitParticle", " FitParticle: 13\n"),
    ("SeedErrors", " SeedErrors: [15, 0.05, 10, 0.02, 2, 1]\n"),
    ("ParameterConstraints", " ParameterConstraints: [0, 0, 0, 0, 0, 0.0001]\n"),
])
@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_configuration_requires_exactly_two_occurrences(config_path, name, line, operation):
    text = config_path.read_text()
    config_path.write_text(text.replace(line, "", 1) if operation == "missing" else text + line)
    with pytest.raises(ValueError, match=name):
        S.configuration(config_path)


def test_configuration_rejects_unexpanded_include(config_path):
    config_path.write_text('#include "other.fcl"\n' + config_path.read_text())
    with pytest.raises(ValueError, match="unexpanded"):
        S.configuration(config_path)


@pytest.mark.parametrize("line", [
    "SeedErrors: [15, 0.05, 10, 0.02, 2, 1]",
    "ParameterConstraints: [0, 0, 0, 0, 0, 0.0001]",
    'source: { module_label: "LineFinder" module_type: "LineFinder" }',
])
def test_configuration_comments_cannot_supply_required_source_or_sequences(config_path, line):
    config_path.write_text(config_path.read_text().replace(line, "# " + line, 1))
    with pytest.raises(ValueError):
        S.configuration(config_path)


def test_configuration_rejects_conflicting_dotted_override(config_path):
    config_path.write_text(config_path.read_text() + "first.SeedMomentum: 1500\n")
    with pytest.raises(ValueError):
        S.configuration(config_path)


def test_production_uses_columnar_operations_not_event_or_seed_python_loops():
    tree = ast.parse(inspect.getsource(S))
    assert not any(isinstance(node, ast.Attribute) and node.attr == "to_list"
                   for node in ast.walk(tree))
    # Iteration over the fixed five branch names is not iteration over events.
    for name in ("inputs", "bounds", "generate", "constraint"):
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        for node in ast.walk(function):
            assert not isinstance(node, (ast.While, ast.AsyncFor, ast.comprehension))
            if isinstance(node, ast.For):
                assert name == "inputs"
                assert isinstance(node.iter, ast.Name) and node.iter.id == "SEED_FIELDS"
