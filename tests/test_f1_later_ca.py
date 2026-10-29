"""Synthetic later-reference contracts; no captured data, fits, builds or remote IO."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_later_ca as L
from trkfastsim.reco import nativeca as C


def operands(times=(1.5, .5), hints=(.5,)):
    p = np.zeros((1, len(times), 6))
    p[0, :, 1], p[0, :, 3] = -np.pi/2, np.pi/2
    p[0, :, 4], p[0, :, 5] = times, 2000.
    n = len(hints)
    line = C.sensor(np.tile([100., 0., 0.], (n, 1)), np.tile([-100., 0., 0.], (n, 1)),
                    np.full(n, 4.), np.full(n, 200.))
    return dict(parameters=p, ranges=np.array([[[i, i+1] for i in range(len(times))]], float),
                counts=np.array([len(times)]), parent=np.zeros(n, int), line=line,
                particle_hint=np.array(hints), sensor_hint=np.full(n, 3.), mass=np.full(n, 105.6583755))


def scalar_nearest(ranges, time):
    if time <= ranges[0, 0]:
        return 0
    if time >= ranges[-1, 1]:
        return len(ranges)-1
    return next(i for i, (_, end) in enumerate(ranges) if end >= time)


def scalar_search(args, lane, cap):
    owner = args["parent"][lane]
    r = args["ranges"][owner, :args["counts"][owner]]
    line = {k: v[lane] for k, v in args["line"].items()}
    pt, st = args["particle_hint"][lane], args["sensor_hint"][lane]
    index = scalar_nearest(r, pt)
    solved, moves, exhausted = index, 0, False
    state = C.solve(args["parameters"][owner, index], line, pt, st,
                    precision=L.NATIVE_CA_PRECISION, mass=args["mass"][lane])
    history = [index]
    if cap:
        while True:
            old = index
            index = scalar_nearest(r, state["particle_time"])
            if index == old or not state["usable"]:
                break
            if moves >= cap:
                exhausted = True
                if state["status"] == 0:
                    state["status"] = np.array(1)
                break
            state = C.solve(args["parameters"][owner, index], line, state["particle_time"],
                            state["sensor_time"], precision=L.NATIVE_CA_PRECISION, mass=args["mass"][lane])
            solved = index
            moves += 1
            history.append(index)
    return dict(state, index=index, solution_index=solved, moves=moves, exhausted=exhausted,
                solution_history=np.array(history+[-1]*(cap+1-len(history))))


@pytest.mark.parametrize("times", [(1.5, .5), (1.5, 2.5, 2.75), (1., 1.), (.5,)])
@pytest.mark.parametrize("cap", [0, 1, 2, 10])
def test_piecewise_scalar_search_and_retained_fields(times, cap):
    args = operands(times, (-1., .5, 1., 1.5, 20.))
    original = deepcopy(args)
    actual = L.piecewise(**args, max_moves=cap)
    for lane in range(len(args["parent"])):
        expected = scalar_search(args, lane, cap)
        for key in actual:
            np.testing.assert_array_equal(actual[key][lane], expected[key], err_msg=key)
    for k in args:
        if k == "line":
            for field in args[k]:
                np.testing.assert_array_equal(args[k][field], original[k][field])
        else:
            np.testing.assert_array_equal(args[k], original[k])


def test_cap_retains_solution_and_convergence_changes_only_at_exhaustion():
    args = operands()
    result = L.piecewise(**args, max_moves=10)
    assert result["index"][0] == 1 and result["solution_index"][0] == 0
    assert result["moves"][0] == 10 and result["exhausted"][0]
    assert result["status"][0] == 1 and result["usable"][0]
    np.testing.assert_array_equal(result["solution_history"], [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])
    assert result["particle_time"][0] == pytest.approx(1.5)
    zero = L.piecewise(**args, max_moves=0)
    assert zero["index"][0] == zero["solution_index"][0] == zero["moves"][0] == 0
    assert zero["status"][0] == 0 and not zero["exhausted"][0]


def test_initial_nearest_boundaries_and_zero_duration_begin_precedence():
    times = [-1., 0., np.nextafter(1., 0.), 1., np.nextafter(1., 2.), 2., 3., 4.]
    args = operands((.5, 1.5, 2.5), times)
    actual = L.piecewise(**args, max_moves=0)
    np.testing.assert_array_equal(actual["index"], [0, 0, 0, 0, 1, 1, 2, 2])
    args["ranges"][:] = 0
    args["particle_hint"][:] = 0
    np.testing.assert_array_equal(L.piecewise(**args, max_moves=0)["index"], 0)
    args["particle_hint"][:] = 1
    np.testing.assert_array_equal(L.piecewise(**args, max_moves=0)["index"], 2)


def test_mixed_parents_padding_and_independent_lane_stops():
    args = operands((1.5, 2.5, 2.75), (.5, 1.5, 2.8, .5))
    args["parameters"] = np.concatenate([args["parameters"], args["parameters"]])
    args["ranges"] = np.concatenate([args["ranges"], args["ranges"]])
    args["counts"] = np.array([3, 1])
    args["parent"][-1] = 1
    args["parameters"][1, 1:] = np.nan
    args["ranges"][1, 1:] = np.nan
    result = L.piecewise(**args)
    np.testing.assert_array_equal(result["index"], [2, 2, 2, 0])
    np.testing.assert_array_equal(result["moves"], [2, 1, 0, 0])
    np.testing.assert_array_equal(result["solution_history"][:, :4], [[0, 1, 2, -1], [1, 2, -1, -1],
                                                                   [2, -1, -1, -1], [0, -1, -1, -1]])
    for key in ("parameters", "ranges"):
        args[key] = np.ma.masked_invalid(args[key])
    masked = L.piecewise(**args)
    for key in result:
        np.testing.assert_array_equal(result[key], masked[key])


def test_pointer_advances_before_unusable_check_and_stopped_lanes_not_resolved(monkeypatch):
    args = operands((1.5, 2.5, 2.75), (.5, 1.5, 2.8))
    real_solve, calls = C.solve, []

    def fake(q, line, pt, st, **kwargs):
        calls.append((q.copy(), pt.copy(), st.copy()))
        state = real_solve(q, line, pt, st, **kwargs)
        # Lane 0 is an unusable retained state pointing outside the initial piece.
        # This cannot arise from a fresh parallel straight-line solve, but is the
        # native source branch contract for an arbitrary returned failure state.
        unusable = q[:, 4] == 1.5
        state["status"][unusable], state["usable"][unusable] = 5, False
        return state

    monkeypatch.setattr(L.C, "solve", fake)
    result = L.piecewise(**args)
    np.testing.assert_array_equal(result["index"], [1, 2, 2])
    np.testing.assert_array_equal(result["solution_index"], [0, 2, 2])
    np.testing.assert_array_equal(result["moves"], [0, 1, 0])
    np.testing.assert_array_equal(result["usable"], [False, True, True])
    assert not result["exhausted"].any()
    assert [len(call[0]) for call in calls] == [3, 1]
    np.testing.assert_allclose(calls[1][1], [2.5], atol=1e-15)
    np.testing.assert_allclose(calls[1][2], [3.5], atol=1e-15)


def test_parallel_ca_sentinels_and_empty_batch():
    args = operands((.5,), (.5, .5))
    args["line"]["direction"][0] = [0., -1., 0.]
    result = L.piecewise(**args)
    np.testing.assert_array_equal(result["status"], [4, 0])
    assert np.isposinf(result["particle_update"][0]) and not result["usable"][0]
    empty = operands(hints=())
    result = L.piecewise(**empty)
    assert result["particle_poca"].shape == (0, 3)
    assert result["solution_history"].shape == (0, 11)
    empty["parameters"] = np.empty((0, 1, 6))
    empty["ranges"] = np.empty((0, 1, 2))
    empty["counts"] = np.empty(0, int)
    assert L.piecewise(**empty)["index"].shape == (0,)


def test_exhaustion_does_not_promote_other_usable_status(monkeypatch):
    args = operands()
    real_solve = C.solve

    def fake(*a, **kw):
        state = real_solve(*a, **kw)
        state["status"][:] = 2
        return state

    monkeypatch.setattr(L.C, "solve", fake)
    result = L.piecewise(**args, max_moves=1)
    assert result["exhausted"][0] and result["status"][0] == 2


def test_hint_policies_precedence_native_midpoint_and_no_mutation():
    line = operands(hints=(.5, .5, .5, .5))["line"]
    line["measurement_time"] = np.array([4., 6., 8., 10.])
    line["length"] = np.array([200., 400., 600., 800.])
    own = dict(usable=np.array([1, 1, 0, 0], np.uint8), particle_time=np.arange(4.)+20,
               sensor_time=np.arange(4.)+30)
    associated = dict(usable=np.array([True, False, True, False]), particle_time=np.arange(4.)+40,
                      sensor_time=np.arange(4.)+50)
    before = deepcopy((own, associated, line))
    ph, sh = L.hit_hint(own, line, np.array([1., 2., 3., 4.]))
    np.testing.assert_array_equal(ph, [20., 21., 3.5, 4.])
    np.testing.assert_array_equal(sh, [30., 31., 6.5, 8.])
    ph, sh = L.crossing_hint(own, associated, line)
    np.testing.assert_array_equal(ph, [20., 21., 42., 8.])
    np.testing.assert_array_equal(sh, [30., 31., 52., 8.])
    midpoint = line["measurement_time"]-.5*line["length"]/line["speed"]
    for h in L.crossing_hint(None, None, line):
        np.testing.assert_array_equal(h, midpoint)
    np.testing.assert_array_equal(L.hit_hint(None, line, np.ones(4))[0], midpoint-1.)
    for actual, expected in zip((own, associated, line), before):
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("cap", [True, False, np.bool_(True), 1., -1, 1001, np.nan, "10"])
def test_invalid_move_cap(cap):
    with pytest.raises(ValueError, match="max_moves"):
        L.piecewise(**operands(), max_moves=cap)


def test_numpy_integer_cap_and_native_float_precision(monkeypatch):
    real_solve, observed = C.solve, []

    def spy(*a, **kw):
        observed.append(kw["precision"])
        return real_solve(*a, **kw)

    monkeypatch.setattr(L.C, "solve", spy)
    L.piecewise(**operands(), max_moves=np.int64(0))
    assert observed == [float(np.float32(1e-4))]


@pytest.mark.parametrize("precision", [True, "1e-4", [1e-4], 0., -1., np.nan, np.inf,
                                        np.ma.array(1e-4, mask=True)])
def test_invalid_precision(precision):
    with pytest.raises(ValueError):
        L.piecewise(**operands(), precision=precision)


@pytest.mark.parametrize("key", ["parameters", "ranges", "counts", "parent", "particle_hint", "sensor_hint", "mass"])
def test_masks_rejected_before_conversion(key):
    args = operands()
    args[key] = np.ma.array(args[key], mask=np.ones_like(args[key], bool))
    with pytest.raises(ValueError, match="missing"):
        L.piecewise(**args)


@pytest.mark.parametrize("key", ["end", "mid", "direction", "length", "measurement_time", "speed"])
def test_sensor_mask_shape_and_nonfinite_guards(key):
    args = operands()
    args["line"][key] = np.ma.array(args["line"][key], mask=True)
    with pytest.raises(ValueError, match="missing"):
        L.piecewise(**args)
    args = operands()
    args["line"][key] = args["line"][key][0]
    with pytest.raises(ValueError):
        L.piecewise(**args)
    args = operands()
    args["line"][key][...] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        L.piecewise(**args)


@pytest.mark.parametrize("key,value", [
    ("counts", [0]), ("counts", [3]), ("counts", [2.]), ("counts", [True]),
    ("counts", np.array([2**64-1], np.uint64)), ("parent", [-1]), ("parent", [1]),
    ("parent", [0.]), ("parent", [False]), ("parent", np.array([2**64-1], np.uint64)),
    ("particle_hint", [np.nan]), ("sensor_hint", [np.inf]), ("mass", [0.]), ("mass", [-1.]),
    ("mass", [np.inf]), ("mass", 105.), ("particle_hint", [".5"]),
    ("sensor_hint", [False]), ("parameters", np.zeros((2, 6))), ("ranges", np.zeros((2, 2))),
])
def test_type_shape_and_value_guards(key, value):
    args = operands()
    args[key] = value
    with pytest.raises(ValueError):
        L.piecewise(**args)


@pytest.mark.parametrize("ranges", [[[[0., 1.], [1.1, 2.]]], [[[0., 1.], [.9, 2.]]],
                                     [[[1., 0.], [0., 2.]]], [[[0., 1.], [1., np.nan]]]])
def test_ranges_must_be_finite_sorted_and_contiguous(ranges):
    args = operands()
    args["ranges"] = ranges
    with pytest.raises(ValueError):
        L.piecewise(**args)


@pytest.mark.parametrize("field,value", [("speed", 0.), ("length", -1.), ("direction", 2.)])
def test_sensor_physical_guards(field, value):
    args = operands()
    args["line"][field][...] = value
    with pytest.raises(ValueError):
        L.piecewise(**args)


@pytest.mark.parametrize("key,field", [("own", "usable"), ("own", "particle_time"), ("own", "sensor_time"),
                                      ("associated", "usable"), ("associated", "particle_time"),
                                      ("associated", "sensor_time")])
def test_retained_state_masks_even_when_unselected(key, field):
    states = {k: dict(usable=np.array([False]), particle_time=np.array([1.]), sensor_time=np.array([2.]))
              for k in ("own", "associated")}
    states[key][field] = np.ma.array(states[key][field], mask=True)
    with pytest.raises(ValueError, match="missing"):
        L.crossing_hint(**states, line=operands()["line"])


@pytest.mark.parametrize("flag", [[-1], [2], [1.], [np.nan], ["True"], [[True]]])
def test_hint_usable_strict(flag):
    state = dict(usable=flag, particle_time=[1.], sensor_time=[2.])
    with pytest.raises(ValueError):
        L.hit_hint(state, operands()["line"], [1.])


def test_hint_bad_state_drift_and_nonfinite_unselected_state():
    line = operands()["line"]
    for state in ({}, dict(usable=[False], particle_time=[np.nan], sensor_time=[1.])):
        with pytest.raises(ValueError):
            L.hit_hint(state, line, [1.])
    for drift in (np.ma.array([1.], mask=True), [np.inf], 1., [True], ["1"]):
        with pytest.raises(ValueError):
            L.hit_hint(None, line, drift)


@pytest.mark.parametrize("kind", ["particle_position", "particle_momentum", "mass", "sensor_position", "unit_norm"])
def test_solver_overflow_fails_closed(kind):
    args = operands()
    if kind == "particle_position":
        args["particle_hint"][:] = 1e308
    elif kind == "particle_momentum":
        args["parameters"][..., 5] = 1e308
    elif kind == "mass":
        args["mass"][:] = 1e308
    elif kind == "sensor_position":
        args["sensor_hint"][:] = 1e308
    else:
        args["line"]["direction"][:] = 1e308
    with pytest.raises(ValueError, match="nonfinite"):
        L.piecewise(**args)


def test_hint_overflow_fails_closed():
    line = operands()["line"]
    line["measurement_time"][:] = -1e308
    with pytest.raises(ValueError, match="nonfinite"):
        L.hit_hint(None, line, [1e308])
    line["length"][:] = 1e308
    line["speed"][:] = 1e-308
    with pytest.raises(ValueError, match="nonfinite"):
        L.crossing_hint(None, None, line)
