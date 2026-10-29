"""Fail-closed all-target CA census tests; synthetic inputs, no fits or captured data."""
from contextlib import nullcontext
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import f1_ca_population_audit as A

C = A.C


def replace(table, field, row, value):
    """Fixture-only edit preserving the other Awkward fields."""
    values = ak.to_list(table[field])
    values[row] = value
    return ak.with_field(table, ak.Array(values), field)


@pytest.fixture
def measured():
    # Repeated event numbers across run/subrun, an empty event, ragged source hits,
    # and unrelated geometry order deliberately defeat positional/event-only joins.
    events = ak.Array({
        "run": [8, 7, 7, 7, 7], "subrun": [2, 1, 2, 2, 3],
        "event": [9, 9, 9, 10, 9],
        "s_sid": [[0x18A2], [], [0x1421, 0x18A2, 0x2024], [0x1C23, 0x1421], [0x2024]],
        "s_time": [[81.], [], [21., 22., 23.], [31., 32.], [91.]],
        "s_et0": [[84.], [], [24., 25., 26.], [34., 35.], [94.]],
        "s_et1": [[82.], [], [27., 25., 26.], [33., 37.], [95.]],
        "s_edep": [[.00005], [], [0., .0002, .0001], [.0003, .0002], [.0001]],
    })
    cal = np.array([[31., 32., 33.], [41., 42., 43.], [11., 12., 13.], [21., 22., 23.]])
    hv = cal + [100., -200., 300.]
    geometry = {"sid": np.array([[0x1C23, 0x2024, 0x1421, 0x18A2]])}
    for prefix, xyz in (("wcal", cal), ("whv", hv)):
        for i, axis in enumerate("xyz"):
            geometry[prefix + axis] = xyz[None, :, i].copy()
    hkeys = np.array([[7, 2, 9, 2], [8, 2, 9, 0], [7, 2, 10, 1],
                      [7, 2, 9, 0], [7, 2, 10, 0], [7, 3, 9, 0]])
    raw = np.array([0x2024, 0x18A2, 0x1421, 0x1421, 0x1C23, 0x2024])
    cond = {"straw": {"halfvp": 80. + np.arange(5000) / 1000.},
            "halfvpscale": np.array([1., 1.5, 2.])}
    return events, geometry, hkeys, raw, cond


def test_hit_inputs_full_event_source_and_geometry_joins_signs_ties_counts(measured):
    before = deepcopy(measured)
    line, corrected = A.hit_inputs(*measured)
    cal = np.array([[41., 42., 43.], [21., 22., 23.], [11., 12., 13.],
                    [11., 12., 13.], [31., 32., 33.], [41., 42., 43.]])
    hv_early = np.array([False, True, False, False, True, False])
    displacement = np.array([100., -200., 300.])
    np.testing.assert_array_equal(line["end"], cal + hv_early[:, None] * displacement)
    np.testing.assert_array_equal(line["mid"], cal + .5 * displacement)
    np.testing.assert_allclose(line["direction"],
                               np.where(hv_early[:, None], 1., -1.) * displacement / np.linalg.norm(displacement))
    np.testing.assert_allclose(line["length"], np.full(6, np.linalg.norm(displacement)))
    np.testing.assert_array_equal(line["measurement_time"], [26., 82., 35., 24., 33., 94.])
    np.testing.assert_array_equal(corrected, [23., 81., 32., 21., 31., 91.])
    # MeV -> keV, factor two, unique-straw lookup, interpolation AND extrapolation.
    unique = np.array([4644, 3586, 2913, 2913, 4067, 4644])
    np.testing.assert_allclose(line["speed"], 2 * np.array([1.5, 1.25, 2., 1., 2.5, 1.5]) * (80. + unique / 1000.))
    assert corrected.shape == (6,) and line["end"].shape == (6, 3)
    assert ak.to_list(measured[0]) == ak.to_list(before[0])
    for field in measured[1]:
        np.testing.assert_array_equal(measured[1][field], before[1][field])
    for actual, saved in zip(measured[2:4], before[2:4]):
        np.testing.assert_array_equal(actual, saved)
    np.testing.assert_array_equal(measured[4]["straw"]["halfvp"], before[4]["straw"]["halfvp"])


def test_hit_inputs_empty_batch_and_repeated_source_hits_are_not_lost(measured):
    events, geometry, hkeys, raw, cond = measured
    line, corrected = A.hit_inputs(events, geometry, hkeys[:0], raw[:0], cond)
    assert line["end"].shape == (0, 3) and corrected.shape == (0,)
    repeated = np.array([0, 3, 0, 1, 3])
    baseline, times = A.hit_inputs(*measured)
    actual, corrected = A.hit_inputs(events, geometry, hkeys[repeated], raw[repeated], cond)
    for field in actual:
        np.testing.assert_array_equal(actual[field], baseline[field][repeated])
    np.testing.assert_array_equal(corrected, times[repeated])


@pytest.mark.parametrize("failure,message", [
    ("missing_event", "missing full identity"), ("duplicate_event", "duplicate identities"),
    ("missing_geometry", "missing full identity"), ("duplicate_geometry", "duplicate available"),
    ("negative_source", "negative source-hit"), ("past_source", "source-hit index outside"),
    ("empty_event", "source-hit index outside"), ("float_keys", "invalid event/source-hit"),
    ("short_keys", "invalid event/source-hit"), ("raw_mismatch", "native/source raw straw"),
    ("conditions_range", "straw conditions index outside"),
])
def test_hit_inputs_reject_identity_and_index_errors(measured, failure, message):
    events, geometry, hkeys, raw, cond = measured
    if failure == "missing_event":
        hkeys[0, 0] = 99
    elif failure == "duplicate_event":
        events = ak.concatenate([events, events[:1]])
    elif failure == "missing_geometry":
        geometry["sid"][0, 1] = 0x7777
    elif failure == "duplicate_geometry":
        geometry["sid"][0, 0] = geometry["sid"][0, 1]
    elif failure == "negative_source":
        hkeys[0, 3] = -1
    elif failure == "past_source":
        hkeys[0, 3] = 3
    elif failure == "empty_event":
        hkeys[0] = [7, 1, 9, 0]
    elif failure == "float_keys":
        hkeys = hkeys.astype(float)
    elif failure == "short_keys":
        hkeys = hkeys[:, :3]
    elif failure == "raw_mismatch":
        raw[0] = 0x18A2
    else:
        cond["straw"]["halfvp"] = cond["straw"]["halfvp"][:4000]
    with pytest.raises(ValueError, match=message):
        A.hit_inputs(events, geometry, hkeys, raw, cond)


@pytest.mark.parametrize("field", ["s_time", "s_et0", "s_et1", "s_edep"])
def test_hit_inputs_reject_differing_jagged_counts_even_in_unselected_event(measured, field):
    events, geometry, hkeys, raw, cond = measured
    events = replace(events, field, 1, [123.])
    with pytest.raises(ValueError, match="measured hit lengths differ"):
        A.hit_inputs(events, geometry, hkeys, raw, cond)


@pytest.mark.parametrize("field,value", [
    ("s_time", np.nan), ("s_time", np.inf), ("s_edep", np.nan), ("s_edep", np.inf),
    ("s_et0", np.nan), ("s_et1", -np.inf),
    ("s_et1", np.nan), ("s_et1", np.inf),
])
def test_hit_inputs_reject_nonfinite_measured_fields(measured, field, value):
    events, geometry, hkeys, raw, cond = measured
    values = ak.to_list(events[field][2])
    values[2] = value
    events = replace(events, field, 2, values)
    with np.errstate(invalid="ignore", over="ignore"), pytest.raises(ValueError, match="nonfinite"):
        A.hit_inputs(events, geometry, hkeys, raw, cond)


@pytest.mark.parametrize("field", ["wcalx", "whvy", "halfvp", "halfvpscale"])
def test_hit_inputs_reject_nonfinite_geometry_and_conditions(measured, field):
    if field == "halfvp":
        measured[4]["straw"][field][4644] = np.nan
    elif field == "halfvpscale":
        measured[4][field][1] = np.nan
    else:
        measured[1][field][0, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        A.hit_inputs(*measured)


@pytest.fixture
def synthetic2tracks():
    qseed = np.array([[45., -1.1, 280., 1.4, 73000., 2000.],
                      [57., -1., 200., 1.1, 73005., 2030.]])
    qref = qseed + [[2., .02, -10., -.01, .5, 5.], [-3., -.03, 15., .02, -.4, -7.]]
    mid = np.array([[250., 180., 390.], [290., 170., 480.]])
    wire = np.array([[.8, .6, 0.], [-.8, -.6, 0.]])
    line = C.sensor(mid + 500. * wire, mid - 500. * wire,
                    np.array([73003., 73010.]), np.array([180., 220.]))
    corrected = np.array([73000., 73006.])
    identities = np.array([[8, 2, 9, 0, 7], [8, 2, 9, 1, 2]])
    return qseed, qref, np.array([105.658367, 200.]), line, corrected, identities


def test_census_constructor_then_changed_reference_exact_tolerant_and_no_mutation(synthetic2tracks, monkeypatch):
    qseed, qref, mass, line, corrected, ids = synthetic2tracks
    hints = C.constructor_hint(qseed, line, corrected, mass)
    built = C.solve(qseed, line, *hints, mass=mass)
    state = C.solve(qref, line, built["particle_time"], built["sensor_time"], mass=mass)
    native = np.nextafter(state["particle_time"], np.inf)
    native[1] += 2.
    ctime = np.nextafter(built["particle_time"], -np.inf)
    arrays = [qseed, qref, mass, corrected, ids, native, ctime, *line.values()]
    snapshots = [a.copy() for a in arrays]
    for a in arrays:
        a.setflags(write=False)
    solve, constructor = C.solve, C.constructor_hint
    calls = []

    def record_hint(q, sensor, time, m):
        assert q is qseed and sensor is line and time is corrected and m is mass
        calls.append("constructor")
        return constructor(q, sensor, time, m)

    def record_solve(q, sensor, ph, sh, *, mass):
        n = len(calls)
        assert q is (qseed if n == 1 else qref) and sensor is line
        assert mass is synthetic2tracks[2]
        expected = hints if n == 1 else (built["particle_time"], built["sensor_time"])
        np.testing.assert_array_equal(ph, expected[0])
        np.testing.assert_array_equal(sh, expected[1])
        calls.append("solve")
        return solve(q, sensor, ph, sh, mass=mass)

    monkeypatch.setattr(C, "constructor_hint", record_hint)
    monkeypatch.setattr(C, "solve", record_solve)
    result = A.census(qseed, qref, mass, line, corrected, native, ctime, ids)
    assert calls == ["constructor", "solve", "solve"]
    assert result["n_hits"] == result["n_targets"] == 2
    assert result["identities"] == ids.tolist()
    assert result["identity_fields"] == ["run", "subrun", "event", "iseed", "shindex"]
    for prefix, expected, saved in (("constructor", built, ctime), ("updated", state, native)):
        assert result[prefix + "_time"] == A._metric(expected["particle_time"], saved)
        assert result[prefix + "_time_exact"] == A._metric(expected["particle_time"], saved, exact=True)
    assert result["constructor_time"]["passed"] and not result["constructor_time_exact"]["passed"]
    assert result["updated_time_exact"]["mismatches"] == 2
    assert result["updated_time"]["mismatches"] == 1
    assert result["exact_mismatch_keys"] == ids.tolist()
    assert result["tolerant_mismatch_keys"] == ids[1:].tolist()
    assert result["native_seed_vs_reference"] == A._metric(qseed, qref, exact=True)
    for prefix, expected in (("constructor", built), ("reference", state)):
        for field in ("updates", "status"):
            assert result[prefix + "_" + field] == expected[field].tolist()
    np.testing.assert_array_equal(result["particle_time"], state["particle_time"])
    np.testing.assert_array_equal(result["sensor_time_reconstructed_only"], state["sensor_time"])
    delta = state["particle_time"] - native
    np.testing.assert_array_equal(result["time_delta_ns"], delta)
    np.testing.assert_array_equal(result["native_time_ulp_ns"], np.abs(np.spacing(native)))
    np.testing.assert_array_equal(result["time_delta_quantiles_ns"], np.quantile(delta, [0, .16, .5, .84, 1]))
    assert result["max_abs_time_delta_ns"] == np.max(np.abs(delta))
    for actual, saved in zip(arrays, snapshots):
        np.testing.assert_array_equal(actual, saved)
    json.dumps(result, allow_nan=False)


def test_census_keeps_parallel_unsupported_z_retry_and_zero_distance_lanes():
    # Two targets, four hits: no filtering of failed local CA or large-z lanes.
    q = np.tile([0., 0., 0., 0., 10., 2000.], (4, 1))
    mid = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    direction = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
    # Updated transverse track in lanes 1/2 remains +/-101 mm from wire z.
    ref = q.copy()
    ref[1:3, 2] = [101., -101.]
    ref[1:3, 1] = np.pi / 2
    ref[1:3, 3] = np.pi / 2
    line = C.sensor(mid + 500 * direction, mid - 500 * direction,
                    np.full(4, 12.5), np.full(4, 200.))
    corrected = np.full(4, 10.)
    mass = np.full(4, C.N.MASS)
    ids = np.array([[1, 2, 3, 0, 0], [1, 2, 3, 0, 4], [1, 2, 3, 1, 2], [1, 2, 3, 1, 8]])
    built = C.solve(q, line, *C.constructor_hint(q, line, corrected, mass), mass=mass)
    state = C.solve(ref, line, built["particle_time"], built["sensor_time"], mass=mass)
    result = A.census(q, ref, mass, line, corrected, state["particle_time"], built["particle_time"], ids)
    assert result["n_hits"] == 4 and result["n_targets"] == 2
    assert result["identities"] == ids.tolist()
    assert result["reference_status"] == [0, 0, 0, 4]
    assert result["unusable_keys"] == ids[3:].tolist()
    assert result["reconstructed_z_retry_keys"] == ids[1:].tolist()
    assert result["zero_distance_keys"] == ids[[0, 3]].tolist()
    np.testing.assert_allclose(result["z_distance_mm"], [0., 101., -101., 0.], atol=1e-10)
    np.testing.assert_allclose(result["distance_mm"], [0., 101., 101., 0.], atol=1e-10)
    assert result["max_abs_z_distance_mm"] == pytest.approx(101.)
    for field, direction in (("delta_dot_particle_mm", state["direction"]), ("delta_dot_sensor_mm", line["direction"])):
        np.testing.assert_array_equal(result[field], np.sum(state["delta"] * direction, axis=-1))
    assert result["updated_time_exact"]["n_components"] == 4
    assert result["updated_time_exact"]["passed"]
    json.dumps(result, allow_nan=False)


@pytest.fixture
def captured():
    """All four target-key coordinates matter; pieces/seeds/hits are scrambled."""
    targets = np.array([[8, 2, 9, 0], [7, 2, 9, 0], [7, 3, 9, 0], [7, 2, 9, 1], [7, 2, 10, 0]])
    iterations, pieces, seeds, effects = [], [], [], []
    for i, target in enumerate(targets):
        key = dict(zip(A.KEYS, target.tolist()))
        q = [45. + i, -1.1, 280. + i, 1.4, 73000. + i, 2000.]
        iterations.append(dict(key, iteration_ordinal=0, iter=0, capture_valid=True,
                               effects_captured=True, nold=1))
        pieces.append(dict(key, iteration_ordinal=0, owner=0, piece_index=0, parameters=q, mass=105. + i))
        indices = [7, 2] if i == 0 else [4]
        sid = [0x2024, 0x1421] if i == 0 else [0x18A2]
        times = [100. + i, 200. + i] if i == 0 else [100. + i]
        seeds.append(dict(key, valid=True, parameters=q, mass=105. + i,
                          sh_shindex=indices, sh_strawid=sid, sh_time=times))
        for j, source in enumerate(indices):
            effects.append(dict(key, iteration_ordinal=0, iter=0, identity_kind=1,
                                identity_valid=True, has_reference=True, payload_supported=True,
                                shindex=source, strawid=sid[j], reference_parameters=q, time=times[j]))
    # Out-of-scope records must not corrupt initial straw membership.
    iterations.append(dict(iterations[0], iteration_ordinal=1, iter=1, capture_valid=False, nold=3))
    pieces.append(dict(pieces[0], owner=1, parameters=[999.] * 6))
    effects.append(dict(effects[0], iteration_ordinal=1, iter=1, identity_valid=False))
    effects.append(dict(effects[0], identity_kind=2, shindex=999, identity_valid=False))
    return (ak.Array(iterations), ak.Array(pieces)[[3, 1, 5, 4, 0, 2]],
            ak.Array(effects)[[4, 1, 5, 2, 0, 3, 6, 7]], ak.Array(seeds)[[2, 4, 0, 3, 1]])


def test_select_initial_fullkeys_membership_and_all_targets_not_numeric_closure(captured):
    before = [ak.to_list(table) for table in captured]
    hits, qseed, qref, mass, ctime, hkeys = A.select_initial(*captured)
    assert len(hits) == 6 and len(np.unique(hkeys[:, :4], axis=0)) == 5
    np.testing.assert_array_equal(hkeys, A.keys(captured[2][:6], A.KEYS + ("shindex",)))
    np.testing.assert_array_equal(qseed, qref)
    np.testing.assert_array_equal(qref, A.dense(hits, "reference_parameters", 6))
    np.testing.assert_array_equal(mass, [108., 105., 109., 106., 105., 107.])
    np.testing.assert_array_equal(ctime, [103., 200., 104., 101., 100., 102.])
    assert [ak.to_list(table) for table in captured] == before


@pytest.mark.parametrize("table", [0, 1, 2, 3])
@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_select_initial_rejects_missing_duplicate_target_piece_seed_or_hit(captured, table, failure):
    tables = list(captured)
    tables[table] = tables[table][1:] if failure == "missing" else ak.concatenate([tables[table], tables[table][:1]])
    with pytest.raises(ValueError, match="missing|duplicate|census"):
        A.select_initial(*tables)


@pytest.mark.parametrize("table,field,row,value,message", [
    (0, "iter", 0, 1, "first ordinal is not initial"),
    (0, "capture_valid", 0, False, "invalid initial capture"),
    (0, "effects_captured", 0, False, "invalid initial capture"),
    (0, "nold", 0, 2, "not one piece"),
    (0, "nold", 0, 0, "not one piece"),
    (1, "piece_index", 0, 1, "piece index differs"),
    (1, "owner", 0, 1, "missing"),
    (3, "valid", 0, False, "invalid constructor seed"),
    (3, "parameters", 0, [0.] * 6, "old parameters differ from constructor"),
    (2, "reference_parameters", 0, [0.] * 6, "reference differs from old"),
    (2, "reference_parameters", 0, [np.nan] * 6, "nonfinite reference_parameters"),
    (2, "reference_parameters", 0, [1.] * 5, "reference_parameters: shape"),
    (2, "identity_valid", 0, False, "unsupported initial straw"),
    (2, "has_reference", 0, False, "unsupported initial straw"),
    (2, "payload_supported", 0, False, "unsupported initial straw"),
    (2, "iter", 0, 1, "hit initial iteration differs"),
    (2, "shindex", 0, 99, "missing full identity"),
    (2, "shindex", 0, -1, "invalid integer identity"),
    (2, "iseed", 0, 99, "missing full identity"),
    (2, "strawid", 0, 0x1421, "constructor raw straw differs"),
    (3, "mass", 0, 999., "constructor/reference mass differs"),
    (3, "sh_shindex", 2, [7], "initial hit census differs"),
    (3, "sh_shindex", 2, [7, 7], "duplicate available identities"),
    (3, "sh_shindex", 2, [7, 99], "missing full identity"),
])
def test_select_initial_rejects_bad_flags_onepiece_reference_and_membership(captured, table, field, row, value, message):
    tables = list(captured)
    tables[table] = replace(tables[table], field, row, value)
    with pytest.raises(ValueError, match=message):
        A.select_initial(*tables)


def test_select_initial_requires_every_target_census_not_just_total_hits(captured):
    tables = list(captured)
    # Move one hit to another otherwise valid target, preserving total hit count.
    for field in A.KEYS:
        tables[2] = replace(tables[2], field, 0, ak.to_list(tables[2][field][2]))
    tables[2] = replace(tables[2], "shindex", 0, 99)
    with pytest.raises(ValueError, match="initial hit census differs"):
        A.select_initial(*tables)


@pytest.mark.parametrize("table,field,value,message", [
    (0, "iteration_ordinal", 1, "no captured first iterations"),
    (2, "identity_kind", 2, "no initial straw hits"),
])
def test_select_initial_rejects_empty_initial_populations(captured, table, field, value, message):
    tables = list(captured)
    tables[table] = ak.with_field(tables[table], np.full(len(tables[table]), value), field)
    with pytest.raises(ValueError, match=message):
        A.select_initial(*tables)


@pytest.mark.parametrize("later_only_target", [False, True])
def test_run_requires_all_captured_targets_but_not_numeric_closure(captured, monkeypatch, later_only_target):
    """Exercise the orchestration gate without file writes, ROOT data, or a fit."""
    iterations, pieces, effects, seeds = captured
    if later_only_target:
        # A target occurring only in ordinal 1 cannot vanish from this census.
        iterations = replace(iterations, "event", len(iterations) - 1, 999)
    output = []
    checks = []
    args = SimpleNamespace(out=Path("/synthetic-ca-test/out.json"),
                           check=Path("/synthetic-ca-test/check.json"),
                           conditions=Path("/synthetic-ca-test/conditions.json"),
                           native_source=Path("/synthetic-ca-test/native"))
    for name in list(os.environ):
        if name.startswith("F1_"):
            monkeypatch.delenv(name)

    def evidence(path, report):
        assert path == args.check
        report.update(inputs={"capture": {"path": "capture"}, "baseline": {"path": "baseline"}},
                      highest_cycles={"capture": {}}, check_sha256="synthetic-hash")
        return iterations, pieces

    root = {
        "trkdump/events": SimpleNamespace(arrays=lambda *a, **kw: "events"),
        "trkdump/conditions": SimpleNamespace(arrays=lambda *a, **kw: "geometry"),
    }

    def inputs(events, geometry, hkeys, raw_sid, conditions):
        assert events == "events" and geometry == "geometry" and conditions == "conditions"
        expected = A.keys(effects[:6], A.KEYS + ("shindex",))
        np.testing.assert_array_equal(hkeys, expected[:, [0, 1, 2, 4]])
        np.testing.assert_array_equal(raw_sid, A.column(effects[:6], "strawid"))
        return "line", "corrected"

    def count(qseed, qref, mass, line, corrected, native_time, constructor_time, identities):
        checks.append(identities.copy())
        assert line == "line" and corrected == "corrected"
        assert len(qseed) == len(qref) == len(mass) == len(native_time) == len(constructor_time) == 6
        # Deliberately disagree: numerical agreement is NOT a coverage precondition.
        return {"n_hits": 6, "n_targets": 5,
                "constructor_time_exact": {"passed": False},
                "updated_time_exact": {"passed": False}, "updated_time": {"passed": False},
                "max_abs_time_delta_ns": 3., "max_abs_z_distance_mm": 101.,
                "unusable_keys": identities[:1].tolist(),
                "reconstructed_z_retry_keys": identities[:2].tolist(), "zero_distance_keys": []}

    def open_output(path, mode="r", *args, **kwargs):
        assert path == Path("/synthetic-ca-test/out.json") and mode == "x"
        output.append(io.StringIO())
        return nullcontext(output[-1])

    monkeypatch.setattr(Path, "exists", lambda path: False)
    monkeypatch.setattr(Path, "glob", lambda path, pattern: [])
    monkeypatch.setattr(Path, "read_text", lambda path: "TPOCAPrecision: 1e-4\n")
    monkeypatch.setattr(Path, "open", open_output)
    monkeypatch.setattr(A, "digest", lambda path: "synthetic-hash")
    monkeypatch.setattr(A, "load_evidence", evidence)
    monkeypatch.setattr(A.uproot, "open", lambda path: nullcontext(root))
    monkeypatch.setattr(A, "read_tree", lambda root, group, tree, cycles: effects if tree == "effects" else seeds)
    monkeypatch.setattr(A.hitreco, "load_conditions", lambda path: "conditions")
    monkeypatch.setattr(A, "hit_inputs", inputs)
    monkeypatch.setattr(A, "census", count)
    if later_only_target:
        with pytest.raises(ValueError, match="first-iteration hit target coverage differs"):
            A.run(args)
        assert not output
    else:
        A.run(args)
        assert len(output) == 1
        report = json.loads(output[0].getvalue())
        assert report["completed"]
        assert not report["full_closure"] and not report["full_numerical_validation"]
        assert not report["f1_generated_operands"]
        assert len(report["captured_first_iteration_targets"]) == 5
        assert report["census"]["updated_time"]["passed"] is False
        assert len(report["census"]["unusable_keys"]) == 1
    assert len(checks) == 1 and len(checks[0]) == 6
