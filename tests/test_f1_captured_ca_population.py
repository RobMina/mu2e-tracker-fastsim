"""Synthetic ALL147 driver contracts; no native evidence, fit, build or remote jobs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_captured_ca_population as P


def change(table, field, value, row=0):
    records = ak.to_list(table)
    records[row][field] = value
    return ak.Array(records)


def oracle_information(q, H, residual, weight):
    W = np.einsum("nri,nr,nrj->nij", H, weight, H)
    b = np.einsum("nij,nj->ni", W, q)-np.einsum("nri,nr,nr->ni", H, weight, residual)
    return W, b


@pytest.fixture
def measured():
    events = ak.Array(dict(run=[7, 8, 7, 7], subrun=[2, 2, 3, 1], event=[9, 9, 9, 9],
                          s_sid=[[0x1421, 0x18A2], [0x18A2], [0x1421], []],
                          s_time=[[20., 22.], [81.], [90.], []],
                          s_et0=[[24., 25.], [84.], [95.], []],
                          s_et1=[[27., 25.], [82.], [94.], []],
                          s_edep=[[0., .0002], [.00005], [.0003], []]))
    # Values where recovering a late endpoint through 2*mid-end loses bits.
    cal = np.array([[1.1, 32., 33.], [41., 42., 43.]])
    hv = np.array([[-1e-16, -168., 333.], [141., -158., 343.]])
    geometry = {"sid": np.array([[0x18A2, 0x1421]])}
    for prefix, xyz in (("wcal", cal), ("whv", hv)):
        for i, axis in enumerate("xyz"):
            geometry[prefix+axis] = xyz[None, :, i].copy()
    hkeys = np.array([[7, 2, 9, 1], [8, 2, 9, 0], [7, 3, 9, 0], [7, 2, 9, 0]])
    raw = np.array([0x18A2, 0x18A2, 0x1421, 0x1421])
    conditions = {"straw": {"halfvp": 80.+np.arange(5000)/1000.}, "halfvpscale": np.array([1., 1.5, 2.])}
    config = dict(scale=[1., 1.4, 1.9], bin_width=.1, mean_half_velocity=87.)
    return events, geometry, hkeys, raw, conditions, config


def test_measured_full_join_ties_energy_and_both_endpoints(measured, monkeypatch):
    events, geometry, hkeys, raw, cond, config = measured
    expected_early = np.array([[1.1, 32., 33.], [-1e-16, -168., 333.], [141., -158., 343.], [41., 42., 43.]])
    expected_late = np.array([[-1e-16, -168., 333.], [1.1, 32., 33.], [41., 42., 43.], [141., -158., 343.]])
    actual_sensor, actual_loader = P.A.sensor, P.hit_inputs
    calls = []

    def sensor(early, late, time, speed):
        np.testing.assert_array_equal(early, expected_early)
        np.testing.assert_array_equal(late, expected_late)
        calls.append("sensor")
        return actual_sensor(early, late, time, speed)

    def loader(*args):
        calls.append("loader")
        return actual_loader(*args)

    monkeypatch.setattr(P.A, "sensor", sensor)
    monkeypatch.setattr(P, "hit_inputs", loader)
    monkeypatch.setattr(ak, "to_list", lambda *a, **k: pytest.fail("no per-event record conversion"))
    data, lines = P.measured_sensors(*measured)
    assert calls == ["loader", "sensor", "sensor"]
    assert tuple(lines) == P.SPEED_CONTROLS
    np.testing.assert_array_equal(data["corrected_time"], [22., 81., 90., 20.])
    np.testing.assert_array_equal(data["measurement_time"], [25., 82., 94., 24.])
    np.testing.assert_array_equal(data["energy"], [.0002, .00005, .0003, 0.])
    assert (2*lines[P.SPEED_CONTROLS[0]]["mid"]-data["early"])[0, 0] != data["late"][0, 0]
    # Independent interpolation/extrapolation of source FCL coefficients, no response inversion.
    expected_source = 174.*np.array([1.9, 1.2, 2.4, 1.])
    np.testing.assert_allclose(lines[P.SPEED_CONTROLS[1]]["speed"], expected_source, rtol=1e-15)
    expected_dump = 2*np.array([2., 1.25, 2.5, 1.])*(80.+np.array([3586, 3586, 2913, 2913])/1000.)
    np.testing.assert_allclose(lines[P.SPEED_CONTROLS[0]]["speed"], expected_dump, rtol=1e-15)


def test_repeated_wanted_hit_valid_across_seeds(measured):
    events, geometry, hkeys, raw, *_ = measured
    repeated = np.array([0, 2, 0, 1])
    once = P.endpoint_measurements(events, geometry, hkeys, raw)
    again = P.endpoint_measurements(events, geometry, hkeys[repeated], raw[repeated])
    for field in once:
        np.testing.assert_array_equal(again[field], once[field][repeated])


@pytest.mark.parametrize("failure,message", [
    ("duplicate_event", "duplicate"), ("duplicate_geometry", "duplicate"),
    ("missing_event", "missing full identity"), ("missing_straw", "missing full identity"),
    ("past_source", "outside event"), ("empty_event", "outside event"),
    ("negative_source", "negative"), ("float_keys", "invalid event/source"),
    ("raw_mismatch", "raw straw"), ("geometry_length", "endpoint lengths"),
    ("extra_geometry", "one dumped geometry"), ("negative_energy", "measured energy"),
    ("length_mismatch", "hit lengths"), ("nonfinite_geometry", "nonfinite"),
])
def test_endpoint_join_guards(measured, failure, message):
    events, geom, hkeys, raw, *_ = measured
    if failure == "duplicate_event":
        events = ak.concatenate([events, events[:1]])
    elif failure == "duplicate_geometry":
        geom["sid"][0, 1] = geom["sid"][0, 0]
    elif failure == "missing_event":
        hkeys[0, 0] = 100
    elif failure == "missing_straw":
        geom["sid"][0, 0] = 999
    elif failure == "past_source":
        hkeys[0, 3] = 2
    elif failure == "empty_event":
        hkeys[0] = [7, 1, 9, 0]
    elif failure == "negative_source":
        hkeys[0, 3] = -1
    elif failure == "float_keys":
        hkeys = hkeys.astype(float)
    elif failure == "raw_mismatch":
        raw[0] = 0x1421
    elif failure == "geometry_length":
        geom["wcalx"] = geom["wcalx"][:, :1]
    elif failure == "extra_geometry":
        geom["wcalx"] = np.repeat(geom["wcalx"], 2, axis=0)
    elif failure == "negative_energy":
        events = change(events, "s_edep", [0., -.0001])
    elif failure == "length_mismatch":
        events = change(events, "s_edep", [3.], row=3)
    else:
        geom["whvy"][0, 0] = np.inf
    with pytest.raises(ValueError, match=message):
        P.endpoint_measurements(events, geom, hkeys, raw)


@pytest.mark.parametrize("field", ["s_edep", "s_time", "s_et0", "s_et1"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_measured_rejected_before_speed_computation(measured, field, value, monkeypatch):
    events, geom, hkeys, raw, cond, config = measured
    events = change(events, field, [0., value])
    monkeypatch.setattr(P, "hit_inputs", lambda *a: pytest.fail("must validate energy/times first"))
    with pytest.raises(ValueError, match="nonfinite measured"):
        P.measured_sensors(events, geom, hkeys, raw, cond, config)


@pytest.fixture
def sample():
    qseed = np.array([[45., -1.1, 280., 1.4, 73000., 2000.], [57., -1., 200., 1.1, 73005., 2030.]])
    # Changed reference deliberately distinguishes seed and reference solves in this unit fixture.
    qref = qseed+[[2., .02, -10., -.01, .5, 5.], [-3., -.03, 15., .02, -.4, -7.]]
    mass = np.array([105.658367, 200.])
    mid = np.array([[250., 180., 390.], [290., 170., 480.]])
    wire = np.array([[.8, .6, 0.], [-.8, -.6, 0.]])
    line = P.A.sensor(mid+500*wire, mid-500*wire, np.array([73003., 73010.]), np.array([180., 220.]))
    corrected = np.array([73000., 73006.])
    built = P.C.solve(qseed, line, *P.C.constructor_hint(qseed, line, corrected, mass),
                      precision=P.NATIVE_CA_PRECISION, mass=mass)
    state = P.C.solve(qref, line, built["particle_time"], built["sensor_time"], precision=P.NATIVE_CA_PRECISION, mass=mass)
    generated = P.fixed_state_rows(qref, line, state, mass)
    H = generated["H"]
    r = np.column_stack((-generated["doca"], [2.5, -3.], [-7., 4.]))
    mvar, scale = np.array([[2., 4., 8.], [3., 5., 9.]]), np.array([2., 3.])
    weight = 1/(mvar*scale[:, None])
    W, b = oracle_information(qref, H, r, weight)
    records, native = [], []
    for i in range(2):
        identity = dict(run=8, subrun=i, event=42, iseed=3, iteration_ordinal=0,
                        effect_index=4, shindex=[21, 7][i], strawid=100+i, input_index=10+i,
                        miter=2, iter=0, active=True, in_fit=True)
        g = H[i]*np.array([1., -1., -1.])[:, None]
        records.append(dict(identity, ca_usable=True, constrain_tot=True, constrain_long=True,
                            constrain_absdrift=False, hit_state=0, row_active=[1, 1, 1],
                            precision=P.NATIVE_CA_PRECISION, row_derivative=g.ravel().tolist(),
                            row_value=(r[i]*[-1., 1., 1.]).tolist(), row_mvar=mvar[i].tolist(),
                            variance_scale=scale[i], row_pvar=[1e30]*3, dDdP=g[0].tolist(), dTdP=g[1].tolist(),
                            doca=generated["doca"][i], lsign=generated["lsign"][i],
                            particle_poca=np.r_[state["particle_poca"][i], state["particle_time"][i]].tolist(),
                            sensor_poca=np.r_[state["sensor_poca"][i], state["sensor_time"][i]].tolist(),
                            particle_direction=state["direction"][i].tolist(), sensor_direction=state["sensor_direction"][i].tolist()))
        native.append(dict(identity, time=state["particle_time"][i], reference_parameters=qref[i].tolist(),
                           weight_matrix=W[i].ravel().tolist(), weight_vector=b[i].tolist()))
    sensors = {label: deepcopy(line) for label in P.SPEED_CONTROLS}
    return SimpleNamespace(native=ak.Array(native), rows=ak.Array(records), qseed=qseed, qref=qref, mass=mass,
                           constructor_time=built["particle_time"], sensors=sensors, corrected=corrected,
                           built=built, state=state, H=H, r=r, weight=weight, W=W, b=b)


def compare(s, output=None):
    return P.compare_population(s.native, s.rows, s.qseed, s.qref, s.mass, s.constructor_time,
                                s.sensors, s.corrected, output)


def test_all_lanes_four_solves_mass_precision_and_retained_hints(sample, monkeypatch):
    s = sample
    actual, calls = P.C.solve, []

    def solve(q, line, ph, sh, *, mass, precision):
        constructor = len(calls) % 2 == 0
        assert q is (s.qseed if constructor else s.qref)
        assert mass is s.mass
        assert precision == float(np.float32(1e-4)) != 1e-4
        if not constructor:
            np.testing.assert_array_equal(ph, s.built["particle_time"])
            np.testing.assert_array_equal(sh, s.built["sensor_time"])
        assert len(q) == 2
        calls.append(constructor)
        return actual(q, line, ph, sh, mass=mass, precision=precision)

    monkeypatch.setattr(P.C, "solve", solve)
    monkeypatch.setattr(ak, "to_list", lambda *a, **k: pytest.fail("no per-hit record loop"))
    result = compare(s)
    assert calls == [True, False, True, False]
    assert result["n_hits"] == result["n_targets"] == 2 and result["n_excluded_hits"] == 0
    assert result["hit_keys"] == P.keys(s.native, P.HIT_KEYS).tolist()
    assert tuple(result["variants"]) == P.SPEED_CONTROLS
    for item in result["variants"].values():
        for field in P.STATE_FIELDS:
            assert item["geometry"][field]["exact"]["passed"]
        assert item["H_vs_captured"]["exact"]["passed"]
        assert len(item["row_metrics"]) == 3
        info = item["information_captured_values_weights"]
        assert "NOT independent residual" in info["scope"]
        assert info["weight_matrix"]["criterion"]["passed"]
        assert info["weight_vector"]["criterion"]["passed"]
    json.dumps(result, allow_nan=False)


def test_independent_information_and_native_prediction_signs(sample):
    s = sample
    _, H, r, w = P.join_null_rows(s.rows[::-1], s.native)
    np.testing.assert_array_equal(H, s.H)
    np.testing.assert_array_equal(r, s.r)
    np.testing.assert_array_equal(w, s.weight)
    g = P._dense(s.rows, "row_derivative", (3, 6))
    value = P._dense(s.rows, "row_value", (3,))
    W = np.einsum("nri,nr,nrj->nij", g, w, g)
    b = np.einsum("nij,nj->ni", W, s.qref)+np.einsum("nri,nr,nr->ni", g, w, value)
    actual_W, actual_b = P.A.information(s.qref, H, r, w)
    np.testing.assert_allclose(actual_W, W, rtol=2e-15, atol=1e-10)
    np.testing.assert_allclose(actual_b, b, rtol=2e-14, atol=1e-7)
    assert not np.allclose(b, np.einsum("nij,nj->ni", W, s.qref)-np.einsum("nri,nr,nr->ni", g, w, value))


def test_capture_state_not_used_to_solve_and_failing_full_keys_not_masked(sample):
    s = sample
    changed = np.r_[s.state["sensor_poca"][1]+[.1, -.2, .3], s.state["sensor_time"][1]+.4].tolist()
    s.rows = change(s.rows, "sensor_poca", changed, row=1)
    result = compare(s)
    for item in result["variants"].values():
        np.testing.assert_array_equal(item["state"]["sensor_poca"], s.state["sensor_poca"])
        metric = item["geometry"]["sensor_poca"]["criterion"]
        assert metric["failing_hit_keys"] == P.keys(s.native, P.HIT_KEYS)[1:].tolist()
        assert metric["n_failing_hits"] == 1
        assert metric["n_components"] == 6
        assert not item["captured_complete_state_control"]["H_vs_captured"]["criterion"]["passed"]


@pytest.mark.parametrize("field,value,message", [
    ("hit_state", 1, "null hits"), ("ca_usable", False, "ca_usable"),
    ("row_active", [1, 0, 1], "three residual"), ("constrain_tot", False, "constrain_tot"),
    ("constrain_long", False, "constrain_long"), ("constrain_absdrift", True, "absolute drift"),
    ("precision", 1e-4, "precision"), ("variance_scale", 0., "scale"),
    ("particle_direction", [np.nan, 0., 0.], "nonfinite captured state"),
    ("sensor_poca", [0., 0., 0., np.inf], "nonfinite captured state"),
])
def test_fail_closed_support_before_any_solve(sample, field, value, message, monkeypatch):
    sample.rows = change(sample.rows, field, value)
    monkeypatch.setattr(P.C, "solve", lambda *a, **k: pytest.fail("unsupported capture must fail before solve"))
    with pytest.raises(ValueError, match=message):
        compare(sample)


@pytest.mark.parametrize("missing", [False, True])
def test_full_key_join_duplicate_and_missing_rejected(sample, missing):
    sample.rows = sample.rows[:1] if missing else ak.concatenate([sample.rows, sample.rows[:1]])
    with pytest.raises(ValueError, match="missing full identity|duplicate"):
        compare(sample)


@pytest.mark.parametrize("stage", [0, 1])
def test_unusable_lane_aborts_whole_population_with_identity(sample, stage, monkeypatch):
    original, calls, output = P.C.solve, [], {}

    def failed(*a, **k):
        state = original(*a, **k)
        if len(calls) == stage:
            state["usable"][1] = False
            state["status"][1] = 4
        calls.append(1)
        return state

    monkeypatch.setattr(P.C, "solve", failed)
    with pytest.raises(ValueError, match="unusable.*CA"):
        compare(sample, output)
    item = output["variants"][P.SPEED_CONTROLS[0]]
    prefix = "constructor" if stage == 0 else "reference"
    assert item[prefix+"_unusable_hit_keys"] == P.keys(sample.native, P.HIT_KEYS)[1:].tolist()
    assert output["n_excluded_hits"] == 0 and output["n_hits"] == 2
    assert len(calls) == stage+1


def test_nonfinite_information_and_unknown_control_fail_closed(sample):
    sample.native = change(sample.native, "weight_vector", [np.inf]*6)
    with pytest.raises(ValueError, match="nonfinite comparison"):
        compare(sample)
    sample.sensors["optimized"] = sample.sensors[P.SPEED_CONTROLS[0]]
    with pytest.raises(ValueError, match="two declared"):
        compare(sample)


def test_bounded_component_metrics_full_failing_identities_and_finite_strictness():
    ids = np.column_stack([np.arange(40), np.zeros((40, 5), dtype=int)])
    a, b = np.ones((40, 3, 6)), np.zeros((40, 3, 6))
    result = P.comparison(a, b, ids)
    assert len(result["criterion"]["first_mismatch_indices"]) == 12
    assert result["criterion"]["failing_hit_keys"] == ids.tolist()
    assert result["criterion"]["per_hit_failed_components"] == [18]*40
    assert P.comparison(np.nextafter(a, np.inf), a, ids)["criterion"]["passed"]
    assert not P.comparison(np.nextafter(a, np.inf), a, ids)["exact"]["passed"]
    with pytest.raises(ValueError, match="nonfinite"):
        P.comparison(a*np.inf, b, ids)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="nonfinite comparison arithmetic"):
        P.comparison(np.full_like(a, 1e308), np.full_like(a, -1e308), ids)


def test_frozen_population_guard_and_no_count_overrides():
    targets = np.column_stack([np.ones((147, 3), dtype=int), np.arange(147)])
    iterations = ak.Array({name: targets[:, i] for i, name in enumerate(P.KEYS)})
    repeated = targets[np.arange(2538) % 147]
    hkeys = np.column_stack([repeated, np.arange(2538)])
    np.testing.assert_array_equal(P.population_guard(iterations, hkeys), targets)
    with pytest.raises(ValueError, match="ALL147 / 2538"):
        P.population_guard(iterations, hkeys[:-1])
    with pytest.raises(ValueError, match="coverage differs"):
        P.population_guard(change(iterations, "run", 2), hkeys)


def args(tmp_path):
    for name in ("check.json", "operand.json", "conditions.npz"):
        (tmp_path/name).write_text("{}")
    return SimpleNamespace(check=tmp_path/"check.json", operand_check=tmp_path/"operand.json",
                           conditions=tmp_path/"conditions.npz", native_source=tmp_path/"native", out=tmp_path/"out.json")


def test_policy_guard_exclusive_failure_json_and_source_hashes(tmp_path, monkeypatch):
    a = args(tmp_path)
    monkeypatch.setenv("F1_SYNTHETIC_NONDEFAULT", "1")
    report = P.run(a)
    assert not report["completed"] and "default F1 environment" in report["error"]
    assert report["hashes_unchanged"]
    assert report["before_sha256"] == report["end_sha256"]
    required = {"f1_captured_ca_population.py", "f1_captured_ca_audit.py", "f1_ca_population_audit.py", "nativearith.py", "nativeca.py"}
    assert required <= {Path(p).name for p in report["source_sha256"]}
    assert json.loads(a.out.read_text()) == report
    before = a.out.read_bytes()
    with pytest.raises(FileExistsError):
        P.run(a)
    assert a.out.read_bytes() == before


def test_gate_failure_rehashes_known_evidence_and_detects_mutation(tmp_path, monkeypatch):
    a = args(tmp_path)
    evidence = tmp_path/"evidence.root"
    evidence.write_text("original")
    original_hash = P.digest(evidence)

    def fail(check, metadata):
        metadata.update(check=str(check), check_sha256=P.digest(check),
                        inputs={"capture": dict(path=str(evidence), sha256=original_hash)})
        evidence.write_text("changed during gate")
        raise ValueError("synthetic gate failure")

    monkeypatch.setattr(P, "load_evidence", fail)
    report = P.run(a)
    assert not report["completed"] and not report["hashes_unchanged"]
    assert "synthetic gate failure" in report["error"]
    assert "changed during audit" in report["integrity_error"]
    assert report["before_sha256"][str(evidence)] == original_hash
    assert report["end_sha256"][str(evidence)] == P.digest(evidence)
    assert json.loads(a.out.read_text()) == report


@pytest.mark.parametrize("mutate_conditions", [False, True])
def test_driver_wires_frozen_selector_receipt_endpoints_and_both_controls(tmp_path, monkeypatch, sample, mutate_conditions):
    """Synthetic I/O harness; the real selector/numerical kernels have separate tests."""
    a = args(tmp_path)
    for field in ("old.root", "new.root", "baseline.root", "control.root", "targets.json", "local.py", "native.hh"):
        (tmp_path/field).write_text("synthetic")
    expanded = tmp_path/"capture.expanded.fcl"
    expanded.write_text("""TPOCAPrecision : 1e-4
strawResponse : {
useDb : false
strawHalfPropVelocity : []
eBinWidth : 0.1
defaultHalfPropVelocity : 87
eBins : 3
eHalfPVScale : [1, 1.4, 1.9]
}
""")
    sidecheck = tmp_path/"side"/"check.json"
    sidecheck.parent.mkdir()
    sidecheck.write_text("{}")
    a.operand_check = sidecheck.parent/"operands.json"
    a.operand_check.write_text(json.dumps(dict(end_sha256={str(sidecheck): P.digest(sidecheck)})))
    iterations, pieces, effects, seeds = (object() for _ in range(4))
    calls = []

    def metadata(check, current):
        capture = "new.root" if current else "old.root"
        paths = dict(capture=capture, control="control.root", baseline="baseline.root", targets="targets.json")
        return dict(check=str(check), check_sha256=P.digest(check),
                    inputs={k: dict(path=str(tmp_path/p), sha256=P.digest(tmp_path/p)) for k, p in paths.items()},
                    highest_cycles={k: {} for k in ("capture", "control", "baseline")})

    def load(check, meta):
        current = check == sidecheck
        calls.append("new_gate" if current else "frozen_gate")
        meta.update(metadata(check, current))
        return iterations, pieces

    def receipt(path, check, frozen):
        calls.append("receipt")
        assert path == a.operand_check and check == sidecheck and Path(frozen) == tmp_path/"old.root"
        return dict(check=str(sidecheck), n_hits=2), {str(a.operand_check): P.digest(a.operand_check)}

    def select(it, pc, eff, sd):
        calls.append("select_initial")
        assert it is iterations and pc is pieces and eff is effects and sd is seeds
        return (sample.native, sample.qseed, sample.qref, sample.mass, sample.constructor_time,
                P.keys(sample.native, P.KEYS+("shindex",)))

    class File:
        def __init__(self, path):
            self.path = Path(path)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    def read(file, directory, name, cycles, fields=None):
        if name in ("effects", "seed_contract"):
            assert file.path == tmp_path/"old.root"
            return effects if name == "effects" else seeds
        if name == "hit_operands":
            assert file.path == tmp_path/"new.root"
            return sample.rows[::-1]
        assert file.path == tmp_path/"baseline.root" and directory == "trkdump"
        if name == "events":
            assert fields == [*P.KEYS[:3], "s_sid", "s_time", "s_et0", "s_et1", "s_edep"]
            return "synthetic measured events"
        return ak.Array({field: [[100., 101.]] for field in fields})

    def sensors(events, geometry, hkeys, raw, conditions, config):
        calls.append("measured_sensors")
        assert events == "synthetic measured events" and conditions == "synthetic conditions"
        np.testing.assert_array_equal(hkeys, P.keys(sample.native, P.KEYS+("shindex",))[:, [0, 1, 2, 4]])
        np.testing.assert_array_equal(raw, [100, 101])
        assert config["path"] == str(expanded) and config["scale"] == [1., 1.4, 1.9]
        if mutate_conditions:
            a.conditions.write_text("changed during numerical audit")
        return dict(corrected_time=sample.corrected, energy=np.array([.0001, .0002])), sample.sensors

    monkeypatch.setattr(P, "FROZEN_EXPANDED_SHA256", P.digest(expanded))
    monkeypatch.setattr(P, "local_sources", lambda: [tmp_path/"local.py"])
    monkeypatch.setattr(P, "native_sources", lambda _: [tmp_path/"native.hh"])
    monkeypatch.setattr(P, "root_arithmetic_sources", lambda _: {})
    monkeypatch.setattr(P, "load_evidence", load)
    monkeypatch.setattr(P, "receipt", receipt)
    monkeypatch.setattr(P, "select_initial", select)
    monkeypatch.setattr(P, "population_guard", lambda it, hk: np.unique(hk[:, :4], axis=0))
    monkeypatch.setattr(P.uproot, "open", File)
    monkeypatch.setattr(P, "read_tree", read)
    monkeypatch.setattr(P.hitreco, "load_conditions", lambda _: "synthetic conditions")
    monkeypatch.setattr(P, "measured_sensors", sensors)
    result = P.run(a)
    assert calls == ["frozen_gate", "receipt", "new_gate", "select_initial", "measured_sensors"]
    assert result["completed"] is (not mutate_conditions)
    assert result["hashes_unchanged"] is (not mutate_conditions)
    assert set(result["population"]["variants"]) == set(P.SPEED_CONTROLS)
    assert result["full_closure"] is False and result["independent_residual_variance_closure"] is False
    assert str(expanded) in result["before_sha256"]
    assert str(sidecheck) in result["before_sha256"]
    assert str(tmp_path/"old.root") in result["before_sha256"]
    assert str(tmp_path/"new.root") in result["before_sha256"]
    assert json.loads(a.out.read_text()) == result


def test_exact_selector_is_reused_not_reimplemented():
    import f1_ca_population_audit
    assert P.select_initial is f1_ca_population_audit.select_initial


def test_missing_measured_values_fail_closed(measured):
    events, geom, hkeys, raw, *_ = measured
    events = change(events, "s_edep", [0., None])
    with pytest.raises(ValueError, match="missing values"):
        P.endpoint_measurements(events, geom, hkeys, raw)


def test_receipt_failure_still_rehashes_declared_sidecar_inputs(tmp_path, monkeypatch):
    a = args(tmp_path)
    evidence = tmp_path/"new-sidecar.root"
    evidence.write_text("before")
    before = P.digest(evidence)
    a.operand_check.write_text(json.dumps(dict(end_sha256={str(evidence): before})))

    def frozen(check, metadata):
        metadata.update(inputs={"capture": dict(path=str(evidence), sha256=before)})
        return None, None

    def reject(*args):
        evidence.unlink()
        raise ValueError("synthetic receipt rejected")

    monkeypatch.setattr(P, "load_evidence", frozen)
    monkeypatch.setattr(P, "receipt", reject)
    report = P.run(a)
    assert not report["completed"] and not report["hashes_unchanged"]
    assert "receipt rejected" in report["error"]
    assert report["before_sha256"][str(evidence)] == before
    assert report["end_sha256"][str(evidence)] is None
    assert json.loads(a.out.read_text()) == report


def test_invalid_declared_hash_emits_strict_failure_json(tmp_path, monkeypatch):
    a = args(tmp_path)
    a.operand_check.write_text('{"end_sha256": {"/synthetic": NaN}}')
    monkeypatch.setattr(P, "load_evidence", lambda check, meta: (None, None))
    result = P.run(a)
    assert not result["completed"] and "invalid input/source SHA256" in result["error"]
    assert json.loads(a.out.read_text()) == result
    assert "NaN" not in a.out.read_text() and "Infinity" not in a.out.read_text()


def test_snapshot_row_derivatives_against_ordinary_contraction_oracle(sample):
    s = sample
    line, state = s.sensors[P.SPEED_CONTROLS[0]], s.state
    _, _, speed, dx, dm = P.A.kinematics(s.qref, state["particle_time"], s.mass)
    u, wire, delta = state["direction"], state["sensor_direction"], state["delta"]
    dot = np.sum(wire*u, axis=1)
    den = 1-dot**2
    sign = np.copysign(1., np.sum(np.cross(wire, u)*delta, axis=1))
    unit = delta/np.linalg.norm(delta, axis=1)[:, None]
    contract = lambda v, matrix: np.einsum("ni,nij->nj", v, matrix)
    beta = (wire*dot[:, None]-u)/(speed*den)[:, None]
    gamma = (wire-dot[:, None]*u)/(line["speed"]*den)[:, None]
    ds, dp, ddm = contract(wire, dx), contract(u, dx), contract(delta, dm)
    oracle = np.stack((contract(sign[:, None]*unit, dx), -contract(beta-gamma, dx),
                       ds+((dot[:, None]*ds-dp-ddm)/den[:, None])*dot[:, None]), axis=1)
    np.testing.assert_allclose(s.H, oracle, rtol=2e-14, atol=2e-11)
    np.testing.assert_array_equal(P.column(s.rows, "lsign"), sign)
