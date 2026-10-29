"""Synthetic initial residual contracts; no native capture, fit, build or remote jobs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_initial_residual_audit as D


CAD = [400, 800000, 50, 50, "TOT:LongVal", "Null:Inactive", "", 0]
DRIFT = "double DriftInfo::maxdvar_(2.4*2.4/3.0);\n"


def fit_settings(cad=None, schedule=None, extra=""):
    return ("  FitSettings : {\n"
            f"    CADSHUSettings : [{json.dumps(CAD if cad is None else cad)}]\n"
            f"    MetaIterationSettings : [{json.dumps([5, 'CADSHU'] if schedule is None else schedule)}]\n"
            f"{extra}  }}\n")


@pytest.fixture
def policy_files(tmp_path):
    fcl, drift = tmp_path/"expanded.fcl", tmp_path/"DriftInfo.cc"
    fcl.write_text(fit_settings()*2)
    drift.write_text(DRIFT)
    return fcl, drift


@pytest.fixture
def policy(policy_files):
    return D.initial_policy(*policy_files)


def test_initial_policy_identical_blocks_first_schedule_and_explicit_scope(policy_files):
    fcl, drift = policy_files
    result = D.initial_policy(fcl, drift)
    assert result["cad"] == CAD
    assert result["temperature"] == 5.
    assert result["variance_scale"] == 36.
    assert result["distance_variance"] == 1.92
    assert result["hit_state"] == 0
    assert result["constrain_tot"] is result["constrain_long"] is True
    assert result["constrain_absdrift"] is result["null_drift_variance"] is False
    assert result["state_eligibility_reconstructed"] is False
    assert "unbiased CAD eligibility remains untested" in result["scope"]
    assert any("not reconstructed" in item for item in D.LIMITS)
    fcl.write_text(fit_settings().replace('"", 0]]', '"", 0], [1, 2, "later"]]'))
    assert D.initial_policy(fcl, drift) == result


@pytest.mark.parametrize("text,message", [
    ("ExtensionSettings : {}\n", "ambiguous initial FitSettings"),
    (fit_settings()+fit_settings(schedule=[4, "CADSHU"]), "ambiguous initial FitSettings"),
    (fit_settings().replace("CADSHUSettings", "OtherSettings"), "missing initial updater/schedule"),
    (fit_settings().replace("MetaIterationSettings", "OtherSettings"), "missing initial updater/schedule"),
    (fit_settings(extra='    CADSHUSettings : [[400]]\n'), "missing initial updater/schedule"),
    (fit_settings(schedule=[4, "CADSHU"]), "unsupported initial schedule"),
    (fit_settings(schedule=[5, "Chi2SHU"]), "unsupported initial schedule"),
    (fit_settings(schedule=[5, "CADSHU", 0]), "unsupported initial schedule"),
])
def test_initial_policy_rejects_ambiguous_missing_or_wrong_schedule(policy_files, text, message):
    fcl, drift = policy_files
    fcl.write_text(text)
    with pytest.raises(ValueError, match=message):
        D.initial_policy(fcl, drift)


@pytest.mark.parametrize("index,value", [(0, 401), (1, 800001), (2, 49), (3, 51),
                                          (4, "TOT"), (5, "Null"), (6, "extra"), (7, 1)])
def test_initial_policy_requires_exact_first_cad(policy_files, index, value):
    fcl, drift = policy_files
    cad = CAD.copy()
    cad[index] = value
    fcl.write_text(fit_settings(cad=cad))
    with pytest.raises(ValueError, match="unsupported initial CADSHU policy"):
        D.initial_policy(fcl, drift)


@pytest.mark.parametrize("updater", ["BkgANNSHU", "DriftANNSHU", "PanelDiagSHU", "Chi2SHU"])
def test_initial_policy_rejects_extra_updater(policy_files, updater):
    fcl, drift = policy_files
    fcl.write_text(fit_settings(extra=f"    {updater}Settings : []\n"))
    with pytest.raises(ValueError, match="additional initial state updater"):
        D.initial_policy(fcl, drift)


@pytest.mark.parametrize("source", ["", DRIFT.replace("2.4*2.4", "2.5*2.5"),
                                    DRIFT.replace("maxdvar_", "other_"),
                                    DRIFT.replace("2.4*2.4/3.0", "1.92")])
def test_initial_policy_guards_source_constant(policy_files, source):
    fcl, drift = policy_files
    drift.write_text(source)
    with pytest.raises(ValueError, match="null distance variance source changed"):
        D.initial_policy(fcl, drift)


@pytest.fixture
def measured_tables():
    fields = dict(sid=[[0x1421, 0x18A2], [0x18A2], [0x1421], []],
                  time=[[20., 22.], [81.], [90.], []],
                  et0=[[24., 25.], [84.], [95.], []],
                  et1=[[27., 25.], [82.], [94.], []],
                  edep=[[0., .0002], [.00005], [.0003], []],
                  tot0=[[30., 31.], [32.], [33.], []],
                  tot1=[[40., 41.], [42.], [43.], []],
                  dtime=[[1., 2.], [3.], [4.], []],
                  wdist=[[10., -20.], [30.], [-40.], []],
                  timevar=[[2., 3.], [4.], [5.], []],
                  uvar=[[6., 7.], [8.], [9.], []],
                  eend=[[0, 0], [1], [1], []])
    data = dict(run=[7, 8, 7, 7], subrun=[2, 2, 3, 1], event=[9, 9, 9, 9])
    for name, values in fields.items():
        data["s_"+name] = ak.values_astype(ak.Array(values), np.int64 if name in ("sid", "eend") else np.float32)
    direction = np.array([[.6, .8, 0.], [0., 0., 1.]])
    center = np.array([[10., 20., 30.], [40., 50., 60.]])
    hlen = np.array([250., 300.])
    geometry = dict(sid=np.array([[0x18A2, 0x1421]]), hlen=hlen[None])
    for prefix, xyz in (("wcal", center+hlen[:, None]*direction),
                        ("whv", center-hlen[:, None]*direction), ("dir", direction)):
        for i, axis in enumerate("xyz"):
            geometry[prefix+axis] = xyz[None, :, i].copy()
    hkeys = np.array([[7, 2, 9, 1], [8, 2, 9, 0], [7, 3, 9, 0], [7, 2, 9, 0]])
    raw = np.array([0x18A2, 0x18A2, 0x1421, 0x1421])
    return ak.Array(data), geometry, hkeys, raw


def replace_field(events, name, values):
    dtype = ak.to_numpy(ak.flatten(events["s_"+name])).dtype
    return ak.with_field(events, ak.values_astype(ak.Array(values), dtype), "s_"+name)


def test_measured_inputs_full_event_source_join_float32_direction_and_tie(measured_tables, monkeypatch):
    events, geometry, hkeys, raw = measured_tables
    monkeypatch.setattr(ak, "to_list", lambda *a, **k: pytest.fail("no per-event records in hot path"))
    actual, geom = D.measured_inputs(*measured_tables)
    for field in D.EVENT_FIELDS:
        expected = ak.to_numpy(ak.flatten(events["s_"+field]))[[1, 2, 3, 0]].astype(float)
        np.testing.assert_array_equal(actual[field], expected)
    np.testing.assert_array_equal(actual["measurement_time"], [25., 82., 94., 24.])
    np.testing.assert_array_equal(actual["corrected_time"], [22., 81., 90., 20.])
    np.testing.assert_array_equal(actual["early"], [[160., 220., 30.], [-140., -180., 30.],
                                                  [40., 50., -240.], [40., 50., 360.]])
    direction = np.array([[.6, .8, 0.], [.6, .8, 0.], [0., 0., 1.], [0., 0., 1.]])
    np.testing.assert_array_equal(geom["direction"], direction)
    np.testing.assert_array_equal(geom["hlen"], [250., 250., 300., 300.])
    np.testing.assert_array_equal(actual["udir"], direction.astype(np.float32).astype(float))
    assert actual["udir"][0, 0] != geom["direction"][0, 0]


def test_measured_inputs_repeated_wanted_valid(measured_tables):
    events, geometry, hkeys, raw = measured_tables
    expected, expected_geom = D.measured_inputs(*measured_tables)
    order = np.array([0, 2, 0, 1, 3])
    actual, geom = D.measured_inputs(events, geometry, hkeys[order], raw[order])
    for field in expected:
        np.testing.assert_array_equal(actual[field], expected[field][order])
    for field in expected_geom:
        np.testing.assert_array_equal(geom[field], expected_geom[field][order])


@pytest.mark.parametrize("available", ["event", "geometry"])
def test_measured_inputs_duplicate_available_rejected(measured_tables, available):
    events, geometry, hkeys, raw = measured_tables
    if available == "event":
        events = ak.concatenate([events, events[-1:]])
    else:
        geometry = {name: np.concatenate([value, value[:, :1]], axis=1) for name, value in geometry.items()}
    with pytest.raises(ValueError, match="duplicate"):
        D.measured_inputs(events, geometry, hkeys, raw)


@pytest.mark.parametrize("field", [f for f in D.EVENT_FIELDS if f not in ("sid", "eend")])
def test_measured_inputs_float32_required(measured_tables, field):
    events, geometry, hkeys, raw = measured_tables
    events = ak.with_field(events, ak.values_astype(events["s_"+field], np.float64), "s_"+field)
    with pytest.raises(ValueError, match="ComboHit field is not float32: "+field):
        D.measured_inputs(events, geometry, hkeys, raw)


@pytest.mark.parametrize("field", D.EVENT_FIELDS)
def test_measured_inputs_masked_lane_rejected(measured_tables, field):
    events, geometry, hkeys, raw = measured_tables
    masked = ak.mask(events["s_"+field], ak.Array([[True, False], [True], [True], []]))
    events = ak.with_field(events, masked, "s_"+field)
    with pytest.raises(ValueError, match="missing|mask|invalid|nonfinite"):
        D.measured_inputs(events, geometry, hkeys, raw)


@pytest.mark.parametrize("field,value,message", [
    ("dtime", np.nan, "nonfinite measured"), ("tot0", np.inf, "nonfinite measured"),
    ("timevar", 0., "nonpositive measured time variance"),
    ("timevar", -1., "nonpositive measured time variance"),
    ("uvar", 0., "nonpositive measured longitudinal variance"),
    ("uvar", -1., "nonpositive measured longitudinal variance"),
    ("eend", 2, "invalid measured early end"),
])
def test_measured_inputs_value_guards(measured_tables, field, value, message):
    events, geometry, hkeys, raw = measured_tables
    values = ak.to_list(events["s_"+field])
    values[0][1] = value
    events = replace_field(events, field, values)
    with pytest.raises(ValueError, match=message):
        D.measured_inputs(events, geometry, hkeys, raw)


def test_measured_inputs_lengths_checked(measured_tables):
    events, geometry, hkeys, raw = measured_tables
    events = replace_field(events, "tot1", [[40.], [42.], [43.], []])
    with pytest.raises(ValueError, match="measured lengths differ: tot1"):
        D.measured_inputs(events, geometry, hkeys, raw)


@pytest.mark.parametrize("field,value,message", [
    ("dirx", np.nan, "nonfinite straw direction"), ("dirx", 2., "nonunit straw direction"),
    ("hlen", 0., "invalid straw half length"), ("hlen", -1., "invalid straw half length"),
    ("hlen", np.inf, "invalid straw half length"),
])
def test_measured_inputs_geometry_guards(measured_tables, field, value, message):
    events, geometry, hkeys, raw = measured_tables
    geometry[field][0, 0] = value
    with pytest.raises(ValueError, match=message):
        D.measured_inputs(events, geometry, hkeys, raw)


@pytest.fixture
def operands(policy):
    generated = dict(doca=np.array([2., -3.]), longitudinal=np.array([10., -20.]))
    state = dict(sensor_direction=np.array([[.6, .8, 0.], [-.6, -.8, 0.]]),
                 sensor_time=np.array([15., 9.]), particle_time=np.array([10., 12.]))
    measured = dict(dtime=np.array([2., 4.]), wdist=np.array([3., 5.]),
                    timevar=np.array([2., 4.]), uvar=np.array([8., 16.]),
                    udir=np.array([[.6, .8, 0.], [.6, .8, 0.]]))
    return generated, state, measured, policy


def test_residual_operands_both_orientations_native_f1_sign_and_time_delta(operands):
    generated, state, measured, policy = operands
    saved = deepcopy(operands)
    actual = D.residual_operands(*operands)
    np.testing.assert_array_equal(actual["orientation_dot"], [1., -1.])
    np.testing.assert_array_equal(actual["value"], [[2., 3., 7.], [-3., -7., -15.]])
    np.testing.assert_array_equal(actual["residual"], [[-2., 3., 7.], [3., -7., -15.]])
    np.testing.assert_array_equal(actual["mvar"], [[1.92, 2., 8.], [1.92, 4., 16.]])
    np.testing.assert_array_equal(actual["scale"], [36., 36.])
    np.testing.assert_array_equal(actual["weight"], 1./(np.array([[1.92, 2., 8.], [1.92, 4., 16.]])*36.))
    for before, after in zip(saved[:3], operands[:3]):
        for field in before:
            np.testing.assert_array_equal(before[field], after[field])
    assert policy == saved[3]
    assert policy["state_eligibility_reconstructed"] is False


@pytest.mark.parametrize("field,value", [("hit_state", 1), ("constrain_tot", False),
                                        ("constrain_long", False), ("constrain_absdrift", True),
                                        ("null_drift_variance", True)])
def test_residual_operands_unsupported_policy(operands, field, value):
    operands[3][field] = value
    with pytest.raises(ValueError, match="unsupported residual policy"):
        D.residual_operands(*operands)


@pytest.mark.parametrize("field", ["dtime", "wdist", "timevar", "uvar", "udir"])
@pytest.mark.parametrize("failure", ["shape", "nonfinite"])
def test_residual_operands_invalid_measurements(operands, field, failure):
    measured = operands[2]
    if failure == "shape":
        measured[field] = measured[field][:1]
    else:
        measured[field][0] = np.nan
    with pytest.raises(ValueError, match="invalid measur"):
        D.residual_operands(*operands)


@pytest.mark.parametrize("dot", [0., .99, -.99])
def test_residual_operands_ambiguous_orientation_rejected(operands, dot):
    operands[1]["sensor_direction"][0] = [dot, 0., 0.]
    operands[2]["udir"][0] = [1., 0., 0.]
    with pytest.raises(ValueError, match="orientation ambiguous"):
        D.residual_operands(*operands)


@pytest.mark.parametrize("field", ["timevar", "uvar", "distance_variance", "variance_scale"])
@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_residual_operands_reject_invalid_variances_and_scale(operands, field, value):
    if field in operands[2]:
        operands[2][field][0] = value
    else:
        operands[3][field] = value
    with pytest.raises(ValueError, match="invalid measur|invalid generated"):
        D.residual_operands(*operands)


@pytest.mark.parametrize("stage", ["time", "longitudinal", "variance_product", "reciprocal"])
def test_residual_operands_overflow_fails_without_excluding_lane(operands, stage):
    generated, state, measured, policy = operands
    huge = np.finfo(float).max
    if stage == "time":
        state["sensor_time"][1], state["particle_time"][1] = huge, -huge
    elif stage == "longitudinal":
        generated["longitudinal"][1] = measured["wdist"][1] = huge
    elif stage == "variance_product":
        measured["timevar"][1] = huge
    else:
        policy["distance_variance"] = np.nextafter(0., 1.)
    with pytest.raises(ValueError, match="nonfinite arithmetic"):
        D.residual_operands(*operands)


@pytest.fixture
def ca_receipt(tmp_path, monkeypatch):
    evidence = tmp_path/"synthetic_input.root"
    evidence.write_text("synthetic input only\n")
    hashes = {str(evidence): D.digest(evidence)}
    data = dict(completed=True, hashes_unchanged=True, before_sha256=hashes, end_sha256=hashes.copy(),
                population=dict(n_hits=2538, n_targets=147, n_excluded_hits=0), rtol=1e-5, atol=1e-7)
    path = tmp_path/"ca_report.json"

    def save():
        path.write_text(json.dumps(data))
        monkeypatch.setattr(D, "CA_REPORT_SHA256", D.digest(path))
        return path

    save()
    return path, evidence, data, save


def test_verified_ca_receipt_validates_and_tracks_report(ca_receipt):
    path, evidence, data, _ = ca_receipt
    actual, watched = D.verified_ca_receipt(path)
    assert actual == data
    assert watched == {str(evidence): D.digest(evidence), str(path.resolve()): D.digest(path)}
    watched["new"] = "test copy"
    assert "new" not in actual["before_sha256"]


def test_verified_ca_receipt_pinned_report_hash_first(ca_receipt):
    path, _, _, _ = ca_receipt
    path.write_text("not even JSON")
    with pytest.raises(ValueError, match="captured CA report hash differs"):
        D.verified_ca_receipt(path)


@pytest.mark.parametrize("field,value,message", [
    ("completed", False, "incomplete CA receipt"), ("completed", 1, "incomplete CA receipt"),
    ("hashes_unchanged", False, "incomplete CA receipt"),
    ("before_sha256", {}, "missing/unequal CA hash manifest"),
    ("before_sha256", [], "missing/unequal CA hash manifest"),
    ("end_sha256", {}, "missing/unequal CA hash manifest"),
    ("population", dict(n_hits=2537, n_targets=147, n_excluded_hits=0), "CA receipt census differs"),
    ("population", dict(n_hits=2538, n_targets=146, n_excluded_hits=0), "CA receipt census differs"),
    ("population", dict(n_hits=2538, n_targets=147, n_excluded_hits=1), "CA receipt census differs"),
    ("rtol", 1e-4, "CA receipt tolerances differ"), ("atol", 1e-6, "CA receipt tolerances differ"),
])
def test_verified_ca_receipt_rejects_bad_contract(ca_receipt, field, value, message):
    path, _, data, save = ca_receipt
    data[field] = value
    save()
    with pytest.raises(ValueError, match=message):
        D.verified_ca_receipt(path)


@pytest.mark.parametrize("sha", ["g"*64, "A"*64, "a"*63, 123, None])
def test_verified_ca_receipt_rejects_invalid_digest(ca_receipt, sha):
    path, evidence, data, save = ca_receipt
    data["before_sha256"] = data["end_sha256"] = {str(evidence): sha}
    save()
    with pytest.raises(ValueError, match="invalid receipt SHA256"):
        D.verified_ca_receipt(path)


def test_verified_ca_receipt_rehashes_each_input(ca_receipt):
    path, evidence, _, _ = ca_receipt
    evidence.write_text("changed input\n")
    with pytest.raises(ValueError, match="prior CA input/source changed"):
        D.verified_ca_receipt(path)


def run_args(tmp_path):
    return SimpleNamespace(ca_report=tmp_path/"ca.json", native_source=tmp_path/"native", out=tmp_path/"out.json")


def test_run_nondefault_environment_failure_json_exclusive_and_scope(tmp_path, monkeypatch):
    args = run_args(tmp_path)
    monkeypatch.setenv("F1_SYNTHETIC_NONDEFAULT", "1")
    monkeypatch.setattr(D, "verified_ca_receipt", lambda _: pytest.fail("environment must be checked first"))
    report = D.run(args)
    assert not report["completed"]
    assert not report["hashes_unchanged"]
    assert "requires default F1 environment" in report["error"]
    assert "missing or changed" in report["integrity_error"]
    assert report["full_closure"] is report["f1_generated_operands"] is report["independent_state_selection"] is False
    assert report["conditional_null_residual_variance_audit"] is True
    assert report["rtol"] == 1e-5 and report["atol"] == 1e-7
    assert json.loads(args.out.read_text()) == report
    before = args.out.read_bytes()
    with pytest.raises(FileExistsError):
        D.run(args)
    assert args.out.read_bytes() == before


@pytest.mark.parametrize("mutation", ["unchanged", "changed", "deleted"])
def test_run_failure_rehashes_every_known_input(tmp_path, monkeypatch, mutation):
    args = run_args(tmp_path)
    evidence = tmp_path/"evidence.root"
    evidence.write_text("original")
    original = D.digest(evidence)
    for name in list(D.os.environ):
        if name.startswith("F1_"):
            monkeypatch.delenv(name)
    def synthetic_receipt(path, watched):
        watched[str(evidence)] = original
        return {}, watched

    monkeypatch.setattr(D, "verified_ca_receipt", synthetic_receipt)
    actual_digest = D.digest

    def fail_at_source(path):
        if Path(path).name == "f1_initial_residual_audit.py":
            if mutation == "changed":
                evidence.write_text("changed")
            elif mutation == "deleted":
                evidence.unlink()
            raise ValueError("synthetic source failure")
        return actual_digest(path)

    monkeypatch.setattr(D, "digest", fail_at_source)
    report = D.run(args)
    assert not report["completed"]
    assert "synthetic source failure" in report["error"]
    assert report["before_sha256"] == {str(evidence): original}
    assert report["end_sha256"] == {str(evidence): None if mutation == "deleted" else actual_digest(evidence)}
    assert report["hashes_unchanged"] is (mutation == "unchanged")
    assert ("integrity_error" in report) is (mutation != "unchanged")
    assert json.loads(args.out.read_text()) == report


@pytest.mark.parametrize("field", D.GEOMETRY_FIELDS)
def test_geometry_mask_rejected_before_join(measured_tables, field):
    events, geometry, hkeys, raw = measured_tables
    geometry[field] = np.ma.array(geometry[field], mask=[[True, False]])
    with pytest.raises(ValueError, match="missing"):
        D.measured_inputs(events, geometry, hkeys, raw)


def test_geometry_false_mask_supported(measured_tables):
    events, geometry, hkeys, raw = measured_tables
    expected, _ = D.measured_inputs(*measured_tables)
    actual, _ = D.measured_inputs(events, {k: np.ma.array(v, mask=False) for k, v in geometry.items()}, hkeys, raw)
    for k in expected:
        np.testing.assert_array_equal(actual[k], expected[k])


@pytest.mark.parametrize("mutation", ["changed", "deleted"])
def test_receipt_failure_retains_declared_hashes(ca_receipt, tmp_path, monkeypatch, mutation):
    path, evidence, data, _ = ca_receipt
    original = data["before_sha256"][str(evidence)]
    if mutation == "deleted":
        evidence.unlink()
    else:
        evidence.write_text("changed")
    for name in list(D.os.environ):
        if name.startswith("F1_"):
            monkeypatch.delenv(name)
    args = run_args(tmp_path)
    args.ca_report = path
    report = D.run(args)
    assert not report["completed"] and not report["hashes_unchanged"]
    assert report["before_sha256"][str(evidence)] == original
    assert report["before_sha256"][str(path)] == D.CA_REPORT_SHA256
    assert report["end_sha256"][str(evidence)] == (None if mutation == "deleted" else D.digest(evidence))


def test_executing_module_must_be_receipt_copy(tmp_path):
    original, other = tmp_path/"prior"/"f1_helper.py", tmp_path/"other"/"f1_helper.py"
    original.parent.mkdir()
    other.parent.mkdir()
    original.write_text("pass\n")
    other.write_text("pass\n")
    manifest = {str(original): D.digest(original)}
    with pytest.raises(ValueError, match="executing inherited module differs"):
        D.executing_sources(manifest, {"f1_helper": SimpleNamespace(__file__=str(other))})
    out = D.executing_sources(manifest, {"f1_helper": SimpleNamespace(__file__=str(original))})
    assert out["f1_helper"]["path"] == str(original)


def test_executing_new_core_dependency_added(tmp_path):
    helper = tmp_path/"corehelper.py"
    helper.write_text("pass\n")
    manifest = {}
    out = D.executing_sources(manifest, {"mu2efastsim.helper": SimpleNamespace(__file__=str(helper))})
    assert manifest == {str(helper): D.digest(helper)}
    assert out["mu2efastsim.helper"]["sha256"] == D.digest(helper)
