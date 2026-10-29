"""Independent synthetic native operands; no ROOT/native runtime or fitted outputs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("hit_operand_check", SCRIPTS / "f1_hit_operand_check.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def sample():
    rows, effects, iterations = [], [], []
    for subrun in (0, 1):
        identity = dict(run=8, subrun=subrun, event=42, iseed=3, iteration_ordinal=0,
                        miter=2, iter=0, schema_version=1)
        iterations.append(dict(identity, capture_valid=True, capture_error="", effects_captured=True,
                               neffects=6, decision_valid=bool(subrun)))
        for index, state in enumerate((-3, -2, -1, 0, 1)):
            usable = state > -2
            flags = np.arange(3) < max(0, state + 2)
            g = np.array([[1., -2., 0., .5, 0., 1.], [-1., 0., 3., 0., 1., 0.],
                          [0., 1., 1., -2., 0., 2.]])
            value, mvar, pvar = [2., -3., .75], [2., 4., 8.], [999., 33., 77.]
            q, scale = [1., 2., 3., 4., 5., 6.], 2.
            # Scalar, independently assembled expected information (not checker einsums).
            w = [[sum(g[r, i]*g[r, j]/(mvar[r]*scale) for r in range(3) if flags[r])
                  for j in range(6)] for i in range(6)]
            b = [sum(w[i][j]*q[j] for j in range(6)) +
                 sum(g[r, i]*value[r]/(mvar[r]*scale) for r in range(3) if flags[r]) for i in range(6)]
            common = dict(identity, effect_index=index, input_index=index+10, shindex=index+20,
                          strawid=index+100, active=usable, in_fit=bool(index % 2))
            row = dict(common, variance_scale=scale, ca_status=(6, 3, 0, 1, 2)[index], ca_usable=usable,
                       hit_state=state, precision=9.999999747378752e-5, constrain_tot=True, constrain_long=False,
                       constrain_absdrift=False, particle_poca=[1., 2., 3., 12.] if usable else [],
                       sensor_poca=[4., 5., 6., 13.] if usable else [],
                       particle_direction=[0., 0., 1.] if usable else [],
                       sensor_direction=[0., 1., 0.] if usable else [],
                       dDdP=[1.]*6 if usable else [], dTdP=[2.]*6 if usable else [],
                       doca=2. if usable else np.nan, doca_variance=3. if usable else np.nan,
                       toca_variance=4. if usable else np.nan, lsign=-1. if usable else np.nan,
                       row_active=flags.astype(int).tolist(), row_value=value, row_mvar=mvar, row_pvar=pvar,
                       row_derivative=np.where(flags[:, None], g, np.nan).ravel().tolist())
            effects.append(dict(common, identity_kind=1, kind=1, identity_valid=True, payload_supported=True,
                                has_reference=True, has_weight=True, time=12., reference_parameters=q,
                                weight_matrix=np.array(w).ravel().tolist(), weight_vector=b))
            rows.append(row)
        effects.append(dict(effects[-1], effect_index=5, identity_kind=2, kind=4, input_index=99))
    return ak.Array(rows), ak.Array(effects), ak.Array(iterations)


def change(table, field, value, row=0):
    records = ak.to_list(table)
    records[row][field] = value
    return ak.Array(records)


def test_population_joins_masks_and_information():
    r, e, i = sample()
    with np.errstate(all="raise"):
        report = H.validate(r[::-1], e[::-1], i[::-1])
    assert report["passed"] and report["n_hits"] == report["n_expected_hits"] == 10
    assert report["n_excluded_hits"] == 0 and report["n_unusable_ca"] == 4
    assert report["n_active_residual_rows"] == 12
    assert report["weight_matrix"]["n_components"] == 360
    assert report["weight_vector"]["n_components"] == 60
    assert not report["full_closure"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("table,field,value,row,message", [
    (0, "schema_version", 2, 0, "schema"), (1, "schema_version", 2, 0, "schema"),
    (2, "schema_version", 2, 0, "schema"), (2, "capture_valid", False, 0, "invalid iteration"),
    (2, "capture_error", "failed", 0, "capture error"), (2, "effects_captured", False, 0, "not captured"),
    (2, "neffects", 7, 0, "effect count"), (1, "effect_index", 7, 0, "effect_index"),
    (1, "miter", 3, 0, "miter"), (0, "miter", 3, 0, "miter"),
    (0, "input_index", 0, 0, "input_index"), (0, "shindex", 0, 0, "shindex"),
    (0, "strawid", 0, 0, "strawid"), (0, "iter", 2, 0, "iter"),
    (0, "in_fit", True, 0, "in_fit"), (0, "active", True, 0, "active"),
    (1, "kind", 2, 0, "kind"), (1, "identity_valid", False, 0, "identity_valid"),
    (1, "has_reference", False, 0, "has_reference"), (1, "has_weight", False, 0, "has_weight"),
    (1, "payload_supported", False, 0, "payload_supported"),
    (0, "ca_status", 7, 0, "ca_status"), (0, "ca_status", -1, 0, "ca_status"),
    (0, "ca_status", 0.5, 0, "dtype"), (0, "ca_usable", True, 0, "ca_usable"),
    (0, "ca_usable", 2, 0, "bool/0/1"), (0, "hit_state", -4, 0, "hit_state"),
    (0, "hit_state", 2, 0, "hit_state"), (0, "hit_state", -1, 0, "active/hit_state"),
    (0, "precision", 0., 0, "precision"), (0, "precision", 1e-3, 0, "precision"),
    (0, "precision", np.nan, 0, "precision"), (0, "lsign", 0., 2, "lsign"),
    (0, "doca", 0., 0, "must be NaN"), (0, "doca", np.nan, 2, "nonfinite usable"),
    (0, "particle_poca", [1., 2., 3., 12.1], 2, "exact effect time"),
    (0, "particle_poca", [1., 2., 3., 12.], 0, "shape"),
    (0, "sensor_direction", [0., 1.], 2, "shape"),
    (0, "dDdP", [np.nan]*6, 2, "nonfinite"), (0, "dTdP", [1.]*5, 2, "shape"),
    (0, "row_active", [2, 0, 0], 2, "row_active"),
    (0, "row_active", [1., 0., 0.], 2, "row_active"),
    (0, "row_value", [np.nan, 0., 0.], 0, "nonfinite"),
    (0, "row_pvar", [-1., 0., 0.], 2, "parameter variance"),
    (0, "row_mvar", [0., 1., 1.], 2, "measurement variance"),
    (0, "row_derivative", [0.]*18, 0, "inactive derivative"),
    (0, "row_derivative", [np.nan]*18, 2, "active derivative"),
    (0, "row_derivative", [np.nan]*17, 0, "shape"),
    (0, "variance_scale", 0., 0, "variance_scale"),
    (0, "variance_scale", np.inf, 0, "variance_scale"),
    (0, "constrain_tot", 2, 0, "bool/0/1"),
])
def test_invalid_schema_flags_and_operands(table, field, value, row, message):
    tables = list(sample())
    tables[table] = change(tables[table], field, value, row)
    with pytest.raises(ValueError, match=message):
        H.validate(*tables)


@pytest.mark.parametrize("table", [0, 1, 2])
def test_duplicate_and_missing(table):
    tables = list(sample())
    tables[table] = ak.concatenate([tables[table], tables[table][:1]])
    with pytest.raises(ValueError, match="duplicate"):
        H.validate(*tables)
    tables = list(sample())
    tables[table] = tables[table][1:]
    with pytest.raises(ValueError, match="count|missing"):
        H.validate(*tables)


@pytest.mark.parametrize("precision", [1e-4, np.nextafter(9.999999747378752e-5, 0.),
                                       np.nextafter(9.999999747378752e-5, np.inf)])
def test_precision_requires_exact_native_float_promotion(precision):
    r, e, i = sample()
    assert H.NATIVE_CA_PRECISION == 9.999999747378752e-5
    assert H.validate(r, e, i)["ca_precision_exact"]
    with pytest.raises(ValueError, match="unsupported CA precision"):
        H.validate(change(r, "precision", precision), e, i)


def test_wrong_fullkey_and_nonstraw():
    r, e, i = sample()
    for altered in (change(r, "event", 99), change(r, "iseed", 99), change(r, "effect_index", 5)):
        with pytest.raises(ValueError, match="identity"):
            H.validate(altered, e, i)
    with pytest.raises(ValueError, match="population"):
        H.validate(r, change(e, "identity_kind", 2), i)


def test_bool_active_and_inactive_variance_masks():
    r, e, i = sample()
    r = ak.with_field(r, r.row_active != 0, "row_active")
    r = change(r, "row_mvar", [0., -10., 0.])
    r = change(r, "row_pvar", [-1., -2., -3.])
    assert H.validate(r, e, i)["passed"]


def test_prediction_sign_scale_and_parameter_variance():
    r, e, i = sample()
    r = change(r, "row_pvar", [1e20]*3, 4)
    assert H.validate(r, e, i)["passed"]
    negative = change(r, "row_derivative", (-np.array(r.row_derivative[4])).tolist(), 4)
    report = H.validate(negative, e, i)
    assert report["weight_matrix"]["passed"] and not report["weight_vector"]["passed"]
    assert report["weight_vector"]["n_failed_hits"] == 1
    assert report["weight_vector"]["failed_hit_keys"] == [[8, 0, 42, 3, 0, 4]]
    with pytest.raises(ValueError, match="variance_scale differs within iteration"):
        H.validate(change(r, "variance_scale", 1., 4), e, i)
    all_scaled = ak.with_field(r, np.ones(len(r)), "variance_scale")
    assert not H.validate(all_scaled, e, i)["passed"]
    assert not H.validate(r, change(e, "reference_parameters", [0.]*6, 4), i)["passed"]


@pytest.mark.parametrize("field,value", [("weight_matrix", [np.nan]*36), ("weight_vector", [np.inf]*6),
                                         ("reference_parameters", [np.nan]*6)])
def test_numeric_failure_keeps_population(field, value):
    r, e, i = sample()
    report = H.validate(r, change(e, field, value, 4), i)
    assert not report["passed"] and report["n_hits"] == 10 and report["n_excluded_hits"] == 0
    metric = report["weight_matrix" if field == "weight_matrix" else "weight_vector"]
    assert metric["n_failed_hits"] == 1 and metric["n_nonfinite_components"] > 0
    json.dumps(report, allow_nan=False)


def test_overflow_and_fixed_tolerance():
    r, e, i = sample()
    report = H.validate(change(r, "row_derivative", [1e308]*18, 4), e, i)
    assert not report["passed"] and report["weight_matrix"]["n_nonfinite_components"] > 0
    assert report["n_hits"] == 10
    # A tiny nonzero against native zero must obey atol, not a global scale correction.
    w = np.array(e.weight_matrix[0])
    w[0] = 2e-7
    report = H.validate(r, change(e, "weight_matrix", w.tolist()), i)
    assert not report["passed"]
    assert report["rtol"] == 1e-5 and report["atol"] == 1e-7


def original_tables():
    rows, effects, iterations = sample()
    seeds = ak.Array([dict(run=8, subrun=s, event=42, iseed=3, optional_nan=np.nan,
                           payload=[1., np.nan, 3.], label="frozen") for s in (0, 1)])
    pieces = ak.Array([dict(run=8, subrun=s, event=42, iseed=3, iteration_ordinal=0,
                            owner=0, piece_index=0, parameters=[1.]*6) for s in (0, 1)])
    return rows, dict(seed_contract=seeds, iterations=iterations, pieces=pieces, effects=effects)


def test_exact_original_trees_nan_order_and_all_fields():
    _, tables = original_tables()
    assert all(x["passed"] for x in H.exact_originals(tables, tables).values())
    for name in tables:
        other = dict(tables)
        other[name] = other[name][::-1]
        with pytest.raises(ValueError, match="keys/order"):
            H.exact_originals(tables, other)
    for field, value in (("label", "changed"), ("optional_nan", 1.), ("payload", [1., 0., 3.])):
        other = dict(tables, seed_contract=change(tables["seed_contract"], field, value))
        with pytest.raises(ValueError, match="values differ"):
            H.exact_originals(tables, other)
    other = dict(tables, seed_contract=change(tables["seed_contract"], "payload", [1., 2.]))
    with pytest.raises(ValueError, match="shape"):
        H.exact_originals(tables, other)


def evidence(tmp_path):
    rows, tables = original_tables()
    events = ak.Array([dict(run=8, subrun=s, event=42) for s in (0, 1)])
    trace = ak.Array([dict(run=8, subrun=s, event=42, iseed=3, iteration_capture_expected=1,
                           iteration_capture_written=1, iteration_capture_requested=True,
                           iteration_capture_valid=True, iteration_capture_errors="") for s in (0, 1)])
    capture, reference = tmp_path/"capture.root", tmp_path/"frozen.root"
    with uproot.recreate(capture) as root:
        for name, table in {"events": events}.items():
            root["trkdump/"+name] = {f: table[f] for f in table.fields}
        for name, table in dict(tables, trace=trace, hit_operands=rows).items():
            root["kktrace/"+name] = {f: table[f] for f in table.fields}
        # Deliberately distinct lower-cycle content: always use highest cycles.
        root["kktrace/iterations"] = {f: tables["iterations"][f] for f in tables["iterations"].fields}
    with uproot.recreate(reference) as root:
        for name, table in tables.items():
            root["kktrace/"+name] = {f: table[:1][f] for f in table.fields}
            root["kktrace/"+name] = {f: table[f] for f in table.fields}
    targets = tmp_path/"targets.json"
    targets.write_text(json.dumps({"keys": H.keys(trace).tolist()}))
    inputs = {}
    for name in ("control", "capture", "baseline", "targets"):
        path = targets if name == "targets" else capture if name == "capture" else tmp_path/(name+".root")
        if name in ("control", "baseline"):
            path.write_bytes(capture.read_bytes())
        if name == "control":
            with uproot.update(path) as root:
                root["kktrace/hit_operands"] = {f: rows[:0][f] for f in rows.fields}
        inputs[name] = {"path": str(path), "sha256": H.digest(path)}
    gate = dict.fromkeys(("passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage",
                          "recorded_capture_equivalent"), True)
    gate.update(smoke_only=False, inputs=inputs, n_events=2, n_seeds=2, n_iterations=2, n_targets=2, n_targets_found=2)
    check = tmp_path/"gate.json"
    check.write_text(json.dumps(gate))
    return SimpleNamespace(check=check, out=tmp_path/"new.json", reference_capture=reference)


def test_wrapper_highest_cycles_hashes_and_exclusive_output(tmp_path):
    args = evidence(tmp_path)
    report = H.run(args)
    assert report["passed"], report
    assert report["hashes_unchanged"] and report["all_four_evidence_hashes_verified"]
    assert report["highest_cycles"]["capture"]["kktrace/iterations"].endswith(";2")
    assert report["highest_cycles"]["reference_capture"]["kktrace/effects"].endswith(";2")
    assert report["reference_capture"]["sha256"] == H.digest(args.reference_capture)
    assert report["source_sha256"] and not report["full_closure"]
    before = args.out.read_bytes()
    with pytest.raises(FileExistsError):
        H.run(args)
    assert args.out.read_bytes() == before


@pytest.mark.parametrize("flag", ["passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage",
                                   "recorded_capture_equivalent", "smoke_only"])
def test_invalid_gate_fails_before_operand_check(tmp_path, monkeypatch, flag):
    args = evidence(tmp_path)
    gate = json.loads(args.check.read_text())
    gate[flag] = flag == "smoke_only"
    args.check.write_text(json.dumps(gate))
    monkeypatch.setattr(H, "validate", lambda *a: pytest.fail("invalid gate reached operand validation"))
    report = H.run(args)
    assert not report["passed"] and "passivity gate" in report["error"]
    assert json.loads(args.out.read_text())["passed"] is False


def test_wrapper_original_tree_change_and_source_hash_race(tmp_path, monkeypatch):
    args = evidence(tmp_path)
    _, tables = original_tables()
    changed = change(tables["seed_contract"], "label", "changed")
    with uproot.update(args.reference_capture) as root:
        root["kktrace/seed_contract"] = {f: changed[f] for f in changed.fields}
    report = H.run(args)
    assert not report["passed"] and "values differ" in report["error"]
    args.out = tmp_path/"race.json"
    # Simulate mutation after initial snapshots, not a native fit/run.
    original = H.exact_originals
    def mutate(current, reference):
        result = original(current, current)
        with args.reference_capture.open("ab") as stream:
            stream.write(b"changed")
        return result
    monkeypatch.setattr(H, "exact_originals", mutate)
    report = H.run(args)
    assert not report["passed"] and not report["hashes_unchanged"]
    assert "changed during audit" in report["integrity_error"]


def test_evidence_hash_mismatch(tmp_path):
    args = evidence(tmp_path)
    gate = json.loads(args.check.read_text())
    gate["inputs"]["capture"]["sha256"] = "0"*64
    args.check.write_text(json.dumps(gate))
    report = H.run(args)
    assert not report["passed"] and "hash changed" in report["error"]


def test_all_unusable_payloads_and_remaining_status_codes():
    r, e, i = sample()
    # Empty usable selection must not lose expected payload widths or drop hits.
    records = ak.to_list(r)
    for row in records:
        row.update(hit_state=-3, active=False, ca_status=4, ca_usable=False,
                   row_active=[0, 0, 0], row_derivative=[np.nan]*18)
        for field in ("particle_poca", "sensor_poca", "particle_direction", "sensor_direction", "dDdP", "dTdP"):
            row[field] = []
        for field in ("doca", "doca_variance", "toca_variance", "lsign"):
            row[field] = np.nan
    # Preserve native double vector types even when every vector is empty.
    r = ak.Array(records)
    for field in ("particle_poca", "sensor_poca", "particle_direction", "sensor_direction", "dDdP", "dTdP"):
        r = ak.with_field(r, ak.unflatten(np.array([], dtype=float), np.zeros(len(r), dtype=int)), field)
    er = ak.to_list(e)
    for effect in er:
        effect.update(active=False, weight_matrix=[0.]*36, weight_vector=[0.]*6)
    report = H.validate(change(r, "ca_status", 5), ak.Array(er), i)
    assert report["passed"] and report["n_unusable_ca"] == 10


def test_one_ulp_toca_is_not_exact_and_missing_payload_rejected():
    r, e, i = sample()
    with pytest.raises(ValueError, match="exact effect time"):
        H.validate(change(r, "particle_poca", [1., 2., 3., np.nextafter(12., 13.)], 2), e, i)
    with pytest.raises(ValueError, match="missing"):
        H.validate(change(r, "row_value", [None, 0., 0.]), e, i)


def test_overflowed_variance_is_numeric_failure_not_zero_information():
    r, e, i = sample()
    r = change(r, "row_mvar", [1e308]*3, 4)
    e = change(change(e, "weight_matrix", [0.]*36, 4), "weight_vector", [0.]*6, 4)
    report = H.validate(r, e, i)
    assert not report["passed"] and report["n_variance_arithmetic_failures"] == 1
    assert report["weight_matrix"]["n_failed_hits"] == 1 and report["n_hits"] == 10


def test_exact_original_missing_duplicate_and_subtolerance_difference():
    _, tables = original_tables()
    for changed, message in (
        (ak.without_field(tables["seed_contract"], "label"), "field set"),
        (ak.concatenate([tables["seed_contract"], tables["seed_contract"][:1]]), "duplicate"),
        (change(tables["seed_contract"], "payload", [1. + 1e-12, np.nan, 3.]), "values differ"),
        (change(tables["seed_contract"], "payload", [1., None, 3.]), "type differs|missing"),
    ):
        with pytest.raises(ValueError, match=message):
            H.exact_originals(tables, dict(tables, seed_contract=changed))


def test_capture_off_sidecar_must_be_empty(tmp_path):
    args = evidence(tmp_path)
    gate = json.loads(args.check.read_text())
    control = Path(gate["inputs"]["control"]["path"])
    rows, _ = original_tables()
    with uproot.update(control) as root:
        root["kktrace/hit_operands"] = {f: rows[:1][f] for f in rows.fields}
    gate["inputs"]["control"]["sha256"] = H.digest(control)
    args.check.write_text(json.dumps(gate))
    report = H.run(args)
    assert not report["passed"] and "capture-off hit operand" in report["error"]


def test_current_capture_cannot_be_its_own_frozen_reference(tmp_path):
    args = evidence(tmp_path)
    gate = json.loads(args.check.read_text())
    args.reference_capture = Path(gate["inputs"]["capture"]["path"])
    report = H.run(args)
    assert not report["passed"] and "independent of current" in report["error"]
