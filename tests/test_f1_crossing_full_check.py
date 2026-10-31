"""Synthetic fixed-population driver tests; never native/remote execution.

Pins/constants and receipt-bound helper files remain unchanged. Fixtures are not
historical receipts and these tests do not certify the real 3k capture or peak RSS.
"""
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_crossing_full_check as F


def gate_fixture(tmp_path):
    inputs = {}
    for name in ("control", "capture", "baseline", "targets"):
        path = tmp_path / (name + ".data")
        path.write_text(name)
        inputs[name] = {"path": str(path), "sha256": F.A.digest(path)}
    gate = dict(F.COUNTS, **{name: True for name in F.TRUE_FLAGS}, smoke_only=False,
                full_numerical_validation=False, missing_target_keys=[], inputs=inputs)
    path = tmp_path / "check.json"
    path.write_text(json.dumps(gate))
    return path, gate


def test_full_receipt_watches_every_input(tmp_path):
    path, gate = gate_fixture(tmp_path)
    watched = {}
    assert F.full_receipt(path, watched) == gate
    assert len(watched) == 5
    assert all(F.A.digest(p) == h for p, h in watched.items())


@pytest.mark.parametrize("field,bad", [(name, bad) for name in F.TRUE_FLAGS
    for bad in (1, "true", False, None, [])] + [(name, bad) for name in
    ("smoke_only", "full_numerical_validation") for bad in (0, "false", True, None)])
def test_full_flags_must_be_literal(tmp_path, field, bad):
    path, gate = gate_fixture(tmp_path)
    gate[field] = bad
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        F.full_receipt(path, {})


@pytest.mark.parametrize("field", list(F.COUNTS))
@pytest.mark.parametrize("bad", [True, False, None, -1, 0, 100, 1.5, 3000.0, "3000", 2**63])
def test_fixed_counts_are_strict_integers(tmp_path, field, bad):
    path, gate = gate_fixture(tmp_path)
    gate[field] = bad
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        F.full_receipt(path, {})


@pytest.mark.parametrize("bad", [None, {}, [[1, 0, 9, 0]], False, "[]"])
def test_missing_targets_field_is_explicit_empty_list(tmp_path, bad):
    path, gate = gate_fixture(tmp_path)
    gate["missing_target_keys"] = bad
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        F.full_receipt(path, {})


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":{"b":0,"b":1}}',
                                  '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}', '{"a":1e999}'])
def test_json_rejects_ambiguity_and_nonfinite(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        F.strict_json(path)


def test_hash_mismatch_and_conflicting_watches(tmp_path):
    path, gate = gate_fixture(tmp_path)
    changed = Path(gate["inputs"]["capture"]["path"])
    changed.write_text("changed")
    watched = {}
    with pytest.raises(ValueError, match="hash mismatch"):
        F.full_receipt(path, watched)
    assert str(changed) in watched
    old = watched[str(changed)]
    changed.write_text("changed again")
    with pytest.raises(ValueError, match="changed watched"):
        F.watch(changed, watched)
    assert watched[str(changed)] == old


def test_pinned_receipt_cannot_be_replaced(tmp_path):
    path, _ = gate_fixture(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        F.full_receipt(path, {}, F.REFERENCE_SHA)


def test_actual_helper_paths_and_pins():
    watched = {}
    assert F.verify_helpers(watched) == F.MODULE_SHA
    assert len(watched) == 6
    assert F.C.MANIFEST_SHA == "ce3762ab9a5610e5838a5939ccff47c67d362812a41ddb1428eca13cc269cd5d"
    assert F.C.PREPARATION_SHA == "9b25e7f783b0949f8a7e1c5c27548a0928a59eb45f1d1090d63a6fe082967c84"
    assert F.C.REFERENCE_CHECK_SHA == "683ed29ea38ed004993acf9786e4c2ad4db269e6ce76f8ad4d88ffc7311b9209"


def test_other_checkout_module_is_rejected(monkeypatch):
    # Replace the driver's module reference, never mutate a pinned helper/constant.
    shadow = SimpleNamespace(__name__="f1_crossing_geometry_check", __file__="/path/to/other/f1_crossing_geometry_check.py")
    monkeypatch.setattr(F, "G", shadow)
    with pytest.raises(ValueError, match="other checkout"):
        F.verify_helpers({})


def test_transitive_executing_code_path_is_checked(monkeypatch):
    original = F.G
    shadow = SimpleNamespace(**vars(original))
    shadow.validate = original.validate.__class__(original.validate.__code__.replace(co_filename="/path/to/other.py"),
                                                original.validate.__globals__, "validate")
    shadow.validate.__module__ = original.__name__
    monkeypatch.setattr(F, "G", shadow)
    with pytest.raises(ValueError, match="transitive|executing"):
        F.verify_helpers({})


def historical_fixture(tmp_path):
    source = tmp_path / "source"
    source.write_text("source")
    hashes = {str(source): F.A.digest(source)}
    path = tmp_path / "historical.json"
    document = dict(passed=True, hashes_unchanged=True, before_sha256=hashes, end_sha256=hashes)
    path.write_text(json.dumps(document))
    return path, source, document


def test_historical_hashes_checked_before_and_recorded(tmp_path):
    path, source, document = historical_fixture(tmp_path)
    watched = {}
    assert F.historical(path, F.A.digest(path), watched) == document
    assert set(watched) == {str(path), str(source)}
    source.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        F.historical(path, F.A.digest(path), {})


def test_historical_before_end_cannot_disagree(tmp_path):
    path, _, document = historical_fixture(tmp_path)
    document["before_sha256"] = {}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="before/end"):
        F.historical(path, F.A.digest(path), {})


def targets_fixture():
    return np.column_stack((np.ones(147, dtype=int), np.zeros(147, dtype=int), np.arange(147), np.zeros(147, dtype=int)))


def test_targets_exact_fixed_population():
    expected = targets_fixture()
    assert np.array_equal(F.targets({"keys": expected.tolist()}), expected)


@pytest.mark.parametrize("kind", ["short", "duplicate", "negative", "float", "bool", "overflow", "masked", "shape"])
def test_targets_reject_bad_identity_before_conversion(kind):
    raw = targets_fixture()
    if kind == "short": raw = raw[:-1]
    if kind == "duplicate": raw[1] = raw[0]
    if kind == "negative": raw[0, 0] = -1
    if kind == "float": raw = raw.astype(float)
    if kind == "bool":
        raw = raw.tolist()
        raw[0][0] = True
    if kind == "overflow":
        raw = raw.astype(np.uint64)
        raw[0, 0] = 2**63
    if kind == "masked": raw = np.ma.array(raw, mask=False)
    if kind == "shape": raw = raw[:, :3]
    with pytest.raises(ValueError):
        F.targets({"keys": raw})


def population_fixture():
    seed = {name: col for name, col in zip(F.A.KEYS, np.column_stack((np.ones(2319, dtype=int),
        np.zeros(2319, dtype=int), np.arange(2319), np.zeros(2319, dtype=int))).T)}
    counts = np.zeros(2319, dtype=int)
    counts[:147] = 23
    counts[:68] += 1
    assert counts.sum() == 3449
    iterations = {name: np.repeat(value, counts) for name, value in seed.items()}
    iterations["iteration_ordinal"] = np.arange(3449) - np.repeat(np.cumsum(counts) - counts, counts)
    trace = dict(seed, iteration_capture_expected=counts, iteration_capture_written=counts,
                 iteration_capture_requested=counts > 0, iteration_capture_valid=counts > 0,
                 iteration_capture_errors=[""] * 2319)
    seed = ak.Array(dict(seed, capture_requested=counts > 0))
    event_keys = np.column_stack((np.ones(3000, dtype=int), np.zeros(3000, dtype=int), np.arange(3000)))
    return seed, ak.Array(iterations), ak.Array(trace), event_keys, targets_fixture()


def test_all_147_targets_and_3449_iterations():
    assert F.selected(*population_fixture()).shape == (2319, 4)


@pytest.mark.parametrize("field", ["iteration_capture_expected", "iteration_capture_written"])
@pytest.mark.parametrize("kind", ["fractional", "bool", "overflow", "masked", "wrong"])
def test_trace_counts_reject_invalid_types_and_values(field, kind):
    seed, iterations, trace, events, expected = population_fixture()
    values = F.A.array(trace[field]).copy()
    if kind == "fractional": values = values.astype(float) + .5
    if kind == "bool": values = values.astype(bool)
    if kind == "overflow":
        values = values.astype(np.uint64)
        values[0] = 2**63
    if kind == "masked":
        values = values.tolist()
        values[0] = None
    if kind == "wrong": values[0] += 1
    trace = ak.with_field(trace, values, field)
    with pytest.raises(ValueError):
        F.selected(seed, iterations, trace, events, expected)


@pytest.mark.parametrize("kind", ["missing_target", "missing_seed", "ordinal_gap", "request", "valid", "error", "seed_event"])
def test_selected_does_not_infer_targets_from_surviving_rows(kind):
    seed, it, trace, events, expected = population_fixture()
    if kind == "missing_target": expected[-1, 2] = 2999
    if kind == "missing_seed": seed = seed[:-1]
    if kind == "ordinal_gap": it = ak.with_field(it, it.iteration_ordinal + 1, "iteration_ordinal")
    if kind == "request": seed = ak.with_field(seed, np.zeros(len(seed), dtype=bool), "capture_requested")
    if kind == "valid": trace = ak.with_field(trace, np.zeros(len(trace), dtype=bool), "iteration_capture_valid")
    if kind == "error": trace = ak.with_field(trace, ["failure"] * len(trace), "iteration_capture_errors")
    if kind == "seed_event": events = events[1:]
    with pytest.raises(ValueError):
        F.selected(seed, it, trace, events, expected)


@pytest.mark.parametrize("value", [2**63, 2**64-1])
def test_integer_keys_unsigned_overflow(value):
    table = ak.Array({"run": np.array([value], dtype=np.uint64)})
    with pytest.raises(ValueError, match="overflow"):
        F.integer_keys(table, ("run",))


def test_full_event_payload_seedless_and_nan():
    events = ak.Array(dict(run=np.ones(3000, dtype=int), subrun=np.zeros(3000, dtype=int),
                           event=np.arange(3000), payload=np.full(3000, np.nan)))
    assert F.C.exact_table(events, events, F.A.KEYS[:3], "events")["n_rows"] == 3000
    changed = np.full(3000, np.nan)
    changed[-1] = 0.
    with pytest.raises(ValueError, match="values differ"):
        F.C.exact_table(events, ak.with_field(events, changed, "payload"), F.A.KEYS[:3], "events")
    with pytest.raises(ValueError, match="keys/order"):
        F.C.exact_table(events, events[::-1], F.A.KEYS[:3], "events")


def test_exact_jagged_nan_all_fields():
    a = ak.Array([dict(run=1, subrun=0, event=2, iseed=0, q=[1., np.nan]),
                  dict(run=1, subrun=0, event=3, iseed=0, q=[])])
    assert F.C.exact_table(a, a, F.A.KEYS, "old")["passed"]
    for b in (ak.with_field(a, [[np.nextafter(1., 2.), np.nan], []], "q"),
              ak.with_field(a, [[1.], [np.nan]], "q"), ak.without_field(a, "q"),
              ak.with_field(a, [[None, np.nan], []], "q")):
        with pytest.raises(ValueError):
            F.C.exact_table(a, b, F.A.KEYS, "old")


def test_highest_cycle_read_and_complete_empty_schema(tmp_path):
    path = tmp_path / "cycles.root"
    full = ak.Array(dict(run=np.array([1], dtype=np.int32), q=np.array([[1., 2., 3.]])))
    with uproot.recreate(path) as root:
        root["kktrace/example"] = full
        root["kktrace/example"] = full[:0]
    cycles = {}
    empty = F.read(path, "kktrace", "example", cycles)
    assert len(empty) == 0 and cycles["kktrace/example"].endswith("example;2")
    F.empty_schema(path, "example", full, cycles)
    with pytest.raises(ValueError, match="partial schema"):
        F.empty_schema(path, "example", ak.with_field(full, [1], "extra"), {})
    with pytest.raises(ValueError, match="type differs"):
        F.empty_schema(path, "example", ak.with_field(full, np.array([1.]), "run"), {})
    with uproot.update(path) as root:
        root["kktrace/example"] = full
    with pytest.raises(ValueError, match="not empty"):
        F.empty_schema(path, "example", full, {})


def test_missing_tree_rejected(tmp_path):
    path = tmp_path / "missing.root"
    with uproot.recreate(path) as root:
        root["kktrace/other"] = {"run": np.array([1])}
    with pytest.raises(ValueError, match="missing tree"):
        F.read(path, "kktrace", "example", {})


def test_hit_summary_removes_only_unbounded_lists():
    metric = dict(passed=True, n_hits=61622, n_components=2218392, n_failed_components=0,
                  n_nonfinite_components=0, n_failed_hits=0, max_finite_absolute_residual=1e-8,
                  failed_hit_keys=[], per_hit_failed_components=[0] * 61622)
    report = dict(passed=True, n_hits=61622, hit_keys=[[1, 0, 1, 0, 0, 0]] * 61622,
                  weight_matrix=copy.deepcopy(metric), weight_vector=copy.deepcopy(metric), n_excluded_hits=0)
    result = F.compact_hit(report)
    assert len(json.dumps(result, allow_nan=False)) < 1500
    assert result["weight_matrix"]["n_components"] == 2218392
    assert "per_hit_failed_components" not in result["weight_matrix"]
    assert result["n_excluded_hits"] == 0


def test_manifest_fields_reject_duplicate_and_bind_input(tmp_path):
    path = tmp_path / "manifest.txt"
    path.write_text("date\narea=/native\ninput=/input\nevents=3000\nconfig=/config\n")
    assert F.manifest_fields(path)["events"] == "3000"
    path.write_text(path.read_text() + "events=100\n")
    with pytest.raises(ValueError, match="ambiguous"):
        F.manifest_fields(path)


def test_expanded_only_unique_debug_filename(tmp_path):
    a, b = tmp_path / "a.fcl", tmp_path / "b.fcl"
    a.write_text(f'cut: 2\nfileName: "{a}"\n')
    b.write_text(f'cut: 2\nfileName: "{b}"\n')
    F.C.expanded_equal(a, b)
    a.write_text(a.read_text().replace("cut: 2", "cut: 3"))
    with pytest.raises(ValueError):
        F.C.expanded_equal(a, b)


def run_args(tmp_path):
    return SimpleNamespace(**{name: tmp_path / (name + ".json") for name in
        ("check", "reference_check", "reference_hit_check", "smoke_check", "preparation", "out")},
        native_source=tmp_path / "native")


def test_failure_report_strict_and_exclusive(tmp_path):
    args = run_args(tmp_path)
    report = F.run(args)  # actual helper checks, then missing native manifest
    assert report["passed"] is False and report["full_target_capture_valid"] is False
    assert report["full_fit_closure"] is report["full_f1_closure"] is False
    assert report["hashes_unchanged"] is True
    assert len(report["before_sha256"]) == 6
    assert json.loads(args.out.read_text()) == report
    with pytest.raises(FileExistsError):
        F.run(args)


def test_end_rehash_runs_on_failure_and_detects_change(tmp_path, monkeypatch):
    args = run_args(tmp_path)
    source = tmp_path / "synthetic_input"
    source.write_text("before")
    def changed(watched):
        F.watch(source, watched)
        source.write_text("after")
        raise ValueError("synthetic failure")
    monkeypatch.setattr(F, "verify_helpers", changed)
    report = F.run(args)
    assert report["passed"] is report["hashes_unchanged"] is False
    assert report["before_sha256"][str(source)] != report["end_sha256"][str(source)]


@pytest.mark.parametrize("field", ["check", "reference_check", "reference_hit_check", "smoke_check", "preparation"])
def test_protected_receipt_output_rejected(tmp_path, field):
    args = run_args(tmp_path)
    path = getattr(args, field)
    path.write_text("untouched")
    args.out = path
    with pytest.raises(ValueError, match="protected"):
        F.run(args)
    assert path.read_text() == "untouched"


def test_source_output_rejected(tmp_path):
    args = run_args(tmp_path)
    args.out = args.native_source / "new.json"
    with pytest.raises(ValueError, match="protected"):
        F.run(args)


def test_cli_flags_are_separate_and_complete():
    process = subprocess.run([sys.executable, "-B", F.__file__, "--help"], text=True, capture_output=True)
    assert process.returncode == 0
    for flag in ("check", "reference-check", "reference-hit-check", "smoke-check", "native-source", "preparation", "out"):
        assert "--" + flag in process.stdout


def test_memory_lifetimes_and_no_native_execution():
    source = inspect.getsource(F.population)
    assert source.count("G.validate(") == source.count("R.validate(") == 1
    assert source.index("data.clear()") < source.index("for suffix in G.FIELDS")
    assert "del reference" in source and "empty_schema" in source
    driver = inspect.getsource(F)
    assert "subprocess" not in driver and "os.system" not in driver
    assert "entry_stop=0" in driver and "array_cache=None" in driver


def history_bindings():
    previous = {"inputs": {"capture": {"path": "/old", "sha256": "hash"}}}
    hit = dict(passed=True, all_four_evidence_hashes_verified=True, capture_off_hit_operands_empty=True,
               check_sha256=F.REFERENCE_SHA, inputs=previous["inputs"], n_hits=61622, n_expected_hits=61622,
               n_excluded_hits=0, n_iterations=3449, full_closure=False, full_numerical_validation=False,
               rtol=F.H.RTOL, atol=F.H.ATOL,
               original_tree_passivity={name: {"passed": True} for name in F.H.ORIGINAL_KEYS})
    smoke = dict(smoke_only=True, full_fit_closure=False, n_events=100, n_targets=2,
                 before_sha256={"/historical/" + name + ".py": digest for name, digest in F.MODULE_SHA.items()})
    return previous, hit, smoke


def test_history_binds_exact_full_predecessor_and_smoke_helpers():
    F.bind_history(*history_bindings())


@pytest.mark.parametrize("field,bad", [("all_four_evidence_hashes_verified", 1),
    ("capture_off_hit_operands_empty", 1), ("check_sha256", F.SMOKE_SHA), ("inputs", {}),
    ("n_hits", 61622.0), ("n_expected_hits", True), ("n_excluded_hits", False),
    ("n_iterations", 49), ("full_closure", 0), ("full_numerical_validation", 0),
    ("rtol", 1e-4), ("atol", 1e-6), ("original_tree_passivity", {})])
def test_history_rejects_wrong_hit_binding(field, bad):
    previous, hit, smoke = history_bindings()
    hit[field] = bad
    with pytest.raises(ValueError):
        F.bind_history(previous, hit, smoke)


@pytest.mark.parametrize("field,bad", [("smoke_only", 1), ("full_fit_closure", 0),
    ("n_events", 100.0), ("n_targets", 147), ("before_sha256", {})])
def test_history_rejects_wrong_smoke_binding(field, bad):
    previous, hit, smoke = history_bindings()
    smoke[field] = bad
    with pytest.raises(ValueError):
        F.bind_history(previous, hit, smoke)


def test_source_pins_are_exact_recorded_preparation_pins(tmp_path):
    area = tmp_path / "native"
    area.mkdir()
    pins = {"Offline": "a", "KinKal": "b", "Production": "c", "mu2e-trig-config": "d", "EventNtuple": "e"}
    preparation = tmp_path / "preparation.json"
    preparation.write_text(json.dumps({"pins": pins}))
    path = area / "source-pins.txt"
    path.write_text("".join(f"{name} {pin}\n" for name, pin in pins.items()))
    watched = {}
    assert F.source_pins(area, preparation, watched) == pins
    assert str(path) in watched
    path.write_text(path.read_text().replace("Offline a", "Offline changed"))
    with pytest.raises(ValueError, match="pins differ"):
        F.source_pins(area, preparation, {})


def geometry_binding_fixture():
    expected = targets_fixture()
    table = ak.Array(dict(nplanes=np.full(294, 36, dtype=np.int64),
                          npanels=np.full(294, 216, dtype=np.int64),
                          nstraws=np.full(294, 20736, dtype=np.int64)))
    smoke = {"geometry": dict(passed=True, n_targets=2, n_planes=72, n_panels=432, n_straws=41472)}
    return {F.G.PREFIX + "bindings": table}, expected, smoke


def test_geometry_census_binds_per_target_smoke_detector_population():
    F.geometry_census(*geometry_binding_fixture())


@pytest.mark.parametrize("field", ["nplanes", "npanels", "nstraws"])
@pytest.mark.parametrize("bad", [0, 1, True, 20736.0, 2**63, None])
def test_empty_or_invalid_geometry_cannot_pass_self_consistently(field, bad):
    tables, expected, smoke = geometry_binding_fixture()
    table = tables[F.G.PREFIX + "bindings"]
    if bad == 2**63:
        values = np.full(294, bad, dtype=np.uint64)
    else:
        values = [bad] * 294
    tables[F.G.PREFIX + "bindings"] = ak.with_field(table, values, field)
    with pytest.raises(ValueError):
        F.geometry_census(tables, expected, smoke)


def test_success_and_end_hash_failure_are_not_full_fit_claims(tmp_path, monkeypatch):
    """Orchestration only: seams replace I/O, not helpers/constants or validators."""
    args = run_args(tmp_path)
    path, gate = gate_fixture(tmp_path)
    target_path = Path(gate["inputs"]["targets"]["path"])
    target_path.write_text(json.dumps({"keys": targets_fixture().tolist()}))
    previous = copy.deepcopy(gate)
    for name in ("control", "capture"):
        previous["inputs"][name]["path"] += ".old"
    def receipt(path, watched, expected=None):
        F.watch(target_path, watched)
        return previous if expected else gate
    monkeypatch.setattr(F, "full_receipt", receipt)
    monkeypatch.setattr(F, "historical", lambda *args: {})
    monkeypatch.setattr(F, "bind_history", lambda *args: None)
    monkeypatch.setattr(F, "source_pins", lambda *args: {})
    monkeypatch.setattr(F, "configuration", lambda *args: {"passed": True})
    monkeypatch.setattr(F, "manifest_integrity", lambda *args: {"passed": True})
    # A module proxy avoids modifying the actual immutable smoke helper binding.
    proxy = SimpleNamespace(**vars(F.C))
    proxy.source_manifest = lambda *args: None
    monkeypatch.setattr(F, "C", proxy)
    monkeypatch.setattr(F, "verify_helpers", lambda watched: {})
    def successful(gate, previous, hit, expected, report, smoke):
        assert expected.shape == (147, 4)
        report["n_targets"] = 147
    monkeypatch.setattr(F, "population", successful)
    report = F.run(args)
    assert report["passed"] is report["full_target_capture_valid"] is True
    assert report["full_fit_closure"] is report["full_f1_closure"] is False
    assert report["hashes_unchanged"] is True
    args.out = tmp_path / "out2.json"
    def changed(*args):
        target_path.write_text("changed during validation")
    monkeypatch.setattr(F, "population", changed)
    report = F.run(args)
    assert report["passed"] is report["full_target_capture_valid"] is report["hashes_unchanged"] is False


def configuration_fixture(tmp_path, monkeypatch):
    """Synthetic file contents: replace only hash I/O, never any immutable pin."""
    directories = {name: tmp_path / name for name in ("current", "previous", "smoke", "native", "config")}
    for directory in directories.values():
        directory.mkdir()
    (tmp_path / "au_dig_3f.txt").write_text(str(tmp_path / "input.root") + "\n")
    fields = f'area={directories["native"]}\ninput={tmp_path / "input.root"}\nconfig={directories["config"]}\n'
    for name in ("current", "previous", "smoke"):
        (directories[name] / "manifest.txt").write_text(fields + f'events={100 if name == "smoke" else 3000}\n')
    for name in ("current", "previous"):
        for arm in ("control", "capture"):
            expanded = directories[name] / (arm + ".expanded.fcl")
            expanded.write_text(f'cut: 2\nfileName: "{expanded}"\n')
            (directories[name] / (arm + ".exit")).write_text("0\n")
    def gate(name):
        return {"inputs": {**{arm: {"path": str(directories[name] / (arm + ".root"))} for arm in ("control", "capture")},
                           "targets": {"sha256": "f32c3e49bec3746fc15a57bc588a3858a15195820fd1705a4d756cb7504fce90"}}}
    requested_pins = {}
    def watch(path, watched, expected=None):
        requested_pins[str(path)] = expected
    monkeypatch.setattr(F, "watch", watch)
    smoke = {"before_sha256": {str(directories["smoke"] / "manifest.txt"): "synthetic"}}
    return directories, gate("current"), gate("previous"), smoke, requested_pins


def test_configuration_checks_fixed_launch_and_all_pinned_predecessor_configs(tmp_path, monkeypatch):
    dirs, gate, previous, smoke, pins = configuration_fixture(tmp_path, monkeypatch)
    result = F.configuration(gate, previous, smoke, dirs["native"], {})
    assert result["passed"] and result["same_input_config_as_smoke"]
    assert pins[str(dirs["previous"] / "manifest.txt")] == F.PRED_MANIFEST_SHA
    for arm in ("control", "capture"):
        assert pins[str(dirs["previous"] / (arm + ".expanded.fcl"))] == F.EXPANDED_SHA[arm]
    assert str(tmp_path / "input.root") not in pins
    assert pins[str(tmp_path / "au_dig_3f.txt")] == "91c49c9a256a0fd6f7af34620db96ad40cab4336f616316ef31855e130090802"
    assert str(dirs["config"] / "targets.json") in pins


@pytest.mark.parametrize("kind", ["events", "input", "config", "area", "smoke_input", "smoke_events", "exit", "expanded", "duplicate", "pair"])
def test_configuration_fails_on_every_unallowed_change(tmp_path, monkeypatch, kind):
    dirs, gate, previous, smoke, _ = configuration_fixture(tmp_path, monkeypatch)
    manifest = dirs["smoke" if kind.startswith("smoke_") else "current"] / "manifest.txt"
    field = kind.removeprefix("smoke_")
    if field in ("events", "input", "config", "area"):
        lines = manifest.read_text().splitlines()
        manifest.write_text("\n".join(field + "=changed" if line.startswith(field + "=") else line for line in lines))
    elif kind == "exit":
        (dirs["current"] / "control.exit").write_text("1\n")
    elif kind == "expanded":
        path = dirs["current"] / "capture.expanded.fcl"
        path.write_text(path.read_text().replace("cut: 2", "cut: 3"))
    elif kind == "duplicate":
        manifest.write_text(manifest.read_text() + "events=3000\n")
    elif kind == "pair":
        gate["inputs"]["control"]["path"] = str(tmp_path / "different/control.root")
    with pytest.raises(ValueError):
        F.configuration(gate, previous, smoke, dirs["native"], {})


def test_manifest_relative_and_absolute_hashes(tmp_path):
    p = tmp_path / "manifest.txt"
    sha = "a" * 64
    p.write_text(f"date\narea=/native\n{sha}  KinKal/Fit/Track.hh\n{sha}  {tmp_path}/binary.so\n")
    assert F.manifest_hashes(p, tmp_path) == {str(tmp_path / "KinKal/Fit/Track.hh"): sha, str(tmp_path / "binary.so"): sha}
    p.write_text(p.read_text() + f"{'b' * 64}  {tmp_path}/binary.so\n")
    with pytest.raises(ValueError, match="conflicting manifest"):
        F.manifest_hashes(p, tmp_path)


def test_manifest_rejects_pnfs_hash_entries_without_touching_namespace(tmp_path):
    p = tmp_path / "manifest.txt"
    p.write_text(f"{'a' * 64}  {'/pnfs/'}never-query-this\n")
    with pytest.raises(ValueError, match="PNFS"):
        F.manifest_hashes(p, tmp_path)


def test_manifest_integrity_requires_recorded_repo_pins(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    area = tmp_path / "native"
    area.mkdir()
    (d / "manifest.txt").write_text("events=3000\n")
    (area / "capture-source-manifest.sha256").write_text(f"{'a' * 64}  Track.hh\n")
    args = SimpleNamespace(native_source=area)
    gate = {"inputs": {"capture": {"path": str(d / "capture.root")}}}
    with pytest.raises(ValueError, match="repository pins"):
        F.manifest_integrity(args, gate, {"KinKal": "b" * 40}, {})
