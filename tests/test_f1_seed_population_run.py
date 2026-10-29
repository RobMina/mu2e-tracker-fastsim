"""Inherited-control runner tests: real toy receipts, mocked numerical work only.

These are orchestration/provenance tests, not native population fidelity evidence.
No receipt-bound source or existing test fixture is rewritten.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

import f1_seed_population_run as U


ARMS = ("parameters", "momentum", "geometry_time", "covariance", "range")
FLAGS = ("completed", "controls_passed", "hashes_unchanged", "controls_only")
CLOSURES = ("full_fit_closure", "full_f1_closure", "full_numerical_validation")


@pytest.fixture
def case(tmp_path, monkeypatch):
    """Keep receipt/watch/strict_json/code_origin/save/serialize completely real."""
    paths = {}
    for name in ("source", "runtime", "input", "control_artifact"):
        paths[name] = tmp_path/name
        paths[name].write_bytes(("immutable toy "+name).encode())
    args = SimpleNamespace(out=tmp_path/"new-populations", controls=tmp_path/"control.json",
                           controls_sha="", mode="own", seed_contract=tmp_path/"seed.json",
                           covariance_contract=tmp_path/"covariance.json", check=tmp_path/"check.json",
                           conditions=tmp_path/"conditions", straw_axis=tmp_path/"axis",
                           baseline=tmp_path/"baseline")
    execution = dict(backend="cpu", x64=True, jax="toy-jax", numpy="toy-numpy")
    prepared = dict(mode="own", keys=[[1400, 0, 3700, 0], [1400, 0, 7033, 0]],
                    schedule={"temperature": [1., 0.5], "maxniter": [3, 10]},
                    baseline_metadata={"own_states": True, "time_model": "stock"},
                    native_time={"provenance": "native_ParameterHit_time_only", "n": 2},
                    native_seed={"provenance": "captured", "n": 2})
    before = {str(paths[name]): U.A.digest(paths[name]) for name in ("source", "runtime", "input")}
    control = dict(**deepcopy(prepared), **{name: True for name in FLAGS},
                   **{name: False for name in CLOSURES}, arms={},
                   controls={"passed": True, "keys": deepcopy(prepared["keys"]), "batches": [{"exact": True}]},
                   execution=deepcopy(execution), before_sha256=before, end_sha256=deepcopy(before),
                   artifacts={"bounded_control": {"path": str(paths["control_artifact"]),
                                                   "sha256": U.A.digest(paths["control_artifact"])}})
    calls, events = [], []
    data = {"toy_population": True}

    def seal():
        args.controls.write_text(json.dumps(control, allow_nan=False), encoding="utf-8")
        args.controls_sha = U.A.digest(args.controls)

    def helpers(watched):
        events.append("helpers")
        for name in ("source", "runtime"):
            U.D.watch(paths[name], watched)
        return deepcopy(execution)

    def prepare(received, watched, report):
        assert received is args
        events.append("prepare")
        U.D.watch(paths["input"], watched)
        report.update(deepcopy(prepared))
        return data

    def population(received, arm, out, report):
        assert received is data and out == args.out.resolve()
        assert report["controls_passed"] is True
        calls.append(arm)
        events.append(arm)
        # NPZ may retain nonfinite output leaves; the JSON summary must be strict.
        U.R.save(out, arm, {"value": np.array([1., np.nan, np.inf, -np.inf])}, report)
        report["arms"][arm] = {"summary": {"n": 4, "nonfinite": 3}}

    def forbidden_controls(*unused, **kwargs):
        pytest.fail("inherited controls must never be refitted")

    monkeypatch.setattr(U.R, "helpers", helpers)
    monkeypatch.setattr(U.R, "prepare", prepare)
    monkeypatch.setattr(U.R, "population", population)
    monkeypatch.setattr(U.R, "controls", forbidden_controls)
    monkeypatch.setattr(U.jax, "clear_caches", lambda: events.append("clear"))
    seal()
    return SimpleNamespace(args=args, paths=paths, control=control, execution=execution,
                           prepared=prepared, calls=calls, events=events, seal=seal,
                           population=population, data=data)


def manifest(case, report):
    path = case.args.out/"manifest.json"
    assert U.D.strict_json(path) == report
    assert path.read_bytes().endswith(b"\n")
    assert all(report[name] is False for name in CLOSURES)


def refused(case, message=None):
    report = U.run(case.args)
    assert report["completed"] is False
    assert report["controls_passed"] is False
    if message:
        assert message in report["error"]
    assert case.calls == []
    assert report["arms"] == {}
    manifest(case, report)
    return report


@pytest.mark.parametrize("mode", ["own", "fixed"])
def test_completed_control_is_reauthenticated_and_only_five_new_arms_execute(case, mode):
    case.args.mode = mode
    case.prepared["mode"] = case.control["mode"] = mode
    case.prepared["baseline_metadata"]["own_states"] = mode == "own"
    case.control["baseline_metadata"] = deepcopy(case.prepared["baseline_metadata"])
    case.seal()
    original = {path: path.read_bytes() for path in [case.args.controls, *case.paths.values()]}
    report = U.run(case.args)
    assert report["completed"] is report["controls_passed"] is report["hashes_unchanged"] is True
    assert U.EXPECTED == ARMS
    assert case.calls == list(ARMS) and tuple(report["arms"]) == ARMS
    assert case.events == ["helpers", "prepare"] + [item for arm in ARMS for item in (arm, "clear")]
    assert report["controls"] == case.control["controls"]
    assert report["controls_receipt"] == str(case.args.controls)
    assert report["controls_sha256"] == case.args.controls_sha
    assert report["execution"] == case.execution
    expected_paths = {str(path.resolve()) for path in [case.args.controls, *case.paths.values()]}
    expected_paths.add(str(Path(U.__file__).resolve()))
    assert set(report["before_sha256"]) == expected_paths
    assert report["before_sha256"] == report["end_sha256"]
    assert report["hash_errors"] == []
    assert set(report["artifacts"]) == set(ARMS)
    assert {p.name for p in case.args.out.iterdir()} == {arm+".npz" for arm in ARMS} | {"manifest.json"}
    for arm in ARMS:
        artifact = report["artifacts"][arm]
        assert U.A.digest(artifact["path"]) == artifact["sha256"]
        with np.load(artifact["path"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["value"], [1., np.nan, np.inf, -np.inf])
    assert all(path.read_bytes() == value for path, value in original.items())
    manifest(case, report)


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("bad", ["missing", False, None, 0, 1, "true", [], {}])
def test_real_receipt_requires_literal_true_control_flags(case, flag, bad):
    if bad == "missing":
        case.control.pop(flag)
    else:
        case.control[flag] = bad
    case.seal()
    refused(case, "missing literal receipt flag: "+flag)
    assert case.events == []


@pytest.mark.parametrize("bad", ["", "a"*63, "a"*65, "G"*64, "A"*64, "0"*64+"\n", None])
def test_invalid_control_digest_stops_before_helpers(case, bad):
    case.args.controls_sha = bad
    refused(case)
    assert case.events == []


def test_valid_but_wrong_control_digest_fails_real_watch(case):
    case.args.controls_sha = "0"*64
    refused(case, "hash mismatch")
    assert case.events == []


@pytest.mark.parametrize("field,value", [
    ("mode", "fixed"), ("arms", {"parameters": {}}), ("arms", []),
    ("controls", {"passed": False}), ("controls", {"passed": 1}),
    ("controls", {"passed": "true"}), ("controls", {}),
    ("full_f1_closure", True), ("full_f1_closure", 0), ("full_f1_closure", None),
])
def test_control_mode_and_scope_cannot_be_reinterpreted(case, field, value):
    case.control[field] = value
    case.seal()
    refused(case)
    assert case.events == []


@pytest.mark.parametrize("field", ["mode", "arms", "controls", "full_f1_closure", "artifacts", "execution",
                                    "keys", "schedule", "baseline_metadata", "native_time", "native_seed"])
def test_missing_control_metadata_fails_closed(case, field):
    case.control.pop(field)
    case.seal()
    refused(case)


@pytest.mark.parametrize("field,value", [("backend", "gpu"), ("x64", False),
                                        ("jax", "other-jax"), ("numpy", "other-numpy")])
def test_current_runtime_must_match_inherited_runtime(case, field, value):
    case.execution[field] = value
    refused(case, "control runtime differs")
    assert case.events == ["helpers"]


@pytest.mark.parametrize("field,value", [
    ("keys", [[1400, 0, 7033, 0], [1400, 0, 3700, 0]]),
    ("keys", [[1400, 0, 3700, 0]]),
    ("keys", [[1400, 1, 3700, 0], [1400, 0, 7033, 0]]),
    ("schedule", {"temperature": [1., 0.5], "maxniter": [3, 9]}),
    ("baseline_metadata", {"own_states": False, "time_model": "stock"}),
    ("native_time", {"provenance": "different", "n": 2}),
    ("native_seed", {"provenance": "captured", "n": 3}),
])
def test_prepared_keys_schedule_baseline_and_native_metadata_must_match(case, field, value):
    case.prepared[field] = value
    refused(case, "prepared inputs differ from controls")
    assert case.events == ["helpers", "prepare"]


@pytest.mark.parametrize("mutation", ["empty", "different_end", "missing_end", "relative_path", "bad_digest"])
def test_real_receipt_manifest_is_required_and_validated(case, mutation):
    if mutation == "empty":
        case.control["before_sha256"] = case.control["end_sha256"] = {}
    elif mutation == "different_end":
        case.control["end_sha256"][str(case.paths["source"])] = "0"*64
    elif mutation == "missing_end":
        case.control.pop("end_sha256")
    else:
        case.control["before_sha256"] = {"relative.py": "0"*64} if mutation == "relative_path" else {
            str(case.paths["source"]): "not-a-digest"}
        case.control["end_sha256"] = deepcopy(case.control["before_sha256"])
    case.seal()
    refused(case)
    assert case.events == []


@pytest.mark.parametrize("name", ["source", "runtime", "input", "control_artifact"])
@pytest.mark.parametrize("missing", [False, True])
def test_control_sources_runtime_inputs_and_artifacts_are_rehashed_before_execution(case, name, missing):
    if missing:
        case.paths[name].unlink()
    else:
        case.paths[name].write_bytes(b"drifted since completed control")
    refused(case, "FileNotFoundError" if missing else "hash mismatch")
    assert case.events == []


@pytest.mark.parametrize("text", ['{"completed":true,"completed":true}', '{"x":NaN}', '{"x":1e999}', '{'])
def test_control_json_is_parsed_strictly_after_authentication(case, text):
    case.args.controls.write_text(text)
    case.args.controls_sha = U.A.digest(case.args.controls)
    refused(case)
    assert case.events == []


@pytest.mark.parametrize("stage", ["helpers", "prepare"])
def test_runtime_and_preparation_exceptions_are_manifested_without_arms(case, monkeypatch, stage):
    def failure(*args):
        raise RuntimeError("deliberate "+stage)
    monkeypatch.setattr(U.R, stage, failure)
    report = refused(case, "RuntimeError: deliberate "+stage)
    assert report["hashes_unchanged"] is True


@pytest.mark.parametrize("failed_index", range(5))
def test_population_exception_stops_later_arms_and_preserves_completed_artifacts(case, monkeypatch, failed_index):
    def population(data, arm, out, report):
        if arm == ARMS[failed_index]:
            raise RuntimeError("deliberate arm failure")
        case.population(data, arm, out, report)
    monkeypatch.setattr(U.R, "population", population)
    report = U.run(case.args)
    assert report["completed"] is report["controls_passed"] is False
    assert report["error"] == "RuntimeError: deliberate arm failure"
    assert tuple(report["arms"]) == ARMS[:failed_index]
    assert tuple(report.get("artifacts", {})) == ARMS[:failed_index]
    assert case.calls == list(ARMS[:failed_index])
    assert report["hashes_unchanged"] is True
    manifest(case, report)


@pytest.mark.parametrize("inventory", [(), ARMS[:-1], ARMS[::-1], ARMS+("ordinary",)])
def test_incomplete_extra_or_reordered_completed_arm_inventory_is_refused(case, monkeypatch, inventory):
    def population(data, arm, out, report):
        if arm == ARMS[-1]:
            report["arms"] = {name: {} for name in inventory}
    monkeypatch.setattr(U.R, "population", population)
    report = U.run(case.args)
    assert report["completed"] is report["controls_passed"] is False
    assert "incomplete population inventory" in report["error"]
    manifest(case, report)


def test_upstream_new_arm_inventory_must_match_before_receipt_work(case, monkeypatch):
    monkeypatch.setattr(U.R, "ARMS", ARMS+("full",))
    refused(case, "new component inventory differs")
    assert case.events == []


@pytest.mark.parametrize("name", ["source", "runtime", "input", "control_artifact", "receipt", "new_artifact"])
@pytest.mark.parametrize("missing", [False, True])
def test_post_execution_source_and_artifact_drift_revokes_success(case, monkeypatch, name, missing):
    def population(data, arm, out, report):
        case.population(data, arm, out, report)
        if arm == ARMS[-1]:
            path = (case.args.controls if name == "receipt" else out/(ARMS[0]+".npz")
                    if name == "new_artifact" else case.paths[name])
            if missing:
                path.unlink()
            else:
                path.write_bytes(b"changed during population execution")
    monkeypatch.setattr(U.R, "population", population)
    report = U.run(case.args)
    assert report["hashes_unchanged"] is report["controls_passed"] is False
    assert case.calls == list(ARMS)
    if missing or name == "new_artifact":
        assert report["hash_errors"]
    manifest(case, report)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), object(), np.array([1.])])
def test_strict_serialization_failure_writes_failure_receipt_not_nan(case, monkeypatch, bad):
    def population(data, arm, out, report):
        case.population(data, arm, out, report)
        report["not_json"] = bad
    monkeypatch.setattr(U.R, "population", population)
    report = U.run(case.args)
    assert report["completed"] is report["controls_passed"] is False
    assert "strict JSON serialization failed" in report["error"]
    manifest(case, report)


def test_existing_output_directory_is_never_overwritten(case):
    case.args.out.mkdir()
    marker = case.args.out/"manifest.json"
    marker.write_bytes(b"owned existing receipt")
    with pytest.raises(FileExistsError):
        U.run(case.args)
    assert marker.read_bytes() == b"owned existing receipt"
    assert case.events == []


def test_second_run_preserves_all_first_run_outputs(case):
    report = U.run(case.args)
    manifest(case, report)
    before = {p: p.read_bytes() for p in case.args.out.iterdir()}
    with pytest.raises(FileExistsError):
        U.run(case.args)
    assert all(path.read_bytes() == value for path, value in before.items())
    assert case.calls == list(ARMS)


@pytest.mark.parametrize("symlink", [False, True])
def test_checkout_outputs_are_rejected_before_any_work(case, tmp_path, symlink):
    root = Path(U.__file__).resolve().parents[1]
    if symlink:
        alias = tmp_path/"checkout-alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    case.args.out = root/"never-created-inherited-runner-test-output"
    assert not case.args.out.exists()
    with pytest.raises(ValueError, match="protected checkout output"):
        U.run(case.args)
    assert not case.args.out.exists() and case.events == []


def test_manifest_creation_remains_exclusive_after_numerical_work(case, monkeypatch):
    sentinel = b"another writer owns this receipt"
    def population(data, arm, out, report):
        case.population(data, arm, out, report)
        if arm == ARMS[-1]:
            (out/"manifest.json").write_bytes(sentinel)
    monkeypatch.setattr(U.R, "population", population)
    with pytest.raises(FileExistsError):
        U.run(case.args)
    assert (case.args.out/"manifest.json").read_bytes() == sentinel


def test_arm_artifact_creation_remains_exclusive(case, monkeypatch):
    sentinel = b"another writer owns this artifact"
    def population(data, arm, out, report):
        (out/(arm+".npz")).write_bytes(sentinel)
        case.population(data, arm, out, report)
    monkeypatch.setattr(U.R, "population", population)
    report = U.run(case.args)
    assert report["completed"] is report["controls_passed"] is False
    assert "FileExistsError" in report["error"]
    assert (case.args.out/(ARMS[0]+".npz")).read_bytes() == sentinel
    assert report["arms"] == {}
    manifest(case, report)


@pytest.mark.parametrize("function", ["inherited", "main"])
def test_runner_rejects_foreign_callable_code_origins(case, monkeypatch, function):
    monkeypatch.setattr(U, function, lambda *args: None)
    refused(case, "function code from different checkout")
    assert case.events == []


@pytest.mark.parametrize("completed,passed,code", [(True, True, 0), (False, True, 1), (True, False, 1), (False, False, 1)])
def test_cli_resolves_paths_passes_receipt_pin_and_reports_exit_status(tmp_path, monkeypatch, capsys, completed, passed, code):
    fields = ("out", "controls", "seed-contract", "covariance-contract", "check", "conditions", "straw-axis", "baseline")
    argv = ["f1_seed_population_run.py", "--mode", "fixed", "--controls-sha", "a"*64]
    for field in fields:
        argv.extend(["--"+field, str(tmp_path/".."/tmp_path.name/field)])
    seen = []
    def run(args):
        seen.append(args)
        return dict(completed=completed, controls_passed=passed, hashes_unchanged=True)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(U, "run", run)
    with pytest.raises(SystemExit) as exit_info:
        U.main()
    assert exit_info.value.code == code
    assert len(seen) == 1 and seen[0].mode == "fixed" and seen[0].controls_sha == "a"*64
    for field in fields:
        assert getattr(seen[0], field.replace("-", "_")) == (tmp_path/field).resolve()
    printed = json.loads(capsys.readouterr().out)
    assert printed["completed"] is completed and printed["controls_passed"] is passed
