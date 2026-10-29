"""Bounded momentum trace guards, provenance, execution gates, and actual toy passivity."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import f1_momentum_trace as M
from test_f1_seed_population_kernel import exact_tree, inputs as seed_inputs


@pytest.mark.parametrize("values,expected", [([1], 3), ([0, 1, 3], 3), ([2, 4, 6], 3),
                                             ([3], 10), ([0, 0, 3], 10), ([2, 5, 5], 10)])
def test_cap_uses_stage_deltas_not_cumulative_total(values, expected):
    assert M.cap(np.array(values, dtype=np.int64)) == expected


@pytest.mark.parametrize("values", [[], [0], [0, 0], [-1], [1, 0], [2, 1, 3], [1.],
                                   [True], ["1"], [[1]], np.array(1),
                                   np.array([2**63], dtype=np.uint64),
                                   np.array([2**64-1], dtype=np.uint64)])
def test_cap_rejects_invalid_cumulative_history(values):
    with pytest.raises(ValueError):
        M.cap(values)


def forbidden(*args, **kwargs):
    pytest.fail("guard allowed forbidden downstream work")


@pytest.mark.parametrize("arm,cap,tracing", [("full", 3, False), (None, 3, False),
    ("momentum", None, True), ("momentum", True, True), ("momentum", 3., True),
    ("momentum", np.int64(3), True), ("momentum", 0, True), ("momentum", 4, True),
    ("momentum", 11, True), ("ordinary", 3, True), ("control", 3, True),
    ("momentum", 3, 1), ("momentum", 3, np.bool_(True))])
def test_kernel_rejects_invalid_options_before_numerics(monkeypatch, arm, cap, tracing):
    monkeypatch.setattr(M.F, "fit", forbidden)
    monkeypatch.setattr(M.F.S, "baseline", forbidden)
    monkeypatch.setattr(M.K, "fit_track", forbidden)
    with pytest.raises(ValueError):
        M.kernel(arm, None, None, None, None, None, None, None, None, maxniter=cap, tracing=tracing)


@pytest.mark.parametrize("arm", ["ordinary", "control", "momentum"])
@pytest.mark.parametrize("maxniter", [3, 10])
def test_untraced_kernel_delegates_complete_tree_and_final_stock_states(monkeypatch, arm, maxniter):
    h = dict(active=jnp.array([True, False]), lr=jnp.array([-1., 0.]))
    x, c, native, time, drift, stages, ann = [object() for _ in range(7)]
    result, calls = object(), []

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(M.F, "fit", fit)
    assert M.kernel(arm, x, h, c, native, time, drift, stages, ann, maxniter=maxniter) is result
    args, kwargs = calls.pop()
    assert len(args) == 9 and args[0] == arm
    assert all(a is b for a, b in zip(args[1:], (x, h, c, drift, stages, ann, native, time)))
    fixed = kwargs.pop("fixed")
    assert set(fixed) == {"active", "lr", "nulldv"}
    assert fixed["active"] is h["active"] and fixed["lr"] is h["lr"]
    np.testing.assert_array_equal(fixed["nulldv"], np.ones(2, dtype=bool))
    assert kwargs == {"maxniter": maxniter} and not calls


@pytest.mark.parametrize("size,maxniter", [(5, 3), (6, 10)])
def test_traced_kernel_selects_only_momentum_inside_same_jit(seed_inputs, monkeypatch, size, maxniter):
    original, original_select = M.F.S.baseline, M.F.select
    calls = []
    h = dict(seed_inputs["h"], active=np.array([True, False, True]), lr=np.array([1., 0., -1.]))
    c, native = seed_inputs["c"], seed_inputs["native"]
    drift, stages, ann = {}, {}, {}

    def baseline(x, hits, crossings):
        assert isinstance(x, jax.core.Tracer)
        assert isinstance(hits["mid"], jax.core.Tracer)
        base = original(x, hits, crossings)
        assert all(isinstance(v, jax.core.Tracer) for v in base.values())
        calls.append("baseline")
        return base

    def select(arm, base, n):
        assert arm == "momentum" and isinstance(n["parameters"], jax.core.Tracer)
        calls.append("select")
        return original_select(arm, base, n)

    def fit(x, hits, crossings, d, s, a, **kwargs):
        assert d is drift and s is stages and a is ann
        assert kwargs["fixed"]["active"] is hits["active"]
        assert kwargs["fixed"]["lr"] is hits["lr"]
        seed, time, fixed = kwargs.pop("seed_input"), kwargs.pop("constraint_time"), kwargs.pop("fixed")
        assert set(fixed) == {"active", "lr", "nulldv"}
        assert kwargs == dict(maxniter=maxniter, time_model="stock", reference_model="previous",
                              crossing_time="current", xing_activity="hit_or_doca", trace=True)
        calls.append("fit")
        return seed, time, fixed

    monkeypatch.setattr(M.F.S, "baseline", baseline)
    monkeypatch.setattr(M.F, "select", select)
    monkeypatch.setattr(M.F, "fit", forbidden)
    monkeypatch.setattr(M.K, "fit_track", fit)

    @jax.jit
    def caller(x, h, c, n, t):
        return M.kernel("momentum", x, h, c, n, t, drift, stages, ann,
                        maxniter=maxniter, tracing=True), original(x, h, c)

    x, time = seed_inputs["x"][:size], seed_inputs["constraint_time"]
    results = []
    for q in (x, x + np.array([2., 1., .03, .02, 7., 5.])[:size]):
        (seed, actual_time, fixed), base = caller(q, h, c, native, time)
        np.testing.assert_array_equal(seed["parameters"][:5], base["parameters"][:5])
        np.testing.assert_array_equal(seed["parameters"][5:], native["parameters"][5:])
        exact_tree(seed["covariance"], base["covariance"])
        exact_tree(seed["range"], base["range"])
        np.testing.assert_array_equal(actual_time, time)
        np.testing.assert_array_equal(fixed["nulldv"], np.ones(3, dtype=bool))
        results.append(base)
    assert calls == ["baseline", "select", "fit"]
    assert not np.array_equal(results[0]["range"], results[1]["range"])
    assert not np.array_equal(results[0]["covariance"], results[1]["covariance"])


def traced_result():
    valid = np.array([[True, True, False], [True, False, False]])
    records = dict(valid=valid, iteration=np.array([[0, 1, 999], [0, 999, 999]]),
                   swapped=np.array([[True, False, True], [False, True, True]]),
                   operand=np.arange(24.).reshape(2, 3, 4))
    return dict(niter=np.array(3), value=np.array([1., np.nan]), trace=records), (np.array([2, 3]),), np.array(False)


def test_executed_retains_rejections_ignores_speculation_and_does_not_mutate():
    result = traced_result()
    before = deepcopy(result)
    ordinary, kept = M.executed(result)
    assert "trace" not in ordinary[0] and "trace" in result[0]
    assert ordinary[1] is result[1] and ordinary[2] is result[2]
    exact_tree(result, before)
    assert not bool(ordinary[2])
    np.testing.assert_array_equal(kept["valid"], [True, True, True])
    np.testing.assert_array_equal(kept["swapped"], [True, False, False])
    np.testing.assert_array_equal(kept["iteration"], [0, 1, 0])
    np.testing.assert_array_equal(kept["stage_index"], [0, 0, 1])
    np.testing.assert_array_equal(kept["operand"], before[0]["trace"]["operand"][[0, 0, 1], [0, 1, 0]])
    assert set(kept) == set(before[0]["trace"]) | {"stage_index"}


@pytest.mark.parametrize("fault,message", [("missing", "missing trace"), ("ndim", "validity"),
    ("dtype", "validity"), ("empty", "validity"), ("count", "trace count"),
    ("first", "initial executed"), ("local", "local iteration"), ("prefix", "nonprefix")])
def test_executed_rejects_invalid_trace_contract(fault, message):
    result = traced_result()
    cf, _, _ = result
    records = cf["trace"]
    if fault == "missing":
        cf.pop("trace")
    elif fault == "ndim":
        records["valid"] = records["valid"].ravel()
    elif fault == "dtype":
        records["valid"] = records["valid"].astype(int)
    elif fault == "empty":
        records["valid"][:] = False
    elif fault == "count":
        cf["niter"] = 2
    elif fault == "first":
        records["valid"][:] = [[False, False, False], [True, True, True]]
        records["iteration"][:] = [0, 1, 2]
    elif fault == "local":
        records["iteration"][0, 1] = 0
    else:
        records["valid"][:] = [[True, False, True], [True, False, False]]
        records["iteration"][:] = [0, 1, 2]
    with pytest.raises(ValueError, match=message):
        M.executed(result)


def test_save_tree_preserves_every_leaf_and_maps_paths_exclusively(tmp_path):
    result = (dict(extra=np.array([np.nan, np.inf]), nested={"mask": np.array([False, True])}),
              (np.array([1, 3], dtype=np.int32),), np.array(False))
    report = {}
    M.save_tree(tmp_path, "control", result, report)
    artifact = report["artifacts"]["control"]
    leaves, _ = jax.tree_util.tree_flatten_with_path(result)
    assert artifact["path"] == str(tmp_path/"control.npz")
    assert artifact["sha256"] == M.A.digest(artifact["path"])
    assert artifact["leaf_paths"] == {f"leaf_{i:03d}": jax.tree_util.keystr(p) for i, (p, _) in enumerate(leaves)}
    with np.load(artifact["path"], allow_pickle=False) as saved:
        assert set(saved.files) == set(artifact["leaf_paths"])
        for i, (_, leaf) in enumerate(leaves):
            np.testing.assert_array_equal(saved[f"leaf_{i:03d}"], leaf)
            assert saved[f"leaf_{i:03d}"].dtype == leaf.dtype
    before, receipt = Path(artifact["path"]).read_bytes(), deepcopy(report)
    with pytest.raises(FileExistsError):
        M.save_tree(tmp_path, "control", (np.array(999),), report)
    assert Path(artifact["path"]).read_bytes() == before and report == receipt


@pytest.fixture
def execution_case(tmp_path, monkeypatch):
    events = []
    report = dict(maxniter=3, controls_passed=False, completed=False, passed=False)
    flags = dict(baseline=True, passive=True, compatible=True)
    data = dict(h={"valid": np.ones((1, 4), bool)}, c={"valid": np.ones((1, 5), bool)}, drift={}, stages={}, ann={})
    inputs, expected = (None, None, None, None, None), {"saved": True}
    identities = dict(key=np.array([1, 2, 3, 4]), hit_ids=np.array([7]), crossing_ids=np.array([8]),
                      seed_hit_mask=np.array([True]), crossing_partner=np.array([7]))
    save, save_tree = M.R.save, M.save_tree

    def kernel(arm, *args, tracing=False, **kwargs):
        events.append("kernel:"+arm+("_traced" if tracing else ""))
        assert kwargs == {"maxniter": 3}
        result = traced_result()
        if arm == "control" and not flags["baseline"] or tracing and not flags["passive"]:
            result[0]["value"] = np.array([2., np.nan])
        if not tracing:
            result[0].pop("trace")
        return result

    def save_records(out, name, values, receipt):
        events.append("save:"+name)
        save(out, name, values, receipt)

    def save_output(out, name, result, receipt):
        events.append("tree:"+name)
        save_tree(out, name, result, receipt)

    def compare(a, b):
        name = "baseline" if "kernel:momentum" not in events else "passive"
        events.append("compare:"+name)
        return original_compare(a, b)

    def project(result):
        events.append("project")
        return {"toy": np.array([2., 3.])}

    def expand(projected, nh, nx):
        assert (nh, nx) == (4, 5) and projected["toy"].shape == (1, 2)
        events.append("expand")
        return projected

    def payload(received, projected, rows):
        assert received is data
        np.testing.assert_array_equal(rows, [0])
        events.append("payload")
        return projected

    def compatibility(actual, saved):
        assert saved is expected and (tmp_path/"projected.npz").exists()
        events.append("compatibility")
        return {"passed": flags["compatible"], "exact": False}

    def native_first(*args):
        assert report["controls_passed"] is True
        assert {"ordinary", "control", "momentum", "momentum_traced_raw", "momentum_traced", "trace", "projected"} == set(report["artifacts"])
        events.append("native")
        return dict(iterations=object(), pieces=object(), effects=object())

    def compare_first(first, *args):
        assert first["valid"] and first["iteration"] == 0 and "stage_index" not in first
        events.append("first")
        return {"scope": "toy comparison"}

    original_compare = M.P.compare
    monkeypatch.setattr(M.jax, "jit", lambda f: f)
    monkeypatch.setattr(M.jax, "block_until_ready", lambda x: events.append("ready"))
    for obj, name, value in [(M, "kernel", kernel), (M.R, "save", save_records), (M, "save_tree", save_output),
                             (M.P, "compare", compare), (M.F, "project", project), (M.R, "expand", expand),
                             (M.R, "payload", payload), (M.R, "compatibility", compatibility),
                             (M.Q, "summary", lambda x: {"toy": True}), (M, "native_first", native_first),
                             (M.T, "compare_first", compare_first)]:
        monkeypatch.setattr(obj, name, value)
    return SimpleNamespace(events=events, report=report, flags=flags, out=tmp_path,
                           call=lambda out=tmp_path, receipt=report, execute=M.execute: execute(
                               data, inputs, expected, 0, identities, tmp_path/"capture", out, receipt))


def test_execute_preserves_controls_then_gates_scientific_comparison(execution_case):
    case = execution_case
    case.call()
    assert case.events == ["kernel:ordinary", "ready", "tree:ordinary", "save:ordinary",
        "kernel:control", "ready", "tree:control", "save:control", "compare:baseline",
        "kernel:momentum", "ready", "tree:momentum", "save:momentum",
        "kernel:momentum_traced", "ready", "tree:momentum_traced_raw", "save:momentum_traced_raw", "save:trace", "tree:momentum_traced", "save:momentum_traced",
        "compare:passive", "project", "expand", "payload", "save:projected", "compatibility", "native", "first"]
    assert case.report["completed"] is case.report["passed"] is case.report["controls_passed"] is True
    assert case.report["population_compatibility"] == {"passed": True, "exact": False}
    assert case.report["n_iterations"] == 3
    assert case.report["trace_sha256"] == case.report["artifacts"]["trace"]["sha256"]
    with np.load(case.out/"trace.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["valid"], [True, True, True])
        np.testing.assert_array_equal(saved["swapped"], [True, False, False])
        np.testing.assert_array_equal(saved["key"], [1, 2, 3, 4])


@pytest.mark.parametrize("flag,error,blocked,artifacts", [
    ("baseline", "ordinary/control differs", "kernel:momentum", {"ordinary", "control"}),
    ("passive", "trace changes momentum", "project", {"ordinary", "control", "momentum", "momentum_traced_raw", "momentum_traced", "trace"}),
    ("compatible", "compact execution differs", "native", {"ordinary", "control", "momentum", "momentum_traced_raw", "momentum_traced", "trace", "projected"})])
def test_execute_stops_at_first_failed_gate(execution_case, flag, error, blocked, artifacts):
    case = execution_case
    case.flags[flag] = False
    with pytest.raises(ValueError, match=error):
        case.call()
    assert blocked not in case.events and "first" not in case.events
    assert set(case.report["artifacts"]) == artifacts
    assert case.report["completed"] is case.report["passed"] is case.report["controls_passed"] is False


@pytest.fixture
def run_case(tmp_path, monkeypatch):
    source, capture = tmp_path/"source.py", tmp_path/"capture.root"
    source.write_bytes(b"sealed toy source")
    capture.write_bytes(b"toy capture")
    args = SimpleNamespace(out=tmp_path/"new-trace")
    events = []

    def helpers(watched):
        events.append("helpers")
        M.D.watch(source, watched)
        return {"backend": "cpu", "x64": True}

    def prepare(received, watched, report):
        assert received is not args and received.out == args.out.resolve()
        events.append("prepare")
        report["maxniter"] = 3
        return {}, (), {}, 0, {}, {"inputs": {"capture": str(capture)}}

    def execute(*values):
        out, report = values[-2:]
        events.append("execute")
        M.save_tree(out, "ordinary", (np.array([1., np.nan]),), report)
        report.update(completed=True, passed=True, controls_passed=True)

    monkeypatch.setattr(M, "helpers", helpers)
    monkeypatch.setattr(M, "prepare", prepare)
    monkeypatch.setattr(M, "execute", execute)
    monkeypatch.setattr(M.C, "CAPTURE_SHA", M.A.digest(capture))
    return SimpleNamespace(args=args, source=source, capture=capture, events=events, execute=execute)


def check_manifest(case, report):
    path = case.args.out/"manifest.json"
    assert M.D.strict_json(path) == report
    assert path.read_bytes().endswith(b"\n")
    assert all(report[name] is False for name in ("full_fit_closure", "full_f1_closure", "full_numerical_validation"))


def assert_saved_tree(report, name, result):
    artifact = report["artifacts"][name]
    assert M.A.digest(artifact["path"]) == artifact["sha256"]
    leaves, _ = jax.tree_util.tree_flatten_with_path(result)
    assert artifact["leaf_paths"] == {
        f"leaf_{i:03d}": jax.tree_util.keystr(path) for i, (path, _) in enumerate(leaves)}
    with np.load(artifact["path"], allow_pickle=False) as saved:
        assert set(saved.files) == set(artifact["leaf_paths"])
        for i, (_, leaf) in enumerate(leaves):
            np.testing.assert_array_equal(saved[f"leaf_{i:03d}"], leaf)
            assert saved[f"leaf_{i:03d}"].dtype == np.asarray(leaf).dtype


@pytest.mark.parametrize("fault,message", [("valid", "invalid/empty trace validity"),
    ("count", "trace count"), ("prefix", "nonprefix")])
def test_run_invalid_trace_preserves_raw_tree_and_manifest(execution_case, run_case, monkeypatch, fault, message):
    case, run = execution_case, run_case
    kernel = M.kernel
    raw = []

    def invalid_kernel(*args, **kwargs):
        result = kernel(*args, **kwargs)
        if kwargs.get("tracing"):
            records = result[0]["trace"]
            if fault == "valid":
                records["valid"] = records["valid"].astype(int)
            elif fault == "count":
                result[0]["niter"] = np.array(2)
            else:
                records["valid"][:] = [[True, False, True], [True, False, False]]
                records["iteration"][:] = [0, 1, 2]
            raw.append(deepcopy(result))
        return result

    monkeypatch.setattr(M, "kernel", invalid_kernel)
    monkeypatch.setattr(M, "execute", lambda *args: case.call(out=args[-2], receipt=args[-1]))
    report = M.run(run.args)
    assert report["completed"] is report["passed"] is report["controls_passed"] is False
    assert message in report["error"]
    assert case.events == ["kernel:ordinary", "ready", "tree:ordinary", "save:ordinary",
        "kernel:control", "ready", "tree:control", "save:control", "compare:baseline",
        "kernel:momentum", "ready", "tree:momentum", "save:momentum",
        "kernel:momentum_traced", "ready", "tree:momentum_traced_raw", "save:momentum_traced_raw"]
    assert set(report["artifacts"]) == {"ordinary", "control", "momentum", "momentum_traced_raw"}
    ordinary = traced_result()
    ordinary[0].pop("trace")
    for name in ("ordinary", "control", "momentum"):
        assert_saved_tree(report, name, ordinary)
    assert len(raw) == 1
    assert_saved_tree(report, "momentum_traced_raw", raw[0])
    assert report["hashes_unchanged"] is True and report["hash_errors"] == []
    assert report["before_sha256"] == report["end_sha256"]
    assert set(report["before_sha256"]) == {str(run.source), str(run.capture)}
    assert {p.name for p in run.args.out.iterdir()} == {
        "manifest.json", "ordinary.npz", "control.npz", "momentum.npz", "momentum_traced_raw.npz"}
    check_manifest(run, report)


def test_run_success_rehashes_inputs_outputs_and_preserves_sources(run_case):
    case = run_case
    before = {p: p.read_bytes() for p in (case.source, case.capture)}
    report = M.run(case.args)
    assert case.events == ["helpers", "prepare", "execute"]
    assert report["completed"] is report["passed"] is report["controls_passed"] is report["hashes_unchanged"] is True
    assert set(report["before_sha256"]) == {str(p) for p in before}
    assert report["before_sha256"] == report["end_sha256"] and report["hash_errors"] == []
    assert all(p.read_bytes() == contents for p, contents in before.items())
    artifact = report["artifacts"]["ordinary"]
    assert M.A.digest(artifact["path"]) == artifact["sha256"]
    check_manifest(case, report)


@pytest.mark.parametrize("stage", ["helpers", "prepare", "execute"])
def test_run_records_failure_without_later_work(run_case, monkeypatch, stage):
    case = run_case

    def failure(*args):
        raise RuntimeError("deliberate "+stage)

    monkeypatch.setattr(M, stage, failure)
    report = M.run(case.args)
    assert report["completed"] is report["passed"] is report["controls_passed"] is False
    assert report["error"] == "RuntimeError: deliberate "+stage
    assert case.events == ["helpers", "prepare"][:["helpers", "prepare", "execute"].index(stage)]
    assert report["hashes_unchanged"] is (stage != "helpers")
    check_manifest(case, report)


@pytest.mark.parametrize("missing", [False, True])
def test_run_authenticates_capture_before_execution(run_case, missing):
    case = run_case
    if missing:
        case.capture.unlink()
    else:
        case.capture.write_bytes(b"wrong capture")
    report = M.run(case.args)
    assert report["completed"] is report["passed"] is report["controls_passed"] is False
    assert "execute" not in case.events
    assert ("FileNotFoundError" if missing else "hash mismatch") in report["error"]
    check_manifest(case, report)


@pytest.mark.parametrize("target", ["source", "capture", "artifact"])
@pytest.mark.parametrize("missing", [False, True])
def test_run_end_hash_drift_revokes_success(run_case, monkeypatch, target, missing):
    case = run_case

    def execute(*args):
        case.execute(*args)
        path = case.args.out/"ordinary.npz" if target == "artifact" else getattr(case, target)
        if missing:
            path.unlink()
        else:
            path.write_bytes(b"changed during execution")

    monkeypatch.setattr(M, "execute", execute)
    report = M.run(case.args)
    assert report["passed"] is report["controls_passed"] is False
    if target != "artifact":
        assert report["hashes_unchanged"] is False
    if missing or target == "artifact":
        assert report["hash_errors"]
    check_manifest(case, report)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), object(), np.array([1.])])
def test_run_serialization_failure_preserves_artifacts_and_provenance(run_case, monkeypatch, bad):
    case = run_case
    raw = traced_result()
    _, records = M.executed(raw)
    provenance = dict(key=[1, 2, 3, 4], maxniter=3, n_hits=4, n_crossings=5,
        n_iterations=3, trace_bytes_bound=4096, population_manifest="saved/manifest.json",
        population_artifact={"path": "saved/momentum.npz", "sha256": "a" * 64},
        comparison_scope="toy compact single-target compiler context", inputs={"capture": str(case.capture)})
    saved_artifacts = {}

    def execute(*args):
        case.execute(*args)
        out, report = args[-2:]
        M.save_tree(out, "momentum_traced_raw", raw, report)
        M.R.save(out, "trace", records, report)
        saved_artifacts.update(deepcopy(report["artifacts"]))
        report.update(provenance, trace_sha256=report["artifacts"]["trace"]["sha256"])
        report["first_iteration_comparison"] = {"nested_diagnostic": [bad]}

    monkeypatch.setattr(M, "execute", execute)
    monkeypatch.setattr(M.D, "serialize", forbidden)
    report = M.run(case.args)
    assert report["completed"] is report["passed"] is report["controls_passed"] is False
    assert "strict JSON serialization failed" in report["error"]
    assert "first_iteration_comparison" not in report
    assert report["artifacts"] == saved_artifacts
    assert set(report["artifacts"]) == {"ordinary", "momentum_traced_raw", "trace"}
    assert_saved_tree(report, "ordinary", (np.array([1., np.nan]),))
    assert_saved_tree(report, "momentum_traced_raw", raw)
    assert report["trace_sha256"] == M.A.digest(case.args.out/"trace.npz") == saved_artifacts["trace"]["sha256"]
    with np.load(case.args.out/"trace.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(records)
        for name, value in records.items():
            np.testing.assert_array_equal(saved[name], value)
    assert all(report[name] == value for name, value in provenance.items())
    assert report["limits"] == M.LIMITS
    assert report["execution"] == {"backend": "cpu", "x64": True}
    assert report["hashes_unchanged"] is True and report["hash_errors"] == []
    assert report["before_sha256"] == report["end_sha256"] == {
        str(path): M.A.digest(path) for path in (case.source, case.capture)}
    assert {p.name for p in case.args.out.iterdir()} == {
        "manifest.json", "ordinary.npz", "momentum_traced_raw.npz", "trace.npz"}
    check_manifest(case, report)


@pytest.mark.parametrize("completed_first", [False, True])
def test_run_output_directory_creation_is_exclusive(run_case, completed_first):
    case = run_case
    if completed_first:
        M.run(case.args)
    else:
        case.args.out.mkdir()
        (case.args.out/"manifest.json").write_bytes(b"existing owner")
    before = {p: p.read_bytes() for p in case.args.out.iterdir()}
    events = list(case.events)
    with pytest.raises(FileExistsError):
        M.run(case.args)
    assert case.events == events and all(p.read_bytes() == b for p, b in before.items())


@pytest.mark.parametrize("symlink", [False, True])
def test_run_refuses_checkout_outputs_before_work(run_case, tmp_path, symlink):
    root = Path(M.__file__).resolve().parents[1]
    if symlink:
        alias = tmp_path/"checkout-alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    run_case.args.out = root/"never-created-momentum-trace-test"
    assert not run_case.args.out.exists()
    with pytest.raises(ValueError, match="protected checkout output"):
        M.run(run_case.args)
    assert run_case.events == [] and not run_case.args.out.exists()


def test_run_manifest_write_is_exclusive_even_after_execution(run_case, monkeypatch):
    case = run_case

    def execute(*args):
        case.execute(*args)
        (case.args.out/"manifest.json").write_bytes(b"another writer")

    monkeypatch.setattr(M, "execute", execute)
    with pytest.raises(FileExistsError):
        M.run(case.args)
    assert (case.args.out/"manifest.json").read_bytes() == b"another writer"


def test_helpers_watch_local_trace_comparator_and_allocation_sources(monkeypatch):
    monkeypatch.setattr(M.R, "helpers", lambda watched: {"toy_runtime": True})
    watched = {}
    assert M.helpers(watched) == {"toy_runtime": True}
    expected = {str(Path(module.__file__).resolve()) for module in (M, M.T, M.fittrace, M.C)}
    assert expected <= set(watched)
    assert all(M.A.digest(path) == watched[path] for path in expected)


def test_helpers_reject_foreign_callable_origin(monkeypatch):
    monkeypatch.setattr(M.R, "helpers", lambda watched: {})
    monkeypatch.setattr(M, "kernel", lambda *args: None)
    with pytest.raises(ValueError, match="function code from different checkout"):
        M.helpers({})


@pytest.mark.slow
@pytest.mark.parametrize("reject", [False, True])
def test_actual_fixed_momentum_trace_and_baseline_control_are_complete_tree_exact(reject):
    from test_kalman_jax import _anns_available, _toy_track

    if not _anns_available():
        pytest.skip("ANN weights unavailable")
    x, _, h, c, _, drift = _toy_track(3)
    h = dict(h, inseed=jnp.arange(len(h["valid"])) < 15)
    x = jnp.asarray(x) + jnp.array([1., -1., .003, .002, .5])
    base = M.F.S.baseline(x, h, c)
    native = dict(parameters=base["parameters"] + jnp.array([10., -20., .1, -.2, 30., .0002]),
                  covariance=base["covariance"] * 2, range=base["range"] + 1000.)
    time = jnp.mean(base["range"]) + .2
    stages = {k: v[:3] for k, v in M.K.stock_stages().items()}
    if reject:
        stages["minndof"] = jnp.full_like(stages["minndof"], 10000)
    ann = M.K.load_anns()

    def run(arm, tracing=False):
        return jax.jit(lambda p, n, t: M.kernel(arm, p, h, c, n, t, drift, stages, ann,
                                               maxniter=3, tracing=tracing))(x, native, time)

    ordinary, control = run("ordinary"), run("control")
    plain, traced = run("momentum"), run("momentum", True)
    jax.block_until_ready((ordinary, control, plain, traced))
    exact_tree(control, ordinary)
    stripped, kept = M.executed(traced)
    exact_tree(stripped, plain)
    assert len(kept["valid"]) == int(plain[0]["niter"]) > 0
    assert np.all(kept["valid"])
    np.testing.assert_array_equal(kept["owner_parameters"][0, 0, :5], x)
    np.testing.assert_array_equal(kept["owner_parameters"][0, 0, 5], native["parameters"][5])
    if reject:
        assert not bool(plain[2]) and len(kept["valid"]) == 1
        assert not np.any(kept["swapped"])
        assert int(plain[0]["status"]) == M.K.LOWNDOF
    else:
        assert np.any(kept["swapped"])
