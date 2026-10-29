"""Local probe guards and synthetic dispatch only: no native data or actual fits.

These tests certify diagnostic mechanics, not physics or full-fit/F1 closure.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_seed_arm_probe as P


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    for name in tuple(P.os.environ):
        if name.startswith("F1_"):
            monkeypatch.delenv(name)


def arguments(tmp_path):
    return SimpleNamespace(**{name: tmp_path/name for name in
                              ("out", "check", "conditions", "straw_axis", "seed_contract")},
                           seed_contract_sha="0"*64, key=P.identity("12:0:34:0"), fixed=False)


@pytest.mark.parametrize("text,expected", [("12:0:34:0", [12, 0, 34, 0]),
                                          ("001:02:003:0", [1, 2, 3, 0])])
def test_identity(text, expected):
    key = P.identity(text)
    assert key.dtype == np.int64 and key.shape == (4,)
    np.testing.assert_array_equal(key, expected)


@pytest.mark.parametrize("text", ["", "1:2:3", "1:2:3:0:0", "0:2:3:0", "1:2:0:0",
                                  "1:2:3:1", "-1:2:3:0", "1:-2:3:0", "+1:2:3:0",
                                  "1:2:3.0:0", "1: 2:3:0", "1:2:3:0\n", "1::3:0",
                                  "9223372036854775808:0:1:0"])
def test_identity_rejects_invalid(text):
    with pytest.raises((ValueError, OverflowError)):
        P.identity(text)


def result_tree():
    return ({"niter": np.array(0, dtype=np.int32), "status": np.array(7, dtype=np.int32),
             "parameters": np.array([1., np.nan, np.inf, -np.inf]),
             "owner": {"range": np.array([-2., 9.]), "index": np.array([0, 2])}},
            {"steps": np.array([[np.nan, 1.], [np.inf, -np.inf]]),
             "covariance": np.eye(2)}, np.array(False))


def test_compare_preserves_exact_nonfinite_tree():
    result = result_tree()
    report = P.compare(result, deepcopy(result))
    assert report == dict(exact=True, unequal_fields=[],
                          actual_nonfinite_fields=["[0]['parameters']", "[1]['steps']"],
                          n_leaves=8)
    assert np.isnan(result[0]["parameters"][1])
    assert np.isposinf(result[0]["parameters"][2])
    assert np.isneginf(result[0]["parameters"][3])


@pytest.mark.parametrize("change,field", [
    ("roundoff", "[1]['covariance']"), ("dtype", "[0]['niter']"),
    ("owner", "[0]['owner']['range']"), ("history", "[1]['steps']"),
    ("nan_position", "[0]['parameters']"), ("inf_sign", "[0]['parameters']"),
    ("good", "[2]"), ("shape", "[0]['owner']['index']"),
])
def test_compare_all_fields_without_tolerance(change, field):
    expected, actual = result_tree(), result_tree()
    if change == "roundoff":
        actual[1]["covariance"][0, 0] = np.nextafter(1., 2.)
    elif change == "dtype":
        actual[0]["niter"] = actual[0]["niter"].astype(np.int64)
    elif change == "owner":
        actual[0]["owner"]["range"][0] = -3.
    elif change == "history":
        actual[1]["steps"][0, 1] = 2.
    elif change == "nan_position":
        actual[0]["parameters"][[0, 1]] = actual[0]["parameters"][[1, 0]]
    elif change == "inf_sign":
        actual[0]["parameters"][2] = -np.inf
    elif change == "good":
        actual = (*actual[:2], np.array(True))
    else:
        actual[0]["owner"]["index"] = np.array([[0, 2]])
    comparison = P.compare(actual, expected)
    assert comparison["exact"] is False
    assert comparison["unequal_fields"] == [field]


@pytest.mark.parametrize("change", ["missing", "extra", "list"])
def test_compare_rejects_tree_structure_change(change):
    expected, actual = result_tree(), result_tree()
    if change == "missing":
        del actual[0]["owner"]
    elif change == "extra":
        actual[0]["extra"] = np.array(0)
    else:
        actual = list(actual)
    with pytest.raises(ValueError, match="structures differ"):
        P.compare(actual, expected)


def test_helpers_pin_fitter_and_watch_local_sources():
    watched = {}
    metadata = P.helpers(watched)
    root = Path(P.__file__).resolve().parents[1]
    assert metadata["fitter_sha256"] == P.FITTER_SHA
    assert watched[str(root/"trkfastsim/reco/kalman_jax.py")] == P.FITTER_SHA
    for name in ("scripts/f1_seed_arms.py", "scripts/f1_seed_arm_probe.py"):
        assert watched[str(root/name)] == P.A.digest(root/name)
    for path in (root/"trkfastsim/reco").glob("*.py"):
        assert watched[str(path)] == P.A.digest(path)
    assert metadata["jax_version"] == P.jax.__version__
    assert metadata["numpy_version"] == np.__version__


@pytest.mark.parametrize("module,name", [(P.S, name) for name in
                                         ("validate", "baseline", "select", "fit")]
                         + [(P, name) for name in ("identity", "compare", "prepare", "execute", "run")])
def test_helpers_reject_foreign_required_code(monkeypatch, module, name):
    monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="function code"):
        P.helpers({})


@pytest.mark.parametrize("module", [P.S, P, P.nativetime, P.vr_f1])
def test_helpers_reject_foreign_module_path(monkeypatch, module):
    root = Path(P.__file__).resolve().parents[1]
    monkeypatch.setattr(module, "__file__", str(root/"scripts/foreign-module.py"))
    with pytest.raises(ValueError):
        P.helpers({})


@pytest.mark.parametrize("module", [P.S, P.nativeseed])
def test_helpers_reject_fitter_rebinding(monkeypatch, module):
    monkeypatch.setattr(module, "K", SimpleNamespace())
    with pytest.raises(ValueError, match="fitter binding differs"):
        P.helpers({})


def test_helpers_reject_spoofed_reco_function(monkeypatch):
    def foreign(*args, **kwargs):
        return None
    foreign.__module__ = P.K.__name__
    monkeypatch.setattr(P.K, "fit_track", foreign)
    with pytest.raises(ValueError, match="function code"):
        P.helpers({})


@pytest.fixture
def receipt_inputs(tmp_path):
    args = arguments(tmp_path)
    baseline = tmp_path/"baseline.bin"
    baseline.write_bytes(b"synthetic baseline marker, not native data")
    capture = tmp_path/"capture.bin"
    capture.write_bytes(b"synthetic capture marker, not native data")
    args.conditions.write_text("synthetic conditions")
    args.straw_axis.write_text("synthetic axis")
    args.check.write_text(json.dumps({"inputs": {"baseline": {"path": str(baseline)}}}))
    manifest = {str(p): P.A.digest(p) for p in (baseline, capture)}
    document = dict(completed=True, passed=True, hashes_unchanged=True, full_f1_closure=False,
                    diagnostics={"n_seeds": 2319},
                    targets={"n_targets": 147, "generated": {"keys": [args.key.tolist()]}},
                    before_sha256=manifest, end_sha256=manifest.copy())
    return args, baseline, capture, document


def save_receipt(args, document):
    args.seed_contract.write_text(json.dumps(document))
    args.seed_contract_sha = P.A.digest(args.seed_contract)


def forbid_datawork(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("data/fit work occurred before receipt gate rejection")
    for module, name in ((P.hitreco, "load_conditions"), (P.vr_f1, "read_inputs"),
                         (P.nativetime, "load_times"), (P.nativeseed, "load_seed"),
                         (P.K, "stock_stages"), (P.K, "load_anns"), (P.S, "validate"),
                         (P.S, "fit")):
        monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize("change", ["completed", "passed", "hashes_unchanged", "numeric_flag",
                                    "population", "target_count", "scope", "missing_scope", "target",
                                    "hash", "malformed_hash", "baseline_drift", "capture_drift",
                                    "unbound_baseline", "manifest", "environment"])
def test_receipt_gate_precedes_all_datawork(receipt_inputs, monkeypatch, change):
    args, baseline, capture, document = receipt_inputs
    if change in ("completed", "passed", "hashes_unchanged"):
        document[change] = False
    elif change == "numeric_flag":
        document["passed"] = 1
    elif change == "population":
        document["diagnostics"]["n_seeds"] -= 1
    elif change == "target_count":
        document["targets"]["n_targets"] -= 1
    elif change == "scope":
        document["full_f1_closure"] = True
    elif change == "missing_scope":
        del document["full_f1_closure"]
    elif change == "target":
        args.key[2] += 1
    elif change == "unbound_baseline":
        del document["before_sha256"][str(baseline)]
        del document["end_sha256"][str(baseline)]
    elif change == "manifest":
        document["end_sha256"] = {}
    save_receipt(args, document)
    if change == "hash":
        args.seed_contract_sha = "f"*64
    elif change == "malformed_hash":
        args.seed_contract_sha = "A"*64
    elif change.endswith("_drift"):
        (baseline if change == "baseline_drift" else capture).write_text("changed")
    elif change == "environment":
        monkeypatch.setenv("F1_UNDECLARED_POLICY", "1")
    forbid_datawork(monkeypatch)
    with pytest.raises(ValueError):
        P.prepare(args, {}, {})


@pytest.mark.parametrize("fixed", [False, True])
def test_prepare_one_target_own_or_fixed_routing(receipt_inputs, monkeypatch, fixed):
    args, baseline, capture, document = receipt_inputs
    args.fixed = fixed
    save_receipt(args, document)
    # Two eligible rows ensure that prepare selects, trims and routes only the requested row.
    x = np.array([[2., 3., .1, .2, 4., 900.], [5., 6., .3, .4, 7., 1000.]])
    h = {key: np.zeros((2, 3)) for key in P.HIT_KEYS}
    h.update(valid=np.array([[True, False, False], [True, True, False]]),
             inseed=np.array([[True, False, False], [True, False, False]]),
             active=np.array([[True, False, False], [False, True, False]]),
             lr=np.array([[1, 0, 0], [-1, 1, 0]]), mid=np.zeros((2, 3, 3)),
             wdir=np.tile([1., 0., 0.], (2, 3, 1)))
    c = dict(valid=np.array([[True, False], [True, False]]), hit=np.array([[0, -1], [0, -1]]),
             mid=np.zeros((2, 2, 3)), wdir=np.tile([1., 0., 0.], (2, 2, 1)))
    ref = dict(zip(P.B.KEYS[:3], (np.array([12, 12]), np.array([0, 0]), np.array([33, 34]))))
    ref["kl_status"] = np.array([0, P.vr_f1.FITOK])
    drift = {key: np.array([1., 2.]) for key in P.DRIFT_KEYS}
    q, covariance, ranges = x[1:2].copy(), np.eye(6)[None], np.array([[-1., 20.]])
    meta = {"inputs": {"capture": {"path": str(capture), "sha256": P.A.digest(capture)}}}
    seed_meta = {"synthetic": True}
    events = []
    condition_marker, ann = object(), object()
    monkeypatch.setattr(P.hitreco, "load_conditions", lambda path: condition_marker)
    def read_inputs(path, cond, mode, **kwargs):
        assert path == str(baseline) and cond is condition_marker and mode == "mc"
        assert kwargs == dict(straw_axis=str(args.straw_axis), fitok_only=False)
        events.append("read")
        return x, h, c, ref, drift, 2
    monkeypatch.setattr(P.vr_f1, "read_inputs", read_inputs)
    def native_loader(check, path, keys):
        assert check == args.check and path == baseline
        np.testing.assert_array_equal(keys, args.key[None])
        return np.array([9.125]), meta
    monkeypatch.setattr(P.nativetime, "load_times", native_loader)
    def seed_loader(check, path, keys):
        native_loader(check, path, keys)
        return q, covariance, ranges, seed_meta
    monkeypatch.setattr(P.nativeseed, "load_seed", seed_loader)
    original_validate = P.S.validate
    def validate(*values):
        events.append("validate")
        return original_validate(*values)
    monkeypatch.setattr(P.S, "validate", validate)
    ann_file = baseline.parent/"synthetic-ann.dat"
    ann_file.write_text("not a trained network")
    monkeypatch.setattr(P.driftann, "find", lambda name: ann_file)
    stages = {"temperature": np.array([4., 1.])}
    monkeypatch.setattr(P.K, "stock_stages", lambda: stages)
    def load_anns():
        assert events == ["read", "validate"]
        return ann
    monkeypatch.setattr(P.K, "load_anns", load_anns)
    report, watched = {}, {}
    inputs = P.prepare(args, watched, report)
    np.testing.assert_array_equal(inputs["x"], x[1])
    for key in P.HIT_KEYS+("valid", "inseed"):
        np.testing.assert_array_equal(inputs["hits"][key], h[key][1, :2])
    for key in c:
        np.testing.assert_array_equal(inputs["crossings"][key], c[key][1, :1])
    for key, value in zip(("parameters", "covariance", "range"), (q[0], covariance[0], ranges[0])):
        np.testing.assert_array_equal(inputs["native"][key], value)
    assert float(inputs["constraint_time"]) == 9.125
    assert inputs["stages"] is stages and inputs["ann"] is ann
    if fixed:
        assert set(inputs["fixed"]) == {"active", "lr", "nulldv"}
        np.testing.assert_array_equal(inputs["fixed"]["active"], h["active"][1, :2])
        np.testing.assert_array_equal(inputs["fixed"]["lr"], h["lr"][1, :2])
        np.testing.assert_array_equal(inputs["fixed"]["nulldv"], np.ones_like(h["active"][1, :2]))
    else:
        assert inputs["fixed"] is None
    assert report["key"] == args.key.tolist()
    assert (report["n_eligible"], report["n_hits"], report["n_crossings"]) == (2, 2, 1)
    assert report["fixed_states"] is fixed and report["stock_fitok"] is True
    assert watched[str(capture)] == P.A.digest(capture)
    assert watched[str(args.seed_contract)] == args.seed_contract_sha


@pytest.fixture
def dispatch_inputs():
    return dict(x=np.arange(6.), hits={"sentinel": np.array([1])},
                crossings={"sentinel": np.array([2])}, drift=object(), stages=object(), ann=object(),
                native={"parameters": np.arange(6.)+10., "covariance": np.eye(6)*2.,
                        "range": np.array([-10., 100.])},
                constraint_time=np.array(8.25), fixed=None)


def immediate_dispatch(monkeypatch):
    # The orchestration is under test, not XLA or the already separately tested fitter.
    monkeypatch.setattr(P.jax, "jit", lambda function: function)
    monkeypatch.setattr(P.jax, "block_until_ready", lambda value: value)


@pytest.mark.parametrize("fixed", [False, True])
def test_execute_dispatch_preserves_policies_and_nonfinite_artifacts(tmp_path, monkeypatch, dispatch_inputs, fixed):
    inputs = dispatch_inputs
    if fixed:
        inputs["fixed"] = {"active": np.array([True]), "lr": np.array([-1]), "nulldv": np.array([True])}
    immediate_dispatch(monkeypatch)
    base = {"parameters": np.arange(6.), "covariance": np.eye(6), "range": np.array([1., 2.])}
    monkeypatch.setattr(P.S, "baseline", lambda *args: base)
    calls, report = [], {}
    expected_arms = ("ordinary", "control", "parameters", "covariance", "range", "full")
    def fit_track(x, h, c, drift, stages, ann, **kwargs):
        index = len(calls)
        arm = expected_arms[index]
        if index >= 2:
            assert report["arms"]["control"]["comparison_to_ordinary"]["exact"] is True
        for actual, key in zip((x, h, c, drift, stages, ann), ("x", "hits", "crossings", "drift", "stages", "ann")):
            assert actual is inputs[key]
        assert kwargs["fixed"] is inputs["fixed"]
        assert kwargs["constraint_time"] is inputs["constraint_time"]
        assert {k: kwargs[k] for k in ("maxniter", "time_model", "reference_model", "crossing_time", "xing_activity")} == dict(
            maxniter=10, time_model="stock", reference_model="previous", crossing_time="current", xing_activity="hit_or_doca")
        seed = kwargs["seed_input"]
        if arm == "ordinary":
            assert seed is None
        else:
            for key in base:
                assert seed[key] is (inputs["native"] if arm in (key, "full") else base)[key]
        calls.append(arm)
        result = result_tree()
        if arm == "range":
            result[0]["owner"]["range"] = np.array([-10., 100.])
        return result
    monkeypatch.setattr(P.K, "fit_track", fit_track)
    P.execute(inputs, tmp_path, report)
    assert tuple(calls) == expected_arms
    assert report["completed"] is True and report["controls_passed"] is True
    assert report["arms"]["range"]["comparison_to_ordinary"]["exact"] is False
    assert report["arms"]["range"]["n_iterations"] == 0
    for arm, values in report["arms"].items():
        assert values["good"] is False  # Failure/nonfiniteness is data, not repaired success.
        artifact = tmp_path/(arm+".npz")
        assert values["artifact"] == str(artifact) and values["sha256"] == P.A.digest(artifact)
        expected = result_tree()
        if arm == "range":
            expected[0]["owner"]["range"] = np.array([-10., 100.])
        leaves, _ = P.jax.tree_util.tree_flatten_with_path(expected)
        with np.load(artifact, allow_pickle=False) as stored:
            assert set(stored.files) == set(values["leaf_paths"])
            for index, (path, value) in enumerate(leaves):
                name = f"leaf_{index:03d}"
                assert values["leaf_paths"][name] == P.jax.tree_util.keystr(path)
                assert stored[name].dtype == value.dtype
                np.testing.assert_array_equal(stored[name], value)
    # Nonfinite operands belong in NPZ, not invalid JSON constants in the report.
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("field", ["history", "owner", "dtype", "nan", "good"])
def test_execute_rejects_control_before_any_component_arm(tmp_path, monkeypatch, dispatch_inputs, field):
    immediate_dispatch(monkeypatch)
    calls, report = [], {}
    def fit(arm, *args, **kwargs):
        calls.append(arm)
        assert arm in ("ordinary", "control"), "component arm executed before control validation"
        result = result_tree()
        if arm == "control":
            if field == "history":
                result[1]["covariance"][0, 0] = np.nextafter(1., 2.)
            elif field == "owner":
                result[0]["owner"]["range"][0] -= 1.
            elif field == "dtype":
                result[0]["niter"] = np.array(0, dtype=np.int64)
            elif field == "nan":
                result[0]["parameters"][1] = 0.
            else:
                result = (*result[:2], np.array(True))
        return result
    monkeypatch.setattr(P.S, "fit", fit)
    with pytest.raises(ValueError, match="baseline injection control differs"):
        P.execute(dispatch_inputs, tmp_path, report)
    assert calls == ["ordinary", "control"]
    assert set(report["arms"]) == {"ordinary", "control"}
    assert report["arms"]["control"]["comparison_to_ordinary"]["exact"] is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["control.npz", "ordinary.npz"]
    assert not report.get("completed", False)


@pytest.mark.parametrize("arms", [(), ("ordinary",), ("ordinary", "parameters"), tuple(reversed(P.S.ARMS))])
def test_execute_rejects_changed_inventory_before_fitting(tmp_path, monkeypatch, dispatch_inputs, arms):
    monkeypatch.setattr(P.S, "ARMS", arms)
    monkeypatch.setattr(P.S, "fit", lambda *a, **k: pytest.fail("fit before inventory check"))
    with pytest.raises(ValueError, match="inventory"):
        P.execute(dispatch_inputs, tmp_path, {})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name,value", [("SCATSCALE", 0.), ("ELOSS", 0.), ("INV", "chol"),
                                        ("PARALLEL", True), ("FIXED_NULLDV_ALL", True), ("UNROLL", 1)])
def test_helpers_reject_stale_import_settings_without_environment(monkeypatch, name, value):
    monkeypatch.setattr(P.K, name, value)
    with pytest.raises(ValueError, match="fitter constants differ"):
        P.helpers({})


def test_helpers_reject_stale_material_setting(monkeypatch):
    monkeypatch.setattr(P.K.kkmaterial, "GAS_SCATTERFRAC", .9999999)
    with pytest.raises(ValueError, match="material settings"):
        P.helpers({})


def test_helpers_reject_mutated_material_cache(monkeypatch):
    cache = dict(P.K.kkmaterial._MC)
    cache["straw-wall"] = dict(cache["straw-wall"], density=99.)
    monkeypatch.setattr(P.K.kkmaterial, "_MC", cache)
    with pytest.raises(ValueError, match="cached constants"):
        P.helpers({})


def test_execute_artifacts_are_exclusive(tmp_path, monkeypatch, dispatch_inputs):
    immediate_dispatch(monkeypatch)
    monkeypatch.setattr(P.S, "fit", lambda *args, **kwargs: result_tree())
    artifact = tmp_path/"ordinary.npz"
    artifact.write_bytes(b"preserved existing evidence")
    with pytest.raises(FileExistsError):
        P.execute(dispatch_inputs, tmp_path, {})
    assert artifact.read_bytes() == b"preserved existing evidence"
    assert not (tmp_path/"control.npz").exists()


def assert_failure_scope(report):
    assert report["controls_passed"] is False
    for flag in ("full_fit_closure", "full_f1_closure", "full_numerical_validation"):
        assert report[flag] is False


def test_missing_receipt_writes_strict_exclusive_failure_json(tmp_path):
    args = arguments(tmp_path)
    report = P.run(args)
    assert report["completed"] is False and "FileNotFoundError" in report["error"]
    assert_failure_scope(report)
    assert report["hashes_unchanged"] is True and report["hash_errors"] == []
    assert report["before_sha256"] == report["end_sha256"]
    manifest = args.out/"manifest.json"
    before = manifest.read_bytes()
    assert P.D.strict_json(manifest) == report
    with pytest.raises(FileExistsError):
        P.run(args)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("existing", ["directory", "file"])
def test_existing_output_rejected_before_helpers(tmp_path, monkeypatch, existing):
    args = arguments(tmp_path)
    if existing == "directory":
        args.out.mkdir()
        evidence = args.out/"evidence"
    else:
        evidence = args.out
    evidence.write_bytes(b"do not replace")
    monkeypatch.setattr(P, "helpers", lambda *args: pytest.fail("helper ran before output exclusivity"))
    with pytest.raises(FileExistsError):
        P.run(args)
    assert evidence.read_bytes() == b"do not replace"


@pytest.mark.parametrize("symlink", [False, True])
def test_checkout_output_rejected_before_helpers(tmp_path, monkeypatch, symlink):
    args = arguments(tmp_path)
    root = Path(P.__file__).resolve().parents[1]
    if symlink:
        link = tmp_path/"checkout-link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    args.out = root/"seed-arm-probe-must-not-be-created"
    monkeypatch.setattr(P, "helpers", lambda *args: pytest.fail("protected checkout reached helpers"))
    with pytest.raises(ValueError, match="protected checkout output"):
        P.run(args)
    assert not args.out.exists()


@pytest.mark.parametrize("change", ["edit", "delete"])
@pytest.mark.parametrize("complete", [False, True])
def test_hash_drift_invalidates_controls_and_retains_evidence(tmp_path, monkeypatch, change, complete):
    args = arguments(tmp_path)
    source = tmp_path/"watched-source"
    source.write_text("before")
    digest = P.A.digest(source)
    def helpers(watched):
        P.D.watch(source, watched)
        return {"synthetic": True}
    monkeypatch.setattr(P, "helpers", helpers)
    monkeypatch.setattr(P, "prepare", lambda *args: {})
    def execute(inputs, out, report):
        if change == "edit":
            source.write_text("after")
        else:
            source.unlink()
        if complete:
            report.update(completed=True, controls_passed=True)
        else:
            raise ValueError("synthetic execution failure")
    monkeypatch.setattr(P, "execute", execute)
    report = P.run(args)
    assert report["completed"] is complete
    assert_failure_scope(report)
    assert report["hashes_unchanged"] is False
    assert report["before_sha256"] == {str(source): digest}
    if change == "edit":
        assert report["end_sha256"] == {str(source): P.A.digest(source)}
        assert report["hash_errors"] == []
    else:
        assert report["end_sha256"] == {}
        assert len(report["hash_errors"]) == 1 and "FileNotFoundError" in report["hash_errors"][0]
    if not complete:
        assert report["error"] == "ValueError: synthetic execution failure"
    assert P.D.strict_json(args.out/"manifest.json") == report


@pytest.mark.parametrize("module,name", [(P.K, "fit_track"), (P.K, "stock_stages"),
                                         (P.K, "load_anns"), (P.nativetime, "load_times"),
                                         (P.nativeseed, "load_seed"), (P.hitreco, "load_conditions"),
                                         (P.vr_f1, "read_inputs"), (P.driftann, "find")])
def test_helpers_reject_foreign_reco_entrypoints_without_spoofing(monkeypatch, module, name):
    # A replacement keeps its real foreign __module__; provenance must not skip it.
    monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="function code"):
        P.helpers({})


@pytest.mark.parametrize("bad", [np.nan, np.inf, object()])
def test_serialization_failure_preserves_fail_closed_probe_flags(tmp_path, monkeypatch, bad):
    args = arguments(tmp_path)
    source = tmp_path/"watched"
    source.write_text("stable")
    def helpers(watched):
        P.D.watch(source, watched)
        return {"synthetic": True}
    monkeypatch.setattr(P, "helpers", helpers)
    monkeypatch.setattr(P, "prepare", lambda *args: {})
    def execute(inputs, out, report):
        report.update(completed=True, controls_passed=True, invalid=bad)
    monkeypatch.setattr(P, "execute", execute)
    report = P.run(args)
    assert P.D.strict_json(args.out/"manifest.json") == report
    assert report["completed"] is False
    assert "strict JSON serialization failed" in report["error"]
    assert report["hashes_unchanged"] is True
    assert_failure_scope(report)


def test_run_control_failure_keeps_artifacts_and_failure_manifest(tmp_path, monkeypatch, dispatch_inputs):
    args = arguments(tmp_path)
    source = tmp_path/"watched"
    source.write_text("stable")
    def helpers(watched):
        P.D.watch(source, watched)
        return {"synthetic": True}
    monkeypatch.setattr(P, "helpers", helpers)
    monkeypatch.setattr(P, "prepare", lambda *args: dispatch_inputs)
    immediate_dispatch(monkeypatch)
    calls = []
    def fit(arm, *args, **kwargs):
        calls.append(arm)
        assert arm in ("ordinary", "control"), "component arm ran despite failed control"
        result = result_tree()
        if arm == "control":
            result[1]["covariance"][0, 0] = 2.
        return result
    monkeypatch.setattr(P.S, "fit", fit)
    report = P.run(args)
    assert calls == ["ordinary", "control"]
    assert report["completed"] is False
    assert_failure_scope(report)
    assert "baseline injection control differs" in report["error"]
    assert report["hashes_unchanged"] is True
    assert P.D.strict_json(args.out/"manifest.json") == report
    assert sorted(p.name for p in args.out.iterdir()) == ["control.npz", "manifest.json", "ordinary.npz"]
    values = report["arms"]["control"]
    parameter_leaf = next(key for key, path in values["leaf_paths"].items() if path == "[0]['parameters']")
    with np.load(args.out/"control.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored[parameter_leaf], result_tree()[0]["parameters"])
