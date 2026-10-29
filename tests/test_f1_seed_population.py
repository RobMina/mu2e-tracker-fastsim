"""Analytic population-driver guards and mocked orchestration; no native fits.

These tests establish mechanics, not population acceptance, fidelity or closure.
Receipt-bound helpers and the actual kernel/comparator are not edited here.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import f1_seed_population as R


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    for name in tuple(R.os.environ):
        if name.startswith("F1_"):
            monkeypatch.delenv(name)


def operands(n=3, nf=4, mf=5):
    """Prefix-padded geometry with finite, nonzero sentinel wire directions."""
    x = np.tile([1., 2., .3, .4, 5., 2000.], (n, 1))
    x[:, 0] = np.arange(n)
    nh = np.maximum(1, nf - np.arange(n) % min(nf, 3))
    nx = np.maximum(1, mf - np.arange(n) % min(mf, 3))
    h = dict(valid=np.arange(nf)[None] < nh[:, None],
             mid=np.ones((n, nf, 3)), wdir=np.tile([1., 0., 0.], (n, nf, 1)),
             inseed=np.zeros((n, nf), dtype=bool))
    h["inseed"][:, 0] = True
    c = dict(valid=np.arange(mf)[None] < nx[:, None],
             mid=np.full((n, mf, 3), 2.), wdir=np.tile([0., 1., 0.], (n, mf, 1)),
             hit=np.full((n, mf), -1, dtype=np.int64))
    c["hit"][:, 0] = 0
    native = dict(parameters=x.copy(), covariance=np.tile(np.eye(6), (n, 1, 1)),
                  range=np.tile([-10., 20.], (n, 1)))
    return dict(x=x, h=h, c=c, native=native, times=np.arange(n, dtype=float))


def projected(n=3, nf=4, mf=5):
    keys = np.column_stack((np.full(n, 1400), np.zeros(n, dtype=int),
                            np.arange(n)+100, np.zeros(n, dtype=int))).astype(np.int64)
    return dict(keys=keys, event_index=np.arange(n, dtype=np.int64),
                good=np.ones(n, dtype=bool), stockok=np.ones(n, dtype=bool),
                kl=np.zeros((n, 6)), stock_params=np.zeros((n, 6)), stock_errors=np.ones((n, 6)),
                kl_status=np.full(n, R.vr_f1.FITOK, dtype=np.int64),
                xt0=np.tile([1., 2., .3, .4, 5., 2000.], (n, 1)),
                Ct0=np.tile(np.eye(6), (n, 1, 1)), stat=np.zeros((n, 3), dtype=np.int64),
                niter=np.tile([1, 2, 3], (n, 1)), chi2=np.ones(n), ndof=np.full(n, 7, dtype=np.int64),
                fact=np.ones((n, nf), dtype=bool), flr=np.ones((n, nf), dtype=np.int64),
                valid=np.ones((n, nf), dtype=bool), stock_active=np.ones((n, nf), dtype=bool),
                stock_lr=np.ones((n, nf), dtype=np.int64), xact_fit=np.ones((n, mf), dtype=bool),
                xvalid=np.ones((n, mf), dtype=bool))


@pytest.mark.parametrize("width", [5, 6])
def test_validate_batch_accepts_vectorized_finite_operands_without_mutation(width):
    data = operands(7)
    data["x"] = data["x"][:, :width].copy()
    before = deepcopy(data)
    assert R.validate_batch(**data) is None
    for key, value in data.items():
        if isinstance(value, dict):
            for name in value:
                np.testing.assert_array_equal(value[name], before[key][name])
        else:
            np.testing.assert_array_equal(value, before[key])


@pytest.mark.parametrize("case,message", [
    ("empty", "seed shape"), ("seed_shape", "seed shape"), ("integer_x", "operand: x"),
    ("masked_x", "operand: x"), ("nonfinite_x", "operand: x"),
    ("time_shape", "operand shape"), ("time_nan", "operand: times"),
    ("q_shape", "operand shape"), ("q_integer", "operand: parameters"),
    ("momentum", "momentum"), ("range_shape", "operand shape"),
    ("range_reverse", "reversed"), ("range_inf", "operand: range"),
    ("cov_shape", "operand shape"), ("cov_asymmetric", "asymmetric"),
    ("cov_zero", "nonpositive covariance"), ("cov_negative", "nonpositive covariance"),
    ("hit_nonprefix", "nonprefix hit"), ("crossing_nonprefix", "nonprefix crossing"),
    ("hit_empty", "empty hit"), ("crossing_empty", "empty crossing"),
    ("mask_integer", "invalid hit mask"), ("mask_masked", "invalid hit mask"),
    ("geometry_nan", "geometry"), ("geometry_shape", "geometry"),
    ("geometry_masked", "geometry"), ("direction_zero", "zero wire direction"),
    ("membership_empty", "empty seed membership"), ("membership_padding", "empty seed membership"),
    ("membership_integer", "seed membership"), ("partner_float", "partner array"),
    ("partner_negative", "partner out of range"), ("partner_large", "partner out of range"),
    ("partner_padding", "partner points to padding"),
])
def test_validate_batch_rejects_bad_operands_in_nonfirst_row(case, message):
    d = operands()
    if case == "empty":
        d = operands(0)
    elif case == "seed_shape":
        d["x"] = d["x"][:, :4]
    elif case == "integer_x":
        d["x"] = d["x"].astype(int)
    elif case == "masked_x":
        d["x"] = np.ma.array(d["x"], mask=False)
    elif case == "nonfinite_x":
        d["x"][2, 0] = np.inf
    elif case == "time_shape":
        d["times"] = d["times"][:, None]
    elif case == "time_nan":
        d["times"][2] = np.nan
    elif case == "q_shape":
        d["native"]["parameters"] = d["native"]["parameters"][:, :5]
    elif case == "q_integer":
        d["native"]["parameters"] = d["native"]["parameters"].astype(int)
    elif case == "momentum":
        d["native"]["parameters"][2, 5] = 0.
    elif case == "range_shape":
        d["native"]["range"] = d["native"]["range"][:, :1]
    elif case == "range_reverse":
        d["native"]["range"][2] = [21., 20.]
    elif case == "range_inf":
        d["native"]["range"][2, 0] = -np.inf
    elif case == "cov_shape":
        d["native"]["covariance"] = d["native"]["covariance"][:, :5]
    elif case == "cov_asymmetric":
        d["native"]["covariance"][2, 1, 0] = 1.
    elif case in ("cov_zero", "cov_negative"):
        d["native"]["covariance"][2, 5, 5] = 0. if case == "cov_zero" else -1.
    elif case in ("hit_nonprefix", "crossing_nonprefix"):
        collection = d["h" if case.startswith("hit") else "c"]
        collection["valid"][2, [0, -1]] = [False, True]
    elif case in ("hit_empty", "crossing_empty"):
        d["h" if case.startswith("hit") else "c"]["valid"][2] = False
    elif case == "mask_integer":
        d["h"]["valid"] = d["h"]["valid"].astype(int)
    elif case == "mask_masked":
        d["h"]["valid"] = np.ma.array(d["h"]["valid"], mask=False)
    elif case == "geometry_nan":
        d["c"]["mid"][2, 0, 0] = np.nan
    elif case == "geometry_shape":
        d["h"]["mid"] = d["h"]["mid"][..., :2]
    elif case == "geometry_masked":
        d["c"]["wdir"] = np.ma.array(d["c"]["wdir"], mask=False)
    elif case == "direction_zero":
        d["c"]["wdir"][2, -1] = 0.
    elif case in ("membership_empty", "membership_padding"):
        d["h"]["inseed"][2] = False
        if case == "membership_padding":
            d["h"]["inseed"][2, -1] = True
    elif case == "membership_integer":
        d["h"]["inseed"] = d["h"]["inseed"].astype(int)
    elif case == "partner_float":
        d["c"]["hit"] = d["c"]["hit"].astype(float)
    elif case == "partner_negative":
        d["c"]["hit"][2, 0] = -2
    elif case == "partner_large":
        d["c"]["hit"][2, 0] = 4
    elif case == "partner_padding":
        d["c"]["hit"][2, 0] = 3
    with pytest.raises(ValueError, match=message):
        R.validate_batch(**d)


def test_layout_thresholds_stable_order_and_width_clamping():
    nh = np.array([65, 32, 33, 64, 1, 32, 64, 1])
    nx = np.array([1, 64, 65, 128, 129, 63, 129, 64])
    h, c = {"valid": np.arange(70)[None] < nh[:, None]}, {"valid": np.arange(140)[None] < nx[:, None]}
    classes, group, order = R.layout(h, c)
    assert classes == [(32, 64), (64, 128), (70, 140)]
    np.testing.assert_array_equal(group, [2, 0, 1, 1, 2, 0, 2, 0])
    np.testing.assert_array_equal(order, [7, 5, 1, 2, 3, 4, 6, 0])
    d = operands()
    assert R.layout(d["h"], d["c"])[0] == [(4, 5)]*3


def test_control_indices_cover_each_stratum_plus_full_key_diagnostics():
    d = operands(27, 70, 140)
    classes = np.repeat(np.arange(3), 8)
    retry = np.tile(np.repeat([False, True], 4), 3)
    good = np.tile(np.repeat([False, True], 2), 6)
    nh = np.r_[np.choose(classes, [2, 40, 68]), [2, 2, 2]]
    nx = np.r_[np.choose(classes, [3, 90, 139]), [3, 3, 3]]
    d["h"]["valid"] = np.arange(70)[None] < nh[:, None]
    d["c"]["valid"] = np.arange(140)[None] < nx[:, None]
    d["baseline"] = projected(27, 70, 140)
    d["baseline"]["good"] = np.r_[good, [False]*3]
    # Cumulative iterations [2,4,6] do NOT retry; [1,4,5] do.
    d["baseline"]["niter"] = np.where(np.r_[retry, [False]*3,][:, None], [1, 4, 5], [2, 4, 6])
    d["keys"] = d["baseline"]["keys"].copy()
    d["keys"][24:] = [[1400, 0, 3700, 0], [1400, 0, 7033, 0], [1401, 0, 3700, 0]]
    selected, strata = R.control_indices(d)
    np.testing.assert_array_equal(selected, np.r_[np.arange(0, 24, 2), [24, 25]])
    assert len(strata) == 12 and sum(s["population"] for s in strata) == 27
    assert strata[0] == dict(size_class=0, retry=False, good=False, population=5)
    assert all(s["population"] == 2 for s in strata[1:])


def test_arguments_slice_widths_and_repeat_last_real_lane():
    d = operands(5, 8, 10)
    before = deepcopy(d)
    x, h, c, native, times = R.arguments(d, np.array([4, 1]), 4, 6, 5)
    selected = [4, 1, 1, 1, 1]
    np.testing.assert_array_equal(x, d["x"][selected])
    np.testing.assert_array_equal(times, d["times"][selected])
    for name in h:
        np.testing.assert_array_equal(h[name], d["h"][name][selected, :4])
    for name in c:
        np.testing.assert_array_equal(c[name], d["c"][name][selected, :6])
    for name in native:
        np.testing.assert_array_equal(native[name], before["native"][name][selected])
    np.testing.assert_array_equal(d["h"]["valid"], before["h"]["valid"])


@pytest.mark.parametrize("indices,lanes", [(np.array([], dtype=int), 256), (np.array([0, 1]), 1)])
def test_arguments_reject_invalid_lane_count(indices, lanes):
    with pytest.raises(ValueError, match="lane count"):
        R.arguments(operands(), indices, 4, 5, lanes)


def test_expand_zero_extends_activity_without_aliasing_or_changing_other_leaves():
    source = projected(2, 2, 3)
    result = R.expand(source, 6, 7)
    for name, width in (("fact", 6), ("flr", 6), ("xact_fit", 7)):
        old = source[name]
        assert result[name].shape == (2, width) and result[name].dtype == old.dtype
        np.testing.assert_array_equal(result[name][:, :old.shape[1]], old)
        assert not np.any(result[name][:, old.shape[1]:])
    for name in source:
        assert not np.shares_memory(source[name], result[name])
        if name not in ("fact", "flr", "xact_fit"):
            np.testing.assert_array_equal(result[name], source[name])


@pytest.mark.parametrize("field", ["xt0", "Ct0", "chi2"])
def test_compatibility_floats_keep_exactness_separate_from_tolerance(field):
    expected = projected()
    actual = deepcopy(expected)
    actual[field].flat[0] += 1e-6
    result = R.compatibility(actual, expected)
    assert result["passed"] and not result["exact"] and result["fidelity_predicates_exact"]
    metric = result["metrics"][field]
    assert metric["n_differing"] == 1 and metric["n_failed"] == 0
    assert metric["max_absolute_residual"] == pytest.approx(1e-6)
    actual[field].flat[0] += 1.
    assert not R.compatibility(actual, expected)["passed"]


@pytest.mark.parametrize("field", ["good", "stat", "niter", "ndof", "fact", "flr", "xact_fit"])
def test_compatibility_discrete_fields_are_exact_not_float_toleranced(field):
    expected = projected()
    actual = deepcopy(expected)
    if actual[field].dtype.kind == "b":
        actual[field].flat[0] = False
    else:
        expected[field].flat[0] = 1_000_000
        actual[field].flat[0] = 1_000_001
    result = R.compatibility(actual, expected)
    assert not result["passed"] and not result["exact"]
    assert result["metrics"][field]["n_failed"] == 1


def test_compatibility_padding_ignored_but_shape_checked():
    expected = projected()
    expected["valid"][:, -1] = False
    actual = deepcopy(expected)
    actual["fact"][:, -1] = False
    actual["flr"][:, -1] = -99
    assert R.compatibility(actual, expected)["exact"]
    actual["xt0"] = actual["xt0"][:, :5]
    with pytest.raises(ValueError, match="comparison shape: xt0"):
        R.compatibility(actual, expected)


def test_compatibility_requires_identical_strict_fidelity_predicates_with_phi_wrapping():
    expected = projected()
    expected["kl"][0, 0] = np.nextafter(.1, 0.)
    actual = deepcopy(expected)
    actual["kl"][0, 0] = .1
    result = R.compatibility(actual, expected)
    assert result["exact"] and not result["passed"] and not result["fidelity_predicates_exact"]
    expected["kl"][0, 0] = .1
    expected["kl"][0, 1] = np.pi-.01
    expected["stock_params"][0, 1] = -np.pi+.01
    actual = deepcopy(expected)
    actual["kl"][0, 1] -= 2*np.pi
    assert R.compatibility(actual, expected)["passed"]


@pytest.mark.parametrize("change", [None, "nan_position", "inf_sign", "finite"])
def test_compatibility_matching_nonfinite_preserved_not_repaired(change):
    expected = projected()
    expected["xt0"][0, :3] = [np.nan, np.inf, -np.inf]
    expected["kl"][0, :3] = [np.nan, np.inf, -np.inf]
    actual = deepcopy(expected)
    if change == "nan_position":
        actual["xt0"][0, [0, 3]] = actual["xt0"][0, [3, 0]]
    elif change == "inf_sign":
        actual["xt0"][0, 1] = -np.inf
    elif change == "finite":
        actual["xt0"][0, 0] = 0.
    with np.errstate(invalid="ignore"):
        result = R.compatibility(actual, expected)
    assert result["passed"] is (change is None)
    assert result["exact"] is (change is None)
    assert np.isnan(actual["kl"][0, 0])


@pytest.mark.parametrize("invalid_error", [0., -1., np.nan, np.inf, -np.inf])
def test_compatibility_invalid_stock_errors_are_failed_fidelity_predicates(invalid_error):
    expected = projected()
    expected["stock_errors"][0, 0] = invalid_error
    actual = deepcopy(expected)
    actual["kl"][0, 0] = np.nan
    # Both pulls are invalid, so both fail fidelity; no finite-only denominator.
    assert R.compatibility(actual, expected)["fidelity_predicates_exact"]


def test_save_is_exclusive_hash_bound_and_preserves_nonfinite(tmp_path):
    report = {}
    values = dict(x=np.array([np.nan, np.inf, -np.inf]), flag=np.array([True, False, True]))
    R.save(tmp_path, "arm", values, report)
    path = tmp_path/"arm.npz"
    before = path.read_bytes()
    assert report["artifacts"]["arm"] == dict(path=str(path), sha256=R.A.digest(path))
    with np.load(path, allow_pickle=False) as stored:
        for name in values:
            np.testing.assert_array_equal(stored[name], values[name])
    with pytest.raises(FileExistsError):
        R.save(tmp_path, "arm", {"changed": np.array([1.])}, report)
    assert path.read_bytes() == before


@pytest.fixture
def baseline_case(tmp_path, monkeypatch):
    directory = tmp_path/"preserved"
    directory.mkdir()
    input_path, gate_path = tmp_path/"input.root", tmp_path/"check.json"
    keys = np.array([[1400, 0, 7033, 0], [1400, 1, 7033, 0], [1401, 0, 7033, 0]])
    p = projected()
    p.update(keys=keys, event_index=np.array([17, 3, 91]), subrun=keys[:, 1], event=keys[:, 2])
    p["stock_params"] = np.arange(18, dtype=float).reshape(3, 6)
    p["stock_errors"][2, 4] = np.nan
    ref = dict(event_index=p["event_index"].copy(), subrun=keys[:, 1].copy(),
               event=keys[:, 2].copy(), kl_status=p["kl_status"].copy())
    for col, name in enumerate(R.vr_f1.KL+("mom",)):
        ref["kl_"+name] = p["stock_params"][:, col].copy()
        ref["kl_"+name+"err"] = p["stock_errors"][:, col].copy()
    metadata = dict(input=str(input_path), own_states=True, all_fits=False, time_model="stock",
                    time_legacy=[], xing_activity="hit_or_doca", seed_time_provenance="native_ParameterHit_time_only",
                    native_time={"check": str(gate_path)}, timing={"fast_niter": 3}, env_f1={"F1_JAX_CACHE": "cache"})
    watched_calls = []
    def watch(path, watched, sha):
        watched_calls.append((Path(path), sha))
        watched[str(path)] = sha
    monkeypatch.setattr(R.D, "watch", watch)
    def call(mode="own", order=(2, 0, 1)):
        saved = {k: v[np.asarray(order)] for k, v in p.items() if k not in ("keys", "kl")}
        np.savez(directory/"ids.npz", **saved)
        (directory/"vr2f1.json").write_text(json.dumps(metadata))
        return R.baseline(directory, mode, ref, keys, {}, input_path, gate_path)
    return SimpleNamespace(call=call, payload=p, ref=ref, keys=keys, metadata=metadata,
                           watched=watched_calls, directory=directory)


@pytest.mark.parametrize("mode", ["own", "fixed"])
def test_baseline_joins_full_identity_and_pins_preserved_files(baseline_case, mode):
    c = baseline_case
    c.metadata["own_states"] = mode == "own"
    result, metadata = c.call(mode)
    for name in ("keys", "event_index", "stock_params", "stock_errors", "subrun", "event"):
        np.testing.assert_array_equal(result[name], c.payload[name])
    assert metadata == c.metadata and result["kl"].shape == (3, 6)
    np.testing.assert_allclose(result["kl"][:, 5], 2000., rtol=0, atol=0)
    assert c.watched == [(c.directory/name, sha) for name, sha in
                         {**R.BASE_PINS[mode], "source.sha256": R.SOURCE_LIST_SHA}.items()]


@pytest.mark.parametrize("case,message", [
    ("duplicate", "duplicate"), ("missing", "population differs"), ("foreign", "missing full identity"),
    ("event", "identity/reference"), ("subrun", "identity/reference"), ("status", "identity/reference"),
    ("stock_flag", "stock flags"), ("stock_params", "stock operands"), ("stock_errors", "stock operands"),
])
def test_baseline_rejects_census_identity_and_reference_corruption(baseline_case, case, message):
    c = baseline_case
    order = (2, 0, 1)
    if case == "duplicate":
        order = (2, 0, 0)
    elif case == "missing":
        order = (0, 1)
    elif case == "foreign":
        c.payload["event_index"][1] = 1000
    elif case in ("event", "subrun"):
        c.payload[case][1] += 1
    elif case == "status":
        c.payload["kl_status"][1] ^= R.vr_f1.FITOK
    elif case == "stock_flag":
        c.payload["stockok"][1] = False
    else:
        c.payload[case][1, 0] += 1e-12
    with pytest.raises(ValueError, match=message):
        c.call(order=order)


@pytest.mark.parametrize("field,value", [("own_states", False), ("all_fits", True),
    ("time_model", "legacy"), ("time_legacy", ["anything"]), ("xing_activity", "all"),
    ("seed_time_provenance", "substituted_seed"), ("input", "different.root"),
    ("native_time", {"check": "different.json"}), ("timing", {"fast_niter": 10}),
    ("env_f1", {"F1_MOM": "1999"})])
def test_baseline_configuration_is_not_silently_reinterpreted(baseline_case, field, value):
    baseline_case.metadata[field] = value
    with pytest.raises(ValueError, match="configuration|physics overrides"):
        baseline_case.call()


def args_for(tmp_path, controls_only=False):
    return SimpleNamespace(out=tmp_path/"out", controls_only=controls_only, mode="own",
                           seed_contract=tmp_path/"seed.json", covariance_contract=tmp_path/"cov.json",
                           check=tmp_path/"check.json", conditions=tmp_path/"conditions", straw_axis=tmp_path/"axis",
                           baseline=tmp_path/"baseline")


def mock_run(monkeypatch, tmp_path):
    source = tmp_path/"watched-source"
    source.write_bytes(b"unchanged analytic fixture")
    calls = []
    def helpers(watched):
        watched[str(source)] = R.A.digest(source)
        return {"synthetic": True}
    def controls(data, out, report):
        calls.append("controls")
        report["controls"] = {"passed": True}
    def population(data, arm, out, report):
        calls.append(arm)
        R.save(out, arm, {"value": np.array([np.nan, np.inf])}, report)
        report["arms"][arm] = {"synthetic": True}
    monkeypatch.setattr(R, "helpers", helpers)
    monkeypatch.setattr(R, "prepare", lambda *a: {"synthetic": True})
    monkeypatch.setattr(R, "controls", controls)
    monkeypatch.setattr(R, "population", population)
    return source, calls


@pytest.mark.parametrize("controls_only", [False, True])
def test_run_exclusive_complete_inventory_provenance_and_strict_manifest(tmp_path, monkeypatch, controls_only):
    source, calls = mock_run(monkeypatch, tmp_path)
    args = args_for(tmp_path, controls_only)
    report = R.run(args)
    assert report["completed"] and report["controls_passed"] and report["hashes_unchanged"]
    assert calls == ["controls"] + ([] if controls_only else list(R.ARMS))
    assert tuple(report["arms"]) == (() if controls_only else R.ARMS)
    assert report["before_sha256"] == report["end_sha256"] == {str(source): R.A.digest(source)}
    assert not any(report[k] for k in ("full_fit_closure", "full_f1_closure", "full_numerical_validation"))
    manifest = args.out/"manifest.json"
    before = manifest.read_bytes()
    assert json.loads(before) == report
    with pytest.raises(FileExistsError):
        R.run(args)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("stage", ["helpers", "prepare", "controls", "population"])
def test_run_failures_are_manifested_and_stop_later_arms(tmp_path, monkeypatch, stage):
    _, calls = mock_run(monkeypatch, tmp_path)
    def fail(*args):
        raise RuntimeError("deliberate "+stage+" failure")
    monkeypatch.setattr(R, stage, fail)
    args = args_for(tmp_path)
    report = R.run(args)
    assert not report["completed"] and not report["controls_passed"]
    assert "RuntimeError: deliberate "+stage+" failure" == report["error"]
    assert not report["arms"]
    assert calls == (["controls"] if stage == "population" else [])
    assert json.loads((args.out/"manifest.json").read_text()) == report
    assert report["hashes_unchanged"] is (stage != "helpers")


@pytest.mark.parametrize("mutation", ["source_changed", "source_missing", "artifact_changed", "artifact_missing", "unwatched"])
def test_run_provenance_drift_never_passes(tmp_path, monkeypatch, mutation):
    source, _ = mock_run(monkeypatch, tmp_path)
    def controls(data, out, report):
        report["controls"] = {"passed": True}
        if mutation == "source_changed":
            source.write_bytes(b"changed")
        elif mutation == "source_missing":
            source.unlink()
        elif mutation.startswith("artifact"):
            R.save(out, "control", {"value": np.array([1.])}, report)
            artifact = out/"control.npz"
            if mutation == "artifact_changed":
                artifact.write_bytes(b"changed")
            else:
                artifact.unlink()
    monkeypatch.setattr(R, "controls", controls)
    if mutation == "unwatched":
        monkeypatch.setattr(R, "helpers", lambda watched: {})
    args = args_for(tmp_path, controls_only=True)
    report = R.run(args)
    assert not report["hashes_unchanged"] and not report["controls_passed"]
    if mutation in ("source_missing", "artifact_changed", "artifact_missing"):
        assert report["hash_errors"]
    assert json.loads((args.out/"manifest.json").read_text()) == report


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_run_serialization_failure_is_strict_fail_closed(tmp_path, monkeypatch, bad):
    mock_run(monkeypatch, tmp_path)
    def controls(data, out, report):
        report["not_json"] = bad
    monkeypatch.setattr(R, "controls", controls)
    args = args_for(tmp_path, controls_only=True)
    report = R.run(args)
    assert not report["completed"] and not report["controls_passed"]
    assert not report["full_numerical_validation"]
    assert "strict JSON serialization failed" in report["error"]
    assert json.loads((args.out/"manifest.json").read_text()) == report


def test_run_rejects_checkout_outputs_before_any_work(tmp_path, monkeypatch):
    _, calls = mock_run(monkeypatch, tmp_path)
    args = args_for(tmp_path)
    args.out = Path(R.__file__).resolve().parents[1]/"never-created-population-test-output"
    with pytest.raises(ValueError, match="protected checkout"):
        R.run(args)
    assert not calls and not args.out.exists()


def test_run_rejects_incomplete_arm_inventory(tmp_path, monkeypatch):
    mock_run(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "population", lambda *args: None)
    report = R.run(args_for(tmp_path))
    assert not report["completed"] and not report["controls_passed"]
    assert "incomplete population inventory" in report["error"]


def test_population_batches_refit_restore_input_order_and_preserve_nonfinite(tmp_path, monkeypatch):
    n, nf, mf = 1031, 70, 140
    data = operands(n, nf, mf)
    # Unsorted classes and enough class-0 rows for multiple 512-row chunks.
    group = np.zeros(n, dtype=int)
    group[[0, 517, 1030]] = 1
    group[[1, 700]] = 2
    nh, nx = np.choose(group, [5, 40, 69]), np.choose(group, [8, 90, 139])
    data["h"]["valid"] = np.arange(nf)[None] < nh[:, None]
    data["c"]["valid"] = np.arange(mf)[None] < nx[:, None]
    data["baseline"] = projected(n, nf, mf)
    data["keys"] = data["baseline"]["keys"]
    calls = []
    def compiler(d, arm, cap):
        assert d is data and arm == "momentum" and cap in (3, 10)
        def fit(x, h, c, native, times):
            ids = np.asarray(x)[:, 0].astype(int)
            nb, mb = h["valid"].shape[1], c["valid"].shape[1]
            calls.append((cap, ids.copy(), nb, mb))
            assert len(ids) % 256 == 0
            np.testing.assert_array_equal(np.asarray(times), ids.astype(float))
            result = projected(len(ids), nb, mb)
            for name in ("keys", "event_index", "stockok", "stock_params", "stock_errors", "kl_status", "valid", "stock_active", "stock_lr", "xvalid"):
                result.pop(name)
            retry = ids % 7 == 0
            result["niter"] = np.where(retry[:, None] & (cap == 3), [1, 4, 5], [2, 4, 6])
            result["chi2"] = ids.astype(float) + (10000. if cap == 10 else 0.)
            result["xt0"][:, 0] = result["chi2"]
            result["kl"][:, 0] = ids
            result["kl"][ids == 14, 5] = np.nan
            return result
        return fit
    monkeypatch.setattr(R, "compiler", compiler)
    report = {}
    R.population(data, "momentum", tmp_path, report)
    with np.load(tmp_path/"momentum.npz", allow_pickle=False) as saved:
        expected = np.arange(n, dtype=float) + (np.arange(n) % 7 == 0)*10000.
        np.testing.assert_array_equal(saved["chi2"], expected)
        np.testing.assert_array_equal(saved["xt0"][:, 0], expected)
        np.testing.assert_array_equal(saved["keys"], data["keys"])
        np.testing.assert_array_equal(saved["event_index"], np.arange(n))
        np.testing.assert_array_equal(saved["kl"][:, 0], np.arange(n))
        assert np.isnan(saved["kl"][14, 5])
        assert saved["fact"].shape == (n, nf) and saved["xact_fit"].shape == (n, mf)
        assert not saved["fact"][group == 0, 32:].any()
        assert not saved["xact_fit"][group == 1, 128:].any()
    assert [c[0] for c in calls] == [3]*5 + [10]*3
    assert [(nb, mb) for _, _, nb, mb in calls[:5]] == [(32, 64)]*3+[(64, 128), (70, 140)]
    for cap, ids, _, _ in calls:
        if cap == 10:
            assert np.all(ids % 7 == 0)
    logs = report["arms"]["momentum"]["timing"]
    assert [r["n"] for r in logs if r["pass_index"] == 1] == [512, 512, 2, 3, 2]
    assert sum(r["n"] for r in logs if r["pass_index"] == 2) == len(np.arange(0, n, 7))
    assert report["artifacts"]["momentum"]["sha256"] == R.A.digest(tmp_path/"momentum.npz")


def test_helpers_watch_receipt_bound_helpers_and_all_new_modules():
    watched = {}
    execution = R.helpers(watched)
    root = Path(R.__file__).resolve().parents[1]
    assert execution["fitter_sha256"] == R.P.FITTER_SHA
    for module in (R, R.F, R.Q, R.P, R.S):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(root)
        assert watched[str(path)] == R.A.digest(path)
    assert watched[str(root/"scripts/f1_seed_arms.py")] == "945f22d6ff6059b67d85b8e81bc567cc27dd43fee933f3d7475c73d831188692"
    assert watched[str(root/"scripts/f1_seed_arm_probe.py")] == "a7f9b2f97e01b6cf21ecc443238f7d2670041f379246462467b6ab0a4e9752fa"


@pytest.mark.parametrize("module", [R, R.F, R.Q])
def test_helpers_reject_foreign_new_module_origin(monkeypatch, module):
    monkeypatch.setattr(module, "__file__", str(Path(R.__file__).resolve().parent/"foreign.py"))
    with pytest.raises(ValueError, match="module origin"):
        R.helpers({})


def test_helpers_reject_spoofed_new_runner_callable(monkeypatch):
    def foreign(*args):
        return None
    foreign.__module__ = R.__name__
    monkeypatch.setattr(R, "validate_batch", foreign)
    with pytest.raises(ValueError, match="function code from different checkout"):
        R.helpers({})


def test_prepare_rejects_f1_environment_before_loading_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("F1_JAX_CACHE", "diagnostic-cache")
    with pytest.raises(ValueError, match="default F1 environment"):
        R.prepare(args_for(tmp_path), {}, {})


def test_prepare_requires_original_and_supplemental_receipt_pins(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    calls = []
    def receipt(path, sha, watched, flags):
        calls.append((path, sha, flags))
        if len(calls) == 1:
            return {"diagnostics": {"n_seeds": 2319}}
        raise ValueError("stop after second receipt")
    monkeypatch.setattr(R.D, "receipt", receipt)
    with pytest.raises(ValueError, match="stop after second receipt"):
        R.prepare(args, {}, {})
    assert calls == [
        (args.seed_contract, "c1c7c19c7fc5cabdd85cd195ffa6973998a478fd9444cabd66882f3e677fa633",
         ("completed", "passed", "hashes_unchanged")),
        (args.covariance_contract, "e3b649e3b135bd68131b25c4faaaa2b14641a2c07c664ce1defa04084af4d740",
         ("completed", "passed", "exact", "hashes_unchanged")),
    ]


@pytest.mark.parametrize("mode", ["own", "fixed"])
@pytest.mark.parametrize("full_tree", [False, True])
def test_compiler_vmapped_dispatch_preserves_native_time_and_fixed_state(monkeypatch, mode, full_tree):
    data = operands()
    data.update(mode=mode, drift={"marker": 1}, stages={"marker": 2}, ann={"marker": 3})
    data["h"]["active"] = np.array([[True, False, True, False]]*3)
    data["h"]["lr"] = np.array([[1, -1, 0, 1]]*3)
    calls = []
    def fit(arm, x, h, c, drift, stages, ann, native, t, fixed, maxniter):
        calls.append((arm, drift is data["drift"], stages is data["stages"], ann is data["ann"], maxniter))
        if mode == "own":
            assert fixed is None
        else:
            assert set(fixed) == {"active", "lr", "nulldv"}
        result = dict(x=x, q=native["parameters"], time=t)
        if fixed is not None:
            result.update(fixed)
        return result
    monkeypatch.setattr(R.F, "fit", fit)
    monkeypatch.setattr(R.F, "project", lambda result: {"xt0": result["x"]})
    compiled = R.compiler(data, "geometry_time", 10, full_tree=full_tree)
    args = R.arguments(data, np.array([2, 0]), 4, 5, 3)
    result = compiled(*args)
    R.jax.block_until_ready(result)
    assert calls == [("geometry_time", True, True, True, 10)]
    if not full_tree:
        assert set(result) == {"xt0"}
        np.testing.assert_array_equal(result["xt0"], data["x"][[2, 0, 0]])
    else:
        np.testing.assert_array_equal(result["q"], data["native"]["parameters"][[2, 0, 0]])
        np.testing.assert_array_equal(result["time"], [2., 0., 0.])
        if mode == "fixed":
            np.testing.assert_array_equal(result["active"], data["h"]["active"][[2, 0, 0]])
            np.testing.assert_array_equal(result["lr"], data["h"]["lr"][[2, 0, 0]])
            np.testing.assert_array_equal(result["nulldv"], np.ones((3, 4), dtype=bool))


@pytest.mark.parametrize("mismatch", [None, "owner", "padded_owner", "preserved", "fidelity"])
def test_controls_compare_complete_trees_and_restore_fast_refit_payload(tmp_path, monkeypatch, mismatch):
    data = operands()
    base = projected()
    base["keys"][1:] = [[1400, 0, 3700, 0], [1400, 0, 7033, 0]]
    base["niter"] = np.array([[2, 4, 6]]*3)
    base["chi2"] = np.array([100., 1., 102.])
    data.update(baseline=base, keys=base["keys"])
    if mismatch == "fidelity":
        base["kl"][1, 0] = .1
    calls = []
    def compiler(d, arm, cap, full_tree):
        assert full_tree and d is data
        def fit(x, h, c, native, t):
            ids = np.asarray(x)[:, 0].astype(int)
            calls.append((arm, cap, len(ids)))
            out = projected(len(ids))
            for name in ("keys", "event_index", "stockok", "stock_params", "stock_errors", "kl_status", "valid", "stock_active", "stock_lr"):
                out.pop(name)
            out["chi2"] = ids.astype(float)+(100. if cap == 10 else 0.)
            if mismatch == "preserved":
                out["chi2"] += 10.
            out["niter"] = np.where(((ids % 2 == 0) & (cap == 3))[:, None], [1, 4, 5], [2, 4, 6])
            out["owner"] = np.full((len(ids), 2), np.nan)
            if mismatch in ("owner", "padded_owner") and arm == "control":
                out["owner"][0 if mismatch == "owner" else -1, 0] = 1.
            return out
        return fit
    monkeypatch.setattr(R, "compiler", compiler)
    monkeypatch.setattr(R.F, "project", lambda result: {k: v for k, v in result.items() if k != "owner"})
    report = {}
    if mismatch:
        with pytest.raises(ValueError, match="bounded baseline compatibility failed"):
            R.controls(data, tmp_path, report)
    else:
        R.controls(data, tmp_path, report)
    assert report["controls"]["passed"] is (mismatch is None)
    assert calls == [("ordinary", 3, 256), ("control", 3, 256), ("ordinary", 10, 256), ("control", 10, 256)]
    batch = report["controls"]["batches"][0]
    assert batch["retry"] == [True, False, True]
    assert batch["lanes"] == 256 and batch["hit_width"] == 4 and batch["crossing_width"] == 5
    assert batch["cap3_injection"]["exact"] is (mismatch not in ("owner", "padded_owner"))
    assert batch["preserved_baseline"]["passed"] is (mismatch not in ("preserved", "fidelity"))
    assert len(report["artifacts"]) == 5
    with np.load(tmp_path/"control_class0_projected.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["keys"], data["keys"])
        np.testing.assert_array_equal(saved["chi2"], base["chi2"]+(10. if mismatch == "preserved" else 0.))
    for arm in ("ordinary", "control"):
        for cap in (3, 10):
            with np.load(tmp_path/f"control_class0_{arm}_cap{cap}.npz", allow_pickle=False) as saved:
                assert all(len(saved[name]) == 3 for name in saved.files)
                assert set(saved.files) == set(batch["leaf_paths"])
                assert any("owner" in path for path in batch["leaf_paths"].values())


@pytest.mark.parametrize("name", ["validate_batch", "baseline", "controls", "population"])
def test_helpers_reject_foreign_required_runner_code_without_spoofing(monkeypatch, name):
    monkeypatch.setattr(R, name, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="function code|missing.*function"):
        R.helpers({})


def test_prepare_real_receipt_hash_failure_retains_provenance_and_never_reads_inputs(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    args.seed_contract.write_text('{"completed": true}')
    def forbidden(*args, **kwargs):
        pytest.fail("receipt failure must precede native input loading")
    monkeypatch.setattr(R.vr_f1, "read_inputs", forbidden)
    monkeypatch.setattr(R.hitreco, "load_conditions", forbidden)
    watched = {}
    with pytest.raises(ValueError, match="hash mismatch"):
        R.prepare(args, watched, {})
    assert watched == {str(args.seed_contract): R.A.digest(args.seed_contract)}


def test_run_partial_population_failure_preserves_completed_artifacts(tmp_path, monkeypatch):
    mock_run(monkeypatch, tmp_path)
    original = R.population
    def population(data, arm, out, report):
        if arm == R.ARMS[1]:
            raise RuntimeError("second arm failed")
        original(data, arm, out, report)
    monkeypatch.setattr(R, "population", population)
    args = args_for(tmp_path)
    report = R.run(args)
    assert not report["completed"] and not report["controls_passed"]
    assert report["hashes_unchanged"]
    assert tuple(report["arms"]) == R.ARMS[:1]
    assert tuple(report["artifacts"]) == R.ARMS[:1]
    assert (args.out/(R.ARMS[0]+".npz")).exists()
    assert not (args.out/(R.ARMS[1]+".npz")).exists()


def test_run_manifest_creation_is_exclusive_even_after_work(tmp_path, monkeypatch):
    mock_run(monkeypatch, tmp_path)
    def controls(data, out, report):
        (out/"manifest.json").write_bytes(b"existing owned-by-someone-else receipt")
    monkeypatch.setattr(R, "controls", controls)
    args = args_for(tmp_path, controls_only=True)
    with pytest.raises(FileExistsError):
        R.run(args)
    assert (args.out/"manifest.json").read_bytes() == b"existing owned-by-someone-else receipt"


@pytest.mark.parametrize("completed,passed,code", [(True, True, 0), (False, True, 1), (True, False, 1)])
def test_main_resolves_all_paths_and_exit_status(tmp_path, monkeypatch, capsys, completed, passed, code):
    import sys
    fields = ("out", "seed-contract", "covariance-contract", "check", "conditions", "straw-axis", "baseline")
    argv = ["f1_seed_population.py", "--mode", "fixed", "--controls-only"]
    for field in fields:
        argv.extend(["--"+field, str(tmp_path/".."/tmp_path.name/field)])
    seen = []
    def run(args):
        seen.append(args)
        return dict(completed=completed, controls_passed=passed, hashes_unchanged=True)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(R, "run", run)
    with pytest.raises(SystemExit) as exit_info:
        R.main()
    assert exit_info.value.code == code
    assert len(seen) == 1 and seen[0].mode == "fixed" and seen[0].controls_only
    for field in fields:
        assert getattr(seen[0], field.replace("-", "_")) == (tmp_path/field).resolve()
    printed = json.loads(capsys.readouterr().out)
    assert printed["completed"] is completed and printed["controls_passed"] is passed
