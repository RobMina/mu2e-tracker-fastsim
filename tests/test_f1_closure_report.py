"""Identity, denominator and attribution guards for the stock-conditioned closure report."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("closure_report", Path(__file__).parents[1] / "scripts/f1_closure_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_identity_join_reordered_missing_empty_and_duplicate():
    a = np.array([[1, 7], [2, 3], [9, 8]])
    b = np.array([[2, 3], [1, 7]])
    np.testing.assert_array_equal(report.match(a, b), [1, 0, -1])
    np.testing.assert_array_equal(report.match(a, np.empty((0, 2))), [-1, -1, -1])
    for left, right in ((a[[0, 0]], b), (a, b[[0, 0]])):
        with pytest.raises(ValueError, match="duplicate"):
            report.match(left, right)


def test_exchange_reports_failure_agreement_not_only_net_yield():
    g, s = np.array([True, False, True, False]), np.array([False, True, True, False])
    r = report.exchange(g, s)
    assert r["net"] == 0
    assert r["gains"] == r["losses"] == 1
    assert r["acceptance_agreement"] == 0.5
    assert report.exchange(g, np.zeros(4, bool))["net_rel"] is None


def test_proxy_is_never_causal_membership():
    ids = {"xvalid": np.ones((2, 2), bool), "xact_fit": np.array([[True, False], [False, True]]),
           "xact_stock": np.zeros((2, 2), bool), "xinb": np.array([[True, False], [True, False]]),
           "xhit": np.zeros((2, 2), int), "xdoca": np.ones((2, 2))}
    r = report.attribute_outliers(ids, np.array([False, True]), np.ones(2, bool))
    assert r["causal_attribution"] is False
    assert "NOT actual iteration" in r["membership_scope"]
    assert r["mismatch_inside_proxy"]["n"] == 1
    assert r["mismatch_outside_proxy_only"]["n"] == 1


def test_nonfinite_pull_stays_in_fidelity_denominator():
    pytest.importorskip("jax")
    from trkfastsim.reco import kalman_jax as K
    x = np.array([[10., 20., 0.1, 0.2, 800., 2000.]] * 2)
    ref = np.stack([np.asarray(K.kl_params(row)) for row in x])
    errors = np.ones((2, 6))
    errors[1, 0] = np.nan
    ids = {"xt0": x, "Ct0": np.tile(np.eye(6), (2, 1, 1)), "stock_params": ref, "stock_errors": errors,
           "good": np.ones(2, bool), "stockok": np.ones(2, bool), "fact": np.ones((2, 1), bool),
           "stock_active": np.ones((2, 1), bool), "valid": np.ones((2, 1), bool)}
    r, _, _ = report.fidelity(ids)
    assert r["d0"]["n"] == 2
    assert r["d0"]["n_finite"] == 1
    assert r["d0"]["frac_within_0p1"] == 0.5


def test_report_highest_cycle_and_complete_input_gate(tmp_path):
    ak = pytest.importorskip("awkward")
    uproot = pytest.importorskip("uproot")
    pytest.importorskip("jax")
    from trkfastsim.reco import kalman_jax as K
    trace, inputs = tmp_path / "trace.root", tmp_path / "input.root"
    fields = {"run": np.array([1]), "subrun": np.array([2]), "event": np.array([3]),
              "iseed": np.array([0]), "goodseed": np.array([True]), "hstatus": ak.Array([[-1, 0]]),
              "hmiter": ak.Array([[0, 1000]]), "rejected": np.array([False]),
              "full_schedule": np.array([True]), "stage": np.array([-1]), "maxit": np.array([-1])}
    with uproot.recreate(trace) as f:
        f["kktrace/trace"] = fields
    with uproot.recreate(inputs) as f:
        f["trkdump/events"] = {"run": np.array([99]), "event": np.array([3])}
    with uproot.update(inputs) as f:
        f["trkdump/events"] = {"run": np.array([1]), "subrun": np.array([2]), "event": np.array([3]),
                               "sd_a0": ak.Array([[10.]]), "kl_d0": ak.Array([[20.]]),
                               "kl_status": ak.Array([[1 << 15]])}
    arm = tmp_path / "fixed_toy"
    arm.mkdir()
    x = np.array([[10., 20., 0.1, 0.2, 800., 2000.]])
    np.savez(arm / "ids.npz", subrun=[2], event=[3], stat=np.zeros((1, 13), int), good=[True], stockok=[True],
             xt0=x, Ct0=np.eye(6)[None], stock_params=np.asarray(K.kl_params(x[0]))[None],
             stock_errors=np.ones((1, 6)), fact=np.ones((1, 1), bool), stock_active=np.ones((1, 1), bool),
             valid=np.ones((1, 1), bool))
    (arm / "vr2f1.json").write_text(json.dumps({"input": str(inputs), "n_tracks": 1, "own_states": False}))
    args = ["--arm", f"fixed_toy={arm}", "--trace", str(trace), "--expected-events", "1", "--out", str(tmp_path / "report.json")]
    report.main(args)
    result = json.loads((tmp_path / "report.json").read_text())
    assert result["inputs"][str(inputs)]["tree"] == "trkdump/events;2"
    assert result["arms"]["fixed_toy"]["fidelity"]["d0"]["frac_within_0p1"] == 1.0
    args[5] = "2"
    with pytest.raises(ValueError, match="event count"):
        report.main(args)
