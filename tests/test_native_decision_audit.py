"""Small independent quadratics, full identity joins, provenance and fail-closed guards."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

spec = importlib.util.spec_from_file_location("native_decision", Path(__file__).parents[1] / "scripts/f1_native_decision_audit.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def sample():
    iterations, pieces = [], []
    for row, (subrun, seed, valid, status) in enumerate(((0, 0, True, 0), (0, 1, True, 4), (1, 0, False, 7))):
        identity = dict(run=1, subrun=subrun, event=9, iseed=seed, iteration_ordinal=0,
                        miter=1, iter=0, schema_version=1)
        front = np.array([1., 2*np.pi, 0., .2, 0., 3.])
        back = np.array([2., 1., .5, 0., 2., 4.])
        reference = np.arange(18, dtype=float).reshape(3, 6) + row
        base = np.diag(np.arange(1., 7.))
        base[0, 1] = base[1, 0] = .25
        cov = np.stack([base, base*2, base*3])
        # Independent expected result via solve; candidate cov/deweight deliberately irrelevant.
        expected = np.array([front @ np.linalg.solve(cov[1], front), back @ np.linalg.solve(cov[2], back)])
        it = dict(identity, status=status, capture_valid=True, capture_error="", decision_valid=valid,
                  prior_valid=valid, candidate_present=valid, nold=3, ncandidate=2 if valid else 0,
                  front_step=front.tolist() if valid else [], back_step=back.tolist() if valid else [],
                  reference_indices=[1, 2] if valid else [-1, -1], front_chi2=expected[0] if valid else np.nan,
                  back_chi2=expected[1] if valid else np.nan, threshold=float(expected.max()) if row == 0 else 0., deweight=123.)
        iterations.append(it)
        for index in range(3):
            pieces.append(dict(identity, owner=0, piece_index=index, parameters=reference[index].tolist(),
                               covariance=cov[index].ravel().tolist(), range=[10.*index, 10.*index+9.]))
        if valid:
            for index, step in enumerate((front, back)):
                pieces.append(dict(identity, owner=1, piece_index=index, parameters=(reference[index+1]+step).tolist(),
                                   covariance=(np.eye(6)*999).ravel().tolist(), range=[10.*(index+1), 10.*(index+1)+9.]))
        # Store actual raw subtraction exactly (floating-point cancellation is part of capture).
        if valid:
            iterations[-1]["front_step"] = ((reference[1]+front)-reference[1]).tolist()
            iterations[-1]["back_step"] = ((reference[2]+back)-reference[2]).tolist()
    # Ensure thresholds at equality use precisely the independently computed expression.
    for it in iterations[:2]:
        qs = D.quadratic(np.array([it["front_step"], it["back_step"]]), cov[1:])
        it["front_chi2"], it["back_chi2"] = qs.tolist()
        if it["status"] == 0:
            it["threshold"] = float(qs.max())
    return ak.Array(iterations), ak.Array(pieces)[::-1]


def replace(table, row, name, value):
    records = ak.to_list(table)
    records[row][name] = value
    return ak.Array(records)


def test_quadratic_and_scope():
    it, p = sample()
    report = D.audit(it[::-1], p)
    assert report["passed"]
    assert report["n_iterations"] == report["n_targets"] == 3
    assert report["n_decision_valid"] == 2
    assert report["n_excluded_decision_invalid"] == 1
    assert report["by_recorded_status"]["7"]["n_excluded"] == 1
    assert report["recorded_status_groups"]["failure_status_other"]["n_decisions"] == 1
    assert report["all_decisions"]["threshold_exceedance"]["recomputed_exceeds"] == 1
    assert not report["full_status_replay"] and not report["full_numerical_validation"]
    assert report["regularization"] == "none"
    json.dumps(report, allow_nan=False)


def test_quadratic_known_and_no_regularization():
    s = np.arange(1., 7.)[None, :]
    c = np.diag(np.arange(1., 7.))[None, :, :]
    np.testing.assert_allclose(D.quadratic(s, c), [21.], rtol=1e-14)
    with pytest.raises(ValueError, match="singular"):
        D.quadratic(s, np.zeros((1, 6, 6)))
    with pytest.raises(ValueError, match="nonfinite"):
        D.quadratic(s*np.nan, c)


def test_numerical_and_threshold_mismatch_keys():
    it, p = sample()
    bad = replace(it, 0, "front_chi2", float(it.threshold[0])+10.)
    report = D.audit(bad, p)
    assert not report["passed"]
    expected = [[1, 0, 9, 0, 0]]
    assert report["all_decisions"]["front"]["mismatch_keys"] == expected
    assert report["all_decisions"]["threshold_exceedance"]["mismatch_keys"] == expected
    assert report["all_decisions"]["front"]["max_relative_residual"] > 0
    bad = replace(it, 0, "front_step", [0.]*6)
    report = D.audit(bad, p)
    assert not report["passed"] and report["raw_steps"]["mismatch_keys"] == expected


@pytest.mark.parametrize("mutation,match", [
    (lambda i, p: (i, ak.concatenate([p, p[:1]])), "duplicate"),
    (lambda i, p: (i, p[1:]), "piece count"),
    (lambda i, p: (i, replace(p, 0, "event", 99)), "missing full identity"),
    (lambda i, p: (i, replace(p, 0, "miter", 2)), "miter"),
    (lambda i, p: (i, replace(p, 0, "piece_index", 40)), "piece_index"),
    (lambda i, p: (replace(i, 0, "reference_indices", [3, 0]), p), "reference_indices"),
    (lambda i, p: (replace(i, 0, "reference_indices", [1., 2.]), p), "integer dtype"),
    (lambda i, p: (replace(i, 0, "front_step", [1.]), p), "shape"),
    (lambda i, p: (replace(i, 0, "threshold", np.inf), p), "nonfinite"),
    (lambda i, p: (replace(i, 0, "front_step", [np.nan]*6), p), "nonfinite"),
    (lambda i, p: (replace(i, 0, "capture_valid", False), p), "invalid iteration"),
    (lambda i, p: (i, replace(p, 0, "parameters", [0.]*5)), "shape"),
])
def test_malformed_capture(mutation, match):
    it, p = mutation(*sample())
    with pytest.raises(ValueError, match=match):
        D.audit(it, p)


def evidence(tmp_path):
    it, p = sample()
    event = ak.Array([dict(run=1, subrun=0, event=9), dict(run=1, subrun=1, event=9)])
    trace = ak.Array([dict(zip(D.KEYS, identity), iteration_capture_expected=1, iteration_capture_written=1,
                          iteration_capture_requested=True, iteration_capture_valid=True, iteration_capture_errors="")
                      for identity in ((1, 0, 9, 0), (1, 0, 9, 1), (1, 1, 9, 0))])
    capture = tmp_path / "capture.root"
    with uproot.recreate(capture) as root:
        for name, table in (("trkdump/events", event), ("kktrace/trace", trace), ("kktrace/iterations", it[:1]), ("kktrace/pieces", p)):
            root[name] = {field: table[field] for field in table.fields}
        root["kktrace/iterations"] = {field: it[field] for field in it.fields}
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"keys": D.keys(trace).tolist()}))
    inputs = {}
    for name in ("control", "capture", "baseline", "targets"):
        path = targets if name == "targets" else capture if name == "capture" else tmp_path / (name+".root")
        if name in ("control", "baseline"):
            path.write_bytes(capture.read_bytes())
        inputs[name] = {"path": str(path), "sha256": D.digest(path)}
    gate = dict.fromkeys(("passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage", "recorded_capture_equivalent"), True)
    gate.update(smoke_only=False, n_iterations=3, n_events=2, n_seeds=3, n_targets=3, n_targets_found=3, inputs=inputs)
    check = tmp_path / "check.json"
    check.write_text(json.dumps(gate))
    return check, gate


def test_cli_highest_cycle_and_exclusive_output(tmp_path):
    check, _ = evidence(tmp_path)
    args = SimpleNamespace(check=check, out=tmp_path / "audit.json")
    report = D.run(args)
    assert report["passed"], report
    assert report["highest_cycles"]["capture"]["kktrace/iterations"].endswith("iterations;2")
    assert report["all_four_evidence_hashes_verified"]
    assert report["target_limited"]
    assert json.loads(args.out.read_text()) == report
    with pytest.raises(FileExistsError):
        D.run(args)


@pytest.mark.parametrize("name", ["control", "capture", "baseline", "targets"])
def test_all_evidence_hashes_required(tmp_path, name):
    check, gate = evidence(tmp_path)
    with Path(gate["inputs"][name]["path"]).open("ab") as stream:
        stream.write(b"changed")
    report = D.run(SimpleNamespace(check=check, out=tmp_path / "failed.json"))
    assert not report["passed"]
    assert "evidence hash changed: " + name in report["error"]


def test_smoke_gate_rejected(tmp_path):
    check, gate = evidence(tmp_path)
    gate["smoke_only"] = True
    check.write_text(json.dumps(gate))
    report = D.run(SimpleNamespace(check=check, out=tmp_path / "failed.json"))
    assert not report["passed"] and "full capture passivity" in report["error"]


def test_tolerances_and_zero_residual_denominator():
    ids = np.array([[1, 0, 9, 0, 0], [1, 0, 9, 1, 0]])
    mask = np.ones(2, dtype=bool)
    report = D.residual(np.array([.5*D.ATOL, 10.+.5*D.RTOL*10.]), np.array([0., 10.]), ids, mask)
    assert report["passed"]
    assert report["max_relative_residual"] == .5
    report = D.residual(np.array([2*D.ATOL, 10.+2*D.RTOL*10.]), np.array([0., 10.]), ids, mask)
    assert report["mismatches"] == 2


@pytest.mark.parametrize("value,match", [(np.nan, "nonfinite"), (0., "singular")])
def test_bad_selected_reference_covariance(value, match):
    it, p = sample()
    selected = np.flatnonzero((D.column(p, "owner") == 0) & (D.column(p, "piece_index") == 1) & (D.column(p, "iseed") == 0) & (D.column(p, "subrun") == 0))[0]
    p = replace(p, selected, "covariance", [value]*36)
    with pytest.raises(ValueError, match=match):
        D.audit(it, p)


def test_highest_cycle_census_must_match_gate(tmp_path):
    check, gate = evidence(tmp_path)
    gate["n_iterations"] = 1
    check.write_text(json.dumps(gate))
    report = D.run(SimpleNamespace(check=check, out=tmp_path / "failed.json"))
    assert not report["passed"] and "highest-cycle iteration census" in report["error"]
