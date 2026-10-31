"""Analytic coverage and local receipt fixtures only; no native data/fits."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_seed_outlier_coverage as C


def sample(n=6):
    keys = np.array([[1, 0, 7, 0], [2, 0, 7, 0], [1, 1, 7, 0],
                     [1, 0, 7, 1], [1, 0, 8, 0], [3, 0, 7, 0]])[:n]
    return dict(keys=keys, good=np.ones(n, bool), stockok=np.ones(n, bool),
                kl=np.zeros((n, 6)), stock_params=np.zeros((n, 6)), stock_errors=np.ones((n, 6)),
                stat=np.ones((n, 2), dtype=int), niter=np.ones((n, 2), dtype=int))


def permute(payload, order):
    return {key: value[order] for key, value in payload.items()}


def histories(keys=None, counts=None):
    if keys is None:
        keys = sample(3)["keys"]
    if counts is None:
        counts = np.array([2, 3, 1])
    n = int(counts.sum())
    return dict(zip(C.B.KEYS, np.repeat(keys, counts, axis=0).T),
                iteration_ordinal=np.arange(n)-np.repeat(np.cumsum(counts)-counts, counts),
                capture_valid=np.ones(n, bool), capture_error=np.full(n, "", dtype="U1"))


def test_full_key_distinctions_and_permutation():
    fixed, own = sample(), sample()
    fixed["kl"][:, 5] = -.2
    captured = fixed["keys"][[3, 1, 2]]
    first = C.coverage(fixed, own, captured)
    second = C.coverage(permute(fixed, [5, 2, 0, 3, 4, 1]), permute(own, [1, 4, 3, 2, 5, 0]), captured[::-1])
    assert first == second
    group = first["fixed"]["groups"]["momentum_failures"]
    assert group["n"] == 6 and group["n_covered"] == group["n_uncovered"] == 3
    assert group["covered_keys"] == [[1, 0, 7, 1], [1, 1, 7, 0], [2, 0, 7, 0]]
    assert group["uncovered_keys"] == [[1, 0, 7, 0], [1, 0, 8, 0], [3, 0, 7, 0]]
    example = group["first_covered_example"]
    assert example["key"] == [1, 0, 7, 1]
    assert example["signed_pulls"]["values"][5] == -.2
    assert example["stat"]["values"] == [1, 1]
    assert example["niter"]["values"] == [1, 1]


@pytest.mark.parametrize("error", [0., -1., np.nan, np.inf, -np.inf])
def test_invalid_stock_errors_fail_without_denominator_shrink(error):
    fixed, own = sample(), sample()
    fixed["stock_errors"][0, 5] = error
    fixed["good"][1] = False
    fixed["kl"][1, 5] = np.nan
    result = C.coverage(fixed, own, fixed["keys"])["fixed"]
    assert result["fidelity"]["mom"] == dict(n=5, n_pass=4, n_fail=1, n_nonfinite=1)
    assert result["groups"]["fit_rejected"]["n"] == 1
    ex = result["groups"]["momentum_failures"]["first_covered_example"]
    assert ex["signed_pulls"]["values"][5] is None and ex["signed_pulls"]["nonfinite"][5]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_signed_pull_json_null_and_flag(value):
    fixed = sample()
    fixed["kl"][0, 5] = value
    result = C.coverage(fixed, sample(), fixed["keys"])
    ex = result["fixed"]["groups"]["momentum_failures"]["first_covered_example"]
    assert ex["signed_pulls"]["values"][5] is None
    assert ex["kl"]["values"][5] is None
    assert ex["signed_pulls"]["nonfinite"][5]
    json.dumps(result, allow_nan=False)


def test_exact_strict_cut_phi_wrap_and_other_coordinate_failures():
    fixed = sample()
    fixed["kl"][:, 1] = 2*np.pi
    fixed["kl"][0, 5] = .1
    fixed["kl"][1, 5] = -.1
    fixed["kl"][2, 5] = np.nextafter(.1, 0.)
    fixed["kl"][3, 0] = .2
    fixed["kl"][4, 4] = -.2
    result = C.coverage(fixed, sample(), fixed["keys"])["fixed"]
    assert result["groups"]["momentum_failures"]["n"] == 2
    assert result["groups"]["phi0_failures"]["n"] == 0
    assert result["groups"]["d0_failures"]["n"] == result["groups"]["t0_failures"]["n"] == 1


def test_own_gains_losses_all_eligible_not_finite_filtered():
    own = sample()
    own["stockok"][[0, 2, 3]] = False
    own["good"][[1, 3]] = False
    own["kl"][:] = np.nan
    result = C.coverage(sample(), own, own["keys"][[0, 1]])["own"]
    assert result["n"] == 6 and result["both_good"] == 2
    assert result["groups"]["gains"]["n"] == 2
    assert result["groups"]["gains"]["n_covered"] == 1
    assert result["groups"]["losses"]["n"] == result["groups"]["losses"]["n_covered"] == 1
    assert result["groups"]["neither"]["n"] == 1
    assert result["fidelity"]["mom"]["n_nonfinite"] == 2


@pytest.mark.parametrize("captured", [np.empty((0, 4), dtype=int), np.array([[999, 999, 7, 0]])])
def test_empty_intersection(captured):
    fixed = sample()
    fixed["kl"][:, 5] = .2
    group = C.coverage(fixed, sample(), captured)["fixed"]["groups"]["momentum_failures"]
    assert group["n"] == group["n_uncovered"] == 6
    assert group["n_covered"] == 0 and group["covered_keys"] == []
    assert group["first_covered_example"] is None


@pytest.mark.parametrize("kind", ["duplicate", "negative", "overflow", "float", "width"])
def test_bad_population_keys(kind):
    fixed = sample()
    if kind == "duplicate": fixed["keys"][1] = fixed["keys"][0]
    if kind == "negative": fixed["keys"][0, 0] = -1
    if kind == "overflow":
        fixed["keys"] = fixed["keys"].astype(np.uint64)
        fixed["keys"][0, 0] = 2**63
    if kind == "float": fixed["keys"] = fixed["keys"].astype(float)
    if kind == "width": fixed["keys"] = fixed["keys"][:, :3]
    with pytest.raises(ValueError):
        C.coverage(fixed, sample(), sample()["keys"])


def test_fixed_stock_rejection_fails():
    fixed = sample()
    fixed["stockok"][0] = False
    with pytest.raises(ValueError, match="native rejection"):
        C.coverage(fixed, sample(), sample()["keys"])


def test_history_permuted_contiguous_keys_and_counts():
    h = histories()
    targets, counts = C.history(h, 3, 6)
    reverse = {key: value[::-1] for key, value in h.items()}
    rt, rc = C.history(reverse, 3, 6)
    np.testing.assert_array_equal(targets, rt)
    np.testing.assert_array_equal(counts, rc)
    assert counts.tolist() == [2, 1, 3]


@pytest.mark.parametrize("kind", ["duplicate", "gap", "start_one", "negative", "overflow", "float",
                                  "valid_false", "valid_integer", "error", "error_numeric", "length", "empty", "target_count", "iteration_count"])
def test_malformed_history_fails(kind):
    h = histories()
    if kind == "duplicate": h["iteration_ordinal"][1] = 0
    if kind == "gap": h["iteration_ordinal"][1] = 2
    if kind == "start_one": h["iteration_ordinal"] += 1
    if kind == "negative": h["iteration_ordinal"][0] = -1
    if kind == "overflow":
        h["run"] = h["run"].astype(np.uint64)
        h["run"][0] = 2**63
    if kind == "float": h["event"] = h["event"].astype(float)
    if kind == "valid_false": h["capture_valid"][0] = False
    if kind == "valid_integer": h["capture_valid"] = h["capture_valid"].astype(int)
    if kind == "error": h["capture_error"][0] = "x"
    if kind == "error_numeric": h["capture_error"] = np.zeros(6)
    if kind == "length": h["run"] = h["run"][:-1]
    if kind == "empty": h = {key: value[:0] for key, value in h.items()}
    with pytest.raises(ValueError):
        C.history(h, 4 if kind == "target_count" else 3, 7 if kind == "iteration_count" else 6)


def test_history_byte_strings():
    h = histories()
    h["capture_error"] = h["capture_error"].astype("S1")
    assert len(C.history(h)[0]) == 3


def test_real_helpers_record_current_sources_and_origins():
    watched = {}
    modules = C.helpers(watched)
    assert len(modules) == len(watched) == 9
    assert all(C.A.digest(path) == sha for path, sha in watched.items())
    assert str(Path(C.__file__).resolve()) in watched


@pytest.mark.parametrize("module,field", [(C.Q, "_pulls"), (C.Q, "validate"), (C, "coverage"), (C.A, "highest"), (C.B, "integer_matrix")])
def test_foreign_callable_rejected(monkeypatch, module, field):
    monkeypatch.setattr(module, field, lambda *a, **k: None)
    with pytest.raises(ValueError, match="function code"):
        C.helpers({})


def test_foreign_module_path_rejected(monkeypatch):
    monkeypatch.setattr(C.Q, "__file__", "/path/to/foreign.py")
    with pytest.raises(ValueError, match="module path"):
        C.helpers({})


def test_current_source_pin_mismatch_rejected(monkeypatch):
    monkeypatch.setitem(C.SOURCE_PINS, "scripts/f1_seed_population_compare.py", "0"*64)
    with pytest.raises(ValueError, match="hash mismatch"):
        C.helpers({})


def write_receipt(path, before, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(fields, before_sha256=before, end_sha256=before, hashes_unchanged=True)))
    return C.A.digest(path)


@pytest.fixture
def bound(tmp_path, monkeypatch):
    data = tmp_path/"data"
    data.mkdir()
    inherited = data/"inherited.txt"
    inherited.write_text("unchanged")
    before = {str(inherited): C.A.digest(inherited)}
    payloads = {}
    for mode, n in (("fixed", 2151), ("own", 2315)):
        payload = dict(keys=np.column_stack((np.ones(n, dtype=int), np.zeros(n, dtype=int),
                                            np.arange(n), np.zeros(n, dtype=int))),
                       good=np.ones(n, bool), stockok=np.ones(n, bool), kl=np.zeros((n, 6)),
                       stock_params=np.zeros((n, 6)), stock_errors=np.ones((n, 6)))
        if mode == "fixed":
            payload["good"][2139:] = False
            payload["kl"][:218, 5] = .2
        else:
            payload["stockok"][2151:] = False
            payload["good"][:] = False
            payload["good"][:2108] = True
            payload["good"][2151:2210] = True
        directory = data/f"seed_population_{mode}_v1"
        directory.mkdir()
        artifact = directory/"momentum.npz"
        np.savez(artifact, **payload)
        artifacts = dict(momentum=dict(path=str(artifact), sha256=C.A.digest(artifact)),
                         parameters=dict(path=str(directory/"NOT_READ.npz"), sha256="0"*64))
        sha = write_receipt(directory/"manifest.json", before, completed=True, controls_passed=True,
                            mode=mode, arms={arm: {} for arm in C.ARMS}, artifacts=artifacts)
        monkeypatch.setitem(C.PINS, mode, sha)
        payloads[mode] = payload
    directory = data/"crossing_3k_20261028_v2"
    directory.mkdir()
    capture = directory/"capture.root"
    counts = np.full(147, 23)
    counts[:68] += 1
    h = histories(payloads["own"]["keys"][:147], counts)
    schema = {key: "string" if key == "capture_error" else value.dtype for key, value in h.items()}
    with uproot.recreate(capture) as root:
        root.mktree("kktrace/iterations", schema).extend({key: value[:1] for key, value in h.items()})
        root.mktree("kktrace/iterations", schema).extend(h)
    capture_sha = C.A.digest(capture)
    monkeypatch.setattr(C, "CAPTURE_SHA", capture_sha)
    full_sha = write_receipt(directory/"crossing_full_check_v1.json", dict(before, **{str(capture): capture_sha}),
                             passed=True, full_target_capture_valid=True, full_target_coverage=True,
                             n_events=3000, n_seeds=2319, n_targets=147, n_iterations=3449)
    monkeypatch.setattr(C, "FULL_SHA", full_sha)
    return data, tmp_path/"new.json", inherited


def test_end_to_end_receipt_binding_highest_cycle_momentum_only(bound):
    data, out, inherited = bound
    result = C.run(data, out)
    assert result["passed"], result.get("error")
    assert result == json.loads(out.read_text())
    assert result["completed"] and result["hashes_unchanged"]
    assert result["before_sha256"] == result["end_sha256"]
    assert str(inherited) in result["before_sha256"]
    assert not any("NOT_READ" in path for path in result["before_sha256"])
    assert result["highest_cycles"]["kktrace/iterations"].endswith("iterations;2")
    assert result["history"]["n_targets"] == 147 and result["history"]["n_iterations"] == 3449
    groups = result["coverage"]["fixed"]["groups"]
    assert groups["momentum_failures"]["n"] == 218 and groups["fit_rejected"]["n"] == 12
    assert groups["momentum_failures"]["n_covered"] == 147
    assert groups["momentum_failures"]["n_uncovered"] == 71
    assert result["coverage"]["own"]["groups"]["gains"]["n"] == 59
    assert result["coverage"]["own"]["groups"]["losses"]["n"] == 43
    assert all(result[key] is False for key in C.CLOSURE)


def test_output_is_exclusive(bound):
    data, out, _ = bound
    out.write_text("do not replace")
    with pytest.raises(FileExistsError):
        C.run(data, out)
    assert out.read_text() == "do not replace"


def test_checkout_output_rejected_before_creation(tmp_path):
    target = Path(C.__file__).resolve().parents[1]/"coverage-test-never-created.json"
    assert not target.exists()
    with pytest.raises(ValueError, match="protected checkout"):
        C.run(tmp_path, target)
    assert not target.exists()


def test_symlink_to_checkout_output_rejected(tmp_path):
    link = tmp_path/"checkout"
    link.symlink_to(Path(C.__file__).resolve().parents[1], target_is_directory=True)
    with pytest.raises(ValueError, match="protected checkout"):
        C.run(tmp_path, link/"coverage-test-never-created.json")


def test_missing_receipt_preserves_strict_failure(tmp_path):
    out = tmp_path/"failure.json"
    result = C.run(tmp_path, out)
    assert result == json.loads(out.read_text())
    assert not result["passed"] and not result["completed"]
    assert result["hashes_unchanged"] and "FileNotFoundError" in result["error"]
    assert all(result[key] is False for key in C.CLOSURE)


@pytest.mark.parametrize("target", ["inherited", "artifact", "manifest", "capture"])
def test_changed_pinned_source_fails(bound, target):
    data, out, inherited = bound
    path = {"inherited": inherited,
            "artifact": data/"seed_population_fixed_v1/momentum.npz",
            "manifest": data/"seed_population_fixed_v1/manifest.json",
            "capture": data/"crossing_3k_20261028_v2/capture.root"}[target]
    with path.open("ab") as stream:
        stream.write(b" changed")
    result = C.run(data, out)
    assert not result["passed"] and "hash mismatch" in result["error"]
    assert json.loads(out.read_text()) == result


def test_end_hash_detects_changed_input(bound, monkeypatch):
    data, out, inherited = bound
    real = C.evidence
    def mutate(data, watched, report):
        real(data, watched, report)
        inherited.write_text("changed during read")
    # Exercise the finish guard without bypassing the independently tested origin check in production.
    real_helpers = C.helpers
    modules = real_helpers({})
    monkeypatch.setattr(C, "helpers", lambda watched: (C.D.watch(inherited, watched), modules)[1])
    monkeypatch.setattr(C, "evidence", mutate)
    result = C.run(data, out)
    assert not result["passed"] and not result["hashes_unchanged"]
    assert result["before_sha256"] != result["end_sha256"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, object()])
def test_serialization_failure_preserves_hashes_and_false_closure(bad):
    report = dict(completed=True, passed=True, before_sha256={"source": "0"*64},
                  end_sha256={"source": "0"*64}, hashes_unchanged=True, hash_errors=[], bad=bad)
    result, text = C.serialize(report)
    assert json.loads(text) == result
    assert not result["passed"] and not result["completed"]
    assert result["before_sha256"] == report["before_sha256"]
    assert all(result[key] is False for key in C.CLOSURE)
    assert result["limits"]


def test_serialization_failure_run_writes_failure(tmp_path, monkeypatch):
    source = tmp_path/"source"
    source.write_text("source")
    monkeypatch.setattr(C, "helpers", lambda watched: C.D.watch(source, watched))
    monkeypatch.setattr(C, "evidence", lambda data, watched, report: report.update(bad=np.inf))
    out = tmp_path/"failed.json"
    result = C.run(tmp_path, out)
    assert result == json.loads(out.read_text())
    assert not result["passed"] and result["hashes_unchanged"]
    assert "serialization failed" in result["error"]
