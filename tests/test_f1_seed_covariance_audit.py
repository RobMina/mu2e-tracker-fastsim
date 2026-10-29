"""Source-typed covariance tests; no native-data or binary-provenance claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_seed_covariance_audit as C

CONFIG = dict(momentum=2000., errors=[15., .05, 10., .02, 2., 1.], constraints=[0., 0., 0., 0., 0., .0001])
DIAGONAL = np.array([225., .002500000176951289, 100., .00039999998989515007, 4., 1.])


def fixture():
    i = np.arange(2319, dtype=np.int64)
    keys = np.column_stack((1400+i//1000, i//100%3, i%1000+1, i%2))
    fields = dict(zip(C.B.KEYS, keys.T))
    fields["covariance"] = np.broadcast_to(np.diag(DIAGONAL).reshape(36), (2319, 36)).copy()
    seeds = ak.Array(fields)
    target = seeds[:147]
    target = ak.with_field(target, np.zeros(147, dtype=np.int64), "category")
    target = ak.with_field(target, np.zeros(147, dtype=np.int64), "piece_index")
    target = ak.with_field(target, np.arange(147, dtype=np.int64), "trajectory_index")
    return seeds, target, keys[:147]


def test_float_product_then_double_not_double_product():
    matrix = C.covariance(CONFIG)
    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(np.diag(matrix), DIAGONAL)
    assert not np.array_equal(matrix, np.diag(np.asarray(CONFIG["errors"])**2))
    assert not np.array_equal(matrix, np.diag(np.asarray(CONFIG["errors"], dtype=np.float32).astype(float)**2))
    assert np.count_nonzero(matrix-np.diag(np.diag(matrix))) == 0


@pytest.mark.parametrize("field", ["momentum", "errors", "constraints"])
def test_config_change_rejected(field):
    config = deepcopy(CONFIG)
    config[field] = 1000. if field == "momentum" else [0.]*6
    with pytest.raises(ValueError, match="configuration"):
        C.covariance(config)


def test_exact_full_census_and_permutations():
    seeds, targets, keys = fixture()
    report = C.compare(seeds[::-1], targets[::-1], keys, CONFIG)
    assert report["completed"] and report["passed"] and report["exact"]
    assert report["n_seeds"] == 2319 and report["n_targets"] == 147
    assert report["comparison"]["all_seeds"]["n_components"] == 2319*36


@pytest.mark.parametrize("family", ["seed", "target"])
@pytest.mark.parametrize("delta,passed", [(1e-9, True), (1., False)])
def test_covariance_mutation_is_not_generator_input(family, delta, passed):
    seeds, targets, keys = fixture()
    table = seeds if family == "seed" else targets
    values = C.A.dense(table, "covariance", 36).copy()
    values[5, 7] += delta
    table = ak.with_field(table, values, "covariance")
    if family == "seed":
        seeds = table
    else:
        targets = table
    report = C.compare(seeds, targets, keys, CONFIG)
    assert report["passed"] is passed and report["exact"] is False
    np.testing.assert_array_equal(report["generated_diagonal"], DIAGONAL)
    metric = report["comparison"]["all_seeds" if family == "seed" else "initial_construction"]
    assert metric["n_inexact_rows"] == 1


@pytest.mark.parametrize("kind", ["seed_count", "target_count", "keys_count", "foreign_target", "duplicate", "piece"])
def test_census_and_join_guards(kind):
    seeds, targets, keys = fixture()
    if kind == "seed_count":
        seeds = seeds[:-1]
    elif kind == "target_count":
        targets = targets[:-1]
    elif kind == "keys_count":
        keys = keys[:-1]
    elif kind == "foreign_target":
        keys = keys.copy()
        keys[0, 0] = 9999
    elif kind == "duplicate":
        targets = targets[np.zeros(147, dtype=int)]
    else:
        targets = ak.with_field(targets, np.ones(147, dtype=np.int64), "piece_index")
    with pytest.raises(ValueError):
        C.compare(seeds, targets, keys, CONFIG)


def test_helpers_and_foreign_callable(monkeypatch):
    watched = {}
    C.helpers(watched)
    assert watched[str(Path(C.S.__file__).resolve())] == C.SEED_HELPER_SHA
    monkeypatch.setattr(C, "covariance", lambda *args: None)
    with pytest.raises(ValueError, match="different checkout"):
        C.helpers({})


def seed_stub(tmp_path):
    path = tmp_path/"seed.json"
    value = dict(inputs=dict(native_source=str(tmp_path/"native")), before_sha256={})
    path.write_text(json.dumps(value))
    return path


def test_failure_exclusive_strict_receipt(tmp_path):
    seed = seed_stub(tmp_path)
    out = tmp_path/"out.json"
    report = C.run(seed, out)
    assert not report["passed"] and not report["exact"] and not report["completed"]
    assert "hash mismatch" in report["error"] and report["hashes_unchanged"]
    assert json.loads(out.read_text()) == report
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        C.run(seed, out)
    assert out.read_bytes() == before


@pytest.mark.parametrize("which", ["receipt", "native", "manifest", "checkout"])
def test_protected_paths(tmp_path, which):
    seed = seed_stub(tmp_path)
    if which == "receipt":
        out = seed
    elif which == "native":
        out = tmp_path/"native"/"out.json"
    elif which == "checkout":
        out = Path(C.__file__).resolve().parents[1]/"never-created.json"
    else:
        out = tmp_path/"evidence"
        value = json.loads(seed.read_text())
        value["before_sha256"][str(out)] = "0"*64
        seed.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protected"):
        C.run(seed, out)


@pytest.mark.parametrize("drift", [False, True])
def test_mock_orchestration_and_hash_drift(tmp_path, monkeypatch, drift):
    path = seed_stub(tmp_path)
    source = tmp_path/"source"
    source.write_text("before")
    def helpers(watched):
        C.D.watch(source, watched)
    def receipt(*args):
        return dict(all_seeds=dict(parameters=dict(exact=True)), targets=dict(n_targets=147))
    def population(seed, watched, report):
        report.update(completed=True, passed=True, exact=True)
        if drift:
            source.write_text("after")
    monkeypatch.setattr(C, "helpers", helpers)
    monkeypatch.setattr(C.D, "receipt", receipt)
    monkeypatch.setattr(C, "population", population)
    report = C.run(path, tmp_path/"out.json")
    assert report["completed"]
    assert report["passed"] is not drift and report["exact"] is not drift
    assert report["hashes_unchanged"] is not drift
    assert not report["full_f1_closure"] and not report["full_fit_closure"]
    assert not report["full_numerical_validation"]
