"""Receipt and failure-output guards, independent of remote data or native runtime."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_crossing_constructor_audit as D


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '[NaN]'])
def test_strict_json(tmp_path, text):
    p = tmp_path/"bad.json"
    p.write_text(text)
    with pytest.raises(ValueError):
        D.strict_json(p)


def bound(tmp_path):
    data = tmp_path/"data.txt"
    data.write_text("unchanged")
    p = tmp_path/"receipt.json"
    manifest = {str(data): D.A.digest(data)}
    doc = dict(passed=True, hashes_unchanged=True, before_sha256=manifest, end_sha256=manifest)
    p.write_text(json.dumps(doc))
    return p, data, doc


def test_receipt_all_inputs_bound(tmp_path):
    p, data, doc = bound(tmp_path)
    watched = {}
    assert D.receipt(p, D.A.digest(p), watched, ("passed", "hashes_unchanged")) == doc
    assert watched == {str(p): D.A.digest(p), str(data): D.A.digest(data)}


@pytest.mark.parametrize("change", ["numeric_flag", "wrong_end", "missing_manifest", "bad_digest", "pnfs", "relative"])
def test_receipt_guard(tmp_path, change):
    p, data, doc = bound(tmp_path)
    if change == "numeric_flag":
        doc["passed"] = 1
    elif change == "wrong_end":
        doc["end_sha256"] = {}
    elif change == "missing_manifest":
        doc["before_sha256"] = doc["end_sha256"] = {}
    elif change == "bad_digest":
        doc["before_sha256"] = doc["end_sha256"] = {str(data): "bad"}
    elif change == "pnfs":
        doc["before_sha256"] = doc["end_sha256"] = {"/pnfs/" + "a": "0"*64}
    else:
        doc["before_sha256"] = doc["end_sha256"] = {"relative": "0"*64}
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        D.receipt(p, D.A.digest(p), {}, ("passed",))


def test_receipt_changed_input(tmp_path):
    p, data, _ = bound(tmp_path)
    data.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        D.receipt(p, D.A.digest(p), {}, ("passed",))


def test_receipt_conflict(tmp_path):
    p, data, _ = bound(tmp_path)
    with pytest.raises(ValueError, match="changed watched"):
        D.receipt(p, D.A.digest(p), {str(data): "0"*64}, ("passed",))


def test_watch_no_overwrite(tmp_path):
    p, data, _ = bound(tmp_path)
    watched = {}
    D.watch(data, watched)
    expected = watched.copy()
    data.write_text("changed")
    with pytest.raises(ValueError):
        D.watch(data, watched)
    assert watched == expected


def test_actual_helpers_are_local_and_pinned():
    watched = {}
    modules = D.helpers(watched)
    assert len(modules) == 7 and len(watched) == 7
    assert all(Path(p).exists() for p in modules.values())


def test_transitive_monkeypatch_rejected(monkeypatch):
    monkeypatch.setattr(D.B, "L", SimpleNamespace())
    with pytest.raises(ValueError, match="transitive"):
        D.helpers({})


def test_wrong_module_path_rejected(monkeypatch):
    monkeypatch.setattr(D.L, "__file__", "/path/to/elsewhere.py")
    with pytest.raises(ValueError, match="executing module"):
        D.helpers({})


def args(tmp_path):
    return SimpleNamespace(out=tmp_path/"report.json", full_check=tmp_path/"full.json",
                           observability=tmp_path/"obs.json", capture=tmp_path/"capture.root", native_source=tmp_path/"native")


def test_missing_gate_leaves_exclusive_failure_receipt(tmp_path):
    a = args(tmp_path)
    result = D.run(a)
    assert not result["passed"] and not result["completed"]
    assert result["hashes_unchanged"] and "FileNotFoundError" in result["error"]
    before = a.out.read_bytes()
    with pytest.raises(FileExistsError):
        D.run(a)
    assert a.out.read_bytes() == before


@pytest.mark.parametrize("target", ["full_check", "observability", "capture", "native_source"])
def test_protected_output(tmp_path, target):
    a = args(tmp_path)
    a.out = getattr(a, target)
    if target == "native_source":
        a.out = a.out/"report.json"
    with pytest.raises(ValueError, match="protected"):
        D.run(a)
    assert not a.out.exists()


def test_changed_watched_input_fails_end_check(tmp_path, monkeypatch):
    a = args(tmp_path)
    watched_file = tmp_path/"watched"
    watched_file.write_text("one")
    def helpers(watched):
        D.watch(watched_file, watched)
        watched_file.write_text("two")
        raise ValueError("diagnostic failure")
    monkeypatch.setattr(D, "helpers", helpers)
    result = D.run(a)
    assert not result["passed"] and not result["hashes_unchanged"]
    assert result["before_sha256"] != result["end_sha256"]


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_matrix_key_overflow(dtype):
    key = np.zeros((2, 5), dtype=dtype)
    key[0, 0] = np.iinfo(dtype).max
    key[1, 0] = 1
    if dtype is np.uint64:
        with pytest.raises(ValueError, match="overflow"):
            D.B.integer_matrix(key, 5, "test")
    else:
        np.testing.assert_array_equal(key, D.B.integer_matrix(key, 5, "test"))


@pytest.mark.parametrize("module,name", [(D.B, "compare"), (D.B, "generate"), (D.L, "piecewise"), (D.C, "particle"), (D.A, "read_tree")])
def test_foreign_required_function_rejected(monkeypatch, module, name):
    monkeypatch.setattr(module, name, lambda *a, **kw: True)
    with pytest.raises(ValueError, match="function code"):
        D.helpers({})


def test_imported_function_alias_rejected(monkeypatch):
    monkeypatch.setattr(D.B, "match_rows", lambda *a, **kw: True)
    with pytest.raises(ValueError, match="imported helper"):
        D.helpers({})


@pytest.mark.parametrize("bad", [np.nan, np.inf, object()])
def test_json_failure_is_written_as_minimal_strict_receipt(bad):
    report = dict(passed=True, completed=True, before_sha256={"x": "0"*64}, end_sha256={"x": "0"*64},
                  hashes_unchanged=True, hash_errors=[], nested={"value": bad})
    result, text = D.serialize(report)
    assert json.loads(text) == result
    assert not result["passed"] and not result["completed"]
    assert result["hashes_unchanged"] and result["before_sha256"] == report["before_sha256"]


def test_receipt_symlink_retarget_uses_authenticated_canonical_path(tmp_path, monkeypatch):
    p, data, original = bound(tmp_path)
    other = tmp_path/"other.json"
    other.write_text(json.dumps(dict(original, inserted=True)))
    link = tmp_path/"link.json"
    link.symlink_to(p)
    real_watch = D.watch
    def switch(path, watched, expected=None):
        value = real_watch(path, watched, expected)
        if Path(path) == p:
            link.unlink()
            link.symlink_to(other)
        return value
    monkeypatch.setattr(D, "watch", switch)
    result = D.receipt(link, D.A.digest(p), {}, ("passed",))
    assert result == original and "inserted" not in result
