"""Initial audit provenance, input binding and exclusive failure receipts."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_initial_crossing_audit as D


def args(tmp_path):
    return SimpleNamespace(**{k: tmp_path/k for k in ("out", "full_check", "associated", "capture", "expanded", "native_source")})


def test_helpers_bound():
    watched = {}
    result = D.helpers(watched)
    assert len(result) == 9 and len(watched) == 10


@pytest.mark.parametrize("module,field", [(D.I, "initial_inputs"), (D.I, "generate"), (D.I, "measurements"), (D.C, "constructor_hint"), (D.I, "signal_speed"), (D.I, "speed_configuration")])
def test_foreign_callable_fails(monkeypatch, module, field):
    monkeypatch.setattr(module, field, lambda *a: None)
    with pytest.raises(ValueError):
        D.helpers({})


@pytest.mark.parametrize("module,field", [(D.I, "match_rows"), (D.I, "column"), (D.I, "require")])
def test_foreign_alias_fails(monkeypatch, module, field):
    monkeypatch.setattr(module, field, lambda *a: None)
    with pytest.raises(ValueError):
        D.helpers({})


def test_bound_input_canonical(tmp_path):
    path = tmp_path/"input"
    path.write_text("data")
    link = tmp_path/"link"
    link.symlink_to(path)
    watched = {}
    assert D.bound_input(link, {str(path): D.A.digest(path)}, watched) == path
    assert str(path) in watched and str(link) not in watched


def test_unbound_input_rejected(tmp_path):
    path = tmp_path/"input"
    path.write_text("data")
    with pytest.raises(ValueError, match="not bound"):
        D.bound_input(path, {}, {})


def test_failure_receipt_exclusive(tmp_path):
    a = args(tmp_path)
    report = D.run(a)
    assert not report["passed"] and not report["completed"]
    assert report["hashes_unchanged"]
    assert "FileNotFoundError" in report["error"]
    before = a.out.read_bytes()
    assert json.loads(before) == report
    with pytest.raises(FileExistsError):
        D.run(a)
    assert a.out.read_bytes() == before


@pytest.mark.parametrize("key", ["full_check", "associated", "capture", "expanded", "native_source"])
def test_protected_paths(tmp_path, key):
    a = args(tmp_path)
    a.out = getattr(a, key)/("report.json") if key == "native_source" else getattr(a, key)
    with pytest.raises(ValueError, match="protected"):
        D.run(a)
    assert not a.out.exists()


def test_end_hash_drift(tmp_path, monkeypatch):
    a = args(tmp_path)
    f = tmp_path/"changing"
    f.write_text("one")
    def helpers(watched):
        D.D.watch(f, watched)
        f.write_text("two")
        raise ValueError("abort")
    monkeypatch.setattr(D, "helpers", helpers)
    report = D.run(a)
    assert not report["passed"] and not report["hashes_unchanged"]


def test_wrapper_closure_cannot_spoof_metadata(monkeypatch):
    original = D.N.sensor
    fake = D.N._checked(lambda *a, **kw: None)
    fake.__wrapped__ = original.__wrapped__
    monkeypatch.setattr(D.N, "sensor", fake)
    with pytest.raises(ValueError, match="wrapper execution"):
        D.helpers({})
