"""Pinned-history provenance and exclusive failure receipts without remote inputs."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_crossing_history_audit as D


def args(tmp_path):
    return SimpleNamespace(**{k: tmp_path/k for k in ("out", "full_check", "initial", "associated", "capture", "native_source")})


def test_helpers_bound():
    watched = {}
    result = D.helpers(watched)
    assert len(result) == 11 and len(watched) == 12


@pytest.mark.parametrize("name", ["pack", "schedule", "bootstrap", "solve_trace", "replay", "compare", "partners"])
def test_foreign_function_rejected(monkeypatch, name):
    monkeypatch.setattr(D.H, name, lambda *a: None)
    with pytest.raises(ValueError):
        D.helpers({})


def test_foreign_alias_rejected(monkeypatch):
    monkeypatch.setattr(D.H, "match_rows", lambda *a: None)
    with pytest.raises(ValueError, match="alias"):
        D.helpers({})


def test_failure_output_exclusive(tmp_path):
    a = args(tmp_path)
    result = D.run(a)
    assert not result["passed"] and not result["completed"] and result["hashes_unchanged"]
    assert "FileNotFoundError" in result["error"]
    before = a.out.read_bytes()
    assert json.loads(before) == result
    with pytest.raises(FileExistsError):
        D.run(a)
    assert a.out.read_bytes() == before


@pytest.mark.parametrize("name", ["full_check", "initial", "associated", "capture", "native_source"])
def test_protected_output(tmp_path, name):
    a = args(tmp_path)
    a.out = getattr(a, name)/"report.json" if name == "native_source" else getattr(a, name)
    with pytest.raises(ValueError, match="protected"):
        D.run(a)


def test_end_hash_change_fails(tmp_path, monkeypatch):
    a = args(tmp_path)
    p = tmp_path/"input"
    p.write_text("one")
    def helpers(watched):
        D.D.watch(p, watched)
        p.write_text("two")
        raise ValueError("stop")
    monkeypatch.setattr(D, "helpers", helpers)
    result = D.run(a)
    assert not result["passed"] and not result["hashes_unchanged"]
