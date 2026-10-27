"""Saved-report hard checks fail closed on absent evidence and reported errors."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("jaxcheck_audit", Path(__file__).parents[1] / "scripts/jaxcheck_audit.py")
auditmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditmod)


def _report():
    entry = {"timing": {"total": 1.0, "to_plane": .2}, "stats": {"rec_overflow": 0, "capped": 0}}
    return {"backend": "gpu", "device": "mock", "files": [
        {"file": "mock.root", "generated": 100, "jax": [entry, entry], "numpy": [entry],
         "reproducible": {"jax": True, "numpy": True}}]}


def test_missing_reports_fail(tmp_path):
    assert not auditmod.audit([])["hard_gates_pass"]
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"files": []}))
    assert not auditmod.audit([path])["hard_gates_pass"]


def test_reported_failure_and_cross_requirement(tmp_path):
    path = tmp_path / "report.json"
    report = _report()
    path.write_text(json.dumps(report))
    assert auditmod.audit([path])["hard_gates_pass"]
    assert not auditmod.audit([path], require_cross=True)["hard_gates_pass"]
    report["files"][0]["reproducible"]["numpy"] = False
    path.write_text(json.dumps(report))
    assert not auditmod.audit([path])["hard_gates_pass"]


def test_missing_or_nonzero_counters_fail(tmp_path):
    path = tmp_path / "report.json"
    for stats in ({}, {"rec_overflow": 2}, {"capped": 0}):
        report = _report()
        report["files"][0]["jax"][0]["stats"] = stats
        path.write_text(json.dumps(report))
        assert not auditmod.audit([path])["hard_gates_pass"]
