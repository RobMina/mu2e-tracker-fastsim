"""Crossing report failures preserve provenance and never claim unavailable closure."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_crossing_observability as X
from test_f1_crossing_observability import hints_fixture, hints_call, reference_fixture, reference_call


@pytest.fixture(autouse=True)
def default_environment(monkeypatch):
    for key in list(X.os.environ):
        if key.startswith("F1_"):
            monkeypatch.delenv(key)


def args(tmp_path):
    return SimpleNamespace(transition_report=tmp_path/"missing.json", native_source=tmp_path/"native", out=tmp_path/"out.json")


def failure(report, path):
    assert report["completed"] is False
    assert json.loads(path.read_text()) == report
    for name in ("full_history_closure", "full_state_policy_closure", "full_fit_closure", "native_crossing_constructor_offsets_compared"):
        assert report[name] is False
    json.dumps(report, allow_nan=False)


def test_missing_receipt_preserves_expected_hash(tmp_path):
    a = args(tmp_path)
    r = X.run(a)
    failure(r, a.out)
    assert r["before_sha256"] == {str(a.transition_report): X.TRANSITION_SHA256}
    assert r["end_sha256"] == {str(a.transition_report): None}
    assert r["hashes_unchanged"] is False


def test_nondefault_environment_and_exclusive_output(tmp_path, monkeypatch):
    a = args(tmp_path)
    monkeypatch.setenv("F1_UNEXPECTED", "1")
    r = X.run(a)
    failure(r, a.out)
    assert "requires default" in r["error"]
    saved = a.out.read_bytes()
    with pytest.raises(FileExistsError):
        X.run(a)
    assert a.out.read_bytes() == saved


@pytest.mark.parametrize("mutation", ["none", "change", "delete"])
def test_failure_rehashes_known_inputs(tmp_path, monkeypatch, mutation):
    a = args(tmp_path)
    p = tmp_path/"evidence"
    p.write_text("original")
    original = X.digest(p)
    def fail(path, watched, expected):
        assert expected == X.TRANSITION_SHA256
        watched[str(p)] = original
        if mutation == "change":
            p.write_text("changed")
        elif mutation == "delete":
            p.unlink()
        raise ValueError("forced failure")
    monkeypatch.setattr(X.H, "receipt", fail)
    r = X.run(a)
    failure(r, a.out)
    assert r["before_sha256"][str(p)] == original
    assert r["hashes_unchanged"] is (mutation == "none")
    assert "forced failure" in r["error"]


@pytest.mark.parametrize("bad", [None, [], "bad", float("nan")])
def test_malformed_manifest_strict_json(tmp_path, monkeypatch, bad):
    a = args(tmp_path)
    p = tmp_path/"evidence"
    p.write_text("original")
    manifest = {str(p): bad}
    a.transition_report.write_text(json.dumps(dict(completed=True, hashes_unchanged=True,
        before_sha256=manifest, end_sha256=manifest, rtol=X.R.RTOL, atol=X.R.ATOL)))
    receipt, pin = X.H.receipt, X.digest(a.transition_report)
    monkeypatch.setattr(X.H, "receipt", lambda path, watched, expected: receipt(path, watched, expected=pin))
    r = X.run(a)
    failure(r, a.out)
    assert "invalid manifest digest" in r["error"]


@pytest.mark.parametrize("family", ["hits", "crossings"])
@pytest.mark.parametrize("bad", ["negative", "overflow", "float", "duplicate", "shape"])
def test_first_row_indices_fail_closed(family, bad):
    life, rows, previous = hints_fixture()
    values = life[family]["first_rows"]
    if bad == "negative":
        values[1] = -1
    elif bad == "overflow":
        values[1] = len(rows)
    elif bad == "float":
        life[family]["first_rows"] = values.astype(float)
    elif bad == "duplicate":
        values[1] = values[0]
    else:
        life[family]["first_rows"] = values[:, None]
    with pytest.raises(ValueError, match="first-row"):
        hints_call(life, rows, previous)


@pytest.mark.parametrize("bad", ["same", "overlap", "reversed", "zero"])
def test_ambiguous_old_owner_ranges_rejected(bad):
    e, p, i = reference_fixture()
    p[2]["range"] = {"same": [0., 10.], "overlap": [5., 15.], "reversed": [20., 10.], "zero": [10., 10.]}[bad]
    with pytest.raises(ValueError, match="range"):
        reference_call(e, p, i)


def test_one_ulp_range_needs_no_representable_midpoint():
    e, p, i = reference_fixture()
    p[0]["range"] = e[0]["reference_range"] = [0., 1.]
    p[2]["range"] = e[1]["reference_range"] = [1., np.nextafter(1., np.inf)]
    result = reference_call(e, p, i)
    assert result["exact"] is True
    assert result["piece_index"][:2] == [0, 1]
