"""Synthetic provenance and exact-passivity tests for the crossing smoke gate."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_crossing_check as C


def fixture_gate(tmp_path):
    inputs = {}
    for name in ("control", "capture", "baseline", "targets"):
        path = tmp_path / (name + ".data")
        path.write_text(name)
        inputs[name] = dict(path=str(path), sha256=C.A.digest(path))
    gate = dict(passed=True, recorded_capture_equivalent=True, smoke_only=True,
                final_capture_gate=False, n_events=100, inputs=inputs)
    path = tmp_path / "check.json"
    path.write_text(json.dumps(gate))
    return path, gate


def test_receipt_literal_flags_and_hashes(tmp_path):
    path, gate = fixture_gate(tmp_path)
    watched = {}
    assert C.receipt(path, watched) == gate
    assert len(watched) == 5


@pytest.mark.parametrize("field,bad", [(field, bad) for field in
    ("passed", "recorded_capture_equivalent", "smoke_only") for bad in (1, "true", False, None)] +
    [("final_capture_gate", 0), ("final_capture_gate", True), ("n_events", True), ("n_events", 100.0), ("n_events", 99)])
def test_receipt_rejects_nonliteral_and_wrong_population(tmp_path, field, bad):
    path, gate = fixture_gate(tmp_path)
    gate[field] = bad
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        C.receipt(path, {})


def test_changed_evidence_rejected(tmp_path):
    path, gate = fixture_gate(tmp_path)
    Path(gate["inputs"]["capture"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        C.receipt(path, {})


def test_exact_table_nan_and_jagged():
    a = ak.Array([dict(run=1, subrun=0, event=7, iseed=0, q=[1., np.nan]),
                  dict(run=1, subrun=0, event=8, iseed=0, q=[])])
    assert C.exact_table(a, a, C.A.KEYS, "test")["passed"]
    with pytest.raises(ValueError, match="keys/order"):
        C.exact_table(a, a[::-1], C.A.KEYS, "test")
    b = ak.with_field(a, ak.Array([[np.nextafter(1., 2.), np.nan], []]), "q")
    with pytest.raises(ValueError, match="values differ"):
        C.exact_table(a, b, C.A.KEYS, "test")


def test_exact_table_missing_field_or_mask():
    a = ak.Array([dict(run=1, subrun=0, event=7, iseed=0, q=[1.])])
    with pytest.raises(ValueError, match="fields differ"):
        C.exact_table(a, ak.without_field(a, "q"), C.A.KEYS, "test")
    b = ak.with_field(a, ak.Array([[None]]), "q")
    with pytest.raises(ValueError):
        C.exact_table(b, b, C.A.KEYS, "test")


def selected_fixture():
    events = ak.Array(dict(run=np.ones(100, dtype=int), subrun=np.zeros(100, dtype=int), event=np.arange(100)))
    seed = ak.Array([dict(run=1, subrun=0, event=5, iseed=0, capture_requested=True),
                     dict(run=1, subrun=0, event=7, iseed=1, capture_requested=False)])
    it = ak.Array([dict(run=1, subrun=0, event=5, iseed=0, iteration_ordinal=0)])
    return {"seed_contract": seed, "iterations": it}, events, [[1, 0, 5, 0], [1, 0, 120, 0]], dict(n_seeds=2, n_targets_found=1, n_iterations=1)


def test_selected_targets_come_from_event_prefix_not_captured_subset():
    c, e, t, g = selected_fixture()
    assert C.selected(c, e, t, g).tolist() == [[1, 0, 5, 0]]
    t.append([1, 0, 7, 1])
    g["n_targets_found"] = 2
    with pytest.raises(ValueError, match="requests"):
        C.selected(c, e, t, g)


@pytest.mark.parametrize("bad", [[], [[-1, 0, 5, 0]], [[1., 0, 5, 0]], [[1, 0, 5]], [[1, 0, 5, 0]] * 2])
def test_invalid_targets_fail(bad):
    c, e, _, g = selected_fixture()
    with pytest.raises(ValueError):
        C.selected(c, e, bad, g)


def test_failure_report_exclusive_and_strict_json(tmp_path, monkeypatch):
    args = SimpleNamespace(check=tmp_path / "check.json", reference_check=tmp_path / "ref.json",
                           preparation=tmp_path / "prep.json", native_source=tmp_path / "native", out=tmp_path / "out.json")
    def fail(*args):
        raise ValueError("synthetic manifest failure")
    monkeypatch.setattr(C, "source_manifest", fail)
    report = C.run(args)
    assert report["passed"] is False and report["hashes_unchanged"] is True
    assert "synthetic manifest failure" in report["error"]
    assert json.loads(args.out.read_text())["full_fit_closure"] is False
    with pytest.raises(FileExistsError):
        C.run(args)


def test_protected_report_path_rejected(tmp_path):
    p = tmp_path / "receipt.json"
    p.write_text("unchanged")
    args = SimpleNamespace(check=p, reference_check=p, preparation=p, native_source=tmp_path / "native", out=p)
    with pytest.raises(ValueError, match="protected"):
        C.run(args)
    assert p.read_text() == "unchanged"


def test_expanded_only_debug_filename_diff_allowed(tmp_path):
    a, b = tmp_path / "new.fcl", tmp_path / "old.fcl"
    a.write_text(f'physics: {{cut: 2}}\nfileName: "{a}"\n')
    b.write_text(f'physics: {{cut: 2}}\nfileName: "{b}"\n')
    C.expanded_equal(a, b)
    a.write_text(f'physics: {{cut: 3}}\nfileName: "{a}"\n')
    with pytest.raises(ValueError, match="configuration differs"):
        C.expanded_equal(a, b)
    a.write_text(f'fileName: "{a}"\nfileName: "{a}"\n')
    with pytest.raises(ValueError, match="ambiguous"):
        C.expanded_equal(a, b)


def test_all_event_keys_checked_even_seedless():
    a = ak.Array(dict(run=[1, 1], subrun=[0, 0], event=[1, 2]))
    b = ak.with_field(a, ak.Array([1, 3]), "event")
    with pytest.raises(ValueError, match="keys/order"):
        C.exact_table(a, b, C.A.KEYS[:3], "events")
