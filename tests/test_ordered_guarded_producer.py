"""Whole-event refusal publication/primary inventory and immutable source gate contracts."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from scripts import ordered_guarded_producer as P


def setup(tmp_path, monkeypatch):
    gen_path, gdml, source = tmp_path / "gen.root", tmp_path / "geom.gdml", tmp_path / "reviewed"
    gen_path.write_bytes(b"fixture")
    gdml.write_bytes(b"fixture")
    source.mkdir()
    for n in ("navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope"):
        (source / (n + ".py")).write_text("fixture")
    gen = pd.DataFrame(dict(run=[1, 1, 1], subrun=[2, 2, 2], event=[3, 3, 4], pdg=[13, -13, 13],
        x=[0.]*3, y=[1000.]*3, z=[0.]*3, px=[0.]*3, py=[-80.]*3, pz=[0.]*3, t=[7.]*3))
    monkeypatch.setattr(P.tables, "read", lambda *a: gen.copy())
    monkeypatch.setattr(P.geometry, "from_gdml", lambda *a: None)
    monkeypatch.setattr(P.steps, "TrackerInterior", lambda *a: SimpleNamespace())
    monkeypatch.setattr(P.crossings, "panel_boxes", lambda *a: (np.zeros((1, 3)), np.ones((1, 3))))
    monkeypatch.setattr(P, "load_pair_sources", lambda *a: dict(navigationcensus=SimpleNamespace(census=lambda *a: {})))
    monkeypatch.setattr(P.orderedhandoff, "plan", lambda *a, **k: dict(policy="fixture", plane_y_mm=870.))
    monkeypatch.setattr(P.steps, "load_world", lambda *a, **k: None)
    monkeypatch.setattr(P, "StoredHitQuery", lambda *a: None)
    monkeypatch.setattr(P, "save_navigation", lambda *a: None)
    monkeypatch.setattr(P.guardednavigation, "GuardedNavigator", lambda **k: SimpleNamespace(policy="fixture", calls=[]))
    return gen_path, gdml, source


def test_refused_whole_event_has_no_partial_physics_and_preserves_all_input_primaries(tmp_path, monkeypatch):
    gen, gdml, source = setup(tmp_path, monkeypatch)
    def refuse(generated, *a, **kw):
        assert len(generated) == 2 and generated.event.tolist() == [3, 3]
        refuse.handoff_ledger = None
        raise ValueError("whole batch geometry refused")
    monkeypatch.setattr(P.steps, "make_steps", refuse)
    out = tmp_path / "out"
    result = P.run(gen, gdml, source, out, max_events=1, arm="inner")
    assert result["status"] == "refused" and result["input_primary_muons"] == 2
    assert not (out / "ordered_steps.root").exists()
    assert len(pd.read_csv(out / "input_primaries.csv")) == 2
    assert json.loads((out / "complete.json").read_text())["handoff_evidence_status"] == "not-yet-available"
    with pytest.raises(FileExistsError):
        P.run(gen, gdml, source, out, max_events=1, arm="inner")


def test_missing_material_refuses_with_evidence_not_unhandled_abort(tmp_path, monkeypatch):
    gen, gdml, source = setup(tmp_path, monkeypatch)
    def refuse(*a, **kw):
        refuse.handoff_ledger = None
        raise KeyError("explicit material constants missing")
    monkeypatch.setattr(P.steps, "make_steps", refuse)
    out = tmp_path / "missing"
    result = P.run(gen, gdml, source, out, max_events=1, arm="inner")
    assert result["status"] == "refused" and not (out / "ordered_steps.root").exists()
    assert "constants missing" in result["reason"] and (out / "complete.json").is_file()


def test_changed_input_never_publishes_completion_or_physics(tmp_path, monkeypatch):
    gen, gdml, source = setup(tmp_path, monkeypatch)
    def refuse(*a, **kw):
        refuse.handoff_ledger = None
        gdml.write_text("changed")
        raise ValueError("refused")
    monkeypatch.setattr(P.steps, "make_steps", refuse)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="immutable inputs"):
        P.run(gen, gdml, source, out, max_events=1, arm="inner")
    assert not (out / "complete.json").exists() and not (out / "ordered_steps.root").exists()
