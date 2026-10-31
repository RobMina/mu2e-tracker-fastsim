"""Bounded replay reports full identity refusals without source or lane omission."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import orderedinterior_probe as P


def test_replay_retains_refusal_and_no_overwrite(tmp_path, monkeypatch):
    for name in ("capture.npz", "complete.json", "geom.gdml"):
        (tmp_path / name).write_bytes(b"fixture")
    initial = dict(identity=np.array([[1, 2, 3, 4]]), position_mm=np.zeros((1, 3)),
                   direction=np.array([[0., -1., 0.]]), model_ke_MeV=np.array([20.]))
    monkeypatch.setattr(P, "load_capture", lambda p: ({}, {"interior": (None, {})}))
    monkeypatch.setattr(P, "flatten", lambda *a: initial)
    monkeypatch.setattr(P.geometry, "from_gdml", lambda p: None)
    def refuse(*a, **kw):
        raise ValueError("cross-medium conflict")
    monkeypatch.setattr(P.steps, "TrackerInterior", lambda *a: SimpleNamespace(
        transport=refuse, calo_skipped={("torus", "example"): 2}))
    original = P.steps._kink_core
    result = P.run(tmp_path, tmp_path / "geom.gdml", tmp_path / "out", scatter=False)
    assert result["status"] == "refused" and result["identities"] == [[1, 2, 3, 4]]
    assert result["reason"] == "cross-medium conflict"
    assert result["admitted_for_production"] is False
    assert P.steps._kink_core is original
    with pytest.raises(FileExistsError):
        P.run(tmp_path, tmp_path / "geom.gdml", tmp_path / "out")
