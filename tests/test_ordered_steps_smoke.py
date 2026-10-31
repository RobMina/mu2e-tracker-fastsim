"""Full-chain smoke keeps generated events whole and refusal explicit."""
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import ordered_steps_smoke as P


def test_smoke_retains_multimuon_event_and_refusal(tmp_path, monkeypatch):
    source, gdml = tmp_path / "gen.root", tmp_path / "geom.gdml"
    source.write_bytes(b"fixture")
    gdml.write_bytes(b"fixture")
    gen = pd.DataFrame(dict(run=[1, 1, 1], subrun=[1, 1, 1], event=[1, 1, 2]))
    monkeypatch.setattr(P.tables, "read", lambda *a: gen.copy())
    monkeypatch.setattr(P.geometry, "from_gdml", lambda *a: None)
    monkeypatch.setattr(P.steps, "TrackerInterior", lambda *a: None)
    monkeypatch.setattr(P.steps, "load_world", lambda *a: None)
    def refuse(generated, *a, **kw):
        assert generated.event.tolist() == [1, 1]
        raise ValueError("geometry refusal")
    monkeypatch.setattr(P.steps, "make_steps", refuse)
    result = P.run(source, gdml, tmp_path / "out", max_events=1)
    assert result["generated_events"] == 1 and result["particles_generated"] == 2
    assert result["status"] == "refused" and not result["admitted_for_production"]
    with pytest.raises(FileExistsError):
        P.run(source, gdml, tmp_path / "out")
