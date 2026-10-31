"""Opt-in file names and receipts cannot overwrite default physics products."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from trkfastsim.analyses import steps as P


@pytest.mark.parametrize("evaluation", ["entry", "midpoint"])
def test_ordered_file_has_distinct_name_receipt_and_exclusive_output(tmp_path, monkeypatch, evaluation):
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text("synthetic")
    generated = pd.DataFrame(dict(run=[1], subrun=[1], event=[1]))
    rows = generated.assign(sp=1)
    monkeypatch.setattr(P.geometry, "from_gdml", lambda p: None)
    monkeypatch.setattr(P.st, "load_world", lambda *a, **kw: None)
    monkeypatch.setattr(P.st, "TrackerInterior", lambda *a, **kw: SimpleNamespace(
        shells=SimpleNamespace(coax=[], n_coaxial=0, calo=None)))
    monkeypatch.setattr(P.tables, "expand", lambda p: p)
    monkeypatch.setattr(P.tables, "read", lambda *a: generated.copy())
    def make(*a, **kw):
        assert kw["navigation"] == "ordered-stored-mean-v1"
        assert kw.get("ordered_options", {}).get("stopping_evaluation", "entry") == evaluation
        return rows, rows
    make.interior_stats, make.timing, make.path_vertices = {}, {}, None
    monkeypatch.setattr(P.st, "make_steps", make)
    monkeypatch.setattr(P.surfaces, "surface_steps", lambda *a: None)
    written = []
    monkeypatch.setattr(P.core_tables, "write", lambda path, *a, **kw: written.append(path))
    outdir = tmp_path / "output"
    outdir.mkdir()
    options = dict(gdml=str(gdml), stage="C", navigation="ordered-stored-mean-v1", ordered_stopping_evaluation=evaluation)
    P.run(["gen.example.root"], str(outdir), **options)
    name = "fasttrkC_orderedmean" + ("_midpoint" if evaluation == "midpoint" else "") + ".example.root"
    assert written == [str(outdir / name)]
    receipt = json.loads((outdir / (name + ".json")).read_text())
    assert receipt["stopping_evaluation"] == evaluation
    assert receipt["navigation"] == "ordered-stored-mean-v1"
    assert "orderedinterior.py" in receipt["sources"]
    with pytest.raises(FileExistsError):
        P.run(["gen.example.root"], str(outdir), **options)


def test_legacy_cannot_silently_take_signed_endpoint_arm():
    with pytest.raises(ValueError, match="explicit ordered navigation"):
        P.run([], ".", ordered_signed_endpoint_budget_mm=1e-10)


def test_legacy_cannot_silently_take_midpoint_evaluation():
    with pytest.raises(ValueError, match="explicit ordered navigation"):
        P.run([], ".", ordered_stopping_evaluation="midpoint")
