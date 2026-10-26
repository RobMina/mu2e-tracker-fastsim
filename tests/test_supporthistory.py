"""Bounded CLI runner contracts with synthetic input and stored support geometry."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trkfastsim.analyses import supporthistory as runner
from trkfastsim.transportledger import SegmentHistory
from test_interiorpath import support


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    path, gdml = tmp_path / "seeds.csv", tmp_path / "geometry.gdml"
    pd.DataFrame(dict(run=[1], subrun=[2], event=[3], sp=[4], x=[0.], y=[0.], z=[0.],
                      px=[100.], py=[0.], pz=[0.], length_mm=[6.])).to_csv(path, index=False)
    gdml.write_text("synthetic geometry fixture")
    monkeypatch.setattr(runner.geometry, "from_gdml", lambda p: object())
    monkeypatch.setattr(runner.steps, "TrackerInterior", lambda p, g: SimpleNamespace(shells=support(), calo_skipped={}))
    return dict(seeds=path, gdml=gdml, outdir=tmp_path / "out", tracker_ref="1" * 40, core_ref="2" * 40,
                mode="ordered-support-mean-replace-v1")


def test_runner_persists_provenance_and_refuses_overwrite(inputs):
    summary = runner.run(**inputs)
    assert summary["seed_rows"] == 1 and summary["seed_events"] == 1
    assert summary["complete"]
    h = SegmentHistory.load(inputs["outdir"] / "history.npz")
    assert h.metadata["settings"]["seed_domain"].startswith("supplied seeds only")
    assert h.metadata["geometry_sha256"] == runner.sha256(inputs["gdml"])
    assert "orderedtransport.py" in h.metadata["settings"]["sources_sha256"]["tracker"]
    with pytest.raises(FileExistsError):
        runner.run(**inputs)


def test_row_bound_is_not_silent_truncation(inputs):
    df = pd.read_csv(inputs["seeds"])
    pd.concat([df, df.assign(sp=5)]).to_csv(inputs["seeds"], index=False)
    with pytest.raises(ValueError, match="row bound"):
        runner.run(**inputs, max_rows=1)
    assert not inputs["outdir"].exists()


def test_skipped_calo_solids_are_refused(inputs, monkeypatch):
    monkeypatch.setattr(runner.steps, "TrackerInterior", lambda p, g: SimpleNamespace(shells=support(), calo_skipped={"bad": 1}))
    with pytest.raises(ValueError, match="skipped solids"):
        runner.run(**inputs)
    assert not inputs["outdir"].exists()


def test_cli_wiring(inputs, capsys):
    runner.main([str(inputs["seeds"]), "--gdml", str(inputs["gdml"]), "--outdir", str(inputs["outdir"]),
                 "--tracker-ref", inputs["tracker_ref"], "--core-ref", inputs["core_ref"], "--mode", inputs["mode"],
                 "--timing", "linear-in-path", "--max-step-mm", ".5"])
    assert json.loads(capsys.readouterr().out)["timing"] == "linear-in-path"
