"""Bounded candidate runner provenance, full-identity output and fail-closed tests."""
import json

import numpy as np
import pandas as pd
import pytest

from trkfastsim.analyses import supportnavigation as runner, navigationresolve
from trkfastsim.navigationcensus import census


@pytest.fixture
def saved(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text('''<gdml><solids><box name="box" x="100" y="100" z="100"/></solids>
    <structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="box"/></volume></structure>
    <setup><world ref="World"/></setup></gdml>''')
    seeds = pd.DataFrame(dict(run=[1, 1], subrun=[2, 2], event=[3, 3], sp=[4, 5], x=[0., 0.],
        y=[0., 1.], z=[0., 0.], px=[1., 1.], py=[0., 0.], pz=[0., 0.], length_mm=[2., 2.]))
    seeds.to_csv(audit / "seeds.csv", index=False)
    pd.DataFrame(columns=["lane", "s0_mm", "s1_mm", "material", "source", "sign"]).to_csv(audit / "raw_hits.csv", index=False)
    (audit / "loader_census.json").write_text(json.dumps(census(gdml, [[-10.]*3, [10.]*3])))
    prior = dict(schema="trkfastsim.navigation-audit.v1", complete=True, gdml=dict(sha256=runner.sha256(gdml)),
                 counts=dict(selected_muons=2), tracker_ref="1"*40, core_ref="2"*40)
    (audit / "summary.json").write_text(json.dumps(prior))
    reviewed = tmp_path / "reviewed"
    navigationresolve.run(audit, gdml=gdml, outdir=reviewed, min_cell_mm=1e-7)
    return dict(audit_dir=audit, resolution_dir=reviewed, gdml=gdml, outdir=tmp_path / "candidate")


def test_replay_no_transport_and_no_overwrite_multiple_primaries(saved):
    result = runner.run(**saved)
    assert result["complete"] and result["seed_rows"] == 2
    manifest = json.loads((saved["outdir"] / "summary.json").read_text())
    assert manifest["seed_events"] == 1 and not manifest["admitted_for_transport"]
    assert manifest["reviewed_cells_and_reports_identical"] and manifest["regenerated_census_identical"]
    assert len(manifest["sources_sha256"]["tracker"]["supportnavigation.py"]) == 64
    assert len(manifest["reviewed_resolution_inputs"]) == 3
    for arm in manifest["candidates"]:
        assert arm["resolved_rays"] == 2 and arm["all_rays_resolved"]
        assert arm["invariance"]["reversed_source_and_cell_order_identical"]
        assert all(x["same_lane_decisions"] for x in arm["stored_endpoint_perturbations"])
    table = pd.read_csv(next(saved["outdir"].glob("*_intervals.csv")))
    assert table.sp.tolist() == [4, 5] and table.event.tolist() == [3, 3]
    assert table.material.tolist() == ["vacuum", "vacuum"]
    with pytest.raises(FileExistsError):
        runner.run(**saved)


@pytest.mark.parametrize("change,match", [("gdml", "GDML hash"), ("audit", "different saved audit"),
    ("incomplete", "completed navigation resolution"), ("settings", "primary probe settings"),
    ("cells", "columns"), ("reports", "reports differ")])
def test_saved_resolution_mismatch_refused(saved, change, match):
    path = saved["resolution_dir"] / "summary.json"
    summary = json.loads(path.read_text())
    if change == "gdml":
        with saved["gdml"].open("a") as stream:
            stream.write("\n")
    elif change == "audit":
        with (saved["audit_dir"] / "seeds.csv").open("a") as stream:
            stream.write("\n")
    elif change == "incomplete":
        summary["complete"] = False
        path.write_text(json.dumps(summary))
    elif change == "settings":
        saved["min_cell_mm"] = 2e-7
    elif change == "cells":
        (saved["resolution_dir"] / "omitted_solid_intervals.csv").write_text("wrong\n")
    else:
        (saved["resolution_dir"] / "omitted_placements.json").write_text('[{"source":"orphan"}]')
    with pytest.raises((ValueError, KeyError, AssertionError), match=match):
        runner.run(**saved)
    assert not saved["outdir"].exists()


@pytest.mark.parametrize("budgets", [[], [0., 0.], [-1.], [np.nan], [np.inf]])
def test_bad_budgets_refused(saved, budgets):
    with pytest.raises(ValueError, match="budgets"):
        runner.run(**saved, budgets=budgets)
    assert not saved["outdir"].exists()


def test_current_census_drift_refused(saved, monkeypatch):
    from trkfastsim import navigationcensus
    monkeypatch.setattr(navigationcensus, "census", lambda *a: {})
    with pytest.raises(ValueError, match="saved census differs"):
        runner.run(**saved)
    assert not saved["outdir"].exists()


def test_duplicate_primary_identity_refused(saved):
    path = saved["audit_dir"] / "seeds.csv"
    table = pd.read_csv(path)
    table["sp"] = 4
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique finite integers"):
        runner.run(**saved)


def test_pair_limit_and_probe_limit_refused(saved):
    with pytest.raises(ValueError, match="max_pairs"):
        runner.run(**saved, max_pairs=0)
    with pytest.raises(ValueError, match="primary probe settings"):
        runner.run(**saved, max_cells=1)


def test_partial_hit_identity_schema_is_not_relabelled(saved):
    path = saved["audit_dir"] / "raw_hits.csv"
    hits = pd.read_csv(path)
    hits["event"] = pd.Series(dtype="int64")
    hits.to_csv(path, index=False)
    summary_path = saved["resolution_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["inputs"]["raw_hits.csv"]["sha256"] = runner.sha256(path)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="partial raw-hit identity schema"):
        runner.run(**saved)
    assert not saved["outdir"].exists()
