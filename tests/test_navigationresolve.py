"""Saved-audit replay contract; no actual transport or default changes."""
import json

import numpy as np
import pandas as pd
import pytest

from trkfastsim.analyses import navigationresolve as replay


def raw():
    return pd.DataFrame(dict(lane=[0, 0], s0_mm=[0., 1.-1e-11], s1_mm=[1., 2.],
                             material=["Al", "Cu"], sign=[1, 1], component=["tube", "tube"]))


def test_explicit_budget_arm_separation():
    products, summaries = replay.replay_boundaries(raw(), [2.], [0., 1e-10, 1e-9])
    assert [r["partition_passed"] for r in summaries] == [False, True, True]
    assert summaries[1]["evidence"]["actual_max_shift_mm"] < 1e-10
    assert len(products) == 3
    with pytest.raises(ValueError, match="unique nonempty"):
        replay.replay_boundaries(raw(), [2.], [1e-9, 1e-9])


def test_sub_resolution_source_collapse_is_refused():
    hit = raw().iloc[:1].assign(s0_mm=0., s1_mm=1e-12)
    _, summary = replay.replay_boundaries(hit, [2.], [1e-9])
    assert not summary[0]["coalesced"]


@pytest.fixture
def saved(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text("synthetic fixture")
    pd.DataFrame(dict(run=[1], subrun=[2], event=[3], sp=[4], x=[0.], y=[0.], z=[0.],
                      px=[1.], py=[0.], pz=[0.], length_mm=[2.])).to_csv(audit / "seeds.csv", index=False)
    raw().to_csv(audit / "raw_hits.csv", index=False)
    (audit / "loader_census.json").write_text(json.dumps(dict(records=[])))
    prior = dict(schema="trkfastsim.navigation-audit.v1", complete=True, gdml=dict(sha256=replay.sha256(gdml)),
                 counts=dict(selected_muons=1), tracker_ref="1"*40, core_ref="2"*40)
    (audit / "summary.json").write_text(json.dumps(prior))
    return audit, gdml, tmp_path / "output"


def test_saved_csv_round_trip_and_bounds(saved):
    audit, gdml, _ = saved
    _, hits, _, _, inputs = replay.read_saved(audit, gdml)
    np.testing.assert_array_equal(hits[["s0_mm", "s1_mm"]], raw()[["s0_mm", "s1_mm"]])
    assert set(inputs) == {"summary.json", "seeds.csv", "raw_hits.csv", "loader_census.json"}
    with pytest.raises(ValueError, match="row bound"):
        replay.read_saved(audit, gdml, max_hit_rows=1)
    gdml.write_text("changed")
    with pytest.raises(ValueError, match="GDML hash"):
        replay.read_saved(audit, gdml)


def test_no_overwrite_and_no_transport_admission(saved):
    audit, gdml, out = saved
    result = replay.run(audit, gdml=gdml, outdir=out, skip_omitted=True)
    assert result["complete"]
    summary = json.loads((out / "summary.json").read_text())
    assert not summary["admitted_for_transport"]
    assert summary["omitted"] is None
    with pytest.raises(FileExistsError):
        replay.run(audit, gdml=gdml, outdir=out, skip_omitted=True)


def test_parent_material_is_replaced_not_added():
    spans = pd.DataFrame(dict(lane=[0], s0_mm=[1.], s1_mm=[2.], state=["inside"],
        ancestors_verified=[True], leaf_verified=[True], material=["Al"], replaces_material=["Honeycomb"]))
    stored = pd.DataFrame(dict(lane=[0], s0_mm=[0.], s1_mm=[3.], material=["Honeycomb"]))
    pairs, result = replay.replacement_check(spans, stored, [3.])
    assert pairs.expected_parent.all() and result["replacement_partition_passed"]
    wrong = stored.assign(material="Cu")
    _, result = replay.replacement_check(spans, wrong, [3.])
    assert result["unexpected_parent_cells"] == 1 and not result["replacement_partition_passed"]
    with pytest.raises(ValueError, match="pair bound"):
        replay.replacement_check(spans, stored, [3.], max_pairs=0)


def test_resolution_comparison_keeps_unresolved_places():
    fine = [dict(source="a", resolved=True, solid_lower_mm=1., solid_upper_mm=1.01, hit_lanes=1),
            dict(source="b", resolved=False)]
    coarse = [dict(fine[0], solid_lower_mm=.99, solid_upper_mm=1.02), fine[1]]
    result = replay.compare_resolution(fine, coarse)
    assert result[0]["bounds_overlap"] and result[0]["refined_width_no_larger"]
    assert not result[1]["compared"]


def test_actual_probe_empty_fixture_is_explicit(saved, monkeypatch):
    audit, gdml, out = saved
    def empty(*a, **kw):
        return pd.DataFrame(columns=["lane", "s0_mm", "s1_mm", "state", "ancestors_verified", "leaf_verified", "replaces_material", "scope", "material"]), []
    monkeypatch.setattr(replay.omittedpath, "probe", empty)
    summary = replay.run(audit, gdml=gdml, outdir=out)
    assert summary["omitted"]["placements"] == 0
    assert (out / "omitted_solid_intervals.csv").exists()
