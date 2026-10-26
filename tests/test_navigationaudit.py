"""Geometry-only audit identities, explicit bounds and refusal evidence."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.analyses import navigationaudit as audit
from trkfastsim import materialpath, geometry
from test_materialpath import shells, tube


def hits(rows):
    return pd.DataFrame(rows, columns=["lane", "s0_mm", "s1_mm", "material", "sign"])


def test_sweep_retains_nested_overlaps_and_negative_occupancy():
    h = hits([(0, 0., 10., "Al", 1), (0, 2., 3., "Al", 1), (0, 4., 5., "Cu", 1),
              (1, 1., 2., "Al", -1)])
    _, faults, summary = audit.summarize_occupancy(h)
    assert summary["affected_lanes"] == 2
    assert summary["negative_spans"] == 2
    assert summary["max_fault_length_mm"] == 1.
    assert ((faults.kind == "total") & (faults.s0_mm == 4)).any()


def test_sweep_coincident_cancellation_and_shared_boundaries():
    h = hits([(0, 0., 2., "Al", 1), (0, 1., 2., "Al", -1), (0, 1., 3., "Cu", 1)])
    _, faults, summary = audit.summarize_occupancy(h)
    assert faults.empty and summary["affected_lanes"] == 0
    assert audit.summarize_occupancy(h.iloc[:0])[2]["fault_spans"] == 0


def test_raw_tube_helper_preserves_overlaps_and_empty_schema():
    model = shells(tube(), tube())
    h = materialpath.tube_hits(model, [[0., 0., 0.]], [[1., 0., 0.]], [4.])
    assert len(h) == 2
    assert h.material.tolist() == ["Al", "Al"]
    with pytest.raises(ValueError, match="overlap"):
        materialpath.tube_intervals(model, [[0., 0., 0.]], [[1., 0., 0.]], [4.])
    empty = materialpath.tube_hits(shells(), np.empty((0, 3)), np.empty((0, 3)), [])
    assert empty.dtypes.astype(str).to_dict() == h.dtypes.astype(str).to_dict()


def reference():
    particles = pd.DataFrame(dict(run=[1]*3, subrun=[2]*3, event=[3, 3, 4], key=[5, 6, 7],
                                  ismcprim=[1]*3, pdg=[13, -13, 13]))
    surface = pd.DataFrame(dict(run=[1]*5, subrun=[2]*5, event=[3, 3, 3, 3, 4], sp=[5, 5, 6, 6, 7],
        surf=[0, 2, 0, 4, 0], t=[2., 1., 3., 2., 1.], px=[1.]*5, py=[0.]*5, pz=[0.]*5))
    for prefix in ("0", "1"):
        surface[["x" + prefix, "y" + prefix, "z" + prefix]] = geometry.to_detector(np.zeros((5, 3)))
    return dict(particles=particles, surfsteps=surface, gen=particles[["run", "subrun", "event"]].drop_duplicates())


def test_seed_selection_retains_multiple_primaries_and_explicit_bound():
    seeds, count = audit.reference_seeds(reference(), (np.full(3, -10.), np.full(3, 10.)), 2, 100.)
    assert seeds.sp.tolist() == [5, 6]
    assert seeds.reference_surface_row.tolist() == [1, 3]
    assert seeds.length_mm.tolist() == [10., 10.]
    assert count["bounded_out_muons"] == 1
    assert count["selected_events"] == 1 and count["selected_muons"] == 2


def test_seed_empty_duplicate_invalid_and_outside():
    frames = reference()
    frames["surfsteps"] = frames["surfsteps"].iloc[:0]
    seeds, count = audit.reference_seeds(frames, (np.full(3, -10.), np.full(3, 10.)), 2, 100.)
    assert seeds.empty and count["envelope_muons"] == 0
    frames = reference()
    frames["particles"] = pd.concat([frames["particles"], frames["particles"].iloc[:1]])
    with pytest.raises(ValueError, match="duplicate primary"):
        audit.reference_seeds(frames, (np.full(3, -10.), np.full(3, 10.)), 2, 100.)
    frames = reference()
    frames["surfsteps"].loc[1, "px"] = 0.
    frames["surfsteps"].loc[3, "x0"] += 100.
    _, count = audit.reference_seeds(frames, (np.full(3, -10.), np.full(3, 10.)), 2, 100.)
    assert count["invalid_state_muons"] == 1 and count["outside_bbox_muons"] == 1


@pytest.mark.parametrize("bad_time", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("all_rows", [False, True])
def test_nonfinite_selection_times_are_rejected(bad_time, all_rows):
    frames = reference()
    if all_rows:
        frames["surfsteps"]["t"] = bad_time
    else:
        frames["surfsteps"].loc[0, "t"] = bad_time
    with pytest.raises(ValueError, match="nonfinite primary-envelope time"):
        audit.reference_seeds(frames, (np.full(3, -10.), np.full(3, 10.)), 2, 100.)


def test_reader_selects_highest_cycles_and_refuses_large_tree(tmp_path):
    path = tmp_path / "ref.root"
    with uproot.recreate(path) as root:
        for name, frame in reference().items():
            root[f"dump/{name}"] = {key: frame[key].to_numpy() for key in frame}
        root["dump/gen"] = {key: np.array([9]) for key in ["run", "subrun", "event"]}
    frames, info = audit.read_reference(path)
    assert info["gen"]["cycle"] == 2 and info["gen"]["entries"] == 1
    assert frames["gen"].event.tolist() == [9]
    with pytest.raises(ValueError, match="row bound"):
        audit.read_reference(path, max_tree_rows=1)


def test_witnesses_preserve_signed_contributors_and_positions():
    h = hits([(0, 0., 2., "Al", 1), (0, 1., 3., "Al", 1)])
    h["component"], h["source"] = "tube", ["a", "b"]
    _, bad, _ = audit.summarize_occupancy(h)
    bad["component"] = "combined"
    seeds = pd.DataFrame(dict(x=[0.], y=[0.], z=[0.], px=[1.], py=[0.], pz=[0.]))
    witness = audit.fault_witnesses(bad, h, seeds)
    assert set(witness.source) == {"a", "b"}
    np.testing.assert_allclose(witness.x_mm, 1.5)
    assert len(witness) == 4
    assert audit.fault_witnesses(bad.iloc[:0], h, seeds).empty


def test_omitted_aabb_candidates_preserve_unknowns():
    item = dict(classification="unsupported", placement_path="omitted", material="Al", solid_tag="torus",
                scope="calorimeter", geometry_bounds_mm=[[1, -1, -1], [2, 1, 1]])
    result = dict(records=[item, dict(item, placement_path="unknown", geometry_bounds_mm=None)])
    rows, summary = audit.omitted_candidates(result, np.array([[0., 0., 0.], [0., 2., 0.]]),
                                             np.array([[1., 0., 0.]]*2), np.array([3., 3.]))
    assert rows.lane.tolist() == [0]
    assert summary["unresolved_bounds"] == 1
    assert rows.s0_mm.tolist() == [1.]


def test_legacy_column_comparison_is_labelled_not_admission():
    model = shells(tube())
    o, d, length = np.array([[0., 0., 0.]]), np.array([[1., 0., 0.]]), np.array([4.])
    h = materialpath.tube_hits(model, o, d, length).assign(sign=1, component="tube")
    comparison = audit.column_comparison(model, h, o, d, length)
    assert set(comparison) == {"8", "32", "128"}
    for rows in comparison.values():
        assert rows[0]["max_abs_difference_mm"] < 1e-12


@pytest.mark.parametrize("empty", [False, True])
def test_runner_writes_audit_not_transport_admission(tmp_path, monkeypatch, empty):
    from trkfastsim import navigationcensus
    from trkfastsim.steps import InteriorColumn
    frames = reference()
    if empty:
        frames["surfsteps"] = frames["surfsteps"].iloc[:0]
    path, gdml = tmp_path / "ref.root", tmp_path / "geom.gdml"
    path.write_text("fixture")
    gdml.write_text("fixture")
    monkeypatch.setattr(audit, "read_reference", lambda *a: (frames, {}))
    monkeypatch.setattr(audit.geometry, "from_gdml", lambda *a: object())
    monkeypatch.setattr(audit.crossings, "panel_boxes", lambda *a: (np.array([[-10.]*3]), np.array([[10.]*3])))
    model = shells(tube(), tube())
    interior = SimpleNamespace(shells=InteriorColumn(model), calo_skipped={})
    monkeypatch.setattr(audit.steps, "TrackerInterior", lambda *a: interior)
    def stored(gdml, bbox, o, d, length):
        h = materialpath.tube_hits(model, o, d, length).assign(sign=1, component="tube", source="fixture")
        return h, {}
    monkeypatch.setattr(audit, "stored_hits", stored)
    monkeypatch.setattr(navigationcensus, "census", lambda *a: dict(records=[], counts={}, placement_warnings=[]))
    out = tmp_path / "output"
    summary = audit.run(path, gdml=gdml, outdir=out, tracker_ref="1"*40, core_ref="2"*40)
    assert summary["counts"]["selected_muons"] == (0 if empty else 3)
    assert summary["strict_navigation"]["tube"]["passed"] == empty
    saved = json.loads((out / "summary.json").read_text())
    assert saved["complete"] and not saved["admitted_for_transport"]
    assert not (out / "history.npz").exists()


def test_run_refuses_overwrite_before_read(tmp_path):
    with pytest.raises(FileExistsError):
        audit.run("missing.root", gdml="missing.gdml", outdir=tmp_path, tracker_ref="1"*40, core_ref="2"*40)
    with pytest.raises(ValueError, match="commit hashes"):
        audit.run("missing.root", gdml="missing.gdml", outdir=tmp_path / "new", tracker_ref="short", core_ref="2"*40)
