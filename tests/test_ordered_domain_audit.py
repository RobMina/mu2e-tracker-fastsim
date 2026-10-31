"""Synthetic diagnostic contracts only; these do not certify geometry or transport."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import ordered_domain_audit as audit


def test_box_complete_forward_intervals_parallel_outside_and_boundaries():
    origin = np.array([[-3., 0., 0.], [3., 0., 0.], [0., 2., 0.],
                       [-1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [-2., 0., 0.]])
    direction = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
                          [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    entry, exit, hit = audit.ray_box(origin, direction, [-1.] * 3, [1.] * 3)
    np.testing.assert_array_equal(hit, [True, False, False, True, True, True, True])
    np.testing.assert_allclose(entry[hit], [2., 0., 0., 0., 1.])
    np.testing.assert_allclose(exit[hit], [4., 2., 0., 1., 1.])
    assert np.isnan(entry[~hit]).all() and np.isnan(exit[~hit]).all()
    split = audit.split_interval(np.zeros(7), np.full(7, 3.), entry, exit, hit)
    np.testing.assert_allclose(split["inside_mm"], [1., 0., 0., 2., 0., 1., 0.])
    np.testing.assert_allclose(split["outside_mm"], [2., 3., 3., 1., 3., 2., 3.])
    assert split["outside_before_entry_mm"][0] == 2.
    assert split["outside_after_exit_mm"][3] == 1.


def test_cylinder_axial_parallel_tangent_and_outside_approaches():
    origin = np.array([[0., 0., -5.], [2., 0., -5.], [3., 0., 0.], [3., 0., 0.],
                       [-2., 1., 0.], [0., 0., 3.], [1., 0., 0.], [0., 0., -2.]])
    direction = np.array([[0., 0., 1.], [0., 0., 1.], [-1., 0., 0.], [1., 0., 0.],
                          [1., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 0., -1.]])
    entry, exit, hit = audit.ray_cylinder(origin, direction, 1., -2., 2.)
    np.testing.assert_array_equal(hit, [True, False, True, False, True, False, True, True])
    np.testing.assert_allclose(entry[hit], [3., 2., 2., 0., 0.])
    np.testing.assert_allclose(exit[hit], [7., 4., 2., 2., 0.])
    assert np.isnan(exit[3])  # Outside pointing away is not a fabricated exit at zero.
    split = audit.split_interval(0., np.full(8, 10.), entry, exit, hit)
    np.testing.assert_allclose(split["inside_mm"], [4., 0., 2., 0., 0., 0., 2., 0.])


def test_cylinder_stable_roots_and_no_arbitrary_horizon():
    entry, exit, hit = audit.ray_cylinder([[1.e8, 0., 0.]], [[-1., 0., 0.]], 1., -2., 2.)
    assert hit[0]
    np.testing.assert_allclose(entry, [1.e8 - 1.], rtol=0, atol=1.e-7)
    np.testing.assert_allclose(exit, [1.e8 + 1.], rtol=0, atol=1.e-7)
    entry, exit, hit = audit.ray_box([[-1.e6, 0., 0.]], [[1., 0., 0.]], [-1.] * 3, [1.] * 3)
    np.testing.assert_array_equal(entry, [999999.])
    np.testing.assert_array_equal(exit, [1000001.])
    assert hit[0]


@pytest.mark.parametrize("function,args", [(audit.ray_box, ([-1.] * 3, [1.] * 3)),
                                           (audit.ray_cylinder, (1., -2., 2.))])
def test_invalid_and_empty_rays(function, args):
    with pytest.raises(ValueError, match="nonzero"):
        function([[0., 0., 0.]], [[0., 0., 0.]], *args)
    with pytest.raises(ValueError, match="finite"):
        function([[np.nan, 0., 0.]], [[1., 0., 0.]], *args)
    entry, exit, hit = function(np.empty((0, 3)), np.empty((0, 3)), *args)
    assert entry.size == exit.size == hit.size == 0


def vertices():
    # Same event/sp in another subrun and another sp in the same event must remain distinct.
    return pd.DataFrame(dict(run=[1400] * 6, subrun=[22, 22, 22, 23, 23, 22],
        event=[349] * 6, sp=[1, 1, 1, 1, 1, 2], path=[10., 15., 20., 0., 2., 0.],
        x=[-3., 2., 7., 0., 2., 3.], y=0., z=0., ux=1., uy=0., uz=0.))


def domains(v=None):
    lower = audit.geometry.to_mu2e([-1., -1., -1.])
    upper = audit.geometry.to_mu2e([1., 1., 1.])
    return audit.domain_columns(audit.prepare_vertices(vertices() if v is None else v), lower, upper, (1., -1., 1.))


def witness_rows():
    return pd.DataFrame(dict(run=1400, subrun=22, event=349, sp=1, path=10.,
        x=-3., y=0., z=0., ux=1., uy=0., uz=0., cell_start_path_mm=[11., 12., 14.],
        cell_end_path_mm=[12., 14., 15.], state=["unresolved", "inside", "inside"],
        material="concrete", source="World/concrete", ancestors_verified=True, replaces_material="vacuum"))


def test_omission_offsets_split_inside_outside_and_unresolved_retained():
    v = domains()
    cells = audit.omission_splits(witness_rows(), v)
    np.testing.assert_allclose(cells.cell_local_start_mm, [1., 2., 4.])
    np.testing.assert_allclose(cells.cell_length_mm, [1., 2., 1.])
    assert cells.state.tolist() == ["unresolved", "inside", "inside"]
    for name in audit.DOMAINS:
        np.testing.assert_allclose(cells[f"{name}_inside_mm"], [0., 2., 0.])
        np.testing.assert_allclose(cells[f"{name}_outside_mm"], [1., 0., 1.])
        np.testing.assert_allclose(cells[f"{name}_outside_before_entry_mm"], [1., 0., 0.])
        np.testing.assert_allclose(cells[f"{name}_outside_after_exit_mm"], [0., 0., 1.])
    assert cells.subrun.tolist() == [22, 22, 22]


def test_witness_nonzero_slice_and_tiny_unresolved_band():
    v = vertices().iloc[:3].copy()
    v["path"] = [100., 200., 4000.]
    v["x"] = [0., 100., 3900.]
    v = domains(v)
    band = 1.864e-8
    w = witness_rows().iloc[:2].copy()
    w["path"] = 200.
    w["x"] = 100.
    w["cell_start_path_mm"] = [3529.2808214475, 3529.2808214475 + band]
    w["cell_end_path_mm"] = [3529.2808214475 + band, 4000.]
    result = audit.omission_splits(w, v)
    np.testing.assert_allclose(result.cell_local_start_mm, w.cell_start_path_mm - 200., rtol=0, atol=0)
    assert result.state.iloc[0] == "unresolved"
    assert 0. < result.cell_length_mm.iloc[0] < 2.e-8
    np.testing.assert_allclose(result.loader_aabb_outside_mm, result.cell_length_mm)
    assert result.vd_cylinder_inside_mm.sum() == 0.


def test_gas_and_lane_coverage_keeps_no_gas_zero_length_and_unobserved_lanes():
    v = domains()
    seg = pd.DataFrame(dict(run=1400, subrun=[22, 23], event=349, sp=1, path_mm=[15., 0.]))
    entries = audit.gas_entries(seg, v)
    assert entries.loader_aabb_outside.tolist() == [True, False]
    assert entries.vd_cylinder_outside.tolist() == [True, False]
    mu = pd.DataFrame(dict(run=1400, subrun=[22, 23, 22, 24], event=349, sp=[1, 1, 2, 1],
                           path_interior_mm=[20., 2., 0., np.nan]))
    lanes = audit.lane_coverage(mu, v, entries, audit.omission_splits(witness_rows(), v))
    assert len(lanes) == 4
    np.testing.assert_array_equal(lanes.vertices, [3, 2, 1, 0])
    np.testing.assert_array_equal(lanes.slices, [2, 1, 0, 0])
    np.testing.assert_array_equal(lanes.gas_entries, [1, 1, 0, 0])
    np.testing.assert_allclose(lanes.loader_aabb_inside_mm, [2., 1., 0., 0.])
    np.testing.assert_allclose(lanes.loader_aabb_outside_mm, [8., 1., 0., 0.])
    assert lanes.has_vertices.tolist() == [True, True, True, False]
    terminal = v[(v.subrun == 22) & (v.sp == 1)].iloc[-1]
    assert not terminal.vd_cylinder_forward_hit and np.isnan(terminal.vd_cylinder_exit_mm)


def test_bad_saved_links_and_extents_refuse():
    bad = vertices()
    bad.loc[1, "x"] += 1.
    with pytest.raises(ValueError, match="endpoints disagree"):
        audit.prepare_vertices(bad)
    with pytest.raises(ValueError, match="duplicate"):
        audit.prepare_vertices(pd.concat([vertices(), vertices().iloc[:1]]))
    w = witness_rows()
    w.loc[0, "cell_end_path_mm"] = 16.
    with pytest.raises(ValueError, match="outside its actual"):
        audit.omission_splits(w, domains())
    w = witness_rows()
    w["subrun"] = 99
    with pytest.raises(ValueError, match="exact full-identity"):
        audit.omission_splits(w, domains())
    seg = pd.DataFrame(dict(run=[1400], subrun=[22], event=[349], sp=[1], path_mm=[11.]))
    with pytest.raises(ValueError, match="exact full-identity"):
        audit.gas_entries(seg, domains())
    with pytest.raises(ValueError, match="ordered forward"):
        audit.split_interval(2., 1., 0., 3., True)


def test_saved_run_hash_receipts_outputs_and_no_overwrite(tmp_path, monkeypatch):
    saved = tmp_path / "saved"
    saved.mkdir()
    v = vertices()
    v.to_csv(saved / "vertices.csv", index=False)
    pd.DataFrame(dict(run=[1400], subrun=[22], event=[349], sp=[1], path_mm=[15.])).to_csv(
        saved / "segments.csv", index=False)
    pd.DataFrame(dict(run=1400, subrun=[22, 23, 22, 24], event=349, sp=[1, 1, 2, 1],
                     path_interior_mm=[20., 2., 0., np.nan])).to_csv(saved / "muons.csv", index=False)
    gdml = tmp_path / "explicit_v04.gdml"
    gdml.write_text("synthetic explicit geometry")
    receipt = dict(status="bounded_full_chain_complete", input_sha256={str(gdml): audit.digest(gdml)},
                   source_sha256={"producer.py": "saved-source-hash"})
    (saved / "complete.json").write_text(json.dumps(receipt))
    witness = tmp_path / "witness.json"
    witness.write_text(json.dumps(dict(input_sha256={str(saved / name): audit.digest(saved / name)
        for name in ("vertices.csv", "segments.csv")}, witnesses=witness_rows().to_dict("records"))))
    monkeypatch.setattr(audit.geometry, "from_gdml", lambda _: object())
    monkeypatch.setattr(audit.crossings, "panel_boxes", lambda _: (
        audit.geometry.to_mu2e([[299., 299., 299.]]), audit.geometry.to_mu2e([[-299., -299., -299.]])))
    out = tmp_path / "new"
    result = audit.run(saved, witness, gdml, out)
    assert result["input_sha256_before"] == result["input_sha256_after"]
    assert result["source_sha256_before"] == result["source_sha256_after"]
    assert result["producer_receipt"] == receipt
    assert not result["admitted_for_production"]
    assert result["coverage"]["muon_lanes"] == 4
    assert result["coverage"]["lanes_without_vertices"] == 1
    assert {p.name for p in out.iterdir()} == {"complete.json", "vertices.csv", "lanes.csv", "gas_entries.csv", "omitted_cells.csv"}
    with pytest.raises(FileExistsError):
        audit.run(saved, witness, gdml, out)
    (saved / "segments.csv").write_text((saved / "segments.csv").read_text() + "\n")
    with pytest.raises(ValueError, match="does not bind"):
        audit.run(saved, witness, gdml, tmp_path / "changed")


def test_actual_omission_witness_first_slice_and_outside_vd():
    direction = np.array([.85292125, -.33863173, -.39730831])
    direction /= np.linalg.norm(direction)
    origin = np.array([2534.7298067, 850., 3545.6126005])
    v = pd.DataFrame(dict(run=1400, subrun=22, event=349, sp=1, path=[0., 4000.],
        x=[origin[0], origin[0] + 4000. * direction[0]],
        y=[origin[1], origin[1] + 4000. * direction[1]],
        z=[origin[2], origin[2] + 4000. * direction[2]],
        ux=direction[0], uy=direction[1], uz=direction[2]))
    lower = audit.geometry.to_mu2e([-1000.] * 3)
    upper = audit.geometry.to_mu2e([1000.] * 3)
    v = audit.domain_columns(audit.prepare_vertices(v), lower, upper,
                            (audit.surfaces.R_OUT, audit.surfaces.Z_FRONT, audit.surfaces.Z_BACK))
    start = 3529.2808214475
    band = 1.864e-8
    w = pd.DataFrame(dict(run=1400, subrun=22, event=349, sp=1, path=0.,
        cell_start_path_mm=[start, start + band], cell_end_path_mm=[start + band, 4000.],
        state=["unresolved", "inside"], material="concrete", source="World/concrete",
        ancestors_verified=True, replaces_material="vacuum"))
    cells = audit.omission_splits(w, v)
    assert cells.path.tolist() == [0., 0.]
    assert not cells.vd_cylinder_forward_hit.any()
    assert cells.vd_cylinder_exit_mm.isna().all()
    np.testing.assert_allclose(cells.vd_cylinder_inside_mm, [0., 0.])
    np.testing.assert_allclose(cells.vd_cylinder_outside_mm, cells.cell_length_mm)
    assert cells.cell_length_mm.iloc[0] > 0.
