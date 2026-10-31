"""Omission AABB candidates remain candidates; unknown bounds remain visible."""
import numpy as np

from scripts.ordered_geometry_exposure import exposures


def test_slab_candidates_and_unknown_bounds():
    record = dict(classification="unsupported", geometry_bounds_mm=[[1., -1., -1.], [2., 1., 1.]],
                  path="World/pipe", material="G4_Cu", solid="pipe", solid_tag="torus")
    unknown = dict(record, path="World/unknown", geometry_bounds_mm=None)
    rows, missing = exposures([record, unknown], np.array([[0., 0., 0.], [0., 2., 0.]]),
        np.array([[1., 0., 0.], [1., 0., 0.]]), np.array([3., 3.]), np.array([7, 8]))
    assert rows.lane.tolist() == [7]
    assert rows.s0_mm.tolist() == [1.] and rows.s1_mm.tolist() == [2.]
    assert len(missing) == 1 and missing[0]["path"] == "World/unknown"


def test_vertices_reconstruct_only_traversed_slices(tmp_path):
    import pandas as pd
    import pytest
    from scripts.ordered_geometry_exposure import saved_slices
    vertices = pd.DataFrame(dict(run=1, subrun=1, event=1, sp=1, path=[0., 2., 5.],
        x=[0., 2., 5.], y=0., z=0., ux=1., uy=0., uz=0.))
    vertices.to_csv(tmp_path / "vertices.csv", index=False)
    _, _, lengths, lane = saved_slices(tmp_path, {"status": "bounded_full_chain_complete"})
    np.testing.assert_array_equal(lengths, [2., 3.])
    np.testing.assert_array_equal(lane, [0, 0])
    vertices.loc[2, "x"] += 1.
    vertices.to_csv(tmp_path / "vertices.csv", index=False)
    with pytest.raises(ValueError, match="endpoints disagree"):
        saved_slices(tmp_path, {"status": "bounded_full_chain_complete"})
