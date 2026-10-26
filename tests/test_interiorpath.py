"""Combined stored-support navigation and opt-in straight history integration."""
import numpy as np
import pytest

from mu2efastsim import eloss
from mu2efastsim.world import CoaxialShells, SignedPieceSet
from trkfastsim.interiorpath import interior_intervals
from trkfastsim.steps import InteriorColumn, TrackerInterior
from test_convexpath import box
from test_materialpath import tube, assert_spans
from test_orderedtransport import metadata, deterministic


def support():
    shells = CoaxialShells([tube("G4_Al", ri=2, ro=3, hz=2)], axis_point=[0, 0, 0])
    calo = SignedPieceSet([(1, *box("G4_Cu", x0=4, x1=5))])
    return InteriorColumn(shells, calo)


def test_combined_navigation_and_legacy_columns():
    geometry = support()
    origin, direction, length = np.array([[0., 0., 0.]]), np.array([[1., 0., 0.]]), np.array([6.])
    spans = interior_intervals(geometry, origin, direction, length)
    assert_spans(spans, [(0, 2, "vacuum"), (2, 3, "G4_Al"), (3, 4, "vacuum"), (4, 5, "G4_Cu"), (5, 6, "vacuum")])
    columns = geometry.column(origin, direction, length)
    for material in ("G4_Al", "G4_Cu"):
        selected = spans[spans.material == material]
        np.testing.assert_allclose((selected.s1_mm - selected.s0_mm).sum(), columns[material][0])


def test_cross_representation_overlap_is_not_silently_resolved():
    geometry = support()
    geometry.calo = SignedPieceSet([(1, *box("G4_Cu", x0=2.5, x1=3.5))])
    with pytest.raises(ValueError, match="overlap"):
        interior_intervals(geometry, [[0, 0, 0]], [[1, 0, 0]], 6.)


def test_tracker_api_constructs_real_history(metadata, monkeypatch):
    deterministic(monkeypatch)
    interior = TrackerInterior.__new__(TrackerInterior)
    interior.shells = support()
    result = interior.transport_support_history(np.random.default_rng(8), np.array([[1, 2, 3, 4]]),
        [[0, 0, 0]], [[1, 0, 0]], [20.], [6.], metadata,
        mode="ordered-support-mean-replace-v1", timing="linear-in-path")
    np.testing.assert_allclose(result["final_ke_MeV"], [18.])
    np.testing.assert_allclose(result["position_mm"], [[6., 0., 0.]])
    assert result["history"]["dt_status"].tolist() == ["known"] * len(result["history"])
    np.testing.assert_allclose(result["history"].accounting()["primary_residual_MeV"], 0., atol=1e-13)


def test_combined_empty_and_zero_length():
    geometry = support()
    assert interior_intervals(geometry, np.empty((0, 3)), np.empty((0, 3)), []).empty
    assert interior_intervals(geometry, [[0, 0, 0]], [[1, 0, 0]], 0.).empty
