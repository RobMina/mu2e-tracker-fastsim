"""Combined stored-support navigation and opt-in straight history integration."""
import numpy as np
import pytest

from mu2efastsim.world import CoaxialShells, SignedPieceSet
from trkfastsim.interiorpath import interior_intervals
from trkfastsim.steps import InteriorColumn
from test_convexpath import box
from test_materialpath import tube, assert_spans



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


def test_combined_empty_and_zero_length():
    geometry = support()
    assert interior_intervals(geometry, np.empty((0, 3)), np.empty((0, 3)), []).empty
    assert interior_intervals(geometry, [[0, 0, 0]], [[1, 0, 0]], 0.).empty
