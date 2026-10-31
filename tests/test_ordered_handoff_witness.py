"""Handoff coordinates use actual local cell offsets and require valid forward states."""
import numpy as np
import pandas as pd
import pytest

from scripts.ordered_handoff_witness import cell_geometry
from trkfastsim import geometry


def test_nonzero_slice_path_and_world_frame():
    cells = pd.DataFrame(dict(path=[100.], cell_start_path_mm=[102.], cell_end_path_mm=[105.],
        x=[1.], y=[2.], z=[3.], ux=[0.], uy=[-1.], uz=[0.]))
    a, b, direction, length = cell_geometry(cells)
    np.testing.assert_allclose(a, geometry.to_mu2e([[1., 0., 3.]]))
    np.testing.assert_allclose(b, geometry.to_mu2e([[1., -3., 3.]]))
    np.testing.assert_array_equal(length, [3.])
    with pytest.raises(ValueError, match="forward"):
        cell_geometry(cells.assign(cell_end_path_mm=101.))
    with pytest.raises(ValueError, match="unit"):
        cell_geometry(cells.assign(uy=-2.))
