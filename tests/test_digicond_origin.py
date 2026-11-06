"""Conditions use the explicit sample detector origin without changing extracted defaults."""
import numpy as np
import pytest
from test_digi import _geom
from trkfastsim.digi import digicond as DC
from trkfastsim.geometry import DETECTOR_ORIGIN_EXTRACTED


def test_explicit_origin_changes_only_translation():
    g = _geom()
    old = DC.sim_straws(g)
    origin = np.array([-3904., 0., 10171.])
    new = DC.sim_straws(g, detector_origin=origin)
    for key in old:
        if key in ("straw_mid", "wire_mid"):
            np.testing.assert_allclose(new[key] - old[key], np.broadcast_to(DETECTOR_ORIGIN_EXTRACTED - origin, old[key].shape))
        else:
            np.testing.assert_array_equal(new[key], old[key])


@pytest.mark.parametrize("origin", [[1., 2.], [1., 2., float("nan")]])
def test_bad_origin_rejected(origin):
    with pytest.raises(ValueError, match="detector_origin"):
        DC.sim_straws(_geom(), detector_origin=origin)
