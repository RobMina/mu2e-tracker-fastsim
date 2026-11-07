"""Bounded pilot endpoint and geometry diagnostics."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import entry_pilot as ep


def test_strict_outward_predicate_and_geometric_excess():
    cut = dict(axisXY=[3., -4.], radius=5., zMin=-2., zMax=8.)
    particles = pd.DataFrame(dict(ex=[8., 9., 9., 3., 3., 9.], ey=[-4.] * 6,
                                  ez=[0., 0., 0., 9., 9., 9.],
                                  epx=[1., 1., -1., 0., 0., -1.], epy=[0.] * 6,
                                  epz=[0., 0., 0., 1., -1., 1.]))
    leaving, excess = ep.endpoints(particles, cut)
    np.testing.assert_array_equal(leaving, [False, True, False, True, False, True])
    np.testing.assert_allclose(excess, [0., 1., 1., 1., 1., np.sqrt(2.)])


@pytest.mark.parametrize("column", ["ex", "epz"])
def test_nonfinite_endpoint_rejected(column):
    row = dict(ex=0., ey=0., ez=0., epx=1., epy=0., epz=0.)
    row[column] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        ep.endpoints(pd.DataFrame([row]), dict(axisXY=[0., 0.], radius=1., zMin=-1., zMax=1.))


def test_geometry_difference_report(monkeypatch):
    @dataclass
    class Geometry:
        mid: np.ndarray
        nstraws: int = 1
        wall_rmax: float = None
    models = {"a": Geometry(np.zeros((1, 3))), "b": Geometry(np.ones((1, 3)) * 0.01)}
    monkeypatch.setattr(ep.geometry, "from_gdml", models.__getitem__)
    assert ep.geometry_check("a", "a")["within_1e_minus_6"]
    result = ep.geometry_check("a", "b")
    assert not result["within_1e_minus_6"]
    assert result["max_abs_differences"]["mid"] == 0.01
