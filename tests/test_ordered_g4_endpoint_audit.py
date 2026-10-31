"""Endpoint diagnostics do not turn a finite fast path into a stopping agreement."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss
from scripts.ordered_g4_endpoint_audit import compare


def test_stopped_disagreement_and_full_identity_coverage():
    initial = pd.DataFrame(dict(run=1, subrun=1, event=[1, 2], sp=1, t=7., x0=0., y0=0., z0=0.))
    final = dict(ke=np.array([.5, 10.]), stopped=np.array([True, False]), dt=np.array([1., 2.]),
                 path=np.array([10., 20.]), o=np.array([[1., 2., 3.], [1., 2., 3.]]))
    reference = initial[["run", "subrun", "event", "sp"]].assign(ee=[eloss.MMU + 2., eloss.MMU],
                                                                ex=1., ey=2., ez=3., et=8.)
    result, rows = compare(initial, final, reference)
    assert result["fast_stop_g4_not_stopped"] == 1
    assert result["finite_path_but_g4_stopped"] == 1
    assert result["both_stopped"] == 0
    np.testing.assert_array_equal(rows.endpoint_distance_mm, [0., 0.])
    with pytest.raises(ValueError, match="missing reference"):
        compare(initial, final, reference.iloc[:1])
