"""Persisted table identity never hides changed values or ragged populations."""
import numpy as np
import pytest

import digi_tables_identical as check
from trkfastsim.analyses.fastdigi import write_digis


def table(path, tdc=10):
    keys = np.array([[1400, 21, 1], [1400, 21, 2]])
    d = dict(ev=np.array([1]), sid=np.array([520]), tdc=np.full((1, 2), tdc),
             tot=np.ones((1, 2), int), pmp=np.array([5]), adc=np.ones((1, 15), int),
             step=np.zeros((1, 2), int))
    write_digis(path, keys, d, 15)


def test_tables_identity_and_diagnostic_difference(tmp_path):
    a, b = tmp_path / "a.root", tmp_path / "b.root"
    table(a)
    table(b)
    assert check.compare(a, b)["identical"]
    table(b, 11)
    with pytest.raises(ValueError, match="persisted columns differ"):
        check.compare(a, b)
    got = check.compare(a, b, differences=True)
    assert not got["identical"]
    assert got["events"] == 2 and got["digis"] == 1
    assert got["differences"]["d_tdc0"] == dict(different=1, total=1, max_abs=1)
