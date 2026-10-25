import numpy as np
import pytest

from test_reco_combo import _sh
from trkfastsim.reco import combo, linefind
from trkfastsim.reco.hitreco import FLAG


@pytest.mark.parametrize("case", ["empty", "dead", "unselected"])
def test_panel_hits_empty_result_preserves_schema(case):
    reference = combo.panel_hits(_sh([0], [1], [10]))
    sh = _sh([0], [1], [10])
    if case == "empty":
        sh = {key: value[:0] for key, value in sh.items()}
    if case == "dead":
        sh["flag"] |= np.uint32(1 << FLAG["dead"])
    elif case == "unselected":
        sh["flag"][:] = 0
    out = combo.panel_hits(sh)
    assert out.keys() == reference.keys()
    for key, value in out.items():
        assert value.shape == (0,), key
        assert value.dtype == reference[key].dtype, key


@pytest.mark.parametrize("case", ["empty", "dead", "noisy", "one_hit"])
def test_time_clusters_empty_result_preserves_schema(case):
    reference = linefind.time_clusters(_sh([0, 0], [1, 2], [10, 11]))
    sh = _sh([0], [1], [10])
    if case == "empty":
        sh = {key: value[:0] for key, value in sh.items()}
    if case in ("dead", "noisy"):
        sh["flag"] |= np.uint32(1 << FLAG[case])
    out = linefind.time_clusters(sh)
    assert out.keys() == reference.keys()
    for key, value in out.items():
        assert value.shape == (0,), key
        assert value.dtype == reference[key].dtype, key
