import numpy as np

from trkfastsim.reco import combo
from trkfastsim.reco.hitreco import FLAG

GOOD = (1 << FLAG["energysel"]) | (1 << FLAG["timesel"]) | (1 << FLAG["radsel"])


def _sh(ev, straw, time, wdist=None):
    n = len(ev)
    wdist = np.zeros(n) if wdist is None else np.asarray(wdist, float)
    f = np.float32
    return {"ev": np.asarray(ev), "sid": np.asarray(straw, np.uint16), "upanel": np.zeros(n, np.int64),
            "straw": np.asarray(straw), "time": np.asarray(time, f), "wdist": wdist.astype(f),
            "uvar": np.full(n, 100.0, f), "timevar": np.full(n, 1.0, f), "flag": np.full(n, GOOD, np.uint32),
            "edep": np.full(n, 1e-3, f), "et0": np.asarray(time, f), "et1": np.asarray(time, f),
            "x": np.full(n, 500.0, f), "y": np.zeros(n, f), "z": np.zeros(n, f),
            "udx": np.ones(n, f), "udy": np.zeros(n, f)}


def test_combo_indices_are_per_event():
    # two events, each one panel: (straws 0, 1 close in time) + (straw 40 alone)
    sh = _sh([0, 0, 0, 1, 1, 1], [0, 1, 40, 0, 1, 40], [10, 12, 500, 10, 12, 500])
    ph = combo.panel_hits(sh)
    assert ph["ncombo"].tolist() == [2, 1, 2, 1]
    assert ph["ind"].tolist() == [0, 1, 2, 0, 1, 2]
    assert ph["ind_file"].tolist() == [0, 1, 2, 3, 4, 5]
