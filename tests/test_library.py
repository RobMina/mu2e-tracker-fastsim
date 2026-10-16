"""Unit tests: the in-straw secondary sampler on a synthetic library."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trkfastsim import library as L                                    # noqa: E402


def _lib():
    # 4 crossings in the same (|p|, path) bin: 0 and 2 have secondaries (1 and 2 steps)
    ins = pd.DataFrame({"xing": [0, 1, 2, 3], "p": 3000.0, "mu_len": 4.8, "nsec": [1, 0, 2, 0]})
    ins["pbin"] = np.digitize(ins.p, L.P_BINS) - 1
    ins["lbin"] = np.digitize(ins.mu_len, L.LEN_BINS) - 1
    st = pd.DataFrame({"xing": [0, 2, 2], "eion": [1e-3, 2e-3, 3e-3], "len": [1.0, 2.0, 3.0], "ion": 1, "shape": 0,
                       "pdg": 11})
    return {"instraw": ins, "instraw_steps": st}


def test_draw_frequency_and_expansion():
    s = L.InStrawSampler(_lib(), np.random.default_rng(3))
    n = 40_000
    xing, st = s.draw(np.full(n, 3000.0), np.full(n, 4.8))
    assert set(np.unique(xing)) == {0, 1, 2, 3}
    assert abs(len(st) / n - (1 + 2) / 4) < 0.02          # mean secondaries per crossing
    assert np.all(np.diff(st.i.to_numpy()) >= 0) and st.i.max() < n


def test_empty_bin_falls_back_to_a_neighbour():
    s = L.InStrawSampler(_lib(), np.random.default_rng(4))
    xing, _ = s.draw(np.array([3000.0]), np.array([2.5]))   # path bin with no library rows
    assert xing[0] in (0, 1, 2, 3)
