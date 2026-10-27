"""Same-straw timing must match the generated particle, not event/straw alone."""
import pandas as pd

from trkfastsim.analyses.v1compare import _same_primary_straw


def test_two_muons_in_one_straw_are_paired_by_generator_identity():
    gen = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1, 1], "idx": [0, 1, 2], "pdg": [2212, 13, -13]})
    fast = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1], "sp": [1, 2], "sid": [10, 10]})
    g4 = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1], "sp": [3, 2], "sid": [10, 10]})
    a, b = _same_primary_straw(fast, g4, gen)
    assert list(zip(a, b)) == [(0, 1), (1, 0)]


def test_different_muons_and_ambiguous_recrossings_do_not_match():
    gen = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1], "idx": [0, 1], "pdg": [13, -13]})
    fast = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1, 1], "sp": [1, 1, 1], "sid": [10, 20, 20]})
    g4 = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1], "sp": [2, 1], "sid": [10, 20]})
    a, b = _same_primary_straw(fast, g4, gen)
    assert len(a) == len(b) == 0
