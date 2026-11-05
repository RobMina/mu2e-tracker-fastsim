"""Fast CE seeds share one denominator and baseline, never independent additional events."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import ceseeds as cs


def frame():
    return pd.DataFrame({"run": 1, "subrun": 0, "event": np.arange(2**54, 2**54 + 4, dtype=np.uint64),
                         "g4_track": [True, True, False, False], "chain_track": [True, True, False, False],
                         "g4_steps": 20, "chain_steps": 20, "g4_tracker_term": [True, True, False, False],
                         "chain_tracker_term": [True, True, False, False]})


def test_identical_seeds_have_no_spread_or_extra_denominator():
    r = cs.report({"a": frame(), "b": frame()})
    assert r["events"] == 4
    assert r["descriptive_seed_sd"] == [0, 0]
    assert r["fast_track_count_across_seeds"] == {0: 2, 1: 0, 2: 2}


def test_fast_fast_conditioning_can_create_deficit_without_model_change():
    a = frame()
    b = a.copy()
    b["chain_track"] = [True, False, True, False]
    b["chain_tracker_term"] = b.chain_track
    r = cs.report({"a": a, "b": b})["fast_fast_history_controls"]["a"]["b"]
    assert r["all_events"]["paired_fast_minus_g4_error_n"][0] == 0
    assert r["reference_fast_tracker_term"]["paired_fast_minus_g4_error_n"][0] == -0.5
    assert r["candidate_fast_tracker_term"]["paired_fast_minus_g4_error_n"][0] == 0.5


def test_rejects_changed_baseline_denominator_and_duplicates():
    a = frame()
    b = a.copy()
    b.loc[0, "g4_track"] = False
    for bad in (b, a.iloc[:-1]):
        with pytest.raises(AssertionError):
            cs.report({"a": a, "b": bad})
    with pytest.raises(ValueError, match="duplicate"):
        cs.report({"a": a, "b": pd.concat([a, a.iloc[:1]])})
