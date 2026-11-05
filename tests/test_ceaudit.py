"""Saved-arm integrity and truth-diagnostic attribution contracts."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import ceaudit


def event_frame():
    n = 8
    d = pd.DataFrame({"run": np.full(n, 1430), "subrun": np.zeros(n, dtype=int),
                      "event": np.arange(n, dtype=np.int64) + 2**54})
    for arm in ("g4", "fast"):
        for col, value in (("track", True), ("surface", True), ("p_truth", 104.0),
                           ("p_reco", 104.1), ("nactive", 35), ("nstraw", 17)):
            d[f"{arm}_{col}"] = value
    return d


def test_join_retains_large_integer_keys_and_missing_track_flags():
    a, b = event_frame(), event_frame()
    b.loc[0, "fast_track"] = False
    d = ceaudit.join_events(a, b.iloc[::-1])
    np.testing.assert_array_equal(d.event, a.event)
    assert not d.chain_track.iloc[0] and d.entry_track.all()


def test_join_accepts_integer_storage_types_but_rejects_float_keys():
    a, b = event_frame(), event_frame()
    b["run"] = b.run.astype(np.uint32)
    b["event"] = b.event.astype(np.uint64)
    d = ceaudit.join_events(a, b)
    np.testing.assert_array_equal(d.event, a.event)
    b["event"] = b.event.astype(float)
    with pytest.raises(ValueError, match="identities"):
        ceaudit.join_events(a, b)


def test_join_rejects_changed_denominator_or_baseline_or_duplicate_identity():
    a, b = event_frame(), event_frame()
    with pytest.raises(AssertionError):
        ceaudit.join_events(a, b.iloc[1:])
    b.loc[0, "g4_p_truth"] += 1
    with pytest.raises(AssertionError):
        ceaudit.join_events(a, b)
    with pytest.raises(ValueError, match="duplicate"):
        ceaudit.join_events(a, pd.concat([a, a.iloc[:1]]))


def test_time_cut_uses_only_exiting_fast_clock_and_preserves_keys():
    base = event_frame()[ceaudit.KEY].iloc[:3]
    s = pd.concat([base.assign(t=10), base.assign(t=12)], ignore_index=True)
    f = base.assign(t=10, status=[0, 1, 2])
    cut = ceaudit.time_cut(s, f)
    assert len(cut) == 5
    assert not ((cut.event == base.event.iloc[0]) & (cut.t == 12)).any()
    assert cut.event.dtype == np.uint64


def test_step_summary_separates_passages_distinct_straws_and_planes():
    s = pd.DataFrame({"run": [1, 1, 1], "subrun": [0, 0, 0], "event": [2, 2, 2],
                      "sid": [4, 4, 8], "plane": [0, 0, 1], "panel": [0, 0, 0], "straw": [4, 4, 8],
                      "eion": [0.1, 0.2, 0.3]})
    r = ceaudit.step_summary(s).iloc[0]
    assert r.steps == 3 and r.straws == 2 and r.planes == 2
    np.testing.assert_allclose(r.eion, 0.6)


def test_yield_partition_adds_to_total_without_dropping_missed_handoffs():
    d = ceaudit.join_events(event_frame(), event_frame())
    d["chain_has_vd13"] = [False, True, True, True, True, True, True, True]
    d["chain_steps"] = [0, 0, 20, 20, 20, 20, 20, 20]
    d.loc[:2, "chain_track"] = False
    for k in ("g4_steps", "entry_steps", "entry_planes", "chain_planes"):
        d[k] = 20
    d["g4_late_steps"] = 0
    d["delta_p"] = 0.1
    r = ceaudit.audit(d)
    p = r["yield_partition"]
    assert sum(x["signed_track_difference"] for x in p.values()) == -3
    assert p["no_chain_vd13"]["events"] == 1 and p["vd13_no_steps"]["events"] == 1
    assert r["chain_entry_categories"]["lost"]["events"] == 3
