"""Independent parent reporting rejects overlap and keeps the explored pilot separate."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import fieldparents as fp


def frame(subrun, delta=0, n=20):
    p = 1500.0
    return pd.DataFrame({"run": np.uint64(1430), "subrun": np.uint64(subrun),
                         "event": np.arange(2**54, 2**54 + n, dtype=np.uint64),
                         "p_first": p, "fast_ntrk": 1, "g4_ntrk": 1, "fast_sid": 0, "g4_sid": 0,
                         "fast_nactive": 40.0 + delta, "g4_nactive": 40.0,
                         "fast_p_first_own": p, "g4_p_first_own": p,
                         "fast_p_reco": p + np.linspace(-10, 10, n),
                         "g4_p_reco": p + np.linspace(-10, 10, n)})


def test_identical_and_pilot_separation():
    r = fp.report({"pilot": frame(0, 100), "a": frame(1), "b": frame(2)}, "pilot")
    c = r["confirmation_only"]
    assert c["events"] == 40
    assert c["parents"] == ["a", "b"]
    assert c["estimates"]["active_hit_difference"]["pooled"] == 0
    assert r["including_pilot"]["estimates"]["active_hit_difference"]["pooled"] == pytest.approx(100 / 3)
    width = c["estimates"]["bin1_relative_width_difference"]
    assert width["pooled"] == width["delete_one_parent_error"] == 0
    assert c["event_paired_yield_difference_error_n"] == [0.0, 0.0, 40]


def test_parent_jackknife_not_event_error():
    r = fp.report({"pilot": frame(0), "a": frame(1, 1), "b": frame(2, 3)}, "pilot")
    hits = r["confirmation_only"]["estimates"]["active_hit_difference"]
    assert hits["pooled"] == 2.0
    assert hits["delete_one_parent_error"] == 1.0
    assert hits["delete_one_parent_estimates"] == [3.0, 1.0]


def test_overlap_duplicates_float_negative_and_missing_rejected():
    parents = {"pilot": frame(0), "a": frame(1), "b": frame(2)}
    for bad in (frame(0), pd.concat([frame(1), frame(1).iloc[:1]]),
                frame(1).assign(event=1.5), frame(1).assign(event=-1), frame(1).drop(columns="p_first")):
        with pytest.raises(ValueError):
            fp.prepare({**parents, "a": bad})
    with pytest.raises(ValueError, match="pilot"):
        fp.report(parents, "missing")
    with pytest.raises(ValueError, match="at least three"):
        fp.prepare({"a": frame(0), "b": frame(1)})


def test_step_pooling_weights_and_count_guards():
    def steps(n, g, f):
        return {"first_pass": {"tracks": n, "straws_per_track": {
            "g4": [g, 0, n], "fast": [f, 0, n], "fast_minus_g4": [f - g, 0, n]}}}
    r = fp.step_summary({"a": steps(10, 20, 18), "b": steps(30, 20, 20)})
    assert r["tracks"] == 40 and r["g4_steps"] == 800 and r["fast_steps"] == 780
    assert r["difference_steps_per_track"] == -0.5
    assert r["relative_step_difference"] == -0.025
    np.testing.assert_allclose(r["delete_one_parent_errors"], [1.0, 0.05])
    with pytest.raises(ValueError, match="count"):
        fp.step_summary({"a": steps(0, 20, 18), "b": steps(10, 20, 20)})
    with pytest.raises(ValueError, match="exactly"):
        fp.report({"pilot": frame(0), "a": frame(1), "b": frame(2)}, "pilot", {"a": steps(10, 20, 18)})


def test_uint64_identity_survives_and_missing_bin_is_explicit():
    parents = {"pilot": frame(0), "a": frame(1).astype({"run": "int64"}), "b": frame(2)}
    d = fp.prepare(parents)
    assert d.event.dtype == np.uint64
    np.testing.assert_array_equal(d[d.parent == "a"].event.to_numpy(), frame(1).event.to_numpy())
    assert np.isnan(fp.metrics(d)["bin0_relative_width_difference"])
