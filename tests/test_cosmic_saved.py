"""Saved comparison statistics retain event pairing and empty-event denominators."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation.cosmic_saved import compare, efficiency, paired_histogram, plot


def test_efficiency_paired_difference():
    out = efficiency([1, 1, 0, 0], [1, 0, 1, 0])
    assert out["events"] == 4
    assert out["gained"] == out["lost"] == 1
    assert out["difference_pp"]["value"] == 0
    assert out["difference_pp"]["error"] == pytest.approx(100 / np.sqrt(6))
    assert out["ratio"]["value"] == 1
    assert out["ratio"]["error"] == pytest.approx(np.sqrt(2 / 3))


def test_histogram_identical_pairs_have_zero_error():
    values = np.tile([0.2, 0.5, 1.2, 2.0], 10)
    out = paired_histogram(values, values, [0, 1, 2], np.ones(40, bool), np.ones(40, bool))
    assert out["fast"] == out["stock"] == [20, 20]
    assert out["ratio"] == [1, 1]
    assert out["ratio_error"] == [0, 0]
    assert out["shape"]["chi2"] == 0
    assert out["shape"]["ndf"] == 0


def test_empty_events_and_overflows_are_retained():
    a = np.tile([0.2, 1.5, 4.0, np.nan], 10)
    b = np.tile([0.2, 0.5, 1.5, np.nan], 10)
    mask = np.tile([True, True, True, False], 10)
    out = paired_histogram(a, b, [0, 1, 2], mask, mask)
    assert out["fast"] == [10, 10]
    assert out["stock"] == [20, 10]
    assert out["fast_outside"] == 10
    assert out["stock_outside"] == 0
    assert out["shape"]["ndf"] == 1
    assert np.isfinite(out["shape"]["chi2"])


def test_missing_stock_efficiency_has_no_ratio():
    assert efficiency([0, 1], [0, 0])["ratio"] is None


def test_low_counts_stay_in_histogram_but_not_in_shape_statistics():
    fast = np.r_[np.zeros(20), np.ones(10), 2.0, 2.0]
    stock = np.r_[np.zeros(15), np.ones(15), 2.0, 2.0]
    selected = np.ones(32, bool)
    out = paired_histogram(fast, stock, [-0.5, 0.5, 1.5, 2.5], selected, selected)
    assert out["fast"] == [20, 10, 2]
    assert out["stock"] == [15, 15, 2]
    assert out["shape"]["included"] == [True, True, False]
    assert np.isnan(out["shape"]["pulls"][-1])
    assert out["shape"]["excluded_fast_entries"] == 2
    assert out["shape"]["excluded_stock_entries"] == 2
    assert out["shape"]["ndf"] == 1
    isolated = paired_histogram(fast[:-2], stock[:-2], [-0.5, 0.5, 1.5], selected[:-2], selected[:-2])
    assert out["shape"]["chi2"] == pytest.approx(isolated["shape"]["chi2"] * (32 / 31) ** -1 * (30 / 29))


def test_all_low_count_bins_have_no_shape_claim():
    mask = np.ones(4, bool)
    result = paired_histogram(np.zeros(4), np.ones(4), [-0.5, 0.5, 1.5], mask, mask)
    assert result["shape"]["chi2"] is None
    assert result["shape"]["included"] == [False, False]
    assert np.isnan(result["shape"]["pulls"]).all()


def test_single_event_ratio_has_no_estimated_error():
    assert efficiency([1], [1])["ratio"] == {"value": 1.0, "error": None}


def synthetic_events():
    events = pd.DataFrame(dict(run=1, subrun=1, event=np.arange(6),
                              p_first=[100, 350, -1, np.nan, np.inf, 350],
                              legacy_ntrk=[1, 0, 1, 1, np.nan, 0], g4_ntrk=[1, 1, 0, 1, 0, 0]))
    for arm in ("legacy", "g4"):
        events[f"{arm}_p_reco"] = [120, 360, 200, 300, np.nan, 400]
        events[f"{arm}_nactive"] = [20, 25, 35, 30, np.nan, 25]
        events[f"{arm}_chi2ndf"] = [1.0, 2.0, 1.5, 2.0, np.nan, 1.0]
    return events


def test_compare_reports_unbinned_events_and_preserves_total_denominator():
    result = compare(synthetic_events())
    assert result["efficiency"]["events"] == 6
    assert result["efficiency"]["fast"] == result["efficiency"]["stock"] == 3
    assert result["efficiency_outside"]["events"] == 3
    assert result["efficiency_outside"]["fast"] == 2
    assert result["efficiency_outside"]["stock"] == 1
    assert sum(b["events"] for b in result["efficiency_bins"]) == 3
    assert result["efficiency_bins"][0]["fast_percent"]["error"] is None
    assert result["efficiency_bins"][2]["fast_percent"]["value"] is None
    assert "solid gas" in result["configuration"]
    assert "no radiative" in result["configuration"]


def test_compare_rejects_duplicate_event_keys():
    events = synthetic_events()
    with pytest.raises(ValueError, match="distinct paired event keys"):
        compare(pd.concat([events, events.iloc[:1]]))


@pytest.mark.parametrize("empty_tracks", [False, True])
def test_plot_smoke_with_empty_and_single_event_bins(tmp_path, empty_tracks):
    events = synthetic_events()
    if empty_tracks:
        events[["legacy_ntrk", "g4_ntrk"]] = 0
    plot(compare(events), tmp_path / "cosmic_trk.png")
    for name in ("cosmic_trk.png", "cosmic_pulls.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
