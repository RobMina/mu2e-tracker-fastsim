"""Saved comparison statistics retain event pairing and empty-event denominators."""
import numpy as np
import pytest

from trkfastsim.validation.cosmic_saved import efficiency, paired_histogram


def test_efficiency_paired_difference():
    out = efficiency([1, 1, 0, 0], [1, 0, 1, 0])
    assert out["events"] == 4
    assert out["gained"] == out["lost"] == 1
    assert out["difference_pp"]["value"] == 0
    assert out["difference_pp"]["error"] == pytest.approx(100 / np.sqrt(6))
    assert out["ratio"]["value"] == 1
    assert out["ratio"]["error"] == pytest.approx(np.sqrt(2 / 3))


def test_histogram_identical_pairs_have_zero_error():
    values = np.array([0.2, 0.5, 1.2, 2.0])
    out = paired_histogram(values, values, [0, 1, 2], np.ones(4, bool), np.ones(4, bool))
    assert out["fast"] == out["stock"] == [2, 2]
    assert out["ratio"] == [1, 1]
    assert out["ratio_error"] == [0, 0]
    assert out["shape"]["chi2"] == 0
    assert out["shape"]["ndf"] == 0


def test_empty_events_and_overflows_are_retained():
    a = np.array([0.2, 1.5, 4.0, np.nan])
    b = np.array([0.2, 0.5, 1.5, np.nan])
    mask = np.array([True, True, True, False])
    out = paired_histogram(a, b, [0, 1, 2], mask, mask)
    assert out["fast"] == [1, 1]
    assert out["stock"] == [2, 1]
    assert out["fast_outside"] == 1
    assert out["stock_outside"] == 0
    assert out["shape"]["ndf"] == 1
    assert np.isfinite(out["shape"]["chi2"])


def test_missing_stock_efficiency_has_no_ratio():
    assert efficiency([0, 1], [0, 0])["ratio"] is None


def test_deterministic_shape_difference_is_not_reported_as_agreement():
    mask = np.ones(4, bool)
    result = paired_histogram(np.zeros(4), np.ones(4), [-0.5, 0.5, 1.5], mask, mask)
    assert np.isinf(result["shape"]["chi2"])
    assert result["shape"]["unsupported_difference_norm"] > 0
    assert np.isnan(result["shape"]["pulls"]).all()


def test_single_event_ratio_has_no_estimated_error():
    assert efficiency([1], [1])["ratio"] == {"value": 1.0, "error": None}
