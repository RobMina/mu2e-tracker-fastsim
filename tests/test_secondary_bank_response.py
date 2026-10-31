"""Bounded response metrics keep empty recipients and exact event-index coverage."""
import numpy as np
import pytest

from scripts.secondary_bank_response import digi_view, measure


def test_empty_recipient_contributes_to_response_mean():
    digis = dict(ev=np.array([0, 0, 2]), sid=np.array([7, 8, 9]), tdc=np.zeros((3, 2)),
                 tot=np.zeros((3, 2)), adc=np.zeros((3, 4)))
    result = measure(digis, np.array([1, 0, 1, 0]), 4)
    assert result["digis_per_event"] == .75
    assert result["good_fits_per_event"] == result["events_with_fit_fraction"] == .5
    view = digi_view(np.zeros((4, 3), np.int64), digis)
    assert view["nadc"].tolist() == [4, 4, 4]
    assert view["adc"].shape == (12,)
    with pytest.raises(ValueError, match="coverage"):
        measure(digis, [1, 0], 4)


def test_response_refuses_bank_ancestry_from_other_inputs():
    from scripts.secondary_bank_response import validate_campaign_inputs
    receipt = {"input_sha256": {"bank.root": "correct"}}
    validate_campaign_inputs(receipt, {"bank.root": "correct"}, ["bank.root"])
    with pytest.raises(ValueError, match="identity differs"):
        validate_campaign_inputs(receipt, {"bank.root": "different"}, ["bank.root"])
