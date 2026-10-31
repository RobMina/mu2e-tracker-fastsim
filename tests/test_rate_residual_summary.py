"""Saved paired summary checks conditional discordance, not invented cause."""
import copy

import numpy as np
import pytest

from scripts.rate_residual_summary import summarize


def test_paired_counts_and_fates_must_agree():
    arm = dict(gen_events=11000000, any_muon={"all": dict(ref=61864, arm=61315,
        arm_only=18992, ref_only=19541)}, replay={"g4_only": dict(total=19541,
        fates={"missed": 9627, "few_left_volume": 9608, "stopped_above": 216,
               "not_near": 83, "few_below_50MeV": 7}),
        "fast_only": dict(total=18992, fates={"g4_not_kept": 11953,
            "g4_few_through": 3737, "g4_missed": 2770, "g4_few_stopped": 532})})
    result = summarize({"arms": {"legacy": arm}})["arms"]["legacy"]
    assert result["diff"] == -549 and result["discordant"] == 38533
    np.testing.assert_allclose(result["paired_error_percent"], 100 * np.sqrt(38533) / 61864)
    assert result["normal_95_percent"][1] < 0
    assert result["fast_wins_wilson95"][1] < .5
    assert result["explicit_fast_interior_stops_in_g4_only"] == 0
    bad = copy.deepcopy(arm)
    bad["replay"]["g4_only"]["total"] += 1
    with pytest.raises(ValueError, match="disagree"):
        summarize({"arms": {"bad": bad}})
