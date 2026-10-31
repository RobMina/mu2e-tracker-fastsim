"""Source-event replicas retain empty donors and identical component multiplicities."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.bankreplica import replica, validate


def bank():
    return dict(events=pd.DataFrame(dict(run=1, subrun=1, event=[10, 20, 30], pbin=[1, 1, 2], noff=[0, 1, 2])),
        instraw=pd.DataFrame(dict(run=1, subrun=1, event=[10, 10, 20, 30], xing=[0, 1, 2, 3],
                                  pbin=[1, 1, 1, 2], lbin=1, nsec=[0, 0, 1, 2])),
        instraw_steps=pd.DataFrame(dict(xing=[2, 3, 3], eion=[.01, .02, .03])),
        offaxis=pd.DataFrame(dict(run=1, subrun=1, event=[20, 30, 30], eion=[.01, .02, .03])))


def test_empty_donors_and_joint_multiplicity_are_not_dropped():
    original = bank()
    result, ancestry = replica(original, None, selection=[0, 0, 2])
    assert result["events"].noff.tolist() == [0, 0, 2]
    assert result["instraw"].nsec.tolist() == [0, 0, 0, 0, 2]
    assert len(result["instraw_steps"]) == 2 and len(result["offaxis"]) == 2
    assert ancestry["events"].source_event.tolist() == [10, 10, 30]
    assert original["events"].event.tolist() == [10, 20, 30]
    assert result["instraw"].xing.tolist() == list(range(5))


def test_explicit_identity_selection_preserves_payload_except_new_ids():
    original = bank()
    result, _ = replica(original, None, selection=np.arange(3))
    for tree in original:
        left = original[tree].drop(columns=["run", "subrun", "event"], errors="ignore")
        right = result[tree].drop(columns=["run", "subrun", "event"], errors="ignore")
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=False)


def test_source_event_resampling_is_reproducible_and_pbin_stratified():
    first, a = replica(bank(), np.random.default_rng(19))
    second, b = replica(bank(), np.random.default_rng(19))
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert a["events"].source_event_row.iloc[-1] == b["events"].source_event_row.iloc[-1] == 2


def test_invalid_bank_and_selection_refuse():
    original = bank()
    original["events"].loc[0, "noff"] = 1
    with pytest.raises(ValueError, match="off-axis counts"):
        validate(original)
    with pytest.raises(ValueError, match="outside bank"):
        replica(bank(), None, selection=[0, 1, 3])
    with pytest.raises(ValueError, match="integer"):
        replica(bank(), None, selection=[0., 1., 2.])
