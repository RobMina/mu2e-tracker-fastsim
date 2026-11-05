"""KinKal active states, event-local straw identity and persisted count checks."""
import awkward as ak
import numpy as np
import pytest

from trkfastsim.validation.active_straws import count_straws, summarize_hits


def hit(state, straw=2, plane=0, panel=0):
    return dict(state=state, straw=straw, plane=plane, panel=panel)


def test_active_states_and_cross_event_deduplication():
    hits = ak.Array([[hit(-3), hit(-2), hit(-1), hit(0), hit(1)], [hit(1)], []])
    rows, straws = count_straws(hits, hits.state > -2)
    np.testing.assert_array_equal(rows, [3, 1, 0])
    np.testing.assert_array_equal(straws, [1, 1, 0])
    result = summarize_hits(hits, np.array([True, True, False]), [3, 1, 0])
    np.testing.assert_array_equal(result["right_only_straws"], [1, 1, np.nan])


def test_distinct_identity_includes_plane_and_panel():
    hits = ak.Array([[hit(1), hit(-1, plane=1), hit(0, panel=1)]])
    assert count_straws(hits, hits.state > -2)[1].tolist() == [3]


def test_nactive_disagreement_rejected():
    with pytest.raises(ValueError, match="persisted nactive"):
        summarize_hits(ak.Array([[hit(-1), hit(1)]]), np.array([True]), [1])
