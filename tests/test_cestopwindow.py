"""Exact stop-window metadata coverage and disjoint-source safeguards."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import cestopwindow as C


def metadata():
    return [np.array([1, 1, 1, 1], np.uint32), np.zeros(4, np.uint32),
            np.array([0, 7, 8, 9], np.uint32), np.array([0, 2, 0, 1], np.int64)]


def test_complete_physical_order_and_explicit_byteswap():
    out = C.ordered_keys(*metadata(), 3)
    assert out.event.tolist() == [8, 9, 7]
    swapped = [v.byteswap() for v in metadata()]
    with pytest.raises(ValueError, match="physical Events"):
        C.ordered_keys(*swapped, 3)
    pd.testing.assert_frame_equal(C.ordered_keys(*swapped, 3, byteswap=True), out)


@pytest.mark.parametrize("field", range(4))
def test_noninteger_metadata_rejected(field):
    args = metadata()
    args[field] = args[field].astype(float)
    with pytest.raises(ValueError, match="integers"):
        C.ordered_keys(*args, 3)


def test_missing_duplicate_and_negative_metadata_rejected():
    with pytest.raises(ValueError, match="physical Events"):
        C.ordered_keys(*metadata(), 4)
    args = metadata()
    args[2][-1] = 8
    with pytest.raises(ValueError, match="duplicate"):
        C.ordered_keys(*args, 3)
    args = metadata()
    args[3][-1] = 0
    with pytest.raises(ValueError, match="physical Events"):
        C.ordered_keys(*args, 3)


def test_disjoint_rejects_overlap_and_preserves_exact_large_keys():
    old = C.ordered_keys(*metadata(), 3)
    new = old.assign(event=old.event + np.uint64(2**54))
    r = C.disjoint(old, new)
    assert r["shared_event_identities"] == 0
    assert r["new_keys_sha256"] != r["previous_keys_sha256"]
    with pytest.raises(ValueError, match="overlaps"):
        C.disjoint(old, old)


@pytest.mark.parametrize("field,value", [(0, 0), (0, 2**32 - 1), (0, 2**32), (1, -1), (2, 2**32)])
def test_native_art_identity_bounds(field, value):
    args = [v.astype(np.int64) for v in metadata()]
    args[field][-1] = value
    with pytest.raises(ValueError, match="native art identity"):
        C.ordered_keys(*args, 3)


def test_highest_cycle_metadata_and_finished_count(tmp_path):
    import awkward as ak
    import uproot

    path = tmp_path / "window.art"
    run, subrun, event, entry = metadata()
    record = ak.Array(dict(eventID=dict(subRun_=dict(run_=dict(run_=run), subRun_=subrun), event_=event),
                           entry=entry))
    with uproot.recreate(path) as root:
        root["FileIndex"] = {"Element": record}
        root.mktree("Events", {"dummy": "int32"}).extend({"dummy": np.zeros(2, np.int32)})
        root.mktree("Events", {"dummy": "int32"}).extend({"dummy": np.zeros(3, np.int32)})
    assert C.load(path, 3).event.tolist() == [8, 9, 7]
    with pytest.raises(ValueError, match="Events, expected"):
        C.load(path, 2)
