"""Exact full-key joins and an unselected, lossless stock track/hit census."""
import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

from trkfastsim.validation import stockcensus as sc


TREE = "EventNtuple/ntuple"


def sample():
    counts = np.array([2, 0, 1, 1])
    nh = np.array([3, 1, 0, 2])
    trk = {"status": [-1, 2, 0, 1], "goodfit": [0, 1, 0, 1],
           "fitalg": [3, 1, 2, 3], "nhits": nh, "nactive": [0, 1, 0, 2],
           "ndof": [-9999, 1, 0, 7], "chisq": [np.nan, -9999., np.inf, 2.],
           "fitcon": [-9999., np.nan, 0., 0.5]}
    trk = {f"trk.{name}": ak.unflatten(np.asarray(value, dtype=np.float32 if name in sc.TRACK_FLOATS else np.int32), counts)
           for name, value in trk.items()}
    hit = {"plane": [0, 0, 0, 35, 2, 2], "panel": [0, 0, 0, 5, 1, 1],
           "straw": [2, 2, 2, 95, 7, 9], "state": [-3, -2, -1, 0, 1, 8]}
    hits = ak.zip({name: ak.unflatten(ak.unflatten(np.asarray(value, np.int32), nh), counts)
                   for name, value in hit.items()})
    evt = ak.zip({"run": np.array([2, 1, 2, 2], np.int32),
                  "subrun": np.array([7, 7, 8, 7], np.int32),
                  "event": np.array([99, 99, 99, 105], np.int32)})
    return {"evtinfo": evt, "trk": ak.zip(trk, depth_limit=1), "trkhits": hits}


def write_root(tmp_path, arrays=None, tree=TREE):
    path = tmp_path / "stock.root"
    with uproot.recreate(path) as root:
        root[tree] = sample() if arrays is None else arrays
    return path


def replace_field(arrays, branch, name, values):
    arrays[branch] = ak.with_field(arrays[branch], values, name)
    return arrays


def empty_sample(n=3, tracks=False):
    counts = np.ones(n, dtype=np.int64) if tracks else np.zeros(n, dtype=np.int64)
    nt = int(counts.sum())
    hit_counts = np.zeros(nt, dtype=np.int64)
    evt = ak.zip({"run": np.ones(n, np.int32), "subrun": np.ones(n, np.int32),
                  "event": np.arange(n, dtype=np.int32)})
    trk = ak.zip({f"trk.{name}": ak.unflatten(np.zeros(nt, np.float64 if name in sc.TRACK_FLOATS else np.int32), counts)
                  for name in sc.TRACK_INTS + sc.TRACK_FLOATS}, depth_limit=1)
    hits = ak.zip({name: ak.unflatten(ak.unflatten(np.empty(0, np.int32), hit_counts), counts)
                   for name in sc.HIT_FIELDS})
    return {"evtinfo": evt, "trk": trk, "trkhits": hits}


def test_full_key_sparse_reordered_exact_join_and_no_mutation():
    ref = np.array([[2, 7, 99], [1, 7, 99], [2, 8, 99], [2, 7, 105], [2**62, 0, 9]], np.int64)
    query = ref[[3, 1, 4, 0, 2]].copy()
    query = np.vstack([query, [2, 7, 100], [2, 9, 99]])
    before_ref, before_query = ref.copy(), query.copy()
    np.testing.assert_array_equal(sc.event_index(ref, query), [3, 1, 4, 0, 2, -1, -1])
    np.testing.assert_array_equal(ref, before_ref)
    np.testing.assert_array_equal(query, before_query)
    valid = sc.validate_keys(ref)
    assert valid.dtype == np.int64 and not np.shares_memory(valid, ref)
    np.testing.assert_array_equal(sc.event_index(ref, np.empty((0, 3), np.int64)), [])
    np.testing.assert_array_equal(sc.event_index(np.empty((0, 3), np.int64), query), -np.ones(len(query)))
    np.testing.assert_array_equal(sc.event_index(ref[::-1], ref), [4, 3, 2, 1, 0])


@pytest.mark.parametrize("keys", [np.array([1, 2, 3]), np.zeros((2, 2), int),
                                    [[1., 2., 3.]], [[1., np.nan, 3.]], [[1., np.inf, 3.]],
                                    [[1, -1, 3]], [[True, False, True]],
                                    np.array([[1, 2, 2**63]], np.uint64),
                                    np.array([[1, 2, 3]], dtype=object),
                                    [[1, 2, 3], [1, 2, 3]]])
def test_invalid_keys(keys):
    with pytest.raises(ValueError):
        sc.validate_keys(keys)


@pytest.mark.parametrize("side", [0, 1])
def test_join_rejects_duplicates_on_either_side(side):
    args = [np.array([[1, 2, 3]]), np.array([[1, 2, 4]])]
    args[side] = np.repeat(args[side], 2, axis=0)
    with pytest.raises(ValueError, match="duplicate"):
        sc.event_index(*args)


def test_key_int64_boundaries_and_mask():
    keys = np.array([[0, 0, 0], [np.iinfo(np.int64).max, 1, 2]], np.uint64)
    np.testing.assert_array_equal(sc.validate_keys(keys), keys)
    with pytest.raises(ValueError, match="missing"):
        sc.validate_keys(np.ma.array([[1, 2, 3]], mask=[[False, True, False]]))


def test_every_track_and_recorded_hit_preserved(tmp_path):
    path = write_root(tmp_path)
    data, meta = sc.read_stock(path, chunk_entries=2)
    assert set(data) == {"evkey", "track_ev", "track_index", "hit_track", "hit_index", "hit_sid", "hit_state",
                         *(f"track_{name}" for name in sc.TRACK_INTS + sc.TRACK_FLOATS)}
    np.testing.assert_array_equal(data["evkey"], [[2, 7, 99], [1, 7, 99], [2, 8, 99], [2, 7, 105]])
    np.testing.assert_array_equal(data["track_ev"], [0, 0, 2, 3])
    np.testing.assert_array_equal(data["track_index"], [0, 1, 0, 0])
    np.testing.assert_array_equal(data["track_status"], [-1, 2, 0, 1])
    np.testing.assert_array_equal(data["track_goodfit"], [0, 1, 0, 1])
    np.testing.assert_array_equal(data["track_fitalg"], [3, 1, 2, 3])
    np.testing.assert_array_equal(data["track_nhits"], [3, 1, 0, 2])
    np.testing.assert_array_equal(data["track_nactive"], [0, 1, 0, 2])
    np.testing.assert_array_equal(data["track_ndof"], [-9999, 1, 0, 7])
    np.testing.assert_array_equal(data["track_chisq"], [np.nan, -9999., np.inf, 2.])
    np.testing.assert_array_equal(data["track_fitcon"], [-9999., np.nan, 0., 0.5])
    np.testing.assert_array_equal(data["hit_track"], [0, 0, 0, 1, 3, 3])
    np.testing.assert_array_equal(data["hit_index"], [0, 1, 2, 0, 0, 1])
    np.testing.assert_array_equal(data["hit_sid"], [2, 2, 2, (35 << 10) | (5 << 7) | 95, 2183, 2185])
    np.testing.assert_array_equal(data["hit_state"], [-3, -2, -1, 0, 1, 8])
    assert all(value.dtype == (np.float32 if name in ("track_chisq", "track_fitcon") else np.int64)
               for name, value in data.items())
    assert meta["tree_class"] == "ROOT::RNTuple" and meta["tree_key"] == TREE + ";1"
    assert meta["cycle"] == 1 and meta["num_entries"] == 4
    assert meta["raw_tracks"] == 4 and meta["raw_hits"] == 6
    assert meta["hit_counts_match_nhits"] and meta["recorded_hit_census_complete"]
    assert set(meta["read_paths"]) == {"evtinfo", "trk", "trkhits"}
    assert meta["fields"]["trk"] == [f"trk.{name}" for name in sc.TRACK_INTS + sc.TRACK_FLOATS]
    assert meta["limits"] == dict(max_events=1000000, max_tracks=1000000, max_hits=10000000)


@pytest.mark.parametrize("chunk", [1, 2, 3, 4096])
def test_chunk_invariance_and_no_input_mutation(tmp_path, chunk):
    path = write_root(tmp_path)
    before = path.read_bytes()
    ref, _ = sc.read_stock(path)
    got, _ = sc.read_stock(path, chunk_entries=chunk)
    for name in ref:
        np.testing.assert_array_equal(got[name], ref[name])
    assert path.read_bytes() == before


@pytest.mark.parametrize("n,tracks", [(0, False), (3, False), (3, True)])
def test_empty_tree_events_and_hit_lists(tmp_path, n, tracks):
    path = write_root(tmp_path, empty_sample(n, tracks))
    data, meta = sc.read_stock(path, chunk_entries=1, max_events=n, max_tracks=n if tracks else 0, max_hits=0)
    assert data["evkey"].shape == (n, 3)
    assert data["track_ev"].shape == ((n if tracks else 0),)
    assert data["hit_sid"].shape == (0,)
    assert meta["raw_hits"] == 0


def test_custom_tree_and_highest_cycle_only(tmp_path):
    path = tmp_path / "cycles.root"
    with uproot.recreate(path) as root:
        root["custom/events"] = sample()
        root["custom/events"] = empty_sample(2)
    data, meta = sc.read_stock(path, tree="custom/events", max_events=2, max_tracks=0, max_hits=0)
    assert meta["cycle"] == 2 and meta["tree_key"] == "custom/events;2"
    assert data["evkey"].shape == (2, 3) and meta["raw_tracks"] == 0
    with pytest.raises(ValueError, match="missing required tree"):
        sc.read_stock(path)
    with pytest.raises(ValueError, match="uncycled"):
        sc.read_stock(path, tree="custom/events;1")


@pytest.mark.parametrize("limit,value", [("max_events", 3), ("max_tracks", 3), ("max_hits", 5)])
def test_hard_caps_no_truncation(tmp_path, limit, value):
    path = write_root(tmp_path)
    with pytest.raises(ValueError, match=limit):
        sc.read_stock(path, chunk_entries=1, **{limit: value})
    sc.read_stock(path, chunk_entries=2, max_events=4, max_tracks=4, max_hits=6)


@pytest.mark.parametrize("name,value", [("chunk_entries", 0), ("chunk_entries", 1.5), ("chunk_entries", True),
                                        ("max_events", -1), ("max_tracks", False), ("max_hits", np.inf),
                                        ("max_hits", 2**64)])
def test_invalid_limits_before_file_read(name, value):
    with pytest.raises(ValueError, match=name):
        sc.read_stock("does-not-exist.root", **{name: value})


@pytest.mark.parametrize("branch", ["evtinfo", "trk", "trkhits"])
def test_missing_required_branch(tmp_path, branch):
    arrays = sample()
    del arrays[branch]
    with pytest.raises(ValueError, match="missing required branch"):
        sc.read_stock(write_root(tmp_path, arrays))


@pytest.mark.parametrize("branch,name", [("evtinfo", "run"), ("trk", "trk.goodfit"), ("trkhits", "state")])
def test_missing_required_field(tmp_path, branch, name):
    arrays = sample()
    arrays[branch] = ak.without_field(arrays[branch], name)
    with pytest.raises(ValueError, match="missing required field"):
        sc.read_stock(write_root(tmp_path, arrays))


@pytest.mark.parametrize("chunk", [1, 4])
def test_duplicate_keys_across_whole_file(tmp_path, chunk):
    arrays = sample()
    arrays["evtinfo"] = arrays["evtinfo"][[0, 1, 2, 0]]
    with pytest.raises(ValueError, match="duplicate"):
        sc.read_stock(write_root(tmp_path, arrays), chunk_entries=chunk)


@pytest.mark.parametrize("value", [np.array([1., 2., 3., 4.]), np.array([np.nan, 2., 3., 4.]),
                                   np.array([-1, 2, 3, 4]), np.array([2**63, 2, 3, 4], np.uint64)])
def test_invalid_persisted_keys(tmp_path, value):
    arrays = replace_field(sample(), "evtinfo", "event", value)
    with pytest.raises(ValueError):
        sc.read_stock(write_root(tmp_path, arrays))


@pytest.mark.parametrize("name,value", [("plane", -1), ("plane", 36), ("panel", 6), ("straw", 96)])
def test_invalid_hit_identifiers(tmp_path, name, value):
    arrays = sample()
    old = arrays["trkhits"][name]
    arrays = replace_field(arrays, "trkhits", name, ak.where(old == old, value, old))
    with pytest.raises(ValueError, match="invalid hitid"):
        sc.read_stock(write_root(tmp_path, arrays))


@pytest.mark.parametrize("branch,name,values,match", [
    ("trk", "trk.nhits", [[2, 1], [], [0], [2]], "trk.nhits.*counts"),
    ("trk", "trk.nhits", [[-1, 1], [], [0], [2]], "trk.nhits.*counts"),
    ("trk", "trk.nactive", [[1], [], [0, 2], [2]], "alignment"),
    ("trk", "trk.status", [[1.5, 2.], [], [0.], [1.]], "integer"),
    ("trk", "trk.chisq", [[1, 2], [], [0], [1]], "floating"),
    ("trkhits", "state", [[[-3, -2], [0, 1]], [], [[]], [[1, 8]]], "trk.nhits.*counts"),
    ("trkhits", "state", [[[-3, -2, -1, 0]], [], [[], []], [[1, 8]]], "track alignment"),
])
def test_malformed_jagged_counts_and_dtypes(tmp_path, branch, name, values, match):
    arrays = sample()
    if branch == "trkhits":
        arrays[branch] = ak.zip({field: arrays[branch][field] for field in sc.HIT_FIELDS}, depth_limit=1)
    arrays = replace_field(arrays, branch, name, ak.Array(values))
    with pytest.raises(ValueError, match=match):
        sc.read_stock(write_root(tmp_path, arrays))


def test_no_mc_and_reads_are_chunk_bounded(tmp_path, monkeypatch):
    path = write_root(tmp_path)
    with uproot.open(path) as root:
        cls = type(root[TREE]["trk"])
    original = cls.array
    reads = []

    def guard(self, *args, **kwargs):
        assert self.name in ("evtinfo", "trk", "trkhits")
        assert 0 <= kwargs["entry_start"] <= kwargs["entry_stop"] <= 4
        assert kwargs["entry_stop"] - kwargs["entry_start"] <= 2
        reads.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "array", guard)
    sc.read_stock(path, chunk_entries=2)
    assert reads == ["evtinfo", "trk", "trkhits"] * 2


def test_real_local_stock_ttree():
    default = Path.home() / "mu2e/remote-data/crv_efficiency_mdc2025/kkts_smoke/nts.kkts.smoke.root"
    path = Path(os.environ.get("STOCKCENSUS_TEST_ROOT", default))
    if not path.is_file():
        pytest.skip("set STOCKCENSUS_TEST_ROOT to an existing local stock EventNtuple")
    with uproot.open(path) as root:
        tree = root[TREE]
        if "trk" not in tree:
            pytest.skip("local smoke ntuple uses another track branch name")
        assert tree.classname == "TTree"
        expected_events = tree.num_entries
        tracks = tree["trk"].array()
        hits = tree["trkhits"].array()
        expected_status = ak.to_numpy(ak.flatten(tracks["trk.status"]))
        expected_state = ak.to_numpy(ak.flatten(ak.flatten(hits["state"], axis=2)))
    data, meta = sc.read_stock(path, chunk_entries=17)
    assert meta["tree_class"] == "TTree" and len(data["evkey"]) == expected_events
    np.testing.assert_array_equal(data["track_status"], expected_status)
    np.testing.assert_array_equal(data["hit_state"], expected_state)


def test_stale_lower_ttree_cycle_is_not_read(tmp_path):
    path = tmp_path / "stale.root"
    with uproot.recreate(path) as root:
        stale = root.mktree(TREE, {"unrelated": "int64"})
        stale.extend({"unrelated": np.arange(100, dtype=np.int64)})
        root[TREE] = empty_sample(2)
    data, meta = sc.read_stock(path, max_events=2)
    assert meta["cycle"] == 2 and len(data["evkey"]) == 2
    assert meta["tree_class"] == "ROOT::RNTuple"


def test_invalid_newest_cycle_never_falls_back(tmp_path):
    path = write_root(tmp_path)
    with uproot.update(path) as root:
        root[TREE] = {"unrelated": np.arange(2)}
    with pytest.raises(ValueError, match="missing required branch"):
        sc.read_stock(path)


def test_wrong_object_and_nonlocal_inputs_rejected(tmp_path):
    path = tmp_path / "string.root"
    with uproot.recreate(path) as root:
        root[TREE] = "not a tree"
    with pytest.raises(ValueError, match="not a TTree or RNTuple"):
        sc.read_stock(path)
    with pytest.raises(ValueError, match="existing local file"):
        sc.read_stock("https://example.invalid/stock.root")


@pytest.mark.parametrize("branch,name,values,match", [
    ("trk", "trk.goodfit", [[True, False], [], [False], [True]], "integer"),
    ("trk", "trk.ndof", [[1, None], [], [0], [1]], "missing"),
    ("trk", "trk.status", [1, 0, 0, 1], "alignment"),
    ("trk", "trk.nactive", [[1, 2], None, [0], [2]], "alignment"),
    ("trkhits", "plane", [[[0, 0], [35, 1]], [], [[]], [[2, 2]]], "alignment"),
    ("trkhits", "panel", [[[0, 0, 0], [5]], [], [[]], [[1.5, 1.]]], "integer"),
    ("trkhits", "straw", [[[2, None, 2], [95]], [], [[]], [[7, 9]]], "missing"),
    ("trkhits", "state", [[[-3, -2, -1], [0]], None, [[]], [[1, 8]]], "alignment"),
    ("trkhits", "state", [[[-3, -2, -1], [0]], [], [None], [[1, 8]]], "missing"),
])
def test_nulls_bad_depth_and_independent_hit_leaf_alignment(tmp_path, branch, name, values, match):
    arrays = sample()
    if branch == "trkhits":
        arrays[branch] = ak.zip({field: arrays[branch][field] for field in sc.HIT_FIELDS}, depth_limit=1)
    arrays = replace_field(arrays, branch, name, ak.Array(values))
    with pytest.raises(ValueError, match=match):
        sc.read_stock(write_root(tmp_path, arrays))


def test_zero_tracks_cannot_hide_hit_track_alignment(tmp_path):
    arrays = empty_sample(3)
    arrays["trkhits"] = empty_sample(3, tracks=True)["trkhits"]
    with pytest.raises(ValueError, match="track alignment"):
        sc.read_stock(write_root(tmp_path, arrays))


def test_empty_tree_does_not_skip_required_fields(tmp_path):
    arrays = empty_sample(0)
    arrays["trk"] = ak.without_field(arrays["trk"], "trk.goodfit")
    with pytest.raises(ValueError, match="missing required field"):
        sc.read_stock(write_root(tmp_path, arrays))


def test_track_cap_precedes_hit_branch_read(tmp_path, monkeypatch):
    path = write_root(tmp_path)
    with uproot.open(path) as root:
        cls = type(root[TREE]["trk"])
    original = cls.array
    reads = []

    def guard(self, *args, **kwargs):
        reads.append(self.name)
        assert self.name != "trkhits"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "array", guard)
    with pytest.raises(ValueError, match="max_tracks"):
        sc.read_stock(path, max_tracks=1)
    assert reads == ["evtinfo", "trk"]


def test_event_cap_precedes_any_branch_read(tmp_path, monkeypatch):
    path = write_root(tmp_path)
    with uproot.open(path) as root:
        cls = type(root[TREE]["trk"])

    def forbidden(self, *args, **kwargs):
        pytest.fail("event cap must be checked before reading a branch")

    monkeypatch.setattr(cls, "array", forbidden)
    with pytest.raises(ValueError, match="max_events"):
        sc.read_stock(path, max_events=3)


def test_float_bits_and_signed_sentinels_are_not_normalized(tmp_path):
    arrays = sample()
    bits = np.array([0x7fc00001, 0x80000000, 0xff800000, 0x7f800000], np.uint32)
    values = ak.unflatten(bits.view(np.float32), [2, 0, 1, 1])
    arrays = replace_field(arrays, "trk", "trk.chisq", values)
    data, _ = sc.read_stock(write_root(tmp_path, arrays), chunk_entries=1)
    np.testing.assert_array_equal(data["track_chisq"].view(np.uint32), bits)


def test_no_source_array_mutation(tmp_path):
    arrays = sample()
    original = {name: ak.copy(value) for name, value in arrays.items()}
    sc.read_stock(write_root(tmp_path, arrays))
    for name in arrays:
        np.testing.assert_array_equal(ak.to_buffers(arrays[name])[2].keys(), ak.to_buffers(original[name])[2].keys())
        buffers = ak.to_buffers(arrays[name])[2]
        saved = ak.to_buffers(original[name])[2]
        for key in buffers:
            np.testing.assert_array_equal(buffers[key], saved[key])
