"""Read-only stock comparison: independent tiny-loop census/overlap oracle.

No reconstruction or optimizer runs: fixtures explicitly record both LR lanes.
The oracle deliberately uses Python loops/sets rather than the vectorized join.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from trkfastsim.validation import run1a_stockcompare as compare


FLAGS = ("success", "objective_valid", "analytic_stationary", "certified", "lr_stationary",
         "refit_error", "refit_numerical_failure")


def _fast(keys, group_ev=(), hits=()):
    """hits are (event row, packed straw, group row or -1, selected)."""
    ng = len(group_ev)
    fast = dict(evkey=np.asarray(keys, dtype=np.int64).reshape(-1, 3),
                group_ev=np.asarray(group_ev, dtype=np.int64),
                source_hit_ev=np.array([h[0] for h in hits], dtype=np.int64),
                source_hit_sid=np.array([h[1] for h in hits], dtype=np.int64),
                hit_group=np.array([h[2] for h in hits], dtype=np.int64),
                source_selected=np.array([h[3] for h in hits], dtype=bool),
                group_support=np.array([sum(h[2] == g and h[3] for h in hits)
                                        for g in range(ng)], dtype=np.int64))
    fast['source_lr_mask'] = np.repeat(fast['source_selected'][:, None], 2, axis=1)
    fast.update({"lr_" + name: np.zeros((ng, 2), dtype=bool) for name in FLAGS})
    fast["lr_objective_valid"][:] = fast["group_support"][:, None] > 0
    fast["lr_lr_stationary"][:] = True
    fast["lr_status"] = np.full((ng, 2), "uncertified", dtype="U32")
    # Opaque diagnostics must survive, including failed and nonfinite lanes.
    fast["lr_direction"] = np.tile(np.array([-1, 1]), (ng, 1))
    fast["lr_objective"] = np.full((ng, 2), np.nan)
    fast["source_time"] = np.arange(len(hits), dtype=float) + 0.25
    return fast


def _succeed(fast, group, direction):
    for name in ("success", "objective_valid", "analytic_stationary", "certified", "lr_stationary"):
        fast["lr_" + name][group, direction] = True
    fast["lr_status"][group, direction] = "converged_certified"


def _stock(keys, tracks=(), hits=()):
    """tracks=(event,status,algorithm,goodfit), hits=(track,straw,state)."""
    stock = dict(evkey=np.asarray(keys, dtype=np.int64).reshape(-1, 3))
    for col, name in enumerate(("ev", "status", "fitalg", "goodfit")):
        stock["track_" + name] = np.array([t[col] for t in tracks], dtype=np.int64)
    for col, name in enumerate(("track", "sid", "state")):
        stock["hit_" + name] = np.array([h[col] for h in hits], dtype=np.int64)
    stock["track_index"] = np.array([sum(t[0] == old[0] for old in tracks[:i])
                                     for i, t in enumerate(tracks)], dtype=np.int64)
    stock["hit_index"] = np.array([sum(h[0] == old[0] for old in hits[:i])
                                   for i, h in enumerate(hits)], dtype=np.int64)
    stock["track_nhits"] = np.array([sum(h[0] == t for h in hits)
                                    for t in range(len(tracks))], dtype=np.int64)
    stock["track_chisq"] = np.full(len(tracks), np.nan)
    return stock


@pytest.fixture
def census():
    # Same event number, different run/subrun: never join on event alone.
    keys = [(10, 1, 7), (11, 1, 7), (10, 2, 7), (10, 1, 8),
            (10, 1, 9), (10, 1, 10), (10, 1, 11)]
    hits = [(0, 0, 0, True), (0, 1, 0, True), (0, 2, 0, True),
            (0, 0, 0, False), (0, 0, -1, False), (0, 0, 8, True),
            (1, 0, 2, True), (1, 3, 2, True), (2, 128, 3, True),
            (3, 1024, 4, True), (4, 1025, 5, True), (5, 2, 6, True),
            (6, 9, 7, False)]
    fast = _fast(keys, [0, 0, 1, 2, 3, 4, 5, 6, 0], hits)
    for group, direction in ((0, 0), (4, 1), (5, 0), (8, 1)):
        _succeed(fast, group, direction)
    # Failed hypotheses with individually plausible flags stay failed and retained.
    fast["lr_certified"][2:4] = True
    fast["lr_refit_numerical_failure"][2] = True
    fast["lr_refit_error"][3] = True
    fast["lr_status"][2] = "numerical_failure"
    fast["lr_status"][3] = "refit_error"
    fast["lr_lr_stationary"][6, 1] = False
    fast["lr_status"][6, 1] = "nonstationary"
    fast["lr_objective"][0] = [1.0, np.inf]
    skeys = [keys[i] for i in (2, 0, 5, 3, 6, 1)] + [(99, 1, 7)]
    stock = _stock(skeys,
                   [(1, 1, 2, 0), (5, -1, 2, 1), (1, 0, 2, 1),
                    (0, 2, 0, 1), (2, 2, 2, 1), (6, 1, 2, 1),
                    (1, 2, 2, 1), (5, 1, 1, 1)],
                   [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, 4, 1),
                    (1, 0, -1), (2, 2, 0), (3, 128, 1), (4, 2, 1),
                    (5, 0, 1), (7, 3, 1)])
    return fast, stock


def _oracle(fast, stock):
    """Exhaustively visit all group/track pairs and all hit rows in Python."""
    keys = [tuple(k) for k in fast["evkey"]]
    skeys = [tuple(k) for k in stock["evkey"]]
    supports = [{int(s) for s, g, use in zip(fast["source_hit_sid"], fast["hit_group"],
                                            fast["source_selected"]) if use and g == group}
                for group in range(len(fast["group_ev"]))]
    straws = [{int(s) for s, t in zip(stock["hit_sid"], stock["hit_track"]) if t == track}
              for track in range(len(stock["track_ev"]))]
    pairs = []
    for g, event in enumerate(fast["group_ev"]):
        for t, stock_event in enumerate(stock["track_ev"]):
            if keys[event] == skeys[stock_event]:
                shared = len(supports[g] & straws[t])
                fs, ss = len(supports[g]), len(straws[t])
                union = len(supports[g] | straws[t])
                pairs.append((g, t, shared, fs, ss, shared/fs if fs else np.nan,
                              shared/ss if ss else np.nan, shared/union if union else np.nan))
    expected = {}
    for col, name in enumerate(("group", "track", "shared_straws", "fast_support", "stock_straws",
                                "fast_fraction", "stock_fraction", "jaccard")):
        expected["comparison_pair_" + name] = np.array([row[col] for row in pairs],
                                                       dtype=np.int64 if col < 5 else float)
    row = [skeys.index(k) if k in skeys else -1 for k in keys]
    expected["comparison_event_stock_row"] = row
    expected["comparison_event_stock_present"] = [r >= 0 for r in row]
    expected["comparison_stock_event_fast_row"] = [keys.index(k) if k in keys else -1 for k in skeys]
    expected["comparison_stock_track_unique_straws"] = [len(s) for s in straws]
    expected["comparison_stock_track_recorded_hits"] = [sum(t == track for t in stock["hit_track"])
                                                       for track in range(len(straws))]
    for name in ("all", "ch", "ch_positive_status", "ch_positive_status_goodfit"):
        counts = []
        for event in row:
            n = 0
            for te, status, alg, good in zip(stock["track_ev"], stock["track_status"],
                                            stock["track_fitalg"], stock["track_goodfit"]):
                eligible = name == "all" or (alg == 2 and
                            (name == "ch" or (status > 0 and
                             (name == "ch_positive_status" or good == 1))))
                n += int(te == event and eligible)
            counts.append(n if event >= 0 else -1)
        expected["comparison_event_stock_" + name] = counts
    for name in ("groups", "seed_groups", "valid_groups", "combined_groups"):
        counts = []
        for event in range(len(keys)):
            n = 0
            for g, ge in enumerate(fast["group_ev"]):
                eligible = {"groups": True, "seed_groups": bool(supports[g]),
                            "valid_groups": any(fast["lr_objective_valid"][g]),
                            "combined_groups": any(fast["lr_success"][g])}[name]
                n += int(ge == event and eligible)
            counts.append(n)
        expected["comparison_event_fast_" + name] = counts
    expected["comparison_event_fast_hits"] = [sum(h == e for h in fast["source_hit_ev"])
                                              for e in range(len(keys))]
    contingency = dict(both=0, fast_group_only=0, stock_track_only=0, neither=0)
    for e, r in enumerate(row):
        if r >= 0:
            f = expected["comparison_event_fast_combined_groups"][e] > 0
            s = expected["comparison_event_stock_ch_positive_status"][e] > 0
            contingency[("both" if s else "fast_group_only") if f else
                        ("stock_track_only" if s else "neither")] += 1
    thresholds = {}
    for threshold in (1, 4, 8):
        thresholds[str(threshold)] = dict(all_pairs=0, combined_group_positive_status_ch_pairs=0)
        for g, t, shared, *_ in pairs:
            if shared >= threshold:
                thresholds[str(threshold)]["all_pairs"] += 1
                if (any(fast["lr_success"][g]) and stock["track_fitalg"][t] == 2
                        and stock["track_status"][t] > 0):
                    thresholds[str(threshold)]["combined_group_positive_status_ch_pairs"] += 1
    return expected, contingency, thresholds


def _assert_oracle(fast, stock, **kwargs):
    expected, contingency, thresholds = _oracle(fast, stock)
    report, out = compare.overlap(fast, stock, **kwargs)
    assert {k for k in out if k.startswith("comparison_")} == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(out[name], value, rtol=0, atol=0, equal_nan=True, err_msg=name)
    assert report["covered_event_contingency"] == contingency
    assert report["overlap_counts"] == thresholds
    assert report["same_event_group_track_pairs"] == len(expected["comparison_pair_group"])
    for prefix, original in (("", fast), ("stock_", stock)):
        for name, value in original.items():
            np.testing.assert_array_equal(out[prefix + name], value, err_msg=prefix + name)
            assert out[prefix + name].dtype == value.dtype
            assert not np.shares_memory(out[prefix + name], value)
    json.dumps(report, allow_nan=False)
    return report, out


def test_all_pairs_full_keys_failure_lanes_duplicates_and_unknown_events(census):
    fast, stock = census
    report, out = _assert_oracle(fast, stock)
    assert report["covered_event_contingency"] == dict(both=1, fast_group_only=1,
                                                       stock_track_only=1, neither=3)
    assert report["missing_stock_events_unknown"] == 1
    assert report["missing_stock_events_with_combined_group"] == 1
    assert report["stock_events_outside_fast_census"] == 1
    assert report["stock_repeated_straw_hit_entries"] == 1
    assert report["combined_hypotheses"] == report["combined_groups"] == 4
    assert report["groups"] == 9 and report["hits"] == 13
    assert report["stock_status_counts"] == {"-1": 1, "0": 1, "1": 3, "2": 3}
    assert report["stock_fitalg_counts"] == {"0": 1, "1": 1, "2": 6}
    assert report["stock_goodfit_counts"] == {"0": 1, "1": 7}
    np.testing.assert_array_equal(out["comparison_event_stock_all"], [3, 2, 1, 0, -1, 1, 0])
    np.testing.assert_array_equal(out["comparison_event_stock_ch_positive_status"], [2, 0, 0, 0, -1, 1, 0])
    np.testing.assert_array_equal(out["comparison_event_stock_ch_positive_status_goodfit"], [1, 0, 0, 0, -1, 1, 0])
    assert set(out["comparison_pair_track"]) == {0, 1, 2, 3, 4, 6, 7}
    assert 1 in out["comparison_pair_group"]  # zero-support group is not pruned
    assert np.any(out["comparison_pair_shared_straws"] == 0)
    assert out["stock_hit_sid"][0] == out["stock_hit_sid"][1] == 0
    assert out["comparison_stock_track_unique_straws"][0] == 3
    assert out["comparison_stock_track_recorded_hits"][0] == 4


def test_exhaustive_tiny_straw_sets():
    # Exhaust all 8 x 8 support/track-set combinations, with duplicated recorded hits.
    subsets = [tuple(s for s in range(3) if mask & (1 << s)) for mask in range(8)]
    for fs, ss in itertools.product(subsets, repeat=2):
        fast = _fast([(1, 2, 3)], [0, 0], [(0, s, 0, True) for s in fs])
        if fs:
            _succeed(fast, 0, 1)
        stock = _stock([(1, 2, 3)], [(0, 2, 2, 0), (0, -1, 0, 1)],
                       [(0, s, state) for s in ss for state in (-1, 1)])
        _assert_oracle(fast, stock)


@pytest.mark.parametrize("n", [0, 1, 3, 4, 7, 8, 9])
def test_threshold_boundaries_and_both_direction_success(n):
    fast = _fast([(1, 0, 1)], [0], [(0, s, 0, True) for s in range(n)])
    if n:
        _succeed(fast, 0, 0)
        _succeed(fast, 0, 1)
    stock = _stock([(1, 0, 1)], [(0, 1, 2, 0)], [(0, s, -1) for s in range(n)])
    report, _ = _assert_oracle(fast, stock)
    assert report["combined_hypotheses"] == (2 if n else 0)
    assert report["combined_groups"] == (1 if n else 0)


def _permuted(fast, stock, rng):
    fast, stock = ({k: v.copy() for k, v in d.items()} for d in (fast, stock))
    ep, gp, hp = (rng.permutation(len(fast[k])) for k in ("evkey", "group_ev", "hit_group"))
    ei, gi = np.argsort(ep), np.argsort(gp)
    fast["evkey"] = fast["evkey"][ep]
    fast["group_ev"] = ei[fast["group_ev"][gp]]
    for k in fast:
        if k.startswith("lr_") or k == "group_support":
            fast[k] = fast[k][gp]
        elif k.startswith("source_") or k == "hit_group":
            fast[k] = fast[k][hp]
    fast["source_hit_ev"] = ei[fast["source_hit_ev"]]
    grouped = fast["hit_group"] >= 0
    fast["hit_group"][grouped] = gi[fast["hit_group"][grouped]]
    ep, tp, hp = (rng.permutation(len(stock[k])) for k in ("evkey", "track_ev", "hit_track"))
    ei, ti = np.argsort(ep), np.argsort(tp)
    for k in stock:
        stock[k] = stock[k][ep if k == "evkey" else tp if k.startswith("track_") else hp]
    stock["track_ev"] = ei[stock["track_ev"]]
    stock["hit_track"] = ti[stock["hit_track"]]
    return fast, stock


def test_reordered_events_groups_tracks_hits(census):
    baseline, _ = _assert_oracle(*census)
    rng = np.random.default_rng(241)
    for _ in range(12):
        report, _ = _assert_oracle(*_permuted(*census, rng))
        assert report == baseline


@pytest.mark.parametrize("fast_events,groups,stock_events,tracks", [
    (0, 0, 0, 0), (2, 0, 0, 0), (0, 0, 2, 2), (2, 2, 0, 0),
    (2, 0, 2, 2), (2, 2, 2, 0), (2, 2, 2, 2), (2, 0, 2, 0),
])
def test_empty_censuses_and_all_zero_support(fast_events, groups, stock_events, tracks):
    keys = [(1, 0, 1), (1, 0, 2)]
    fast = _fast(keys[:fast_events], [0] * groups)
    stock = _stock(keys[:stock_events], [(0, -1, 0, 0)] * tracks)
    report, out = _assert_oracle(fast, stock)
    assert report["combined_hypotheses"] == 0
    assert np.isnan(out["comparison_pair_fast_fraction"]).all()
    assert np.isnan(out["comparison_pair_stock_fraction"]).all()
    assert np.isnan(out["comparison_pair_jaccard"]).all()


def test_source_hits_survive_without_any_group():
    fast = _fast([(1, 0, 1), (1, 0, 2)], (), [(0, 3, -1, False), (0, 3, -1, False)])
    stock = _stock([(1, 0, 1)], [(0, 1, 2, 1)], [(0, 3, 1)])
    report, out = _assert_oracle(fast, stock, max_pairs=0, max_cells=0)
    assert report["hits"] == 2 and report["groups"] == 0
    np.testing.assert_array_equal(out["comparison_event_fast_hits"], [2, 0])
    assert report["covered_event_contingency"]["stock_track_only"] == 1


def test_input_snapshots_read_only_and_independent(census):
    fast, stock = census
    snapshots = [{k: v.copy() for k, v in d.items()} for d in census]
    for d in census:
        for value in d.values():
            value.flags.writeable = False
    _, out = _assert_oracle(fast, stock)
    out["source_hit_sid"][0] = 42
    out["stock_hit_sid"][0] = 43
    out["lr_status"][0, 0] = "changed"
    for original, snapshot in zip(census, snapshots):
        for k in snapshot:
            np.testing.assert_array_equal(original[k], snapshot[k], err_msg=k)
            assert not original[k].flags.writeable


@pytest.mark.parametrize("field", FLAGS)
def test_combined_flag_disagreement_rejected(census, field):
    fast, stock = census
    fast["lr_" + field][0, 0] = not fast["lr_" + field][0, 0]
    with pytest.raises(ValueError, match="success flags"):
        compare.overlap(fast, stock)


@pytest.mark.parametrize("lane,value", [((0, 0), "failed"), ((0, 1), "converged_certified")])
def test_status_disagreement_rejected(census, lane, value):
    fast, stock = census
    fast["lr_status"][lane] = value
    with pytest.raises(ValueError, match="success/status"):
        compare.overlap(fast, stock)


@pytest.mark.parametrize("field", FLAGS + ("status",))
@pytest.mark.parametrize("problem", ["shape", "dtype"])
def test_invalid_lr_lanes_rejected(census, field, problem):
    fast, stock = census
    key = "lr_" + field
    fast[key] = fast[key][:, :1] if problem == "shape" else fast[key].astype(float if field != "status" else bool)
    with pytest.raises(ValueError):
        compare.overlap(fast, stock)


@pytest.mark.parametrize("side,field,value", [
    (0, "evkey", [[1, 2]]), (0, "evkey", [[1, 2, 3], [1, 2, 3]]),
    (0, "group_ev", [[0]]), (0, "group_ev", [-1]), (0, "group_ev", [7]),
    (0, "group_ev", [0.0]), (0, "source_hit_ev", [[0]]),
    (0, "source_hit_ev", [-1] * 13), (0, "source_hit_ev", [7] * 13),
    (0, "source_hit_sid", [0] * 12), (0, "source_hit_sid", [65536] * 13),
    (0, "source_hit_sid", [96] * 13), (0, "source_hit_sid", [6 << 7] * 13),
    (0, "source_hit_sid", [36 << 10] * 13), (0, "source_hit_sid", [-1] * 13),
    (0, "hit_group", [-2] * 13), (0, "hit_group", [9] * 13),
    (0, "hit_group", [0] * 12), (0, "source_selected", [1] * 13),
    (0, "source_selected", [True] * 12), (0, "group_support", [-1] * 9),
    (0, "group_support", [0] * 8), (0, "group_support", [0] * 9),
    (1, "evkey", [[-1, 0, 0]]), (1, "evkey", [[1.0, 0.0, 0.0]]),
    (1, "track_ev", [[0]]), (1, "track_ev", [-1] * 8), (1, "track_ev", [7] * 8),
    (1, "hit_track", [[0]]), (1, "hit_track", [-1] * 10), (1, "hit_track", [8] * 10),
    (1, "hit_sid", [0] * 9), (1, "hit_sid", [65536] * 10),
    (1, "hit_sid", [96] * 10), (1, "hit_sid", [6 << 7] * 10),
    (1, "hit_sid", [36 << 10] * 10), (1, "hit_sid", [-1] * 10),
    (1, "track_status", [0.0] * 8), (1, "track_status", [0] * 7),
    (1, "track_status", [-2147483649] * 8), (1, "track_fitalg", [-2] * 8),
    (1, "track_goodfit", [2] * 8), (1, "track_goodfit", [-1] * 8),
])
def test_invalid_census_shapes_bounds_and_types(census, side, field, value):
    census[side][field] = np.asarray(value)
    with pytest.raises(ValueError):
        compare.overlap(*census)


@pytest.mark.parametrize("problem", ["duplicate_selected", "selected_ungrouped", "wrong_event", "empty_valid"])
def test_invalid_support_or_provenance_rejected(census, problem):
    fast, stock = census
    if problem == "duplicate_selected":
        fast["source_hit_sid"][1] = 0
    elif problem == "selected_ungrouped":
        fast["source_selected"][4] = True
    elif problem == "wrong_event":
        fast["source_hit_ev"][0] = 1
    else:
        fast["lr_objective_valid"][1, 0] = True
    with pytest.raises(ValueError):
        compare.overlap(fast, stock)


@pytest.mark.parametrize("name", ["max_pairs", "max_cells"])
@pytest.mark.parametrize("value", [-1, True, np.bool_(False), 1.5, None, 4000001])
def test_invalid_allocation_limits(census, name, value):
    with pytest.raises(ValueError, match="allocation bounds"):
        compare.overlap(*census, **{name: value})


def test_exact_allocation_bounds_and_fail_without_truncation(census):
    fast, stock = census
    pairs = cells = 0
    for ge in fast["group_ev"]:
        for t, te in enumerate(stock["track_ev"]):
            if tuple(fast["evkey"][ge]) == tuple(stock["evkey"][te]):
                pairs += 1
                cells += len({int(s) for s, ht in zip(stock["hit_sid"], stock["hit_track"]) if ht == t})
    _assert_oracle(fast, stock, max_pairs=pairs, max_cells=cells)
    for limits in (dict(max_pairs=pairs - 1), dict(max_cells=cells - 1),
                   dict(max_pairs=0), dict(max_cells=0)):
        with pytest.raises(ValueError, match="bound exceeded; no truncation"):
            compare.overlap(fast, stock, **limits)
    _assert_oracle(_fast([]), _stock([]), max_pairs=0, max_cells=0)


def test_bounds_count_empty_support_groups_but_not_duplicate_stock_hits():
    fast = _fast([(1, 0, 1)], [0, 0])
    stock = _stock([(1, 0, 1)], [(0, -1, 0, 0)], [(0, 0, 0)] * 20)
    # Both zero-support groups still allocate a pair and one unique-hit cell each.
    _assert_oracle(fast, stock, max_pairs=2, max_cells=2)
    with pytest.raises(ValueError, match="no truncation"):
        compare.overlap(fast, stock, max_pairs=1)
    with pytest.raises(ValueError, match="no truncation"):
        compare.overlap(fast, stock, max_cells=1)
    empty_track = _stock([(1, 0, 1)], [(0, -1, 0, 0)])
    _assert_oracle(fast, empty_track, max_pairs=np.int64(2), max_cells=np.int64(0))
    with pytest.raises(ValueError, match="allocation bounds"):
        compare.overlap(fast, empty_track, max_pairs=2000001)


def test_full_int64_keys_and_maximum_valid_packed_straw():
    large = 2**53
    last = np.iinfo(np.int64).max
    keys = [(large, 1, last), (large + 1, 1, last), (large, 2, last)]
    sid = (35 << 10) | (5 << 7) | 95
    fast = _fast(keys, [0, 1, 2], [(e, sid, e, True) for e in range(3)])
    stock = _stock(keys[::-1], [(e, 1, 2, 0) for e in range(3)],
                   [(e, sid, 1) for e in range(3)])
    _, out = _assert_oracle(fast, stock)
    np.testing.assert_array_equal(out["comparison_pair_track"], [2, 1, 0])
    np.testing.assert_array_equal(out["comparison_pair_shared_straws"], [1, 1, 1])


@pytest.mark.parametrize("side,field", [(0, "evkey"), (1, "evkey"),
                                       (0, "group_support"), (1, "track_status"),
                                       (1, "track_fitalg")])
def test_unsigned_int64_overflow_rejected(census, side, field):
    census[side][field] = census[side][field].astype(np.uint64)
    census[side][field].flat[0] = np.uint64(2**63)
    with pytest.raises(ValueError):
        compare.overlap(*census)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_lr(tmp_path, fast):
    hits = tmp_path / "hits.npz"
    np.savez(hits, evkey=fast["evkey"], s_ev=fast["source_hit_ev"], s_sid=fast["source_hit_sid"])
    directory = tmp_path / "lr"
    directory.mkdir()
    payload = directory / "diagnostics.npz"
    np.savez(payload, **fast)
    report = dict(output_payload_sha256=_sha(payload), events=len(fast["evkey"]),
                  groups=len(fast["group_ev"]), hits=len(fast["hit_group"]),
                  fixed_selected_hits=int(fast["source_selected"].sum()),
                  final_combined_checked=dict(hypotheses=int(fast["lr_success"].sum())),
                  inputs=dict(hits=dict(path=str(hits), sha256=_sha(hits))))
    (directory / "report.json").write_text(json.dumps(report))
    return directory, report


def test_load_fast_binds_saved_payload_and_measured_identity(tmp_path, census):
    fast, _ = census
    directory, original = _save_lr(tmp_path, fast)
    loaded, report = compare.load_fast(directory)
    assert report == original
    assert set(loaded) == set(fast)
    for key in fast:
        np.testing.assert_array_equal(loaded[key], fast[key], err_msg=key)


@pytest.mark.parametrize("problem", ["payload_hash", "missing_hash", "input_hash", "events", "groups",
                                     "hits", "fixed_selected_hits", "hypotheses"])
def test_load_fast_hash_and_report_tampering(tmp_path, census, problem):
    fast, _ = census
    directory, report = _save_lr(tmp_path, fast)
    if problem == "payload_hash":
        (directory / "diagnostics.npz").write_bytes(b"not an npz")
    elif problem == "missing_hash":
        del report["output_payload_sha256"]
    elif problem == "input_hash":
        (tmp_path / "hits.npz").write_bytes(b"changed input")
    elif problem == "hypotheses":
        report["final_combined_checked"]["hypotheses"] += 1
    else:
        report[problem] += 1
    (directory / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="hash|census"):
        compare.load_fast(directory)


@pytest.mark.parametrize("field", ["evkey", "s_ev", "s_sid"])
def test_load_fast_rehashed_measured_identity_tampering(tmp_path, census, field):
    directory, report = _save_lr(tmp_path, census[0])
    hits = tmp_path / "hits.npz"
    with np.load(hits, allow_pickle=False) as data:
        payload = dict(data)
    payload[field] = payload[field].copy()
    payload[field].flat[0] += 1
    np.savez(hits, **payload)
    report["inputs"]["hits"]["sha256"] = _sha(hits)
    (directory / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity mismatch"):
        compare.load_fast(directory)


@pytest.mark.parametrize("field", ["stock_evkey", "comparison_pair_group"])
def test_load_fast_rejects_comparison_as_lr_even_with_updated_hash(tmp_path, census, field):
    directory, report = _save_lr(tmp_path, census[0])
    payload = dict(census[0], **{field: np.array([0])})
    np.savez(directory / "diagnostics.npz", **payload)
    report["output_payload_sha256"] = _sha(directory / "diagnostics.npz")
    (directory / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="comparison output"):
        compare.load_fast(directory)


@pytest.mark.parametrize("field", ["source_selected", "group_support", "lr_success", "lr_status"])
def test_load_fast_rehashed_invalid_fixed_support_and_flags(tmp_path, census, field):
    directory, report = _save_lr(tmp_path, census[0])
    payload = {k: v.copy() for k, v in census[0].items()}
    if field == "group_support":
        payload[field][0] += 1
    elif field == "lr_status":
        payload[field][0, 0] = "failed"
    else:
        payload[field].flat[0] = not payload[field].flat[0]
    np.savez(directory / "diagnostics.npz", **payload)
    report["output_payload_sha256"] = _sha(directory / "diagnostics.npz")
    (directory / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare.load_fast(directory)


def _cli(tmp_path, monkeypatch, census):
    directory, _ = _save_lr(tmp_path, census[0])
    stockpath = tmp_path / "stock.root"
    stockpath.write_bytes(b"mocked recorded stock file")
    calls = []

    def read_stock(path, **kwargs):
        assert path == stockpath
        calls.append(kwargs)
        return census[1], dict(tree_key="custom/tree;3", recorded_hit_census_complete=True)

    monkeypatch.setattr(compare.stockcensus, "read_stock", read_stock)
    argv = ["--lr", str(directory), "--stock", str(stockpath), "--tree", "custom/tree",
            "--chunk-entries", "2"]
    return argv, calls


def test_cli_new_output_exact_replay_no_overwrite_and_immutable_inputs(tmp_path, monkeypatch, census, capsys):
    argv, calls = _cli(tmp_path, monkeypatch, census)
    inputs = {p: p.read_bytes() for p in (tmp_path / "hits.npz", tmp_path / "stock.root",
                                         tmp_path / "lr" / "report.json", tmp_path / "lr" / "diagnostics.npz")}
    first = tmp_path / "first"
    report = compare.main(argv + ["--outdir", str(first)])
    assert json.loads(capsys.readouterr().out) == report
    assert json.loads((first / "report.json").read_text()) == report
    assert report["output_payload_sha256"] == _sha(first / "diagnostics.npz")
    assert calls == [dict(tree="custom/tree", chunk_entries=2)]
    _, expected = compare.overlap(*census)
    with np.load(first / "diagnostics.npz", allow_pickle=False) as actual:
        assert set(actual.files) == set(expected)
        for k, value in expected.items():
            np.testing.assert_array_equal(actual[k], value, err_msg=k)
    for category in ("inputs", "implementation"):
        for info in report[category].values():
            assert info["sha256"] == _sha(Path(info["path"]))
    saved = {p: p.read_bytes() for p in first.iterdir()}
    with pytest.raises(FileExistsError):
        compare.main(argv + ["--outdir", str(first)])
    assert len(calls) == 1  # reject existing output before reading anything
    for p, content in saved.items():
        assert p.read_bytes() == content
    second = tmp_path / "second"
    replay = compare.main(argv + ["--outdir", str(second), "--compare-saved", str(first / "diagnostics.npz")])
    assert replay["saved_comparison"] == dict(exact=True, sha256=_sha(first / "diagnostics.npz"))
    for p, content in inputs.items():
        assert p.read_bytes() == content


@pytest.mark.parametrize("problem", ["value", "missing", "extra", "shape", "failed_lane", "dtype"])
def test_cli_exact_comparison_rejects_changes_before_output(tmp_path, monkeypatch, census, problem):
    argv, _ = _cli(tmp_path, monkeypatch, census)
    _, saved = compare.overlap(*census)
    if problem == "value":
        saved["comparison_pair_shared_straws"][0] += 1
    elif problem == "missing":
        del saved["source_time"]
    elif problem == "extra":
        saved["extra"] = np.array([0])
    elif problem == "shape":
        saved["comparison_pair_group"] = saved["comparison_pair_group"][:, None]
    elif problem == "dtype":
        saved["comparison_pair_shared_straws"] = saved["comparison_pair_shared_straws"].astype(float)
    else:
        saved["lr_objective"][0, 1] = 123.0
    previous = tmp_path / "previous.npz"
    np.savez(previous, **saved)
    destination = tmp_path / "never-created"
    with pytest.raises(ValueError, match="saved comparison arrays differ"):
        compare.main(argv + ["--outdir", str(destination), "--compare-saved", str(previous)])
    assert not destination.exists()


def test_cli_detects_stock_file_changed_during_read(tmp_path, monkeypatch, census):
    argv, _ = _cli(tmp_path, monkeypatch, census)

    def changing_reader(path, **kwargs):
        path.write_bytes(b"changed while reading")
        return census[1], {}

    monkeypatch.setattr(compare.stockcensus, "read_stock", changing_reader)
    destination = tmp_path / "never-created"
    with pytest.raises(ValueError, match="changed during read"):
        compare.main(argv + ["--outdir", str(destination)])
    assert not destination.exists()


@pytest.mark.parametrize('problem', ['false', 'shape', 'dtype', 'missing'])
def test_saved_lr_hit_mask_matches_selected_support(census, problem):
    fast, stock = census
    if problem == 'false':
        fast['source_lr_mask'][0, 0] = False
    elif problem == 'shape':
        fast['source_lr_mask'] = fast['source_lr_mask'][:, :1]
    elif problem == 'dtype':
        fast['source_lr_mask'] = fast['source_lr_mask'].astype(int)
    else:
        del fast['source_lr_mask']
    with pytest.raises((ValueError, KeyError)):
        compare.overlap(fast, stock)


@pytest.mark.parametrize('field', ['source_lr_mask', 'lr_analytic_stationary'])
@pytest.mark.parametrize('problem', ['false', 'missing'])
def test_rehashed_lr_mask_and_analytic_flag_contradictions_rejected(tmp_path, census, field, problem):
    fast = {k: v.copy() for k, v in census[0].items()}
    if problem == 'false':
        fast[field][0, 0] = False
    else:
        del fast[field]
    directory, _ = _save_lr(tmp_path, fast)
    with pytest.raises((ValueError, KeyError)):
        compare.load_fast(directory)


@pytest.mark.parametrize('side,field', [(0, 'source_hit_sid'), (0, 'source_selected'),
                                      (0, 'source_lr_mask'), (0, 'lr_success'), (1, 'hit_sid'),
                                      (0, 'lr_status'), (0, 'lr_objective'), (0, 'source_time'),
                                      (1, 'hit_state'), (1, 'track_chisq')])
def test_masked_missing_census_values_rejected(census, side, field):
    value = census[side][field]
    mask = np.zeros(value.shape, bool)
    mask.flat[0] = True
    census[side][field] = np.ma.array(value, mask=mask)
    with pytest.raises(ValueError, match='masked'):
        compare.overlap(*census)


def test_masked_string_cannot_hide_success_status_contradiction(census):
    fast, stock = census
    fast['lr_status'][0, 1] = 'converged_certified'
    mask = np.zeros(fast['lr_status'].shape, bool)
    mask[0, 1] = True
    fast['lr_status'] = np.ma.array(fast['lr_status'], mask=mask)
    with pytest.raises(ValueError, match='masked'):
        compare.overlap(fast, stock)


def test_coverage_details_exclusive_event_stage_hierarchy():
    keys = [(1, 2, e) for e in range(7)]
    fast = _fast(keys, [1, 2, 3, 4, 5], [(2, 1, 1, True), (3, 1, 2, True),
                                       (4, 1, 3, True), (5, 1, 4, True)])
    fast['lr_objective_valid'][1] = False
    _succeed(fast, 3, 0)
    stock = _stock(keys[:5] + [keys[6]], [(i, 1, 2, 1) for i in range(5)])
    report, _ = compare.overlap(fast, stock)
    assert report['recorded_ch_event_fast_stage_counts'] == dict(
        no_group=1, no_seed_support=1, no_valid_objective=1, no_combined_pass=1, combined_pass=1)
    assert sum(report['recorded_ch_event_fast_stage_counts'].values()) == 5
    assert report['groups_in_recorded_ch_events'] == 4
    assert report['recorded_ch_group_lane_status_counts']['all_groups'] == {
        'converged_certified': 1, 'uncertified': 7}
    assert report['recorded_ch_group_lane_status_counts']['groups_without_combined_pass'] == {'uncertified': 6}
    assert report['missing_stock_events_unknown'] == 1


def test_pair_overlap_summaries_independent_oracle(census):
    fast, stock = census
    report, out = compare.overlap(fast, stock)
    good_group = np.any(fast['lr_success'], axis=1)
    for label, expected in report['pair_overlap_summaries'].items():
        pair_rows = []
        for row, (g, t) in enumerate(zip(out['comparison_pair_group'], out['comparison_pair_track'])):
            ch = stock['track_fitalg'][t] == 2 and stock['track_status'][t] > 0
            include = (label == 'all_pairs' or (ch and (label == 'positive_status_ch_pairs' or
                (good_group[g] if label == 'combined_group_positive_status_ch_pairs' else not good_group[g]))))
            if include:
                pair_rows.append(row)
        rows = np.array(pair_rows, dtype=int)
        assert expected['pairs'] == len(rows)
        for field in ('shared_straws', 'fast_support', 'stock_straws', 'fast_fraction', 'stock_fraction', 'jaccard'):
            values = out['comparison_pair_'+field][rows]
            finite = values[np.isfinite(values)]
            assert expected[field]['finite'] == len(finite)
            want = dict(zip(('min', 'median', 'p90', 'max'), np.quantile(finite, [0, .5, .9, 1]))) if len(finite) else {}
            assert expected[field]['quantiles'] == want
        counts = dict(fast_subset_of_stock_nonempty=0, stock_subset_of_fast_nonempty=0,
                      identical_nonempty_straw_sets=0)
        for row in rows:
            shared = out['comparison_pair_shared_straws'][row]
            fs = out['comparison_pair_fast_support'][row]
            ss = out['comparison_pair_stock_straws'][row]
            counts['fast_subset_of_stock_nonempty'] += int(fs > 0 and shared == fs)
            counts['stock_subset_of_fast_nonempty'] += int(ss > 0 and shared == ss)
            counts['identical_nonempty_straw_sets'] += int(fs > 0 and shared == fs == ss)
        for key, value in counts.items():
            assert expected[key] == value


def test_report_empty_pairs_and_unknown_events_are_not_failure_stages():
    fast = _fast([(1, 0, 1)], [0])
    stock = _stock([(2, 0, 1)], [(0, 1, 2, 1)], [(0, 1, 0)])
    report, _ = compare.overlap(fast, stock)
    assert sum(report['recorded_ch_event_fast_stage_counts'].values()) == 0
    for summary in report['pair_overlap_summaries'].values():
        assert summary['pairs'] == 0 and summary['fast_fraction'] == dict(finite=0, quantiles={})
    assert report['recorded_ch_group_lane_status_counts'] == dict(all_groups={}, groups_without_combined_pass={})
