"""Full-key timing routing and writer metadata, without running digitization kernels."""
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.digi import fastdigi as F


def _keys():
    return np.array([[1400, 11, e] for e in (9, 2, 8, 1, 4, 7, 3)], np.int64)


def _frame(keys):
    rows = []
    for i, key in enumerate(keys):
        row = dict.fromkeys(F.STEP_COLS, 0)
        row.update(zip(("run", "subrun", "event"), key))
        row.update(straw=i, t=200. + i)
        rows.append(row)
    return pd.DataFrame(rows, columns=F.STEP_COLS)


def _timing(n):
    return {"pbtime": np.arange(n, dtype=np.float64) * 11.25 - 20.,
            "event_length": 1700. + np.arange(n, dtype=np.float64)}


def _digis(st):
    n = len(st["ev"])
    return {"ev": st["ev"].copy(), "sid": st["sid"].copy(),
            "step": np.repeat(np.arange(n)[:, None], 2, axis=1),
            "tdc": np.ones((n, 2), np.int32), "tot": np.ones((n, 2), np.int32),
            "pmp": np.ones(n, np.int32), "adc": np.zeros((n, 3), np.int32)}


@pytest.mark.parametrize("backend", ["numpy", "jax", "fallback"])
@pytest.mark.parametrize("compact", [False, True])
def test_full_population_timing_routes_through_chunks(monkeypatch, backend, compact):
    keys = _keys()
    frame = _frame(keys[[5, 0, 2]])
    timing = _timing(len(keys))
    calls, fallback_calls, stats = [], [], {}

    def fake(st, nev, c, rng, counters, *, event_timing):
        assert c.onspill and counters is stats
        calls.append((nev, {k: v.copy() for k, v in event_timing.items()}))
        if backend == "fallback":
            raise OverflowError("mock device capacity")
        return _digis(st)

    def fallback(st, nev, c, rng, counters, *, event_timing):
        assert isinstance(rng, np.random.Generator)
        fallback_calls.append(event_timing)
        return _digis(st)

    monkeypatch.setattr(F.D, "digitize", fallback if backend == "fallback" else fake)
    monkeypatch.setattr(F, "digitize_jax", fake)
    rng = np.random.default_rng(2)
    if backend != "numpy":
        rng = pytest.importorskip("jax").random.key(2)
    got_keys, d, n = F.digitize_steps(frame, SimpleNamespace(onspill=True, digitizationEnd=1705., timeFromProtonsToDRMarker=200.), rng, keys,
                                    chunk=2, compact=compact, stats=stats,
                                    backend="numpy" if backend == "numpy" else "jax", event_timing=timing)
    np.testing.assert_array_equal(got_keys, keys)
    np.testing.assert_array_equal(d["ev"], [0, 2, 5])
    np.testing.assert_array_equal(d["step"], [[1, 1], [2, 2], [0, 0]])
    assert n == 3
    population = np.array([0, 2, 5]) if compact else np.arange(len(keys))
    assert len(calls) == (len(population) + 1) // 2
    for i, (nev, values) in enumerate(calls):
        selection = population[2 * i:2 * i + 2]
        assert nev == len(selection)
        for name in timing:
            np.testing.assert_array_equal(values[name], timing[name][selection])
            if backend == "fallback":
                np.testing.assert_array_equal(fallback_calls[i][name], values[name])
    if backend == "fallback":
        assert stats["numpy_fallback_chunks"] == len(calls)
    # Empty events survive in the full output population, including their pbtime.
    read = F.digis_as_read(keys, d, 3, event_timing=timing)
    np.testing.assert_array_equal(read["pbt"], timing["pbtime"].astype(np.float32))
    np.testing.assert_array_equal(read["evkey"], keys)


@pytest.mark.parametrize("compact", [False, True])
def test_all_events_empty_keep_timing_and_keys(monkeypatch, compact):
    keys, calls = _keys(), []
    timing = _timing(len(keys))

    def fake(st, nev, c, rng, stats, *, event_timing):
        assert len(st["ev"]) == 0
        calls.append(event_timing["pbtime"])
        return _digis(st)

    monkeypatch.setattr(F.D, "digitize", fake)
    got_keys, d, n = F.digitize_steps(_frame([]), SimpleNamespace(onspill=True, digitizationEnd=1705., timeFromProtonsToDRMarker=200.), None, keys,
                                    chunk=3, compact=compact, event_timing=timing)
    assert n == 0 and len(d["ev"]) == 0
    np.testing.assert_array_equal(got_keys, keys)
    np.testing.assert_array_equal(np.concatenate(calls), timing["pbtime"])
    np.testing.assert_array_equal(F.events_as_read(keys, event_timing=timing)["pbt"], timing["pbtime"])


@pytest.mark.parametrize("keys", [_keys(), np.empty((0, 3), np.int64)])
def test_onspill_requires_timing_even_without_steps(keys):
    with pytest.raises(ValueError, match="required on spill"):
        F.digitize_steps(_frame([]), SimpleNamespace(onspill=True, digitizationEnd=1705., timeFromProtonsToDRMarker=200.), None, keys)


@pytest.mark.parametrize("bad", [
    {}, {"pbtime": [0.]},
    {"pbtime": np.zeros(3), "event_length": np.ones(3)},
    {"pbtime": np.zeros((7, 1)), "event_length": np.ones(7)},
    {"pbtime": np.full(7, np.nan), "event_length": np.ones(7)},
    {"pbtime": np.zeros(7), "event_length": np.full(7, np.inf)},
    {"pbtime": np.zeros(7), "event_length": np.array([1.] * 6 + [0.])},
    {"pbtime": ["0"] * 7, "event_length": np.ones(7)},
])
def test_bad_timing_rejected_before_compact_slicing(monkeypatch, bad):
    def fail(*args, **kw):
        pytest.fail("invalid timing reached kernel")
    monkeypatch.setattr(F.D, "digitize", fail)
    keys = _keys()
    with pytest.raises(ValueError, match="event_timing|event_length"):
        F.digitize_steps(_frame(keys[:1]), SimpleNamespace(onspill=True, digitizationEnd=1705., timeFromProtonsToDRMarker=200.), None, keys,
                         compact=True, event_timing=bad)
    with pytest.raises(ValueError, match="event_timing|event_length"):
        F.events_as_read(keys, event_timing=bad)


def test_timing_requires_explicit_full_keys():
    with pytest.raises(ValueError, match="full event keys"):
        F.digitize_steps(_frame([]), SimpleNamespace(onspill=True, digitizationEnd=1705., timeFromProtonsToDRMarker=200.), None, event_timing=_timing(0))


def test_table_wrapper_forwards_only_explicit_timing(monkeypatch):
    frame, keys, calls = _frame([]), _keys(), []
    monkeypatch.setattr(F, "load_steps", lambda path: frame)
    def fake(*args, **kw):
        calls.append((args, kw))
        return "result"
    monkeypatch.setattr(F, "digitize_steps", fake)
    assert F.digitize_table("unused", None, None, keys) == "result"
    assert calls[-1][1] == {}
    timing = _timing(len(keys))
    F.digitize_table("unused", None, None, keys, event_timing=timing)
    assert calls[-1][1]["event_timing"] is timing


def test_jax_wrapper_forwards_only_explicit_timing(monkeypatch):
    calls = []
    def fake(*args, **kw):
        calls.append(kw)
        return "result"
    monkeypatch.setitem(sys.modules, "trkfastsim.digi.digi_jax", SimpleNamespace(digitize=fake))
    assert F.digitize_jax({}, 2, None, None) == "result"
    assert calls[-1] == {}
    timing = _timing(2)
    F.digitize_jax({}, 2, None, None, event_timing=timing)
    assert calls[-1]["event_timing"] is timing


@pytest.mark.parametrize("backend", ["numpy", "jax", "fallback"])
def test_none_timing_keeps_legacy_kernel_signature(monkeypatch, backend):
    def fake(st, nev, c, rng, stats):  # deliberately no kwargs
        return _digis(st)
    def overflow(st, nev, c, rng, stats):
        raise OverflowError("mock capacity")
    monkeypatch.setattr(F.D, "digitize", fake)
    monkeypatch.setattr(F, "digitize_jax", overflow if backend == "fallback" else fake)
    rng = None if backend == "numpy" else pytest.importorskip("jax").random.key(3)
    keys, d, n = F.digitize_steps(_frame(_keys()[:1]), SimpleNamespace(onspill=False), rng,
                                _keys(), chunk=2, backend="numpy" if backend == "numpy" else "jax")
    assert n == 1 and len(d["ev"]) == 1


@pytest.mark.parametrize("compact", [False, True])
def test_large_event_ids_do_not_collide(monkeypatch, compact):
    # Each pair collided under (run << 40) | (subrun << 20) | event.
    keys = np.array([[1, 0, 2**20 + 5], [1, 1, 5],
                     [2, 0, 2**40 + 7], [3, 0, 7]], np.int64)
    frame = _frame(np.vstack([keys[[3, 0, 2, 1]], [1, 1, 2**20 + 5]]))
    monkeypatch.setattr(F.D, "digitize", lambda st, *args: _digis(st))
    got_keys, d, n = F.digitize_steps(frame, SimpleNamespace(onspill=False), None, keys,
                                    chunk=1, compact=compact)
    assert n == 4  # the fifth row is NOT among the supplied keys
    np.testing.assert_array_equal(got_keys, keys)
    np.testing.assert_array_equal(d["ev"], [0, 1, 2, 3])
    np.testing.assert_array_equal(d["step"], [[1, 1], [3, 3], [2, 2], [0, 0]])


@pytest.mark.parametrize("onspill", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_output_timing_matches_readers_and_retains_empty_events(tmp_path, onspill, empty):
    from trkfastsim.reco import hitreco, tracktable
    keys = _keys()
    d = _digis({"ev": np.array([] if empty else [1, 5], np.int64),
                "sid": np.array([] if empty else [4, 5], np.int64)})
    timing = _timing(len(keys)) if onspill else None
    path = tmp_path / "digis.root"
    F.write_digis(path, keys, d, 3, event_timing=timing)
    with uproot.open(path) as f:
        a = f["trkdump/events"].arrays()
        assert len(a) == len(keys)
        np.testing.assert_array_equal(ak.to_numpy(a.spill), np.full(len(keys), int(onspill)))
        expected = timing["pbtime"] if onspill else np.zeros(len(keys))
        np.testing.assert_array_equal(ak.to_numpy(a.pbt), expected.astype(np.float32))
    for actual, expected in ((F.digis_as_read(keys, d, 3, event_timing=timing), hitreco.read_digis([path])),
                             (F.events_as_read(keys, event_timing=timing), tracktable._events(path))):
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])
            assert actual[name].dtype == expected[name].dtype


@pytest.mark.parametrize("compact", [False, True])
def test_empty_event_window_is_checked_before_compaction(compact):
    keys = _keys()[:2]
    timing = {"pbtime": np.zeros(2), "event_length": np.array([1700., 1504.])}
    cond = SimpleNamespace(onspill=True, digitizationEnd=1705., timeFromProtonsToDRMarker=200.)
    with pytest.raises(ValueError, match="past next event"):
        F.digitize_steps(_frame(keys[:1]), cond, None, keys, compact=compact, event_timing=timing)
