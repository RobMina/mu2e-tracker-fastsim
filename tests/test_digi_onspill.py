"""Explicit marker timing, stock ghost rounding and unchanged default NumPy output."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from trkfastsim.digi import digitize as DG
from trkfastsim.digi.digicond import Conditions
from trkfastsim.digi.timing import ghost_indices, step_times
from test_digi import cond, _step, _steps


def _onspill(cond):
    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = True
    return c


def _timing(lengths, pbtime=None):
    lengths = np.asarray(lengths, np.float64)
    return {"event_length": lengths, "pbtime": np.zeros_like(lengths) if pbtime is None else np.asarray(pbtime, np.float64)}


def test_ghost_strict_boundaries_and_no_recursive_folding():
    t = np.array([-4000, 0, 200, 1495, 1695, 6000], np.float32)
    idx, gt = ghost_indices(t, 1695., 200.)
    np.testing.assert_array_equal(idx, [0, 1, 2, 3, 4, 5, 0, 1, 4, 5])
    np.testing.assert_array_equal(gt, [-4000, 0, 200, 1495, 1695, 6000, -2305, 1695, 0, 4305])
    assert gt.dtype == np.float32 and idx.dtype == np.int64
    t = np.array([np.nextafter(np.float32(200), np.float32(0)),
                  np.nextafter(np.float32(1495), np.float32(2000))])
    np.testing.assert_array_equal(ghost_indices(t, 1695., 200.)[0], [0, 1, 0, 1])
    idx, gt = ghost_indices(np.array([500.], np.float32), 1000., 700.)
    np.testing.assert_array_equal(idx, [0, 0, 0])
    np.testing.assert_array_equal(gt, [500., 1500., -500.])


def test_ghost_double_shift_then_float_rounding():
    t = np.array([123.45678, 1600.125], np.float32)
    period = 1695.00007
    idx, gt = ghost_indices(t, period, 200.)
    expected = np.array([t[0], t[1], float(t[0]) + period, float(t[1]) - period], np.float32)
    np.testing.assert_array_equal(idx, [0, 1, 0, 1])
    np.testing.assert_array_equal(gt, expected)
    assert gt[-1] != np.float32(t[1] - np.float32(period))
    for bad in (np.array([[1.]]), np.array([np.inf]), np.array([1e100])):
        with pytest.raises(ValueError):
            ghost_indices(bad, period, 200.)


def test_step_times_marker_offset_and_single_fold(cond):
    c = _onspill(cond)
    st = _steps([_step(c, 520, t=t) for t in (-5100., -30., 0., 3400., 10000.)], ev=[0, 0, 1, 1, 1])
    timing = _timing([1695., 2000.], [15., -5.])
    time, lengths = step_times(st, 2, c, timing)
    np.testing.assert_array_equal(time, [0., 1680., 1690., 5., 1520.])
    np.testing.assert_array_equal(lengths, timing["event_length"])
    assert time.dtype == lengths.dtype == np.float64
    c.s["onspill"] = False
    time, _ = step_times(st, 2, c, timing)
    np.testing.assert_array_equal(time, [-5085., -15., -5., 3395., 9995.])


@pytest.mark.parametrize("field,bad", [("ev", [-1]), ("ev", [1]), ("ev", [.5]),
    ("sid", [96]), ("sid", [6 << 7]), ("sid", [36 << 10]), ("sid", [-1]), ("sid", [1.5]),
    ("t", [np.nan]), ("x0", [np.inf]), ("t", [[1.]])])
def test_invalid_step_inputs(cond, field, bad):
    st = _steps([_step(cond, 520)])
    st[field] = np.asarray(bad)
    with pytest.raises(ValueError):
        step_times(st, 1, cond, None)


@pytest.mark.parametrize("timing", [None, {}, {"pbtime": [0.]}, _timing([1695.], [np.nan]),
    _timing([np.inf]), _timing([0.]), _timing([-1.]), _timing([1504.]), _timing([1695., 1695.])])
def test_required_valid_timing_even_without_steps(cond, timing):
    c = _onspill(cond)
    st = {k: v[:0] for k, v in _steps([_step(c, 520)]).items()}
    rng = np.random.default_rng(6)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        DG.digitize(st, 1, c, rng, event_timing=timing)
    assert rng.bit_generator.state == before


def test_empty_event_populations(cond):
    c = _onspill(cond)
    st = {k: v[:0] for k, v in _steps([_step(c, 520)]).items()}
    for nev in (0, 3):
        d = DG.digitize(st, nev, c, np.random.default_rng(4), event_timing=_timing(np.full(nev, 1695.)))
        assert d["adc"].shape == (0, c.nADCSamples) and d["tdc"].shape == (0, 2)


def test_default_offspill_array_and_rng_regression(cond):
    st = _steps([_step(cond, 520), _step(cond, 521, t=1800.), _step(cond, 92, t=100.),
                 _step(cond, 522, t=3000.)], ev=[0, 0, 2, 2])
    rng = np.random.default_rng(23)
    d = DG.digitize(st, 3, cond, rng)
    expected = {
        "ev": "4e7de98684a483af6a8683603ec7241196683c7a4e0952e86b102c5dbf15cd5c",
        "sid": "f19c56b30294920151984ed9844a6cd5f21fcf7169b25ad91aed5a1951995478",
        "tdc": "eff7134f6aa2426308d886aaec162048f6fd6f15f86f6f2eb643ce15960d8bcc",
        "tot": "04f454743c3a03ce1b0ca7d6e2c234dd205561042860a923876e33bc68a91b45",
        "pmp": "2ee83820324102956c09d51fa48074fd6d95b932257874090a5e9227d2adb5fe",
        "adc": "98baeaef8d85ddbb35bceefcc8ce614fa899d0f0b6f7299b45f1302f9b2f7e11",
        "step": "180941be1558d74985419655da4d740bfd03c74d183912260494a08af4cf9bee",
        "t_analog": "1092e33996efa101f58d4b06706f04fb8a334cfd17df1bfadb414e9a58f88023",
    }
    assert {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in d.items()} == expected
    assert rng.bit_generator.state["state"]["state"] == 67086340549928322506611325226831435541
    other_rng = np.random.default_rng(23)
    other = DG.digitize(st, 3, cond, other_rng, event_timing=_timing([cond.eventWindowLength] * 3))
    assert all(np.array_equal(d[k], other[k]) for k in d)
    assert rng.bit_generator.state == other_rng.bit_generator.state


def test_periodic_equivalence_multiple_events_steps_and_lengths(cond):
    c = _onspill(cond)
    st = _steps([_step(c, 520, t=950.), _step(c, 521, t=1350.), _step(c, 92, t=50.),
                 _step(c, 522, t=700.), _step(c, 523, t=1200.)], ev=[0, 0, 2, 2, 2])
    timing = _timing([1695., 1800., 2300.], [50., -30., 20.])
    shifted = dict(st, t=st["t"] + np.array([-7, 12, 0, 100, -19]) * c.mbtime)
    a_rng, b_rng = np.random.default_rng(10), np.random.default_rng(10)
    a = DG.digitize(st, 3, c, a_rng, event_timing=timing)
    b = DG.digitize(shifted, 3, c, b_rng, event_timing=timing)
    assert len(a["ev"]) == 5 and np.array_equal(np.unique(a["ev"]), [0, 2])
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a_rng.bit_generator.state == b_rng.bit_generator.state
    assert np.array_equal(np.sort(a["step"][:, 0]), np.arange(5))


def test_event_specific_tdc_limits_and_readall(cond):
    c = _onspill(cond)
    st = _steps([_step(c, us, t=1580.) for us in (92, 520, 92, 520)], ev=[0, 0, 1, 1])
    timing = _timing([1505., 2200.])
    d = DG.digitize(st, 2, c, np.random.default_rng(4), event_timing=timing)
    np.testing.assert_array_equal(d["ev"], [1])
    np.testing.assert_array_equal(d["sid"], [92])
    np.testing.assert_array_equal(d["step"], [[2, 2]])
    c.s["onspill"] = False
    d = DG.digitize(st, 2, c, np.random.default_rng(4), event_timing=timing)
    np.testing.assert_array_equal(d["ev"], [1, 1])


@pytest.mark.parametrize("onspill,dend,expected", [(True, 1505., []), (True, 2200., [0, 1]),
                                                           (False, 1505., [1])])
def test_crossing_bound_is_not_onspill_event_length(cond, monkeypatch, onspill, dend, expected):
    c = Conditions(cond.s, cond.a)
    c.s.update(onspill=onspill, digitizationEnd=dend + c.timeFromProtonsToDRMarker)
    calls = 0

    def crosses(wf, state, rows):
        nonlocal calls
        state["T"][rows] = 1800.
        calls += 1
        return np.full(len(rows), calls == 1)

    monkeypatch.setattr(DG, "_crosses", crosses)
    ps, _, _, _ = DG.find_crossings(SimpleNamespace(c=c), np.random.default_rng(5),
        np.array([92, 520]), np.array([True, False]), event_lengths=np.array([1695., 4000.]))
    np.testing.assert_array_equal(ps, expected)


def test_ghosts_keep_per_end_charge_distance_and_provenance(cond, monkeypatch):
    c = _onspill(cond)
    captured = []
    original = DG.Waveforms

    def waveforms(c, row, times, charge, distance, us, steps):
        captured.append((row, times, charge, distance, us, steps))
        return original(c, row, times, charge, distance, us, steps)

    monkeypatch.setattr(DG, "Waveforms", waveforms)
    st = _steps([_step(c, 92, t=50., zpos=400.), _step(c, 520, t=1650., zpos=-400.)])
    DG.digitize(st, 1, c, np.random.default_rng(12), event_timing=_timing([2000.]))
    row, times, charge, distance, us, steps = captured[0]
    nprimary = np.count_nonzero((times >= 0) & (times < c.mbtime))
    idx, expected = ghost_indices(times[:nprimary], c.mbtime, 200.)
    assert len(idx) > nprimary
    np.testing.assert_array_equal(times, expected)
    for values in (row, charge, distance, us, steps):
        np.testing.assert_array_equal(values, values[:nprimary][idx])
