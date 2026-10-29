"""Synthetic producer edge observations are not fabricated physical terminal states."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import ledger, steps
from test_transportcapture import _geometry, interior, phase, run_direct


def _exact(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def _record(obs, kind):
    records = obs.records(kind)
    assert len(records) == 1
    return records[0][1]


def _legacy(obs):
    return {key: value for key, value in _record(obs, "legacy_restart").items()
            if key not in ("identity", "lane", "iteration")}


def _rng_exact(off, on):
    assert off.bit_generator.state == on.bit_generator.state
    assert off.bit_generator.seed_seq.state == on.bit_generator.seed_seq.state
    np.testing.assert_array_equal(off.random(10), on.random(10))


def _support_physics(monkeypatch, obj, *, top=-8., bottom=-10., rate=1., supported=None):
    def support(o, d, length, moments):
        lo = np.clip((top - o[:, 1]) / d[:, 1], 0., length)
        hi = np.clip((bottom - o[:, 1]) / d[:, 1], 0., length)
        chord = np.maximum(hi - lo, 0.)
        if supported is not None:
            chord = np.where(np.isin(o[:, 0], supported), chord, 0.)
        name = materials.ALUMINIUM_6061.name
        mom = {name: (chord, chord * (lo + hi) / 2.,
                      chord * (lo * lo + lo * hi + hi * hi) / 3.)} if moments else {}
        return {name: chord}, mom

    def loss(rng, mat, p, length):
        rng.random(len(length))
        return rate * np.asarray(length)

    def zero_deflection(rng, variance, w1, w2):
        rng.normal(size=(len(variance), 4))
        return np.zeros((len(variance), 2)), np.zeros((len(variance), 2)), np.zeros(len(variance))

    # Deterministic physics still consumes draws, making capture RNG checks non-vacuous.
    monkeypatch.setattr(obj, "_support", support)
    monkeypatch.setattr(steps.tr, "slab_loss", loss)
    monkeypatch.setattr(steps, "_kink_core", zero_deflection)


@pytest.fixture
def exit_run(monkeypatch):
    # Keep synthetic origins inside a mocked envelope without loading detector geometry.
    monkeypatch.setattr(steps, "_envelope_exit", lambda o, d: np.full(len(o), 1e9))

    def run(case, capture=True):
        obj = interior()
        config = dict(top=-8., bottom=-10., rate=1.)
        p, length = np.array([80.]), 10.
        if case == "not_processed":
            p = np.array([0., 1.])
        elif case == "short_extent":
            length = 1.
        elif case == "stop":
            p = np.array([12.])
        elif case == "mixed":
            p = np.array([80., 12., 80., 1.])
            config["supported"] = (0., 1.)
        elif case == "cap64":
            length = 1e6
            config.update(top=0., bottom=-length, rate=0.)
        elif case == "empty":
            p = np.zeros(0)
        elif case not in ("no_support", "survive"):
            raise AssertionError(case)
        if case != "no_support":
            _support_physics(monkeypatch, obj, **config)
        n = len(p)
        o = np.column_stack((np.arange(n, dtype=float), np.zeros(n), np.zeros(n)))
        d = np.tile([0., -1., 0.], (n, 1))
        xc = np.arange(n, dtype=float) / 100.
        initial = dict(o=o.copy(), d=d.copy(), p=p.copy(), xcum=xc.copy())
        obs, rng = phase(n) if capture else None, np.random.default_rng(37)
        rng_before = copy.deepcopy(rng.bit_generator.state)
        kinks = obj.exit_scatter(rng, o, d, p, xc, length=length, tail=False, capture=obs)
        _exact(initial, dict(o=o, d=d, p=p, xcum=xc))
        return SimpleNamespace(obs=obs, rng=rng, rng_before=rng_before, initial=initial,
                               kinks=kinks, length=length)
    return run


@pytest.mark.parametrize("case,status,iteration", [
    ("no_support", "no_support", 1), ("not_processed", "not_processed", 0),
    ("short_extent", "not_processed", 0),
])
def test_exit_unprocessed_and_no_support_keep_initial_state(exit_run, case, status, iteration):
    result = exit_run(case)
    summary = result.obs.summary()
    assert result.obs.complete and not len(result.kinks["it"])
    assert summary.disposition.tolist() == [status] * len(summary)
    assert summary.disposition_iteration.tolist() == [iteration] * len(summary)
    assert summary.last_kind.tolist() == ["initial"] * len(summary)
    np.testing.assert_array_equal(summary.last_path_mm, np.zeros(len(summary)))
    np.testing.assert_array_equal(summary.last_model_ke_MeV, ledger.kinetic(result.initial["p"]))
    _exact(_legacy(result.obs), dict(result.initial, path=np.zeros(len(summary))))
    assert not result.obs.records("piece_attempt") and not result.obs.records("restart")
    assert result.rng.bit_generator.state == result.rng_before


def test_exit_surviving_kink_is_extent_complete(exit_run):
    result = exit_run("survive")
    summary, kink = result.obs.summary().iloc[0], result.kinks
    assert result.obs.complete and summary.disposition == "extent_complete"
    assert summary.last_kind == "restart" and summary.last_path_mm == 10.
    assert summary.last_lost_support_MeV == 2. and summary.last_lost_thin_MeV == 0.
    assert summary.last_model_ke_MeV == ledger.kinetic(80.) - 2.
    attempt, restart = _record(result.obs, "piece_attempt"), _record(result.obs, "restart")
    np.testing.assert_array_equal(attempt["accepted"], [True])
    np.testing.assert_array_equal(attempt["lower_local_path_mm"], [0.])
    np.testing.assert_array_equal(attempt["attempted_local_path_mm"], [10.])
    np.testing.assert_array_equal(kink["end"], [[0., -10., 0.]])
    np.testing.assert_array_equal(restart["position_mm"], kink["pos"])
    np.testing.assert_array_equal(restart["direction"], kink["dir"])
    np.testing.assert_array_equal(restart["path_mm"], kink["path"])
    np.testing.assert_array_equal(restart["momentum_MeV_c"], kink["p"])
    legacy = _legacy(result.obs)
    for old, new in (("o", "pos"), ("d", "dir"), ("p", "p"), ("path", "path")):
        np.testing.assert_array_equal(legacy[old], kink[new])
    assert legacy["xcum"][0] > 0.
    kink["pos"][:] = 123.
    np.testing.assert_array_equal(_record(result.obs, "restart")["position_mm"], [[0., -10., 0.]])


def test_exit_stop_attempt_does_not_emit_kink_or_promote_legacy_endpoint(exit_run):
    result = exit_run("stop")
    summary = result.obs.summary().iloc[0]
    assert result.obs.complete and summary.disposition == "unresolved_stop"
    assert summary.last_kind == "initial" and summary.last_path_mm == 0.
    assert summary.last_model_ke_MeV == ledger.kinetic(12.)
    assert summary.last_lost_support_MeV == 0.
    assert not len(result.kinks["it"])
    attempt = _record(result.obs, "piece_attempt")
    np.testing.assert_array_equal(attempt["accepted"], [False])
    np.testing.assert_array_equal(attempt["attempted_local_path_mm"], [10.])
    np.testing.assert_array_equal(attempt["attempted_position_mm"], [[0., -10., 0.]])
    np.testing.assert_array_equal(attempt["sampled_support_MeV"], [2.])
    np.testing.assert_array_equal(attempt["model_ke_after_MeV"], [ledger.kinetic(12.) - 2.])
    assert attempt["model_ke_after_MeV"][0] < 0.
    assert all(not len(record["lane"]) for _, record in result.obs.records("kink_end"))
    assert all(not len(record["lane"]) for _, record in result.obs.records("restart"))
    legacy = _legacy(result.obs)
    np.testing.assert_array_equal(legacy["path"], [10.])
    np.testing.assert_array_equal(legacy["o"], attempt["attempted_position_mm"])
    # The legacy momentum floor is a restart artifact, not a surviving endpoint.
    np.testing.assert_array_equal(legacy["p"], [np.sqrt((eloss.MMU + .5) ** 2 - eloss.MMU ** 2)])


def test_exit_mixed_lanes_classified_without_cross_lane_state(exit_run):
    result = exit_run("mixed")
    summary = result.obs.summary()
    assert result.obs.complete
    assert summary.disposition.tolist() == ["extent_complete", "unresolved_stop", "no_support", "not_processed"]
    assert summary.last_kind.tolist() == ["restart", "initial", "initial", "initial"]
    np.testing.assert_array_equal(summary.last_path_mm, [10., 0., 0., 0.])
    np.testing.assert_array_equal(summary.last_lost_support_MeV, [2., 0., 0., 0.])
    np.testing.assert_array_equal(result.kinks["it"], [0])
    attempt = _record(result.obs, "piece_attempt")
    np.testing.assert_array_equal(attempt["lane"], [0, 1])
    np.testing.assert_array_equal(attempt["identity"][:, -1], [1, 2])
    np.testing.assert_array_equal(attempt["accepted"], [True, False])
    np.testing.assert_array_equal(_legacy(result.obs)["path"], [10., 10., 0., 0.])


def test_exit_64_piece_cap_retains_last_observed_restart(exit_run):
    result = exit_run("cap64")
    summary = result.obs.summary().iloc[0]
    assert result.obs.complete and summary.disposition == "iteration_cap"
    assert summary.disposition_iteration == summary.last_iteration == 64
    assert summary.last_kind == "restart" and len(result.kinks["it"]) == 64
    assert summary.last_path_mm == result.kinks["path"][-1] < result.length - 1.
    assert np.all(np.diff(result.kinks["path"]) > 0.)
    assert summary.last_lost_support_MeV == 0.
    attempts = result.obs.records("piece_attempt")
    np.testing.assert_array_equal(np.concatenate([q["iteration"] for _, q in attempts]), np.arange(1, 65))
    np.testing.assert_array_equal(np.concatenate([q["attempted_local_path_mm"] for _, q in attempts]),
                                  result.kinks["path"])
    legacy = _legacy(result.obs)
    for old, new in (("o", "pos"), ("d", "dir"), ("p", "p"), ("path", "path")):
        np.testing.assert_array_equal(legacy[old], result.kinks[new][-1:])


def test_exit_empty_capture_finishes(exit_run):
    result = exit_run("empty")
    assert result.obs.complete and result.obs.summary().empty
    assert not len(result.kinks["it"])
    _exact(_legacy(result.obs), dict(result.initial, path=np.zeros(0)))


@pytest.mark.parametrize("case", ["no_support", "not_processed", "short_extent", "survive", "stop", "mixed", "cap64", "empty"])
def test_exit_capture_on_off_exact_and_rng_identical(exit_run, case):
    off, on = exit_run(case, capture=False), exit_run(case)
    _exact(off.kinks, on.kinks)
    if case in ("survive", "stop", "mixed", "cap64"):
        assert on.rng_before != on.rng.bit_generator.state
    _rng_exact(off.rng, on.rng)


def test_transport_mixed_lane_loses_candidates_after_restart(monkeypatch):
    crossings = steps.crossings.straw_crossings

    def lose_first_lane(o, d, geom):
        rows = crossings(o, d, geom)
        lane = rows[:, 0].astype(int)
        return rows[~((o[lane, 0] == 0.) & (o[lane, 1] < 1002.5))]

    monkeypatch.setattr(steps.crossings, "straw_crossings", lose_first_lane)

    def run(capture):
        obj = interior()
        _support_physics(monkeypatch, obj, top=-2.75, bottom=-3.75)
        return run_direct(momentum=(80., 80.), xs=(0., 1.), thin=False, obj=obj, capture=capture)

    off, on = run(False), run(True)
    for key in ("rows", "final", "kinks"):
        _exact(getattr(off, key), getattr(on, key))
    assert off.stats == on.stats
    _rng_exact(off.rng, on.rng)
    _exact(_legacy(on.obs), on.final)
    summary = on.obs.summary()
    assert on.obs.complete
    assert summary.disposition.tolist() == ["no_candidates", "final_non_kink"]
    assert summary.disposition_iteration.tolist() == [2, 2]
    assert summary.last_kind.tolist() == ["restart", "entry"]
    np.testing.assert_array_equal(summary.last_path_mm, [1009.95, 1020.])
    np.testing.assert_array_equal(on.kinks["it"], [0, 1])
    np.testing.assert_array_equal(on.rows["it"], [0, 1, 1, 1])
    np.testing.assert_array_equal(summary.last_lost_support_MeV, [1., 1.])
    assert not on.obs.records("stop_attempt")


@pytest.mark.parametrize("split", [False, True])
def test_transport_stop_at_proposed_support_kink_is_only_attempt(monkeypatch, split):
    def run(capture):
        obj = interior()
        _support_physics(monkeypatch, obj, top=-3., bottom=-7., rate=10.)
        rng, obs = np.random.default_rng(19), phase(1) if capture else None
        rows = obj.transport(rng, np.array([[0., 1002.5, 0.]]), np.array([[0., -1., 0.]]),
                             np.array([12.]), _geometry((0., -10., -20.)), tail=False,
                             kink_xx0=np.inf, kink_sup_xx0=.01 if split else None,
                             thin_loss=False, capture=obs)
        return SimpleNamespace(rows=rows, obs=obs, final=obj.last_final, kinks=obj.last_kinks,
                               stats=obj.last_stats, rng=rng)

    off, on = run(False), run(True)
    for key in ("rows", "final", "kinks"):
        _exact(getattr(off, key), getattr(on, key))
    assert off.stats == on.stats
    _rng_exact(off.rng, on.rng)
    _exact(_legacy(on.obs), on.final)
    summary = on.obs.summary().iloc[0]
    assert on.obs.complete and summary.disposition == "unresolved_stop"
    assert summary.last_kind == "entry" and summary.last_path_mm == 1000.
    assert summary.last_model_ke_MeV == ledger.kinetic(12.)
    assert summary.last_lost_support_MeV == summary.last_lost_thin_MeV == 0.
    np.testing.assert_array_equal(on.rows["path"], [1000.])
    assert not len(on.kinks["it"]) and not on.obs.records("restart")
    assert not on.obs.records("kink_end")
    attempt = _record(on.obs, "stop_attempt")
    np.testing.assert_array_equal(attempt["lower_local_path_mm"], [1000.])
    np.testing.assert_array_equal(attempt["candidate_straw"], [1])
    np.testing.assert_array_equal(attempt["committed_candidate"], [True])
    if split:
        np.testing.assert_array_equal(attempt["attempted_local_path_mm"], [1006.5625])
        np.testing.assert_array_equal(attempt["attempted_support_MeV"], [10.625])
        assert on.stats["support_splits"] > 0
    else:
        np.testing.assert_array_equal(attempt["attempted_local_path_mm"], [1010.])
        np.testing.assert_array_equal(attempt["attempted_support_MeV"], [40.])
    np.testing.assert_array_equal(attempt["attempted_position_mm"][:, 1],
                                  1002.5 - attempt["attempted_local_path_mm"])
    assert attempt["model_ke_after_MeV"][0] <= .5
    assert attempt["attempted_support_MeV"][0] > ledger.kinetic(12.)
    np.testing.assert_array_equal(on.final["stopped"], [True])
    np.testing.assert_array_equal(on.final["path"], [0.])
    np.testing.assert_array_equal(on.final["lost_support"], [0.])
