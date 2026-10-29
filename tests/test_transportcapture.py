"""Passive producer observations preserve transport; strict legacy physics failures remain."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import materials
from mu2efastsim.world import CoaxialShells
from trkfastsim import ledger, steps
from trkfastsim.transportcapture import PhaseCapture, TransportCapture
from test_rate_ledger_limits import _geometry


def interior():
    obj = steps.TrackerInterior.__new__(steps.TrackerInterior)
    obj.shells = CoaxialShells([], axis_point=[0., 0., 0.])
    obj.nsub = 8
    return obj


def phase(n):
    return PhaseCapture(np.column_stack((np.ones((n, 3), int), np.arange(n)+1)))


def run_direct(momentum=(80.,), *, xs=(0.,), ys=(0., -10., -20.), capture=True, max_iter=400,
               thin=True, kink=np.inf, obj=None):
    geom, obj = _geometry(ys), interior() if obj is None else obj
    p = np.array(momentum)
    o = np.column_stack((np.array(xs), np.full(len(p), 1002.5), np.zeros(len(p))))
    d = np.tile([0., -1., 0.], (len(p), 1))
    rng = np.random.default_rng(19)
    obs = phase(len(p)) if capture else None
    result = obj.transport(rng, o, d, p, geom, tail=False, kink_xx0=kink,
                           kink_sup_xx0=None, thin_loss=thin, max_iter=max_iter, capture=obs)
    return SimpleNamespace(rows=result, obs=obs, final=obj.last_final, kinks=obj.last_kinks,
                           stats=obj.last_stats, rng=rng, obj=obj)


def test_final_non_kink_is_last_entry_not_stale_restart():
    result = run_direct()
    summary = result.obs.summary().iloc[0]
    assert summary.disposition == "final_non_kink" and summary.last_kind == "entry"
    assert summary.last_path_mm == 1020.
    assert summary.last_lost_thin_MeV == result.rows["lost_thin"][-1] > 0.
    assert result.final["path"][0] == result.final["lost_thin"][0] == 0.
    assert summary.last_model_ke_MeV == ledger.kinetic(80.) - result.rows["lost"][-1]
    assert result.obs.complete


def test_stop_attempt_is_unresolved_not_a_fake_endpoint():
    result = run_direct(momentum=(12.,))
    summary = result.obs.summary().iloc[0]
    assert summary.disposition == "unresolved_stop" and summary.last_kind == "entry"
    assert summary.last_path_mm == 1000.
    name, attempt = result.obs.records("stop_attempt")[0]
    np.testing.assert_array_equal(attempt["lower_local_path_mm"], [1000.])
    np.testing.assert_array_equal(attempt["attempted_local_path_mm"], [1010.])
    assert attempt["model_ke_after_MeV"][0] <= .5
    assert not result.obs.records("restart")
    assert result.final["stopped"][0]


@pytest.mark.parametrize("max_iter,expected", [(0, "iteration_cap"), (400, "no_candidates")])
def test_missing_and_capped_lanes(max_iter, expected):
    result = run_direct(xs=(10000.,), max_iter=max_iter)
    summary = result.obs.summary().iloc[0]
    assert summary.disposition == expected and summary.last_kind == "initial"
    assert summary.last_path_mm == 0.


def test_empty_direct_transport_capture():
    result = run_direct(momentum=(), xs=())
    assert result.obs.summary().empty and result.obs.complete


def test_mixed_lanes_all_receive_dispositions():
    result = run_direct(momentum=(80., 12., 80.), xs=(0., 0., 10000.))
    summary = result.obs.summary()
    assert summary.disposition.tolist() == ["final_non_kink", "unresolved_stop", "no_candidates"]
    assert summary.sp.tolist() == [1, 2, 3]


@pytest.mark.parametrize("momentum,thin,kink,max_iter", [
    (80., False, np.inf, 400), (80., True, np.inf, 400), (12., True, np.inf, 400),
    (80., True, 1e-5, 400), (80., True, 1e-5, 1), (80., True, 1e-5, 0),
])
def test_direct_capture_on_off_exact(momentum, thin, kink, max_iter):
    args = dict(momentum=(momentum, 80.), xs=(0., 10000.), thin=thin, kink=kink, max_iter=max_iter)
    off, on = run_direct(capture=False, **args), run_direct(**args)
    for a, b in ((off.rows, on.rows), (off.final, on.final), (off.kinks, on.kinks)):
        assert set(a) == set(b)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
    assert off.stats == on.stats
    assert off.rng.bit_generator.state == on.rng.bit_generator.state
    assert off.rng.bit_generator.seed_seq.state == on.rng.bit_generator.seed_seq.state
    np.testing.assert_array_equal(off.rng.random(10), on.rng.random(10))


@pytest.fixture
def make_run(monkeypatch):
    def run(capture, momentum=80., arms="AM", exit_scatter=False, history=True):
        geom, obj = _geometry((0., -10., -20.)), interior()
        generated = pd.DataFrame(dict(run=[1, 1], subrun=[1, 1], event=[1, 1], pdg=[13, -13],
            x=[0., 0.], y=[1502.5, 1502.5], z=[0., 0.], px=[0., 0.], py=[-momentum, -80.],
            pz=[0., 0.], t=[7., 9.]))
        def to_plane(gen, geom, rng, world, **kwargs):
            return (steps._muons(gen, kwargs["all_muons"]), np.array([[0., 1002.5, 0.]]*2),
                    np.array([[0., -1., 0.]]*2), np.array([momentum, 80.]),
                    np.array([500., 500.]), np.ones(2, bool))
        to_plane.last_history = None
        monkeypatch.setattr(steps, "to_plane", to_plane)
        rng = np.random.default_rng(19)
        rows, particles = steps.make_steps(generated, geom, rng, stage="C", interior=obj, arms=arms,
            history=history, msc_tail=False, exit_scatter=exit_scatter, kink_xx0=np.inf, kink_sup_xx0=None,
            capture_transport=capture)
        return SimpleNamespace(rows=rows, particles=particles, obs=steps.make_steps.transport_capture,
            vertices=steps.make_steps.path_vertices.copy(), seg=steps.make_steps.segment_ledger,
            mu=steps.make_steps.muon_ledger, rng=rng)
    return run


@pytest.mark.parametrize("momentum,arms,exit_scatter,history", [
    (80., "legacy", False, False), (80., "AM", False, True), (12., "AMT", True, True),
    (80., "T", True, True),
])
def test_make_steps_capture_exact_and_legacy_exit_restart(make_run, momentum, arms, exit_scatter, history):
    off = make_run(False, momentum, arms, exit_scatter, history)
    on = make_run(True, momentum, arms, exit_scatter, history)
    for name in ("rows", "particles", "vertices", "seg", "mu"):
        a, b = getattr(off, name), getattr(on, name)
        if a is not None:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        else:
            assert b is None
    assert off.obs is None and on.obs.complete
    assert off.rng.bit_generator.state == on.rng.bit_generator.state
    assert off.rng.bit_generator.seed_seq.state == on.rng.bit_generator.seed_seq.state
    np.testing.assert_array_equal(off.rng.random(10), on.rng.random(10))
    assert on.obs.get_phase("interior").summary().sp.tolist() == [1, 2]
    if exit_scatter:
        exit_phase = on.obs.get_phase("exit")
        assert exit_phase.summary().disposition.tolist() == ["not_processed", "not_processed"]
        assert (exit_phase.summary().last_path_mm == 0).all()
        if momentum == 12.:
            assert exit_phase.records("interior_status")[0][1]["legacy_stopped"].tolist() == [True, False]


def test_capture_serialization_no_overwrite_or_pickle(make_run, tmp_path):
    captured = make_run(True).obs
    target = tmp_path / "capture.npz"
    captured.save(target)
    with np.load(target, allow_pickle=False) as data:
        meta = json.loads(str(data["manifest_json"]))
        assert meta["schema"] == "trkfastsim.transport-observations.v1"
        for key in data.files:
            assert data[key].dtype.kind != "O"
        assert data["interior_summary_disposition"].tolist() == ["final_non_kink"]*2
    with pytest.raises(FileExistsError):
        captured.save(target)


@pytest.mark.parametrize("option", [dict(stage="A"), dict(backend="jax"), dict(interior_msc=False),
    dict(sampler=object()), dict(offaxis=object()), dict(entry_ledger="vacuum"), dict(arms="E"),
    dict(capture_transport="yes")])
def test_invalid_capture_options_before_rng(option):
    rng = np.random.default_rng(2)
    before = copy.deepcopy(rng.bit_generator.state)
    kwargs = dict(stage="C", interior=interior(), capture_transport=True, exit_scatter=False)
    kwargs.update(option)
    with pytest.raises(ValueError):
        steps.make_steps(pd.DataFrame(), _geometry((0.,)), rng, **kwargs)
    assert rng.bit_generator.state == before
    assert steps.make_steps.transport_capture is None


def test_unfinished_capture_refuses_publication():
    capture = TransportCapture({})
    with pytest.raises(ValueError, match="incomplete"):
        capture.save("unused.npz")


def test_copies_are_detached_and_phase_classification_is_total():
    result = run_direct()
    values = result.obs.records("entry")[0][1]
    values["path_mm"][:] = -1.
    assert (result.obs.records("entry")[0][1]["path_mm"] > 0).all()
    summary = result.obs.summary()
    summary.loc[:, "last_path_mm"] = -1.
    assert result.obs.summary().last_path_mm.iloc[0] == 1020.
    with pytest.raises(ValueError):
        result.obs.terminate([0], "no_candidates")
    fresh = phase(1)
    with pytest.raises(ValueError, match="unclassified"):
        fresh.finish()


@pytest.mark.parametrize("dtype", ["uint64", "Int64", "UInt64", "int64"])
def test_bind_preserves_large_mixed_integer_keys(dtype):
    mu = pd.DataFrame(dict(run=[1], subrun=[2], event=pd.Series([2**53+1], dtype=dtype), mukey=[1]))
    capture = TransportCapture({})
    capture.bind(mu)
    assert capture.handoff["identity"][0, 2] == 2**53+1
    assert capture.handoff["identity"].dtype == np.int64


@pytest.mark.parametrize("value", [pd.Series([None], dtype="Int64"), pd.Series([2**63], dtype="uint64"),
                                   pd.Series([1.]), pd.Series([-1])])
def test_bind_rejects_noninteger_missing_or_overflow_key(value):
    mu = pd.DataFrame(dict(run=[1], subrun=[2], event=value, mukey=[1]))
    with pytest.raises(ValueError, match="identities"):
        TransportCapture({}).bind(mu)


def test_nonzero_phase_offset_keeps_attempt_coordinates_explicit():
    obj, geom = interior(), _geometry((0., -10., -20.))
    captured = PhaseCapture(np.array([[1, 1, 1, 1]]), path_offset=[500.])
    obj.transport(np.random.default_rng(19), np.array([[0., 1002.5, 0.]]), np.array([[0., -1., 0.]]),
                  np.array([12.]), geom, tail=False, thin_loss=True, kink_xx0=np.inf,
                  kink_sup_xx0=None, capture=captured)
    assert captured.summary().last_path_mm.iloc[0] == 1500.
    attempt = captured.records("stop_attempt")[0][1]
    assert attempt["lower_local_path_mm"][0] == 1000.
    assert attempt["attempted_local_path_mm"][0] == 1010.
    assert "attempted_path_mm" not in attempt


def test_capture_provenance_contains_identity_and_box_crossing_dependencies():
    from trkfastsim.transportcapture import source_hashes
    capture = TransportCapture({})
    assert {"mu2efastsim.tables", "mu2efastsim.crossings", "trkfastsim.surfaces"} <= capture.sources.keys()
    assert capture.sources == source_hashes()
    assert capture.environment["numpy"] == np.__version__
