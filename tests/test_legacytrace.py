"""Capture-on/off equivalence through the real legacy producer, including lost lanes."""
import json

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import materials
from trkfastsim import legacytrace, steps
from test_step_energy_ledger import generated, empty_interior, plane_state, _geometry
from test_transportledger import meta


class Slabs:
    """Two fixed y slabs supply real sequential columns but no boundary-history interface."""

    groups = []
    c = np.zeros(3)

    def column(self, o, d, t_max, nsub=8, moments=False):
        result = {}
        for name, y0, y1 in (("G4_Al", 20., 22.), ("G4_Cu", 10., 11.)):
            lo = (y1 - o[:, 1]) / d[:, 1]
            hi = (y0 - o[:, 1]) / d[:, 1]
            lo = np.maximum(lo, 0.)
            hi = np.minimum(hi, np.asarray(t_max))
            length = np.maximum(hi - lo, 0.)
            result[name] = (length, np.where(length > 0, (hi ** 2 - lo ** 2) / 2, 0.),
                            np.where(length > 0, (hi ** 3 - lo ** 3) / 3, 0.)) if moments else length
        return result


@pytest.mark.parametrize("interior_msc", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_make_steps_capture_is_bitwise_inert(generated, empty_interior, meta, interior_msc, tail):
    empty_interior.shells = Slabs()
    gen = pd.concat([generated.assign(py=-300.), generated.assign(py=-80.)], ignore_index=True)
    geom = _geometry((0., -10.))
    a, b = np.random.default_rng(29), np.random.default_rng(29)
    args = dict(interior=empty_interior, interior_msc=interior_msc, exit_scatter=False, msc_tail=tail)
    before, parts = steps.make_steps(gen, geom, a, **args)
    vertices = steps.make_steps.path_vertices
    trace = legacytrace.LegacyTrace(meta)
    after, parts_after = steps.make_steps(gen, geom, b, transport_trace=trace, **args)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    pd.testing.assert_frame_equal(parts, parts_after, check_exact=True)
    if vertices is not None:
        pd.testing.assert_frame_equal(vertices, steps.make_steps.path_vertices, check_exact=True)
    np.testing.assert_array_equal(a.random(100), b.random(100))
    begins = [r for r in trace.records if r["kind"] == "slab_begin"]
    assert begins
    keys = np.concatenate([r["identity"] for r in begins])
    assert set(keys[:, 3]) == {1, 2}
    assert {r["material"]["material"]["name"] for r in begins} == {"G4_Al", "G4_Cu"}
    assert any(r["kind"] == "slab_core" for r in trace.records)
    gas = [r for r in trace.records if r["kind"] == "gas_sample"]
    assert len(gas) == 1 and gas[0]["transport_debit_status"] == "unmodeled"


def test_stopped_lane_is_recorded_before_crossing_filter(generated, empty_interior, meta, monkeypatch):
    empty_interior.shells = Slabs()
    original = steps.tr.slab_loss

    def exaggerated(rng, mat, p, length, **kwargs):
        return original(rng, mat, p, length * 1000, **kwargs)

    monkeypatch.setattr(steps.tr, "slab_loss", exaggerated)
    trace = legacytrace.LegacyTrace(meta)
    out, _ = steps.make_steps(generated, _geometry(), np.random.default_rng(2), interior=empty_interior,
                              exit_scatter=False, transport_trace=trace)
    assert out.empty
    states = [r for r in trace.records if r["kind"] == "candidate_energy"]
    assert states and states[0]["first_stop"].any() and states[0]["stopped"].any()
    assert not states[0]["emitted"].any()


def test_exit_capture_equivalence_and_scope(meta, monkeypatch):
    monkeypatch.setattr(steps, "_envelope_exit", lambda o, d: np.full(len(o), 40.))
    interior = steps.TrackerInterior.__new__(steps.TrackerInterior)
    interior.shells, interior.nsub = Slabs(), 8
    mu = pd.DataFrame(dict(run=[1, 1], subrun=[2, 2], event=[3, 3], mukey=[1, 2]))
    trace = legacytrace.LegacyTrace(meta)
    scope = trace.bind(mu).subset(np.array([1]))
    o = np.array([[0., 30., 0.]])
    d = np.array([[0., -1., 0.]])
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    x = interior.exit_scatter(a, o, d, np.array([300.]), np.array([0.]), length=40., tail=True)
    y = interior.exit_scatter(b, o, d, np.array([300.]), np.array([0.]), length=40., tail=True, trace=scope)
    for key in x:
        np.testing.assert_array_equal(x[key], y[key])
    np.testing.assert_array_equal(a.random(20), b.random(20))
    assert trace.records
    for record in trace.records:
        if "identity" in record:
            assert (record["identity"][:, 3] == 2).all()


def test_handoff_and_npz(generated, empty_interior, plane_state, meta, tmp_path):
    plane_state(80., origin_y=steps.plane_y(_geometry()))
    trace = legacytrace.LegacyTrace(meta)
    steps.make_steps(generated, _geometry(), np.random.default_rng(1), stage="C", interior=empty_interior,
                     exit_scatter=False, transport_trace=trace)
    assert [r["kind"] for r in trace.records[:2]] == ["plane", "interior_top"]
    path = tmp_path / "trace.npz"
    trace.save(path)
    with np.load(path, allow_pickle=False) as saved:
        manifest = json.loads(str(saved["manifest_json"]))
        assert "not physical" in manifest["scope"]
        np.testing.assert_array_equal(saved[manifest["identities"]["array"]], [[1, 1, 1, 1]])
    with pytest.raises(FileExistsError):
        trace.save(path)
    with pytest.raises(ValueError, match="reused"):
        trace.bind(steps._muons(generated, True))


def test_empty_collector_and_empty_straight(meta):
    trace = legacytrace.LegacyTrace(meta)
    mu = pd.DataFrame({k: np.array([], dtype=np.int64) for k in ("run", "subrun", "event", "mukey")})
    scope = trace.bind(mu)
    interior = steps.TrackerInterior.__new__(steps.TrackerInterior)
    loss, stats = interior.loss_to_straws(np.random.default_rng(1), np.empty((0, 3)), np.empty((0, 3)),
                                         np.empty(0), np.empty(0, dtype=int), np.empty(0), trace=scope)
    assert loss.size == 0 and stats == {} and trace.records == []


@pytest.mark.parametrize("interior_msc", [False, True])
def test_trace_rejects_secondary_libraries_before_rng(generated, empty_interior, meta, interior_msc):
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="primary-only"):
        steps.make_steps(generated, _geometry(), rng, sampler=object(), transport_trace=legacytrace.LegacyTrace(meta),
                         interior=empty_interior, interior_msc=interior_msc)
    assert rng.bit_generator.state == state


@pytest.mark.parametrize("interior_msc", [False, True])
def test_frozen_published_steps_equivalence(generated, empty_interior, meta, interior_msc):
    import os
    import subprocess
    import types

    git_dir = os.environ.get("TRACKER_BASELINE_GIT_DIR")
    if not git_dir:
        pytest.skip("set TRACKER_BASELINE_GIT_DIR for published-source regression")
    source = subprocess.check_output(["git", f"--git-dir={git_dir}", "show",
                                      "56e66bfa44a3e5d58fa647ac2e38955c353d26ba:trkfastsim/steps.py"], text=True)
    baseline = types.ModuleType("trkfastsim._published_steps")
    baseline.__package__ = "trkfastsim"
    exec(compile(source, "published-56e66bf-steps.py", "exec"), baseline.__dict__)
    old = baseline.TrackerInterior.__new__(baseline.TrackerInterior)
    old.shells, old.nsub = Slabs(), 8
    empty_interior.shells = Slabs()
    gen = pd.concat([generated.assign(py=-300.), generated.assign(py=-80.)], ignore_index=True)
    a, b = np.random.default_rng(29), np.random.default_rng(29)
    args = dict(interior_msc=interior_msc, exit_scatter=False, msc_tail=True)
    before, parts = baseline.make_steps(gen, _geometry((0., -10.)), a, interior=old, **args)
    after, parts_after = steps.make_steps(gen, _geometry((0., -10.)), b, interior=empty_interior,
                                         transport_trace=legacytrace.LegacyTrace(meta), **args)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    pd.testing.assert_frame_equal(parts, parts_after, check_exact=True)
    if baseline.make_steps.path_vertices is not None:
        pd.testing.assert_frame_equal(baseline.make_steps.path_vertices, steps.make_steps.path_vertices, check_exact=True)
    np.testing.assert_array_equal(a.random(50), b.random(50))


def test_straight_writer_stop_decision(generated, empty_interior, meta, monkeypatch):
    empty_interior.shells = Slabs()
    monkeypatch.setattr(steps.eloss, "dedx_restricted", lambda mat, p, cut, *a: np.full_like(p, 100.))
    trace = legacytrace.LegacyTrace(meta)
    out, _ = steps.make_steps(generated, _geometry(), np.random.default_rng(2), interior=empty_interior,
                              interior_msc=False, exit_scatter=False, transport_trace=trace)
    record = next(r for r in trace.records if r["kind"] == "writer_candidate")
    assert record["stopped"].all() and not record["kept_by_legacy_filter"].any() and out.empty


def test_interior_iteration_cap_is_explicit(generated, empty_interior, meta):
    trace = legacytrace.LegacyTrace(meta)
    scope = trace.bind(steps._muons(generated, True))
    empty_interior.transport(np.random.default_rng(1), np.array([[0., 100., 0.]]), np.array([[0., -1., 0.]]),
                             np.array([80.]), _geometry(), max_iter=0, trace=scope)
    record = next(r for r in trace.records if r["kind"] == "termination")
    assert record["status"].tolist() == ["iteration_cap"] and record["stats"]["capped"] == 1


def test_real_plane_handoff_distinguishes_rejection_and_stop(generated, meta, monkeypatch):
    gen = pd.concat([generated.assign(x=1e6), generated], ignore_index=True)

    def world_loss(rng, o, d, p, world, edges, **kwargs):
        n, nb = len(p), len(edges) - 1
        z = np.zeros((n, nb))
        return np.zeros(n), np.zeros(n), np.zeros(n, bool), z, z, z

    monkeypatch.setattr(steps.tr, "world_loss", world_loss)
    trace = legacytrace.LegacyTrace(meta)
    steps.make_steps(gen, _geometry(), np.random.default_rng(1), stage="C", msc_tail=False,
                     exit_scatter=False, transport_trace=trace)
    record = trace.records[0]
    assert record["handoff_status"].tolist() == ["generated_rejected", "upstream_stopped_plane_proxy"]
    assert not record["alive"].any()


def test_exit_iteration_cap_is_explicit(generated, empty_interior, meta, monkeypatch):
    trace = legacytrace.LegacyTrace(meta)
    scope = trace.bind(steps._muons(generated, True))
    monkeypatch.setattr(steps, "_envelope_exit", lambda o, d: np.full(len(o), 100.))
    monkeypatch.setattr(empty_interior, "_support", lambda o, d, seg, moments: ({"G4_Al": np.ones(len(o))}, {}))
    monkeypatch.setattr(empty_interior, "_split_length", lambda o, d, seg, target: np.full(len(o), .01))
    monkeypatch.setattr(steps, "_kink_core", lambda rng, v, w1, w2: (np.zeros((len(v), 2)), np.zeros((len(v), 2)), np.zeros(len(v))))
    empty_interior.exit_scatter(np.random.default_rng(2), np.array([[0., 100., 0.]]), np.array([[0., -1., 0.]]),
                                np.array([10000.]), np.zeros(1), length=100., tail=False, trace=scope)
    record = next(r for r in trace.records if r["kind"] == "termination")
    assert record["status"].tolist() == ["iteration_cap"]
    np.testing.assert_allclose(record["nominal_remaining_mm"], [99.36])
