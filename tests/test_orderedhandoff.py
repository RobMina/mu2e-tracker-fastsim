"""Explicit world cutoff, no-backprojection and full-primary forward handoff contracts."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials, transport
from mu2efastsim.world import CoaxialShells
from trkfastsim import crossings, geometry, orderedhandoff as H, orderedinterior as O, steps


def geom():
    return geometry.StrawGeometry(plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=geometry.to_mu2e([[0., 0., 0.]]), dir=np.array([[1., 0., 0.]]), halflen=np.array([500.]), radius=2.5,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))


def record(top=850., **kw):
    return dict(scope="tracker", classification="approximate", geometry_bounds_mm=[[-10., -10., -10.], [10., top, 10.]],
                placement_assumptions=[]) | kw


def test_plan_above_actual_and_loaded_support_not_legacy_plane():
    g = geom()
    interior = SimpleNamespace(shells=CoaxialShells([], geometry.DETECTOR_ORIGIN_EXTRACTED))
    result = H.plan(g, interior, dict(schema="tracker-navigation-census-v1", records=[record()]), clearance_mm=20.)
    assert result["plane_y_mm"] > 870.
    assert result["approximate_records"] == 1 and not result["admitted_for_physics"]
    assert steps.plane_y(g) < result["plane_y_mm"]
    with pytest.raises(ValueError, match="unknown tracker/calo"):
        H.plan(g, interior, dict(schema="tracker-navigation-census-v1", records=[record(geometry_bounds_mm=None)]), clearance_mm=20.)


def test_world_endpoint_gate_requires_exact_matching_decreasing_edges():
    expected = np.r_[steps.BAND_EDGES_Y[steps.BAND_EDGES_Y > 870.], 870.]
    H.validate_world(SimpleNamespace(edges=expected), 870.)
    for w in (None, SimpleNamespace(edges=[1000., 869.]), SimpleNamespace(edges=[870., 1000.])):
        with pytest.raises(ValueError):
            H.validate_world(w, 870.)


def test_capture_retains_undelivered_upward_and_offplane_primary_identities():
    mu = pd.DataFrame(dict(run=[1]*4, subrun=[2]*4, event=[3, 3, 4, 5], mukey=[1, 2, 1, 1]))
    origin = np.array([[0., 870., 0.], [0., 870., 0.], [0., 869., 0.], [0., 1000., 0.]])
    direction = np.array([[0., -1., 0.], [0., 1., 0.], [0., -1., 0.], [0., -1., 0.]])
    result = H.capture(mu, origin, direction, np.ones(4)*80., np.ones(4)*130., np.array([1, 1, 1, 0]),
                       np.array([1., 1., 1., np.nan]), plane=870.)
    assert result.sp.tolist() == [1, 2, 1, 1]
    assert result.disposition.tolist() == ["forward-handoff", "refused-handoff", "refused-handoff", "not-delivered-by-world"]
    assert len(result) == 4


@pytest.mark.parametrize("mode", O.MODES)
def test_actual_producer_no_backprojection_and_no_handoff_time_subtraction(monkeypatch, mode):
    g = geom()
    interior = SimpleNamespace(shells=CoaxialShells([], geometry.DETECTOR_ORIGIN_EXTRACTED))
    def evolve(rng, origin, direction, p, geometry_, **kw):
        return O.transport(interior, rng, origin, direction, p, geometry_, tail=kw["tail"], tail_k=kw["tail_k"],
                           mode=kw["navigation"], **kw["ordered_options"])
    interior.transport = evolve
    generated = pd.DataFrame(dict(run=[1], subrun=[2], event=[3], pdg=[13], x=[0.], y=[1000.], z=[0.],
                                  px=[0.], py=[-80.], pz=[0.], t=[7.]))
    initial = geometry.to_mu2e([[0., 870., 0.]])
    def plane(gen, geom, rng, world, **kw):
        assert kw["endpoint_y_mm"] == 870.
        plane.last_history = dict(t_world=np.array([10.]))
        return steps._muons(gen, True), initial.copy(), np.array([[0., -1., 0.]]), np.array([80.]), np.array([130.]), np.array([True])
    monkeypatch.setattr(steps, "to_plane", plane)
    def fail(*a):
        raise AssertionError("legacy backprojection invoked")
    monkeypatch.setattr(steps, "_back_to_interior_top", fail)
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut, *a: np.zeros_like(p))
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut: np.zeros_like(p))
    rows, particles = steps.make_steps(generated, g, np.random.default_rng(19), stage="C", interior=interior,
        world=SimpleNamespace(edges=np.r_[steps.BAND_EDGES_Y[steps.BAND_EDGES_Y > 870.], 870.]), history=True, navigation=mode,
        handoff=H.POLICY, handoff_plane_y_mm=870., ordered_options={"max_path_mm": 1000.}, msc_tail=False)
    assert len(rows) == 1 and len(particles) == 1
    np.testing.assert_allclose(interior.last_slices["pos0"][0], initial[0])
    assert steps.make_steps.muon_ledger.t_interior_start_ns.iloc[0] == 17.
    assert steps.make_steps.handoff_ledger.disposition.tolist() == ["forward-handoff"]
    assert (steps.make_steps.path_vertices.time_ns >= 17.).all()


def test_legacy_endpoint_argument_forbidden_without_explicit_policy():
    with pytest.raises(ValueError, match="requires forward"):
        steps.make_steps(None, None, None, handoff_plane_y_mm=870.)
    with pytest.raises(ValueError, match="requires ordered"):
        steps.make_steps(None, None, None, handoff=H.POLICY, handoff_plane_y_mm=870.)


@pytest.mark.parametrize("bounds", [[[0., 0., 0.], [1., np.nan, 1.]], [[0., 0., 0.], [-1., 1., 1.]], [0., 1.]])
def test_nonfinite_unordered_or_malformed_bounds_refuse_before_plane_reduction(bounds):
    g = geom()
    interior = SimpleNamespace(shells=CoaxialShells([], geometry.DETECTOR_ORIGIN_EXTRACTED))
    with pytest.raises(ValueError, match="bounds"):
        H.plan(g, interior, dict(schema="tracker-navigation-census-v1", records=[record(geometry_bounds_mm=bounds)]), clearance_mm=20.)


def test_infinite_delivered_offset_refused_and_shapes_validated():
    mu = pd.DataFrame(dict(run=[1], subrun=[2], event=[3], mukey=[1]))
    args = (mu, np.array([[0., 870., 0.]]), np.array([[0., -1., 0.]]), np.array([80.]))
    result = H.capture(*args, np.array([np.inf]), np.array([True]), np.array([1.]), plane=870.)
    assert result.disposition.tolist() == ["refused-handoff"]
    with pytest.raises(ValueError, match="one-dimensional"):
        H.capture(*args, np.array([[130.]]), np.array([True]), np.array([1.]), plane=870.)


def test_forward_domain_enforced_without_guarded_provider():
    g = geom()
    interior = SimpleNamespace(shells=CoaxialShells([], geometry.DETECTOR_ORIGIN_EXTRACTED))
    with pytest.raises(ValueError, match="crosses world handoff plane"):
        O.transport(interior, np.random.default_rng(19), np.array([[0., 870., 0.]]), np.array([[0., 1., 0.]]),
                    np.array([80.]), g, tail=False, tail_k=1., max_path_mm=1., forward_plane_y_mm=870.)
    assert interior.last_forward_refusal["interior_lanes"].tolist() == [0]
    assert interior.last_forward_refusal["iteration"] == 0
    np.testing.assert_array_equal(interior.last_forward_refusal["direction"], [[0., 1., 0.]])
