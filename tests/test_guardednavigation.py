"""Strict refinement, conditional ownership and actual loss/kick/re-navigation contracts."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import geometry, guardednavigation as G, ledger, orderedinterior as O, steps


def intervals(start=(1.,), end=(2.,), material=("steel",)):
    return pd.DataFrame(dict(lane=np.zeros(len(start), np.int64), s0_mm=start, s1_mm=end, material=material))


def test_refinement_preserves_straw_and_rejects_even_same_material_overlap():
    support = intervals((1., 2.), (2., 3.), ("steel", "steel"))
    thin = intervals((4.,), (5.,), ("gas",))
    result = G.refine(support, thin, np.array([6.]))
    assert result.material.tolist() == ["vacuum", "steel", "vacuum", "gas", "vacuum"]
    assert result.kind.tolist() == ["vacuum", "support", "vacuum", "thin", "vacuum"]
    assert result.loc[result.material == "steel", "s1_mm"].item() == 3.
    with pytest.raises(ValueError, match="overlapping"):
        G.refine(support, intervals(), np.array([6.]))


def make_provider(monkeypatch, *, y=0., admitted=True, arm="inner"):
    cells = intervals().assign(state="inside")
    cells["s0_mm"], cells["s1_mm"] = 0., 1.
    result = dict(summary={"all_pairs_admitted": admitted}, inner_intervals=intervals(), outer_intervals=intervals())
    modules = dict(omittedpath=SimpleNamespace(probe=lambda *a, **k: (cells, [])),
                   supportenvelope=SimpleNamespace(compose=lambda *a, **k: result))
    monkeypatch.setattr(G.strawpath, "straw_intervals", lambda *a: intervals((), (), ()))
    return G.GuardedNavigator(arm=arm, gdml="fixture", census=dict(schema="tracker-navigation-census-v1", records=[]),
        modules=modules, raw_hits=lambda *a: intervals().assign(sign=1, source="fixture"), world_plane_y_mm=y,
        endpoint_budget_mm=1e-10, min_cell_mm=1e-7, max_component_mm=1e-6, max_ray_unknown_mm=1e-5)


def test_provider_preserves_pair_and_refuses_boundary_contact(monkeypatch):
    provider = make_provider(monkeypatch, y=1.)
    result = provider(None, None, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([6.]), endpoint_budget_mm=1e-10)
    assert "steel" in result.material.to_numpy()
    assert provider.calls[0]["replacements_strictly_below_plane"]
    provider = make_provider(monkeypatch, y=0.)
    with pytest.raises(ValueError, match="world endpoint plane"):
        provider(None, None, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([6.]), endpoint_budget_mm=1e-10)
    assert len(provider.calls) == 1


def test_provider_refusal_never_evolves_partial_success_or_mismatched_budget(monkeypatch):
    provider = make_provider(monkeypatch, y=1., admitted=False)
    args = (None, None, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([6.]))
    with pytest.raises(ValueError, match="pair refused"):
        provider(*args, endpoint_budget_mm=1e-10)
    with pytest.raises(ValueError, match="budgets disagree"):
        provider(*args, endpoint_budget_mm=1e-9)


@pytest.mark.parametrize("arm", ["nominal", "", None])
def test_no_nominal_boundary_arm(monkeypatch, arm):
    with pytest.raises(ValueError, match="inner or outer"):
        make_provider(monkeypatch, arm=arm)


@pytest.mark.parametrize("mode", O.MODES)
def test_actual_evolution_debits_before_gas_and_renavigates_after_kick(monkeypatch, mode):
    geom = geometry.StrawGeometry(plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=np.zeros((1, 3)), dir=np.array([[0., 0., 1.]]), halflen=np.array([100.]), radius=2.5,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))
    calls = []

    def navigate(shells, geom, origin, direction, length, **kw):
        calls.append((origin.copy(), direction.copy()))
        gas = len(calls) > 1
        return pd.DataFrame(dict(lane=[0], s0_mm=[0.], s1_mm=[float(length[0])],
            material=[steps.STRAW_GAS.name if gas else materials.ALUMINIUM_6061.name],
            kind=["thin" if gas else "support"]))
    navigate.policy = G.POLICY + ":synthetic"
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full_like(p, .01))
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut: length * .01)
    monkeypatch.setattr(steps, "_kink_core", lambda rng, v, a, b:
                        (np.tile([.001, 0.], (len(v), 1)), np.zeros((len(v), 2)), np.zeros(len(v))))
    monkeypatch.setattr(steps.crossings, "point_straw", lambda *a: np.array([0]))
    obj = SimpleNamespace(shells=None)
    O.transport(obj, np.random.default_rng(19), np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([80.]), geom,
                tail=False, tail_k=1., mode=mode, max_path_mm=2., max_step_mm=1., navigation_provider=navigate)
    assert len(calls) == 2
    assert not np.array_equal(calls[0][1], calls[1][1])
    np.testing.assert_allclose(calls[1][0], [[1., 0., 0.]])
    slices = obj.last_slices
    assert slices["kind"].tolist() == ["support", "thin"]
    np.testing.assert_allclose(slices["ke0"][1], ledger.kinetic(80.) - .01)
    assert slices["t0"][1] > 0
    np.testing.assert_allclose(obj.last_final["ke"], ledger.kinetic(80.) - .02)
    np.testing.assert_allclose(obj.last_final["lost_support"], [.01])
    assert obj.last_stats["navigation"] == navigate.policy


def test_invalid_provider_refuses_before_transport():
    with pytest.raises(ValueError, match="named callable"):
        O.transport(None, None, None, None, None, None, tail=False, tail_k=1., navigation_provider=lambda: None)


@pytest.mark.parametrize("fault", ["gap", "overlap", "missing_lane", "bad_kind", "vacuum_kind", "fractional_lane", "zero_width"])
def test_provider_partition_validation_fail_closed(fault):
    partition = pd.DataFrame(dict(lane=[0, 0], s0_mm=[0., 1.], s1_mm=[1., 2.],
                                  material=["vacuum", "steel"], kind=["vacuum", "support"]))
    if fault == "gap":
        partition.loc[1, "s0_mm"] = 1.1
    elif fault == "overlap":
        partition.loc[1, "s0_mm"] = .9
    elif fault == "missing_lane":
        partition = partition.iloc[:0]
    elif fault == "bad_kind":
        partition.loc[1, "kind"] = "unsupported"
    elif fault == "vacuum_kind":
        partition.loc[0, "kind"] = "support"
    elif fault == "fractional_lane":
        partition["lane"] = .5
    else:
        partition.loc[0, "s1_mm"] = 0.
    with pytest.raises(ValueError):
        O.validate_provider_partition(partition, np.array([2.]))


def test_complete_provider_partition_accepts_unsorted_lanes_and_empty():
    partition = pd.DataFrame(dict(lane=[1, 0], s0_mm=[0., 0.], s1_mm=[2., 1.],
                                  material=["steel", "vacuum"], kind=["support", "vacuum"]))
    O.validate_provider_partition(partition, np.array([1., 2.]))
    O.validate_provider_partition(partition.iloc[:0], np.array([]))


def test_outer_only_straw_overlap_refuses_both_arms(monkeypatch):
    provider = make_provider(monkeypatch, y=1.)
    provider.modules["supportenvelope"].compose = lambda *a, **k: dict(summary={"all_pairs_admitted": True},
        inner_intervals=intervals((), (), ()), outer_intervals=intervals())
    monkeypatch.setattr(G.strawpath, "straw_intervals", lambda *a: intervals(material=("steel",)))
    with pytest.raises(ValueError, match="overlapping"):
        provider(None, None, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([6.]), endpoint_budget_mm=1e-10)
    assert provider.calls[-1]["status"] == "refused"
    assert provider.calls[-1]["stage"] == "paired-straw-refinement"


def test_endpoint_callback_failure_retains_first_call_evidence(monkeypatch):
    provider = make_provider(monkeypatch, y=1.)
    def refuse(evidence):
        raise RuntimeError("leading budget exceeded")
    with pytest.raises(RuntimeError, match="leading budget"):
        provider(None, None, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([6.]),
                 endpoint_budget_mm=1e-10, endpoint_evidence=refuse)
    call = provider.calls[0]
    assert call["stage"] == "endpoint-callback" and call["status"] == "refused"
    assert all(key in call for key in ("raw_hits", "cells", "adjusted_hits", "endpoints"))


@pytest.mark.parametrize("mode", O.MODES)
def test_explicit_unchanged_provider_matches_default_state_slices_and_rng(monkeypatch, mode):
    from mu2efastsim.world import CoaxialShells
    from trkfastsim import storednavigation
    geom = geometry.StrawGeometry(plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=np.zeros((1, 3)), dir=np.array([[1., 0., 0.]]), halflen=np.array([100.]), radius=2.5,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))
    def provider(*a, **k):
        return storednavigation.intervals(*a, **k)
    provider.policy = storednavigation.POLICY
    states = []
    for navigator in (None, provider):
        obj = SimpleNamespace(shells=CoaxialShells([], geometry.DETECTOR_ORIGIN_EXTRACTED))
        rng = np.random.default_rng(19)
        rows = O.transport(obj, rng, np.array([[0., 5., 0.]]), np.array([[0., -1., 0.]]), np.array([80.]), geom,
            tail=False, tail_k=1., mode=mode, max_path_mm=10., navigation_provider=navigator)
        states.append((rows, obj.last_final, obj.last_slices, rng.bit_generator.state))
    for left, right in zip(states[0][:3], states[1][:3]):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    assert states[0][3] == states[1][3]


def test_bound_queries_retain_full_primary_identity_and_forward_refusal(monkeypatch):
    provider = make_provider(monkeypatch, y=1.)
    provider.forward_only = True
    provider.bind_primary_keys(np.array([[1400, 22, 1, 1], [1400, 22, 1, 2]]))
    provider.bind_query_lanes(np.array([1]), 7)
    with pytest.raises(ValueError, match="crosses world handoff plane"):
        provider(None, None, np.zeros((1, 3)), np.array([[0., 1., 0.]]), np.array([6.]), endpoint_budget_mm=1e-10)
    call = provider.calls[0]
    assert call["primary_keys"].tolist() == [[1400, 22, 1, 2]]
    assert call["interior_lanes"].tolist() == [1] and call["iteration"] == 7
    assert call["stage"] == "forward-domain" and call["status"] == "refused"
    assert "cells" not in call


def test_exact_touch_equal_gas_in_distinct_straw_rows_must_not_merge():
    thin = intervals((1., 2.), (2., 3.), (steps.STRAW_GAS.name, steps.STRAW_GAS.name))
    result = G.refine(intervals((), (), ()), thin, np.array([4.]))
    gas = result[result.kind == "thin"]
    assert len(gas) == 2
    assert gas.s0_mm.tolist() == [1., 2.] and gas.s1_mm.tolist() == [2., 3.]
