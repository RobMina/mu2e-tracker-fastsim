"""Bounded candidate contracts; production T3 xfails remain separate."""
import copy

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from mu2efastsim.world import CoaxialShells
from trkfastsim import geometry, historystate, orderedpath as P, strawpath, transportledger as L
from trkfastsim.flighttime import linear_kinetic_dt_ns
from trkfastsim.steps import STRAW_GAS


def geom(ys=(0., -10., -20.)):
    n = len(ys)
    zeros = np.zeros(n, dtype=int)
    return geometry.StrawGeometry(plane=zeros, panel=zeros, straw=np.arange(n),
        mid=np.column_stack((np.zeros(n), ys, np.zeros(n))),
        dir=np.tile([1., 0., 0.], (n, 1)), halflen=np.full(n, 500.), radius=2.5,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None],
        plane_t=np.zeros((1, 3)), wall_rmax=2.515, metal_t=.0002,
        wire_r=.0125, wire_plate_t=.001)


@pytest.fixture
def meta():
    mats = [materials.ALUMINIUM_6061, materials.MYLAR, STRAW_GAS, materials.TUNGSTEN, materials.GOLD]
    return L.provenance(geometry_sha256="0" * 64, tracker_commit="1" * 40, core_commit="2" * 40,
        inputs=[{"name": "synthetic", "sha256": "3" * 64}], frame="detector", settings={},
        media=[L.medium(), *[L.medium(m, cut_MeV=1e9, loss_basis="unrestricted") for m in mats]])


def run(meta, *, kinetic=20., origin=(0., 5., 0.), length=30., **kw):
    return P.transport(CoaxialShells([], [0, 0, 0]), geom(), np.array([[1, 1, 1, 1]]),
        [origin], [[0., -1., 0.]], [kinetic], [length], meta, mode=P.MODE, **kw)


def test_concentric_straw_order_and_lengths():
    result = strawpath.straw_intervals(geom((0.,)), [[0., 5., 0.]], [[0., -1., 0.]], 10.)
    assert result.material.tolist() == ["vacuum", materials.ALUMINIUM_6061.name, materials.MYLAR.name,
        STRAW_GAS.name, materials.GOLD.name, materials.TUNGSTEN.name,
        materials.GOLD.name, STRAW_GAS.name, materials.MYLAR.name,
        materials.ALUMINIUM_6061.name, "vacuum"]
    lengths = result.s1_mm - result.s0_mm
    np.testing.assert_allclose(lengths, [2.485, .0002, .0148, 2.4875, .001, .023, .001,
                                        2.4875, .0148, .0002, 2.485], atol=1e-14)
    assert result.s0_mm.iloc[0] == 0 and result.s1_mm.iloc[-1] == 10
    np.testing.assert_array_equal(result.s0_mm.iloc[1:], result.s1_mm.iloc[:-1])


def test_wire_miss_start_inside_zero_and_empty():
    g = geom((0.,))
    result = strawpath.straw_intervals(g, [[0, 0, 1.]], [[0, -1., 0]], 1.)
    assert result.material.tolist() == [STRAW_GAS.name]
    for origin, direction, length in [(np.empty((0, 3)), np.empty((0, 3)), []),
                                       ([[0., 5., 0.]], [[0., -1., 0.]], 0.)]:
        assert strawpath.straw_intervals(g, origin, direction, length).empty


def test_material_order_uses_updated_ke_and_terminal_never_regresses(meta, monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda m, p, cut: np.ones_like(p))
    result = run(meta, max_step_mm=100., fractional_loss=1.)
    h = result["history"]
    assert result["traveled_mm"][0] == 30.
    assert result["final_ke_MeV"][0] < h["ke_out_MeV"][3]
    assert result["traveled_mm"][0] >= result["intervals"].s1_mm.max()
    np.testing.assert_array_equal(h["ke_in_MeV"][1:], h["ke_out_MeV"][:-1])
    np.testing.assert_allclose(h.accounting()["primary_residual_MeV"], 0., atol=1e-14)
    assert not h.accounting()["partition_complete"][h["medium"] > 0].any()


def test_stop_absorbs_before_unreachable_straw_without_extension(meta, monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda m, p, cut: np.ones_like(p))
    result = run(meta, kinetic=3., max_step_mm=.1, fractional_loss=.1)
    assert result["status"].tolist() == ["model_stop"]
    stop = result["traveled_mm"][0]
    assert stop < 12.5
    h = result["history"]
    np.testing.assert_allclose(h["path_mm"].sum(), stop)
    state = P.at_entries(result, np.tile([1, 1, 1, 1], (3, 1)), [2.5, stop, 12.5])
    assert state["available"].tolist() == [True, True, False]
    assert state["alive"].tolist() == [True, False, False]
    assert h["terminal"][-1] == "model_stop" and h["ke_out_MeV"][-1] > 0


def test_single_clock_queries_match_integral_not_entry_to_entry_smearing(meta, monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda m, p, cut: np.ones_like(p))
    result = run(meta, max_step_mm=100., fractional_loss=1.)
    h = result["history"]
    a, b, t0, t1 = historystate.path_coordinates(h)
    path = a + (b - a) / 2
    state = P.at_entries(result, np.tile([1, 1, 1, 1], (len(h), 1)), path)
    ke = .5 * (h["ke_in_MeV"] + h["ke_out_MeV"])
    np.testing.assert_allclose(state["dt_from_start_ns"], t0 +
        linear_kinetic_dt_ns(path - a, h["ke_in_MeV"], ke), atol=1e-13)
    assert (np.diff(state["dt_from_start_ns"]) > 0).all()


def test_surface_clock_uses_same_partial_segment_query(meta, monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda m, p, cut: np.ones_like(p))
    origin = geometry.to_mu2e([[0., 5., 0.]])[0]
    g = geom((0.,))
    g.mid = geometry.to_mu2e(g.mid)
    result = P.transport(CoaxialShells([], [0, 0, 0]), g, np.array([[1, 1, 1, 1]]),
        [origin], [[0., -1., 0.]], [20.], [10.], meta, mode=P.MODE)
    starts = pd.DataFrame([[1, 1, 1, 1, 7.]], columns=["run", "subrun", "event", "sp", "t_start_ns"])
    fake = lambda a, b: (np.array([2]), np.array([.5]), np.array([1]))
    monkeypatch.setattr("trkfastsim.surfaces._crossings", fake)
    rows = P.surface_steps(result, starts)
    a, b, _, _ = historystate.path_coordinates(result["history"])
    q = P.at_entries(result, np.array([[1, 1, 1, 1]]), [(a[2] + b[2]) / 2])
    np.testing.assert_allclose(rows.t, 7 + q["dt_from_start_ns"], atol=1e-13)


def test_opt_in_refusals_do_not_change_production_or_metadata(meta):
    before = copy.deepcopy(meta)
    with pytest.raises(ValueError, match="explicit mode"):
        P.transport(None, None, [], [], [], [], [], meta, mode="default")
    run(meta)
    assert meta == before
    with pytest.raises(ValueError, match="overlapping"):
        strawpath.straw_intervals(geom((0., 0.)), [[0., 5., 0.]], [[0., -1., 0.]], 10.)
