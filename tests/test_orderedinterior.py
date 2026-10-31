"""Actual make_steps ordered-mode regressions; legacy strict xfails remain controls."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from mu2efastsim.world import CoaxialShells
from trkfastsim import geometry, ledger, orderedinterior, steps, surfaces
from trkfastsim.flighttime import linear_kinetic_dt_ns


def geometry_at(ys):
    n = len(ys)
    zeros = np.zeros(n, dtype=int)
    return geometry.StrawGeometry(plane=zeros, panel=zeros, straw=np.arange(n),
        mid=geometry.to_mu2e(np.column_stack((np.zeros(n), ys, np.zeros(n)))),
        dir=np.tile([1., 0., 0.], (n, 1)), halflen=np.full(n, 500.), radius=2.5,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None],
        plane_t=np.zeros((1, 3)), wall_rmax=2.515, metal_t=.0002, wire_r=0.)


@pytest.fixture
def run(monkeypatch):
    def execute(momentum=80., ys=(0., -400., -900.), support=False, loss=.001,
                max_path=2000., zero_scatter=True, navigation=orderedinterior.MODE, sample_value=999., **options):
        geom = geometry_at(ys)
        origin = geometry.to_mu2e([[0., 1002.5, 0.]])
        generated = pd.DataFrame(dict(run=[1], subrun=[1], event=[1], pdg=[13],
            x=origin[:, 0], y=origin[:, 1] + 500., z=origin[:, 2],
            px=[0.], py=[-momentum], pz=[0.], t=[7.]))

        def plane(gen, geom, rng, world, **kw):
            return (steps._muons(gen, kw["all_muons"]), origin.copy(), np.array([[0., -1., 0.]]),
                    np.array([momentum]), np.array([500.]), np.array([True]))
        plane.last_history = None
        monkeypatch.setattr(steps, "to_plane", plane)
        interior = steps.TrackerInterior.__new__(steps.TrackerInterior)
        tube = [(materials.ALUMINIUM_6061.name, np.eye(3), geometry.to_mu2e([[0., -100.5, 0.]])[0],
                 0., .5, 100., 0., 2 * np.pi)] if support else []
        interior.shells = CoaxialShells(tube, geometry.DETECTOR_ORIGIN_EXTRACTED)
        interior.nsub = 8
        monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full_like(p, loss))
        monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut: np.full_like(p, sample_value))
        if zero_scatter:
            monkeypatch.setattr(steps, "_kink_core", lambda rng, v, a, b:
                                (np.zeros((len(v), 2)), np.zeros((len(v), 2)), np.zeros(len(v))))
        rows, particles = steps.make_steps(generated, geom, np.random.default_rng(19), stage="C",
            interior=interior, history=True, msc_tail=False, navigation=navigation,
            ordered_options=dict(max_path_mm=max_path, **options))
        return SimpleNamespace(rows=rows, particles=particles, interior=interior,
            seg=steps.make_steps.segment_ledger.copy(), mu=steps.make_steps.muon_ledger.copy(),
            vertices=steps.make_steps.path_vertices.copy(), geom=geom, origin=origin)
    return execute


def test_terminal_includes_final_nonkink_group_in_actual_writer(run):
    r = run()
    assert len(r.rows) >= 3
    assert r.interior.last_stats["kinks"] == 0
    last, final = r.seg.iloc[-1], r.mu.iloc[0]
    assert final.path_interior_mm >= last.path_mm
    assert final.lost_thin_MeV >= last.lost_thin_MeV
    assert final.ke_final_MeV <= last.ke_entry_MeV
    np.testing.assert_allclose(final.ke_final_MeV + final.lost_thin_MeV + final.lost_support_MeV,
                               ledger.kinetic(80.), atol=1e-12)
    assert r.vertices.path.max() == final.path_interior_mm == 2000.
    np.testing.assert_allclose(r.seg.writer_debit_MeV, 0., atol=1e-12)


def test_absorbing_stop_truncates_gas_and_has_no_surface_extension(run):
    r = run(momentum=12., ys=(0., -10., -20.), loss=.1)
    assert r.mu.stopped_interior.iloc[0] == 1
    assert r.rows.straw.unique().tolist() == [0]
    assert r.vertices.path.max() < 1010.
    assert r.vertices.path.max() == r.mu.path_interior_mm.iloc[0]
    assert r.interior.last_final["ke"][0] == .5
    np.testing.assert_allclose(r.interior.last_final["lost_support"] + r.interior.last_final["lost_thin"],
                               ledger.kinetic(12.) - .5, atol=1e-12)
    np.testing.assert_allclose(r.rows["len"], np.linalg.norm(
        r.rows[["x1", "y1", "z1"]].to_numpy() - r.rows[["x0", "y0", "z0"]].to_numpy(), axis=1))
    assert r.rows["len"].sum() < 5.
    surf = surfaces.surface_steps(r.rows, r.vertices)
    assert (surf.y0 > -20.).all()


def test_support_loss_step_vertex_and_surface_use_same_physical_clock(run, monkeypatch):
    r = run(support=True, loss=.01)
    sl = r.interior.last_slices
    assert r.mu.lost_support_MeV.iloc[0] > 0
    v = r.vertices.sort_values("path")
    index = np.searchsorted(v.path.to_numpy(), r.seg.path_mm, side="right") - 1
    np.testing.assert_allclose(r.rows.t, v.time_ns.to_numpy()[index], atol=1e-12)
    np.testing.assert_allclose(sl["t1"] - sl["t0"],
                               linear_kinetic_dt_ns(sl["end"] - sl["start"], sl["ke0"], sl["ke1"]), atol=1e-12)
    # Force a surface inside a lossy slice to test the same partial-KE integral.
    material = np.flatnonzero(sl["ke1"] < sl["ke0"])[0]
    sv = v.path.to_numpy()
    j = np.searchsorted(sv, sl["start"][material])
    monkeypatch.setattr(surfaces, "_crossings", lambda a, b: (np.array([j]), np.array([.5]), np.array([1])))
    surf = surfaces.surface_steps(r.rows, v)
    expected = v.time_ns.iloc[j] + linear_kinetic_dt_ns(
        .5 * (sv[j + 1] - sv[j]), v.kinetic_MeV.iloc[j],
        .5 * (v.kinetic_MeV.iloc[j] + v.kinetic_MeV.iloc[j + 1]))
    np.testing.assert_allclose(surf.t, [expected], atol=1e-12)


def test_gas_response_partition_never_overdraws_primary(run):
    r = run(loss=.01)
    sl = r.interior.last_slices
    gas = sl["material"] == steps.STRAW_GAS.name
    np.testing.assert_allclose(sl["deposited"][gas] + sl["unassigned"][gas],
                               sl["ke0"][gas] - sl["ke1"][gas], atol=1e-14)
    np.testing.assert_allclose(r.rows.eion.sum(), sl["deposited"][gas].sum())
    assert (r.rows.eion >= 0).all()


def test_scattering_recomputes_crossings_and_preserves_path_continuity(run):
    r = run(zero_scatter=False, loss=.001)
    sl = r.interior.last_slices
    assert r.interior.last_stats["kinks"] > 0
    assert not np.all(sl["dir"][:, 0] == 0)
    np.testing.assert_allclose(sl["pos0"][1:], sl["pos1"][:-1], atol=1e-10)
    np.testing.assert_allclose(sl["ke0"][1:], sl["ke1"][:-1], atol=1e-12)
    np.testing.assert_array_equal(sl["start"][1:], sl["end"][:-1])
    assert (r.rows.px != 0).any()


def test_ordered_rejects_legacy_arm_combination():
    with pytest.raises(ValueError, match="ordered navigation requires"):
        steps.make_steps(None, None, None, navigation=orderedinterior.MODE, arms="AMT")


def test_no_positive_path_for_initial_cutoff_and_cap_is_error(run):
    r = run(momentum=1.)
    assert r.rows.empty and r.vertices.empty
    assert r.mu.path_interior_mm.iloc[0] == 0.
    assert r.mu.stopped_interior.iloc[0] == 1.
    with pytest.raises(RuntimeError, match="iteration cap"):
        run(max_iter=1)


@pytest.mark.parametrize("options", [{"max_iter": 0}, {"fractional_loss": 2.}, {"max_step_mm": -1.}])
def test_invalid_policy_rejected(run, options):
    with pytest.raises(ValueError):
        run(**options)


def test_gas_unassigned_not_fabricated_escape(run):
    r = run()
    assert r.seg.escaping_gas_MeV.isna().all()
    np.testing.assert_allclose(r.seg.deposited_gas_MeV + r.seg.unassigned_gas_MeV,
                               r.seg.gas_primary_debit_MeV, atol=1e-14)


def test_strict_boundary_control_and_relabel_budget(run):
    r = run(boundary_resolution_mm=0., max_boundary_relabel_mm=0.)
    np.testing.assert_array_equal(r.interior.last_final["boundary_relabel_mm"], [0.])
    with pytest.raises(RuntimeError, match="boundary relabel budget"):
        run(boundary_resolution_mm=1000., max_boundary_relabel_mm=.01)


def test_exact_batch_identity_and_initial_stopped_lane(monkeypatch):
    geom = geometry_at((0.,))
    obj = steps.TrackerInterior.__new__(steps.TrackerInterior)
    obj.shells = CoaxialShells([], geometry.DETECTOR_ORIGIN_EXTRACTED)
    origin = geometry.to_mu2e([[0., 5., 0.], [0., 5., 0.]])
    monkeypatch.setattr(steps, "_kink_core", lambda rng, v, a, b:
                        (np.zeros((len(v), 2)), np.zeros((len(v), 2)), np.zeros(len(v))))
    result = obj.transport(np.random.default_rng(7), origin, np.tile([0., -1., 0.], (2, 1)),
        np.array([80., 1.]), geom, tail=False, navigation=orderedinterior.MODE,
        ordered_options={"max_path_mm": 10.})
    assert (result["it"] == 0).all()
    np.testing.assert_array_equal(obj.last_final["stopped"], [False, True])
    np.testing.assert_array_equal(obj.last_final["path"], [10., 0.])


def test_multiple_gas_slices_do_not_inflate_filter_step_count(run):
    r = run(loss=.01, max_step_mm=.5)
    assert len(r.rows) == 3
    assert not r.rows.duplicated(["run", "subrun", "event", "sp", "sid"]).any()
    gas = r.interior.last_slices["material"] == steps.STRAW_GAS.name
    assert gas.sum() > len(r.rows)
    np.testing.assert_allclose(r.rows.eion.sum(), r.interior.last_slices["deposited"][gas].sum())
    np.testing.assert_allclose(r.rows["len"].sum(),
        (r.interior.last_slices["end"] - r.interior.last_slices["start"])[gas].sum())


def test_constant_dedx_cutoff_location_independent_of_slice_size(run):
    coarse = run(momentum=12., ys=(0., -10., -20.), loss=.1, max_step_mm=10.)
    fine = run(momentum=12., ys=(0., -10., -20.), loss=.1, max_step_mm=.1)
    np.testing.assert_allclose(coarse.mu.path_interior_mm, fine.mu.path_interior_mm, atol=1e-10)
    np.testing.assert_allclose(coarse.vertices.time_ns.iloc[-1], fine.vertices.time_ns.iloc[-1], atol=1e-12)
    assert coarse.mu.ke_final_MeV.iloc[0] == fine.mu.ke_final_MeV.iloc[0] == .5


def test_progress_observation_consumes_no_rng_or_physics(run):
    baseline = run()
    observed = []
    captured = run(progress=observed.append)
    pd.testing.assert_frame_equal(baseline.rows, captured.rows, check_exact=True)
    pd.testing.assert_frame_equal(baseline.vertices, captured.vertices, check_exact=True)
    assert observed[0]["iteration"] == 0


def test_capped_response_bias_is_reported(run):
    r = run()
    assert r.interior.last_stats["gas_capped_slices"] > 0
    assert r.interior.last_stats["gas_sampled_MeV"] > r.interior.last_stats["gas_applied_MeV"]
    np.testing.assert_allclose(r.interior.last_stats["gas_applied_MeV"], r.rows.eion.sum())


def test_sampled_gas_debits_actual_response_not_mean_cap(run):
    r = run(navigation=orderedinterior.SAMPLED_MODE, sample_value=.01)
    assert len(r.rows) == 3
    assert r.interior.last_stats["mode"] == orderedinterior.SAMPLED_MODE
    np.testing.assert_allclose(r.rows.eion, .01)
    np.testing.assert_allclose(r.seg.gas_primary_debit_MeV, r.rows.eion, atol=1e-12)
    np.testing.assert_allclose(r.seg.unassigned_gas_MeV, 0., atol=1e-12)
    assert r.interior.last_stats["gas_capped_slices"] == 0
    final = r.mu.iloc[0]
    np.testing.assert_allclose(final.ke_final_MeV + final.lost_support_MeV + final.lost_thin_MeV,
                               ledger.kinetic(80.), atol=1e-12)


def test_sampled_overshoot_stops_at_cutoff_without_extension(run):
    r = run(momentum=12., navigation=orderedinterior.SAMPLED_MODE, sample_value=999.)
    assert r.mu.stopped_interior.iloc[0] == 1
    assert r.mu.ke_final_MeV.iloc[0] == .5
    assert r.vertices.path.max() < 1000.
    assert r.rows.empty


def test_explicit_entry_evaluation_preserves_current_rows_and_rng(run):
    implicit = run(navigation=orderedinterior.SAMPLED_MODE, sample_value=.01, zero_scatter=False)
    explicit = run(navigation=orderedinterior.SAMPLED_MODE, sample_value=.01, zero_scatter=False,
                   stopping_evaluation="entry")
    pd.testing.assert_frame_equal(implicit.rows, explicit.rows, check_exact=True)
    pd.testing.assert_frame_equal(implicit.vertices, explicit.vertices, check_exact=True)


def test_midpoint_sampled_budget_and_invalid_evaluation(run):
    r = run(navigation=orderedinterior.SAMPLED_MODE, sample_value=.01, stopping_evaluation="midpoint")
    np.testing.assert_allclose(r.seg.gas_primary_debit_MeV, r.rows.eion, atol=1e-12)
    assert r.interior.last_stats["stopping_evaluation"] == "midpoint"
    with pytest.raises(ValueError, match="stopping evaluation"):
        run(stopping_evaluation="invalid")
