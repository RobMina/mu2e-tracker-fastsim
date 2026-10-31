"""Open T3 limits: terminal bookkeeping, stopping, and support-only flight time.

Synthetic stage-C boundaries exercise the real transport/writers in both modes.
Strict legacy xfails preserve default failure evidence; ordered mode must pass.
Ordered clocks integrate material slices, not loss smeared across vacuum gaps.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import materials
from mu2efastsim.world import CoaxialShells
from trkfastsim import geometry, ledger, steps


RADIUS = 2.5


def _geometry(ys):
    n = len(ys)
    zeros = np.zeros(n, dtype=int)
    return geometry.StrawGeometry(
        plane=zeros, panel=zeros, straw=np.arange(n),
        mid=np.column_stack((np.zeros(n), ys, np.zeros(n))),
        dir=np.tile([1.0, 0.0, 0.0], (n, 1)), halflen=np.full(n, 500.0),
        radius=RADIUS, panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)),
        wall_rmax=RADIUS + 0.015, metal_t=0.0002, wire_r=0.0,
    )


@pytest.fixture(params=[("legacy", "entry"), ("ordered-stored-mean-v1", "entry"),
                        ("ordered-stored-sampled-v1", "entry"), ("ordered-stored-mean-v1", "midpoint"),
                        ("ordered-stored-sampled-v1", "midpoint")])
def navigation_spec(request):
    return request.param


@pytest.fixture
def navigation(navigation_spec):
    return navigation_spec[0]


@pytest.fixture
def run_transport(monkeypatch, navigation, navigation_spec):
    """Keep real interior transport/writers; inject only boundary state and test material."""
    def run(momentum=80.0, ys=(0.0, -10.0, -20.0), arms="AM", support_loss=None):
        geom = _geometry(ys)
        # A long vacuum approach makes any premature slowing of the vertex clock visible.
        origin_y = 1002.5
        generated = pd.DataFrame({
            "run": [1], "subrun": [1], "event": [1], "pdg": [13],
            "x": [0.0], "y": [origin_y + 500.0], "z": [0.0],
            "px": [0.0], "py": [-momentum], "pz": [0.0], "t": [7.0],
        })

        def to_plane(gen, geom, rng, world, **kwargs):
            return (steps._muons(gen, kwargs["all_muons"]),
                    np.array([[0.0, origin_y, 0.0]]), np.array([[0.0, -1.0, 0.0]]),
                    np.array([momentum]), np.array([500.0]), np.array([True]))

        to_plane.last_history = None
        monkeypatch.setattr(steps, "to_plane", to_plane)
        interior = steps.TrackerInterior.__new__(steps.TrackerInterior)
        interior.shells = CoaxialShells([], axis_point=[0.0, 0.0, 0.0])
        interior.nsub = 8

        if support_loss is not None:
            # A 1-mm slab at y=-100..-101, between the first and second entries.
            # Its fixed energy debit isolates time bookkeeping from loss fluctuations.
            def support(o, d, length, moments):
                entry = (-100.0 - o[:, 1]) / d[:, 1]
                exit_ = (-101.0 - o[:, 1]) / d[:, 1]
                chord = np.maximum(np.minimum(exit_, length) - np.maximum(entry, 0.0), 0.0)
                return {materials.ALUMINIUM_6061.name: chord}, {}

            def slab_loss(rng, mat, p, length):
                return support_loss * np.asarray(length)

            def zero_deflection(rng, variance, w1, w2):
                # Retain the support kink/restart, without transverse random motion.
                return np.zeros((len(variance), 2)), np.zeros((len(variance), 2)), np.zeros(len(variance))

            monkeypatch.setattr(interior, "_support", support)
            monkeypatch.setattr(steps.tr, "slab_loss", slab_loss)
            monkeypatch.setattr(steps, "_kink_core", zero_deflection)

        ordered = navigation != "legacy"
        if ordered:
            if support_loss is not None:
                tube = [(materials.COPPER.name, np.eye(3), np.array([0., -100.5, 0.]),
                         0., .5, 100., 0., 2 * np.pi)]
                interior.shells = CoaxialShells(tube, axis_point=[0., 0., 0.])
                monkeypatch.setattr(steps.eloss, "dedx_restricted", lambda mat, p, cut, *args:
                    np.full_like(p, support_loss if mat.name == materials.COPPER.name else 0.))
            if navigation == "ordered-stored-sampled-v1":
                monkeypatch.setattr(steps.eloss, "sample_loss", lambda rng, mat, p, length, cut:
                    steps.eloss.dedx_restricted(mat, p, cut) * length)
            monkeypatch.setattr(steps, "_kink_core", lambda rng, v, a, b:
                (np.zeros((len(v), 2)), np.zeros((len(v), 2)), np.zeros(len(v))))
        rows, particles = steps.make_steps(
            generated, geom, np.random.default_rng(19), stage="C", interior=interior,
            arms=None if ordered else arms, history=True, msc_tail=False, exit_scatter=False,
            navigation=navigation, ordered_options={"max_path_mm": 2500.,
                "stopping_evaluation": navigation_spec[1]} if ordered else None,
            # No thin-material kinks; support still triggers its own kink.
            kink_xx0=np.inf, kink_sup_xx0=None,
        )
        return SimpleNamespace(
            rows=rows, particles=particles, geom=geom, origin_y=origin_y,
            seg=steps.make_steps.segment_ledger.copy(), mu=steps.make_steps.muon_ledger.copy(),
            vertices=steps.make_steps.path_vertices.copy(), stats=interior.last_stats.copy(),
        )
    return run


@pytest.fixture
def nonkink_run(run_transport):
    return run_transport()


def test_non_kink_entries_charge_thin_material(nonkink_run):
    """Control: the entry ledger sees real loss even though no group kinks."""
    r = nonkink_run
    assert len(r.rows) == r.geom.nstraws and len(r.particles) == 1
    assert r.stats["kinks"] == 0 and not r.mu.stopped_interior.iloc[0]
    assert np.all(np.diff(r.seg.path_mm) > 0)
    assert np.all(np.diff(r.seg.lost_thin_MeV) > 0)
    np.testing.assert_allclose(r.seg.lost_support_MeV, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        r.seg.ke_entry_MeV + r.seg.lost_thin_MeV, ledger.kinetic(80.0), atol=1e-10,
    )
    np.testing.assert_allclose(r.seg.ke_written_MeV, r.seg.ke_entry_MeV, atol=1e-10)


def test_terminal_muon_ledger_includes_final_non_kink_group(nonkink_run, navigation, request):
    if navigation == "legacy":
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
            "T3: final non-kink group does not advance the terminal muon path/energy ledger")))
    r = nonkink_run
    final, last = r.mu.iloc[0], r.seg.iloc[-1]
    print(f"terminal: entry losses={r.seg.lost_thin_MeV.tolist()}; "
          f"last entry path={last.path_mm:g} mm, final path={final.path_interior_mm:g} mm; "
          f"final thin loss={final.lost_thin_MeV:g} MeV")
    # Irrespective of whether the endpoint is an entry, exit, or envelope boundary,
    # it cannot precede any emitted entry or undo energy already lost there.
    violations = np.array([
        last.path_mm - final.path_interior_mm,
        last.lost_thin_MeV - final.lost_thin_MeV,
        last.lost_support_MeV - final.lost_support_MeV,
        final.ke_final_MeV - last.ke_entry_MeV,
    ])
    assert np.all(violations <= 1e-9), f"terminal ledger regresses behind last entry: {violations}"


@pytest.fixture
def stopped_run(run_transport):
    return run_transport(momentum=12.0, arms="AMT")


def test_stopped_muon_suppresses_downstream_straw_steps(stopped_run, navigation):
    """Control: the transport stop decision already suppresses subsequent entries."""
    r = stopped_run
    assert r.mu.stopped_interior.iloc[0] == 1
    unique = r.rows.straw.drop_duplicates().tolist()
    assert 0 < len(unique) < r.geom.nstraws
    assert r.stats["kinks"] == 0
    assert unique == list(range(len(unique)))
    if navigation == "legacy":
        assert len(r.rows) == len(unique)
    assert np.all(r.seg.ke_entry_MeV > 0.5)


def test_stopped_muon_path_ends_before_unreachable_entry(stopped_run, navigation, request):
    if navigation == "legacy":
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
            "T3: a stopped muon's path vertices still acquire a downstream straight extension")))
    r = stopped_run
    # Bound the endpoint by the first suppressed straw, NOT the stale terminal ledger.
    # No exact microscopic stopping position is assumed by this conservative check.
    missing = np.setdiff1d(np.arange(r.geom.nstraws), r.rows.straw.to_numpy())
    unreachable_entry = r.origin_y - r.geom.mid[missing[0], 1] - RADIUS
    endpoint = r.vertices.path.max()
    print(f"stopped: emitted={len(r.rows)}/{r.geom.nstraws}, "
          f"first unreachable entry={unreachable_entry:g} mm, path endpoint={endpoint:g} mm")
    assert endpoint <= unreachable_entry + 1e-9, (
        f"stopped path reaches {endpoint:g} mm past unreachable entry at {unreachable_entry:g} mm"
    )


@pytest.mark.parametrize("support_loss", [
    pytest.param(0.0, id="zero-loss-control"),
    pytest.param(5.0, id="support-loss"),
])
def test_support_only_integrated_step_and_vertex_times_agree(run_transport, support_loss, navigation, request):
    if navigation == "legacy" and support_loss:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError,
            reason="T3: integrated step and vertex times partition support-only energy loss differently"))
    r = run_transport(ys=(0.0, -400.0, -900.0),
                      arms=ledger.Arms(writer="none", thin=False, time="integrated"),
                      support_loss=support_loss)
    assert r.rows.straw.nunique() == r.geom.nstraws
    if navigation == "legacy":
        assert r.stats["kinks"] > 0
    np.testing.assert_allclose(r.seg.lost_thin_MeV, 0.0, atol=1e-12)
    np.testing.assert_allclose(r.seg.lost_support_MeV, [0.0, support_loss, support_loss], atol=1e-10)
    # Legacy smears loss entry-to-entry; ordered time follows actual material slices.
    p = ledger.momentum(r.seg.ke_entry_MeV.to_numpy())
    if navigation == "legacy":
        np.testing.assert_allclose(np.diff(r.rows.t),
                                   ledger.flight_time_ns(np.diff(r.seg.path_mm), p[:-1], p[1:]),
                                   rtol=0, atol=1e-9)
    # Same t0 + path/(beta*c) interpolation used by surface crossings. The track
    # is straight, so each entry's path also locates it on the spatial polyline.
    v = r.vertices.sort_values("path").reset_index(drop=True)
    path = r.seg.path_mm.to_numpy()
    left = np.searchsorted(v.path.to_numpy(), path, side="right") - 1
    assert np.all((left >= 0) & (left < len(v) - 1))
    vertex_time = (v.t0.to_numpy()[left]
                   + path / (v.beta.to_numpy()[left] * steps.C_MM_PER_NS))
    if navigation != "legacy":
        from trkfastsim.flighttime import linear_kinetic_dt_ns
        local = path - v.path.to_numpy()[left]
        right = left + 1
        fraction = local / (v.path.to_numpy()[right] - v.path.to_numpy()[left])
        kin = v.kinetic_MeV.to_numpy()[left]
        kout = kin + fraction * (v.kinetic_MeV.to_numpy()[right] - kin)
        vertex_time = v.time_ns.to_numpy()[left] + linear_kinetic_dt_ns(local, kin, kout)
    delta_ps = 1e3 * (r.rows.t.to_numpy() - vertex_time)
    print(f"support loss={support_loss:g} MeV: step-minus-vertex time={delta_ps.tolist()} ps")
    np.testing.assert_allclose(r.rows.t.to_numpy(), vertex_time, rtol=0, atol=1e-9)
