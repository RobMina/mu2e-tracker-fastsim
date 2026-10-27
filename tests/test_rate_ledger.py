"""Transport-ledger arms (trkfastsim.ledger): analytic flight-time limits, world band reconstruction, writer
conventions, thin-material loss on synthetic straws, and default (legacy) equality."""

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from mu2efastsim.world import CoaxialShells
from trkfastsim import geometry, ledger, steps

RADIUS = 2.5
P_START = 80.0
START_Y = 1002.5
T_START = 7.0
C = steps.C_MM_PER_NS


def _numeric_time(L, p_in, p_out, n=200001):
    """Midpoint-rule flight time for kinetic energy linear in path."""
    s = (np.arange(n) + 0.5) / n * L
    ta, tb = ledger.kinetic(p_in), ledger.kinetic(p_out)
    tk = ta + (tb - ta) * s / L
    return float(np.sum(1.0 / ledger.beta(ledger.momentum(tk))) * (L / n) / C)


def test_flight_time_constant_momentum_is_exact():
    L = np.array([0.0, 1.0, 1234.5])
    for p in (30.0, 300.0, 3e4):
        np.testing.assert_allclose(ledger.flight_time_ns(L, p, p), L / (ledger.beta(p) * C), rtol=1e-14)


@pytest.mark.parametrize("p_in,p_out", [(300.0, 200.0), (120.0, 40.0), (2000.0, 1990.0), (500.0, 499.999999)])
def test_flight_time_linear_kinetic_energy_matches_quadrature(p_in, p_out):
    L = 750.0
    np.testing.assert_allclose(ledger.flight_time_ns(L, p_in, p_out), _numeric_time(L, p_in, p_out), rtol=1e-8)


def test_flight_time_is_additive_and_finite_at_stop():
    L, p_in = 400.0, 150.0
    ta = ledger.kinetic(p_in)
    p_mid = ledger.momentum(0.5 * ta)              # half the energy at half the path (linear in path)
    whole = ledger.flight_time_ns(L, p_in, 0.0)
    parts = ledger.flight_time_ns(L / 2, p_in, p_mid) + ledger.flight_time_ns(L / 2, p_mid, 0.0)
    np.testing.assert_allclose(whole, parts, rtol=1e-12)
    assert np.isfinite(whole) and whole > L / C
    np.testing.assert_allclose(whole, _numeric_time(L, p_in, 1e-3, n=2000001), rtol=2e-3)   # integrable 1/beta at 0


def test_world_band_momenta_reconstruct_entry_and_exit():
    p0 = np.array([1000.0, 500.0])
    p_edges = np.array([[1000.0, 900.0, 900.0, 600.0], [500.0, 450.0, 450.0, 0.0]])   # band 1 not crossed
    blen = np.array([[100.0, 0.0, 50.0], [80.0, 0.0, 40.0]])
    band_p = 0.5 * (p_edges[:, [0, 1, 2]] + p_edges[:, [1, 2, 3]])
    band_p[:, 1] = 0.0                              # world_loss leaves uncrossed bands at zero
    p_in, p_out = ledger.world_band_momenta(p0, band_p, blen)
    np.testing.assert_allclose(p_in[:, [0, 2]], p_edges[:, [0, 2]])
    np.testing.assert_allclose(p_out[:, [0, 2]], p_edges[:, [1, 3]])
    np.testing.assert_allclose(p_out[:, 1], p_in[:, 1])


def test_world_time_vacuum_limit():
    o, d = np.array([[0.0, 5000.0, 0.0]]), np.array([[0.0, -1.0, 0.0]])
    p = np.array([700.0])
    blen = np.array([[2000.0, 2980.0]])
    t = ledger.world_time_ns(o, d, 20.0, p, np.c_[p, p], blen, np.array([4980.0]), p)
    np.testing.assert_allclose(t, 4980.0 / (ledger.beta(700.0) * C), rtol=1e-14)


def test_presets_and_core_convention():
    assert ledger.preset("legacy") == ledger.Arms()
    assert ledger.preset("AMT") == ledger.Arms(writer="none", thin=True, time="integrated")
    with pytest.raises(ValueError):
        ledger.Arms(writer="air")
    assert ledger.core_world_convention()["world_convention"] in ("energy", "momentum")


# ---- make_steps on synthetic straws (fixtures as tests/test_step_energy_ledger.py)
def _geometry(ys=(0.0,), wall=0.0, metal=0.0):
    n = len(ys)
    zeros = np.zeros(n, dtype=int)
    return geometry.StrawGeometry(
        plane=zeros, panel=zeros, straw=np.arange(n),
        mid=np.column_stack((np.zeros(n), ys, np.zeros(n))),
        dir=np.tile([1.0, 0.0, 0.0], (n, 1)), halflen=np.full(n, 500.0),
        radius=RADIUS, panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=RADIUS + wall, metal_t=metal)


@pytest.fixture
def generated():
    return pd.DataFrame({"run": [1], "subrun": [1], "event": [1], "pdg": [13], "x": [0.0], "y": [START_Y], "z": [0.0],
                         "px": [0.0], "py": [-P_START], "pz": [0.0], "t": [T_START]})


@pytest.fixture
def empty_interior():
    interior = steps.TrackerInterior.__new__(steps.TrackerInterior)
    interior.shells = CoaxialShells([], axis_point=[0.0, 0.0, 0.0])
    interior.nsub = 8
    return interior


@pytest.fixture
def plane_state(monkeypatch):
    def install(momentum, origin_y=START_Y, upstream_path=500.0):
        def to_plane(gen, geom, rng, world, **kwargs):
            mu = steps._muons(gen, kwargs["all_muons"])
            steps.to_plane.last_history = None
            return (mu, np.array([[0.0, origin_y, 0.0]]), np.array([[0.0, -1.0, 0.0]]), np.array([momentum]),
                    np.array([upstream_path]), np.array([True]))
        monkeypatch.setattr(steps, "to_plane", to_plane)
    return install


def _run(generated, geom, **kw):
    rows, parts = steps.make_steps(generated, geom, np.random.default_rng(19), msc_tail=False, exit_scatter=False, **kw)
    assert len(rows) == geom.nstraws and len(parts) == 1
    return rows


def _p(rows):
    return np.linalg.norm(rows[["px", "py", "pz"]].to_numpy(), axis=1)


def test_integrated_time_uses_local_momentum(generated, empty_interior, plane_state):
    generated = generated.assign(py=-300.0)
    geom = _geometry((0.0, -1000.0))
    plane_state(P_START, origin_y=steps.plane_y(geom))
    rows = _run(generated, geom, stage="C", interior=empty_interior, arms=ledger.Arms(writer="none", time="integrated"))
    np.testing.assert_allclose(_p(rows), P_START, rtol=1e-12)
    sep = np.linalg.norm(np.diff(rows[["x0", "y0", "z0"]].to_numpy(), axis=0))
    np.testing.assert_allclose(rows.t.iloc[1] - rows.t.iloc[0], sep / (ledger.beta(P_START) * C), rtol=1e-12)
    legacy = _run(generated, geom, stage="C", interior=empty_interior, arms="A")
    assert not np.isclose(legacy.t.iloc[1] - legacy.t.iloc[0], sep / (ledger.beta(P_START) * C), rtol=1e-6)


def test_vacuum_writer_has_no_air_debit(generated, empty_interior, plane_state):
    geom = _geometry()
    plane_state(P_START, origin_y=steps.plane_y(geom))
    rows = _run(generated, geom, stage="C", interior=empty_interior, arms="A")      # air_loss default True ignored
    np.testing.assert_allclose(_p(rows), P_START, rtol=1e-12, atol=1e-10)
    legacy = _run(generated, geom, stage="C", interior=empty_interior)
    assert _p(legacy)[0] < P_START


def test_energy_writer_debits_kinetic_energy(generated, empty_interior, plane_state):
    geom = _geometry()
    y_pl = steps.plane_y(geom)
    plane_state(P_START, origin_y=y_pl)
    rows = _run(generated, geom, stage="C", interior=empty_interior, arms="E")
    s_in = y_pl - RADIUS
    loss = float(eloss.dedx_restricted(materials.AIR, P_START, 1e9)) * s_in
    np.testing.assert_allclose(ledger.kinetic(_p(rows)), ledger.kinetic(P_START) - loss, rtol=1e-12)


def test_thin_loss_charges_straw_material_to_the_next_straw(generated, empty_interior, plane_state):
    wall, metal = 0.015, 0.0002
    geom = _geometry((0.0, -10.0), wall=wall, metal=metal)
    plane_state(P_START, origin_y=steps.plane_y(geom))
    rows = _run(generated, geom, stage="C", interior=empty_interior, arms="AM", history=True)
    lens = [(materials.MYLAR, 2 * (wall - metal)), (materials.ALUMINIUM_6061, 2 * metal), (steps.STRAW_GAS, 2 * RADIUS)]
    expect = sum(float(eloss.dedx_restricted(m, P_START, 1e9)) * L for m, L in lens)
    ke = ledger.kinetic(_p(rows))
    np.testing.assert_allclose(ke[0], ledger.kinetic(P_START), rtol=1e-12)
    np.testing.assert_allclose(ke[0] - ke[1], expect, rtol=1e-9)
    seg = steps.make_steps.segment_ledger
    np.testing.assert_allclose(seg.lost_thin_MeV.to_numpy(), [0.0, expect], rtol=1e-9)
    np.testing.assert_allclose(seg.writer_debit_MeV, 0.0, atol=1e-9)              # written = transport (vacuum writer)
    np.testing.assert_allclose(seg.l_thin_mm.to_numpy(), [0.0, sum(L for _, L in lens)], rtol=1e-9)
    off = _run(generated, geom, stage="C", interior=empty_interior, arms="A", history=True)
    np.testing.assert_allclose(_p(off), P_START, rtol=1e-12)                       # legacy: walls take no energy
    np.testing.assert_allclose(steps.make_steps.segment_ledger.l_thin_mm, seg.l_thin_mm, rtol=1e-12)


def test_default_equals_legacy_preset_and_history_draws_nothing(generated, empty_interior, plane_state):
    geom = _geometry((0.0, -10.0, -20.0), wall=0.015, metal=0.0002)
    plane_state(P_START, origin_y=steps.plane_y(geom))
    a = _run(generated, geom, stage="C", interior=empty_interior)
    b = _run(generated, geom, stage="C", interior=empty_interior, arms="legacy", history=True)
    pd.testing.assert_frame_equal(a, b)


def test_thin_and_time_arms_need_the_interior(generated):
    with pytest.raises(ValueError):
        steps.make_steps(generated, _geometry(), np.random.default_rng(1), arms="M")


@pytest.mark.parametrize("arm", ["A", pytest.param("AMT", marks=pytest.mark.xfail(
    strict=True, reason="path vertices exist only at kinks: between them surface times use one speed, while the step "
                        "times follow the thin-material loss at every straw (sub-ps here)"))])
def test_step_and_path_vertex_times_agree_at_the_entries(generated, empty_interior, plane_state, arm):
    """Surface crossings interpolate the path vertices: at each straw entry that time should equal the step time."""
    generated = generated.assign(py=-300.0)
    geom = _geometry((0.0, -400.0, -900.0))
    plane_state(P_START, origin_y=steps.plane_y(geom))
    rows = _run(generated, geom, stage="C", interior=empty_interior, arms=arm)
    v = steps.make_steps.path_vertices
    start = v[np.isclose(v.path.to_numpy(), 0.0)].iloc[0]
    s = np.linalg.norm(rows[["x0", "y0", "z0"]].to_numpy() - start[["x", "y", "z"]].to_numpy(float), axis=1)
    t_vertex = start.t0 + s / (start.beta * C)
    np.testing.assert_allclose(rows.t.to_numpy(), t_vertex, rtol=0, atol=1e-9)
    assert np.all(np.diff(rows.t.to_numpy()) > 0)


def test_g4_keys_map_to_fast_muon_rank_through_the_generator_index():
    """G4 key = generator index (all CRY particles) + 1; fast key = rank among the muons; first muon = fast key 1."""
    from trkfastsim.analyses import ratearms
    ev = {"run": 1, "subrun": 1}
    # event 1: proton, muon (G4 key 2 = fast key 1); event 2: two muons, only the second stored in G4 (key 2)
    fast = pd.DataFrame([{**ev, "event": 1, "genidx": 1, "sp": 1, "ngood": 12},
                         {**ev, "event": 2, "genidx": 0, "sp": 1, "ngood": 0},
                         {**ev, "event": 2, "genidx": 1, "sp": 2, "ngood": 15}])
    g4 = pd.DataFrame([{**ev, "event": 1, "sp": 2, "genidx": 0, "ngood": 13},
                       {**ev, "event": 2, "sp": 2, "genidx": 0, "ngood": 14}])
    m, bad = ratearms.g4_fast_keys(g4, fast)
    assert bad == 0
    assert m.sort_values("event").sp.tolist() == [1, 2]
    e = ratearms._events(m, "g4").sort_values("event")
    assert e.g4_any.tolist() == [True, True] and e.g4_mu1.tolist() == [True, False]
    f = ratearms._events(fast, "fast").sort_values("event")
    assert f.fast_any.tolist() == [True, True] and f.fast_mu1.tolist() == [True, False]
