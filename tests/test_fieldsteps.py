"""sim.fieldsteps: field-on chord transport through straws (synthetic geometry, no data files)."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import ehelix, eloss
from trkfastsim import crossings, geometry
from trkfastsim.sim import fieldsteps as fs

NSTRAW = 48


def _geometry():
    """Two layers of x-axis straws (y pitch 6 mm, layers at z = -2.5 and +2.5), 900 mm long, about the origin."""
    n = NSTRAW
    y = (np.arange(n) // 2) * 6.0 - 70.0
    z = np.where(np.arange(n) % 2 == 0, -2.5, 2.5)
    zeros = np.zeros(n, int)
    return geometry.StrawGeometry(
        plane=zeros, panel=zeros, straw=np.arange(n), mid=np.c_[np.zeros(n), y, z], dir=np.tile([1.0, 0, 0], (n, 1)),
        halflen=np.full(n, 450.0), radius=2.5085, panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=2.5235, metal_t=0.0001, wire_r=0.0125,
        wire_plate_t=0.0002)


def _start(o, d, p, pdg=13):
    n = len(o)
    return pd.DataFrame({"run": 1, "subrun": 1, "event": np.arange(n), "sp": 1, "pdg": pdg,
                         "x": o[:, 0], "y": o[:, 1], "z": o[:, 2], "px": d[:, 0] * p, "py": d[:, 1] * p,
                         "pz": d[:, 2] * p, "t": 0.0})


def _no_loss(rng, m, p, ln, ms):
    return np.zeros(len(p))


def _lines(n, rng):
    """Lines through the straw band along +z with small tilts, starting below it."""
    o = np.c_[rng.uniform(-300, 300, n), rng.uniform(-60, 60, n), np.full(n, -40.0)]
    d = np.c_[rng.normal(0, 0.05, n), rng.normal(0, 0.3, n), np.ones(n)]
    return o, d / np.linalg.norm(d, axis=1)[:, None]


def test_no_field_matches_straight_crossings_across_chord_ends():
    geom = _geometry()
    o, d = _lines(400, np.random.default_rng(1))
    env = fs.Envelope(geom, pad_r=10.0, pad_z=10.0, axis_xy=(0.0, 0.0))
    origin = np.zeros(3)
    steps, final = fs.transport(np.random.default_rng(2), _start(o, d, 500.0), geom, None, chord_mm=1.7,
                                envelope=env, origin=origin, msc=False, loss=_no_loss)
    ref = crossings.straw_crossings(o, d, geom)
    assert len(steps) == len(ref)
    s = steps.sort_values(["event", "t"]).reset_index(drop=True)
    assert np.array_equal(s.event.to_numpy(), ref[:, 0].astype(int))
    assert np.array_equal(s.straw.to_numpy(), ref[:, 1].astype(int))
    np.testing.assert_allclose(s.len.to_numpy(), ref[:, 3] - ref[:, 2], atol=1e-9)
    e0 = o[ref[:, 0].astype(int)] + d[ref[:, 0].astype(int)] * ref[:, 2:3]
    np.testing.assert_allclose(s[["x0", "y0", "z0"]].to_numpy(), e0, atol=1e-9)
    assert (s.width == 0).all() and (final.status == 0).all()


def test_uniform_field_follows_the_helix_and_entries_lie_on_the_gas_surface():
    geom = _geometry()
    rng = np.random.default_rng(3)
    o, d = _lines(300, rng)
    B = np.array([0.0, 0.0, 1.0])
    env = fs.Envelope(geom, pad_r=10.0, pad_z=10.0, axis_xy=(0.0, 0.0))
    steps, final = fs.transport(np.random.default_rng(4), _start(o, d, 105.0), geom, B, chord_mm=5.0,
                                sagitta_mm=1e-4, envelope=env, origin=np.zeros(3), msc=False, loss=_no_loss)
    # exit state equals the exact helix over the same path
    xe, ue = ehelix.helix(o, d, 105.0, -1.0, np.tile(B, (len(o), 1)), final.path.to_numpy())
    np.testing.assert_allclose(final[["x", "y", "z"]].to_numpy(), xe, atol=1e-6)
    # gas entry points sit on the straw's inner surface (chord sagitta below 1e-4 mm)
    j = steps.straw.to_numpy()
    w = steps[["x0", "y0", "z0"]].to_numpy() - geom.mid[j]
    r = np.hypot(w[:, 1], w[:, 2])
    assert np.all(np.abs(r - geom.radius) < 1e-3)
    assert (steps.width > 0).all() and (steps["shape"] == fs.SHAPE_LINE).all()


def test_width_is_the_makestrawgassteps_formula():
    geom = _geometry()
    o = np.array([[0.0, 2.0, -40.0]])
    d = np.array([[0.6, 0.0, 0.8]])
    env = fs.Envelope(geom, pad_r=10.0, pad_z=10.0, axis_xy=(0.0, 0.0))
    B = np.array([0.0, 0.0, 1.0])
    steps, _ = fs.transport(np.random.default_rng(5), _start(o, d, 105.0), geom, B, chord_mm=1.0, envelope=env,
                            origin=np.zeros(3), msc=False, loss=_no_loss)
    row = steps.iloc[0]
    pdir = np.array([row.px, row.py, row.pz])
    mom = np.linalg.norm(pdir)
    pdir0 = (np.array([row.x1, row.y1, row.z1]) - np.array([row.x0, row.y0, row.z0]))
    pdir0 /= np.linalg.norm(pdir0)
    pperp = np.linalg.norm(np.cross(pdir0, B))
    bn = ehelix.K
    sint = np.linalg.norm(np.cross(np.cross(B, pdir0), [1.0, 0, 0]))
    sag = fs.SAG_PRMS * sint * row.len ** 2 * bn * pperp / mom
    bend = 0.5 * min(geom.radius, mom * pperp / bn)
    assert row.width == pytest.approx(min(sag, bend), rel=2e-3)


def test_muon_loss_and_scattering_change_the_state_and_are_seeded():
    geom = _geometry()
    o, d = _lines(200, np.random.default_rng(6))
    env = fs.Envelope(geom, pad_r=10.0, pad_z=10.0, axis_xy=(0.0, 0.0))
    st = _start(o, d, 150.0)
    a = fs.transport(np.random.default_rng(7), st, geom, np.array([0, 0, 1.0]), envelope=env, origin=np.zeros(3))
    b = fs.transport(np.random.default_rng(7), st, geom, np.array([0, 0, 1.0]), envelope=env, origin=np.zeros(3))
    pd.testing.assert_frame_equal(a[0], b[0])
    hit = np.isin(a[1].event, a[0].event)
    pf = np.linalg.norm(a[1][["px", "py", "pz"]].to_numpy(), axis=1)
    assert hit.sum() > 100 and np.all(pf[hit] < 150.0) and np.all(pf > 140.0)
    assert (a[0].eion >= 0).all() and a[0].eion.mean() > 0


def test_refuses_electrons_without_a_loss_model():
    geom = _geometry()
    o, d = _lines(3, np.random.default_rng(8))
    with pytest.raises(ValueError):
        fs.transport(np.random.default_rng(9), _start(o, d, 105.0, pdg=11), geom, None)


def test_electron_loss_mean_follows_the_restricted_totals():
    from mu2efastsim import materials
    rng = np.random.default_rng(10)
    p = np.full(20000, 105.0)
    lost = fs.electron_loss(rng, materials.MYLAR, p, np.full(len(p), 0.5))
    assert np.all(lost >= 0) and np.all(lost <= p)
    # 0.5 mm Mylar at 105 MeV: about 0.1 MeV of ionisation plus brem about T x / X0 (X0 = 287 mm)
    assert 0.1 < lost.mean() < 0.4


def test_electrons_run_with_the_electron_loss_model():
    geom = _geometry()
    o, d = _lines(50, np.random.default_rng(11))
    env = fs.Envelope(geom, pad_r=10.0, pad_z=10.0, axis_xy=(0.0, 0.0))
    steps, final = fs.transport(np.random.default_rng(12), _start(o, d, 105.0, pdg=11), geom, np.array([0, 0, 1.0]),
                                envelope=env, origin=np.zeros(3), loss=fs.electron_loss, tail=False)
    assert len(steps) > 0 and (final.status == 0).all()
