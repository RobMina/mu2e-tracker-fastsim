"""sim.fieldsteps_jax: the device twin of the field-on electron transport (synthetic geometry, no data files)."""
import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")

from mu2efastsim import materials  # noqa: E402
from trkfastsim import geometry  # noqa: E402
from trkfastsim.sim import fieldsteps as fs  # noqa: E402
from trkfastsim.sim import fieldsteps_jax as FJ  # noqa: E402

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


def _grid(B):
    """A uniform field B [T] on a regular grid covering the test volume (G4BL layout: lo, d, n, B[nx, ny, nz, 3])."""
    n = np.array([5, 5, 5])
    return np.array([-1000.0, -1000.0, -1000.0]), np.array([500.0, 500.0, 500.0]), n, np.broadcast_to(
        np.asarray(B, float), (5, 5, 5, 3)).copy()


def _start(o, d, p):
    n = len(o)
    return pd.DataFrame({"run": 1, "subrun": 1, "event": np.arange(n), "sp": 1, "pdg": 11,
                         "x": o[:, 0], "y": o[:, 1], "z": o[:, 2], "px": d[:, 0] * p, "py": d[:, 1] * p,
                         "pz": d[:, 2] * p, "t": 0.0})


def _lines(n, rng):
    o = np.c_[rng.uniform(-300, 300, n), rng.uniform(-60, 60, n), np.full(n, -40.0)]
    d = np.c_[rng.normal(0, 0.05, n), rng.normal(0, 0.3, n), np.ones(n)]
    return o, d / np.linalg.norm(d, axis=1)[:, None]


def _setup(B, **kw):
    geom = _geometry()
    env = fs.Envelope(geom, pad_r=10.0, pad_z=10.0, axis_xy=(0.0, 0.0))
    trk = FJ.Tracker(geom, _grid(B), None, env, np.zeros(3), np.asarray(B, float), **kw)
    return geom, env, trk


def _no_loss(rng, m, p, ln, ms):
    return np.zeros(len(p))


def test_without_loss_and_scattering_the_device_rows_equal_numpy():
    B = np.array([0.0, 0.0, 1.0])
    geom, env, trk = _setup(B, chord_mm=5.0, eloss_on=False, msc=False)
    o, d = _lines(200, np.random.default_rng(3))
    start = _start(o, d, 105.0)
    ref, rfin = fs.transport(np.random.default_rng(4), start, geom, lambda x: np.tile(B, (len(x), 1)), chord_mm=5.0,
                             envelope=env, origin=np.zeros(3), bnom=B, msc=False, loss=_no_loss, tail=False)
    res = FJ.transport(trk, jax.random.PRNGKey(1), o, d * 105.0, np.zeros(len(o)))
    dev = FJ.to_frame(res, start)
    assert len(dev) == len(ref) > 100
    a = dev.sort_values(["event", "t"]).reset_index(drop=True)
    b = ref.sort_values(["event", "t"]).reset_index(drop=True)
    assert np.array_equal(a.straw.to_numpy(), b.straw.to_numpy())
    assert np.array_equal(a.nsgs.to_numpy(), b.nsgs.to_numpy())
    for c in ("len", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz", "width"):
        np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(), atol=1e-8)
    fin = FJ.final_frame(res, start)
    np.testing.assert_allclose(fin[["x", "y", "z"]].to_numpy(), rfin[["x", "y", "z"]].to_numpy(), atol=1e-8)
    assert np.array_equal(fin.status.to_numpy(), rfin.status.to_numpy())
    assert all(v == 0 for k, v in res.stats.items() if k.endswith("overflow"))


def test_electron_loss_pairs_follow_the_numpy_model():
    """Mean and spread of the device per-(track, material) loss against fieldsteps.electron_loss."""
    _, _, trk = _setup(np.array([0.0, 0.0, 1.0]))
    n = 40000
    for mi, ln in ((len(trk.mats) - 3, 5.0), (len(trk.mats) - 5, 0.015), (len(trk.mats) - 2, 0.02)):
        m = trk.mats[mi]
        ref = fs.electron_loss(np.random.default_rng(5), m, np.full(n, 104.0), np.full(n, ln))
        dev, dcap, bcap = FJ._electron_loss(jax.random.PRNGKey(6), trk.T, trk.mc, trk.lnt0, trk.dlnt,
                                            jax.numpy.full(n, 104.0), jax.numpy.full(n, ln),
                                            jax.numpy.full(n, mi, jax.numpy.int32))
        dev = np.asarray(dev)
        se = np.hypot(ref.std(), dev.std()) / np.sqrt(n)
        assert abs(dev.mean() - ref.mean()) < 5 * se + 1e-6, m.name
        assert int(dcap) == 0 and int(bcap) == 0
        np.testing.assert_allclose(np.median(dev), np.median(ref), rtol=0.05)


def test_full_physics_transport_agrees_with_numpy_statistically_and_is_seeded():
    B = np.array([0.0, 0.0, 1.0])
    geom, env, trk = _setup(B)
    o, d = _lines(400, np.random.default_rng(7))
    start = _start(o, d, 105.0)
    ref, rfin = fs.transport(np.random.default_rng(8), start, geom, lambda x: np.tile(B, (len(x), 1)), envelope=env,
                             origin=np.zeros(3), bnom=B, loss=fs.electron_loss, tail=False)
    r1 = FJ.transport(trk, jax.random.PRNGKey(9), o, d * 105.0, np.zeros(len(o)))
    r2 = FJ.transport(trk, jax.random.PRNGKey(9), o, d * 105.0, np.zeros(len(o)))
    for k in r1.steps:
        assert np.array_equal(np.asarray(r1.steps[k]), np.asarray(r2.steps[k]), equal_nan=True)
    dev = FJ.to_frame(r1, start)
    assert abs(len(dev) - len(ref)) < 5 * np.sqrt(len(ref)) * 0.2 + 10
    for c in ("len", "eion"):
        a, b = dev[c].to_numpy(), ref[c].to_numpy()
        se = np.hypot(a.std() / np.sqrt(len(a)), b.std() / np.sqrt(len(b)))
        assert abs(a.mean() - b.mean()) < 5 * se, c
    fin = FJ.final_frame(r1, start)
    pd_, pr = np.linalg.norm(fin[["px", "py", "pz"]].to_numpy(), axis=1), np.linalg.norm(rfin[["px", "py", "pz"]].to_numpy(), axis=1)
    assert abs(np.median(pd_) - np.median(pr)) < 0.05


def test_inactive_tracks_get_no_steps():
    B = np.array([0.0, 0.0, 1.0])
    _, _, trk = _setup(B)
    o, d = _lines(50, np.random.default_rng(10))
    act = np.arange(50) % 2 == 0
    res = FJ.transport(trk, jax.random.PRNGKey(11), o, d * 105.0, np.zeros(50), active=act)
    v = np.asarray(res.steps["valid"])
    assert not v[~act].any() and v[act].any()
    assert (np.asarray(res.final["status"])[~act] == FJ.LEFT).all()


def test_thin_materials_are_the_numpy_set():
    _, _, trk = _setup(np.array([0.0, 0.0, 1.0]))
    names = [m.name for m in trk.mats[-5:]]
    assert names == [materials.MYLAR.name, materials.ALUMINIUM_6061.name, trk.gas.name, materials.TUNGSTEN.name,
                     materials.GOLD.name]
