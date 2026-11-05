"""sim.fieldworld: hand-off surface, map stack, and the stop region of fieldsteps.transport (synthetic, no data files)."""
import numpy as np

from trkfastsim.sim import fieldsteps as fs, fieldworld as fw
from test_fieldsteps import _geometry, _no_loss, _start


def test_handoff_cut_side_and_front():
    h = fw.Handoff(rmax=100.0, z0=0.0, z1=500.0, axis_xy=(0.0, 0.0))
    x0 = np.array([[200.0, 0, 250], [0.0, 0, -100], [200.0, 0, 600], [50.0, 0, 250]])
    x1 = np.array([[0.0, 0, 250], [0.0, 0, 100], [0.0, 0, 600], [60.0, 0, 250]])
    s = h.cut(x0, x1)
    assert np.isclose(s[0], 0.5) and np.isclose(s[1], 0.5) and np.isnan(s[2]) and np.isnan(s[3])
    assert list(h.face(x0 + s[:, None].clip(0, 1) * (x1 - x0))[:2]) == [4, 0]


def test_mapstack_first_inner_then_outer_with_flip(tmp_path):
    def write(name, lo, n, d, B, extend=False):
        hdr = tmp_path / f"{name}.header"
        hdr.write_text("# x\nparam\ngrid X0=%g Y0=%g Z0=%g nX=%d nY=%d nZ=%d dX=%g dY=%g dZ=%g\n%sdata\n"
                       % (*lo, *n, *d, "extendY flip=By\n" if extend else ""))
        with open(tmp_path / f"{name}.bin", "wb") as f:
            np.array([0xDEADBEEF], np.uint32).tofile(f)
            np.broadcast_to(np.asarray(B, float), (*n, 3)).astype(np.float64).tofile(f)
        return str(hdr)
    inner = write("I", (-10, -10, -10), (3, 3, 3), (10, 10, 10), (0, 0, 1.0))
    outer = write("O", (-100, 0, -100), (3, 3, 3), (100, 100, 100), (0, 0.5, 2.0), extend=True)
    ms = fw.MapStack([inner], [outer])
    b = ms(np.array([[0.0, 0, 0], [50.0, 50, 0], [50.0, -50, 0], [500.0, 0, 0]]))
    assert np.allclose(b, [[0, 0, 1], [0, 0.5, 2], [0, -0.5, 2], [0, 0, 0]])


def test_transport_stops_on_the_handoff_surface():
    geom = _geometry()
    n = 50
    rng = np.random.default_rng(3)
    o = np.c_[rng.uniform(-50, 50, n), rng.uniform(-50, 50, n), np.full(n, -400.0)]
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    env = fw.Region(r=1000.0, z=(-1000.0, 1000.0), axis_xy=(0.0, 0.0))
    stop = fw.Handoff(rmax=300.0, z0=-100.0, z1=100.0, axis_xy=(0.0, 0.0))
    steps, fin = fs.transport(rng, _start(o, d, 1000.0), geom, np.array([0, 0, 1.0]), None, envelope=env,
                              origin=np.zeros(3), loss=_no_loss, msc=False, stop=stop, straws=False)
    assert len(steps) == 0 and (fin.status == 3).all()
    assert np.allclose(fin.z, -100.0, atol=1e-6) and np.allclose(fin.path, 300.0, atol=1e-6)
