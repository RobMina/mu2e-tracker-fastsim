"""Upstream saved counts and local axial-orbit diagnostic preserve exact identities."""
import numpy as np

from mu2efastsim import ehelix
from trkfastsim.validation import ceupstream as cu


def test_axial_orbit_radial_max_matches_dense_exact_helix():
    x = np.array([[200.0, 0, 0], [0, 0, 0], [-100.0, 30, 0]])
    pv = np.array([[80.0, 10, 60], [100.0, 0, 0], [20.0, -80, 40]])
    bz = np.array([1.0, -1.0, 1.0])
    got = cu.axial_rmax(x, pv, bz, np.zeros(2))
    p = np.linalg.norm(pv, axis=1)
    phase = np.linspace(0, 2 * np.pi, 4096)
    xx = np.repeat(x, len(phase), axis=0)
    u = np.repeat(pv / p[:, None], len(phase), axis=0)
    pp = np.repeat(p, len(phase))
    b = np.repeat(np.c_[bz * 0, bz * 0, bz], len(phase), axis=0)
    length = (p[:, None] * phase / (ehelix.K * np.abs(bz[:, None]))).ravel()
    xn, _ = ehelix.helix(xx, u, pp, -1.0, b, length)
    ref = np.hypot(xn[:, 0], xn[:, 1]).reshape(3, -1).max(axis=1)
    np.testing.assert_allclose(got, ref, atol=1e-3)


def test_saved_chain_load_keeps_uint64_and_optional_counters(tmp_path):
    f = tmp_path / "chain.npz"
    ev = np.arange(2**54, 2**54 + 3, dtype=np.uint64)
    np.savez(f, run=np.array([1430] * 3), subrun=np.array([0] * 3), event=ev,
             fate=[1, 2, 1], nfoil=[1, 2, 3], nipa=[0, 1, 0], nopa=[0, 0, 0], path=[10, 20, 30], T=[100, 0, 90])
    d = cu.saved_chain(f)
    np.testing.assert_array_equal(d.event, ev)
    assert d.upstream_nfoil.tolist() == [1, 2, 3]
    assert "upstream_nring" not in d
