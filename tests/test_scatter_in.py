"""TrackerInterior.transport(scatter_in=True): support scattering moves muons onto the straws as well as off them."""
import numpy as np
import pytest

from mu2efastsim.world import CoaxialShells
from trkfastsim import crossings, geometry
from trkfastsim.sim import steps as st

NSTRAW = 48


def _geometry():
    """Two layers of x-axis straws (y pitch 6 mm, layers at z = -2.5 and +2.5): a band |z| < 5.0 mm in z."""
    n = NSTRAW
    y = (np.arange(n) // 2) * 6.0 - 70.0
    z = np.where(np.arange(n) % 2 == 0, -2.5, 2.5)
    zeros = np.zeros(n, int)
    return geometry.StrawGeometry(
        plane=zeros, panel=zeros, straw=np.arange(n), mid=np.c_[np.zeros(n), y, z], dir=np.tile([1.0, 0, 0], (n, 1)),
        halflen=np.full(n, 450.0), radius=2.5085, panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=2.5235, metal_t=0.0001, wire_r=0.0125,
        wire_plate_t=0.0002)


def _interior():
    """One aluminium ring (r 600-640 mm about the z axis, |z| < 400 mm) above and below the straws."""
    shells = CoaxialShells([("Al6061", np.eye(3), np.zeros(3), 600.0, 640.0, 400.0, 0.0, 2 * np.pi)],
                           axis_point=np.zeros(3))
    it = st.TrackerInterior.__new__(st.TrackerInterior)
    it.shells, it.nsub = st.InteriorColumn(shells, None), 8
    return it


def _vertical(n, rng, zspan=150.0):
    """Vertical lines from y = 700 through the ring top, uniform in x (|x| < 300) and in z (|z| < zspan)."""
    o = np.c_[rng.uniform(-300, 300, n), np.full(n, 700.0), rng.uniform(-zspan, zspan, n)]
    return o, np.tile([0.0, -1.0, 0.0], (n, 1))


def _reach(rows, n):
    return len(np.unique(rows["it"])) if n else 0


def test_default_is_unchanged_by_the_switch():
    geom, interior = _geometry(), _interior()
    o, d = _vertical(500, np.random.default_rng(3), zspan=20.0)
    p = np.full(len(o), 300.0)
    a = interior.transport(np.random.default_rng([1, 2]), o, d, p, geom)
    b = interior.transport(np.random.default_rng([1, 2]), o, d, p, geom, scatter_in=False)
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _one_kink_reach(o, d, p, geom, rng):
    """Independent reference: each vertical line kinks once, at mid-depth of the ring top, by a Gaussian angle of the
    Highland width of its chord (no loss, no tail, no displacement), then goes straight; muons reaching the straws."""
    x = o[:, 0]
    y1, y0 = np.sqrt(640.0 ** 2 - x ** 2), np.sqrt(600.0 ** 2 - x ** 2)
    from mu2efastsim import eloss, materials
    from mu2efastsim import transport as tr
    X = (y1 - y0) / eloss.radiation_length_mm(materials.get("Al6061"))
    th = np.sqrt(tr.highland_var_increment(p, np.zeros(len(p)), X))[:, None] * rng.standard_normal((len(p), 2))
    start = np.c_[x, 0.5 * (y0 + y1), o[:, 2]]
    dn = np.c_[np.tan(th[:, 0]), -np.ones(len(p)), np.tan(th[:, 1])]
    dn /= np.linalg.norm(dn, axis=1)[:, None]
    return len(np.unique(crossings.straw_crossings(start, dn, geom)[:, 0]))


def test_uniform_illumination_matches_a_one_kink_reference():
    """Muons uniformly spread in z over a 10 mm band of straws, one ring above it: scattering in the ring moves as many
    lines onto the band as off it, and tilts them so they cover more of z. Without scatter_in a line that misses the
    band is never scattered onto it, so the reach falls short of the reference."""
    geom, interior = _geometry(), _interior()
    n = 12000
    o, d = _vertical(n, np.random.default_rng(11))
    p = np.full(n, 1000.0)                      # theta0 ~ 9 mrad in the ring, ~5 mm in z at the straws
    straight = len(np.unique(crossings.straw_crossings(o, d, geom)[:, 0]))
    ref = _one_kink_reach(o, d, p, geom, np.random.default_rng(12))
    legacy = interior.transport(np.random.default_rng([1, 5]), o, d, p, geom, tail=False)
    on = interior.transport(np.random.default_rng([1, 5]), o, d, p, geom, tail=False, scatter_in=True)
    n_leg, n_on = _reach(legacy, n), _reach(on, n)
    assert straight > 300 and ref > straight
    assert abs(n_on - ref) < 4 * np.sqrt(2 * ref)
    assert n_leg < ref - 6 * np.sqrt(2 * ref)
    assert interior.last_stats["scatter_in_kinks"] > 0


def test_scattered_in_rows_are_whole_straw_crossings():
    geom, interior = _geometry(), _interior()
    rng = np.random.default_rng(7)
    o, d = _vertical(3000, rng)
    o[:, 2] = np.where(o[:, 2] >= 0, 20.0, -20.0) + o[:, 2] * 0.2   # every straight line misses the band
    assert not len(crossings.straw_crossings(o, d, geom))
    tr = interior.transport(np.random.default_rng([2, 1]), o, d, np.full(len(o), 200.0), geom, scatter_in=True)
    assert len(tr["it"]) > 0
    for k in ("pos0", "pos1"):
        w = tr[k] - geom.mid[tr["js"]]
        r = np.hypot(w[:, 1], w[:, 2])                     # x-axis straws
        assert np.allclose(r, geom.radius, atol=1e-6)
    assert np.all(tr["path"] > 0) and np.all(np.diff(tr["path"])[np.diff(tr["it"]) == 0] > 0)
    assert np.all(tr["lost"] > 0)                          # every scattered-in muon crossed the ring first


def test_make_steps_refuses_unsupported_combinations():
    with pytest.raises(ValueError, match="scatter_in"):
        st.make_steps(None, _geometry(), np.random.default_rng(1), stage="A", scatter_in=True)
