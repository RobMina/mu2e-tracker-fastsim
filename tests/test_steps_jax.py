"""JAX backend (steps_jax) against the numpy reference on a synthetic tracker: kernels exactly, sampling statistically."""
import warnings

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mu2efastsim import eloss, materials  # noqa: E402
from mu2efastsim import transport as tr  # noqa: E402
from mu2efastsim.world import CoaxialShells, SignedPieceSet  # noqa: E402
from trkfastsim import crossings, geometry, steps as st, steps_jax as sj  # noqa: E402

warnings.filterwarnings("ignore", message=".*donated.*")
NSTRAW = 48


def _geometry():
    """Two layers of x-axis straws in one panel (y pitch 6 mm, layers 5 mm apart in z), Mu2e frame about the origin."""
    n = NSTRAW
    y = (np.arange(n) // 2) * 6.0 - 70.0
    z = np.where(np.arange(n) % 2 == 0, -2.5, 2.5)
    zeros = np.zeros(n, int)
    return geometry.StrawGeometry(
        plane=zeros, panel=zeros, straw=np.arange(n), mid=np.c_[np.zeros(n), y, z], dir=np.tile([1.0, 0, 0], (n, 1)),
        halflen=np.full(n, 450.0), radius=2.5085, panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=2.5235, metal_t=0.0001, wire_r=0.0125,
        wire_plate_t=0.0002)


def _geometry_panels(npanel=4):
    """_geometry with the straws split into `npanel` panels of one plane (consecutive straws), for the panel cut."""
    g = _geometry()
    panel = np.arange(NSTRAW) // (NSTRAW // npanel)
    return geometry.StrawGeometry(
        plane=g.plane, panel=panel, straw=np.arange(NSTRAW) % (NSTRAW // npanel), mid=g.mid, dir=g.dir,
        halflen=g.halflen, radius=g.radius, panel_R=np.repeat(g.panel_R, npanel, 0), panel_t=np.zeros((npanel, 3)),
        plane_R=g.plane_R, plane_t=g.plane_t, wall_rmax=g.wall_rmax, metal_t=g.metal_t, wire_r=g.wire_r,
        wire_plate_t=g.wire_plate_t)


def _box(sign, mat, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    P = np.array([[1, 0, 0, hi[0]], [-1, 0, 0, -lo[0]], [0, 1, 0, hi[1]], [0, -1, 0, -lo[1]], [0, 0, 1, hi[2]],
                  [0, 0, -1, -lo[2]]], float)
    return (sign, mat, P, lo, hi)


def _interior():
    """Coaxial shells about the z axis outside the straws (full rings, sectors sharing z-intervals) plus a signed
    calorimeter-like box set downstream."""
    I = np.eye(3)
    tubes = [("Al6061", I, np.array([0, 0, 0.0]), 600.0, 640.0, 200.0, 0.0, 2 * np.pi),
             ("G4_Cu", I, np.array([0, 0, 50.0]), 640.0, 660.0, 100.0, 0.3, 1.0),
             ("G10", I, np.array([0, 0, 50.0]), 640.0, 660.0, 100.0, 1.5, 0.8),
             ("G4_Cu", I, np.array([0, 0, -150.0]), 640.0, 660.0, 40.0, 4.0, 2.0)]
    tubes += [("Al7075", I, np.array([0, 0, z]), 660.0, 700.0, 20.0, ph, 0.6)
              for z in (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0) for ph in (0.0, 2.0, 4.0)]
    shells = CoaxialShells(tubes, axis_point=np.zeros(3))
    calo = SignedPieceSet([_box(1, "G4_CESIUM_IODIDE", [-300, -300, 500], [300, 300, 600]),
                           _box(-1, "G4_CESIUM_IODIDE", [-50, -50, 520], [50, 50, 580]),
                           _box(1, "G4_Cu", [-60, -60, 530], [60, 60, 540])], cell=100.0)
    it = st.TrackerInterior.__new__(st.TrackerInterior)
    it.shells, it.nsub = st.InteriorColumn(shells, calo), 8
    return it


@pytest.fixture(scope="module")
def setup():
    geom = _geometry()
    interior = _interior()
    return geom, interior, sj.DeviceInterior(interior, geom)


def _lines(rng, n, spread=0.35):
    """Downward lines from y = 720 (through the shells) aimed at the straw layers (|z| < 3 at y = 0)."""
    d = np.c_[rng.normal(0, spread, n), -np.ones(n), rng.normal(0, 0.01, n)]
    d /= np.linalg.norm(d, axis=1)[:, None]
    aim = np.c_[rng.uniform(-350, 350, n), np.zeros(n), rng.uniform(-3, 3, n)]
    return aim - d * (720.0 / -d[:, 1])[:, None], d


def _segments(rng, n):
    o = rng.uniform([-800, -800, -400], [800, 800, 700], (n, 3))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return o, d, rng.uniform(0, 1500, n)


@pytest.mark.parametrize("moments", [False, True])
def test_column_matches_numpy(setup, moments):
    geom, interior, dev = setup
    o, d, L = _segments(np.random.default_rng(3), 3000)
    ref = interior.shells.column(o, d, L, nsub=interior.nsub, moments=moments)
    got = dev.column(jnp.asarray(o), jnp.asarray(d), jnp.asarray(L), jnp.ones(len(L), bool), moments=moments)
    got = [np.asarray(x) for x in (got if moments else (got,))]
    assert set(ref) <= set(dev.sup_names)
    nz = 0
    for m in dev.sup_names:
        i = dev.sup_names.index(m)
        r = ref.get(m)
        r = [np.zeros(len(L))] * len(got) if r is None else ([np.asarray(r)] if not moments else [np.asarray(x) for x in r])
        for k, (a, b) in enumerate(zip(got, r)):
            assert np.allclose(a[:, i], b, rtol=1e-9, atol=1e-6 * (1 + 1000.0 ** k)), (m, k)
        nz += int((r[0] > 0).sum())
    assert nz > 500                                            # the segments do cross the material


def test_column_zero_for_invalid_and_empty(setup):
    geom, interior, dev = setup
    o, d, L = _segments(np.random.default_rng(4), 64)
    v = np.zeros(64, bool)
    got = np.asarray(dev.column(jnp.asarray(o), jnp.asarray(d), jnp.asarray(L), jnp.asarray(v)))
    assert not got.any()


def test_split_length_matches_numpy(setup):
    geom, interior, dev = setup
    o, d, L = _segments(np.random.default_rng(5), 2000)
    x = dev.xsup(dev.column(jnp.asarray(o), jnp.asarray(d), jnp.asarray(L), jnp.ones(len(L), bool)))
    x = np.asarray(x)
    sel = x > 0.05
    o, d, L, x = o[sel], d[sel], L[sel], x[sel]
    target = x / np.ceil(x / 0.25) * (1 - 1e-6)            # off the ties where a probe holds exactly the target
    ref = interior._split_length(o, d, L, target)
    got = np.asarray(dev.split_length(jnp.asarray(o), jnp.asarray(d), jnp.asarray(L), jnp.asarray(target),
                                      jnp.ones(len(L), bool)))
    assert len(L) > 100 and np.allclose(got, ref, rtol=1e-12)


def test_straight_crossings_match_numpy(setup):
    geom, _, _ = setup
    o, d = _lines(np.random.default_rng(6), 4000)
    ref = crossings.straw_crossings(o, d, geom)
    got = sj.straw_crossings(geom, o, d)
    assert len(ref) > 1000 and ref.shape == got.shape
    assert np.array_equal(ref[:, :2], got[:, :2]) and np.allclose(ref[:, 2:], got[:, 2:], rtol=0, atol=1e-8)


def test_exit_gas_backwards_matches_numpy(setup):
    geom, _, dev = setup
    rng = np.random.default_rng(7)
    j = rng.integers(0, NSTRAW, 3000)
    pos = geom.mid[j] + np.c_[rng.uniform(-400, 400, 3000), rng.uniform(-3, 3, (3000, 2))]
    d = rng.normal(size=(3000, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    ref = st._exit_gas_backwards(pos, d, geom)
    got, over = sj._exit_gas_backwards(dev.T, dev.cfg, jnp.asarray(pos), jnp.asarray(d))
    assert int(over) == 0 and np.allclose(np.asarray(got), ref, rtol=0, atol=1e-9)
    assert (np.abs(ref - pos).max(1) > 0).sum() > 500


def _numpy_slab(mat, p, L, seed):
    return tr.slab_loss(np.random.default_rng(seed), mat, p, L)


@pytest.mark.parametrize("name,p,L", [("Al6061", 300.0, 50.0), ("G4_Cu", 200.0, 60.0), ("G4_Cu", 3000.0, 100.0)])
def test_slab_loss_statistically_equal(setup, name, p, L):
    _, _, dev = setup
    n = 20000
    i = dev.sup_names.index(name)
    col = np.zeros((n, len(dev.sup_names)))
    col[:, i] = L
    keys = jax.vmap(lambda k: jax.random.fold_in(jax.random.key(11), k))(jnp.arange(n, dtype=jnp.uint32))
    got = np.asarray(dev.slab(jnp.asarray(col), jnp.full(n, p), keys))
    mat = materials.get(name)
    ref = _numpy_slab(mat, np.full(n, p), np.full(n, L), 2)
    tk = np.sqrt(p * p + eloss.MMU ** 2) - eloss.MMU
    fs_r, fs_g = np.mean(ref >= tk - 1e-9), np.mean(got >= tk - 1e-9)
    assert abs(fs_r - fs_g) < 5 * np.sqrt(max(fs_r * (1 - fs_r), 1e-4) * 2 / n)
    a, b = ref[ref < tk - 1e-9], got[got < tk - 1e-9]
    assert abs(a.mean() / b.mean() - 1) < 0.01
    qa, qb = np.quantile(a, [0.1, 0.5, 0.9]), np.quantile(b, [0.1, 0.5, 0.9])
    assert np.all(np.abs(qa / qb - 1) < 0.02)


def test_group_tail_statistically_equal(setup):
    _, _, dev = setup
    n = 200000
    rng = np.random.default_rng(8)
    M = dev.cfg.m_all
    mats = dev.sup + list(dev.THIN)
    L = np.zeros((n, M))
    L[:, dev.sup_names.index("Al6061")] = 2.0
    L[:, dev.cfg.m_sup + 0] = 0.05        # Mylar
    L[:, dev.cfg.m_sup + 2] = 5.0         # gas
    p = np.full(n, 400.0)
    X = sum(L[:, k] / eloss.radiation_length_mm(m) for k, m in enumerate(mats))
    a = np.sqrt(tr.highland_var_increment(p, np.zeros(n), X))
    parts = [(m, L[:, k]) for k, m in enumerate(mats) if L[:, k].any()]
    ref = st._group_tail(rng, parts, np.arange(n), p, p, a, None)
    keys = jax.vmap(lambda k: jax.random.fold_in(jax.random.key(5), k))(jnp.arange(n, dtype=jnp.uint32))
    got, over = sj._group_tail(dev.T, keys, jnp.asarray(L), jnp.asarray(p), jnp.asarray(a), None)
    got = np.asarray(got)
    assert int(over) == 0
    fr, fg = np.mean(np.any(ref != 0, 1)), np.mean(np.any(got != 0, 1))
    assert fr > 0.01 and abs(fr - fg) < 4 * np.sqrt(fr * (1 - fr) * 2 / n)
    tr_, tg = np.hypot(*ref[np.any(ref != 0, 1)].T), np.hypot(*got[np.any(got != 0, 1)].T)
    assert abs(np.median(tr_) / np.median(tg) - 1) < 0.05


def test_transport_rows_match_numpy_at_high_momentum(setup):
    """At |p| = 1e12 MeV/c without the tail (whose rate does not fall with |p|) the core scattering is ~1e-11 rad: the jax
    rows, kinks and splits follow the numpy ones."""
    geom, interior, dev = setup
    o, d = _lines(np.random.default_rng(9), 300)
    p = np.full(len(o), 1e12)
    ref = interior.transport(np.random.default_rng(1), o, d, p, geom, tail=False)
    got, stats, kinks, final = dev.transport(jax.random.key(1), o, d, p, tail=False)
    for k in ("rec_overflow", "kink_overflow", "row_overflow", "point_overflow", "tail_overflow"):
        assert stats[k] == 0, k
    assert len(ref["it"]) > 500 and np.array_equal(ref["it"], got["it"]) and np.array_equal(ref["js"], got["js"])
    for k in ("pos0", "pos1", "dir"):
        assert np.allclose(ref[k], got[k], rtol=0, atol=1e-6), k
    # the path to a straw also counts where the kinks sat: a split probe or a gas back-off on a boundary tie (float
    # rounding of the column) moves a kink along the line, not the line
    tie = np.unique(ref["it"][np.abs(ref["path"] - got["path"]) > 1e-6])
    assert len(tie) <= 0.02 * len(o)
    assert stats["kinks"] == interior.last_stats["kinks"] and stats["support_splits"] == interior.last_stats["support_splits"]
    a, b = interior.last_kinks, kinks
    oa, ob = np.lexsort((a["path"], a["it"])), np.lexsort((b["path"], b["it"]))
    assert np.array_equal(a["it"][oa], b["it"][ob])
    ok = ~np.isin(a["it"][oa], tie)
    assert np.allclose(a["pos"][oa][ok], b["pos"][ob][ok], atol=1e-6)


def test_transport_reproducible_and_chunk_independent(setup):
    """Bit-identical for one key and chunking; another chunking changes only the float rounding of vectorised math."""
    geom, interior, dev = setup
    o, d = _lines(np.random.default_rng(10), 200)
    p = np.random.default_rng(11).uniform(150, 3000, len(o))
    a = dev.transport(jax.random.key(3), o, d, p)[0]
    a2 = dev.transport(jax.random.key(3), o, d, p)[0]
    b = dev.transport(jax.random.key(3), o, d, p, lanes=37)[0]
    c = dev.transport(jax.random.key(4), o, d, p)[0]
    assert len(a["it"]) > 1000
    for k in a:
        assert np.array_equal(a[k], a2[k]), k
    assert np.array_equal(a["it"], b["it"]) and np.array_equal(a["js"], b["js"])
    for k in ("pos0", "pos1", "path", "lost"):
        assert np.allclose(a[k], b[k], rtol=1e-12, atol=1e-8), k
    assert not np.array_equal(a["pos0"], c["pos0"])


def _mean_err(x):
    return x.mean(), x.std() / np.sqrt(len(x))


def test_transport_statistically_like_numpy(setup):
    """Slow muons (scattering, support loss, splits): muons with straws, straws per muon, the support loss before the
    first straw and the |p| at the end, as numpy within 4 sigma."""
    geom, interior, dev = setup
    o, d = _lines(np.random.default_rng(12), 8000, spread=0.2)
    p = np.full(len(o), 250.0)
    ref = interior.transport(np.random.default_rng(2), o, d, p, geom)
    fin_r = dict(interior.last_final)
    got, stats, _, fin_g = dev.transport(jax.random.key(2), o, d, p)
    assert stats["support_splits"] > 5000 and stats["row_overflow"] == 0
    nr, ng = np.bincount(ref["it"], minlength=len(o)), np.bincount(got["it"], minlength=len(o))
    (m_r, e_r), (m_g, e_g) = _mean_err(nr[nr > 0]), _mean_err(ng[ng > 0])
    assert abs(m_r - m_g) < 4 * np.hypot(e_r, e_g)
    f_r, f_g = np.mean(nr > 0), np.mean(ng > 0)
    assert abs(f_r - f_g) < 4 * np.sqrt(f_r * (1 - f_r) * 2 / len(o))
    first = lambda r: np.r_[True, r["it"][1:] != r["it"][:-1]]
    (l_r, el_r), (l_g, el_g) = _mean_err(ref["lost"][first(ref)]), _mean_err(got["lost"][first(got)])
    assert abs(l_r - l_g) < 4 * np.hypot(el_r, el_g)
    (p_r, ep_r), (p_g, ep_g) = _mean_err(fin_r["p"]), _mean_err(fin_g["p"])
    assert abs(p_r - p_g) < 4 * np.hypot(ep_r, ep_g)


def _gen(n, rng):
    o, d = _lines(rng, n)
    p = rng.uniform(300, 5000, n)
    return pd.DataFrame({"run": 1, "subrun": 1, "event": np.arange(n), "pdg": 13, "x": o[:, 0], "y": o[:, 1],
                         "z": o[:, 2], "px": d[:, 0] * p, "py": d[:, 1] * p, "pz": d[:, 2] * p, "t": 10.0})


def test_make_steps_jax_columns_and_reproducibility(setup):
    geom, interior, _ = setup
    gen = _gen(400, np.random.default_rng(13))
    kw = dict(stage="A", interior=interior, exit_scatter=False)
    a, pa = st.make_steps(gen, geom, np.random.default_rng(1), **kw)
    b, pb = st.make_steps(gen, geom, np.random.default_rng(1), backend="jax", **kw)
    c, _ = st.make_steps(gen, geom, np.random.default_rng(1), backend="jax", **kw)
    assert list(a.columns) == list(b.columns) and (a.dtypes == b.dtypes).all()
    pd.testing.assert_frame_equal(b, c)
    pd.testing.assert_frame_equal(pa, pb)
    assert abs(len(a) / len(b) - 1) < 0.05
    assert abs(np.median(a.eion) / np.median(b.eion) - 1) < 0.05


def test_make_steps_stage_a_jax_same_crossings(setup):
    geom, _, _ = setup
    gen = _gen(300, np.random.default_rng(14))
    a, _ = st.make_steps(gen, geom, np.random.default_rng(1))
    b, _ = st.make_steps(gen, geom, np.random.default_rng(1), backend="jax")
    cols = ["event", "sid", "len", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz", "ion"]
    pd.testing.assert_frame_equal(a[cols], b[cols], rtol=1e-12, atol=1e-9)
    assert not np.array_equal(a.eion.to_numpy(), b.eion.to_numpy())


def test_jax_rejects_unsupported_options(setup):
    geom, interior, _ = setup
    gen = _gen(10, np.random.default_rng(15))
    with pytest.raises(NotImplementedError):
        st.make_steps(gen, geom, np.random.default_rng(1), stage="A", interior=interior, lever_moments=True, backend="jax")
    with pytest.raises(ValueError):
        st.make_steps(gen, geom, np.random.default_rng(1), backend="cuda")


def test_jaxcheck_compare_and_cross(setup):
    from trkfastsim.analyses import jaxcheck
    geom, interior, _ = setup
    gen = _gen(400, np.random.default_rng(16))
    kw = dict(stage="A", interior=interior, exit_scatter=False)
    a, _ = st.make_steps(gen, geom, np.random.default_rng(2), **kw)
    b, _ = st.make_steps(gen, geom, np.random.default_rng(2), backend="jax", **kw)
    c = jaxcheck.compare(a, b, gen)
    assert c["steps"] == [len(a), len(b)] and c["primaryfilter_trk"]["events"] == len(gen)
    assert abs(c["straws_per_muon"]["diff_sigma"]) < 5
    x = jaxcheck.cross_backend(b, b.copy(), gen)
    assert x["identical_frames"] and x["matched"] == len(b) and x["only_this"] == x["only_cross"] == 0


def test_panel_cut_rows_exact(monkeypatch):
    """Lines through more panel boxes than P_CAP: the cut lanes are redone with every box, so the rows equal those of
    a pass that takes every box from the start (same straws; positions to float rounding of other array shapes)."""
    geom = _geometry_panels(4)
    interior = _interior()
    o, d = _lines(np.random.default_rng(17), 300)
    p = np.full(len(o), 1e12)
    ref = sj.DeviceInterior(interior, geom).transport(jax.random.key(1), o, d, p, tail=False)
    monkeypatch.setattr(sj, "P_CAP", 1)
    got = sj.DeviceInterior(interior, geom).transport(jax.random.key(1), o, d, p, tail=False)
    assert got[1]["panel_cut"] > 100 and got[1]["row_overflow"] == 0
    assert len(ref[0]["it"]) > 500
    assert np.array_equal(ref[0]["it"], got[0]["it"]) and np.array_equal(ref[0]["js"], got[0]["js"])
    for k in ("pos0", "pos1", "dir", "path", "lost"):
        assert np.allclose(ref[0][k], got[0][k], rtol=1e-12, atol=1e-9), k
