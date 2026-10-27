"""Device world geometry, transport law, reproducibility and numpy isolation."""
import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from mu2efastsim import transport as tr
from trkfastsim import steps as st, world_jax as wj
from test_steps_jax import _box, _geometry, _gen


def _world():
    pieces = [q[1:] for q in [_box(1, "Al6061", [-100, 300, -100], [100, 350, 100]),
                              _box(1, "G4_Cu", [-80, 100, -80], [80, 120, 80])]]
    return tr.BandedWorld(pieces, [800.0, 250.0, 0.0])


def test_world_column_matches_numpy():
    world = _world()
    dev = wj.DeviceWorld(world, world.edges)
    rng = np.random.default_rng(21)
    o = rng.uniform([-200, 500, -200], [200, 800, 200], (257, 3))
    d = rng.normal(size=(257, 3))
    d[:, 1] = -np.abs(d[:, 1]) - 0.1
    d /= np.linalg.norm(d, axis=1)[:, None]
    for k, (top, bot) in enumerate(zip(world.edges[:-1], world.edges[1:])):
        origin, length, _ = wj._band_segment(jnp.asarray(o), jnp.asarray(d), top, bot, 0.0, jnp.ones(len(o), bool))
        got = np.asarray(dev.column(k, origin, jnp.asarray(d), length, jnp.ones(len(o), bool)))
        ref = world.band(k).column(np.asarray(origin), d, np.asarray(length))
        for m, v in ref.items():
            np.testing.assert_allclose(got[:, world.band(k).mats.index(m)], v, rtol=1e-10, atol=1e-8)
        zero = dev.column(k, origin, jnp.asarray(d), length, jnp.zeros(len(o), bool))
        assert not np.asarray(zero).any()


def _rays(n):
    rng = np.random.default_rng(22)
    o = np.c_[rng.uniform(-60, 60, n), np.full(n, 700.0), rng.uniform(-60, 60, n)]
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    return o, d


@pytest.mark.parametrize("sliced,spread,tail", [(True, False, True), (False, True, False)])
def test_world_reproducible(sliced, spread, tail):
    world = _world()
    dev = wj.DeviceWorld(world, world.edges)
    o, d = _rays(97)
    p = np.full(len(o), 400.0)
    opts = dict(sliced=sliced, spread=spread, tail=tail)
    a = dev.transport(jax.random.key(12), o, d, p, **opts)
    b = dev.transport(jax.random.key(12), o, d, p, **opts)
    c = dev.transport(jax.random.key(12), o, d, p, lanes=37, **opts)
    for x, y, z in zip(a, b, c):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_allclose(x, z, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(a[0][:, 1], 0.0, atol=1e-10)
    assert np.isfinite(a[0]).all()


def test_world_statistically_matches_numpy():
    world = _world()
    dev = wj.DeviceWorld(world, world.edges)
    o, d = _rays(12000)
    p = np.full(len(o), 220.0)
    rng = np.random.default_rng(23)
    ref = tr.world_loss(rng, o, d, p, world, world.edges, return_band_p=True, return_tail=True)
    pw, xx0, ok, bx, bh, bp, chi, tn = ref
    on, dn, ln = tr.msc_to_plane(rng, o, d, np.sqrt(p * np.maximum(pw, 1)), 0.0, 1.0, xx0, bx, bh,
                                 band_p=bp, band_chic2=chi, band_thn=tn)
    got = dev.transport(jax.random.key(23), o, d, p)
    for a, b in [(pw, got[2]), (on[:, 0], got[0][:, 0]), (dn[:, 0], got[1][:, 0]), (ln, got[3])]:
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        assert abs(a.mean() - b.mean()) <= 5 * se
    assert abs(ok.mean() - got[4].mean()) < 0.01
    assert abs(np.std(dn[:, 0]) / np.std(got[1][:, 0]) - 1) < 0.08


def test_empty_air_and_stopped():
    world = tr.BandedWorld([], [800.0, 250.0, 0.0])
    dev = wj.DeviceWorld(world, world.edges)
    o, d = _rays(2)
    p = np.array([0.01, 5000.0])
    out = dev.transport(jax.random.key(5), o, d, p, tail=False)
    assert not out[4][0] and out[4][1]
    ref = tr.world_loss(np.random.default_rng(5), o, d, p, world, world.edges)
    np.testing.assert_allclose(out[2], ref[0], rtol=1e-12)
    empty = dev.transport(jax.random.key(5), o[:0], d[:0], p[:0])
    assert all(len(q) == 0 for q in empty)


def test_world_requires_packed_order_and_matching_edges():
    world = _world()
    with pytest.raises(NotImplementedError, match="BandedWorld"):
        wj.DeviceWorld(world._pieces, world.edges)
    with pytest.raises(ValueError, match="edges"):
        wj.DeviceWorld(world, [800.0, 250.0, 1.0])


def test_to_plane_wrapper_and_cached_edge_guard(monkeypatch):
    geom = _geometry()
    edges = np.r_[st.BAND_EDGES_Y, st.plane_y(geom)]
    world = tr.BandedWorld([], edges)
    gen = _gen(100, np.random.default_rng(25))
    a = wj.to_plane(gen, geom, jax.random.key(2), world, sliced_msc=True)
    b = wj.to_plane(gen, geom, jax.random.key(2), world, sliced_msc=True)
    assert a[-1].any()
    for x, y in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(x, y)
    monkeypatch.setattr(st, "plane_y", lambda g: edges[-1] + 1.0)
    with pytest.raises(ValueError, match="cached world"):
        wj.to_plane(gen, geom, jax.random.key(2), world)


def test_make_steps_numpy_unchanged_and_jax_dispatch(monkeypatch):
    geom = _geometry()
    world = tr.BandedWorld([], np.r_[st.BAND_EDGES_Y, st.plane_y(geom)])
    gen = _gen(100, np.random.default_rng(24))
    a = st.make_steps(gen, geom, np.random.default_rng(1), stage="C", world=world, exit_scatter=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("numpy called device world")
    monkeypatch.setattr(wj, "to_plane", forbidden)
    b = st.make_steps(gen, geom, np.random.default_rng(1), stage="C", world=world, exit_scatter=False)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y, check_exact=True)
    with pytest.raises(AssertionError, match="numpy called device world"):
        st.make_steps(gen, geom, np.random.default_rng(1), stage="C", world=world, backend="jax")


def test_compacted_band_loss_preserves_fixed_lane_reference():
    world = _world()
    dev = wj.DeviceWorld(world, world.edges)
    rng = np.random.default_rng(28)
    n = 257
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(28), i))(jnp.arange(n))
    p = jnp.asarray(np.geomspace(0.01, 10000, n))
    for k in range(len(dev.bands)):
        _, mc, coeff, nm = dev.bands[k]
        col = jnp.asarray(rng.uniform(0, 200, (n, nm)) * (rng.random((n, nm)) < .2))
        seg = col.sum(1) + 50.0
        initial = jnp.zeros(n)
        a = wj._band_loss(mc, coeff, col, seg, p, initial, keys)
        b = wj._compact_band_loss(mc, coeff, col, seg, p, initial, keys)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=2e-12, atol=1e-9)
