"""Support-slice early completion preserves every physical update and ignores padded work."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from mu2efastsim import eloss_jax, materials, transport as tr
from trkfastsim import steps_jax as sj


def _inputs():
    cs = [eloss_jax.consts(m) for m in [materials.ALUMINIUM_6061, materials.get("G4_Cu")]]
    mc = eloss_jax.MatConsts(*(jnp.asarray([getattr(c, f) for c in cs]) for f in eloss_jax.MatConsts._fields))
    n = 64
    mat = jnp.arange(n) % 2
    p = jnp.asarray(np.geomspace(1, 5000, n))
    length = jnp.asarray(np.tile([0, 0.01, 1, 1000], n // 4))
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(19), i))(jnp.arange(n))
    return mc, mat, p, length, keys


@pytest.mark.parametrize("rounds,start", [(1, 0), (8, 0), (16, 0), (8, tr.N_SLICES_MAX - 2)])
def test_rounds_equal_fixed_loop(rounds, start):
    mc, mat, p, length, keys = _inputs()
    carry = (jnp.full(len(p), start, jnp.int32), length, jnp.zeros_like(p))
    c = eloss_jax.MatConsts(*(f[mat] for f in mc))
    old = jax.jit(jax.vmap(lambda k, cc, pp, ll, i, lf, lo: sj._slab_one(k, cc, pp, ll, rounds, (i, lf, lo))))
    ref = old(keys, c, p, length, *carry)
    got, _ = sj._slab_rounds(mc, mat, p, length, keys, carry, rounds)
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(sj._slab_finish(mc, mat, p, ref), sj._slab_finish(mc, mat, p, got))


def test_padding_not_pending():
    mc, mat, p, length, keys = _inputs()
    idx = jnp.asarray([63, sj._NBIG, sj._NBIG, sj._NBIG])
    carry = (jnp.zeros(len(p), jnp.int32), length, jnp.zeros_like(p))
    got, todo, nt = sj._slab_iter(mc, mat, p, length, keys, carry, idx, 1)
    assert int(nt) == 1
    np.testing.assert_array_equal(todo, [True, False, False, False])
    for a, b in zip(carry, got):
        np.testing.assert_array_equal(a[:63], b[:63])
