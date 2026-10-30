"""GPU batching preserves scalar-step/straw RNG domains and dense resource bounds."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from trkfastsim import digi_jax as dj, digi_clusters_jax as dc
from test_digi import cond


def test_grouped_charge_preserves_keys_and_float32_precision(cond, monkeypatch):
    monkeypatch.setattr(dj, "CHARGE_BLOCKS", 2)
    dj._charge_blocks_kernel.cache_clear()
    rng = np.random.default_rng(17)
    us = rng.integers(0, len(cond.straw_mid), 8192)
    pos = cond.straw_mid[us] + rng.normal(size=(8192, 3)) * .2
    ne = rng.integers(1, 4, 8192)
    tm = rng.uniform(1000, 1001, 8192)
    key = jax.random.key(91)
    try:
        grouped = dj._charge_blocks_kernel(cond)(key, jnp.int64(4096), *map(jnp.asarray, (us, pos, ne, tm)))
        chunks = [dj._charge_kernel(cond)(jax.random.fold_in(jax.random.fold_in(key, 1), 4096 + first),
                  *map(jnp.asarray, (us[first:first + 4096], pos[first:first + 4096], ne[first:first + 4096], tm[first:first + 4096])))
                  for first in (0, 4096)]
        for i in range(3):
            actual = np.asarray(grouped[i])
            expected = np.concatenate([np.asarray(x[i]) for x in chunks])
            bound = 2 * np.abs(np.spacing(expected))
            assert np.all(np.abs(actual - expected) <= bound), (i, np.max(np.abs(actual - expected)))
        offsets = jnp.array([4096, 8192], jnp.int64)
        grouped_keys = jax.jit(jax.vmap(lambda x: jax.random.fold_in(jax.random.fold_in(key, 1), x)))(offsets)
        keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(key, 1), x) for x in (4096, 8192)])
        np.testing.assert_array_equal(jax.random.key_data(grouped_keys), jax.random.key_data(keys))
    finally:
        dj._charge_blocks_kernel.cache_clear()


def test_sampling_batch_shape_does_not_change_clusters(cond, monkeypatch):
    from test_digi_clusters_jax import _table
    st = _table(cond, 137, eion=2e-3)
    st["width"][:] = .2
    st["ion"][::3] = 1
    idx = np.arange(137)
    key = jax.random.key(32)
    monkeypatch.setattr(dc, "BATCH_SIZE", 64)
    a = dc.make_clusters(key, cond, st, idx)
    monkeypatch.setattr(dc, "BATCH_SIZE", 256)
    monkeypatch.setattr(dc, "MAX_SAMPLE_LANES", 2048)
    b = dc.make_clusters(key, cond, st, idx)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_large_sampling_capacity_has_fixed_lane_bound():
    for capacity in 2 ** np.arange(13):
        n = dc._sample_batch(int(capacity))
        assert 1 <= n <= dc.BATCH_SIZE
        assert n * capacity <= max(dc.MAX_SAMPLE_LANES, capacity)
