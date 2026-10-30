"""Production digitiser shape bounds without dropping event or charge tails."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from trkfastsim import digi_jax as DJ
from test_digi import cond, _step, _steps


def test_jitter_fixed_shape_and_population_independence():
    key = jax.random.key(42)
    events = np.array([0, 127, 128, 49999, 128, 49999])
    panels = np.array([3, 215, 4, 0, 4, 215])
    before = DJ._jitter_block._cache_size()
    all_values = DJ._panel_jitter(key, events, panels, .5)
    after = DJ._jitter_block._cache_size()
    assert after - before <= 1
    assert all_values[2] == all_values[4]
    for first in (0, 2, 4):
        np.testing.assert_array_equal(all_values[first:first + 2],
            DJ._panel_jitter(key, events[first:first + 2], panels[first:first + 2], .5))
    assert DJ._jitter_block._cache_size() == after


def test_waveform_lane_budget():
    for cap in 2 ** np.arange(13):
        batch = DJ._waveform_batch(int(cap))
        assert 1 <= batch <= DJ.BATCH
        assert batch * cap <= max(DJ.WAVEFORM_CLUSTER_LANES, cap)


def test_sparse_fifty_thousand_events_no_population_jit(cond):
    st = _steps([_step(cond, 1000), _step(cond, 1100)], ev=[1, 49999])
    a = DJ.digitize(st, 50000, cond, jax.random.key(6))
    b = DJ.digitize(st, 50001, cond, jax.random.key(6))
    assert np.array_equal(a["ev"], [1, 49999])
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_batch_lane_budget_does_not_change_straw_rng(cond, monkeypatch):
    st = _steps([_step(cond, 1000), _step(cond, 1100), _step(cond, 1200)], ev=[0, 1, 2])
    a = DJ.digitize(st, 3, cond, jax.random.key(6))
    monkeypatch.setattr(DJ, "WAVEFORM_CLUSTER_LANES", 1)
    b = DJ.digitize(st, 3, cond, jax.random.key(6))
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_saturation_tiles_identical_noisy_output(cond, monkeypatch):
    st = _steps([_step(cond, 1000, eion=.04, ion=1)])
    first = DJ.digitize(st, 1, cond, jax.random.key(5))
    monkeypatch.setattr(DJ, "SATURATION_TILE", 7)
    DJ._electronics_kernel.cache_clear()
    try:
        tiled = DJ.digitize(st, 1, cond, jax.random.key(5))
        assert all(np.array_equal(first[k], tiled[k]) for k in first)
    finally:
        DJ._electronics_kernel.cache_clear()
