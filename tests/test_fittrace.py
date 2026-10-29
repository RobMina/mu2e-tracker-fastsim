"""Actual-operand trace passivity, stopped-stage masking and allocation guards."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trkfastsim.reco import fittrace, kalman_jax as K


def test_allocation_guard():
    assert fittrace.allocation_bytes(3, 4, 2, 10) > 0
    with pytest.raises(ValueError, match="exceeding"):
        fittrace.empty(3, 4, 2, 10, max_bytes=1)
    with pytest.raises(ValueError, match="positive"):
        fittrace.allocation_bytes(3, 4, 2, 0)
    assert fittrace.allocation_bytes(3, 4, 2, 10, batch=2) == 2 * fittrace.allocation_bytes(3, 4, 2, 10)


@pytest.mark.parametrize("reject", [False, True])
def test_trace_is_passive_and_has_only_executed_iterations(reject):
    from test_kalman_jax import _toy_track, _anns_available
    if not _anns_available():
        pytest.skip("ANN weights unavailable")
    x, _, h, c, _, d = _toy_track(3)
    stages = {k: v[:3] for k, v in K.stock_stages().items()}
    if reject:
        stages["minndof"] = jnp.full_like(stages["minndof"], 10000)
    ann = K.load_anns()
    run = lambda trace: jax.jit(lambda p: K.fit_track(p, h, c, d, stages, ann,
        time_model="stock", maxniter=3, trace=trace))(jnp.asarray(x))
    plain, traced = run(False), run(True)
    cf, hist, good = traced
    records = cf.pop("trace")
    assert jax.tree_util.tree_structure(plain) == jax.tree_util.tree_structure((cf, hist, good))
    for a, b in zip(jax.tree_util.tree_leaves(plain), jax.tree_util.tree_leaves((cf, hist, good))):
        np.testing.assert_array_equal(a, b)
    valid = np.asarray(records["valid"])
    assert valid.sum() == int(cf["niter"])
    assert valid[0, 0]
    assert np.all(np.asarray(records["iteration"])[valid] < 3)
    if reject:
        assert not good
        assert valid.sum() == 1
        assert not np.any(records["swapped"])
        assert not np.any(valid[1:])
    else:
        assert np.any(records["swapped"])
    n, m = h["valid"].shape[0], c["valid"].shape[0]
    assert set(records) == set(fittrace.shapes(n, m))
    for name, shape in fittrace.shapes(n, m).items():
        assert records[name].shape == (3, 3) + shape
        assert records[name].dtype == np.dtype(fittrace.dtype(name))
    expected_seed = np.r_[x, K.MOM] if len(x) == 5 else x
    np.testing.assert_array_equal(records["owner_parameters"][0, 0, 0], expected_seed)
    np.testing.assert_array_equal(records["eff_xref"][0, 0],
        jnp.tile(jnp.asarray(expected_seed), (n + m + 1, 1)))
    assert np.any(np.asarray(records["eff_w"])[valid] > 0)
    for stage, iteration in np.argwhere(valid):
        order = np.asarray(records["order"][stage, iteration])
        physical = np.asarray(records["physical_time"][stage, iteration])
        ordering = np.asarray(records["ordering_time"][stage, iteration])
        assert np.all(np.diff(ordering[order]) >= 0)
        assert np.isfinite(physical).all()


def test_trace_rejects_legacy_before_fitting():
    with pytest.raises(ValueError, match="requires stock"):
        K.fit_track(None, None, None, None, None, None, trace=True)


@pytest.mark.parametrize("gate_fail", [False, True])
def test_extension_gate_and_optin_trace_passivity(gate_fail):
    from test_kalman_jax import _toy_track, _anns_available
    if not _anns_available():
        pytest.skip("ANN weights unavailable")
    x, _, h, c, _, d = _toy_track(3)
    if gate_fail:
        x = np.asarray(x).copy()
        x[0] += 3000
        h = dict(h, mid=h["mid"].at[:, 0].add(3000))
        c = dict(c, mid=c["mid"].at[:, 0].add(3000))
    h = dict(h, inseed=jnp.ones_like(h["valid"]))
    stages = {k: v[:6] for k, v in K.stock_stages().items()}
    ann = K.load_anns()
    kwargs = dict(time_model="stock", maxniter=3, reference_model="search", crossing_time="constructed")
    plain = jax.jit(lambda p: K.fit_track(p, h, c, d, stages, ann, **kwargs))(jnp.asarray(x))
    cf, hist, good = jax.jit(lambda p: K.fit_track(p, h, c, d, stages, ann, trace=True, **kwargs))(jnp.asarray(x))
    records = cf.pop("trace")
    traced = (cf, hist, good)
    assert jax.tree_util.tree_structure(plain) == jax.tree_util.tree_structure(traced)
    for a, b in zip(jax.tree_util.tree_leaves(plain), jax.tree_util.tree_leaves(traced)):
        np.testing.assert_array_equal(a, b)
    valid = np.asarray(records["valid"])
    assert np.any(valid[:4])
    assert np.any(valid[4:]) != gate_fail
    assert valid.sum() == int(cf["niter"])
    n = h["valid"].shape[0]
    for stage, iteration in np.argwhere(valid):
        order = np.asarray(records["order"][stage, iteration])
        refs = np.asarray(records["reference_parameters"][stage, iteration])
        np.testing.assert_array_equal(records["eff_xref"][stage, iteration], np.concatenate([refs, refs[:1]])[order])
        times = np.asarray(records["physical_time"][stage, iteration])
        query = np.asarray(records["reference_query_time"][stage, iteration])
        offset = np.asarray(records["crossing_offset"][stage, iteration])
        np.testing.assert_array_equal(query[n:-1], times[n:-1] - offset)


def test_vectorized_seed_membership_preserves_empty_duplicate_shuffled_rows():
    import awkward as ak
    from trkfastsim.reco.vr_f1 import _seed_membership
    hits = [[3, 1, 3, 5], [], [9, 4], [2]]
    seeds = [[5, 3, 3], [1], [], [3, 2]]
    expected = np.concatenate([np.isin(h, s) for h, s in zip(hits, seeds)])
    np.testing.assert_array_equal(_seed_membership(ak.Array(hits), ak.Array(seeds)), expected)
