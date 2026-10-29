"""Slow synthetic full-schedule controls; not native-data acceptance or fidelity tests."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import f1_seed_arms as S
from test_kalman_jax import _toy_track, _anns_available


def exact_tree(actual, expected, label):
    left, tree = jax.tree_util.tree_flatten_with_path(actual)
    right, other_tree = jax.tree_util.tree_flatten_with_path(expected)
    assert tree == other_tree
    for (path, a), (other_path, b) in zip(left, right):
        assert path == other_path
        np.testing.assert_array_equal(a, b, err_msg=f"{label}: {jax.tree_util.keystr(path)}")


@pytest.mark.slow
@pytest.mark.skipif(not _anns_available(), reason="ANN weights unavailable")
@pytest.mark.parametrize("fixed_states", [False, True])
def test_full_schedule_seed_control_and_range_are_exact(fixed_states):
    x, _, h, c, _, d = _toy_track(3)
    h = dict(h, inseed=jnp.arange(len(h["valid"])) < 15)
    x = jnp.asarray(x)+jnp.array([1., -1., .003, .002, .5])
    initial = S.baseline(x, h, c)
    native = dict(initial, range=initial["range"]+1000.)
    constraint_time = jnp.mean(initial["range"])+.2
    S.validate(x, h, c, native, constraint_time)
    fixed = dict(active=h["active"], lr=h["lr"], nulldv=jnp.ones_like(h["active"])) if fixed_states else None
    stages, ann = S.K.stock_stages(), S.K.load_anns()
    def run(arm):
        return jax.jit(lambda p, n, t: S.fit(arm, p, h, c, d, stages, ann, n, t, fixed=fixed))(x, native, constraint_time)
    ordinary = run("ordinary")
    jax.block_until_ready(ordinary)
    assert bool(ordinary[2])
    assert int(ordinary[0]["niter"]) > 0
    assert np.isfinite(np.asarray(ordinary[0]["xe"])).all()
    for arm in ("control", "range"):
        actual = run(arm)
        jax.block_until_ready(actual)
        exact_tree(actual, ordinary, arm)


@pytest.mark.slow
@pytest.mark.skipif(not _anns_available(), reason="ANN weights unavailable")
def test_failed_fit_preserves_only_the_selected_seed_operands():
    x, _, h, c, _, d = _toy_track(7)
    h = dict(h, inseed=jnp.ones_like(h["valid"]))
    x = jnp.asarray(np.r_[x, S.K.MOM])
    initial = S.baseline(x, h, c)
    native = dict(parameters=initial["parameters"]+jnp.array([.2, -.1, .001, -.002, .3, .0002]),
                  covariance=initial["covariance"]*1.1, range=initial["range"]+1000.)
    constraint_time = jnp.mean(initial["range"])+.2
    S.validate(x, h, c, native, constraint_time)
    fixed = dict(active=jnp.zeros_like(h["active"]), lr=h["lr"], nulldv=jnp.ones_like(h["active"]))
    stages, ann = S.K.stock_stages(), S.K.load_anns()
    for arm in S.ARMS:
        cf, hist, good = jax.jit(lambda p, n, t: S.fit(arm, p, h, c, d, stages, ann, n, t, fixed=fixed))(x, native, constraint_time)
        expected = initial if arm == "ordinary" else S.select(arm, initial, native)
        assert not bool(good)
        assert int(cf["niter"]) == 1 and int(cf["status"]) == S.K.LOWNDOF
        np.testing.assert_array_equal(cf["owner"]["P"][0], expected["parameters"])
        np.testing.assert_array_equal(cf["owner"]["ranges"][0], expected["range"])
        np.testing.assert_allclose(cf["owner"]["C"][0], expected["covariance"], rtol=1e-13, atol=1e-13)
        assert np.all(np.asarray(hist[0]) == S.K.LOWNDOF)


@pytest.mark.slow
@pytest.mark.skipif(not _anns_available(), reason="ANN weights unavailable")
def test_batched_control_with_repeated_padding_lane():
    x, _, h, c, _, d = _toy_track(9)
    h = dict(h, inseed=jnp.arange(len(h["valid"])) < 15)
    x = jnp.stack((jnp.asarray(x), jnp.asarray(x)+jnp.array([1., -.2, .001, .002, .1]), jnp.asarray(x)))
    hj = jax.tree_util.tree_map(lambda v: jnp.broadcast_to(v, (3,)+v.shape), h)
    cj = jax.tree_util.tree_map(lambda v: jnp.broadcast_to(v, (3,)+v.shape), c)
    native = jax.vmap(S.baseline)(x, hj, cj)
    time = jnp.mean(native["range"], axis=1)+.2
    stages, ann = S.K.stock_stages(), S.K.load_anns()
    def run(arm):
        one = lambda p, hh, cc, n, t: S.fit(arm, p, hh, cc, d, stages, ann, n, t)
        return jax.jit(jax.vmap(one))(x, hj, cj, native, time)
    ordinary, control = run("ordinary"), run("control")
    exact_tree(control, ordinary, "batched control")
    exact_tree(jax.tree_util.tree_map(lambda v: v[0], ordinary),
               jax.tree_util.tree_map(lambda v: v[2], ordinary), "repeated padding lane")
