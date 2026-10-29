"""Construction-offset isolation and default equivalence on synthetic fits."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from trkfastsim.reco import kalman_jax as K
from test_kalman_jax import _toy_track, _anns_available


@pytest.mark.skipif(not _anns_available(), reason="ANN weights unavailable")
def test_fixed_offsets_and_default_equivalence():
    xt, _, h, c, _, d = _toy_track(3)
    x = jnp.r_[jnp.asarray(xt), K.MOM]
    stages, ann = K.stock_stages(), K.load_anns()
    fit = lambda mode: jax.jit(lambda p: K.fit_track(p, h, c, d, stages, ann, time_model="stock", crossing_time=mode))(x)
    previous = jax.jit(lambda p: K.fit_track(p, h, c, d, stages, ann, time_model="stock"))(x)
    current = fit("current")
    for a, b in zip(jax.tree_util.tree_leaves(previous), jax.tree_util.tree_leaves(current)):
        np.testing.assert_array_equal(a, b)
    constructed = fit("constructed")
    np.testing.assert_allclose(constructed[0]["xing_offset"], np.full(len(c["valid"]), float(K.kkmaterial.STRAW["wrad"] / K.speed(x[5]))), rtol=3e-16)
    assert "xing_offset" not in current[0]
    assert bool(constructed[2])


def test_options_reject_nonstock_and_implicit_constraint():
    xt, _, h, c, _, d = _toy_track(3)
    x = jnp.asarray(xt)
    with pytest.raises(ValueError, match="require stock"):
        K.fit_track(x, h, c, d, K.stock_stages(), None, crossing_time="constructed")
    with pytest.raises(ValueError, match="explicit constraint"):
        K.fit_track(x, h, c, d, K.stock_stages(), None, time_model="stock", seed_input={})


@pytest.mark.skipif(not _anns_available(), reason="ANN weights unavailable")
def test_native_seed_injection_reproduces_explicit_reconstructed_operands():
    xt, _, h, c, _, d = _toy_track(3)
    x = jnp.r_[jnp.asarray(xt), K.MOM]
    s_h = K.straw_poca(x, h["mid"], h["wdir"])[0]
    s_x = K.straw_poca(x, c["mid"], c["wdir"])[0] + K.kkmaterial.STRAW["wrad"]
    t = jnp.r_[x[4] + s_h / K.speed(x[5]), x[4] + s_x / K.speed(x[5])]
    bounds = jnp.array([t.min(), t.max()])
    seed = {"parameters": x, "covariance": K.seed_covariance(x), "range": bounds}
    stages, ann = K.stock_stages(), K.load_anns()
    baseline = K.fit_track(x, h, c, d, stages, ann, time_model="stock", constraint_time=bounds.mean())
    supplied = K.fit_track(x, h, c, d, stages, ann, time_model="stock", constraint_time=bounds.mean(), seed_input=seed)
    for a, b in zip(jax.tree_util.tree_leaves(baseline), jax.tree_util.tree_leaves(supplied)):
        np.testing.assert_array_equal(a, b)


def test_extension_gate_freezes_once_and_counts_only_constructed_valid_searches():
    owner = {"P": jnp.array([[0., 0., 0., 0., .5, 100.], [0., 0., 0., 0., 1.5, 300.]]),
             "ranges": jnp.array([[0., 1.], [1., 2.]]), "count": jnp.array(1)}
    xe = jnp.tile(owner["P"][1], (5, 1))
    crs = {"mid": jnp.zeros((5, 3)),
           "wdir": jnp.array([[1., 0., 0.], [1., 0., 0.], [0., -1., 0.], [0., -1., 0.], [1., 0., 0.]]),
           "valid": jnp.array([True, True, True, False, False])}
    offsets = jnp.full(5, .0125)
    constructed = jnp.array([True, False, False, False, False])
    unsupported = jnp.zeros(5, bool)
    f = jax.jit(K.construct_crossings)
    stopped = f(owner, xe, crs, offsets, constructed, unsupported, False)
    for actual, expected in zip(stopped, (offsets, constructed, unsupported)):
        np.testing.assert_array_equal(actual, expected)
    result = f(owner, xe, crs, offsets, constructed, unsupported, True)
    np.testing.assert_array_equal(result[1], [True, True, True, False, False])
    np.testing.assert_array_equal(result[2], [False, False, True, False, False])
    np.testing.assert_array_equal(np.asarray(result[0])[[0, 3, 4]], [.0125] * 3)
    np.testing.assert_allclose(result[0][1], K.kkmaterial.STRAW["wrad"] / K.speed(300.), rtol=3e-16)
    changed_owner = dict(owner, P=owner["P"].at[:, 5].set(900.))
    retained = f(changed_owner, xe, crs, *result, True)
    for actual, expected in zip(retained, result):
        np.testing.assert_array_equal(actual, expected)
