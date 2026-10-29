"""Population kernel isolation, routing, projection, and opt-in actual-fitter toy controls."""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import f1_seed_population_kernel as P

ARMS = ("ordinary", "control", "parameters", "momentum", "geometry_time", "covariance", "range", "full")
FIELDS = ("parameters", "covariance", "range")


@pytest.fixture
def inputs():
    x = np.array([30., -40., .2, .4, 12., 1500.])
    h = dict(mid=np.array([[20., 50., -30.], [40., -60., -50.], [0., 900., 0.]]),
             wdir=np.tile([1., 0., 0.], (3, 1)), valid=np.array([True, True, False]),
             inseed=np.array([True, True, False]))
    c = dict(mid=h["mid"].copy(), wdir=h["wdir"].copy(), valid=h["valid"].copy(),
             hit=np.array([0, 1, -1]))
    covariance = np.diag([9., 16., .04, .09, 4., 25.])
    covariance[0, 1], covariance[1, 0] = .25, np.nextafter(.25, 1.)
    native = dict(parameters=x + np.array([1., -2., .1, .2, 1., 200.]),
                  covariance=covariance, range=np.array([-8., 55.]))
    return dict(x=x, h=h, c=c, native=native, constraint_time=np.array(9.125))


def expected_seed(arm, base, native):
    if arm == "ordinary":
        return None
    result = {name: native[name] if arm in (name, "full") else base[name] for name in FIELDS}
    if arm in ("momentum", "geometry_time"):
        q = np.asarray(base["parameters"]).copy()
        slots = [5] if arm == "momentum" else list(range(5))
        q[slots] = np.asarray(native["parameters"])[slots]
        result["parameters"] = q
    return result


def exact_tree(actual, expected):
    left, tree = jax.tree_util.tree_flatten_with_path(actual)
    right, other_tree = jax.tree_util.tree_flatten_with_path(expected)
    assert tree == other_tree
    for (path, a), (other_path, b) in zip(left, right):
        assert path == other_path
        np.testing.assert_array_equal(a, b, err_msg=jax.tree_util.keystr(path))


def test_arm_inventory():
    assert P.ARMS == ARMS


@pytest.mark.parametrize("arm", ARMS)
def test_select_isolates_operands_without_mutation(inputs, arm):
    base = dict(parameters=np.arange(6., dtype=float), covariance=np.eye(6), range=np.array([1., 2.]))
    native = inputs["native"]
    before = deepcopy((base, native))
    actual = P.select(arm, base, native)
    exact_tree(actual, expected_seed(arm, base, native))
    if actual is not None:
        for name in FIELDS:
            if name == "parameters" and arm in ("momentum", "geometry_time"):
                continue
            assert actual[name] is (native if arm in (name, "full") else base)[name]
    exact_tree((base, native), before)
    assert native["covariance"][0, 1] != native["covariance"][1, 0]


@pytest.mark.parametrize("arm", P.S.ARMS)
def test_original_six_arms_delegate_to_sealed_select(monkeypatch, arm):
    sentinel, calls = object(), []
    base, native = object(), object()

    def selected(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(P.S, "select", selected)
    assert P.select(arm, base, native) is sentinel
    assert calls == [(arm, base, native)]


def test_ordinary_and_full_do_not_require_baseline(inputs):
    assert P.select("ordinary", None, None) is None
    exact_tree(P.select("full", None, inputs["native"]), inputs["native"])


@pytest.mark.parametrize("arm", ["", "parameter", "FULL", "geometry", None])
def test_unknown_selection_rejected(arm):
    with pytest.raises(ValueError, match="unknown seed arm"):
        P.select(arm, None, None)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("fixed_states", [False, True])
@pytest.mark.parametrize("maxniter", [3, 10])
def test_fit_routes_complete_result_and_stock_policies(inputs, monkeypatch, arm, fixed_states, maxniter):
    base = dict(parameters=np.arange(6.), covariance=np.eye(6), range=np.array([1., 2.]))
    baseline_calls, fit_calls = [], []
    result = ({"extra_cf": object()}, (object(),), object())
    drift, stages, ann = object(), object(), object()
    fixed = dict(active=object(), lr=object(), nulldv=object()) if fixed_states else None

    def baseline(*args):
        baseline_calls.append(args)
        return base

    def fitter(*args, **kwargs):
        fit_calls.append((args, kwargs))
        return result

    monkeypatch.setattr(P.S, "baseline", baseline)
    monkeypatch.setattr(P.K, "fit_track", fitter)
    actual = P.fit(arm, **inputs, drift=drift, stages=stages, ann=ann, fixed=fixed, maxniter=maxniter)
    assert actual is result
    assert len(baseline_calls) == (0 if arm in ("ordinary", "full") else 1)
    if baseline_calls:
        assert all(a is b for a, b in zip(baseline_calls[0], (inputs["x"], inputs["h"], inputs["c"])))
    assert len(fit_calls) == 1
    args, kwargs = fit_calls[0]
    assert len(args) == 6
    assert all(a is b for a, b in zip(args, (inputs["x"], inputs["h"], inputs["c"], drift, stages, ann)))
    exact_tree(kwargs.pop("seed_input"), expected_seed(arm, base, inputs["native"]))
    assert kwargs.pop("fixed") is fixed
    assert kwargs.pop("constraint_time") is inputs["constraint_time"]
    assert kwargs == dict(maxniter=maxniter, time_model="stock", reference_model="previous",
                          crossing_time="current", xing_activity="hit_or_doca")


@pytest.mark.parametrize("size", [5, 6])
@pytest.mark.parametrize("arm", ["control", "parameters", "momentum", "geometry_time", "covariance", "range"])
def test_baseline_recomputed_in_same_jit_with_dynamic_inputs(inputs, monkeypatch, size, arm):
    original, traced = P.S.baseline, []

    def baseline(x, h, c):
        assert isinstance(x, jax.core.Tracer)
        base = original(x, h, c)
        assert all(isinstance(base[name], jax.core.Tracer) for name in FIELDS)
        traced.append(True)
        return base

    def fitter(*args, **kwargs):
        return kwargs["seed_input"], kwargs["constraint_time"]

    monkeypatch.setattr(P.S, "baseline", baseline)
    monkeypatch.setattr(P.K, "fit_track", fitter)

    @jax.jit
    def caller(x, h, c, native, constraint_time):
        result = P.fit(arm, x, h, c, {}, {}, {}, native, constraint_time)
        return result, original(x, h, c)

    inputs["x"] = inputs["x"][:size]
    (actual, time), base = caller(**inputs)
    assert traced
    exact_tree(actual, expected_seed(arm, base, inputs["native"]))
    np.testing.assert_array_equal(time, inputs["constraint_time"])
    moved = dict(inputs, x=inputs["x"] + np.array([2., 1., .03, .02, 7., 5.])[:size])
    (changed, changed_time), new_base = caller(**moved)
    exact_tree(changed, expected_seed(arm, new_base, inputs["native"]))
    np.testing.assert_array_equal(changed_time, time)
    assert not np.array_equal(base["range"], new_base["range"])
    assert not np.array_equal(base["covariance"], new_base["covariance"])
    if arm in ("momentum", "geometry_time"):
        exact_tree(actual["covariance"], base["covariance"])
        exact_tree(changed["range"], new_base["range"])


@pytest.mark.parametrize("maxniter", [None, False, True, -1, 0, 1, 2, 4, 9, 11, 3., 10., "3", np.nan])
def test_invalid_cap_rejected_before_numerical_work(inputs, monkeypatch, maxniter):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid cap reached numerical work")

    monkeypatch.setattr(P.S, "baseline", forbidden)
    monkeypatch.setattr(P.K, "fit_track", forbidden)
    with pytest.raises(ValueError, match="maxniter must be 3 or 10"):
        P.fit("control", **inputs, drift={}, stages={}, ann={}, maxniter=maxniter)


@pytest.mark.parametrize("arm,time,error", [("unknown", 1., "unknown seed arm")]
                         + [(arm, None, "explicit constraint time") for arm in ARMS])
def test_arm_and_time_guards_before_numerical_work(inputs, monkeypatch, arm, time, error):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid wrapper call reached numerical work")

    monkeypatch.setattr(P.S, "baseline", forbidden)
    monkeypatch.setattr(P.K, "fit_track", forbidden)
    inputs["constraint_time"] = time
    with pytest.raises(ValueError, match=error):
        P.fit(arm, **inputs, drift={}, stages={}, ann={})


def test_default_cap_is_ten(inputs, monkeypatch):
    monkeypatch.setattr(P.K, "fit_track", lambda *args, **kwargs: kwargs["maxniter"])
    assert P.fit("ordinary", **inputs, drift={}, stages={}, ann={}) == 10


@pytest.mark.parametrize("good_value", [False, True])
def test_projection_preserves_histories_masks_and_rejected_values(monkeypatch, good_value):
    xt0, Ct0, kl = np.arange(6.), np.eye(6), np.arange(6.) + 100.
    cf = dict(st=dict(active=np.array([True, False]), lr=np.array([1., -1.])),
              xact=np.array([False, True, False]), chi2=-1000., ndof=-1000)
    hist = (np.array([0, 2]), np.array([1., np.nan]), np.array([8, -2]), np.array([3, 4]))
    good, calls = np.array(good_value), []

    def t0_piece(value):
        assert value is cf
        calls.append("t0")
        return xt0, Ct0, 9

    def kl_params(value):
        assert value is xt0
        calls.append("kl")
        return kl

    monkeypatch.setattr(P.K, "t0_piece", t0_piece)
    monkeypatch.setattr(P.K, "kl_params", kl_params)
    actual = P.project((cf, hist, good))
    expected = dict(xt0=xt0, Ct0=Ct0, kl=kl, good=good, stat=hist[0], niter=hist[3],
                    chi2=hist[1][-1], ndof=hist[2][-1], fact=cf["st"]["active"],
                    flr=cf["st"]["lr"], xact_fit=cf["xact"])
    exact_tree(actual, expected)
    for name in set(expected) - {"chi2", "ndof"}:
        assert actual[name] is expected[name]
    assert calls == ["t0", "kl"]


def test_projection_jits_and_vmaps_actual_t0_piece():
    q = jnp.array([20., -30., .2, .4, 10., 1500.])
    cf = dict(owner=dict(P=q[None], C=jnp.eye(6)[None], ranges=jnp.array([[-100., 100.]]),
                         count=jnp.array(1, dtype=jnp.int32)), st=dict(active=jnp.array([True, False]), lr=jnp.array([1., 0.])),
              xact=jnp.array([True]))
    hist = (jnp.array([0, 2]), jnp.array([1., 3.]), jnp.array([9, 8]), jnp.array([2, 5]))
    result = cf, hist, jnp.array(False)
    actual = jax.jit(P.project)(result)
    xt0, Ct0, _ = P.K.t0_piece(cf)
    np.testing.assert_array_equal(actual["xt0"], xt0)
    np.testing.assert_array_equal(actual["Ct0"], Ct0)
    np.testing.assert_allclose(actual["kl"], P.K.kl_params(xt0), rtol=1e-14, atol=1e-14)
    batched = jax.tree_util.tree_map(lambda v: jnp.stack((v, v)), result)
    projected = jax.jit(jax.vmap(P.project))(batched)
    for lane in range(2):
        exact_tree(jax.tree_util.tree_map(lambda v: v[lane], projected), actual)


@pytest.mark.slow
@pytest.mark.parametrize("fixed_states", [False, True])
@pytest.mark.parametrize("maxniter", [3, 10])
def test_actual_fitter_ordinary_control_complete_tree_exact(fixed_states, maxniter):
    from test_kalman_jax import _toy_track, _anns_available

    if not _anns_available():
        pytest.skip("ANN weights unavailable")
    x, _, h, c, _, drift = _toy_track(3)
    h = dict(h, inseed=jnp.arange(len(h["valid"])) < 15)
    x = jnp.asarray(x) + jnp.array([1., -1., .003, .002, .5])
    native = P.S.baseline(x, h, c)
    time = jnp.mean(native["range"]) + .2
    fixed = dict(active=h["active"], lr=h["lr"], nulldv=jnp.ones_like(h["active"])) if fixed_states else None
    stages, ann = P.K.stock_stages(), P.K.load_anns()

    def run(arm):
        return jax.jit(lambda p, n, t: P.fit(arm, p, h, c, drift, stages, ann, n, t,
                                            fixed=fixed, maxniter=maxniter))(x, native, time)

    ordinary, control = run("ordinary"), run("control")
    jax.block_until_ready((ordinary, control))
    exact_tree(control, ordinary)
    assert int(ordinary[0]["niter"]) > 0
    assert np.isfinite(np.asarray(ordinary[0]["xe"])).all()
    if maxniter == 10:
        assert bool(ordinary[2])


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["momentum", "geometry_time"])
def test_actual_failed_fit_preserves_q_split_and_baseline_covariance_range(arm):
    from test_kalman_jax import _toy_track, _anns_available

    if not _anns_available():
        pytest.skip("ANN weights unavailable")
    x, _, h, c, _, drift = _toy_track(7)
    h = dict(h, inseed=jnp.ones_like(h["valid"]))
    x = jnp.asarray(np.r_[x, P.K.MOM])
    initial = P.S.baseline(x, h, c)
    native = dict(parameters=initial["parameters"] + jnp.array([.2, -.1, .001, -.002, .3, .0002]),
                  covariance=initial["covariance"] * 1.1, range=initial["range"] + 1000.)
    time = jnp.mean(initial["range"]) + .2
    fixed = dict(active=jnp.zeros_like(h["active"]), lr=h["lr"], nulldv=jnp.ones_like(h["active"]))
    stages, ann = P.K.stock_stages(), P.K.load_anns()

    @jax.jit
    def run(p, n, t):
        return (P.fit(arm, p, h, c, drift, stages, ann, n, t, fixed=fixed),
                P.S.baseline(p, h, c))

    (cf, hist, good), base = run(x, native, time)
    expected = expected_seed(arm, base, native)
    assert not bool(good)
    assert int(cf["niter"]) == 1 and int(cf["status"]) == P.K.LOWNDOF
    np.testing.assert_array_equal(cf["owner"]["P"][0], expected["parameters"])
    np.testing.assert_array_equal(cf["owner"]["ranges"][0], base["range"])
    np.testing.assert_allclose(cf["owner"]["C"][0], base["covariance"], rtol=1e-13, atol=1e-13)
    assert np.all(np.asarray(hist[0]) == P.K.LOWNDOF)
