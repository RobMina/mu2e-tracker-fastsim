"""Synthetic seed-arm contracts; the fitter is mocked, never executed here.

Baseline operands stay inside the caller's JIT. Native transformed covariance
may have roundoff asymmetry: neither validation nor routing may symmetrize it.
"""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import f1_seed_arms as S
from trkfastsim.reco import kalman_jax as K


ARMS = ("ordinary", "control", "parameters", "covariance", "range", "full")
FIELDS = ("parameters", "covariance", "range")


@pytest.fixture
def inputs():
    x = np.array([30., -40., .2, .4, 12., 1500.])
    direction = np.array([x[2], -1., x[3]])
    direction /= np.linalg.norm(direction)
    point = np.array([x[0], 0., x[1]])

    def geometry(distances):
        distances = np.asarray(distances)
        return dict(mid=point + distances[:, None] * direction,
                    wdir=np.tile([1., 0., 0.], (len(distances), 1)))

    hits = dict(geometry([-120., 180., -900., 15000.]),
                valid=np.array([True, True, True, False]),
                inseed=np.array([True, True, False, True]))
    crossings = dict(geometry([-200., 260., 1200., -2000., 9000., -12000.]),
                     valid=np.array([True, True, True, True, False, False]),
                     hit=np.array([0, 1, 2, -1, 3, 0]))
    covariance = np.diag([9., 16., .04, .09, 4., 25.])
    covariance[0, 1] = .25
    covariance[1, 0] = np.nextafter(.25, 1.)
    native = dict(parameters=np.array([31., -42., .3, .6, 13., 1700.]),
                  covariance=covariance, range=np.array([-8., 55.]))
    return dict(x=x, hits=hits, crossings=crossings, native=native,
                constraint_time=np.array(9.125))


def expected_baseline(x, hits, crossings):
    """Stock fit_track's initial operands, evaluated in the same caller JIT."""
    q = jnp.concatenate([x, jnp.array([K.MOM])]) if x.shape[0] == 5 else x
    covariance = K.seed_covariance(q)
    seed_valid = hits["valid"] & (hits["inseed"] if "inseed" in hits else True)
    hit_s = K.straw_poca(q, hits["mid"], hits["wdir"])[0]
    hit_times = jnp.where(seed_valid, q[4] + hit_s / K.speed(q[5]), jnp.nan)
    crossing_s = (K.straw_poca(q, crossings["mid"], crossings["wdir"])[0]
                  + K.kkmaterial.STRAW["wrad"])
    crossing_valid = (crossings["valid"] & (crossings["hit"] >= 0)
                      & seed_valid[jnp.maximum(crossings["hit"], 0)])
    crossing_times = jnp.where(crossing_valid, q[4] + crossing_s / K.speed(q[5]), jnp.nan)
    times = jnp.concatenate([hit_times, crossing_times])
    return dict(parameters=q, covariance=covariance,
                range=jnp.stack([jnp.nanmin(times), jnp.nanmax(times)]))


def assert_seed_equal(actual, expected):
    assert set(actual) == set(FIELDS)
    for name in FIELDS:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-13, atol=2e-12)


def test_arm_inventory():
    assert S.ARMS == ARMS


@pytest.mark.parametrize("arm", ARMS)
def test_select_isolates_fields_without_copying_or_repairing(inputs, arm):
    native = inputs["native"]
    base = dict(parameters=np.arange(6., dtype=float), covariance=np.eye(6),
                range=np.array([1., 2.]))
    before = deepcopy((base, native))
    actual = S.select(arm, base, native)
    if arm == "ordinary":
        assert actual is None
    else:
        assert set(actual) == set(FIELDS)
        for name in FIELDS:
            source = native if arm in (name, "full") else base
            assert actual[name] is source[name]
    for current, saved in zip((base, native), before):
        for name in FIELDS:
            np.testing.assert_array_equal(current[name], saved[name])
    # This fixture intentionally distinguishes preservation from symmetrization.
    assert native["covariance"][0, 1] != native["covariance"][1, 0]


def test_select_ordinary_does_not_read_seed_operands():
    assert S.select("ordinary", None, None) is None


@pytest.mark.parametrize("arm", ["", "parameter", "FULL", None])
def test_select_rejects_unknown_arm(arm):
    with pytest.raises(ValueError, match="unknown seed arm"):
        S.select(arm, None, None)


@pytest.mark.parametrize("size", [5, 6])
@pytest.mark.parametrize("case", ["masked", "no_inseed", "single", "single_unpadded", "padded_partner"])
def test_baseline_matches_stock_expression_inside_caller_jit(inputs, size, case):
    x, hits, crossings = inputs["x"][:size], inputs["hits"], inputs["crossings"]
    if case == "no_inseed":
        del hits["inseed"]
    elif case in ("single", "single_unpadded"):
        hits["inseed"][:] = False
        hits["inseed"][1] = True
        crossings["valid"][:] = False
        if case == "single_unpadded":
            hits = {name: value[1:2] for name, value in hits.items()}
            crossings = {name: value[:1].copy() for name, value in crossings.items()}
            crossings["hit"][:] = -1
    elif case == "padded_partner":
        crossings["valid"][4] = True

    @jax.jit
    def caller(q, h, c):
        return S.baseline(q, h, c), expected_baseline(q, h, c)

    actual, expected = caller(x, hits, crossings)
    assert_seed_equal(actual, expected)
    assert np.all(np.isfinite(actual["covariance"]))
    assert np.all(np.isfinite(actual["range"]))
    if size == 5:
        assert float(actual["parameters"][5]) == K.MOM
    if case in ("single", "single_unpadded"):
        assert float(actual["range"][0]) == float(actual["range"][1])
    if case in ("masked", "padded_partner"):
        # Only seed-hit-linked, valid crossings extend this owner's endpoints.
        q = np.asarray(actual["parameters"])
        speed = q[5] / np.sqrt(q[5] ** 2 + K.MMU ** 2) * K.C
        expected_range = q[4] + (np.array([-200., 260.]) + K.kkmaterial.STRAW["wrad"]) / speed
        np.testing.assert_allclose(actual["range"], expected_range, rtol=2e-13, atol=2e-12)

    # Reuse the compiled caller with changed dynamic operands, not a host seed cache.
    moved = x.copy()
    moved[0] += 2.
    moved[2] += .07
    moved[4] += 11.
    changed, changed_expected = caller(moved, hits, crossings)
    assert_seed_equal(changed, changed_expected)
    assert not np.allclose(actual["covariance"], changed["covariance"])
    assert not np.allclose(actual["range"], changed["range"])


@pytest.mark.parametrize("size", [5, 6])
@pytest.mark.parametrize("inseed", [False, True])
def test_validate_accepts_valid_inputs_without_mutation(inputs, size, inseed):
    inputs["x"] = inputs["x"][:size]
    if not inseed:
        del inputs["hits"]["inseed"]
    inputs["native"]["range"][:] = 4.  # Closed degenerate range is valid.
    before = deepcopy(inputs)
    assert S.validate(**inputs) is None
    for actual, expected in zip(jax.tree_util.tree_leaves(inputs), jax.tree_util.tree_leaves(before)):
        np.testing.assert_array_equal(actual, expected)


INVALID_INPUTS = [
    (("x",), np.zeros(4), "baseline parameters"),
    (("x",), np.zeros((1, 6)), "baseline parameters"),
    (("x",), np.ones(6, dtype=int), "baseline parameters"),
    (("x",), np.array([1., 2., .2, .3, np.nan, 1000.]), "baseline parameters"),
    (("x",), np.array([1., 2., .2, .3, 4., np.inf]), "baseline parameters"),
    (("x",), np.array([1., 2., .2, .3, 4., 0.]), "baseline momentum"),
    (("x",), np.array([1., 2., .2, .3, 4., -1.]), "baseline momentum"),
    (("hits", "valid"), np.array([], dtype=bool), "hits valid mask"),
    (("hits", "valid"), np.ones(4, dtype=int), "hits valid mask"),
    (("hits", "valid"), np.ones((4, 1), dtype=bool), "hits valid mask"),
    (("crossings", "valid"), np.array([], dtype=bool), "crossings valid mask"),
    (("crossings", "valid"), np.ones(6, dtype=int), "crossings valid mask"),
    (("crossings", "valid"), np.ones((6, 1), dtype=bool), "crossings valid mask"),
    (("hits", "inseed"), np.zeros(4, dtype=bool), "seed"),
    (("hits", "inseed"), np.array([False, False, False, True]), "seed"),
    (("hits", "inseed"), np.ones(3, dtype=bool), "seed"),
    (("hits", "inseed"), np.ones(4, dtype=int), "seed"),
    (("crossings", "hit"), np.zeros(5, dtype=int), "crossing partner"),
    (("crossings", "hit"), np.zeros(6, dtype=float), "crossing partner"),
    (("crossings", "hit"), np.array([-2, 1, 2, -1, 3, 0]), "crossing partner"),
    (("crossings", "hit"), np.array([0, 1, 2, -1, 3, 4]), "crossing partner"),
    (("crossings", "hit"), np.array([3, 1, 2, -1, 3, 0]), "padded hit"),
    (("native", "parameters"), np.ones(5), "native parameters"),
    (("native", "parameters"), np.ones(6, dtype=int), "native parameters"),
    (("native", "parameters"), np.full(6, np.nan), "native parameters"),
    (("native", "parameters"), np.array([1., 2., .2, .3, 4., 0.]), "momentum/range/diagonal"),
    (("native", "parameters"), np.array([1., 2., .2, .3, 4., -1.]), "momentum/range/diagonal"),
    (("native", "covariance"), np.ones((5, 5)), "native covariance"),
    (("native", "covariance"), np.eye(6, dtype=int), "native covariance"),
    (("native", "covariance"), np.full((6, 6), np.inf), "native covariance"),
    (("native", "covariance"), np.diag([1., 1., 0., 1., 1., 1.]), "momentum/range/diagonal"),
    (("native", "covariance"), np.diag([1., 1., -1., 1., 1., 1.]), "momentum/range/diagonal"),
    (("native", "range"), np.zeros((1, 2)), "native range"),
    (("native", "range"), np.array([0, 1]), "native range"),
    (("native", "range"), np.array([np.nan, 1.]), "native range"),
    (("native", "range"), np.array([0., np.inf]), "native range"),
    (("native", "range"), np.array([2., 1.]), "momentum/range/diagonal"),
    (("constraint_time",), None, "constraint time"),
    (("constraint_time",), np.nan, "constraint time"),
    (("constraint_time",), np.inf, "constraint time"),
    (("constraint_time",), 1, "constraint time"),
    (("constraint_time",), np.array([1.]), "constraint time"),
]


@pytest.mark.parametrize("path,value,error", INVALID_INPUTS)
def test_validate_rejects_bad_values(inputs, path, value, error):
    target = inputs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=error):
        S.validate(**inputs)


@pytest.mark.parametrize("collection,count", [("hits", 4), ("crossings", 6)])
@pytest.mark.parametrize("field", ["mid", "wdir"])
@pytest.mark.parametrize("bad", ["shape", "integer", "nan", "inf", "masked"])
def test_validate_geometry_including_padding(inputs, collection, count, field, bad):
    value = inputs[collection][field].copy()
    if bad == "shape":
        value = np.zeros((count, 2))
    elif bad == "integer":
        value = value.astype(int)
    elif bad in ("nan", "inf"):
        value[-1, 0] = np.nan if bad == "nan" else np.inf
    else:
        value = np.ma.array(value, mask=np.zeros_like(value, dtype=bool))
        value.mask[-1, 0] = True
    inputs[collection][field] = value
    with pytest.raises(ValueError, match=collection + " " + field):
        S.validate(**inputs)


@pytest.mark.parametrize("field", FIELDS)
def test_validate_rejects_masked_native_values(inputs, field):
    inputs["native"][field] = np.ma.array(inputs["native"][field], mask=True)
    with pytest.raises(ValueError, match="native " + field):
        S.validate(**inputs)


@pytest.mark.parametrize("path", [("x",), ("hits", "valid"), ("hits", "inseed"), ("crossings", "valid"), ("crossings", "hit")])
def test_validate_rejects_masks_before_conversion(inputs, path):
    target = inputs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = np.ma.array(target[path[-1]], mask=True)
    with pytest.raises(ValueError, match="masked"):
        S.validate(**inputs)


@pytest.mark.parametrize("collection", ["hits", "crossings"])
@pytest.mark.parametrize("slot", [0, -1])
def test_validate_rejects_zero_wire_directions_even_in_padding(inputs, collection, slot):
    inputs[collection]["wdir"][slot] = 0.
    with pytest.raises(ValueError, match="zero .* wdir"):
        S.validate(**inputs)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_validate_requires_exact_native_fields(inputs, change):
    if change == "missing":
        del inputs["native"]["range"]
    else:
        inputs["native"]["constraint_time"] = inputs["constraint_time"]
    with pytest.raises(ValueError, match="native seed fields"):
        S.validate(**inputs)


@pytest.mark.parametrize("kind", ["asymmetric", "singular", "indefinite"])
def test_validate_rejects_bad_covariance_without_regularization(inputs, kind):
    covariance = np.eye(6)
    if kind == "asymmetric":
        covariance[0, 1] = .5
        error = "asymmetric native covariance"
    else:
        covariance[0, 1] = covariance[1, 0] = 1. if kind == "singular" else 2.
        error = "nonpositive native covariance"
    before = covariance.copy()
    inputs["native"]["covariance"] = covariance
    with pytest.raises(ValueError, match=error):
        S.validate(**inputs)
    np.testing.assert_array_equal(covariance, before)


def test_validate_rejects_masked_constraint_time(inputs):
    inputs["constraint_time"] = np.ma.array(1., mask=True)
    with pytest.raises(ValueError, match="constraint time"):
        S.validate(**inputs)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("with_fixed", [False, True])
def test_fit_routes_only_selected_operands_and_fixed_policies(inputs, monkeypatch, arm, with_fixed):
    base = dict(parameters=object(), covariance=object(), range=object())
    baseline_calls, fit_calls = [], []
    result = object()
    drift, stages, ann = object(), object(), object()
    fixed = {"active": np.array([True]), "lr": np.array([1.]),
             "nulldv": np.array([False])} if with_fixed else None

    def baseline(x, hits, crossings):
        baseline_calls.append((x, hits, crossings))
        return base

    def fitter(*args, **kwargs):
        fit_calls.append((args, kwargs))
        return result

    monkeypatch.setattr(S, "baseline", baseline)
    monkeypatch.setattr(K, "fit_track", fitter)
    actual = S.fit(arm, inputs["x"], inputs["hits"], inputs["crossings"], drift,
                   stages, ann, inputs["native"], inputs["constraint_time"], fixed=fixed)
    assert actual is result
    assert len(baseline_calls) == (0 if arm in ("ordinary", "full") else 1)
    if baseline_calls:
        assert all(a is b for a, b in zip(baseline_calls[0],
                   (inputs["x"], inputs["hits"], inputs["crossings"])))
    assert len(fit_calls) == 1
    args, kwargs = fit_calls[0]
    assert len(args) == 6
    assert all(a is b for a, b in zip(args,
               (inputs["x"], inputs["hits"], inputs["crossings"], drift, stages, ann)))
    seed = kwargs.pop("seed_input")
    assert kwargs.pop("fixed") is fixed
    assert kwargs.pop("constraint_time") is inputs["constraint_time"]
    assert kwargs == dict(maxniter=10, time_model="stock", reference_model="previous",
                          crossing_time="current", xing_activity="hit_or_doca")
    if arm == "ordinary":
        assert seed is None
    else:
        assert set(seed) == set(FIELDS)
        for name in FIELDS:
            source = inputs["native"] if arm in (name, "full") else base
            assert seed[name] is source[name]


@pytest.mark.parametrize("arm", ["control", "parameters", "covariance", "range"])
def test_fit_computes_baseline_operands_inside_caller_jit(inputs, monkeypatch, arm):
    original_baseline = S.baseline
    traced = []

    def traced_baseline(x, hits, crossings):
        assert isinstance(x, jax.core.Tracer)
        base = original_baseline(x, hits, crossings)
        assert all(isinstance(base[name], jax.core.Tracer) for name in FIELDS)
        traced.append(True)
        return base

    def fitter(*args, **kwargs):
        return kwargs["seed_input"], kwargs["constraint_time"]

    monkeypatch.setattr(S, "baseline", traced_baseline)
    monkeypatch.setattr(K, "fit_track", fitter)

    @jax.jit
    def caller(x, hits, crossings, native, constraint_time):
        actual, forwarded_time = S.fit(arm, x, hits, crossings, {}, {}, jnp.array([1.]),
                                       native, constraint_time)
        base = expected_baseline(x, hits, crossings)
        expected = {name: native[name] if name == arm else base[name] for name in FIELDS}
        return actual, expected, forwarded_time

    actual, expected, time = caller(**inputs)
    assert traced
    assert_seed_equal(actual, expected)
    np.testing.assert_array_equal(time, inputs["constraint_time"])
    moved = dict(inputs, x=inputs["x"] + np.array([2., 1., .03, .02, 7., 5.]))
    changed, expected_changed, changed_time = caller(**moved)
    assert_seed_equal(changed, expected_changed)
    np.testing.assert_array_equal(changed_time, time)
    if arm == "covariance":
        np.testing.assert_array_equal(changed["covariance"], inputs["native"]["covariance"])


@pytest.mark.parametrize("arm,time,error", [("bogus", 1., "unknown seed arm")]
                         + [(arm, None, "explicit constraint time") for arm in ARMS])
def test_fit_guards_run_before_baseline_or_fitter(inputs, monkeypatch, arm, time, error):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid wrapper call reached numerical work")

    monkeypatch.setattr(S, "baseline", forbidden)
    monkeypatch.setattr(K, "fit_track", forbidden)
    with pytest.raises(ValueError, match=error):
        S.fit(arm, inputs["x"], inputs["hits"], inputs["crossings"], {}, {}, None,
              inputs["native"], time)
