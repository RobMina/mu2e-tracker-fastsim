"""Small explicit expanded-sample reference; production has no event/draw loops."""
import json

import numpy as np
import pytest

from trkfastsim.reco import autonomous_stats as S


def keys(n):
    return np.column_stack((np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64), np.arange(n)))


def reference(event_keys, owners, values, replicates, seed):
    _, inverse = np.unique(event_keys, axis=0, return_inverse=True)
    owners = inverse.ravel()[owners]
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = rng.integers(len(event_keys), size=(replicates, len(event_keys)))
    out = []
    for sample in samples:
        expanded = np.concatenate([values[owners == event] for event in sample])
        expanded = expanded[np.isfinite(expanded)]
        if len(expanded):
            q = np.quantile(expanded, [0.16, 0.5, 0.84], method="linear")
            out.append([*q, (q[2] - q[0]) / 2])
    return np.asarray(out).reshape(-1, 4)


def test_multitrack_ties_empty_events_and_full_identity_against_reference():
    # Event numbers repeat: run AND subrun distinguish the sampling units.
    k = np.array([[2, 0, 7], [1, 1, 7], [1, 0, 7], [2, 1, 7], [3, 0, 7]])
    owners = np.array([0, 0, 0, 1, 2, 2])
    values = np.array([-7., 2., 2., 11., 2., 19.])
    expected = reference(k, owners, values, 401, 52)
    result = S.event_cluster_bootstrap(k, owners, {"x": values}, replicates=401, seed=52)
    row = result["variables"]["x"]
    assert result["events"] == 5
    assert result["events_with_matched_tracks"] == 3
    assert row["effective_events"] == 3
    assert row["bootstrap_empty_replicates"] == 401 - len(expected) > 0
    q = np.quantile(values, [.16, .5, .84], method="linear")
    for j, name in enumerate(("q16", "q50", "q84", "central_width")):
        stat = row["statistics"][name]
        assert stat["estimate"] == pytest.approx([*q, (q[2] - q[0]) / 2][j])
        assert stat["bootstrap_se"] == pytest.approx(expected[:, j].std(ddof=1))
        assert stat["ci95"] == pytest.approx(np.quantile(expected[:, j], [.025, .975], method="linear"))
    json.dumps(result, allow_nan=False)


def test_frequency_quantiles_are_type7_not_inverse_weighted_cdf():
    values = np.array([0., 10., 10., 30.])
    owners = np.array([0, 0, 1, 2])
    counts = np.array([[1, 0, 1], [2, 3, 0], [0, 0, 0], [0, 0, 1]])
    actual = S._frequency_quantiles(values, owners, counts)
    for i, frequency in enumerate(counts):
        expanded = np.repeat(values, frequency[owners])
        if len(expanded):
            assert actual[i] == pytest.approx(np.quantile(expanded, [.16, .5, .84], method="linear"))
        else:
            assert np.isnan(actual[i]).all()
    assert actual[0] == pytest.approx([3.2, 10., 23.6])


def test_reproducible_across_batches_variable_order_and_population_reordering(monkeypatch):
    k = keys(5)
    owners = np.array([0, 0, 1, 3, 3, 3])
    x = np.array([1., 6., -2., 11., 9., 18.])
    y = x.copy()
    y[2] = np.nan
    result = S.event_cluster_bootstrap(k, owners, {"x": x, "y": y}, replicates=75, seed=91)
    monkeypatch.setattr(S, "_WORK_ELEMENTS", 7)  # force single-draw batches
    assert result == S.event_cluster_bootstrap(k, owners, {"y": y, "x": x}, replicates=75, seed=91)
    order = np.array([3, 1, 4, 0, 2])
    track_order = np.array([4, 0, 5, 1, 3, 2])
    remap = np.argsort(order)
    assert result == S.event_cluster_bootstrap(k[order], remap[owners][track_order],
                                              {"x": x[track_order], "y": y[track_order]},
                                              replicates=75, seed=91)
    assert result != S.event_cluster_bootstrap(k, owners, {"x": x, "y": y}, replicates=75, seed=92)


def test_missing_nonfinite_counts_are_per_variable_and_disjoint():
    owners = np.array([0, 0, 1, 1, 2, 2])
    values = np.ma.array([1., np.nan, 5., np.inf, -9., 22.], mask=[0, 0, 0, 0, 1, 1])
    result = S.event_cluster_bootstrap(keys(4), owners, {"x": values, "absent": np.ma.masked_all(6)},
                                       replicates=110, seed=4)
    row = result["variables"]["x"]
    assert [row[k] for k in ("finite_tracks", "missing_tracks", "nonfinite_tracks")] == [2, 2, 2]
    assert row["effective_events"] == row["events_without_finite_tracks"] == 2
    expected = reference(keys(4), owners, values.filled(np.nan), 110, 4)
    assert row["bootstrap_empty_replicates"] == 110 - len(expected)
    assert row["statistics"]["q50"]["bootstrap_se"] == pytest.approx(expected[:, 1].std(ddof=1))
    absent = result["variables"]["absent"]
    assert absent["missing_tracks"] == 6
    assert absent["bootstrap_empty_replicates"] == 110
    assert absent["statistics"]["q16"]["status"] == "no_finite_tracks"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("n,owners,values,status", [
    (0, [], [], "no_finite_tracks"),
    (3, [], [], "no_finite_tracks"),
    (3, [0, 1], [np.nan, np.inf], "no_finite_tracks"),
    (1, [0, 0], [1., 20.], "insufficient_effective_events"),
    (5, [2, 2], [1., 20.], "insufficient_effective_events"),
    (3, [0, 1, 2], [7., 7., 7.], "degenerate_bootstrap"),
])
def test_empty_single_effective_and_degenerate_do_not_invent_precision(n, owners, values, status):
    result = S.event_cluster_bootstrap(keys(n), owners, {"x": values}, replicates=20)
    for row in result["variables"]["x"]["statistics"].values():
        assert row["status"] == status
        assert row["bootstrap_se"] is None
        assert row["ci95"] is None
        assert (row["estimate"] is None) == (status == "no_finite_tracks")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("replicates,seed", [(0, 0), (1, 0), (-1, 0), (2.5, 0), (True, 0),
                                            (2, -1), (2, 1.5), (2, True), (2, 2**64)])
def test_invalid_bootstrap_controls(replicates, seed):
    with pytest.raises(ValueError, match="bootstrap-"):
        S.event_cluster_bootstrap(keys(0), [], {}, replicates=replicates, seed=seed)


@pytest.mark.parametrize("k,owners,values", [
    ([[1., 0., 0.]], [0], [1.]), ([[1, 0, 0], [1, 0, 0]], [0], [1.]),
    ([[1, 0, 0]], [-1], [1.]), ([[1, 0, 0]], [1], [1.]),
    ([[1, 0, 0]], [0.], [1.]), ([[1, 0, 0]], [0], []),
    ([[1, 0, 0]], [[0]], [1.]),
])
def test_invalid_population_or_ownership(k, owners, values):
    with pytest.raises(ValueError):
        S.event_cluster_bootstrap(k, owners, {"x": values})


def test_insufficient_nonempty_draws_are_null_not_zero():
    row = S.event_cluster_bootstrap(keys(100), [0, 1], {"x": [1., 7.]}, replicates=2, seed=2)["variables"]["x"]
    assert row["effective_events"] == 2
    assert row["bootstrap_valid_replicates"] == 1
    assert row["bootstrap_empty_replicates"] == 1
    for statistic in row["statistics"].values():
        assert statistic["status"] == "insufficient_valid_replicates"
        assert statistic["bootstrap_se"] is statistic["ci95"] is None


def test_full_identities_above_float_precision_are_distinct():
    k = np.array([[1, 1, 2**53], [1, 1, 2**53 + 1]], dtype=np.int64)
    row = S.event_cluster_bootstrap(k, [0, 1], {"x": [1., 7.]}, replicates=20)["variables"]["x"]
    assert row["effective_events"] == 2
    assert row["statistics"]["q50"]["bootstrap_se"] > 0


@pytest.mark.parametrize("values", [
    [-np.finfo(float).max, np.finfo(float).max],
    [1e308, 1.2e308], [-1.2e308, -1e308],
    [np.finfo(float).max, np.finfo(float).max],
])
def test_extreme_finite_values_use_stable_quantiles_width_and_se(values):
    values = np.asarray(values)
    scale = np.max(np.abs(values))
    normalized = values / scale
    expected = reference(keys(2), np.arange(2), normalized, 200, 0)
    q = np.quantile(normalized, [.16, .5, .84])
    points = np.r_[q, (q[2] - q[0]) / 2] * scale
    with np.errstate(over="raise", invalid="raise"):
        result = S.event_cluster_bootstrap(keys(2), [0, 1], {"x": values}, replicates=200)
    row = result["variables"]["x"]
    assert row["bootstrap_valid_replicates"] == 200
    assert row["bootstrap_empty_replicates"] == row["bootstrap_arithmetic_failure_replicates"] == 0
    for j, name in enumerate(("q16", "q50", "q84", "central_width")):
        statistic = row["statistics"][name]
        assert statistic["estimate"] == pytest.approx(points[j])
        if np.all(expected[:, j] == expected[0, j]):
            assert statistic["status"] == "degenerate_bootstrap"
        else:
            assert statistic["status"] == "ok"
            assert statistic["bootstrap_se"] == pytest.approx(expected[:, j].std(ddof=1) * scale)
            assert statistic["ci95"] == pytest.approx(np.quantile(expected[:, j], [.025, .975]) * scale)
    json.dumps(result, allow_nan=False)


def test_extreme_frequency_interpolation_preserves_exact_order_statistics():
    limit = np.finfo(float).max
    counts = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    with np.errstate(over="raise", invalid="raise"):
        result = S._frequency_quantiles(np.array([-limit, limit]), np.array([0, 1]), counts)
    assert result[0] / limit == pytest.approx([-.68, 0., .68])
    assert np.all(result[1] == -limit)
    assert np.all(result[2] == limit)
    assert np.isnan(result[3]).all()


def test_unrepresentable_standard_error_fails_closed_not_ok_with_infinity():
    limit = np.finfo(float).max
    # Seed 10 gives two nonempty draws, one at each endpoint: SD > float64 max.
    result = S.event_cluster_bootstrap(keys(2), [0, 1], {"x": [-limit, limit]},
                                       replicates=2, seed=10)
    row = result["variables"]["x"]
    assert row["bootstrap_valid_replicates"] == 2
    assert row["bootstrap_empty_replicates"] == row["bootstrap_arithmetic_failure_replicates"] == 0
    for name in ("q16", "q50", "q84"):
        statistic = row["statistics"][name]
        assert statistic["status"] == "arithmetic_failure"
        assert statistic["estimate"] is not None
        assert statistic["bootstrap_se"] is statistic["ci95"] is None
    json.dumps(result, allow_nan=False)


def test_nonempty_arithmetic_failures_are_separate_from_empty_draws(monkeypatch):
    original = S._frequency_quantiles

    def fail_one_nonempty(*args):
        q = original(*args)
        q[np.flatnonzero(np.isfinite(q[:, 0]))[0]] = np.inf
        return q

    monkeypatch.setattr(S, "_frequency_quantiles", fail_one_nonempty)
    result = S.event_cluster_bootstrap(keys(5), [0, 1], {"x": [1., 7.]}, replicates=71, seed=8)
    row = result["variables"]["x"]
    expected = reference(keys(5), np.array([0, 1]), np.array([1., 7.]), 71, 8)
    assert row["bootstrap_empty_replicates"] == 71 - len(expected) > 0
    assert row["bootstrap_arithmetic_failure_replicates"] == 1
    assert row["bootstrap_valid_replicates"] == len(expected) - 1
    for statistic in row["statistics"].values():
        assert statistic["status"] == "arithmetic_failure"
        assert statistic["bootstrap_se"] is statistic["ci95"] is None
    json.dumps(result, allow_nan=False)
