"""Native piece search cases independent of captured data and ANN weights."""
import numpy as np
import pytest

from trkfastsim.reco import nativeref as R


def owner(times):
    p = np.zeros((1, len(times), 6))
    p[0, :, 1] = -np.pi / 2
    p[0, :, 3] = np.pi / 2
    p[0, :, 4] = times
    p[0, :, 5] = 2000
    ranges = np.array([[[i, i + 1] for i in range(len(times))]], dtype=float)
    return p, ranges, np.array([len(times)])


def test_nearest_edges_and_degenerate_precedence():
    p, ranges, counts = owner([0, 0, 0])
    t = np.array([-1, 0, np.nextafter(1., 0.), 1, np.nextafter(1., 2.), 2, 3, 4])
    np.testing.assert_array_equal(R.nearest(ranges, counts, np.zeros(len(t), int), t), [0, 0, 0, 0, 1, 1, 2, 2])
    ranges[:] = 0
    np.testing.assert_array_equal(R.nearest(ranges, counts, np.array([0]), np.array([0.])), [0])


def search(times, hint=.5, moves=10):
    p, r, c = owner(times)
    return R.search(p, r, c, np.array([0]), np.zeros((1, 3)), np.array([[1., 0, 0]]), np.array([hint]), moves)


def test_recompute_moves_until_solution_piece_is_stable():
    result = search([1.5, 2.5, 2.75])
    assert result["index"][0] == result["solution_index"][0] == 2
    assert result["moves"][0] == 2
    assert result["time"][0] == pytest.approx(2.75)
    assert not result["exhausted"][0]


def test_zero_moves_stays_on_hint_piece_even_when_solution_outside():
    result = search([1.5, .5], moves=0)
    assert result["index"][0] == 0
    assert result["time"][0] == pytest.approx(1.5)
    assert result["moves"][0] == 0


def test_oscillation_cap_keeps_native_reference_solution_split():
    result = search([1.5, .5], moves=10)
    assert result["index"][0] == 1
    assert result["solution_index"][0] == 0
    assert result["moves"][0] == 10
    assert result["exhausted"][0]
    assert result["usable"][0]


def test_exact_boundary_selects_earlier_piece():
    result = search([1., 1.5], hint=1.5)
    assert result["index"][0] == 1
    result = search([1., 1.], hint=1.5)
    assert result["index"][0] == 0


def test_parallel_local_ca_is_unusable():
    p, r, c = owner([.5])
    t, usable = R.toca(p[0], np.zeros((1, 3)), np.array([[0., -1, 0]]))
    assert not usable[0]


def test_many_to_one_join_and_missing_and_duplicate_guard():
    a = np.array([[1, 3], [1, 2]])
    w = np.array([[1, 2], [1, 3], [1, 2], [1, 4]])
    np.testing.assert_array_equal(R.match_rows(w, a), [1, 0, 1, -1])
    with pytest.raises(ValueError, match="duplicate"):
        R.match_rows(w, np.vstack([a, a]))


@pytest.mark.parametrize("times,moves", [([1.5, 2.5, 2.75], 10), ([1.5, .5], 0), ([1.5, .5], 10), ([1., 1.], 10)])
def test_jax_search_matches_independent_native_coordinate_search(times, moves):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from trkfastsim.reco import kalman_jax as K
    p, ranges, counts = owner(times)
    x = np.zeros_like(p[0])
    x[:, 4], x[:, 5] = times, 2000
    ow = {"P": jnp.asarray(x), "ranges": jnp.asarray(ranges[0]), "count": jnp.array(counts[0] - 1)}
    ix, time, unsupported = jax.jit(lambda hint: K.reference_search(
        ow, jnp.zeros((1, 3)), jnp.array([[1., 0, 0]]), hint, max_moves=moves))(jnp.array([.5]))
    expected = search(times, moves=moves)
    np.testing.assert_array_equal(ix, expected["index"])
    np.testing.assert_allclose(time, expected["time"], atol=1e-14)
    np.testing.assert_array_equal(unsupported, ~expected["usable"] | expected["exhausted"])
    detailed = jax.jit(lambda hint: K.reference_search_state(
        ow, jnp.zeros((1, 3)), jnp.array([[1., 0, 0]]), hint, max_moves=moves))(jnp.array([.5]))
    for name in ("index", "solution_index", "moves", "usable", "exhausted"):
        np.testing.assert_array_equal(detailed[name], expected[name])
    np.testing.assert_array_equal(detailed["time"], time)


def test_vector_lanes_stop_independently():
    p, ranges, counts = owner([1.5, 2.5, 2.75])
    hints = np.array([.5, 1.5, 2.8])
    result = R.search(p, ranges, counts, np.zeros(3, int), np.zeros((3, 3)),
                      np.tile([1., 0, 0], (3, 1)), hints)
    np.testing.assert_array_equal(result["index"], [2, 2, 2])
    np.testing.assert_array_equal(result["moves"], [2, 1, 0])


def test_reference_option_guard_and_default_equality():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from test_kalman_jax import _toy_track, _anns_available
    from trkfastsim.reco import kalman_jax as K
    if not _anns_available():
        pytest.skip("ANN weights unavailable")
    xt, _, h, c, _, d = _toy_track(3)
    x = jnp.asarray(xt)
    stages, ann = K.stock_stages(), K.load_anns()
    with pytest.raises(ValueError, match="require stock"):
        K.fit_track(x, h, c, d, stages, ann, reference_model="search")
    for model in ("legacy", "stock"):
        implicit = jax.jit(lambda p: K.fit_track(p, h, c, d, stages, ann, time_model=model))(x)
        explicit = jax.jit(lambda p: K.fit_track(p, h, c, d, stages, ann, time_model=model,
                                                reference_model="previous"))(x)
        for a, b in zip(jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)):
            np.testing.assert_array_equal(a, b)


def test_audit_piece_index_guards():
    import importlib.util
    from pathlib import Path
    import awkward as ak
    spec = importlib.util.spec_from_file_location("native_reference_audit", Path(__file__).parents[1] / "scripts/f1_native_reference_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    it = ak.Array({"run": [1], "subrun": [0], "event": [2], "iseed": [0], "iteration_ordinal": [0]})
    pieces = {"run": [1, 1], "subrun": [0, 0], "event": [2, 2], "iseed": [0, 0],
              "iteration_ordinal": [0, 0], "parameters": [[0.] * 6, [0.] * 6], "range": [[0., 1.], [1., 2.]]}
    for index in ([-1, 0], [0, 0], [0, 2]):
        with pytest.raises(ValueError, match="piece ind"):
            audit.pad_pieces(ak.Array(dict(pieces, piece_index=index)), it)
    _, ranges, counts = audit.pad_pieces(ak.Array(dict(pieces, piece_index=[1, 0])), it)
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_array_equal(ranges[0], [[1, 2], [0, 1]])


def test_state_census_uses_solved_piece_not_updated_pointer():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("native_reference_census", Path(__file__).parents[1] / "scripts/f1_native_reference_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    p, _, _ = owner([1.5, .5])
    p[0, 1, 2] = 500.0
    mid = np.zeros((1, 3))
    result = {"index": np.array([1]), "solution_index": np.array([0]), "time": np.array([1.5]),
              "usable": np.array([True])}
    keys = np.array([[1, 0, 1, 0, 0, 0]])
    report = audit.state_census(p, np.array([0]), mid, np.array([[1., 0, 0]]), mid,
                                np.array([True]), result, keys)
    assert report["pointer_solution_split_count"] == 1
    assert report["hit_retry_trigger_count"] == 0
    assert report["max_abs_hit_dz_mm"] == 0.0
    assert report["pointer_resolve_minus_retained_time_ns"] == [-1.0]
    result["usable"][:] = False
    report = audit.state_census(p, np.array([0]), mid, np.array([[1., 0, 0]]), mid,
                                np.array([True]), result, keys)
    assert report["hit_retry_trigger_keys"] == keys.tolist()
