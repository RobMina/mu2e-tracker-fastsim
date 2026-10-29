"""Dense scalar oracles and edge cases for event-clustered histogram bands."""
import json

import numpy as np
import pytest

from trkfastsim.analyses import clusterband as C


def _key(ev):
    return tuple(np.atleast_1d(ev).tolist())


def _dense(f, r, ef, er, edges, paired):
    """Deliberately scalar independent oracle, including random denominators."""
    keys_f = [_key(e) for e in ef]
    keys_r = [_key(e) for e in er]
    if not paired:
        keys_f = [("f",) + e for e in keys_f]
        keys_r = [("r",) + e for e in keys_r]
    keys = list(dict.fromkeys(keys_f + keys_r))
    pos = {k: i for i, k in enumerate(keys)}
    mats = []
    for values, event_keys in ((f, keys_f), (r, keys_r)):
        a = np.zeros((len(keys), len(edges) - 1))
        for value, key in zip(values, event_keys):
            if np.isfinite(value) and edges[0] <= value <= edges[-1]:
                b = min(np.searchsorted(edges, value, side="right") - 1, len(edges) - 2)
                a[pos[key], b] += 1
        mats.append(a)
    af, ar = mats
    nf, nr = af.sum(axis=1), ar.sum(axis=1)
    sf, sr = af.sum(axis=0) / nf.sum(), ar.sum(axis=0) / nr.sum()
    uf = (af - nf[:, None] * sf) / nf.sum()
    ur = (ar - nr[:, None] * sr) / nr.sum()
    valid = sr > 0
    z = uf[:, valid] / sr[valid] - ur[:, valid] * sf[valid] / sr[valid]**2
    return sf, sr, uf.T @ uf, ur.T @ ur, z.T @ z


def _sample():
    # Event numbers deliberately repeat across runs/subruns; cluster sizes vary.
    ids = np.array([[1, 1, 4], [1, 2, 4], [2, 1, 4], [2, 2, 4], [9, 1, 5]])
    ef = ids[[0, 0, 0, 1, 1, 2, 2, 2, 3, 4]]
    er = ids[[0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 4]]
    f = np.array([0.2, 0.2, 1.2, 1.2, 2.2, 0.2, 2.2, 2.2, -3, 9])
    r = np.array([0.2, 1.2, 0.2, 1.2, 1.2, 2.2, 0.2, 2.2, 2.2, 9, -3])
    return f, r, ef, er


@pytest.mark.parametrize("paired", [True, False])
def test_sparse_covariance_and_bands_match_scalar_dense_oracle(paired):
    f, r, ef, er = _sample()
    edges = [0, 1, 2, 3]
    actual = C.ratio_band(f, r, ef, er, edges=edges, min_ref=1, paired=paired)
    sf, sr, vf, vr, cov = _dense(f, r, ef, er, edges, paired)
    np.testing.assert_allclose(actual["share_fast"], sf)
    np.testing.assert_allclose(actual["share_ref"], sr)
    np.testing.assert_allclose(np.square(actual["share_error_fast"]), np.diag(vf), atol=1e-16)
    np.testing.assert_allclose(np.square(actual["share_error_ref"]), np.diag(vr), atol=1e-16)
    np.testing.assert_allclose(actual["ratio_covariance"], cov, atol=1e-15)
    np.testing.assert_allclose(actual["ratio_error"], np.sqrt(np.diag(cov)))
    dev = np.abs(sf / sr - 1)
    assert actual["max_abs_dev"] == pytest.approx(dev.max())
    assert actual["max_abs_dev_minus_2sigma"] == pytest.approx(np.maximum(dev - 2 * np.sqrt(np.diag(cov)), 0).max())
    assert actual["available"]
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("paired", [True, False])
def test_randomized_scalar_oracle_including_empty_reference_bins(paired):
    rng = np.random.default_rng(540)
    for _ in range(8):
        ef = rng.integers(0, 15, 91)
        er = rng.integers(0, 15, 67)
        f = rng.choice([-10, 0.2, 1.2, 2.2, 3.2, 30], len(ef))
        r = rng.choice([-10, 0.2, 1.2, 3.2, 30], len(er))
        edges = np.arange(5)
        actual = C.ratio_band(f, r, ef, er, edges=edges, min_ref=1, paired=paired)
        sf, sr, _, _, cov = _dense(f, r, ef, er, edges, paired)
        np.testing.assert_allclose(actual["ratio_covariance"], cov, atol=1e-14)
        assert actual["ratio"][2] is None
        assert actual["ratio_error"][2] is None
        assert actual["covariance_bins"] == [0, 1, 3]
        np.testing.assert_allclose(np.asarray(actual["ratio"])[sr > 0].astype(float), sf[sr > 0] / sr[sr > 0])


def test_identical_paired_arms_have_zero_errors_but_cannot_certify_rank_zero():
    values = [0.2, 0.2, 1.2, 1.2, 2.2, 0.2]
    ev = [1, 1, 2, 3, 3, 4]
    actual = C.ratio_band(values, values, ev, ev, edges=[0, 1, 2, 3], min_ref=1)
    assert actual["ratio"] == [1.0, 1.0, 1.0]
    assert actual["ratio_error"] == [0.0, 0.0, 0.0]
    assert actual["covariance_rank"] == 0
    assert not actual["available"]
    assert actual["max_abs_dev"] == 0
    assert actual["max_abs_dev_minus_2sigma"] is None
    independent = C.ratio_band(values, values, ev, ev, edges=[0, 1, 2, 3], min_ref=1, paired=False)
    assert independent["available"]
    assert np.all(np.asarray(independent["ratio_error"]) > 0)


def test_perfectly_correlated_counts_with_arm_duplication_have_zero_error():
    values = np.array([0.2, 0.2, 1.2, 1.2, 2.2, 0.2])
    ev = np.array([1, 1, 2, 3, 3, 4])
    actual = C.ratio_band(np.repeat(values, 3), values, np.repeat(ev, 3), ev,
                          edges=[0, 1, 2, 3], min_ref=1)
    np.testing.assert_allclose(actual["ratio_error"], 0, atol=1e-14)
    assert actual["covariance_rank"] == 0


@pytest.mark.parametrize("paired", [True, False])
def test_within_event_entry_duplication_invariance(paired):
    f, r, ef, er = _sample()
    kwargs = dict(edges=[0, 1, 2, 3], min_ref=1, paired=paired)
    a = C.ratio_band(f, r, ef, er, **kwargs)
    b = C.ratio_band(np.repeat(f, 7), np.repeat(r, 3), np.repeat(ef, 7, axis=0), np.repeat(er, 3, axis=0), **kwargs)
    for key in ("share_fast", "share_ref", "share_error_fast", "share_error_ref", "ratio", "ratio_error", "ratio_covariance"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-14)
    assert a["covariance_rank"] == b["covariance_rank"]


def test_shuffled_multirun_ids_preserve_pairing_not_just_event_number():
    f, r, ef, er = _sample()
    rng = np.random.default_rng(93)
    pf, pr = rng.permutation(len(f)), rng.permutation(len(r))
    kwargs = dict(edges=[0, 1, 2, 3], min_ref=1)
    a = C.ratio_band(f, r, ef, er, **kwargs)
    b = C.ratio_band(f[pf], r[pr], ef[pf], er[pr], **kwargs)
    assert a == b
    assert a["matched_events_with_entries"] == 5
    lossy = C.ratio_band(f, r, ef[:, 2], er[:, 2], **kwargs)
    assert not np.allclose(a["ratio_error"], lossy["ratio_error"])


def test_disjoint_ids_equal_independent_covariance():
    f, r, ef, er = _sample()
    er = er + np.array([100, 0, 0])
    kwargs = dict(edges=[0, 1, 2, 3], min_ref=1)
    a = C.ratio_band(f, r, ef, er, **kwargs)
    b = C.ratio_band(f, r, ef, er, paired=False, **kwargs)
    np.testing.assert_allclose(a["ratio_covariance"], b["ratio_covariance"], atol=1e-15)
    assert a["matched_events_with_entries"] == 0


def test_partial_overlap_and_large_integer_keys():
    f, r, ef, er = _sample()
    ef = ef.copy()
    er = er.copy()
    ef[:, 0] += 2**60
    er[:, 0] += 2**60
    er[er[:, 1] == 2, 0] += 77
    a = C.ratio_band(f, r, ef, er, edges=[0, 1, 2, 3], min_ref=1)
    *_, cov = _dense(f, r, ef, er, [0, 1, 2, 3], True)
    np.testing.assert_allclose(a["ratio_covariance"], cov, atol=1e-15)
    assert a["matched_events_with_entries"] == 3


def test_flow_bookkeeping_and_random_in_range_denominator():
    f = [0, 1, 2, -1, 3, np.nan, np.inf, -np.inf]
    r = [0.2, 1.2, 1.2, 0.2]
    ef, er = np.arange(len(f)), np.arange(len(r))
    a = C.ratio_band(f, r, ef, er, edges=[0, 1, 2], min_ref=1)
    assert a["h_fast"] == [1, 2]  # final right edge included
    np.testing.assert_allclose(a["share_fast"], [1/3, 2/3])
    assert a["flow_fast"] == {"total": 8, "finite": 5, "in_range": 3, "underflow": 1,
                              "overflow": 1, "nonfinite": 3, "events_with_entries": 8, "events_in_range": 3}
    sf, sr, _, _, cov = _dense(f, r, ef, er, [0, 1, 2], True)
    np.testing.assert_allclose(a["ratio"], sf / sr)
    np.testing.assert_allclose(a["ratio_covariance"], cov)
    # Entries outside the range do not enter random in-range normalization.
    b = C.ratio_band(f[:3], r, ef[:3], er, edges=[0, 1, 2], min_ref=1)
    np.testing.assert_allclose(a["ratio_covariance"], b["ratio_covariance"])
    json.dumps(a, allow_nan=False)


def test_default_reference_edges_match_core_construction_and_constant_range():
    r = np.arange(1000, dtype=float)
    a = C.ratio_band(r, r, np.arange(len(r)), np.arange(len(r)), min_ref=1)
    _, expected = np.histogram(r, bins=20, range=tuple(np.quantile(r, [0.05, 0.95])))
    np.testing.assert_array_equal(a["edges"], expected)
    const = C.ratio_band([3, 3], [3, 3], [1, 2], [1, 2], nbins=2, min_ref=1)
    assert const["edges"] == [2.5, 3, 3.5]
    assert not const["available"]


@pytest.mark.parametrize("f,r", [([], []), ([], [0.2, 1.2]), ([0.2, 1.2], []), ([9], [0.2, 1.2]), ([0.2], [np.nan])])
@pytest.mark.parametrize("fixed", [False, True])
def test_empty_arms_fail_closed_and_are_strict_json(f, r, fixed):
    kw = {"edges": [0, 1, 2]} if fixed else {}
    a = C.ratio_band(f, r, np.arange(len(f)), np.arange(len(r)), min_ref=1, **kw)
    assert not a["available"]
    assert a["max_abs_dev"] is None
    assert a["max_abs_dev_minus_2sigma"] is None
    assert a["unavailable_reason"]
    json.dumps(a, allow_nan=False)


def test_no_selected_bins_and_single_bin_rank_limitation():
    a = C.ratio_band([0.2, 1.2], [0.2, 1.2], [1, 2], [3, 4], edges=[0, 1, 2])
    assert a["nbins_used"] == 0
    assert not a["available"]
    assert a["max_abs_dev"] is None
    b = C.ratio_band([0.2, 1.2], [0.2, 1.2], [1, 2], [3, 4], edges=[0, 2], min_ref=1)
    assert b["ratio_error"] == [0.0]
    assert not b["available"]
    assert b["covariance_rank"] == 0


def test_one_event_arm_and_zero_reference_bins_fail_closed():
    a = C.ratio_band([0.2, 1.2], [0.2, 1.2], [1, 1], [2, 3], edges=[0, 1, 2, 3], min_ref=0)
    assert not a["available"]
    assert a["ratio"][2] is None
    assert a["ratio_error"][2] is None
    assert a["bins_used"] == [0, 1]
    assert a["max_abs_dev_minus_2sigma"] is None


@pytest.mark.parametrize("kwargs", [
    {"edges": [0, 0, 1]}, {"edges": [0, np.inf]}, {"edges": [[0, 1]]},
    {"edges": [1]}, {"q": [0.9, 0.1]}, {"q": [-0.1, 1]},
    {"nbins": 0}, {"nbins": 1.5}, {"min_ref": -1}, {"min_ref": np.nan}, {"paired": "yes"},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        C.ratio_band([0.2], [0.2], [1], [1], **kwargs)


@pytest.mark.parametrize("values,events", [([1, 2], [1]), ([1], [1.0]),
    ([1], [[1, 2]]), ([[1]], [1]), ([1], [True]),
    ([1], np.array([2**63], dtype=np.uint64))])
def test_invalid_values_or_event_ids(values, events):
    with pytest.raises(ValueError):
        C.ratio_band(values, [1], events, [1], edges=[0, 2])


def test_mixed_id_representations_rejected_only_for_paired():
    with pytest.raises(ValueError):
        C.ratio_band([1], [1], [1], [[1, 1, 1]], edges=[0, 2])
    C.ratio_band([1], [1], [1], [[1, 1, 1]], edges=[0, 2], paired=False)
    a = C.ratio_band([], [1], [], [[1, 1, 1]], edges=[0, 2])
    assert not a["available"]


@pytest.mark.parametrize("paired", [True, False])
def test_means_match_dense_event_oracle_and_duplication(paired):
    f, r, ef, er = _sample()
    a = C.mean_compare(f, r, ef, er, paired=paired)
    ids = list(dict.fromkeys([_key(e) for e in ef] + [_key(e) for e in er]))
    uf, ur = [], []
    for event in ids:
        xf = f[np.all(ef == event, axis=1)]
        xr = r[np.all(er == event, axis=1)]
        uf.append((xf.sum() - len(xf) * f.mean()) / len(f))
        ur.append((xr.sum() - len(xr) * r.mean()) / len(r))
    uf, ur = np.array(uf), np.array(ur)
    cross = uf @ ur if paired else 0
    assert a["fast"]["mean"] == pytest.approx(f.mean())
    assert a["ref"]["mean"] == pytest.approx(r.mean())
    assert a["cross_covariance"] == pytest.approx(cross)
    assert a["difference_error"]**2 == pytest.approx(uf @ uf + ur @ ur - 2 * cross)
    ratio = f.mean() / r.mean()
    assert a["ratio_error"]**2 == pytest.approx((uf @ uf + ratio**2 * (ur @ ur) - 2 * ratio * cross) / r.mean()**2)
    b = C.mean_compare(np.repeat(f, 3), np.repeat(r, 2), np.repeat(ef, 3, axis=0), np.repeat(er, 2, axis=0), paired=paired)
    for key in ("difference", "difference_error", "ratio", "ratio_error", "cross_covariance"):
        assert a[key] == pytest.approx(b[key])
    assert a["available"]
    json.dumps(a, allow_nan=False)


def test_means_identical_proportional_empty_nonfinite_and_zero_reference():
    x, ev = np.array([1, 2, 3, 5]), np.array([1, 1, 2, 3])
    same = C.mean_compare(x, x, ev, ev)
    assert same["difference_error"] == same["ratio_error"] == 0
    assert not same["available"]
    proportional = C.mean_compare(2 * x, x, ev, ev)
    assert proportional["ratio_error"] == 0
    assert proportional["ratio"] == 2
    empty = C.mean_compare([], x, [], ev)
    assert not empty["available"]
    assert empty["difference_error"] is None
    finite = C.mean_compare([1, 2, np.nan, np.inf], [1, -1], [1, 2, 3, 4], [1, 2])
    assert finite["fast"]["count"] == 2
    assert finite["fast"]["nonfinite"] == 2
    assert finite["ratio"] is None
    assert finite["difference_error"] is not None
    json.dumps(finite, allow_nan=False)


def test_many_sparse_event_clusters_without_event_by_bin_dense_storage():
    # Large-cluster input guards sparse event storage and vectorized execution.
    rng = np.random.default_rng(12)
    f, r = rng.random(100_000), rng.random(100_000)
    ev = np.arange(len(f))
    a = C.ratio_band(f, r, ev, ev, edges=np.linspace(0, 1, 41), min_ref=20)
    assert a["available"]
    assert a["nbins_used"] == 40
    assert a["matched_events_with_entries"] == 100_000


def test_mean_rank_zero_after_permuting_floating_point_entries():
    from trkfastsim.analyses.clusterband import mean_compare
    result = mean_compare([.1, .2, .3], [.3, .2, .1], [0, 0, 1], [1, 0, 0])
    assert not result['available']
    assert result['difference_error'] == 0.0
    assert result['ratio_error'] == 0.0
