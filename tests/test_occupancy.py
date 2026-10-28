"""Panel occupancy shares with event-clustered errors (analyses/occupancy.py)."""
import numpy as np

from trkfastsim.analyses import occupancy as O


def _sample(rng, nev, probs, cluster):
    """Events hit one panel (drawn from probs) with `cluster` hits each: identical physics in every call."""
    p = rng.choice(len(probs), size=nev, p=probs)
    ev = np.repeat(np.arange(nev), cluster)
    return ev, np.repeat(p, cluster)


def test_unique_panel_from_strawid():
    sid = (np.array([0, 26, 27, 35]) << 10) | (np.array([0, 5, 0, 5]) << 7) | np.array([3, 95, 0, 1])
    assert O.unique_panel(sid).tolist() == [0, 26 * 6 + 5, 27 * 6, 35 * 6 + 5]


def test_clustered_chi2_is_one_for_identical_physics_while_poisson_scales_with_cluster_size():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.5, 1.5, O.NPAN)
    probs /= probs.sum()
    cp, cc = [], []
    for _ in range(8):
        a = O.accumulate(*_sample(rng, 40000, probs, 4))
        b = O.accumulate(*_sample(rng, 40000, probs, 4))
        r = O.compare(a, b)
        cp.append(r["chi2ndf_poisson"])
        cc.append(r["chi2ndf_clustered"])
    assert 0.85 < np.mean(cc) < 1.15
    assert 3.4 < np.mean(cp) < 4.6          # cluster size 4
    assert 3.5 < r["overdispersion_median"] < 4.5


def test_clustered_chi2_detects_a_real_shape_difference():
    rng = np.random.default_rng(2)
    probs = np.full(O.NPAN, 1.0 / O.NPAN)
    q = probs.copy()
    q[:6] *= 1.5
    q /= q.sum()
    r = O.compare(O.accumulate(*_sample(rng, 200000, q, 4)), O.accumulate(*_sample(rng, 200000, probs, 4)))
    assert r["panels_beyond_5sigma"] >= 3
    assert {w["plane"] for w in r["worst"][:6]} == {0}


def test_full_covariance_rank_for_plane_clusters():
    rng = np.random.default_rng(11)
    scores = []
    for _ in range(8):
        samples = []
        for arm in range(2):
            pl = rng.integers(0, 36, size=40000)
            ev = np.repeat(np.arange(len(pl)), 6)
            pan = (pl[:, None] * 6 + np.arange(6)).ravel()
            samples.append(O.accumulate(ev, pan))
        result = O.compare(*samples)
        scores.append(result["chi2ndf_covariance"])
        assert result["covariance_rank"] == 35
        assert not result["null_space_incompatible"]
    assert 0.7 < np.mean(scores) < 1.3


def test_covariance_diagonal_matches_marginal_variance():
    a = O.accumulate(np.array([1, 1, 1, 2, 2, 3]), np.array([0, 1, 1, 0, 2, 2]), npan=3)
    assert np.allclose(np.diag(O.covariance(a)), O.shares(a)[1])
    assert np.allclose(O.covariance(a).sum(axis=0), 0)


def test_empty_occupancy_is_not_available():
    a = O.accumulate([], [])
    r = O.compare(a, a)
    assert not r["available"]
    assert r["covariance_rank"] == 0
    assert np.isnan(r["chi2ndf_covariance"])


def _dense(ev, panel, ids, npan):
    from scipy.sparse import csr_matrix
    row = np.searchsorted(ids, ev)
    return csr_matrix((np.ones(len(ev)), (row, panel)), shape=(len(ids), npan)).toarray()


def test_paired_covariance_matches_dense_influences_and_disjoint_events():
    ef, pf = np.array([1, 1, 2, 2, 2, 3, 4]), np.array([0, 1, 0, 2, 2, 1, 2])
    er, pr = np.array([1, 1, 1, 2, 3, 3, 5]), np.array([1, 1, 2, 0, 1, 2, 0])
    af, ar = O.accumulate(ef, pf, 3), O.accumulate(er, pr, 3)
    cross = O.accumulate_cross(ef, pf, er, pr, 3)
    ids = np.union1d(ef, er)
    f, r = _dense(ef, pf, ids, 3), _dense(er, pr, ids, 3)
    uf = (f - f.sum(axis=1)[:, None] * f.sum(axis=0) / f.sum()) / f.sum()
    ur = (r - r.sum(axis=1)[:, None] * r.sum(axis=0) / r.sum()) / r.sum()
    assert np.allclose(O.cross_covariance(af, ar, cross), uf.T @ ur)
    cov = O.covariance(af) + O.covariance(ar) - O.cross_covariance(af, ar, cross) - O.cross_covariance(af, ar, cross).T
    assert np.allclose(cov, (uf - ur).T @ (uf - ur))
    result = O.compare(af, ar, min_ref=1, cross=cross)
    expected = O._quadratic(O.shares(af)[0] - O.shares(ar)[0], cov)
    assert np.isclose(result['chi2ndf_covariance'], expected[0])
    disjoint = O.accumulate_cross(ef, pf, er + 100, pr, 3)
    assert np.allclose(O.cross_covariance(af, ar, disjoint), 0)
    assert np.isclose(O.compare(af, ar, min_ref=1, cross=disjoint)['chi2ndf_covariance'],
                      O.compare(af, ar, min_ref=1)['chi2ndf_covariance'])


def test_identical_paired_arms_are_zero_variance_not_a_rank_zero_pass():
    ev, pan = _sample(np.random.default_rng(4), 4000, np.ones(6) / 6, 3)
    a = O.accumulate(ev, pan, 6)
    cross = O.accumulate_cross(ev, pan, ev, pan, 6)
    result = O.compare(a, a, cross=cross)
    assert result['cross_arm_covariance_included']
    assert result['covariance_rank'] == 0 and not result['available']
    assert not result['null_space_incompatible']
    assert result['max_abs_pull'] == 0


def test_cross_moments_merge_and_full_event_identity():
    ef = np.array([[1, 2, 3], [1, 2, 3], [2, 2, 3], [2, 2, 4]])
    er = ef[[2, 0, 3, 1]]
    pf, pr = np.array([0, 1, 2, 0]), np.array([0, 1, 2, 1])
    cross = O.accumulate_cross(ef, pf, er, pr, 3)
    split = None
    for run in (1, 2):
        mf, mr = ef[:, 0] == run, er[:, 0] == run
        split = O.accumulate_cross(ef[mf], pf[mf], er[mr], pr[mr], 3, acc=split)
    for k in cross:
        assert np.array_equal(cross[k], split[k])
    assert cross['S'].sum() == 6


def test_paired_covariance_invariant_under_duplicate_hits():
    rng = np.random.default_rng(19)
    ev = np.repeat(np.arange(4000), 2)
    pf, pr = rng.integers(0, 6, len(ev)), rng.integers(0, 6, len(ev))
    results = []
    for k in (1, 3):
        ef, f = np.repeat(ev, k), np.repeat(pf, k)
        er, r = np.repeat(ev, k), np.repeat(pr, k)
        results.append(O.compare(O.accumulate(ef, f, 6), O.accumulate(er, r, 6),
                                 cross=O.accumulate_cross(ef, f, er, r, 6)))
    assert np.isclose(results[0]['chi2ndf_covariance'], results[1]['chi2ndf_covariance'])
    assert np.allclose(results[0]['pulls'], results[1]['pulls'])


def test_paired_mismatched_coverage_rejected():
    import pytest
    a = O.accumulate([1, 2], [0, 1], 3)
    cross = O.accumulate_cross([1], [0], [1, 2], [0, 1], 3)
    with pytest.raises(ValueError, match='cover'):
        O.compare(a, a, min_ref=1, cross=cross)


def test_empty_fast_arm_cannot_certify_against_nonempty_reference():
    empty = O.accumulate([], [], 3)
    ref = O.accumulate([1, 2, 3], [0, 1, 2], 3)
    cross = O.accumulate_cross([], [], [1, 2, 3], [0, 1, 2], 3)
    assert not O.compare(empty, ref, min_ref=1, cross=cross)['available']
