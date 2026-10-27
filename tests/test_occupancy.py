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
