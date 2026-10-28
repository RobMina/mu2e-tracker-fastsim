"""Paired score contrast and deletion-moment tests."""
import numpy as np
from scripts import secondary_occupancy_jackknife as J
from trkfastsim.analyses import occupancy as O


def _score(value, rank=3, **kw):
    return {'chi2ndf_covariance': value, 'covariance_rank': rank,
            'available': True, 'null_space_incompatible': False, **kw}


def test_identical_replicas_cancel_and_unstable_rank_is_unavailable():
    reps = [_score(1.), _score(2.), _score(3.)]
    r = J.difference_error(_score(2.), _score(2.), reps, reps)
    assert r['available'] and r['difference'] == r['error'] == 0
    r = J.difference_error(_score(2.), _score(1.), [_score(2.), _score(3.), _score(5.)], reps)
    expected = np.array([1., 1., 2.])
    np.testing.assert_allclose(r['error'], np.sqrt(2 / 3 * ((expected - expected.mean()) ** 2).sum()))
    assert not J.difference_error(_score(2.), _score(2.), [_score(1., rank=2), _score(2.), _score(3.)], reps)['available']
    assert not J.difference_error(_score(2.), _score(2.), [_score(1., support_stable=False)] * 3, reps)['available']


def test_empty_groups_and_unavailable_score_closure():
    import pytest
    with pytest.raises(ValueError, match='one file group'):
        J.evaluate({'base': [], 'g4': []}, {'base': []})
    missing = _score(np.nan, rank=0, available=False)
    assert J.score_closes(np.nan, missing)
    assert not J.score_closes(np.nan, _score(1.))
    reps = [_score(1.), _score(2.), _score(3.)]
    for invalid in [missing, _score(1., null_space_incompatible=True)]:
        assert not J.difference_error(invalid, _score(1.), reps, reps)['available']
    assert not J.difference_error(_score(1.), _score(1.), reps[:1], reps[:1])['available']


def test_group_moment_subtraction_equals_direct_leave_one_recomputation():
    rng = np.random.default_rng(7)
    rows, marginals, crosses = [], {'base': [], 'candidate': [], 'g4': []}, {'base': [], 'candidate': []}
    for group in range(3):
        ev = np.repeat(np.arange(250) + group * 250, 3)
        panels = {a: rng.integers(0, 4, len(ev)) for a in marginals}
        rows.append((ev, panels))
        for a in marginals:
            marginals[a].append(O.accumulate(ev, panels[a], npan=4))
        for a in crosses:
            crosses[a].append(O.accumulate_cross(ev, panels[a], ev, panels['g4'], npan=4))
    result = J.evaluate(marginals, crosses)
    brute = []
    for omit in range(3):
        ev = np.concatenate([x[0] for i, x in enumerate(rows) if i != omit])
        panels = {a: np.concatenate([x[1][a] for i, x in enumerate(rows) if i != omit]) for a in marginals}
        acc = {a: O.accumulate(ev, p, npan=4) for a, p in panels.items()}
        scores = {a: O.compare(acc[a], acc['g4'], cross=O.accumulate_cross(ev, panels[a], ev, panels['g4'], npan=4))
                  for a in crosses}
        brute.append(scores['candidate']['chi2ndf_covariance'] - scores['base']['chi2ndf_covariance'])
    np.testing.assert_allclose(result['contrasts']['candidate']['replica_differences'], brute, rtol=1e-12)
