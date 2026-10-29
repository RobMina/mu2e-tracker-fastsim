"""Shared-draw migration report accounting and exact saved-lattice checks."""
import numpy as np
import pandas as pd
import pytest

from scripts import secondary_migration as D
from trkfastsim import library as L, steps as S
from trkfastsim.analyses import secmigration as M
from test_anchor_topology import _source
from test_library_v2 import _build


def test_matrix_conserves_draws_energy_and_reports_interior_exterior():
    g, source, parts = _source()
    lib = _build(source, parts, g, topology=True)
    mu = source[source.sp == 1]
    mu, first, draws = M.draw_context(mu, L.OffAxisSampler(lib, np.random.default_rng(2)))
    b = M.trace(mu, first, draws, g)
    t = M.trace(mu, first, draws, g, 'topology', 'topology')
    report = D.transition_summary(b, t, mu)
    assert np.asarray(report['transition_by_exterior']).sum() == len(draws)
    np.testing.assert_allclose(np.asarray(report['transition_eion_MeV']).sum(), draws.eion.sum())
    assert sum(report['changed_lattice_bitmask_plane_panel_straw']) == len(draws)
    total = {}
    D.add_summary(total, report)
    D.add_summary(total, report)
    assert total['rows'] == 2 * len(draws)
    np.testing.assert_array_equal(total['transition_by_exterior'], 2 * np.asarray(report['transition_by_exterior']))
    with pytest.raises(ValueError, match='aligned'):
        D.transition_summary(b, t.iloc[::-1], mu)


def test_trace_matches_real_lattice_and_rejects_one_ulp():
    g, source, parts = _source()
    lib = _build(source, parts, g, topology=True)
    mu = source[source.sp == 1]
    ranked, first, draws = M.draw_context(mu, L.OffAxisSampler(lib, np.random.default_rng(2)))
    final = M.aggregate(M.trace(ranked, first, draws, g))
    real, _, _ = S.add_offaxis_secondaries(mu, parts.iloc[:1], L.OffAxisSampler(lib, np.random.default_rng(2)), g)
    saved = real[real.sp >= S.OFFAXIS_KEY0]
    assert D.exact_lattice(saved, final)['passed']
    final.loc[0, 'eion'] = np.nextafter(final.loc[0, 'eion'], np.inf)
    with pytest.raises(ValueError, match='saved off-axis'):
        D.exact_lattice(saved, final)


def test_donor_split_preserves_raw_identity_and_uses_disjoint_classes():
    g, source, parts = _source()
    lib = _build(source, parts, g, topology=True)
    mu, first, draws = M.draw_context(source[source.sp == 1], L.OffAxisSampler(lib, np.random.default_rng(2)))
    b, t = M.trace(mu, first, draws, g), M.trace(mu, first, draws, g, 'topology', 'topology')
    for exterior in (True, False):
        mix = D.donor_split(b, t, exterior)
        take = (b.source_exterior != 0) == exterior
        pd.testing.assert_frame_equal(mix[take].reset_index(drop=True), t[take].reset_index(drop=True))
        pd.testing.assert_frame_equal(mix[~take].reset_index(drop=True), b[~take].reset_index(drop=True))
    with pytest.raises(ValueError, match='aligned'):
        D.donor_split(b, t.iloc[::-1])


def test_paired_contrast_matches_scalar_jackknife_and_empty_group_invariance():
    from scripts.secondary_migration_report import contrast
    arrays = [np.array(x, float) for x in ([3, -1, 7], [20, 15, 10], [100, 80, 50], [90, 90, 45])]
    value = lambda xs: xs[0].sum() * xs[3].sum() / (xs[1].sum() * xs[2].sum())
    result = contrast(*arrays)
    replicas = np.array([value([np.delete(x, i) for x in arrays]) for i in range(3)])
    assert result['relative_change'] == value(arrays)
    np.testing.assert_allclose(result['error'], np.sqrt(2 / 3 * ((replicas - replicas.mean()) ** 2).sum()))
    assert result == contrast(*(np.r_[x, 0] for x in arrays))
    assert contrast([0], [1], [1], [1])['error'] is None


def test_selection_uses_full_keys_and_rejects_duplicates():
    f = pd.DataFrame({'run': [1, 2], 'subrun': 1, 'event': 3, 'value': [7, 8]})
    keys = f.iloc[:1][['run', 'subrun', 'event']]
    assert D.restrict(f, keys).value.tolist() == [7]
    with pytest.raises(pd.errors.MergeError):
        D.restrict(f, pd.concat([keys, keys]))
