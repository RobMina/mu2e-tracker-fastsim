"""Synthetic stereo-source replay and independently known pair-cut outcomes."""
import copy
import json

import numpy as np
import pytest

from trkfastsim.reco.hitreco import FLAG
from trkfastsim.reco.stereoseed import stereo_points
from trkfastsim.validation.stereopairaudit import COUNTS, REJECTIONS, audit, main


def fixture(z=(0., 10.), angle=(0., np.pi / 2), ev=None, time=None):
    z, angle = np.asarray(z, float), np.asarray(angle, float)
    n = len(z)
    d = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(n)))
    crossing = np.column_stack((np.full(n, 10.), np.full(n, 20.), z))
    mid = crossing - 10. * d
    cond = {'straw': {**dict(zip(('midx', 'midy', 'midz'), mid.T)),
                     **dict(zip(('dirx', 'diry', 'dirz'), d.T)), 'hlen': np.full(n, 100.)}}
    sh = dict(ev=np.zeros(n, int) if ev is None else np.asarray(ev, int), sid=np.arange(n),
              flag=np.full(n, 1 << FLAG['tdiv'], np.uint32),
              time=np.zeros(n) if time is None else np.asarray(time, float),
              uvar=np.full(n, 100.), timevar=np.full(n, 4.),
              **dict(zip(('x', 'y', 'z'), crossing.T)))
    return sh, cond


def saved(sh, cond, nevents=5):
    source = stereo_points(sh, cond, nevents=nevents)
    assoc = {k: v.copy() for k, v in source.items() if isinstance(v, np.ndarray)
             and k.startswith(('member', 'hit_group', 'group_', 'event_'))}
    assoc.update(evkey=np.column_stack((np.ones(nevents, int), np.zeros(nevents, int), np.arange(nevents))),
                 source_hit_ev=sh['ev'].copy(), source_hit_sid=sh['sid'].copy(),
                 hit_geometry_supported=np.ones(len(sh['ev']), bool),
                 group_candidates_evaluated=np.ones(len(source['group_n']), bool),
                 association_group_support=np.zeros(len(source['group_n']), int))
    for key in ('positions', 'group', 'hit_index', 'ev', 'wire_pull', 'dt'):
        assoc['candidate_' + key] = source[key].copy()
    return assoc, source


def check(out):
    assert np.array_equal(sum(out['group_' + k] for k in COUNTS[1:]), out['group_considered'])
    assert sum(out['diagnostics'][k] for k in COUNTS[1:]) == out['diagnostics']['considered']
    assert out['diagnostics']['candidate_pair_replay_exact']
    assert out['diagnostics']['exclusive_accounting_exact']
    assert out['group_accepted'].sum() == len(out['candidate_hit_index'])


@pytest.mark.parametrize('reason', ['accepted', *REJECTIONS])
def test_independently_known_cuts_and_order(reason):
    sh, cond = fixture()
    if reason == 'parallel':
        cond['straw']['dirx'][:] = 1.
        cond['straw']['diry'][:] = 0.
        sh['y'][:] = cond['straw']['midy']
        sh['z'][:] = cond['straw']['midz'][:] = 0.
        sh['time'][1] = 10.
    elif reason == 'nonfinite':
        cond['straw'].update(midx=np.array([0., 1e308]), midy=np.zeros(2),
            dirx=np.array([0., .3]), diry=np.array([1., np.sqrt(.91)]), hlen=np.full(2, 1e308))
        for key in 'xyz':
            sh[key] = cond['straw']['mid' + key].copy()
    elif reason == 'same_z':
        sh['z'][:] = cond['straw']['midz'][:] = 0.
        sh['time'][1] = 10.
    elif reason == 'far_z':
        sh['z'][1] = cond['straw']['midz'][1] = 50.001
        sh['time'][1] = 10.
    elif reason == 'time':
        sh['time'][1] = 10.
        cond['straw']['hlen'][:] = 9.
    elif reason == 'wire_length':
        cond['straw']['hlen'][:] = 9.
        sh['x'][0] += 4.
        sh['uvar'][0] = 1.
    elif reason == 'wire_distance':
        sh['x'][0] += 4.
        sh['uvar'][0] = 1.
    assoc, source = saved(sh, cond)
    out = audit(sh, cond, assoc, pair_batch=1)
    expected = 'accepted' if reason == 'accepted' else 'rejected_' + reason
    assert out['group_considered'].tolist() == [1]
    for name in COUNTS[1:]:
        assert out['group_' + name].tolist() == [int(name == expected)]
        assert out['diagnostics'][name] == source['diagnostics'][name]
    check(out)


def test_all_groups_exact_batches_original_identity_and_no_mutation():
    sh, cond = fixture(z=np.arange(18) * 3., angle=np.arange(18) * .41,
                       ev=[3] * 10 + [0] * 8, time=list(range(9, -1, -1)) + list(range(8)))
    assoc, source = saved(sh, cond, nevents=7)
    assoc['candidate_hit_index'] = assoc['candidate_hit_index'][::-1, ::-1].copy()
    for name in ('candidate_group', 'candidate_ev'):
        assoc[name] = assoc[name][::-1].copy()
    assoc['group_reject_bits'] = np.full(2, 8 | 16)
    assoc['control_rejectbits'] = np.full(2, 2**30)
    assoc['candidate_used'] = np.zeros(len(assoc['candidate_group']), bool)
    before = copy.deepcopy((sh, cond, assoc))
    reference = audit(sh, cond, assoc)
    assert reference['group_considered'].tolist() == [28, 45]
    np.testing.assert_array_equal(reference['candidate_hit_index'], source['hit_index'])
    for batch in (1, 17):
        out = audit(sh, cond, assoc, pair_batch=batch, max_pairs=73)
        for key in out:
            if key == 'diagnostics':
                assert out[key] == reference[key]
            else:
                np.testing.assert_array_equal(out[key], reference[key])
        check(out)
    for actual, original in zip((sh, cond['straw'], assoc), (before[0], before[1]['straw'], before[2])):
        for key in actual:
            np.testing.assert_array_equal(actual[key], original[key])
    with pytest.raises(ValueError, match='pair limit exceeded.*no pairs truncated'):
        audit(sh, cond, assoc, max_pairs=72)


def test_wide_groups_singletons_flag_exclusion_and_time_cap():
    sh, cond = fixture(z=np.arange(12) * 3., angle=np.arange(12) * .6,
        ev=[0] * 7 + [2] + [3] * 3 + [4], time=list(np.arange(7) * 19.) + [0., 0., 19., 38., 0.])
    sh['flag'][-1] = 0
    sh['timevar'][:] = 1e4
    assoc, source = saved(sh, cond)
    out = audit(sh, cond, assoc, pair_batch=1)
    assert out['group_n'].tolist() == [7, 1, 3]
    assert out['group_raw_pairs'].tolist() == [21, 0, 3]
    assert out['group_considered'].tolist() == [0, 0, 3]
    assert out['group_rejected_time'].tolist() == [0, 0, 1]
    assert out['diagnostics']['excluded_pairs'] == 21
    assert out['diagnostics']['rejected_flag_hits'] == 1
    np.testing.assert_array_equal(out['candidate_hit_index'], source['hit_index'])
    check(out)


def test_unsupported_geometry_excludes_whole_group_not_hits():
    sh, cond = fixture(z=(0., 10., 20., 0., 10.), angle=(0., 1., 2., 0., 1.), ev=[0, 0, 0, 2, 2])
    assoc, _ = saved(sh, cond)
    for key, value in zip(('dirx', 'diry', 'dirz'), (0., 0., 1.)):
        cond['straw'][key][0] = value
    for key in 'xyz':
        sh[key][0] = cond['straw']['mid' + key][0]
    keep = assoc['candidate_group'] != 0
    for key in tuple(assoc):
        if key.startswith('candidate_'):
            assoc[key] = assoc[key][keep]
    assoc['hit_geometry_supported'][0] = False
    assoc['group_candidates_evaluated'][0] = False
    assoc['group_npairs'][0] = assoc['group_npoints'][0] = 0
    assoc['event_npoints'][0] = 0
    out = audit(sh, cond, assoc, pair_batch=1)
    assert out['group_raw_pairs'].tolist() == [3, 1]
    assert out['group_considered'].tolist() == [0, 1]
    assert out['group_unsupported_geometry'].tolist() == [True, False]
    assert out['candidate_hit_index'].tolist() == [[3, 4]]
    check(out)


def test_tilted_3d_geometry_uses_intersection_z_and_normalized_distance():
    sh, cond = fixture()
    d = np.array([[1., 0., 1.], [0., 1., -1.]]) / np.sqrt(2.)
    crossing = np.array([[10., 20., 0.], [10., 20., 50.]])
    mid = crossing - 10. * d
    cond['straw'].update(dict(zip(('dirx', 'diry', 'dirz'), (7. * d).T)))
    cond['straw'].update(dict(zip(('midx', 'midy', 'midz'), mid.T)))
    measured = mid + np.array([15., 5.])[:, None] * d
    sh.update(dict(zip(('x', 'y', 'z'), measured.T)))
    assoc, source = saved(sh, cond)
    out = audit(sh, cond, assoc)
    np.testing.assert_array_equal(out['candidate_hit_index'], source['hit_index'])
    assert out['group_accepted'].tolist() == [1]
    check(out)


@pytest.mark.parametrize('nevents', [0, 4])
def test_empty(nevents):
    sh, cond = fixture(z=(), angle=())
    assoc, _ = saved(sh, cond, nevents=nevents)
    out = audit(sh, cond, assoc, pair_batch=17)
    assert out['candidate_hit_index'].shape == (0, 2)
    assert out['group_considered'].shape == (0,)
    assert out['diagnostics']['events'] == nevents
    check(out)


@pytest.mark.parametrize('key', ['source_hit_ev', 'source_hit_sid', 'hit_group', 'member', 'member_group',
    'group_ev', 'group_n', 'group_span', 'group_npairs', 'group_npoints', 'event_nhits', 'event_ngood',
    'event_ngroups', 'event_npoints', 'candidate_ev', 'hit_geometry_supported', 'group_candidates_evaluated'])
def test_saved_identity_and_count_mismatch(key):
    sh, cond = fixture()
    assoc, _ = saved(sh, cond)
    if assoc[key].dtype.kind == 'b':
        assoc[key][0] = ~assoc[key][0]
    else:
        assoc[key][0] += 1
    with pytest.raises(ValueError, match='mismatch'):
        audit(sh, cond, assoc)


@pytest.mark.parametrize('case', ['pair', 'duplicate', 'missing_candidate', 'group', 'bounds', 'census',
                                 'nonfinite', 'variance', 'position', 'sid', 'mask', 'missing'])
def test_invalid_or_inconsistent_inputs(case):
    sh, cond = fixture(z=(0., 10., 20.), angle=(0., 1., 2.))
    assoc, _ = saved(sh, cond)
    if case == 'pair':
        assoc['candidate_hit_index'][0] = [0, 0]
    elif case == 'duplicate':
        assoc['candidate_hit_index'][0] = assoc['candidate_hit_index'][1]
    elif case == 'missing_candidate':
        for key in tuple(assoc):
            if key.startswith('candidate_'):
                assoc[key] = assoc[key][1:]
        assoc['group_npoints'][0] -= 1
        assoc['event_npoints'][0] -= 1
    elif case == 'group':
        assoc['candidate_group'][0] = -1
    elif case == 'bounds':
        assoc['candidate_hit_index'][0, 0] = 99
    elif case == 'census':
        assoc['evkey'][1] = assoc['evkey'][0]
    elif case == 'nonfinite':
        sh['x'][0] = np.nan
    elif case == 'variance':
        sh['uvar'][0] = 0.
    elif case == 'position':
        sh['z'][0] += 1.
    elif case == 'sid':
        sh['sid'][0] = 96
    elif case == 'mask':
        sh['time'] = np.ma.array(sh['time'], mask=[True, False, False])
    else:
        del assoc['hit_group']
    with pytest.raises(ValueError):
        audit(sh, cond, assoc)


@pytest.mark.parametrize('setting', ['pair_batch', 'max_pairs'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_limits(setting, value):
    sh, cond = fixture()
    assoc, _ = saved(sh, cond)
    with pytest.raises(ValueError):
        audit(sh, cond, assoc, **{setting: value})


def test_cli_hashes_full_census_and_never_overwrites(tmp_path):
    sh, cond = fixture(ev=[2, 2])
    assoc, _ = saved(sh, cond)
    np.savez(tmp_path/'hits.npz', evkey=assoc['evkey'], **{'s_' + k: v for k, v in sh.items()})
    np.savez(tmp_path/'association.npz', **assoc)
    (tmp_path/'conditions.json').write_text(json.dumps(cond, default=lambda a: a.tolist()))
    args = ['--hits', str(tmp_path/'hits.npz'), '--association', str(tmp_path/'association.npz'),
            '--conditions', str(tmp_path/'conditions.json'), '--outdir', str(tmp_path/'out')]
    report = main(args)
    assert report['diagnostics']['accepted'] == 1
    assert all(len(v['sha256']) == 64 for v in report['inputs'].values())
    assert (tmp_path/'out'/'groups.csv').exists()
    assert (tmp_path/'out'/'diagnostics.npz').exists()
    with pytest.raises(FileExistsError):
        main(args)
    assoc['evkey'][0, 0] = 9
    np.savez(tmp_path/'association.npz', **assoc)
    args[-1] = str(tmp_path/'new')
    with pytest.raises(ValueError, match='evkey mismatch'):
        main(args)
    assert not (tmp_path/'new').exists()


def test_masked_geometry_rejected_before_legacy_conversion():
    sh, cond = fixture()
    assoc, _ = saved(sh, cond)
    cond['straw']['hlen'] = np.ma.array(cond['straw']['hlen'], mask=[True, False])
    with pytest.raises(ValueError, match='masked'):
        audit(sh, cond, assoc)


def test_saved_pair_bound_checked_before_canonical_sort():
    sh, cond = fixture()
    assoc, _ = saved(sh, cond)
    assoc['candidate_hit_index'] = np.repeat(assoc['candidate_hit_index'], 2, axis=0)
    with pytest.raises(ValueError, match='exceeds considered'):
        audit(sh, cond, assoc)


def test_float32_geometry_uses_original_prepass():
    from trkfastsim.validation.run1a_seed import geometry_supported
    sh, cond = fixture()
    cond['straw'] = {k: v.astype(np.float32) for k, v in cond['straw'].items()}
    assoc, _ = saved(sh, cond)
    assoc['hit_geometry_supported'] = geometry_supported(sh, cond)
    check(audit(sh, cond, assoc, pair_batch=1))
