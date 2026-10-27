"""Detector-only candidate identities, empty events and the existing F1 boundary."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from trkfastsim.reco import autonomous as A, hitreco, ntuple, tracktable
from test_reco_jax import _cond
from kalman_toy import DRIFT, make_track


def candidates():
    xt, h, _, _ = make_track(np.random.default_rng(11), n=12, smear=False)
    cond = _cond(96, np.random.default_rng(12))
    st = cond['straw']
    st['hlen'][:] = 0
    for i, c in enumerate('xyz'):
        st['mid' + c][:12] = h['mid'][:, i]
        st['dir' + c][:12] = h['wdir'][:, i]
    st['hlen'][:12] = h['hlen']
    sh = {'ev': np.repeat([0, 0, 2], 12), 'sid': np.tile(np.arange(12), 3),
          'time': np.repeat([10., 210., 10.], 12), 'flag': np.zeros(36, np.uint32),
          'halfpv': np.tile(h['vsig'] / 2, 3), 'eend': np.zeros(36, np.int64)}
    for key, src in [('et0', 'tmeas'), ('et1', 'tmeas'), ('dtime', 'totdt'), ('timevar', 'tvar'),
                     ('uvar', 'uvar'), ('wdist', 'wdist'), ('edep', 'edep')]:
        sh[key] = np.tile(h[src], 3)
    sh['et0'][12:24] += 200
    sh['et1'][12:24] += 200
    sh['tot0'] = sh['tot1'] = np.ones(36)
    sd = {k: np.full(3, xt[i]) for i, k in enumerate(('a0', 'b0', 'a1', 'b1', 't0'))}
    sd['t0'][1] += 200
    sd.update(ev=np.array([0, 0, 2]), n=np.full(3, 12), member=np.arange(36))
    return sh, sd, cond


def test_multiseed_candidates_are_event_and_hit_identity_bound():
    sh, sd, cond = candidates()
    x, H, X = A.build_candidates(sh, sd, cond)
    assert x.shape == (3, 6)
    assert np.array_equal(H['sh'], np.arange(36).reshape(3, 12))
    assert H['inseed'].all()
    assert np.array_equal(X['hit'], np.tile(np.arange(12), (3, 1)))
    assert np.array_equal(X['sid'], np.tile(np.arange(12), (3, 1)))
    assert np.all(np.isfinite(x))
    assert np.allclose(x[:, 5], 2000, atol=.001)


def test_repeat_straw_hits_keep_distinct_associations():
    sh, sd, cond = candidates()
    sh['sid'][1] = sh['sid'][0]
    _, H, X = A.build_candidates(sh, sd, cond)
    assert X['sid'][0, 0] == X['sid'][0, 1]
    assert np.array_equal(X['hit'][0, :2], [0, 1])
    assert np.array_equal(H['sh'][0, :2], [0, 1])


def test_hitless_crossings_include_both_sides_and_exclude_wire_end_overrun():
    sh, sd, cond = candidates()
    st = cond['straw']
    x = A.seed_parameters(sd)[0]
    u = np.array([x[2], -1., x[3]]); u /= np.linalg.norm(u)
    for i, distance in enumerate([-900., 900., 0.], 12):
        p = np.array([x[0], 0., x[1]]) + distance * u
        p[0] += 1000 if i == 14 else 0
        for j, c in enumerate('xyz'):
            st['mid' + c][i] = p[j]
            st['dir' + c][i] = float(c == 'x')
        st['hlen'][i] = 100.
    _, _, X = A.build_candidates(sh, sd, cond)
    assert set(X['sid'][0, X['valid'][0]]) == set(range(14))
    assert np.all(X['hit'][0, 12:14] == -1)


def test_extension_hit_mask_and_wrong_event_membership():
    sh, sd, cond = candidates()
    sd['n'] = np.full(3, 11)
    sd['member'] = np.arange(36).reshape(3, 12)[:, :11].ravel()
    _, H, _ = A.build_candidates(sh, sd, cond)
    assert H['valid'].sum() == 36
    assert H['inseed'].sum() == 33
    sd['member'][-1] = 0
    with pytest.raises(ValueError, match='event identity'):
        A.build_candidates(sh, sd, cond)


def write_empty(path, events=(1, 2, 3)):
    with uproot.recreate(path) as f:
        f.mktree('trkdump/events', {'run': 'int64', 'subrun': 'int64', 'event': 'int64', 'pbt': 'float32',
                 **{k: 'var * int64' for k in ('d_sid', 'd_tdc0', 'd_tdc1', 'd_tot0', 'd_tot1', 'd_nadc', 'd_adc')}})
        empty = ak.unflatten(np.empty(0, np.int64), np.zeros(len(events), np.int64))
        f['trkdump/events'].extend({'run': np.ones(len(events), np.int64), 'subrun': np.ones(len(events), np.int64),
                                   'event': np.asarray(events), 'pbt': np.zeros(len(events), np.float32),
                                   **{k: empty for k in ('d_sid', 'd_tdc0', 'd_tdc1', 'd_tot0', 'd_tot1', 'd_nadc', 'd_adc')}})


def test_real_digi_only_empty_chain_writer_and_roundtrip(tmp_path):
    path = tmp_path / 'digis.root'
    write_empty(path)
    cond = _cond(96, np.random.default_rng(12))
    table = A.reconstruct(path, cond, DRIFT)
    assert np.array_equal(table['events']['event'], [1, 2, 3])
    assert len(table['ev']) == 0
    assert not table['meta']['stock_conditioned']
    tracktable.save(table, tmp_path / 'tracks.npz')
    loaded = tracktable.load(tmp_path / 'tracks.npz')
    assert np.array_equal(loaded['events']['event'], [1, 2, 3])
    ntuple.write(table, tmp_path / 'EventNtuple.root')
    with uproot.open(tmp_path / 'EventNtuple.root') as f:
        tree = f[ntuple.TREE]
        assert tree.num_entries == 3
        assert ak.sum(ak.num(tree['trk'].array()['trk.status'])) == 0


def test_duplicate_full_event_identity_rejected(tmp_path):
    path = tmp_path / 'duplicate.root'
    write_empty(path, (1, 1))
    with pytest.raises(ValueError, match='duplicate full event'):
        A.reconstruct(path, _cond(96, np.random.default_rng(12)), DRIFT)


def test_bad_geometry_rejected():
    sh, sd, cond = candidates()
    cond['straw']['dirx'][0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        A.build_candidates(sh, sd, cond)


def test_reconstruct_preserves_failed_and_multiple_seeds(monkeypatch, tmp_path):
    from trkfastsim.reco import linefind_jax
    sh, sd, cond = candidates()
    d = {'evkey': np.array([[1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4]]), 'pbt': np.zeros(4)}
    monkeypatch.setattr(hitreco, 'read_digis', lambda paths: d)
    monkeypatch.setattr(hitreco, 'straw_hits', lambda *args: sh)
    monkeypatch.setattr(linefind_jax, 'line_seeds', lambda *args: sd)
    def fake_fit(x, H, X, drift, chunk, buckets=False):
        return {'x': x, 'cov': np.tile(np.eye(6), (3, 1, 1)), 'chi2': np.ones(3), 'ndof': np.full(3, 10),
                'status': np.array([0, 2, 0]), 'good': np.array([True, False, True]), 'active': H['valid'],
                'lr': H['lr'], 'xactive': X['valid'],
                'material': {k: np.zeros(X['valid'].shape) for k in ('wall', 'gas', 'doca', 'docavar', 'dirdot')}}, {}
    monkeypatch.setattr(A, 'fit_candidates', fake_fit)
    table = A.reconstruct('no-stock-file', cond, DRIFT)
    assert np.array_equal(table['ev'], [0, 0, 2])
    assert np.array_equal(table['seed_index'], [0, 1, 0])
    assert np.array_equal(table['goodfit'], [1, 0, 1])
    assert table['meta']['events_without_seeds'] == 2
    assert table['meta']['events_with_multiple_seeds'] == 1
    ntuple.write(table, tmp_path / 'multi.root')
    with uproot.open(tmp_path / 'multi.root') as f:
        tree = f[ntuple.TREE]
        assert tree.num_entries == 4
        assert np.array_equal(ak.to_numpy(ak.num(tree['trk'].array()['trk.status'])), [1, 0, 1, 0])


def test_explicit_crossing_association_survives_hit_sort_and_serialization(tmp_path):
    from test_ntuple_writer import _table, VERT
    table = _table([VERT], nhit=4)
    table['sid'][0, 1] = table['sid'][0, 0]
    table['xings']['sid'][0, 1] = table['sid'][0, 0]
    table['state'][0] = [-2, 1, 0, -2]
    table['xings']['hit'] = np.array([[0, 1, -1, 3]])
    table['mid'] = table['mid'][:, ::-1].copy()
    tracktable.save(table, tmp_path / 'association.npz')
    table = tracktable.load(tmp_path / 'association.npz')
    result = ntuple.compute(table)['trkmats']
    assert np.array_equal(result['hashit'], [True, True, False, True])
    assert np.array_equal(result['activehit'], [False, True, False, False])
    assert np.array_equal(result['drifthit'], [False, True, False, False])
    ntuple.write(table, tmp_path / 'association.root')
    with uproot.open(tmp_path / 'association.root') as root:
        mat = ak.flatten(root[ntuple.TREE]['trkmats'].array(), axis=None)
        assert len(mat) > 0
    table['xings']['hit'][0, 1] = 99
    with pytest.raises(ValueError, match='outside original'):
        ntuple.compute(table)


def test_nonempty_fitter_adapter(monkeypatch):
    import jax.numpy as jnp
    from trkfastsim.reco import kalman_jax as K
    sh, sd, cond = candidates()
    x, H, X = A.build_candidates(sh, sd, cond)
    monkeypatch.setattr(K, 'load_anns', lambda: {})
    def fake_fit(xx, hh, cc, drift, stages, ann, **kwargs):
        assert kwargs == {'fixed': None, 'time_model': 'stock'}
        assert set(cc) == {'mid', 'wdir', 'valid', 'hit'}
        n = len(hh['valid']) + len(cc['valid'])
        cf = {'xe': jnp.tile(xx, (n, 1)), 'Ce': jnp.tile(jnp.eye(6), (n, 1, 1)),
              'chi2': jnp.array(1.), 'ndof': jnp.array(10), 'status': jnp.array(0),
              'st': {'active': hh['valid'], 'lr': jnp.zeros_like(hh['lr'])}, 'xact': cc['valid']}
        return cf, None, jnp.array(True)
    monkeypatch.setattr(K, 'fit_track', fake_fit)
    monkeypatch.setattr(K, 't0_piece', lambda cf: (cf['xe'][0], cf['Ce'][0], 0))
    monkeypatch.setattr(K, 'crossing_rows', lambda xx, cov, cc, nsig, varscale:
                        {k: jnp.zeros(cc['valid'].shape) for k in ('wall', 'gas', 'doca', 'docavar', 'dirdot')})
    out, timing = A.fit_candidates(x, H, X, DRIFT, chunk=2)
    assert np.array_equal(out['x'], x)
    assert out['active'].shape == H['valid'].shape
    assert timing['real_tracks'] == 3 and timing['executed_lanes'] == 4
    assert timing['warm_tracks'] == 1 and timing['warm_executed_lanes'] == 2


def test_invalid_padding_association_is_not_dereferenced():
    from test_ntuple_writer import _table, VERT
    table = _table([VERT], nhit=4)
    table['xings']['hit'] = np.array([[0, 1, 2, 999]])
    table['xings']['xvalid'][0, 3] = False
    result = ntuple.compute(table)['trkmats']
    assert len(result['hashit']) == 3


def test_candidate_bucket_plan_preserves_rows_and_masked_holes():
    from trkfastsim.reco.autonomous_batches import batch_plan
    h = np.zeros((5, 84), bool)
    x = np.zeros((5, 219), bool)
    h[:, :10] = True; x[:, :10] = True
    h[1, 50] = True; x[4, 218] = True
    h[3, 20:40] = True; x[3, 100] = True
    plan = batch_plan(h, x, 2, buckets=True)
    assert np.array_equal(np.sort(np.concatenate([p[0] for p in plan])), np.arange(5))
    for ids, nh, nx in plan:
        assert not h[ids, nh:].any() and not x[ids, nx:].any()
        assert len(ids) <= 2
    work = sum(2 * (nh + nx) for _, nh, nx in plan)
    assert work < len(plan) * 2 * (84 + 219)
    assert batch_plan(h[:0], x[:0], 2, buckets=True) == []
    with pytest.raises(ValueError, match='positive integer'):
        batch_plan(h, x, 0)


def test_cached_kernel_binds_calibration_values_not_mutable_arrays(monkeypatch):
    from trkfastsim.reco import kalman_jax as K
    def fit(xx, hh, cc, drift, stages, ann, **kw):
        return {}, None, True
    tree = {'v': np.array([1., 2.])}
    frozen = A._frozen_tree(tree)
    k1 = A._fit_kernel(fit, K.t0_piece, K.crossing_rows, frozen, frozen, frozen)
    assert k1 is A._fit_kernel(fit, K.t0_piece, K.crossing_rows, A._frozen_tree(tree), frozen, frozen)
    tree['v'][0] = 3.
    assert k1 is not A._fit_kernel(fit, K.t0_piece, K.crossing_rows, A._frozen_tree(tree), frozen, frozen)


def test_real_f1_bucket_adapter_against_full_width():
    from trkfastsim.reco import kalman_jax as K
    try:
        K.load_anns()
    except FileNotFoundError:
        pytest.skip('KinKal ANN weights not available')
    sh, sd, cond = candidates()
    x, H, X = A.build_candidates(sh, sd, cond)
    for row, start in ((0, 8), (1, 10)):
        H['valid'][row, start:] = False
        H['inseed'][row, start:] = False
        X['valid'][row, start:] = False
    from trkfastsim.reco.autonomous_batches import restore_width
    for key in H:
        fill = 1 if key in ('vsig', 'hlen', 'uvar', 'tvar', 'esign') else 0
        H[key] = restore_width(H[key], 40, fill)
    H['wdir'][:, 12:, 0] = 1
    for key in X:
        X[key] = restore_width(X[key], 64, -1 if key == 'hit' else 0)
    X['wdir'][:, 12:, 0] = 1
    full, _ = A.fit_candidates(x, H, X, DRIFT, chunk=2)
    bucket, timing = A.fit_candidates(x, H, X, DRIFT, chunk=2, buckets=True)
    assert np.array_equal(full['good'], bucket['good'])
    assert np.array_equal(full['status'], bucket['status'])
    assert np.array_equal(full['active'][H['valid']], bucket['active'][H['valid']])
    assert np.array_equal(full['lr'][H['valid']], bucket['lr'][H['valid']])
    np.testing.assert_allclose(full['x'], bucket['x'], rtol=1e-7, atol=1e-7)
    assert timing['executed_effect_slots'] < timing['full_width_effect_slots']


def test_kpp_reference_config_is_unfiltered_and_ascii():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'art/TrkFastSim/fcl/dump_tracks_data_autonomous.fcl'
    text = path.read_bytes().decode('ascii')
    assert '#include "reco_data_vst.fcl"' in text
    assert 'geom_common_extracted_v04.txt' in text
    assert 'DumpTracks : true' in text and 'SaveAllFits : true' in text
    assert 'outputs : {}' in text
    path_line = next(row for row in text.splitlines() if row.startswith('physics.RecoPath'))
    for rejected in ('KLFilter', 'SelectReco', 'CaloReco'):
        assert rejected not in path_line
    assert all(name in path_line for name in ('PBTFSD', 'makeSH', 'SimpleTimeCluster', 'LineFinder', 'KKLine'))
