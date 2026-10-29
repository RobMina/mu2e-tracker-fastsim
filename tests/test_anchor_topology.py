"""Donor-topology mapping, full self-repasting closure and opt-in identity."""
import numpy as np
import pandas as pd
import pytest
from trkfastsim import library as L, steps as S
from trkfastsim.placement import TOPOLOGY, topology_anchor, validate
from test_library_v2 import _geom, _row, _chord, _build

KEYS = ['run', 'subrun', 'event', 'sp']


def _metadata(**changes):
    row = dict(i=0, source_anchor_plane=10, source_plane_min=10, source_plane_max=12,
               source_exterior=0, source_plane_rank=0, source_plane_count=2, dpl=1)
    return pd.DataFrame([{**row, **changes}])


def test_anchor_metadata_is_opt_in_and_preserves_payload():
    mu = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'k': [0, 1, 2], 'plane': [10, 10, 12],
                       'panel': 2, 'straw': [40, 41, 42], 't': [1., 1.2, 2.]})
    off = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'plane': [9, 11, 13], 'panel': 2,
                        'straw': 50, 't': [1.1, 1.1, 2.1], 'eion': .001, 'len': 2., 'ion': 1,
                        'shape': 0, 'pdg': 11, 'sp': [2, 3, 4]})
    base = L.anchor_offsets(off, mu)
    annotated = L.anchor_offsets(off, mu, topology=True)
    pd.testing.assert_frame_equal(base, annotated[base.columns])
    assert annotated.source_exterior.tolist() == [-1, 0, 1]
    assert annotated.source_anchor_plane.tolist() == [10, 10, 12]
    assert annotated.source_plane_count.tolist() == [2, 2, 1]
    assert (annotated.source_plane_min == 10).all()
    assert (annotated.source_plane_max == 12).all()


@pytest.mark.parametrize('count', [1, 2, 3, 7, 22, 64])
def test_all_within_plane_ranks_self_map_exactly(count):
    mu = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'sp': 1,
                       'k': np.arange(count), 'plane': 10})
    st = pd.concat([_metadata()] * count, ignore_index=True)
    st['source_plane_rank'] = np.arange(count)
    st['source_plane_count'] = count
    st['source_plane_max'] = 10
    st['dpl'] = 0
    np.testing.assert_array_equal(topology_anchor(mu, KEYS, mu.iloc[:1], st), np.arange(count))


@pytest.mark.parametrize('column,value', [('dpl', np.nan), ('dpl', np.inf), ('dpl', .5), ('dpl', -11),
    ('dpl', 26), ('source_anchor_plane', 9), ('source_plane_min', -1), ('source_plane_max', 36),
    ('source_exterior', 1), ('source_plane_rank', -1), ('source_plane_rank', 2),
    ('source_plane_count', 0), ('source_plane_count', 2.5)])
def test_bad_metadata_rejected(column, value):
    with pytest.raises(ValueError):
        validate(_metadata(**{column: value}))


@pytest.mark.parametrize('column', TOPOLOGY + ['dpl'])
def test_missing_metadata_rejected(column):
    with pytest.raises(ValueError, match='annotated'):
        validate(_metadata().drop(columns=column))


@pytest.mark.parametrize('index', [-1, 1, .5, np.nan, np.inf])
def test_bad_recipient_index_rejected(index):
    mu = pd.DataFrame({'run': [1], 'subrun': 1, 'event': 1, 'sp': 1, 'k': 0, 'plane': 10})
    with pytest.raises(ValueError, match='recipient index'):
        topology_anchor(mu, KEYS, mu, _metadata(i=index))


def test_multiple_recipient_keys_and_stable_nearest_plane_tie():
    mu = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'sp': [1, 1, 1, 5, 5, 5],
                       'k': [0, 1, 2, 0, 1, 2], 'plane': [10, 10, 12, 20, 22, 22]})
    first = mu.drop_duplicates(KEYS)
    st = pd.concat([_metadata(i=1, source_anchor_plane=11, dpl=0), _metadata(),
                    _metadata(i=1, source_anchor_plane=12, source_exterior=1)], ignore_index=True)
    np.testing.assert_array_equal(topology_anchor(mu, KEYS, first, st), [0, 0, 1])
    with pytest.raises(ValueError, match='duplicate'):
        topology_anchor(pd.concat([mu, mu.iloc[:1]]), KEYS, first, st)
    with pytest.raises(ValueError, match='missing recipient'):
        topology_anchor(mu[mu.sp == 1], KEYS, first, st)


def test_degenerate_recipient_extent_does_not_guarantee_internal_gap():
    mu = pd.DataFrame({'run': [1], 'subrun': 1, 'event': 1, 'sp': 1, 'k': 0, 'plane': 20})
    k = topology_anchor(mu, KEYS, mu, _metadata())
    assert (mu.iloc[k].plane.to_numpy() + _metadata().dpl.to_numpy()).tolist() == [21]


def _source():
    g = _geom()
    direction = L._unit(np.array([[.2, -1., 0.]]))[0]
    rows = []
    for plane, straw, t in [(10, 20 + k, float(k)) for k in range(22)] + [(12, 40, 30.), (12, 41, 31.)]:
        j = (plane * 6 + 2) * 96 + straw
        q0, q1 = _chord(g, j, direction, .4, t)
        rows.append(_row(g, j, q0, q1, 3000 * direction, 1, t))
    for sp, (plane, t) in enumerate([(9, 15.1), (11, 15.1), (11, 30.1), (13, 31.1), (10, 15.1)], 2):
        j = (plane * 6 + 2) * 96 + 70
        rows.append(_row(g, j, g.mid[j] + [30., .2, .3], g.mid[j] + [31., .3, .4],
                         [0., 0., 0.], sp, t, eion=sp * .001))
    steps = pd.DataFrame(rows).assign(nsgs=len(rows))
    parts = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'key': np.arange(1, 7),
                          'pdg': [13] + [11] * 5, 'isprim': [1] + [0] * 5,
                          'ismcprim': [1] + [0] * 5, 'parent': [-1] + [1] * 5})
    return g, steps, parts


def test_annotated_build_self_repaste_full_payload_and_rng_identity():
    g, source, parts = _source()
    base, annotated = _build(source, parts, g), _build(source, parts, g, topology=True)
    for tree in base:
        pd.testing.assert_frame_equal(base[tree], annotated[tree][base[tree].columns])
    assert annotated['offaxis'].source_exterior.tolist() == [-1, 0, 0, 1, 0]
    assert annotated['offaxis'].source_plane_rank.tolist() == [15, 15, 0, 1, 15]
    mu, p = source[source.sp == 1].copy(), parts.iloc[:1].copy()
    outputs, states, draws = [], [], []
    for lib, options in [(base, {}), (annotated, {'topology_anchor': False}),
                         (annotated, {'topology_anchor': True})]:
        sampler = L.OffAxisSampler(lib, np.random.default_rng(19), **options)
        outputs.append(S.add_offaxis_secondaries(mu, p, sampler, g))
        states.append(sampler.rng.bit_generator.state)
        draws.append(sampler.last_concentration)
    for a, b in zip(outputs[0][:2], outputs[1][:2]):
        pd.testing.assert_frame_equal(a, b)
    assert outputs[0][2] == outputs[1][2]
    assert states[0] == states[1] == states[2] and draws[0] == draws[1] == draws[2]
    out, outparts, stats = outputs[2]
    secondary = out[out.sp >= S.OFFAXIS_KEY0].sort_values('sp').reset_index(drop=True)
    donor = source[source.sp != 1].sort_values('sp').reset_index(drop=True)
    for column in ['sid', 'plane', 'panel', 'straw', 'shape', 'ion', 'eion', 'len', 't']:
        np.testing.assert_array_equal(secondary[column], donor[column])
    for column in ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']:
        np.testing.assert_allclose(secondary[column], donor[column], atol=1e-10, rtol=0)
    assert outparts.loc[outparts.key >= S.OFFAXIS_KEY0, 'parent'].tolist() == [1] * 5
    assert stats['placed'] == 5 and stats['reflected'] == stats['merged_same_particle'] == 0
    with pytest.raises(ValueError, match='mutually exclusive'):
        L.OffAxisSampler(annotated, np.random.default_rng(1), edge_anchor=True, topology_anchor=True)
    with pytest.raises(ValueError, match='annotated'):
        L.OffAxisSampler(base, np.random.default_rng(1), topology_anchor=True)


@pytest.mark.parametrize('shift', [-10, 8, 23])
def test_detector_edge_reflection_and_multiple_muon_parent_identity(shift):
    g, source, parts = _source()
    lib = _build(source, parts, g, topology=True)
    mu = source[source.sp == 1].copy()
    moved = mu.copy().assign(plane=mu.plane + shift, sp=5, t=mu.t + 100.)
    js = ((moved.plane * 6 + moved.panel) * 96 + moved.straw).to_numpy(dtype=int)
    moved['sid'], moved['upanel'] = g.sid[js], g.upanel[js]
    for end in '01':
        moved[f'z{end}'] += shift * 300.
    p = pd.concat([parts.iloc[:1], parts.iloc[:1].assign(key=5)], ignore_index=True)
    fast = pd.concat([mu, moved], ignore_index=True)
    out, outparts, stats = S.add_offaxis_secondaries(fast, p,
        L.OffAxisSampler(lib, np.random.default_rng(3), topology_anchor=True), g)
    sec = out[out.sp >= S.OFFAXIS_KEY0].merge(outparts[['key', 'parent']], left_on='sp', right_on='key')
    assert sec.groupby('parent').size().to_dict() == {1: 5, 5: 5}
    second = sec[sec.parent == 5].sort_values('sp')
    np.testing.assert_array_equal(second.plane, S._mirror(np.array([9, 11, 11, 13, 10]) + shift, 36))
    np.testing.assert_allclose(second.t, np.array([15.1, 15.1, 30.1, 31.1, 15.1]) + 100.)
    assert stats['reflected'] == int(shift in (-10, 23))
    assert stats['placed'] == 10 and stats['merged_same_particle'] == 0


def test_unequal_crossing_density_keeps_true_exterior_on_boundary():
    g, source, parts = _source()
    lib = _build(source, parts, g, topology=True)
    mu = source[(source.sp == 1) & ((source.plane == 12) | (source.t == 0))].copy()
    out, _, _ = S.add_offaxis_secondaries(mu, parts.iloc[:1].copy(),
        L.OffAxisSampler(lib, np.random.default_rng(3), topology_anchor=True), g)
    assert out.loc[out.sp >= S.OFFAXIS_KEY0].sort_values('sp').plane.tolist() == [9, 11, 11, 13, 10]


def test_empty_annotated_offaxis_library():
    g, source, parts = _source()
    lib = _build(source[source.sp == 1], parts.iloc[:1], g, topology=True)
    sampler = L.OffAxisSampler(lib, np.random.default_rng(1), topology_anchor=True)
    mu = source[source.sp == 1].copy()
    out, p, stats = S.add_offaxis_secondaries(mu, parts.iloc[:1], sampler, g)
    pd.testing.assert_frame_equal(out, mu)
    pd.testing.assert_frame_equal(p, parts.iloc[:1])
    assert stats == {'drawn': 0, 'placed': 0}


def test_nearest_plane_tie_uses_exact_rational_distance():
    mu = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'sp': 1,
                       'k': [0, 1, 2, 3], 'plane': [0, 13, 14, 21]})
    st = _metadata(source_anchor_plane=9, source_plane_min=0, source_plane_max=14, dpl=0)
    assert (9 / 14) * 21 > 13.5
    assert topology_anchor(mu, KEYS, mu.iloc[:1], st).tolist() == [1]


@pytest.mark.parametrize('plane', [-1, 36, 1.5, np.nan])
def test_malformed_recipient_plane_refused(plane):
    mu = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'sp': 1, 'k': [0], 'plane': [plane]})
    with pytest.raises(ValueError, match='recipient plane'):
        topology_anchor(mu, KEYS, mu, _metadata())
