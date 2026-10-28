"""Shared-draw diagnostics: production parity, accounting, lattice-only hybrids."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim import geometry as G, library as L, steps as S, tables as T
from trkfastsim.analyses import secmigration as M, seccensus as C
from test_library_v2 import _geom, _row


def _fixture():
    geom = _geom()
    rows = []
    for event, sp, plane, panel, straw, t in [
        (1, 1, 0, 2, 0, 10.), (1, 1, 0, 3, 95, 11.), (1, 1, 2, 4, 40, 12.),
        (1, 5, 1, 2, 1, 20.), (1, 5, 3, 2, 20, 21.),
        (2, 1, 0, 2, 0, 30.), (2, 1, 0, 3, 95, 31.), (2, 1, 2, 4, 40, 32.)]:
        j = (plane * 6 + panel) * 96 + straw
        rows.append({**_row(geom, j, geom.mid[j] - [0, 1, 0], geom.mid[j] + [0, 1, 0],
                            [0, -3000, 0], sp, t), 'event': event})
    primary = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
    primary['nsgs'] = primary.groupby(T.KEY).sid.transform('size')
    parts = primary[T.KEY + ['sp']].drop_duplicates().rename(columns={'sp': 'key'}).assign(
        parent=-1, pdg=13, isprim=1, ismcprim=1)
    # Shared straws exercise same-particle merges, distinct-particle survival, invalid targets and event-wide vetoes.
    off = pd.DataFrame([
        dict(sp=20, k=0, dpl=-1, dst=-1),
        dict(sp=21, k=0, dpl=-1, dst=-2),
        dict(sp=21, k=0, dpl=1, dst=2),
        dict(sp=22, k=0, dpl=0, dst=0),
        dict(sp=23, k=0, dpl=0, dst=-300),
        dict(sp=24, k=21, dpl=1, dst=1),
        dict(sp=25, k=15, dpl=1, dst=2),
        dict(sp=26, k=15, dpl=0, dst=0),
        dict(sp=27, k=0, dpl=1, dst=2),
    ])
    off = off.assign(run=99, subrun=98, event=97, dpa=0, shape=0, ion=1, pdg=11,
                     eion=np.arange(1, len(off) + 1) * .001, len=np.arange(1, len(off) + 1) * 1.,
                     dt=[.9, .8, .2, .3, .4, .5, .6, .7, .1],
                     source_anchor_plane=np.where(off.k == 21, 12, 10),
                     source_plane_min=10, source_plane_max=12,
                     source_plane_rank=np.where(off.k == 15, 15, 0),
                     source_plane_count=np.where(off.k == 21, 2, 20))
    target = off.source_anchor_plane + off.dpl
    off['source_exterior'] = np.where(target < 10, -1, np.where(target > 12, 1, 0))
    events = pd.DataFrame(dict(run=[99], subrun=98, event=97, uy=-1.,
                               pbin=np.digitize([3000.], L.P_BINS) - 1, nstraw_mu=22,
                               plane_min=10, plane_max=12))
    return geom, primary, parts, dict(events=events, offaxis=off)


def _sampler(lib, topology=False):
    return L.OffAxisSampler(lib, np.random.default_rng(19), plane_bin_width=0, topology_anchor=topology)


@pytest.mark.parametrize('mode', ['default', 'topology'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('v2', [False, True])
def test_exact_production_parity_and_conservation(mode, reverse, v2):
    geom, primary, parts, lib = _fixture()
    if v2:
        lib['offaxis'] = lib['offaxis'].assign(du0=1., du1=2., r0=.5, r1=.5, phi0=.1, phi1=.2, mub=.5)
    if reverse:
        lib['offaxis'] = lib['offaxis'].iloc[::-1].reset_index(drop=True)
        # Stable time ties must preserve the input occurrence, not donor sp order.
        lib['offaxis']['dt'] = 0.
        primary = primary.iloc[::-1].reset_index(drop=True)
    before = primary.copy(deep=True)
    sampler = _sampler(lib)
    mu, first, st = M.draw_context(primary, sampler)
    saved = st.copy(deep=True)
    state = sampler.rng.bit_generator.state
    tr = M.trace(mu, first, st, geom, mode, mode)
    final = M.aggregate(tr)
    prod_sampler = _sampler(lib, topology=mode == 'topology')
    out, particles, stats = S.add_offaxis_secondaries(primary, parts, prod_sampler, geom)
    actual = out[out.sp >= S.OFFAXIS_KEY0].reset_index(drop=True)
    cols = T.KEY + ['sp', 'sid', 'plane', 'panel', 'straw', 'upanel', 'eion', 'len', 't']
    pd.testing.assert_frame_equal(final[cols], actual[cols], check_dtype=False)
    parents = particles[particles.key >= S.OFFAXIS_KEY0].rename(columns={'key': 'sp'})
    paired = final.merge(parents[T.KEY + ['sp', 'parent']], on=T.KEY + ['sp'], suffixes=('', '_real'))
    assert (paired.parent == paired.parent_real).all()
    assert len(tr) == len(st) == stats['drawn'] == 27
    assert tr.draw_row.is_unique and first.sp.tolist() == [1, 5, 1]
    assert first.donor_index.tolist() == [0, 0, 0]
    assert tr.keep.sum() == stats['placed'] == final.raw_n.sum()
    assert (~tr.valid).sum() == stats['off_lattice'] == 3
    assert tr.in_muon.sum() == stats['in_muon_straw'] > 0
    assert ((tr.reflected_plane | tr.reflected_straw) & tr.valid).sum() == stats['reflected'] > 0
    assert tr.keep.sum() - len(final) == stats['merged_same_particle'] > 0
    assert len(tr) == (~tr.valid).sum() + tr.in_muon.sum() + final.raw_n.sum()
    assert final.eion.sum() == pytest.approx(tr.loc[tr.keep, 'eion'].sum())
    assert final.len.sum() == pytest.approx(tr.loc[tr.keep, 'len'].sum())
    # The veto is event-wide: parent 1's first row hits parent 5, but survives in event 2.
    veto = tr[(tr.parent == 1) & (tr.gsp == 20)]
    assert veto.in_muon.tolist() == [True, False]
    assert final.duplicated(T.KEY + ['sid']).any()  # distinct particles, same occupancy
    assert state == sampler.rng.bit_generator.state == prod_sampler.rng.bit_generator.state
    pd.testing.assert_frame_equal(st, saved)
    pd.testing.assert_frame_equal(primary, before)
    # Classification agrees with the census while never inspecting endpoints.
    census = C._classify(out.assign(mu=out.sp < S.SECONDARY_KEY0), geom)
    np.testing.assert_array_equal(M.classify(final, mu), census.loc[~census.mu, 'loc'])


def test_hybrids_switch_only_requested_lattice_coordinates_and_not_time():
    geom, primary, _, lib = _fixture()
    mu, first, st = M.draw_context(primary, _sampler(lib))
    base = M.trace(mu, first, st, geom)
    topo = M.trace(mu, first, st, geom, 'topology', 'topology')
    plane = M.trace(mu, first, st, geom, 'topology', 'default')
    transverse = M.trace(mu, first, st, geom, 'default', 'topology')
    np.testing.assert_array_equal(plane.plane, topo.plane)
    np.testing.assert_array_equal(plane[['panel', 'straw']], base[['panel', 'straw']])
    np.testing.assert_array_equal(transverse.plane, base.plane)
    np.testing.assert_array_equal(transverse[['panel', 'straw']], topo[['panel', 'straw']])
    for hybrid in (plane, transverse):
        np.testing.assert_array_equal(hybrid.t, base.t)
        np.testing.assert_array_equal(hybrid.draw_row, base.draw_row)
        assert not any(c in hybrid for c in ('x0', 'y0', 'z0'))
    assert (base.t != topo.t).any()
    assert (base.target_plane != topo.target_plane).any()
    assert (base.target_straw != topo.target_straw).any()
    assert (base.donor_run == 99).all() and (base.run == 1).all()
    with pytest.raises(ValueError, match='modes'):
        M.trace(mu, first, st, geom, 'edge')
    with pytest.raises(ValueError, match='uniquely'):
        M.trace(mu, first, st.assign(draw_row=0), geom)


def test_raw_order_permutation_preserves_occurrence_identity_and_aggregation():
    geom, primary, _, lib = _fixture()
    mu, first, st = M.draw_context(primary, _sampler(lib))
    for mode in ('default', 'topology'):
        a = M.trace(mu, first, st, geom, mode, mode)
        b = M.trace(mu, first, st.sample(frac=1, random_state=4), geom, mode, mode)
        pd.testing.assert_frame_equal(a.sort_values('draw_row').reset_index(drop=True),
                                      b.sort_values('draw_row').reset_index(drop=True))
        pd.testing.assert_frame_equal(M.aggregate(a), M.aggregate(b))


def test_float_relative_rank_is_not_integer_division():
    geom, primary, _, lib = _fixture()
    # 15/22*22 rounds below 15 in binary floating point: production selects 14.
    primary = pd.concat([primary.iloc[:1]] * 22, ignore_index=True).assign(t=np.arange(22.), sp=1)
    mu, first, st = M.draw_context(primary, _sampler(lib))
    tr = M.trace(mu, first, st, geom)
    assert tr.loc[tr.donor_k == 15, 'anchor_default_k'].tolist() == [14, 14]
    no_count = M.trace(mu, first, st.drop(columns='nstraw_mu'), geom)
    assert no_count.loc[no_count.donor_k == 15, 'anchor_default_k'].tolist() == [15, 15]


def test_classify_all_classes_uses_final_lattice_and_all_muons():
    geom, primary, _, _ = _fixture()
    # Same other muon's straw; its near panel; other panel; near plane; far plane; invalid.
    plane = np.array([1, 1, 1, 4, 8, 0, 1])
    panel = np.array([2, 2, 3, 0, 0, 0, 2])
    straw = np.array([1, 4, 1, 1, 1, 0, 5])
    s = pd.DataFrame(dict(run=1, subrun=1, event=1, plane=plane, panel=panel, straw=straw,
                          upanel=plane * 6 + panel, sid=G.straw_id(plane, panel, straw),
                          valid=[True, True, True, True, True, False, True]))
    np.testing.assert_array_equal(M.classify(s, primary), [0, 1, 2, 3, 4, -1, 2])
    s['event'] = 99
    np.testing.assert_array_equal(M.classify(s, primary), [4, 4, 4, 4, 4, -1, 4])


@pytest.mark.parametrize('empty_mu', [False, True])
def test_empty_context_and_aggregate_have_usable_schema(empty_mu):
    geom, primary, _, lib = _fixture()
    if empty_mu:
        primary = primary.iloc[:0]
    else:
        lib['offaxis'] = lib['offaxis'].iloc[:0]
    sampler = _sampler(lib)
    state = sampler.rng.bit_generator.state
    mu, first, st = M.draw_context(primary, sampler)
    tr = M.trace(mu, first, st, geom)
    final = M.aggregate(tr)
    assert tr.empty and final.empty and len(M.classify(final, mu)) == 0
    assert set(['sp', 'raw_n', 'sid', 'eion', 'len', 't']).issubset(final)
    if empty_mu:
        assert sampler.rng.bit_generator.state == state


def test_only_vetoed_rows_do_not_allocate_particle_numbers():
    geom, primary, _, lib = _fixture()
    mu, first, st = M.draw_context(primary, _sampler(lib))
    tr = M.trace(mu, first, st, geom)
    assert M.aggregate(tr[~tr.keep]).empty
    # Existing in-straw/off-axis rows must never become recipients.
    contaminated = pd.concat([primary, primary.assign(sp=S.SECONDARY_KEY0)])
    clean = M.draw_context(contaminated, _sampler(lib))
    pd.testing.assert_frame_equal(mu, clean[0])
    pd.testing.assert_frame_equal(first, clean[1])
    pd.testing.assert_frame_equal(st, clean[2])


def test_primary_time_ties_follow_production_input_order():
    geom, primary, parts, lib = _fixture()
    tied = (primary.event == 1) & (primary.sp == 1) & (primary.plane == 0)
    primary.loc[tied, 't'] = 10.
    panels = []
    for steps in (primary, primary.iloc[::-1]):
        mu, first, st = M.draw_context(steps, _sampler(lib))
        tr = M.trace(mu, first, st, geom)
        panels.append(tr.loc[(tr.event == 1) & (tr.parent == 1), 'anchor_default_panel'].iloc[0])
        final = M.aggregate(tr)
        out, _, _ = S.add_offaxis_secondaries(steps, parts, _sampler(lib), geom)
        cols = T.KEY + ['sp', 'sid', 'eion', 'len', 't']
        pd.testing.assert_frame_equal(final[cols], out.loc[out.sp >= S.OFFAXIS_KEY0, cols].reset_index(drop=True),
                                      check_dtype=False)
    assert set(panels) == {2, 3}
