"""Enabled actual consumer hooks are inert and original-key exposure is retained."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trkfastsim.sim import fieldsteps as fs, fieldwall as fw
from trkfastsim.validation import fieldwallpilot as wp
from test_fieldwall_transport import geom, start


def states():
    out = pd.concat([start((-2., 0., 0.)), start((-2., 1.1, 0.)), start((-2., 10., 0.)),
                     start((100., 0., 0.))], ignore_index=True)
    out['event'] = np.arange(4, dtype=np.uint64)+np.uint64(2**63+9)
    out['sp'] = np.arange(4, dtype=np.uint64)+np.uint64(2**63+13)
    return out


def test_real_enabled_transport_exact_payload_final_rng_and_inputs_inert():
    st, g = states(), geom()
    before = st.copy(deep=True)
    opts = dict(chord_mm=.5, max_chords=12, complete_outer_wall=True,
                envelope=fs.Envelope(g, pad_r=4., pad_z=4., axis_xy=(0., 0.)), origin=np.zeros(3))
    r0, r1 = np.random.default_rng(77), np.random.default_rng(77)
    expected = fs.transport(r0, st, g, np.array([0., 0., 1.]), **opts)
    original = (fs._columns, fs.tr.highland_var_increment, fs.st._group_tail, fw.WallIndex.columns)
    obs = wp.ConsumerLedger(st)
    with wp.observing_consumers(obs):
        actual = fs.transport(r1, st, g, np.array([0., 0., 1.]), **opts)
    assert original == (fs._columns, fs.tr.highland_var_increment, fs.st._group_tail, fw.WallIndex.columns)
    for a, b in zip(actual, expected):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(st, before, check_exact=True)
    assert r0.bit_generator.state == r1.bit_generator.state
    d, r = obs.frame(), obs.report()
    np.testing.assert_array_equal(d.chords, actual[1].chords)
    assert d.event.dtype == np.uint64 and d.sp.dtype == np.uint64
    np.testing.assert_array_equal(d.event, st.event)
    assert d.loc[1, 'wall_only_chords'] > 0
    assert d.loc[3, ['chords', 'wall_mm', 'wall_x0']].eq(0).all()
    assert r['column_calls'] > 0 and r['msc_calls'] == r['tail_calls'] > 0
    assert 'not unmodified' in r['throughput_scope']


def test_observer_context_mismatch_and_exception_restore():
    original = (fs._columns, fs.tr.highland_var_increment, fs.st._group_tail, fw.WallIndex.columns)
    with pytest.raises(ValueError, match='caller'):
        with wp.observing_consumers(wp.ConsumerLedger(states())):
            fs._columns(1, [])
    assert original == (fs._columns, fs.tr.highland_var_increment, fs.st._group_tail, fw.WallIndex.columns)


@pytest.mark.parametrize('changed', [np.nan, np.inf, -1., 1.])
def test_direct_material_consumer_mismatch_fails(changed):
    st = states()
    obs = wp.ConsumerLedger(st)
    loc = dict(act=np.arange(4), f_metal=.1, sp=None)
    obs.wall(loc, (np.ones(4), np.ones(4), np.ones(4)))
    columns = {fs.materials.MYLAR.name: (fs.materials.MYLAR, np.full(4, .9)),
               fs.materials.ALUMINIUM_6061.name: (fs.materials.ALUMINIUM_6061, np.full(4, .1))}
    columns[fs.materials.MYLAR.name][1][0] = changed
    with pytest.raises(ValueError, match='mismatch'):
        obs.columns(loc, columns)


def test_pending_material_ledger_cannot_be_reported_or_overwritten():
    obs = wp.ConsumerLedger(states())
    loc = dict(act=np.arange(4), f_metal=.1, sp=None)
    obs.wall(loc, tuple(np.ones(4) for _ in range(3)))
    with pytest.raises(ValueError, match='incomplete'):
        obs.report()
    with pytest.raises(ValueError, match='not consumed'):
        obs.wall(loc, tuple(np.ones(4) for _ in range(3)))


def test_sgs_export_preserves_native_original_mapping_and_order():
    st, g = states(), geom()
    result = fs.transport(np.random.default_rng(78), st, g, None, msc=False, loss=lambda r,m,p,l,ms: l*0,
                           complete_outer_wall=True, max_chords=1, chord_mm=4., origin=np.zeros(3),
                           envelope=fs.Envelope(g, axis_xy=(0., 0.)))
    rk = st[wp.TRACK+['pdg']].sort_values(wp.TRACK).copy()
    rk['key'] = rk.groupby(wp.tables.KEY).cumcount()+1
    p = rk[wp.tables.KEY+['key','pdg']].assign(parent=-1,isprim=1,ismcprim=1).reset_index(drop=True)
    out = wp.export_candidate(result[0], st, p)
    assert len(out) == len(result[0])
    assert (out.sp == 1).all() and out.event.dtype == np.uint64
    np.testing.assert_array_equal(out.event, result[0].event)
    with pytest.raises(AssertionError):
        wp.export_candidate(result[0], st, p.assign(pdg=11))


def test_prior_source_receipt_mutation_and_binding_fail_closed(tmp_path):
    files = {name: tmp_path/name for name in ['input.root','fasttrk_field.root','g4prim_field.root','g4fp_field.root']}
    for name, path in files.items():
        path.write_text(name)
    receipts = {name: dict(path=str(path), sha256_before=wp.pa.sha256(path), sha256_after=wp.pa.sha256(path))
                for name, path in files.items()}
    receipts['input_table'] = receipts.pop('input.root')
    report = dict(passed=True, g4all_ordered_payload_exact=True, inputs=receipts)
    wp.verify_source_receipts(report, files['input.root'], tmp_path)
    files['fasttrk_field.root'].write_text('changed')
    with pytest.raises(ValueError, match='source receipt changed'):
        wp.verify_source_receipts(report, files['input.root'], tmp_path)


def test_strict_json_nonfinite_unavailable_not_nan_literal():
    result = wp.clean(dict(a=np.nan, b=np.array([np.inf, 1.]), c=np.uint64(2**63+1), d=np.bool_(True)))
    assert json.dumps(result, allow_nan=False)
    assert result == {'a': None, 'b': [None, 1.], 'c': 2**63+1, 'd': True}


def test_supplied_map_geometry_binding_and_prior_dependencies_retained(tmp_path):
    paths = {name: tmp_path/name for name in ('input.root', 'fasttrk_field.root', 'g4prim_field.root',
                                               'g4fp_field.root', 'geom.gdml', 'map.header', 'map.bin', 'oldcode.py')}
    for name, path in paths.items():
        path.write_text(name)
    inputs = {name: dict(path=str(path), sha256_before=wp.pa.sha256(path), sha256_after=wp.pa.sha256(path))
              for name, path in paths.items()}
    inputs['input_table'] = inputs.pop('input.root')
    report = dict(passed=True, g4all_ordered_payload_exact=True, inputs=inputs)
    retained = wp.verify_source_receipts(report, paths['input.root'], tmp_path, paths['geom.gdml'], paths['map.header'])
    assert 'prior/inputs/oldcode.py' in retained
    foreign = tmp_path/'other.header'
    foreign.write_text(paths['map.header'].read_text())
    with pytest.raises(ValueError, match='geometry/map binding'):
        wp.verify_source_receipts(report, paths['input.root'], tmp_path, paths['geom.gdml'], foreign)
    before = wp.pa.receipts(retained)
    paths['oldcode.py'].write_text('changed during transport')
    assert before != wp.pa.receipts(retained)


def test_driver_wires_string_map_path_and_retains_closed_outputs(tmp_path, monkeypatch):
    g, st = geom(), states()
    opts = dict(species=13, inputs=1, seed=20261103, chord_mm=20., sagitta_mm=.02, gas_state='solid',
                mu_radiative=False, origin_mm=[0.,0.,0.], entry_vds=[13], field='map.header', entries=len(st))
    legacy = tmp_path/'legacy'
    legacy.mkdir()
    (legacy/'fieldcompare.json').write_text(json.dumps(opts))
    rk = st[wp.TRACK+['pdg']].sort_values(wp.TRACK).copy()
    rk['key'] = rk.groupby(wp.tables.KEY).cumcount()+1
    particles = rk[wp.tables.KEY+['key','pdg']].assign(parent=-1,isprim=1,ismcprim=1).reset_index(drop=True)
    frame = fs.transport(np.random.default_rng(1), st, g, None, max_chords=1, chord_mm=4.,
                          loss=lambda r,m,p,l,ms: l*0, msc=False, origin=np.zeros(3),
                          envelope=fs.Envelope(g, axis_xy=(0.,0.)))[0]
    exported = wp.export_candidate(frame, st, particles)
    for name in ('fasttrk_field.root','g4prim_field.root','g4fp_field.root'):
        wp.tables.write(legacy/name, {'particles': particles, 'sgsteps': exported}, as_ttree=True)
    inp, gdml, header = tmp_path/'input.root', tmp_path/'geom.gdml', tmp_path/'map.header'
    wp.tables.write(inp, {'sgsteps': frame}, as_ttree=True)
    gdml.write_text('synthetic geometry loader')
    header.write_text('synthetic map loader')
    header.with_suffix('.bin').write_text('synthetic map binary')
    bound = {p.name: p for p in (inp, gdml, header, header.with_suffix('.bin'))}
    bound.update({p.name:p for p in legacy.glob('*.root')})
    receipts = {name: dict(path=str(path), sha256_before=wp.pa.sha256(path), sha256_after=wp.pa.sha256(path))
                for name,path in bound.items()}
    receipts['input_table'] = receipts.pop('input.root')
    source = tmp_path/'source.json'
    source.write_text(json.dumps(dict(passed=True, g4all_ordered_payload_exact=True, entry_particles=len(st), inputs=receipts)))
    monkeypatch.setattr(wp.fc, 'primaries', lambda inputs,species: (st[wp.TRACK+['pdg']],frame,None,None,{'events':len(st)}))
    monkeypatch.setattr(wp.sa, 'reconstruct', lambda *a: dict(particles=particles,sgsteps=exported,entry=st,genealogy=st[wp.TRACK]))
    monkeypatch.setattr(wp.geometry, 'from_gdml', lambda path:g)
    monkeypatch.setattr(fs, 'interior', lambda *a,**k:None)
    def read_map(path):
        assert isinstance(path,str)
        assert path == str(header)
        return ()
    monkeypatch.setattr(fs.ehelix, 'read_g4bl', read_map)
    monkeypatch.setattr(fs.ehelix, 'grid_field', lambda *a:np.array([0.,0.,1.]))
    out = tmp_path/'candidate'
    result = wp.run(inp, legacy, source, gdml, header, out)
    assert result['passed'] and result['entries'] == len(st)
    assert result['hashes_before'] == result['hashes_after']
    assert all((out/p).is_file() for p in ('fasttrk_field.root','states.root','wall_consumers.csv.gz','genealogy.csv.gz','fieldwallpilot.json'))
    with pytest.raises(FileExistsError):
        wp.run(inp, legacy, source, gdml, header, out)


def test_closed_path_audit_binds_map_when_source_audit_has_no_map(tmp_path):
    legacy=tmp_path/'legacy'
    legacy.mkdir()
    inp,gdml,header=tmp_path/'input.root',tmp_path/'geometry.gdml',tmp_path/'DSMap.header'
    expected=dict(input_table=inp,gdml=gdml,map_header=header,map_adjacent_binary=header.with_suffix('.bin'))
    expected.update({'original/'+n:legacy/n for n in ('fasttrk_field.root','g4prim_field.root','g4fp_field.root','fieldcompare.json')})
    for name,path in expected.items():
        path.write_text(name)
    hashes=wp.pa.receipts(expected)
    report=dict(passed=True,hashes_before=hashes,hashes_after=copy.deepcopy(hashes))
    found=wp.verify_path_receipts(report,inp,legacy,gdml,header)
    assert found['path_audit/map_header']==header
    header.with_suffix('.bin').write_text('changed')
    with pytest.raises(ValueError,match='binding mismatch'):
        wp.verify_path_receipts(report,inp,legacy,gdml,header)
