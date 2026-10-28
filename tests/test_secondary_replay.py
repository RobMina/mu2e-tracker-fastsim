"""Saved replay closure and fail-closed output, using synthetic v2 donor geometry."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import secondary_replay as R
from trkfastsim import library as L, steps as S
from test_anchor_topology import _source
from test_library_v2 import _build


@pytest.fixture
def replay_data():
    geom, source, parts = _source()
    # Distinct in-straw donors on every crossing exercise stream and index identity.
    secondary = source[source.sp == 1].copy()
    secondary['sp'] = 7
    secondary['eion'] = np.arange(1, len(secondary) + 1) * .0001
    secondary[['px', 'py', 'pz']] = 0.
    source = pd.concat([source, secondary], ignore_index=True)
    source['nsgs'] = len(source)
    parts = pd.concat([parts, parts.iloc[1:2].assign(key=7)], ignore_index=True)
    original = _build(source, parts, geom)
    annotated = _build(source, parts, geom, topology=True)
    mu = source[source.sp == 1].copy()
    # Multiple primary keys in the SAME event, plus a primary without crossings.
    mu2 = mu.iloc[::4].copy().assign(sp=5, t=mu.iloc[::4].t + 100.)
    primary = pd.concat([mu, mu2], ignore_index=True)
    primary['nsgs'] = len(primary)
    primary_parts = pd.concat([parts.iloc[:1], parts.iloc[:1].assign(key=5),
                              parts.iloc[:1].assign(key=9, event=2)], ignore_index=True)
    saved = {'sgsteps': primary, 'particles': primary_parts,
             'gen': pd.DataFrame({'run': [1, 1], 'subrun': [1, 1], 'event': [1, 2], 'pdg': [13, 13]}),
             'subruns': pd.DataFrame({'run': [1], 'subrun': [1], 'livetime': [2.25]}),
             'surfsteps': pd.DataFrame({'run': [1], 'subrun': [1], 'event': [1], 'sp': [1], 't': [1.25]})}
    # Construct baseline independently of replay helper, matching make_steps.
    off = L.OffAxisSampler(original, np.random.default_rng([23, 17, 2]))
    ins = L.InStrawSampler(original, np.random.default_rng([23, 17, 1]))
    sg, pa, _ = S.add_offaxis_secondaries(primary, primary_parts, off, geom)
    sg, pa = S.add_instraw_secondaries(sg, pa, ins, geom)
    return geom, original, annotated, {**saved, 'sgsteps': sg, 'particles': pa}, saved


def test_reconstruction_order_counts_and_protected_payloads(replay_data):
    geom, original, annotated, baseline, primary = replay_data
    before = {k: v.copy(deep=True) for k, v in baseline.items()}
    sg, pa = R.reconstruct_primary(baseline)
    pd.testing.assert_frame_equal(sg, primary['sgsteps'])
    pd.testing.assert_frame_equal(pa, primary['particles'])
    assert set(pa.key) == {1, 5, 9}
    candidate, report = R.replay_file(baseline, original, annotated, geom, 23, 17)
    assert all(c['passed'] for c in report['default']['checks'].values())
    assert all(c['passed'] for c in report['preserved'].values())
    assert R.counts(candidate)['instraw_steps'] == len(primary['sgsteps'])
    assert R.counts(candidate)['offaxis_steps'] > 0
    for name in baseline:
        pd.testing.assert_frame_equal(baseline[name], before[name])
    # Exact saved default closure: no float tolerance is involved.
    default, _ = R.repaste(baseline, original, geom, 23, 17)
    pd.testing.assert_frame_equal(default['sgsteps'], baseline['sgsteps'], check_exact=True)
    assert not R.exact_check(candidate['sgsteps'], baseline['sgsteps'])['passed']


def test_exact_comparison_dtype_reporting_and_ulp_rejection():
    a = pd.DataFrame({'x': [1.], 'key': np.array([1], dtype='int32')})
    b = a.astype({'key': 'int64'})
    assert R.exact_check(a, b)['passed']
    assert R.exact_check(a, b)['dtype_differences'] == {'key': ['int32', 'int64']}
    b.loc[0, 'x'] = np.nextafter(1., 2.)
    assert not R.exact_check(a, b)['passed']
    assert not R.exact_check(a, a[['key', 'x']])['passed']
    assert not R.exact_check(a, a.rename(columns={'x': 'y'}))['passed']


def test_legacy_payload_rejects_single_ulp(replay_data):
    _, original, annotated, _, _ = replay_data
    R.verify_legacy_payload(original, annotated)
    bad = {k: v.copy() for k, v in annotated.items()}
    bad['offaxis'].loc[0, 'eion'] = np.nextafter(bad['offaxis'].loc[0, 'eion'], np.inf)
    with pytest.raises(R.ReplayIdentityError, match='legacy payload'):
        R.verify_legacy_payload(original, bad)


def test_wrong_original_index_fails_before_topology(replay_data, monkeypatch):
    geom, original, annotated, baseline, _ = replay_data
    calls = []
    paste = R.repaste
    def tracked(*args, **kwargs):
        calls.append(kwargs.get('topology', False))
        return paste(*args, **kwargs)
    monkeypatch.setattr(R, 'repaste', tracked)
    with pytest.raises(R.ReplayIdentityError, match='default identity'):
        R.replay_file(baseline, original, annotated, geom, 23, 0)
    assert calls == [False, False]


def _disk_inputs(tmp_path, replay_data, monkeypatch):
    geom, original, annotated, baseline, _ = replay_data
    lib, ann, gdml = (tmp_path / name for name in ('library.root', 'annotated.root', 'geom.gdml'))
    R.core_tables.write(str(lib), original, as_ttree=True)
    R.core_tables.write(str(ann), annotated, as_ttree=True)
    gdml.write_text('synthetic geometry supplied by test')
    monkeypatch.setattr(R.geometry, 'from_gdml', lambda path: geom)
    return baseline, dict(library_path=lib, annotated_path=ann, gdml=gdml, seed=23)


def test_disk_roundtrip_ttrees_indices_provenance_and_exclusive_output(tmp_path, replay_data, monkeypatch):
    baseline, options = _disk_inputs(tmp_path, replay_data, monkeypatch)
    path = tmp_path / 'fast.saved.root'
    R.core_tables.write(str(path), baseline, as_ttree=True)
    outdir = tmp_path / 'replay'
    report = R.run([path], start_index=17, outdir=outdir, **options)
    assert report['status'] == 'complete' and report['all_defaults_passed']
    entry = report['files'][0]
    assert entry['file_index'] == 17
    assert entry['input']['sha256'] == R.sha256(path)
    assert len(report['sources']) >= 6
    assert json.loads((outdir / 'counts.json').read_text())['status'] == 'complete'
    reread = R.read_baseline(outdir / path.name)
    R.verify_preserved(baseline, reread)
    with R.uproot.open(outdir / path.name) as root:
        assert all(root['dump/' + table].classname == 'TTree' for table in R.TABLES)
    with pytest.raises(FileExistsError):
        R.run([path], start_index=17, outdir=outdir, **options)


def test_later_failed_default_writes_no_candidates_and_preserves_order(tmp_path, replay_data, monkeypatch):
    baseline, options = _disk_inputs(tmp_path, replay_data, monkeypatch)
    # Deliberately reverse lexicographic names; no sorting or index reset allowed.
    first, second = tmp_path / 'z.root', tmp_path / 'a.root'
    R.core_tables.write(str(first), baseline, as_ttree=True)
    geom, original, _, _, _ = replay_data
    bad, _ = R.repaste(baseline, original, geom, 23, 18)
    bad['sgsteps'].loc[0, 'nsgs'] += 1
    R.core_tables.write(str(second), bad, as_ttree=True)
    outdir = tmp_path / 'failed'
    with pytest.raises(R.ReplayIdentityError, match='default identity'):
        R.run([first, second], start_index=17, outdir=outdir, **options)
    assert not list(outdir.glob('*.root'))
    report = json.loads((outdir / 'counts.json').read_text())
    assert report['status'] == 'failed'
    assert [f['file_index'] for f in report['files']] == [17, 18]
    assert [f['output_filename'] for f in report['files']] == ['z.root', 'a.root']
    assert all(c['passed'] for c in report['files'][0]['default']['checks'].values())
    assert not report['files'][1]['default']['checks']['original.sgsteps']['passed']


def test_optional_surfaces_and_protected_instraw_guard(replay_data):
    geom, original, annotated, baseline, _ = replay_data
    baseline = {**baseline, 'surfsteps': None}
    candidate, _ = R.replay_file(baseline, original, annotated, geom, 23, 17)
    assert candidate['surfsteps'] is None
    ins = candidate['sgsteps'].sp.between(S.SECONDARY_KEY0, S.OFFAXIS_KEY0 - 1)
    row = candidate['sgsteps'].index[ins][0]
    candidate['sgsteps'].loc[row, 'eion'] = np.nextafter(candidate['sgsteps'].loc[row, 'eion'], np.inf)
    with pytest.raises(R.ReplayIdentityError, match='protected'):
        R.verify_preserved(baseline, candidate)


def test_sampler_seeds_and_offaxis_then_instraw_order(replay_data, monkeypatch):
    geom, original, _, baseline, _ = replay_data
    seen = []
    off_type, ins_type = L.OffAxisSampler, L.InStrawSampler
    off_paste, ins_paste = S.add_offaxis_secondaries, S.add_instraw_secondaries

    def off_factory(lib, rng, **kwargs):
        assert rng.bit_generator.state == np.random.default_rng([23, 17, 2]).bit_generator.state
        return off_type(lib, rng, **kwargs)

    def ins_factory(lib, rng):
        assert rng.bit_generator.state == np.random.default_rng([23, 17, 1]).bit_generator.state
        return ins_type(lib, rng)

    def off_call(*args):
        seen.append('offaxis')
        assert not (args[0].sp >= S.SECONDARY_KEY0).any()
        return off_paste(*args)

    def ins_call(*args):
        seen.append('instraw')
        assert (args[0].sp >= S.OFFAXIS_KEY0).any()
        assert not args[0].sp.between(S.SECONDARY_KEY0, S.OFFAXIS_KEY0 - 1).any()
        return ins_paste(*args)

    monkeypatch.setattr(L, 'OffAxisSampler', off_factory)
    monkeypatch.setattr(L, 'InStrawSampler', ins_factory)
    monkeypatch.setattr(S, 'add_offaxis_secondaries', off_call)
    monkeypatch.setattr(S, 'add_instraw_secondaries', ins_call)
    R.repaste(baseline, original, geom, 23, 17)
    assert seen == ['offaxis', 'instraw']
