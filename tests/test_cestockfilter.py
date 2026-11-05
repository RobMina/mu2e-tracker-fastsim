"""Stock filter diagnostic parsing fails closed on incomplete or duplicate event exposure."""
import json

import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import cestockfilter as cf


def text(events=(1, 2, 3)):
    return "\n".join(f"[DetectorStepsFilter::filter] run: 1430 subRun: 6000 event: {e} selecttrk = {i == 0:d} selectcalo = {i == 1:d} selectcrv = 0"
                     for i, e in enumerate(events)) + f"\nTrigReport Events total = {len(events)} passed = 2 failed = {len(events) - 2}\n"


def test_parse_exact_keys_and_filter_OR():
    ids = np.arange(2**54, 2**54 + 3, dtype=np.uint64)
    d = cf.decisions(text(ids), 3)
    np.testing.assert_array_equal(d.event, ids)
    assert d.stock_accept.tolist() == [True, True, False]
    assert d.stock_trk.tolist() == [True, False, False]


def test_missing_duplicate_and_wrong_completed_counts_rejected():
    for txt in (text((1, 1, 2)), text().replace("Events total = 3", "Events total = 4"), text().replace("selecttrk = 0", "selecttrk = false", 1)):
        with pytest.raises(ValueError):
            cf.decisions(txt, 3)
    with pytest.raises(ValueError):
        cf.decisions(text(), 4)


# Census mode intentionally has no inferred tracker/calorimeter decision components.
def ledger(accepted=(True, False, False)):
    n = len(accepted)
    return pd.DataFrame(dict(run=np.full(n, 1430, np.uint64), subrun=np.full(n, 6200, np.uint64),
                             event=np.arange(2**54, 2**54 + n, dtype=np.uint64),
                             StockSelectionPath=list(accepted), PrimaryPath=True,
                             status=np.zeros(n, np.uint64), overflowSimP=np.zeros(n, np.uint64),
                             nStepLimit=np.zeros(n, np.uint64)))


def frame(d):
    return d[cf.KEY].assign(g4_track=[True, False, True], chain_track=[False, False, True],
                            g4_steps=[12, 0, 15])


def test_census_native_and_csv_booleans_exact_keys_and_stock_reconciliation():
    d = ledger()
    sel = cf.census_decisions(d, 3)
    np.testing.assert_array_equal(sel.event, d.event)
    assert sel.stock_accept.tolist() == [True, False, False]
    strings = d.assign(StockSelectionPath=['True', 'False', 'False'], PrimaryPath='True',
                       stock_accept=['True', 'False', 'False'])
    pd.testing.assert_frame_equal(cf.census_decisions(strings, 3), sel)
    with pytest.raises(ValueError, match='contradicts'):
        cf.census_decisions(d.assign(stock_accept=True), 3)


@pytest.mark.parametrize('column', ['StockSelectionPath', 'PrimaryPath', 'stock_accept'])
@pytest.mark.parametrize('bad', ['false', 'TRUE', '0', '1', 'yes', ' False', '', None, pd.NA, 0, 1, np.nan])
def test_census_ambiguous_or_missing_boolean_rejected(column, bad):
    d = ledger()
    if column == 'stock_accept':
        d[column] = d.StockSelectionPath
    d[column] = d[column].astype(object)
    d.loc[0, column] = bad
    with pytest.raises(ValueError, match='boolean'):
        cf.census_decisions(d, 3)


@pytest.mark.parametrize('column', cf.KEY + ['StockSelectionPath', 'PrimaryPath', 'status', 'overflowSimP', 'nStepLimit'])
def test_census_missing_columns_rejected(column):
    with pytest.raises(ValueError, match='missing census'):
        cf.census_decisions(ledger().drop(columns=column), 3)


@pytest.mark.parametrize('column', ['status', 'overflowSimP', 'nStepLimit'])
@pytest.mark.parametrize('bad', [1, -1, 0.5, np.nan, '0', False])
def test_invalid_status_is_not_a_zero_response(column, bad):
    d = ledger()
    d[column] = bad
    with pytest.raises(ValueError, match='StatusG4'):
        cf.census_decisions(d, 3)


def test_census_requires_all_physics_eligible_inclusive_events():
    with pytest.raises(ValueError, match='inclusive PrimaryPath'):
        cf.census_decisions(ledger().assign(PrimaryPath=False), 3)
    with pytest.raises(ValueError, match='physics_eligible'):
        cf.census_decisions(ledger().assign(physics_eligible=False), 3)
    for expected in [0, -1, 2, 4, True, 3.0]:
        with pytest.raises(ValueError, match='expected event count'):
            cf.census_decisions(ledger(), expected)
    with pytest.raises(ValueError, match='expected event count'):
        cf.census_decisions(ledger(()), 0)


def test_census_duplicate_and_noninteger_ids_rejected():
    d = ledger()
    d.loc[1, cf.KEY] = d.loc[0, cf.KEY]
    with pytest.raises(ValueError, match='duplicate'):
        cf.census_decisions(d, 3)
    for bad in [ledger().assign(event=1.0), ledger().assign(event=-1), ledger().assign(event='1'),
                ledger().assign(event=np.nan)]:
        with pytest.raises(ValueError, match='identities'):
            cf.census_decisions(bad, 3)


@pytest.mark.parametrize('accepted', [(True, False, False), (True, True, True), (False, False, False)])
def test_census_only_observed_OR_masks_include_zero_and_empty_subsets(accepted):
    d = ledger(accepted)
    f = frame(d)
    result = cf.census_report({'a': f, 'b': f.copy()}, cf.census_decisions(d, 3))
    per = result['per_seed']['a']
    assert set(per) == {'all_events', 'stock_OR_pass', 'stock_OR_fail'}
    assert result['stock_filter_counts'] == {'stock_accept': sum(accepted)}
    for name, mask in [('all_events', np.ones(3, bool)), ('stock_OR_pass', np.array(accepted)),
                       ('stock_OR_fail', ~np.array(accepted))]:
        assert json.dumps(per[name], sort_keys=True) == json.dumps(
            cf.cerootcause.migration(f.g4_track[mask], f.chain_track[mask]), sort_keys=True)
    assert per['all_events']['events'] == 3 and per['all_events']['neither'] == 1
    assert per['stock_OR_pass']['events'] + per['stock_OR_fail']['events'] == 3
    assert 'not an unbiased' in result['scope'] and 'physics closure' in result['scope']


def test_full_ids_foreign_missing_duplicate_and_subset_mismatch_rejected_before_conditioning():
    d = ledger((True, False, False))
    sel, f = cf.census_decisions(d, 3), frame(d)
    foreign = f.copy()
    foreign.loc[2, 'event'] = np.uint64(2**54 + 10)  # outside the pass mask
    duplicate = pd.concat([f.iloc[:2], f.iloc[:1]], ignore_index=True)
    for bad in (foreign, duplicate, f.iloc[:1], f.iloc[1:]):
        with pytest.raises((ValueError, AssertionError)):
            cf.census_report({'a': f, 'b': bad}, sel)
    foreign_ledger = d.copy()
    foreign_ledger.loc[2, 'event'] = np.uint64(2**54 + 10)
    with pytest.raises(AssertionError):
        cf.census_report({'a': f}, cf.census_decisions(foreign_ledger, 3))


def test_full_g4_baseline_exact_before_conditioning_and_accept_column_reconciled():
    d = ledger()
    sel, f = cf.census_decisions(d, 3), frame(d)
    for column, value in [('g4_track', False), ('g4_steps', 16)]:
        bad = f.copy()
        bad.loc[2, column] = value  # fail-mask event must still match
        with pytest.raises(AssertionError):
            cf.census_report({'a': f, 'b': bad}, sel)
    f['g4_energy'] = [1.0, 2.0, 3.0]
    bad = f.copy()
    bad.loc[2, 'g4_energy'] = np.nextafter(3.0, 4.0)
    with pytest.raises(AssertionError):
        cf.census_report({'a': f, 'b': bad}, sel)
    for column in ['stock_accept', 'StockSelectionPath']:
        with pytest.raises(ValueError, match='contradicts'):
            cf.census_report({'a': f.assign(**{column: True})}, sel)
    with pytest.raises(ValueError, match='require g4_track and chain_track'):
        cf.census_report({'a': f.drop(columns='chain_track')}, sel)


def test_census_cli_gzip_exact_and_mutually_exclusive(tmp_path, capsys):
    d = ledger()
    census, events = tmp_path / 'census.csv.gz', tmp_path / 'events.csv.gz'
    d.to_csv(census, index=False)
    frame(d).to_csv(events, index=False)
    args = [f'a={events}', '--census-events', str(census), '--expected-events', '3',
            '--outdir', str(tmp_path / 'report')]
    cf.main(args)
    result = json.loads((tmp_path / 'report' / 'cestockfilter.json').read_text())
    assert result['per_seed']['a']['stock_OR_pass']['events'] == 1
    saved = pd.read_csv(tmp_path / 'report' / 'stock_decisions.csv.gz')
    np.testing.assert_array_equal(saved.event, d.event)
    assert list(saved) == cf.KEY + ['stock_accept']
    with pytest.raises(SystemExit):
        cf.main(args + ['--replay-log', 'unused'])
    # pandas normally infers lowercase false as bool: retain lexical tokens and reject it.
    d.assign(StockSelectionPath=['True', 'false', 'False']).to_csv(census, index=False)
    with pytest.raises(ValueError, match='boolean'):
        cf.main(args)


def test_replay_report_and_cli_keep_existing_components(tmp_path, capsys):
    sel = cf.decisions(text(), 3)
    f = frame(sel).assign(g4_tracker_term=[True, False, False])
    result = cf.report({'old': f}, sel)
    assert result['per_seed']['old']['stock_OR_pass']['events'] == 2
    assert result['per_seed']['old']['stock_calo_only']['events'] == 1
    assert result['per_seed']['old']['all_particle_vs_primary_tracker_decisions'] == {
        'all_particle_only': 0, 'primary_only': 0}
    log, events = tmp_path / 'replay.log', tmp_path / 'events.csv.gz'
    log.write_text(text())
    f.to_csv(events, index=False)
    cf.main([f'old={events}', '--replay-log', str(log), '--expected-events', '3',
             '--outdir', str(tmp_path / 'old')])
    assert json.dumps(json.loads((tmp_path / 'old' / 'cestockfilter.json').read_text()), sort_keys=True) == json.dumps(result, sort_keys=True)


def test_census_reordered_full_frames_align_and_g4_column_sets_remain_exact():
    d = ledger()
    sel = cf.census_decisions(d.iloc[::-1], 3)
    f = frame(d)
    expected = cf.census_report({'a': f, 'b': f}, sel)
    actual = cf.census_report({'a': f.iloc[::-1], 'b': f.iloc[[1, 2, 0]]}, sel)
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    for bad in (f.drop(columns='g4_steps'), f.assign(g4_extra=0), f.assign(g4_track=['True', 'False', 'True'])):
        with pytest.raises(AssertionError):
            cf.census_report({'a': f, 'b': bad}, sel)
