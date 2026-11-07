"""Read-only saved-LR versus recorded stock-track census and straw-set overlap.

Missing stock events are UNKNOWN, never zero-track events. All event/group/hit
and direction lanes are retained; no reconstruction, truth matching, direction
choice, track assignment, efficiency, fake-rate or momentum claim is made.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import stockcensus
from .run1a_wirefit import _sha256, _quantiles

IDENTITY = ('evkey', 'source_hit_ev', 'source_hit_sid', 'hit_group', 'group_ev', 'source_selected', 'group_support')


def _unmasked(a, name):
    if np.ma.isMaskedArray(a) and np.any(np.ma.getmaskarray(a)):
        raise ValueError(f'missing masked values in {name}')
    return np.asarray(a)


def _integer(a, name, shape=None, low=0, high=None):
    a = _unmasked(a, name)
    if (a.dtype.kind not in 'iu' or (shape is not None and a.shape != shape)
            or np.any(a < low) or np.any(a > (np.iinfo(np.int64).max if high is None else high))):
        raise ValueError(f'invalid {name}')
    return a.astype(np.int64)


def _boolean(a, name, shape):
    a = _unmasked(a, name)
    if a.dtype.kind != 'b' or a.shape != shape:
        raise ValueError(f'invalid {name}')
    return a


def validate_fast(fast):
    """Validate census and combined flags without refitting or repairing history."""
    for name, value in fast.items():
        _unmasked(value, name)
    keys = stockcensus.validate_keys(fast['evkey'])
    ne = len(keys)
    gev = _integer(fast['group_ev'], 'group_ev', high=ne-1)
    hev = _integer(fast['source_hit_ev'], 'source_hit_ev', high=ne-1)
    if gev.ndim != 1 or hev.ndim != 1:
        raise ValueError('event indices must be vectors')
    ng, nh = len(gev), len(hev)
    sid = _integer(fast['source_hit_sid'], 'source_hit_sid', (nh,), high=65535)
    if np.any(((sid >> 10) >= 36) | (((sid >> 7) & 7) >= 6) | ((sid & 127) >= 96)):
        raise ValueError('invalid source straw id')
    hg = _integer(fast['hit_group'], 'hit_group', (nh,), low=-1, high=ng-1)
    use = _boolean(fast['source_selected'], 'source_selected', (nh,))
    count = _integer(fast['group_support'], 'group_support', (ng,))
    grouped = hg >= 0
    if np.any(use & ~grouped) or np.any(hev[grouped] != gev[hg[grouped]]):
        raise ValueError('hit/group/event mismatch')
    members = np.column_stack((hg[use], sid[use]))
    if (len(np.unique(members, axis=0)) != len(members)
            or not np.array_equal(count, np.bincount(hg[use], minlength=ng))):
        raise ValueError('nonunique or inconsistent fixed support')
    fit_mask = _boolean(fast['source_lr_mask'], 'source_lr_mask', (nh, 2))
    if not np.array_equal(fit_mask, np.repeat(use[:, None], 2, axis=1)):
        raise ValueError('saved LR hit mask differs from fixed selected support')
    flags = {k: _boolean(fast['lr_'+k], 'lr_'+k, (ng, 2)) for k in
             ('success', 'objective_valid', 'analytic_stationary', 'certified', 'lr_stationary',
              'refit_error', 'refit_numerical_failure')}
    expected = flags['objective_valid'] & flags['analytic_stationary'] & flags['certified'] & flags['lr_stationary']
    expected &= ~flags['refit_error'] & ~flags['refit_numerical_failure']
    if not np.array_equal(flags['success'], expected):
        raise ValueError('combined LR success flags disagree')
    if np.any(flags['objective_valid'] & (count[:, None] == 0)):
        raise ValueError('empty support cannot have valid objective')
    if np.asarray(fast['lr_status']).shape != (ng, 2) or np.asarray(fast['lr_status']).dtype.kind not in 'US':
        raise ValueError('invalid lr_status')
    if not np.array_equal(fast['lr_status'] == 'converged_certified', flags['success']):
        raise ValueError('combined LR success/status disagree')
    return keys, gev, hev, sid, hg, use, count, flags


def _expand(left_event, right_event, ne, limit):
    """All same-event index pairs with a checked allocation bound."""
    count = np.bincount(right_event, minlength=ne)
    widths = count[left_event]
    cells = int(widths.sum())
    if cells > limit:
        raise ValueError('same-event pair/cell bound exceeded; no truncation')
    left = np.repeat(np.arange(len(left_event)), widths)
    order = np.argsort(right_event, kind='stable')
    starts = np.r_[0, np.cumsum(count)]
    rank = np.arange(cells)-np.repeat(np.cumsum(widths)-widths, widths)
    right = order[starts[left_event[left]]+rank]
    return left, right


def _rows(a):
    a = np.ascontiguousarray(a, dtype=np.int64)
    return a.view(np.dtype([('c'+str(i), np.int64) for i in range(a.shape[1])])).reshape(-1)


def overlap(fast, stock, *, max_pairs=2000000, max_cells=4000000):
    """Keep every same-event group/stock-track pair, including zero shared straws."""
    if (isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)) or not 0 <= max_pairs <= 2000000
            or isinstance(max_cells, bool) or not isinstance(max_cells, (int, np.integer)) or not 0 <= max_cells <= 4000000):
        raise ValueError('invalid pair/cell allocation bounds')
    keys, gev, hev, sid, hg, use, support, flags = validate_fast(fast)
    for name, value in stock.items():
        _unmasked(value, name)
    skeys = stockcensus.validate_keys(stock['evkey'])
    ne, ng = len(keys), len(gev)
    te = _integer(stock['track_ev'], 'track_ev', high=len(skeys)-1)
    if te.ndim != 1:
        raise ValueError('track_ev must be a vector')
    nt = len(te)
    ht = _integer(stock['hit_track'], 'hit_track', high=nt-1)
    if ht.ndim != 1:
        raise ValueError('hit_track must be a vector')
    hs = _integer(stock['hit_sid'], 'hit_sid', ht.shape, high=65535)
    if np.any(((hs >> 10) >= 36) | (((hs >> 7) & 7) >= 6) | ((hs & 127) >= 96)):
        raise ValueError('invalid stock straw id')
    status = _integer(stock['track_status'], 'track_status', (nt,), low=np.iinfo(np.int32).min)
    fitalg = _integer(stock['track_fitalg'], 'track_fitalg', (nt,), low=-1)
    goodfit = _integer(stock['track_goodfit'], 'track_goodfit', (nt,), high=1)
    event_row = stockcensus.event_index(skeys, keys)
    stock_event_in_fast = stockcensus.event_index(keys, skeys)
    mapped = stock_event_in_fast[te]
    tracks = np.flatnonzero(mapped >= 0)
    pg, pt_local = _expand(gev, mapped[tracks], ne, max_pairs)
    pt = tracks[pt_local]
    pair_keys = _rows(np.column_stack((pg, pt)))
    pair_order = np.argsort(pair_keys, kind='stable')
    unique_hits = np.unique(np.column_stack((ht, hs)), axis=0)
    unique_count = np.bincount(unique_hits[:, 0], minlength=nt)
    in_fast = mapped[unique_hits[:, 0]] >= 0
    uhit = unique_hits[in_fast]
    ih, group = _expand(mapped[uhit[:, 0]], gev, ne, max_cells)
    membership = np.sort(_rows(np.column_stack((hg[use], sid[use]))))
    query = _rows(np.column_stack((group, uhit[ih, 1])))
    pos = np.searchsorted(membership, query)
    found = pos < len(membership)
    found[found] &= membership[pos[found]] == query[found]
    matched = _rows(np.column_stack((group[found], uhit[ih[found], 0])))
    index = np.searchsorted(pair_keys[pair_order], matched)
    if np.any(index >= len(pair_keys)) or np.any(pair_keys[pair_order[index]] != matched):
        raise RuntimeError('internal pair alignment failed')
    shared = np.bincount(pair_order[index], minlength=len(pg))
    sf, tf, jaccard = (np.full(len(pg), np.nan) for _ in range(3))
    np.divide(shared, support[pg], out=sf, where=support[pg] > 0)
    np.divide(shared, unique_count[pt], out=tf, where=unique_count[pt] > 0)
    union = support[pg]+unique_count[pt]-shared
    np.divide(shared, union, out=jaccard, where=union > 0)
    out = {k: v.copy() for k, v in fast.items()}
    out.update({'stock_'+k: np.asarray(v).copy() for k, v in stock.items()})
    present = event_row >= 0
    event_fields = dict(all=np.bincount(te, minlength=len(skeys)))
    for name, cut in (('ch', fitalg == 2), ('ch_positive_status', (fitalg == 2) & (status > 0)),
                      ('ch_positive_status_goodfit', (fitalg == 2) & (status > 0) & (goodfit == 1))):
        event_fields[name] = np.bincount(te[cut], minlength=len(skeys))
    for name, value in event_fields.items():
        a = np.full(ne, -1, np.int64)
        a[present] = value[event_row[present]]
        out['comparison_event_stock_'+name] = a
    out.update(comparison_event_stock_row=event_row, comparison_event_stock_present=present,
        comparison_stock_event_fast_row=stock_event_in_fast,
        comparison_stock_track_unique_straws=unique_count,
        comparison_stock_track_recorded_hits=np.bincount(ht, minlength=nt),
        comparison_pair_group=pg, comparison_pair_track=pt, comparison_pair_shared_straws=shared,
        comparison_pair_fast_support=support[pg], comparison_pair_stock_straws=unique_count[pt],
        comparison_pair_fast_fraction=sf, comparison_pair_stock_fraction=tf, comparison_pair_jaccard=jaccard)
    group_pass = np.any(flags['success'], axis=1)
    for name, sel in (('groups', np.ones(ng, bool)), ('seed_groups', support > 0),
                      ('valid_groups', np.any(flags['objective_valid'], axis=1)), ('combined_groups', group_pass)):
        out['comparison_event_fast_'+name] = np.bincount(gev[sel], minlength=ne)
    out['comparison_event_fast_hits'] = np.bincount(hev, minlength=ne)
    has_fast = out['comparison_event_fast_combined_groups'] > 0
    has_stock = out['comparison_event_stock_ch_positive_status'] > 0
    table = {label: int(np.sum(present & cut)) for label, cut in
             (('both', has_fast & has_stock), ('fast_group_only', has_fast & ~has_stock),
              ('stock_track_only', ~has_fast & has_stock), ('neither', ~has_fast & ~has_stock))}
    ch_pair = (fitalg[pt] == 2) & (status[pt] > 0)
    successful_pair = ch_pair & group_pass[pg]
    report = dict(events=ne, groups=ng, hits=len(hev), fixed_selected_hits=int(use.sum()),
        combined_hypotheses=int(flags['success'].sum()), combined_groups=int(group_pass.sum()),
        stock_events=len(skeys), stock_tracks=nt, stock_recorded_hits=len(ht),
        covered_fast_events=int(present.sum()), missing_stock_events_unknown=int((~present).sum()),
        stock_events_outside_fast_census=int(np.sum(stock_event_in_fast < 0)),
        covered_event_contingency=table, missing_stock_events_with_combined_group=int(np.sum(~present & has_fast)),
        same_event_group_track_pairs=len(pg), pair_overlap_definition='Distinct packed straw IDs among fixed fast support '
            'and ALL recorded stock hits, not pulse identity, active-hit purity or a physical track match.',
        stock_repeated_straw_hit_entries=int(len(ht)-len(unique_hits)),
        overlap_counts={str(n): dict(all_pairs=int(np.sum(shared >= n)),
            combined_group_positive_status_ch_pairs=int(np.sum(successful_pair & (shared >= n)))) for n in (1, 4, 8)},
        stock_status_meaning='status>0 includes kalmanConverged(1) and kalmanOK(2); goodfit is separate.',
        coverage_scope='Only recorded stock events enter the contingency table. Missing keys remain unknown '
            '(-1 counts), not stock failures. Recorded tracks may already be output-filtered.',
        limitations='Not track efficiency, fake rate, momentum resolution, calibrated fit quality, direction '
            'identification, equal-compute comparison or throughput. No reco settings or support changed.',
        original_arrays_preserved=True, max_pairs=int(max_pairs), max_cells=int(max_cells))
    for name, a in (('stock_status', status), ('stock_fitalg', fitalg), ('stock_goodfit', goodfit)):
        labels, counts = np.unique(a, return_counts=True)
        report[name+'_counts'] = {str(k): int(v) for k, v in zip(labels, counts)}
    report.update(coverage_details(out))
    return report, out


def coverage_details(data):
    """Summarize all pairs and missing-stage counts without choosing a match."""
    gev, pg, pt = (data[k] for k in ('group_ev', 'comparison_pair_group', 'comparison_pair_track'))
    group_pass = np.any(data['lr_success'], axis=1)
    stock_event = data['comparison_event_stock_ch_positive_status'] > 0
    groups = stock_event[gev]
    flags = {name: data['comparison_event_fast_'+name] > 0
             for name in ('groups', 'seed_groups', 'valid_groups', 'combined_groups')}
    stage = dict(no_group=stock_event & ~flags['groups'],
                 no_seed_support=stock_event & flags['groups'] & ~flags['seed_groups'],
                 no_valid_objective=stock_event & flags['seed_groups'] & ~flags['valid_groups'],
                 no_combined_pass=stock_event & flags['valid_groups'] & ~flags['combined_groups'],
                 combined_pass=stock_event & flags['combined_groups'])
    report = dict(recorded_ch_event_fast_stage_counts={k: int(v.sum()) for k, v in stage.items()},
        recorded_ch_event_stage_scope='Exclusive event-level availability hierarchy, not causal failure '
            'attribution or a track match. Groups/hypotheses are counted separately below.',
        groups_in_recorded_ch_events=int(groups.sum()),
        recorded_ch_group_lane_status_counts={}, pair_overlap_summaries={})
    for label, sel in (('all_groups', groups), ('groups_without_combined_pass', groups & ~group_pass)):
        names, counts = np.unique(data['lr_status'][sel], return_counts=True)
        report['recorded_ch_group_lane_status_counts'][label] = {str(k): int(v) for k, v in zip(names, counts)}
    ch = (data['stock_track_fitalg'][pt] == 2) & (data['stock_track_status'][pt] > 0)
    populations = dict(all_pairs=np.ones(len(pg), bool), positive_status_ch_pairs=ch,
                       combined_group_positive_status_ch_pairs=ch & group_pass[pg],
                       nonpassing_group_positive_status_ch_pairs=ch & ~group_pass[pg])
    for label, sel in populations.items():
        summary = dict(pairs=int(sel.sum()))
        for field in ('shared_straws', 'fast_support', 'stock_straws', 'fast_fraction', 'stock_fraction', 'jaccard'):
            value = data['comparison_pair_'+field][sel]
            summary[field] = dict(finite=int(np.isfinite(value).sum()), quantiles=_quantiles(value))
        shared = data['comparison_pair_shared_straws'][sel]
        fs, ss = data['comparison_pair_fast_support'][sel], data['comparison_pair_stock_straws'][sel]
        summary.update(fast_subset_of_stock_nonempty=int(np.sum((fs > 0) & (shared == fs))),
                       stock_subset_of_fast_nonempty=int(np.sum((ss > 0) & (shared == ss))),
                       identical_nonempty_straw_sets=int(np.sum((fs > 0) & (shared == fs) & (shared == ss))))
        report['pair_overlap_summaries'][label] = summary
    return report


def load_fast(directory):
    """Bind the saved LR output and its measured-input event/hit census."""
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    path = directory/'diagnostics.npz'
    if report.get('output_payload_sha256') != _sha256(path):
        raise ValueError('saved LR payload hash mismatch or missing')
    with np.load(path, allow_pickle=False) as data:
        fast = dict(data)
    if any(k.startswith(('stock_', 'comparison_')) for k in fast):
        raise ValueError('comparison output cannot be used as LR input')
    keys, gev, hev, sid, _, use, _, flags = validate_fast(fast)
    for name, value in (('events', len(keys)), ('groups', len(gev)), ('hits', len(hev)), ('fixed_selected_hits', int(use.sum()))):
        if report.get(name) != value:
            raise ValueError(f'saved LR report census mismatch: {name}')
    if report['final_combined_checked']['hypotheses'] != int(flags['success'].sum()):
        raise ValueError('saved LR success census mismatch')
    info = report['inputs']['hits']
    hitpath = Path(info['path'])
    if _sha256(hitpath) != info['sha256']:
        raise ValueError('saved measured-hit input hash mismatch')
    with np.load(hitpath, allow_pickle=False) as data:
        for key, expected in (('evkey', keys), ('s_ev', hev), ('s_sid', sid)):
            if not np.array_equal(data[key], expected):
                raise ValueError(f'saved measured-hit identity mismatch: {key}')
    return fast, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', required=True, type=Path)
    parser.add_argument('--stock', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--tree', default='EventNtuple/ntuple')
    parser.add_argument('--chunk-entries', type=int, default=4096)
    parser.add_argument('--compare-saved', type=Path)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    start = time.perf_counter()
    fast, original = load_fast(args.lr)
    stock_hash = _sha256(args.stock)
    stock, inventory = stockcensus.read_stock(args.stock, tree=args.tree, chunk_entries=args.chunk_entries)
    if _sha256(args.stock) != stock_hash:
        raise ValueError('stock input changed during read')
    loaded = time.perf_counter()
    report, payload = overlap(fast, stock)
    report.update(stock_inventory=inventory, load_seconds=loaded-start, comparison_seconds=time.perf_counter()-loaded,
        processing_scope='Read-only saved outputs; no fit, simulation, digitization or association rerun; output excluded.',
        historical_lr_inputs=original['inputs'], provenance_caveat='Current hashes bind consumed files, not '
            'historical stock binary/config or missing original association/output bindings. No same-digitization '
            'or full-source-coverage claim follows from event keys alone.')
    paths = dict(lr_report=args.lr/'report.json', lr_payload=args.lr/'diagnostics.npz', stock=args.stock)
    report['inputs'] = {k: dict(path=str(p.resolve()), sha256=_sha256(p)) for k, p in paths.items()}
    report['implementation'] = {k: dict(path=str(p.resolve()), sha256=_sha256(p)) for k, p in
                                (('driver', Path(__file__)), ('stockcensus', Path(stockcensus.__file__)))}
    if args.compare_saved:
        with np.load(args.compare_saved, allow_pickle=False) as previous:
            if set(previous.files) != set(payload) or any(v.dtype != previous[k].dtype or
                    not np.array_equal(v, previous[k], equal_nan=v.dtype.kind in 'fc') for k, v in payload.items()):
                raise ValueError('saved comparison arrays differ')
        report['saved_comparison'] = dict(exact=True, sha256=_sha256(args.compare_saved))
    json.dumps(report, allow_nan=False)
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir/'diagnostics.npz', **payload)
    report['output_payload_sha256'] = _sha256(args.outdir/'diagnostics.npz')
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    (args.outdir/'report.json').write_text(text)
    print(text, end='')
    return report


if __name__ == '__main__':
    main()
