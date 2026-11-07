"""Read-only saved-stage attrition, with stock coverage used only for reporting.

No candidate search/refit, threshold change, stock-conditioned recovery or truth
matching. Stage counters describe availability, not causal physics attribution.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import run1a_stockcompare as comparison
from .run1a_wirefit import _sha256

COUNTERS = ('group_hypotheses', 'group_independent', 'group_fit_success', 'group_supported', 'group_support')
LANES = ('status', 'certificate_status', 'termination_status', 'last_refit_status',
         'last_refit_termination_status', 'objective_valid', 'success', 'certified',
         'analytic_stationary', 'analytic_differentiable', 'lr_stationary', 'pending_lr_flips',
         'refit_error', 'refit_numerical_failure', 'accepted_lr_rounds', 'continuous_refits')


def _equal(a, b):
    return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')


def _counts(a):
    names, counts = np.unique(a, return_counts=True)
    return {str(k): int(v) for k, v in zip(names, counts)}


def diagnose(fast, association):
    """Recount saved candidate/source identities and inventory every group/lane."""
    keys, gev, hev, sid, hg, selected, support, flags = comparison.validate_fast(fast)
    ng, nh = len(gev), len(hev)
    for key, value in association.items():
        comparison._unmasked(value, key)
    for key in ('evkey', 'source_hit_ev', 'source_hit_sid', 'hit_group', 'group_ev'):
        if not _equal(np.asarray(association[key]), np.asarray(fast[key])):
            raise ValueError(f'association identity mismatch: {key}')
    comparison._boolean(association['association_source_used'], 'association_source_used', (nh,))
    for field, expected in (('association_source_used', selected), ('association_group_support', support)):
        if not np.array_equal(association[field], expected):
            raise ValueError(f'association support mismatch: {field}')
    n = comparison._integer(association['group_n'], 'group_n', (ng,))
    grouped = hg >= 0
    if not np.array_equal(n, np.bincount(hg[grouped], minlength=ng)):
        raise ValueError('group membership recount mismatch')
    count = comparison._integer(association['group_npoints'], 'group_npoints', (ng,), high=256)
    cg = comparison._integer(association['candidate_group'], 'candidate_group', high=ng-1)
    if cg.ndim != 1:
        raise ValueError('candidate groups must be a vector')
    if 'candidate_ev' in association:
        cev = comparison._integer(association['candidate_ev'], 'candidate_ev', cg.shape, high=len(keys)-1)
        if not np.array_equal(cev, gev[cg]):
            raise ValueError('candidate event mismatch')
    pair = comparison._integer(association['candidate_hit_index'], 'candidate_hit_index', (len(cg), 2), high=nh-1)
    pos = np.asarray(association['candidate_positions'])
    if pos.shape != (len(cg), 3) or pos.dtype.kind != 'f' or not np.isfinite(pos).all():
        raise ValueError('invalid candidate positions')
    if np.any(pair[:, 0] == pair[:, 1]) or np.any(hg[pair] != cg[:, None]):
        raise ValueError('candidate source/group mismatch')
    if len(np.unique(np.sort(pair, axis=1), axis=0)) != len(pair):
        raise ValueError('duplicate candidate source pair')
    if not np.array_equal(count, np.bincount(cg, minlength=ng)):
        raise ValueError('candidate count mismatch')
    counters = {k: comparison._integer(association['association_'+k], k, (ng,)) for k in COUNTERS}
    choose4 = np.array([math.comb(i, 4) if i >= 4 else 0 for i in range(257)], np.int64)
    if not np.array_equal(counters['group_hypotheses'], choose4[count]):
        raise ValueError('quartet count mismatch')
    for prev, cur in zip(COUNTERS[:3], COUNTERS[1:4]):
        if np.any(counters[cur] > counters[prev]):
            raise ValueError('association counters not nested')
    if not np.array_equal(counters['group_supported'] > 0, support > 0):
        raise ValueError('supported hypothesis/winner mismatch')
    if np.any((support > 0) & (support < 8)) or np.any(support > n):
        raise ValueError('invalid winner support')
    geometry = comparison._boolean(association['hit_geometry_supported'], 'hit_geometry_supported', (nh,))
    unsupported = np.zeros(ng, bool)
    np.logical_or.at(unsupported, hg[grouped], ~geometry[grouped])
    evaluated = comparison._boolean(association['group_candidates_evaluated'], 'group_candidates_evaluated', (ng,))
    span = np.asarray(association['group_span'])
    if span.shape != (ng,) or span.dtype.kind != 'f' or not np.isfinite(span).all() or np.any(span < 0):
        raise ValueError('invalid group time span')
    if not np.array_equal(evaluated, ~unsupported) or np.any(((span > 100.) | unsupported) & (count > 0)):
        raise ValueError('candidate eligibility mismatch')
    category = np.select((unsupported, span > 100., count < 4,
        counters['group_independent'] == 0, counters['group_fit_success'] == 0,
        counters['group_supported'] == 0),
        ('unsupported_geometry', 'wide_time_group', 'fewer_than_four_candidates',
         'no_independent_quartet', 'no_fitted_hypothesis', 'no_supported_hypothesis'), default='supported')
    used_hits = np.unique(pair.ravel())
    represented = np.bincount(hg[used_hits], minlength=ng)
    distinct_straw = np.unique(np.column_stack((hg[grouped], sid[grouped])), axis=0)
    nstraw = np.bincount(distinct_straw[:, 0], minlength=ng)
    table = dict(group=np.arange(ng), run=keys[gev, 0], subrun=keys[gev, 1], event=keys[gev, 2],
                 members=n, distinct_member_straws=nstraw, time_span_ns=span, stereo_candidates=count,
                 candidate_source_hits=represented, association_stage=category, fixed_support=support)
    table.update(counters)
    for name in LANES:
        a = np.asarray(fast['lr_'+name])
        if name.endswith('status'):
            good_type = a.dtype.kind in 'US'
        elif name in ('pending_lr_flips', 'accepted_lr_rounds', 'continuous_refits'):
            comparison._integer(a, name, (ng, 2))
            good_type = True
        else:
            good_type = a.dtype.kind == 'b'
        if a.shape != (ng, 2) or not good_type:
            raise ValueError(f'invalid LR lane shape/type: {name}')
        for lane, label in enumerate(('negative_z', 'positive_z')):
            table[label+'_'+name] = a[:, lane]
    certificate_labels = ('not_evaluated', 'analytic_linearization_failed', 'analytic_nondifferentiable',
        'analytic_nonstationary', 'fd_check_failed', 'no_admissible_stencil_pair', 'adjacent_stencil_checked')
    cert = fast['lr_certificate_status']
    if not np.isin(cert, certificate_labels).all() or not np.array_equal(cert == 'adjacent_stencil_checked', flags['certified']):
        raise ValueError('certificate label/flag mismatch')
    if np.any(flags['certified'] & (~flags['objective_valid'] | ~flags['analytic_stationary'] | ~fast['lr_analytic_differentiable'])):
        raise ValueError('certified lane lacks continuous prerequisites')
    if not np.array_equal(fast['lr_accepted_lr_rounds'], fast['lr_continuous_refits']):
        raise ValueError('accepted LR/refit count mismatch')
    present = comparison._boolean(fast['comparison_event_stock_present'], 'stock presence', (len(keys),))
    ch = comparison._integer(fast['comparison_event_stock_ch_positive_status'], 'stock CH counts', (len(keys),), low=-1)
    if np.any((ch < 0) != ~present):
        raise ValueError('stock presence/count mismatch')
    table['stock_event_present'] = present[gev]
    table['stock_positive_status_ch'] = ch[gev]
    passed = np.any(flags['success'], axis=1)
    table['any_combined_pass'] = passed
    populations = dict(all_groups=np.ones(ng, bool), no_support=support == 0,
        supported_nonpassing=(support > 0) & ~passed, recorded_ch=ch[gev] > 0,
        recorded_ch_nonpassing=(ch[gev] > 0) & ~passed)
    report = dict(schema_version=1, populations={}, scope=__doc__,
        population_units='groups counts groups; status histograms and continuous_checked/combined_pass/sticky_numerical_failure count direction lanes, never events.',
        counter_scope='Saved raw/independent/fitted/supported quartet counts are not rerun. '
            'Only candidate and source/group identities/counts and combinatorial totals are independently recounted.',
        control_rejection_bits='The unambiguous-control rejection bits are NOT association vetoes and are not used here.')
    for label, sel in populations.items():
        report['populations'][label] = dict(groups=int(sel.sum()),
            groups_any_combined_pass=int(np.sum(sel & passed)),
            groups_any_continuous_pass=int(np.sum(sel & np.any(flags['certified'], axis=1))),
            association_stages=_counts(category[sel]),
            final_lane_status=_counts(fast['lr_status'][sel]), certificate_status=_counts(fast['lr_certificate_status'][sel]),
            continuous_checked=int(flags['certified'][sel].sum()), combined_pass=int(flags['success'][sel].sum()),
            sticky_numerical_failure=int(flags['refit_numerical_failure'][sel].sum()))
    out = {k: v.copy() for k, v in fast.items()}
    out.update({'attrition_'+k: v for k, v in table.items()})
    return report, out, pd.DataFrame(table)


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--compare-saved', type=Path)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    old = json.loads((args.comparison/'report.json').read_text())
    payload_path = args.comparison/'diagnostics.npz'
    if old.get('output_payload_sha256') != _sha256(payload_path):
        raise ValueError('comparison payload hash mismatch')
    fast = _load(payload_path)
    paths = {k: Path(v['path']) for k, v in old['inputs'].items()}
    for key, path in paths.items():
        if _sha256(path) != old['inputs'][key]['sha256']:
            raise ValueError(f'comparison input hash mismatch: {key}')
    original, lr_report = comparison.load_fast(paths['lr_report'].parent)
    for key, value in original.items():
        if key not in fast or not _equal(value, fast[key]):
            raise ValueError(f'comparison changed saved LR field: {key}')
    stock = {k[6:]: v for k, v in fast.items() if k.startswith('stock_')}
    _, replay = comparison.overlap(original, stock)
    if set(replay) != set(fast) or any(not _equal(v, fast[k]) for k, v in replay.items()):
        raise ValueError('comparison table replay mismatch')
    assoc_info = lr_report['inputs']['association']
    association = Path(assoc_info['path'])
    if _sha256(association) != assoc_info['sha256']:
        raise ValueError('association hash mismatch')
    report, payload, table = diagnose(fast, _load(association))
    report.update(inputs={k: dict(path=str(p.resolve()), sha256=_sha256(p)) for k, p in
        dict(comparison_report=args.comparison/'report.json', comparison_payload=payload_path,
             association=association, **paths).items()},
        implementation_sha256=_sha256(Path(__file__)),
        provenance_caveat='Current hashes and structural replay do not restore original association-input '
            'authentication or prove ROOT/log/binary/config binding; optimizer traces are copied, not rerun.')
    from . import run1a_wirefit, stockcensus
    report['implementation'] = {m.__name__: dict(path=str(Path(m.__file__).resolve()), sha256=_sha256(Path(m.__file__)))
        for m in (comparison, stockcensus, run1a_wirefit)}
    report['repeat_scope'] = 'payload_arrays_only; report/CSV are not compared by --compare-saved'
    report['recorded_ch_groups'] = json.loads(table[table.stock_positive_status_ch > 0].to_json(orient='records'))
    if args.compare_saved:
        previous = _load(args.compare_saved)
        if set(previous) != set(payload) or any(not _equal(v, previous[k]) for k, v in payload.items()):
            raise ValueError('attrition repeat differs')
        report['saved_comparison'] = dict(exact=True, sha256=_sha256(args.compare_saved))
    json.dumps(report, allow_nan=False)
    args.outdir.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.outdir/'groups.csv', index=False)
    np.savez_compressed(args.outdir/'diagnostics.npz', **payload)
    report['output_payload_sha256'] = _sha256(args.outdir/'diagnostics.npz')
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    (args.outdir/'report.json').write_text(text)
    print(text, end='')
    return report


if __name__ == '__main__':
    main()
