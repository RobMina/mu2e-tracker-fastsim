"""Read-only exhaustive saved-group stereo replay, without association or fits.

Fixed production stereo_points defaults and exclusive first-failure cut order.
All saved groups are audited, never selected using support, stock, or ambiguity.
Only whole unsupported-geometry groups, span >100 ns, and n<2 are excluded.
Current hashes establish this replay's inputs, not original production binding.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from trkfastsim.reco.hitreco import FLAG
from trkfastsim.reco.seedassociation import _geometry
from trkfastsim.reco.stereoseed import MIN_XY_NORM, _columns, _integer
from .run1a_seed import validate, geometry_supported

REJECTIONS = ('parallel', 'nonfinite', 'same_z', 'far_z', 'time', 'wire_length', 'wire_distance')
COUNTS = ('considered',) + tuple('rejected_' + k for k in REJECTIONS) + ('accepted',)


def _array(table, key, kind, shape=None):
    try:
        value = table[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'missing {key}') from exc
    if np.ma.isMaskedArray(value):
        raise ValueError(f'masked {key}')
    a = np.asarray(value)
    if a.dtype.kind not in kind or (shape is not None and a.shape != shape):
        raise ValueError(f'invalid shape/type for {key}')
    if not np.isfinite(a).all():
        raise ValueError(f'nonfinite {key}')
    return a


def _match(table, key, expected, kind):
    actual = _array(table, key, kind, expected.shape)
    if not np.array_equal(actual, expected):
        raise ValueError(f'saved {key} mismatch')
    return actual


def _canonical(pair):
    pair = np.sort(pair, axis=1)
    return pair[np.lexsort((pair[:, 1], pair[:, 0]))]


def audit(sh, cond, association, *, pair_batch=65536, max_pairs=2000000):
    """Return group counts and accepted original hit pairs, or raise on mismatch.

    ``sh`` contains measured single-hit columns, and association must retain its
    complete evkey census, source_hit_ev/sid, membership, geometry eligibility,
    group counts and candidate_hit_index/group. CLI also checks measured evkey.
    Membership is independently checked against production flag/time grouping;
    this does not rerun association, candidate pruning, hypothesis search or fit.
    ``group_raw_pairs`` includes excluded groups; ``group_considered`` does not.
    Exclusive rejection counts plus accepted equal considered for EVERY group.
    Accepted unordered original pairs are compared by lexicographic canonical
    order against ALL saved candidates, preserving multiplicity. No ambiguity
    rejection bits or candidate_used masks are consumed. Pair evaluation arrays
    are bounded by pair_batch; canonical replay arrays hold all accepted pairs,
    bounded by max_pairs. No hit/event loops.
    """
    pair_batch = _integer('pair_batch', pair_batch, 1)
    max_pairs = _integer('max_pairs', max_pairs, 1)
    keys = _array(association, 'evkey', 'iu')
    sh = {k: _array(sh, k, 'fiu') for k in sh}
    validate(keys, sh)
    for name in ('midx', 'midy', 'midz', 'dirx', 'diry', 'dirz', 'hlen'):
        _array(cond['straw'], name, 'fiu')
    supported_prepass = geometry_supported(sh, cond)
    h = _columns(sh, 'ev sid x y z time uvar timevar flag', ('ev', 'sid', 'flag'))
    _, mid, direction, measured_u, hlen = _geometry(h, cond)
    ev, sid, flag = h['ev'], h['sid'], h['flag']
    nh, ne = len(ev), len(keys)
    _match(association, 'source_hit_ev', ev, 'iu')
    _match(association, 'source_hit_sid', sid, 'iu')
    xy = direction[:, :2]
    xy_norm = np.linalg.norm(xy, axis=1)
    supported = supported_prepass
    _match(association, 'hit_geometry_supported', supported, 'b')
    bad = (1 << 6) | (1 << FLAG['dead']) | (1 << FLAG['noisy'])
    good = ((flag & bad) == 0) & ((flag & (1 << FLAG['tdiv'])) != 0)
    member = np.flatnonzero(good)
    member = member[np.lexsort((sid[member], sh['time'][member], ev[member]))]
    e, t = ev[member], h['time'][member]
    breaks = np.r_[True, (np.diff(e) != 0) | (np.diff(t) > 20.)][:len(member)]
    start = np.flatnonzero(breaks)
    n = np.diff(np.r_[start, len(member)]) if len(start) else np.empty(0, np.int64)
    mg = np.cumsum(breaks) - 1
    span = t[start + n - 1] - t[start]
    gev, ng = e[start], len(n)
    hg = np.full(nh, -1, np.int64)
    hg[member] = mg
    for key, expected, kind in (('hit_group', hg, 'iu'), ('member', member, 'iu'),
            ('member_group', mg, 'iu'), ('group_ev', gev, 'iu'), ('group_n', n, 'iu'),
            ('group_span', span, 'fiu')):
        _match(association, key, expected, kind)
    unsupported = np.zeros(ng, bool)
    np.logical_or.at(unsupported, mg, ~supported[member])
    _match(association, 'group_candidates_evaluated', ~unsupported, 'b')
    eligible = ~unsupported & (span <= 100.) & (n >= 2)
    raw = n * (n - 1) // 2
    considered = np.where(eligible, raw, 0)
    total = int(considered.sum())
    if total > max_pairs:
        raise ValueError(f'pair limit exceeded: {total} > max_pairs={max_pairs}; no pairs truncated')
    _match(association, 'group_npairs', considered, 'iu')
    saved_pair = _array(association, 'candidate_hit_index', 'iu')
    if saved_pair.ndim != 2 or saved_pair.shape[1] != 2:
        raise ValueError('invalid candidate_hit_index shape')
    if len(saved_pair) > total:
        raise ValueError('saved candidate count exceeds considered pairs')
    saved_group = _array(association, 'candidate_group', 'iu', (len(saved_pair),))
    if (np.any(saved_pair < 0) or np.any(saved_pair >= nh) or np.any(saved_group < 0)
            or np.any(saved_group >= ng)):
        raise ValueError('candidate index outside source/group census')
    saved_pair, saved_group = saved_pair.astype(np.int64), saved_group.astype(np.int64)
    if np.any(saved_pair[:, 0] == saved_pair[:, 1]) or np.any(hg[saved_pair] != saved_group[:, None]):
        raise ValueError('candidate source/group mismatch')
    saved_canonical = _canonical(saved_pair)
    if np.any(np.all(saved_canonical[1:] == saved_canonical[:-1], axis=1)):
        raise ValueError('duplicate candidate source pair')
    saved_count = np.bincount(saved_group, minlength=ng)
    _match(association, 'group_npoints', saved_count, 'iu')
    for key, expected in (('candidate_ev', gev[saved_group]),
            ('event_nhits', np.bincount(ev, minlength=ne)),
            ('event_ngood', np.bincount(e, minlength=ne)),
            ('event_ngroups', np.bincount(gev, minlength=ne)),
            ('event_npoints', np.bincount(gev[saved_group], minlength=ne))):
        if key in association:
            _match(association, key, expected, 'iu')
    counts = {key: np.zeros(ng, np.int64) for key in COUNTS}
    counts['considered'] = considered
    rows = np.where(eligible[mg], n[mg] - 1 - (np.arange(len(member)) - start[mg]), 0)
    row_end = np.cumsum(rows)
    row_start = row_end - rows
    pairs, groups = [], []
    for offset in range(0, total, pair_batch):
        rank = np.arange(offset, min(offset + pair_batch, total))
        a = np.searchsorted(row_end, rank, side='right')
        b = a + 1 + rank - row_start[a]
        i, j, g = member[a], member[b], mg[a]
        cross = xy[i, 0] * xy[j, 1] - xy[i, 1] * xy[j, 0]
        delta = mid[j, :2] - mid[i, :2]
        sine = np.abs(cross) / (xy_norm[i] * xy_norm[j])
        parallel = sine < .25
        denom = np.where(~parallel, cross, 1.)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            ui = (delta[:, 0] * xy[j, 1] - delta[:, 1] * xy[j, 0]) / denom
            uj = (delta[:, 0] * xy[i, 1] - delta[:, 1] * xy[i, 0]) / denom
            pi = mid[i] + ui[:, None] * direction[i]
            pj = mid[j] + uj[:, None] * direction[j]
            pull = np.stack(((ui - measured_u[i]) / np.sqrt(h['uvar'][i]),
                             (uj - measured_u[j]) / np.sqrt(h['uvar'][j])), axis=1)
            dz, dt = np.abs(pi[:, 2] - pj[:, 2]), h['time'][i] - h['time'][j]
            cuts = (parallel,
                    ~np.isfinite(pi).all(axis=1) | ~np.isfinite(pj).all(axis=1)
                    | ~np.isfinite(pull).all(axis=1) | ~np.isfinite(dz),
                    dz <= 1e-3, dz > 50.,
                    (np.abs(dt) > 20.) | (np.abs(dt) > 3. * np.hypot(np.sqrt(h['timevar'][i]), np.sqrt(h['timevar'][j]))),
                    (np.abs(ui) > hlen[i]) | (np.abs(uj) > hlen[j]),
                    np.any(np.abs(pull) > 3., axis=1))
        keep = np.ones(len(rank), bool)
        for name, reject in zip(REJECTIONS, cuts):
            counts['rejected_' + name] += np.bincount(g[keep & reject], minlength=ng)
            keep &= ~reject
        counts['accepted'] += np.bincount(g[keep], minlength=ng)
        if np.any(keep):
            pairs.append(np.column_stack((i[keep], j[keep])))
            groups.append(g[keep])
    pair = np.concatenate(pairs) if pairs else np.empty((0, 2), np.int64)
    group = np.concatenate(groups) if groups else np.empty(0, np.int64)
    if not np.array_equal(sum(counts[k] for k in COUNTS[1:]), considered):
        raise RuntimeError('exclusive per-group accounting failed')
    if not np.array_equal(_canonical(pair), saved_canonical):
        raise ValueError('accepted original candidate_hit_index replay mismatch')
    if not np.array_equal(counts['accepted'], saved_count):
        raise ValueError('accepted per-group replay mismatch')
    support = None
    if 'association_group_support' in association:
        support = _array(association, 'association_group_support', 'iu', (ng,))
        if np.any(support < 0) or np.any(support > n):
            raise ValueError('invalid saved association support')
    out = dict(evkey=keys.copy(), group=np.arange(ng), group_ev=gev, group_n=n, group_span=span,
               group_raw_pairs=raw, group_eligible=eligible, group_unsupported_geometry=unsupported,
               group_wide_time=span > 100., group_singleton=n < 2,
               candidate_hit_index=pair, candidate_group=group)
    out.update({'group_' + k: v for k, v in counts.items()})
    if support is not None:
        out['group_fixed_support'] = support.copy()
    diagnostics = dict(events=ne, hits=nh, good_hits=len(member), rejected_flag_hits=int((~good).sum()),
                       groups=ng, eligible_groups=int(eligible.sum()),
                       unsupported_geometry_groups=int(unsupported.sum()),
                       wide_time_groups=int((span > 100.).sum()), singleton_groups=int((n < 2).sum()),
                       raw_pairs=int(raw.sum()), excluded_pairs=int((raw - considered).sum()),
                       candidate_pair_replay_exact=True, exclusive_accounting_exact=True,
                       **{k: int(v.sum()) for k, v in counts.items()})
    if support is not None:
        no_support = support == 0
        diagnostics['no_support'] = dict(groups=int(no_support.sum()),
            zero_candidate_groups=int((no_support & (counts['accepted'] == 0)).sum()),
            **{k: int(v[no_support].sum()) for k, v in counts.items()})
    out['diagnostics'] = diagnostics
    return out


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hits', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--association', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--pair-batch', type=int, default=65536)
    parser.add_argument('--max-pairs', type=int, default=2000000)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    paths = dict(hits=args.hits, conditions=args.conditions, association=args.association,
                 implementation=Path(__file__), stereo_source=Path(__file__).parents[1]/'reco'/'stereoseed.py',
                 geometry_source=Path(__file__).parents[1]/'reco'/'seedassociation.py',
                 grouping_source=Path(__file__).with_name('run1a_seed.py'))
    inputs = {k: dict(path=str(p.resolve()), sha256=_sha256(p)) for k, p in paths.items()}
    with np.load(args.hits, allow_pickle=False) as saved:
        keys = saved['evkey']
        sh = {k[2:]: saved[k] for k in saved.files if k.startswith('s_')}
    with np.load(args.association, allow_pickle=False) as saved:
        association = dict(saved)
    _match(association, 'evkey', keys, 'iu')
    cond = json.loads(args.conditions.read_text())
    out = audit(sh, cond, association, pair_batch=args.pair_batch, max_pairs=args.max_pairs)
    report = dict(diagnostics=out.pop('diagnostics'), inputs=inputs, scope=__doc__,
                  pair_batch=args.pair_batch, max_pairs=args.max_pairs,
                  rejection_order=list(REJECTIONS),
                  counting='Exclusive first-failure counts; raw_pairs includes ineligible groups; considered excludes them.',
                  provenance_caveat='Current input hashes plus full-census/source/group identity and exact candidate-pair '
                      'replay do not authenticate the original association input files or ROOT/log/binary/config binding.',
                  not_run=['association', 'hypothesis search', 'fits', 'optimizer', 'sampling', 'threshold tuning'],
                  ambiguity_bits='Not consumed; control ambiguity flags are not association vetoes.')
    for key, path in paths.items():
        if _sha256(path) != inputs[key]['sha256']:
            raise ValueError(f'input changed during audit: {key}')
    columns = {k: v for k, v in out.items() if k == 'group' or k.startswith('group_')}
    columns.update(zip(('run', 'subrun', 'event'), out['evkey'][out['group_ev']].T))
    json.dumps(report, allow_nan=False)
    args.outdir.mkdir(parents=True, exist_ok=False)
    with (args.outdir/'groups.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    np.savez_compressed(args.outdir/'diagnostics.npz', **out)
    report['output_payload_sha256'] = _sha256(args.outdir/'diagnostics.npz')
    report['output_groups_sha256'] = _sha256(args.outdir/'groups.csv')
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    (args.outdir/'report.json').write_text(text)
    print(text, end='')
    return report


if __name__ == '__main__':
    main()
