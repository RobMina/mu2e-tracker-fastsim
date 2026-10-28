"""Paired delete-one-file-group uncertainty on saved occupancy score differences.

This is a conditional-library sensitivity diagnostic, not a calibrated test of
model improvement. Recompute the full covariance and pseudoinverse for every
replicate. Rank and reference support must remain stable or the SE is unavailable.
Groups must be independent and comparably sized; few or highly unequal groups
make the unweighted delete-group error unreliable even if numerical guards pass.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from trkfastsim.analyses import occupancy as O
from scripts.secondary_replay import file_record


def difference_error(candidate, base, candidate_reps, base_reps):
    delta = candidate['chi2ndf_covariance'] - base['chi2ndf_covariance']
    reps = np.array([a['chi2ndf_covariance'] - b['chi2ndf_covariance']
                     for a, b in zip(candidate_reps, base_reps)])
    ranks = [r['covariance_rank'] for r in candidate_reps]
    base_ranks = [r['covariance_rank'] for r in base_reps]
    rows = [candidate, base, *candidate_reps, *base_reps]
    valid = (len(reps) > 1 and len(candidate_reps) == len(base_reps) and np.isfinite(reps).all()
             and np.isfinite(delta) and all(r['available'] and not r['null_space_incompatible'] for r in rows)
             and all(k == candidate['covariance_rank'] for k in ranks)
             and all(k == base['covariance_rank'] for k in base_ranks)
             and all(r.get('support_stable', True) for r in rows))
    error = np.sqrt((len(reps) - 1) / len(reps) * ((reps - reps.mean()) ** 2).sum()) if valid else None
    return {'difference': float(delta) if np.isfinite(delta) else None,
            'error': float(error) if error is not None else None, 'available': bool(valid),
            'candidate_rank': candidate['covariance_rank'], 'baseline_rank': base['covariance_rank'],
            'candidate_replica_ranks': ranks, 'baseline_replica_ranks': base_ranks,
            'replica_differences': [float(x) if np.isfinite(x) else None for x in reps],
            'unavailable_reason': None if valid else 'insufficient groups, unstable support/rank or unavailable covariance'}


def evaluate(marginals, crosses):
    arms = [a for a in marginals if a != 'g4']
    n = len(marginals['g4'])
    if n == 0:
        raise ValueError('at least one file group is required')
    if set(crosses) != set(arms) or 'base' not in arms:
        raise ValueError('paired cross moments require baseline and every candidate')
    if any(len(x) != n for x in marginals.values()) or any(len(x) != n for x in crosses.values()):
        raise ValueError('moment groups must align')
    totals = {a: {k: sum(g[k] for g in groups) for k in groups[0]} for a, groups in marginals.items()}
    ct = {a: {k: sum(g[k] for g in groups) for k in groups[0]} for a, groups in crosses.items()}
    full = {a: O.compare(totals[a], totals['g4'], cross=ct[a]) for a in arms}
    support = totals['g4']['C'] >= 20
    reps = {a: [] for a in arms}
    for i in range(n):
        leave = {a: {k: v - marginals[a][i][k] for k, v in moments.items()} for a, moments in totals.items()}
        stable = np.array_equal(leave['g4']['C'] >= 20, support)
        for a in arms:
            cross = {k: v - crosses[a][i][k] for k, v in ct[a].items()}
            result = O.compare(leave[a], leave['g4'], cross=cross)
            result['support_stable'] = stable
            reps[a].append(result)
    return {'groups': n, 'scores': {a: full[a]['chi2ndf_covariance'] for a in arms},
            'contrasts': {a: difference_error(full[a], full['base'], reps[a], reps['base'])
                          for a in arms if a != 'base'}}


def score_closes(score, saved):
    if np.isnan(score) and np.isnan(saved['chi2ndf_covariance']):
        return not saved['available']
    return bool(np.isclose(score, saved['chi2ndf_covariance'], rtol=1e-10))


def run(report_path, moments_path, out):
    report = json.loads(Path(report_path).read_text())
    record = file_record(moments_path)
    if report['status'] != 'complete' or record['sha256'] != report['occupancy_group_moments']['sha256']:
        raise ValueError('incomplete report or mismatched moment archive')
    result = {'interpretation': __doc__, 'inputs': [file_record(report_path), record],
              'sources': [file_record(__file__), file_record(O.__file__)], 'occupancy': {}}
    arms = list(report['occupancy'])
    n = len(report['files'])
    with np.load(moments_path, allow_pickle=False) as archive:
        for kind in report['occupancy']['base']:
            def collect(prefix, arm):
                keys = [k.rsplit('/', 1)[-1] for k in archive if k.startswith(f'0/{prefix}/{arm}/{kind}/')]
                return [{k: archive[f'{i}/{prefix}/{arm}/{kind}/{k}'] for k in keys} for i in range(n)]
            marginal = {a: collect('marginal', a) for a in [*arms, 'g4']}
            cross = {a: collect('cross', a) for a in arms}
            r = evaluate(marginal, cross)
            for a, score in r['scores'].items():
                if not score_closes(score, report['occupancy'][a][kind]):
                    raise ValueError('saved occupancy score closure failed')
            result['occupancy'][kind] = r
    with open(out, 'x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v['contrasts'] for k, v in result['occupancy'].items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report')
    p.add_argument('moments')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run(a.report, a.moments, a.out)


if __name__ == '__main__':
    main()
