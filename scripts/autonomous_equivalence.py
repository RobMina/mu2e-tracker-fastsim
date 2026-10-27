"""Compare saved autonomous track tables without refitting either arm."""
import argparse
import json
from pathlib import Path

import numpy as np

from trkfastsim.reco import tracktable
from trkfastsim.reco.autonomous import digest


def compare(before, after):
    a, b = tracktable.load(before), tracktable.load(after)
    ka, kb = (np.column_stack([t['events'][k] for k in ('run', 'subrun', 'event')]) for t in (a, b))
    if not np.array_equal(ka, kb):
        raise ValueError('different full event identities or order')
    for key in ('ev', 'seed_index'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError('different seed identities/order: ' + key)
    arrays = {}
    for name in ('goodfit', 'status', 'x', 'cov', 'chi2', 'ndof', 'hvalid', 'sh', 'sid', 'state', 'active', 'lr'):
        u, v = a[name], b[name]
        if u.shape != v.shape:
            raise ValueError('different shape for ' + name)
        if name in ('sh', 'sid', 'state', 'active', 'lr'):
            u, v = u[a['hvalid']], v[b['hvalid']]
        finite = np.isfinite(u) & np.isfinite(v)
        delta = v[finite].astype(float) - u[finite].astype(float)
        arrays[name] = {'identical': bool(np.array_equal(u, v, equal_nan=True)),
                        'n_changed': int(np.count_nonzero(~((u == v) | (np.isnan(u) & np.isnan(v))))),
                        'max_abs': float(np.max(np.abs(delta), initial=0)),
                        'nonfinite_pattern_identical': bool(np.array_equal(np.isfinite(u), np.isfinite(v)))}
    crossings = {}
    for name in ('sid', 'hit', 'xvalid', 'active', 'wall', 'gas', 'doca', 'docavar', 'dirdot'):
        u, v = a['xings'][name], b['xings'][name]
        if u.shape != v.shape:
            raise ValueError('different crossing shape for ' + name)
        valid = a['xings']['xvalid'] | b['xings']['xvalid']
        u, v = u[valid], v[valid]
        finite = np.isfinite(u) & np.isfinite(v)
        delta = v[finite].astype(float) - u[finite].astype(float)
        crossings[name] = {'identical': bool(np.array_equal(u, v, equal_nan=True)),
                            'max_abs': float(np.max(np.abs(delta), initial=0)),
                            'nonfinite_pattern_identical': bool(np.array_equal(np.isfinite(u), np.isfinite(v)))}
    accepted = (a['goodfit'] == 1) & (b['goodfit'] == 1)
    accepted_diff = {}
    for name in ('x', 'cov', 'chi2', 'ndof', 'state'):
        u, v = a[name][accepted], b[name][accepted]
        if name == 'state':
            u, v = u[a['hvalid'][accepted]], v[b['hvalid'][accepted]]
        finite = np.isfinite(u) & np.isfinite(v)
        diff = v[finite].astype(float) - u[finite].astype(float)
        accepted_diff[name] = {'identical': bool(np.array_equal(u, v, equal_nan=True)),
                               'max_abs': float(np.max(np.abs(diff), initial=0)),
                               'abs_q50_90_99': np.quantile(np.abs(diff), [.5, .9, .99]).tolist() if diff.size else None}
    track_keys = np.column_stack([ka[a['ev']], a['seed_index']])
    changed = a['goodfit'] != b['goodfit']
    parameter_differences = {}
    for i, name in enumerate(('a0_mm', 'b0_mm', 'a1', 'b1', 't0_ns', 'momentum_MeV')):
        delta = b['x'][accepted, i] - a['x'][accepted, i]
        sigma = np.sqrt(np.maximum(a['cov'][accepted, i, i], 0))
        finite = np.isfinite(delta) & np.isfinite(sigma) & (sigma > 0)
        pull = np.abs(delta[finite]) / sigma[finite]
        parameter_differences[name] = {'both_accepted': int(accepted.sum()), 'finite_positive_sigma': int(finite.sum()),
            'max_abs_delta': float(np.max(np.abs(delta), initial=0)),
            'abs_delta_q50_90_99': np.quantile(np.abs(delta), [.5, .9, .99]).tolist() if delta.size else None,
            'within_0p1_reference_sigma': int((pull < .1).sum()),
            'max_abs_pull': float(np.max(pull, initial=0))}
    return {'inputs': {str(p): digest(p) for p in (before, after)}, 'events': len(ka), 'seeds': len(a['ev']),
            'both_accepted': int(accepted.sum()), 'accepted_differences': accepted_diff,
            'parameter_differences': parameter_differences,
            'acceptance_change_keys_run_subrun_event_seed': track_keys[changed].tolist(),
            'before_after_good': np.column_stack([a['goodfit'][changed], b['goodfit'][changed]]).tolist(),
            'good_before': int(a['goodfit'].sum()), 'good_after': int(b['goodfit'].sum()),
            'gains': int(((a['goodfit'] == 0) & (b['goodfit'] == 1)).sum()),
            'losses': int(((a['goodfit'] == 1) & (b['goodfit'] == 0)).sum()),
            'arrays': arrays, 'crossings': crossings,
            'interpretation': 'same identities; float differences include backend/compiler/shape effects, not a native closure test'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to overwrite ' + str(args.out))
    result = compare(args.before, args.after)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('events', 'seeds', 'good_before', 'good_after', 'gains', 'losses')}, indent=2))
