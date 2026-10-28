"""Summarize saved shared-draw migration and paired file-group contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.secondary_replay import file_record


def contrast(fast, reference, den_fast, den_ref):
    """Contrast/reference rate and delete-one-nonempty-group SE; common paired groups."""
    f, r, df, dr = (np.asarray(x, float) for x in (fast, reference, den_fast, den_ref))
    if any(a.shape != f.shape for a in (r, df, dr)) or f.ndim != 1:
        raise ValueError('paired group arrays must align')
    def value(f, r, df, dr):
        return f * dr / (df * r) if df > 0 and r > 0 and dr > 0 else np.nan
    totals = [a.sum() for a in (f, r, df, dr)]
    estimate = value(*totals)
    active = (f != 0) | (r != 0) | (df != 0) | (dr != 0)
    reps = np.array([value(*(t - a[k] for t, a in zip(totals, (f, r, df, dr))))
                     for k in np.flatnonzero(active)])
    if len(reps) < 2 or not np.isfinite(reps).all():
        error = None
    else:
        error = float(np.sqrt((len(reps) - 1) / len(reps) * ((reps - reps.mean()) ** 2).sum()))
    return {'relative_change': float(estimate) if np.isfinite(estimate) else None, 'error': error}


def summarize(data):
    if data['status'] != 'complete':
        raise ValueError('migration report is not complete')
    files = data['files']
    df, dr = (np.array([f['mu_steps'][a] for f in files]) for a in ('fast', 'g4'))
    loc = {a: np.array([f['locations'][a] for f in files]) for a in files[0]['locations']}
    out = {'files': len(files), 'elapsed_s': data['elapsed_s'],
           'saved_census_closure': data.get('saved_census_closure'),
           'interpretation': 'shared-draw diagnostic; hybrids are lattice-only; not digi/reco closure',
           'contrast_error': 'paired delete-one-file-group; reference and primary denominators fluctuate; fixed-library conditional',
           'location_ratios': data['location_comparisons'],
           'occupancy': {a: {k: v['chi2ndf_covariance'] for k, v in b.items()} for a, b in data['occupancy'].items()},
           'paired_location_changes': {}, 'raw_migration': {}}
    contrasts = [(a + '_minus_base', loc[a] - loc['base']) for a in loc if a not in ('g4', 'base')]
    contrasts.append(('joint_interaction', loc['topology'] - loc['plane_only'] - loc['transverse_only'] + loc['base']))
    if 'exterior_only' in loc:
        contrasts.append(('donor_split_interaction', loc['topology'] - loc['exterior_only'] - loc['nonexterior_only'] + loc['base']))
    for name, f in contrasts:
        out['paired_location_changes'][name] = [
            {'location': row['location'], 'delta_steps': int(f[:, k].sum()),
             **contrast(f[:, k], loc['g4'][:, k], df, dr)}
            for k, row in enumerate(data['location_comparisons']['base'])]
    for scope, record in data['migration'].items():
        matrices = np.asarray(record['transition_by_source_class'])
        raw = {k: record[k] for k in ('rows', 'changed_lattice_bitmask_plane_panel_straw',
                'same_sid_changed_time', 'same_sid_changed_anchor', 'kept_both', 'lost', 'gained',
                'plane_reflections', 'straw_reflections')}
        raw['source_classes'] = {}
        for name, matrix in zip(record['source_class_order'], matrices):
            total = int(matrix.sum())
            base, topo = matrix.sum(axis=1), matrix.sum(axis=0)
            raw['source_classes'][name] = {'rows': total,
                'base_states': dict(zip(record['states'], base.tolist())),
                'topology_states': dict(zip(record['states'], topo.tolist())),
                'state_changes': dict(zip(record['states'], (topo - base).tolist())),
                'same_plane_other_to_1_2': int(matrix[3, 4]), 'same_plane_other_to_ge3': int(matrix[3, 5]),
                'near_panel_to_same_plane_other': int(matrix[2, 3]),
                'same_plane_other_to_near_panel': int(matrix[3, 2]),
                'fraction_class_changed': float(1 - np.trace(matrix) / total) if total else None}
        out['raw_migration'][scope] = raw
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = summarize(json.loads(Path(a.input).read_text()))
    result['input'] = file_record(a.input)
    result['source'] = file_record(__file__)
    text = json.dumps(result, indent=2)
    with open(a.out, 'x') as stream:
        stream.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
