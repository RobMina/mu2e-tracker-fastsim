"""Print bounded reconstruction/evaluation evidence from saved autonomous reports."""
import argparse
import json
from pathlib import Path


def report(path):
    obj = json.loads(Path(path).read_text())
    print(path)
    if 'timing' in obj:
        keys = ('n_events', 'n_hits', 'n_seeds', 'n_good', 'events_without_seeds', 'events_with_multiple_seeds',
                'preparation_s', 'writer_s', 'wall_s', 'invocation_work_s', 'numerical_validity')
        print(json.dumps({k: obj[k] for k in keys if k in obj}, indent=2))
        timing = obj['timing']
        print(json.dumps({k: v for k, v in timing.items() if k not in ('batch_s', 'batches')}, indent=2))
        if timing.get('batch_s'):
            print('cold_batch_s', timing['batch_s'][0])
    else:
        print(json.dumps({k: obj[k] for k in ('events', 'tracks', 'events_with_fit',
                                              'contingency_A_rows_B_columns_0_1_2plus')}, indent=2))
        cmp = obj.get('shared_hit_diagnostics', {}).get('comparison', {})
        print(json.dumps(cmp.get('track_match'), indent=2))
        for key, value in cmp.get('matched_pars_at_TT_Outer_entry', {}).items():
            delta = value['B_minus_A']
            print(key, 'n=', delta['n'], 'q16/50/84=', delta['q'][1:4])
        boot = obj.get('shared_hit_diagnostics', {}).get('matched_parameter_bootstrap')
        if boot is not None:
            print('event bootstrap', boot['bootstrap_replicates'], 'seed', boot['bootstrap_seed'])
            for key, value in boot['variables'].items():
                print(key, 'finite tracks/effective events', value['finite_tracks'], value['effective_events'])
                print(json.dumps(value['statistics'], sort_keys=True))
        print('conditional matched subset, not all-event residual closure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+')
    args = parser.parse_args()
    for path in args.reports:
        report(path)
