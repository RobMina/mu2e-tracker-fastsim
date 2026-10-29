"""Trace shared donor draws into joint lattice migrations without changing physics.

Consume saved paired base/topology census manifests. Check the reconstructed
lattice, energy, length and time against BOTH saved outputs before reporting.
Plane-only and transverse-only arms are lattice diagnostics, never step files.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from trkfastsim import geometry, library, steps, tables as T
from trkfastsim.analyses import occupancy as O, seccensus as C, secmigration as M
from scripts import secondary_replay as R

MODES = {'base': ('default', 'default'), 'plane_only': ('topology', 'default'),
         'transverse_only': ('default', 'topology'), 'topology': ('topology', 'topology')}
ARMS = [*MODES, 'exterior_only', 'nonexterior_only']
LATTICE = T.KEY + ['sid', 'sp', 'plane', 'panel', 'straw', 'upanel', 'eion', 'len', 't']
STATES = ['off_lattice', 'muon_straw_reject', 'near_panel', 'same_plane_other', 'plane_1_2', 'plane_ge3']


def restrict(frame, keys):
    return frame.merge(keys, on=T.KEY, how='inner', validate='many_to_one')


def exact_lattice(saved, reconstructed):
    key = T.KEY + ['sp', 'sid']
    a, b = (x[LATTICE].sort_values(key).reset_index(drop=True) for x in (saved, reconstructed))
    check = R.exact_check(a, b)
    R.require_checks({'lattice': check}, 'trace differs from saved off-axis output')
    return check


def transition_summary(base, candidate, mu):
    if not np.array_equal(base['draw_row'], candidate['draw_row']):
        raise ValueError('raw draw rows are not aligned')
    lb, lt = M.classify(base, mu), M.classify(candidate, mu)
    sb = np.where(~base.valid, 0, np.where(base.in_muon, 1, lb + 1)).astype(int)
    st = np.where(~candidate.valid, 0, np.where(candidate.in_muon, 1, lt + 1)).astype(int)
    if np.any((sb < 0) | (sb >= 6) | (st < 0) | (st >= 6)):
        raise ValueError('invalid raw-row state')
    exterior = base.source_exterior.to_numpy(dtype=int)
    source_class = np.where(exterior < 0, 0, np.where(exterior > 0, 1, np.where(base.dpl != 0, 2, 3)))
    classes = np.bincount(source_class * 36 + sb * 6 + st, minlength=144).reshape(4, 6, 6)
    migration = (exterior + 1) * 36 + sb * 6 + st
    joint = np.bincount(migration, minlength=108).reshape(3, 6, 6)
    eion = np.bincount(migration, weights=base.eion, minlength=108).reshape(3, 6, 6)
    changes = {c: base[c].to_numpy() != candidate[c].to_numpy() for c in ('plane', 'panel', 'straw', 'sid')}
    bit = changes['plane'].astype(int) + 2 * changes['panel'] + 4 * changes['straw']
    summary = {'rows': len(base), 'states': STATES, 'exterior_order': [-1, 0, 1],
               'transition_by_exterior': joint.tolist(), 'transition_eion_MeV': eion.tolist(),
               'transition_by_source_class': classes.tolist(),
               'source_class_order': ['below', 'above', 'internal_gap', 'anchor_plane'],
               'changed_lattice_bitmask_plane_panel_straw': np.bincount(bit, minlength=8).tolist(),
               'same_sid_changed_time': int(((~changes['sid']) & (base.t.to_numpy() != candidate.t.to_numpy())).sum()),
               'same_sid_changed_anchor': int(((~changes['sid']) &
                   (base.anchor_default_k.to_numpy() != candidate.anchor_topology_k.to_numpy())).sum()),
               'kept_both': int((base.keep & candidate.keep).sum()),
               'lost': int((base.keep & ~candidate.keep).sum()),
               'gained': int((~base.keep & candidate.keep).sum()),
               'plane_reflections': [int(base.reflected_plane.sum()), int(candidate.reflected_plane.sum())],
               'straw_reflections': [int(base.reflected_straw.sum()), int(candidate.reflected_straw.sum())]}
    if int(joint.sum()) != len(base):
        raise ValueError('nonconserving raw migration matrix')
    return summary


def add_summary(total, row):
    for key, value in row.items():
        if key in ('states', 'exterior_order', 'source_class_order'):
            total[key] = value
        elif isinstance(value, list):
            total[key] = (np.asarray(total.get(key, np.zeros_like(value))) + np.asarray(value)).tolist()
        else:
            total[key] = total.get(key, 0) + value


def location_counts(frame, mu):
    loc = M.classify(frame.assign(valid=True), mu)
    return np.bincount(loc, minlength=5).astype(int).tolist()


def donor_split(base, topology, exterior=True):
    """Diagnostic split on donor truth; reaggregate after choosing per-row anchors."""
    if not np.array_equal(base.draw_row, topology.draw_row):
        raise ValueError('raw draw rows are not aligned')
    take = (base.source_exterior != 0).to_numpy()
    if not exterior:
        take = ~take
    return pd.concat([base[~take], topology[take]], ignore_index=True).sort_values('draw_row').reset_index(drop=True)


def occupancy_rows(frame, mu, acc):
    s = frame.copy()
    s['mu'] = s.sp < steps.SECONDARY_KEY0
    return C._occ(s, acc)


def run(census_path, topology_path, lib_path, gdml, outdir, max_groups=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    old, top = (json.loads(Path(p).read_text()) for p in (census_path, topology_path))
    if not old['paired'] or not top['paired'] or old['selected_events'] != top['selected_events']:
        raise ValueError('saved censuses must use the same paired selection')
    candidates = {Path(p).name: p for p in top['input_files']['fast']}
    groups = old['file_groups']
    if len(candidates) != len(groups) or {Path(g['fast']).name for g in groups} != set(candidates):
        raise ValueError('different fast-file coverage')
    if set(old['input_files']['g4']) != set(top['input_files']['g4']):
        raise ValueError('different G4 coverage')
    if max_groups is not None:
        if max_groups < 1:
            raise ValueError('max groups must be positive')
        groups = groups[:max_groups]
    geom = geometry.from_gdml(gdml)
    lib = library.read(lib_path)
    report = {'scope': 'shared-draw lattice diagnostic; hybrids have no endpoint/digi/reco interpretation',
              'seed': 1, 'file_index_contract': 'base census manifest must retain full original seed-1 campaign order',
              'uncertainty': 'event-cluster occupancy; file-group jackknife locations; fixed-library conditional',
              'inputs': [R.file_record(p) for p in (census_path, topology_path, lib_path, gdml)],
              'sources': [R.file_record(p) for p in (__file__, M.__file__, steps.__file__, library.__file__, O.__file__)],
              'files': [], 'migration': {'all': {}, 'both_selected': {}}, 'status': 'running'}
    occ = {a: {} for a in [*ARMS, 'g4']}
    cross = {a: {} for a in ARMS}
    moment_rows, cross_rows = [], []
    t0 = time.perf_counter()
    try:
        for index, group in enumerate(groups):
            start = time.perf_counter()
            saved = R.read_baseline(group['fast'])
            other = R.read_baseline(candidates[Path(group['fast']).name])
            mu, _ = R.reconstruct_primary(saved)
            sampler = library.OffAxisSampler(lib, np.random.default_rng([1, index, 2]))
            ranked, first, drawn = M.draw_context(mu, sampler)
            traces = {a: M.trace(ranked, first, drawn, geom, plane_mode=p, transverse_mode=t)
                      for a, (p, t) in MODES.items()}
            traces['exterior_only'] = donor_split(traces['base'], traces['topology'])
            traces['nonexterior_only'] = donor_split(traces['base'], traces['topology'], exterior=False)
            finals = {a: M.aggregate(x) for a, x in traces.items()}
            checks = {a: exact_lattice(s['sgsteps'][s['sgsteps'].sp >= steps.OFFAXIS_KEY0], finals[a])
                      for a, s in [('base', saved), ('topology', other)]}
            reference = pd.concat([C._load(p, True) for p in group['g4']], ignore_index=True)
            selected_fast = C._selected_keys(mu.assign(isprim=1, pdg=13))
            selected_g4 = C._selected_keys(reference)
            both = selected_fast.merge(selected_g4, on=T.KEY, validate='one_to_one')
            smr = restrict(reference, both)
            smu = restrict(mu, both)
            common = restrict(saved['sgsteps'][saved['sgsteps'].sp < steps.OFFAXIS_KEY0], both)
            ref_mu = smr[smr.mu]
            local_occ = {a: {} for a in [*ARMS, 'g4']}
            local_cross = {a: {} for a in ARMS}
            ref_occ = C._occ(smr, local_occ['g4'])
            row = {'index': index, 'inputs': [R.file_record(p) for p in [group['fast'], candidates[Path(group['fast']).name]]],
                   'checks': checks, 'selected': {'fast': len(selected_fast), 'g4': len(selected_g4), 'both': len(both)},
                   'mu_steps': {'fast': len(smu), 'g4': len(ref_mu)},
                   'locations': {'g4': location_counts(smr[~smr.mu], ref_mu)}, 'arms': {}}
            for scope, keys in [('all', None), ('both_selected', both)]:
                b, t = traces['base'], traces['topology']
                if keys is not None:
                    b, t = restrict(b, keys), restrict(t, keys)
                add_summary(report['migration'][scope], transition_summary(b, t, ranked))
            for name in ARMS:
                final = restrict(finals[name], both)
                arm = pd.concat([common, final], ignore_index=True)
                sec = arm[arm.sp >= steps.SECONDARY_KEY0]
                row['locations'][name] = location_counts(sec, smu)
                t = traces[name]
                row['arms'][name] = {'raw': len(t), 'kept': int(t.keep.sum()), 'final': len(finals[name]),
                                     'merges': int(t.keep.sum()) - len(finals[name]),
                                     'selected_offaxis': len(final), 'selected_eion': float(sec.eion.sum())}
                arm_occ = occupancy_rows(arm, smu, local_occ[name])
                for kind in arm_occ:
                    local_cross[name][kind] = O.accumulate_cross(*arm_occ[kind], *ref_occ[kind])
            for a, kinds in local_occ.items():
                for kind, moments in kinds.items():
                    occ[a][kind] = O.merge(occ[a][kind], moments) if kind in occ[a] else moments
            for a, kinds in local_cross.items():
                for kind, moments in kinds.items():
                    cross[a][kind] = O.merge(cross[a][kind], moments) if kind in cross[a] else moments
            moment_rows.append(local_occ)
            cross_rows.append(local_cross)
            row['elapsed_s'] = time.perf_counter() - start
            report['files'].append(row)
            print(f'group {index}: {len(drawn)} shared rows, both {len(both)}, {row["elapsed_s"]:.2f} s', flush=True)
        report['occupancy'] = {a: {k: O.compare(v, occ['g4'][k], cross=cross[a][k])
                                  for k, v in moments.items()} for a, moments in occ.items() if a != 'g4'}
        payload = {}
        for index, (marginal, paired) in enumerate(zip(moment_rows, cross_rows)):
            for prefix, source in [('marginal', marginal), ('cross', paired)]:
                for arm, kinds in source.items():
                    for kind, moments in kinds.items():
                        for key, value in moments.items():
                            payload[f'{index}/{prefix}/{arm}/{kind}/{key}'] = np.asarray(value)
        np.savez_compressed(outdir / 'occupancy_groups.npz', **payload)
        report['occupancy_group_moments'] = R.file_record(outdir / 'occupancy_groups.npz')
        df = np.array([r['mu_steps']['fast'] for r in report['files']])
        dr = np.array([r['mu_steps']['g4'] for r in report['files']])
        ref = np.array([r['locations']['g4'] for r in report['files']])
        report['location_comparisons'] = {}
        for a in ARMS:
            f = np.array([r['locations'][a] for r in report['files']])
            report['location_comparisons'][a] = [
                {'location': name, 'counts': [int(f[:, k].sum()), int(ref[:, k].sum())],
                 'ratio_minus_1': C._jack(f[:, k], df, ref[:, k], dr)[0],
                 'error': C._jack(f[:, k], df, ref[:, k], dr)[1]}
                for k, name in enumerate(C.LOC)]
        if max_groups is None:
            report['saved_census_closure'] = {}
            for name, saved in [('base', old), ('topology', top)]:
                loc = report['location_comparisons'][name]
                if [r['counts'] for r in loc] != [r['steps'] for r in saved['breakdowns']['location']]:
                    raise ValueError(f'{name}: saved location census did not close')
                for kind, value in report['occupancy'][name].items():
                    target = saved['occupancy_steps'][kind]
                    if not np.isclose(value['chi2ndf_covariance'], target['chi2ndf_covariance'], rtol=1e-10):
                        raise ValueError(f'{name}.{kind}: saved covariance did not close')
                report['saved_census_closure'][name] = True
        report['status'] = 'complete'
    except Exception as exc:
        report['status'], report['error'] = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_s'] = time.perf_counter() - t0
        with (outdir / 'migration.json').open('x') as stream:
            json.dump(report, stream, indent=2, default=R._json_scalar)
    print(json.dumps({'locations': report['location_comparisons'], 'occupancy': {
        a: {k: v['chi2ndf_covariance'] for k, v in b.items()} for a, b in report['occupancy'].items()}}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--census', required=True)
    p.add_argument('--topology-census', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--gdml', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--max-groups', type=int)
    a = p.parse_args()
    run(a.census, a.topology_census, a.library, a.gdml, a.outdir, a.max_groups)


if __name__ == '__main__':
    main()
