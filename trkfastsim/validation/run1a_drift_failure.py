"""Read-only boundary/POCA sensitivity study of saved analytic-candidate finals.

No fit, LR change, direction selection or status repair. A polished-root probe
changes the numerical evaluator and is diagnostic only, not a replacement for
the saved objective or a reason to count an old failure as a success.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from trkfastsim.reco.stereoseed import _integer
from . import run1a_drift_diagnostic as prior
from . import run1a_stabledrift as stable
from .run1a_timing import measurement_arrays, prepare_saved
from .run1a_wirefit import _quantiles, _sha256


def _equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')


def validate_candidate(saved, candidate):
    """Keep every historical array, including empty/failed rows and trace data."""
    for key, value in saved.items():
        if key not in candidate or not _equal(value, candidate[key]):
            raise ValueError(f'candidate baseline copy differs: {key}')
    ng, nh = len(saved['group_ev']), len(saved['source_selected'])
    for name, tail, kind in (('local_parameters', (5,), 'f'), ('t0', (), 'f'), ('z_ref', (), 'f'),
                            ('flight_direction', (), 'f'), ('parameter_scale', (6,), 'f'),
                            ('objective_valid', (), 'b'), ('success', (), 'b'),
                            ('initial_objective_valid', (), 'b'), ('final_objective', (), 'f'),
                            ('gradient_norm', (), 'f'), ('analytic_differentiable', (), 'b'),
                            ('status', (), 'US'), ('certificate_status', (), 'US')):
        a = np.asarray(candidate['candidate_'+name])
        if a.shape != (ng, 2)+tail or a.dtype.kind not in kind:
            raise ValueError(f'invalid candidate field: {name}')
    for name, kind in (('mask', 'b'), ('initial_lr', 'f')):
        a = np.asarray(candidate['source_candidate_'+name])
        if a.shape != (nh, 2) or a.dtype.kind not in kind:
            raise ValueError(f'invalid candidate hit field: {name}')
    for new, old in (('source_candidate_mask', 'source_drift_mask'),
                     ('source_candidate_initial_lr', 'source_drift_initial_LR'),
                     ('candidate_parameter_scale', 'drift_parameter_scale'),
                     ('candidate_z_ref', 'drift_z_ref'),
                     ('candidate_flight_direction', 'drift_flight_direction'),
                     ('candidate_initial_objective_valid', 'drift_initial_objective_valid')):
        if not _equal(candidate[new], saved[old]):
            raise ValueError(f'candidate fixed input differs: {new}')
    if (not np.array_equal(candidate['candidate_success'], candidate['candidate_status'] == 'converged_certified')
            or np.any(candidate['candidate_success'] & ~candidate['candidate_objective_valid'])
            or np.any(candidate['candidate_objective_valid'] & ~saved['profile_profile_valid'])):
        raise ValueError('candidate validity/status inconsistent')
    if any(k.startswith(('failure_', 'source_failure_')) for k in candidate):
        raise ValueError('input must be candidate, not failure diagnostic output')


def summarize(candidate, payload):
    """Overlapping sensitivity evidence, never a causal partition or new success."""
    from trkfastsim.reco.driftprobecompare import BRANCHES
    status = candidate['candidate_status']
    valid = payload['failure_original_objective_valid']
    polished = payload['failure_polished_objective_valid']
    same = payload['failure_center_same_branch']
    groups = candidate['hit_group'][candidate['source_selected']]
    selected = candidate['source_selected']
    boundary_names = ('time_boundary', 'phi_boundary', 'offset_boundary', 'signed_error_boundary',
                      'angle_branch', 'doca_branch')
    boundary = {}
    for name in boundary_names:
        key = 'source_failure_'+name
        if key not in payload:
            continue
        a = payload[key][selected]
        rows = np.zeros(status.shape, bool)
        np.logical_or.at(rows, groups, a)
        boundary[name] = (a, rows)
    output = {}
    for label in np.unique(status):
        rows = status == label
        use = rows & valid
        both = use & polished
        hit_use = rows[groups]
        threshold = 1e-6*(1+np.sqrt(2*np.where(valid, payload['failure_original_objective'], 0.)))
        d = dict(hypotheses=int(rows.sum()), original_valid=int(use.sum()),
            polished_valid=int(both.sum()), polished_center_same_branch=int((both & same).sum()),
            original_checked=int((rows & payload['failure_original_certified']).sum()),
            polished_diagnostic_checked=int((rows & payload['failure_polished_certified']).sum()),
            polished_checked_and_same_center_branch=int((rows & same & payload['failure_polished_certified']).sum()),
            polished_minus_original_objective=_quantiles((payload['failure_polished_objective']-
                payload['failure_original_objective'])[both]),
            original_gradient_over_threshold=_quantiles((payload['failure_original_gradient_norm']/threshold)[use]),
            polished_gradient_over_original_threshold=_quantiles((payload['failure_polished_gradient_norm']/threshold)[both]),
            boundary_reasons={name: dict(hypotheses=int((rows & r).sum()),
                hit_hypothesis_entries=int(a[hit_use].sum())) for name, (a, r) in boundary.items()})
        for name in ('original', 'polished'):
            prefix = 'failure_'+name+'_certificate_'
            compatible = payload[prefix+'same_branch'] & payload[prefix+'numeric_good']
            take = rows[..., None] & compatible
            d[name+'_compatible_rows_by_scale'] = np.sum(take, axis=(0, 1)).tolist()
            d[name+'_passing_rows_by_scale'] = np.sum(rows[..., None] & payload[prefix+'scale_pass'], axis=(0, 1)).tolist()
            d[name+'_compatible_gradient_difference_over_original_threshold'] = _quantiles(
                (payload[prefix+'gradient_difference']/threshold[..., None])[take])
        for name in ('poca_shift', 'original_correction', 'polished_correction'):
            d['abs_'+name+'_mm'] = _quantiles(np.abs(payload['source_failure_'+name][selected][hit_use]))
        active = candidate['candidate_trace_trial_evaluated'] & rows[..., None, None]
        accepted = candidate['candidate_trace_accepted_backtrack']
        index = np.maximum(accepted, 0)[..., None]
        branch = np.take_along_axis(candidate['candidate_trace_trial_same_branch'], index, axis=-1)[..., 0]
        d['accepted_cross_branch_updates'] = int(np.sum(rows[..., None] & (accepted >= 0) & ~branch))
        d['unchanged_parameter_trial_attempts'] = int(np.sum(active & candidate['candidate_trace_trial_same_parameters']))
        common = rows[..., None, None] & payload['failure_decomposition_valid'] & payload['failure_decomposition_same_branches'] & same[..., None, None]
        terms = ('gradient_difference', 'probe_root_contribution', 'polished_fd_remainder',
                 'analytic_center_contribution', 'decomposition_roundoff')
        d['common_center_and_probe_branch_coordinate_checks'] = int(common.sum())
        for term in terms:
            a = payload['failure_decomposition_'+term]/threshold[..., None, None]
            d['abs_'+term+'_over_original_threshold'] = _quantiles(np.abs(a[common]))
            d['abs_t0_'+term+'_over_original_threshold'] = _quantiles(np.abs(a[..., 5][common[..., 5]]))
        t0 = rows[..., None] & payload['failure_decomposition_valid'][..., 5]
        d['t0_original_poca_changed_checks'] = int(np.sum(t0 & ~payload['failure_decomposition_t0_original_poca_identical']))
        d['t0_polished_poca_changed_checks'] = int(np.sum(t0 & ~payload['failure_decomposition_t0_polished_poca_identical']))
        for term in ('plus_increment_ratio', 'minus_increment_ratio'):
            a = payload['failure_decomposition_'+term]
            d['abs_'+term+'_minus_one'] = _quantiles(np.abs(a[common]-1.))
        for arm in ('original', 'polished'):
            changed = payload['failure_decomposition_'+arm+'_branch_changes']
            d[arm+'_probe_branch_changed_hypotheses_by_scale'] = {name: np.sum(
                rows[..., None] & np.any(changed[..., i], axis=-1), axis=(0, 1)).tolist() for i, name in enumerate(BRANCHES)}
            d[arm+'_certificate_status_counts'] = stable._counts(payload['failure_'+arm+'_certificate_status'][rows])
            no_pair = rows & (candidate['candidate_certificate_status'] == 'no_admissible_stencil_pair')
            d[arm+'_no_pair_probe_branch_changed_hypotheses_by_scale'] = {name: np.sum(
                no_pair[..., None] & np.any(changed[..., i], axis=-1), axis=(0, 1)).tolist() for i, name in enumerate(BRANCHES)}
        if 'offset_boundary' in boundary:
            offset_hit = hit_use & boundary['offset_boundary'][0]
            d['guarded_offset_abs_edge_jump_mm'] = _quantiles(np.abs(payload['source_failure_offset_edge_jump'][selected][offset_hit]))
            d['guarded_offset_edge_distance_mm'] = _quantiles(payload['source_failure_offset_edge_distance'][selected][offset_hit])
        d['polishing_changed_hit_hypothesis_entries'] = int(payload['source_failure_polish_changed'][selected][hit_use].sum())
        d['polishing_arithmetic_floor_hit_hypothesis_entries'] = int(payload['source_failure_polish_arithmetic_floor'][selected][hit_use].sum())
        d['by_original_certificate_status'] = {str(name): dict(hypotheses=int(np.sum(rows & (candidate['candidate_certificate_status'] == name))),
            polished_diagnostic_checked=int(np.sum(rows & (candidate['candidate_certificate_status'] == name) & payload['failure_polished_certified'])))
            for name in np.unique(candidate['candidate_certificate_status'][rows])}
        output[str(label)] = d
    return dict(by_saved_candidate_status=output, saved_status_counts=stable._counts(status),
        original_arrays_preserved=True, accepted_lr_updates=0, assigned_directions=0,
        diagnostic_only=True, mechanism_tags_are_overlapping_not_causal_partition=True,
        interpretation='Polished-root checks change the numerical evaluator, not saved parameters/LR/support. '
            'They are sensitivity evidence, NOT promotions, fit improvements or exact-root certificates.',
        boundary_scope='Guarded arithmetic proximity, not proof of a mathematical discontinuity or a minimum.',
        input_provenance_caveat='Current hash binding and replay do not restore missing original association '
            'input hashes or historical baseline output-NPZ binding.')


def run(keys, sh, cond, association, wirefit, geom, response, model, saved, candidate, *,
        bz, mass=105.6583755, group_batch=32):
    """Reconstruct measured inputs and inspect fixed saved finals, without a refit."""
    from trkfastsim.reco.driftfitdiagnostic import _Evaluator, _empty_state
    from trkfastsim.reco.driftfailure import diagnose, _PolishedEvaluator
    from trkfastsim.reco.driftjacobian import linearize
    from trkfastsim.reco.driftprobecompare import compare
    group_batch = _integer('group_batch', group_batch, 1)
    data = prepare_saved(keys, sh, cond, association, wirefit)
    prior.validate_saved(data, wirefit, saved)
    validate_candidate(saved, candidate)
    stable.original_seeds(wirefit, saved)
    measurements = measurement_arrays(data, geom, response)
    ng, nh = len(data['gev']), len(data['hg'])
    count, selected = data['count'], data['selected']
    width = int(count.max(initial=0))
    if width > 256:
        raise ValueError('diagnostic support bound exceeded')
    batch = min(group_batch, 512, max(1, 65536//(2*max(1, width))))
    starts = np.r_[0, np.cumsum(count)]
    rows, hits = {}, {}
    for begin in range(0, max(1, ng), batch):
        groups = np.arange(begin, min(ng, begin+batch))
        valid = np.arange(width)[None, :] < count[groups, None]
        idx = np.zeros(valid.shape, np.int64)
        if len(selected):
            idx[valid] = selected[(starts[groups, None]+np.arange(width)[None, :])[valid]]
        padded = {}
        for key, value in measurements.items():
            a = np.zeros(valid.shape+value.shape[1:], value.dtype)
            a[valid] = value[idx[valid]]
            padded[key] = np.repeat(a, 2, axis=0)
        u, variance = np.zeros(valid.shape), np.ones(valid.shape)
        u[valid], variance[valid] = data['u'][idx[valid]], data['uvar'][idx[valid]]
        lr = stable._replay_initial(groups, valid, idx, padded, u, data, wirefit, saved, bz=bz, mass=mass)
        ev = _Evaluator(candidate['candidate_local_parameters'][groups].reshape(-1, 5),
            saved['drift_z_ref'][groups].reshape(-1), candidate['candidate_t0'][groups].reshape(-1),
            saved['drift_flight_direction'][groups].reshape(-1), **padded,
            measured_u=np.repeat(u, 2, axis=0), uvar=np.repeat(variance, 2, axis=0),
            valid=np.repeat(valid, 2, axis=0), model=model, bz=bz, mass=mass, initial_lr=lr,
            parameter_scale=saved['drift_parameter_scale'][groups].reshape(-1, 6),
            seed_valid=saved['profile_profile_valid'][groups].reshape(-1))
        parameters = ev.parameters.copy()
        result = diagnose(ev, parameters)
        state = ev(parameters) if ev.batch and ev.hits else _empty_state(ev)
        wrapper = _PolishedEvaluator(ev)
        polished = wrapper(parameters)
        decomposition = compare(ev, wrapper, parameters, state, polished,
                                linearize(ev, parameters, state), linearize(wrapper, parameters, polished))
        result.update({'decomposition_'+k: v for k, v in decomposition.items()})
        for key, value in result.items():
            a = np.asarray(value)
            if a.shape[:1] != (2*len(groups),):
                raise RuntimeError(f'invalid diagnostic row shape: {key}')
            a = a.reshape((len(groups), 2)+a.shape[1:])
            if key.startswith('hit_'):
                if a.shape[2:3] != (width,):
                    raise RuntimeError(f'invalid diagnostic hit shape: {key}')
                name = key[4:]
                missing = np.nan if a.dtype.kind == 'f' else (-1 if '_bin' in name else 0)
                hits.setdefault(name, np.full((nh, 2)+a.shape[3:], missing, a.dtype))
                hits[name][idx[valid]] = np.moveaxis(a, 1, 2)[valid]
            else:
                rows.setdefault(key, []).append(a)
    payload = {k: v.copy() for k, v in candidate.items()}
    payload.update({'failure_'+k: np.concatenate(v) for k, v in rows.items()})
    payload.update({'source_failure_'+k: v for k, v in hits.items()})
    ok = candidate['candidate_objective_valid']
    replayed = []
    checks = [('objective_valid', 'objective_valid'), ('objective', 'final_objective'),
              ('raw_gradient_norm', 'gradient_norm'), ('raw_rank', 'rank'), ('raw_condition', 'condition'),
              ('differentiable', 'analytic_differentiable'), ('certified', 'success'), ('certified', 'certified')]
    checks.extend((k.removeprefix('failure_original_'), k.removeprefix('failure_original_'))
                  for k in payload if k.startswith('failure_original_certificate_'))
    for diagnostic, original in checks:
        a, b = payload['failure_original_'+diagnostic], candidate['candidate_'+original]
        if diagnostic.startswith('raw_'):
            a, b = a[ok], b[ok]
        if original == 'rank' and b.dtype.kind in 'iu':
            a = a.astype(b.dtype)
        if not _equal(a, b):
            raise ValueError(f'saved candidate replay differs: {original}')
        replayed.append('candidate_'+original)
    hit_ok = ok[data['hg'][selected]]
    for diagnostic in sorted(prior.HIT_FIELDS-{'mask', 'initial_lr', 'lr_preferred', 'lr_improvement'}):
        a = payload['source_failure_original_'+diagnostic][selected]
        b = candidate['source_candidate_'+diagnostic][selected]
        if not _equal(a[hit_ok], b[hit_ok]):
            raise ValueError(f'saved candidate hit replay differs: {diagnostic}')
        replayed.append('source_candidate_'+diagnostic)
    report = summarize(candidate, payload)
    report['exact_replayed_fields'] = sorted(replayed)
    report['replay_scope'] = 'Listed final-state fields; raw linear diagnostics and hit fields on objective-valid lanes. '
    report['replay_scope'] += 'All certificate fields include invalid lanes. Traces/status provenance is copied, not optimizer-replayed.'
    report.update(events=len(keys), groups=ng, hits=nh, fixed_selected_hits=len(selected),
        zero_hit_events=int(np.sum(np.bincount(data['sh']['ev'], minlength=len(keys)) == 0)),
        group_batch=group_batch, effective_group_batch=batch, fixed_full_width=width,
        nominal_axial_bz_T=float(bz), mass_hypothesis_MeV=float(mass),
        fixed_polishing_rounds=3, finite_difference_steps=[1e-3, 1e-4, 1e-5, 1e-6],
        saved_candidate_replayed_exactly=True, caveat=__doc__)
    return report, payload


def load_candidate(directory, saved_dir, baseline_paths):
    """Require the saved candidate's recorded baseline/input hashes before analysis."""
    report = json.loads((directory/'report.json').read_text())
    for name, path in (('report', saved_dir/'report.json'), ('payload', saved_dir/'diagnostics.npz')):
        if report['saved_baseline'][name]['sha256'] != _sha256(path):
            raise ValueError(f'candidate baseline hash differs: {name}')
    for name, path in baseline_paths.items():
        if name in prior.INPUTS and report['inputs'][name]['sha256'] != _sha256(path):
            raise ValueError(f'candidate input hash differs: {name}')
    expected = dict(condition_limit=1e12, gradient_tolerance=1e-6, initial_damping=1e-3,
        max_parameter_step=10., max_backtracks=8, finite_difference_steps=[1e-3, 1e-4, 1e-5, 1e-6],
        gradient_agreement_fraction=.25, jacobian_column_relative_tolerance=1e-3)
    if any(report['numerical_settings'][k] != v for k, v in expected.items()):
        raise ValueError('unsupported saved candidate numerical settings')
    return prior.read_npz(directory/'diagnostics.npz'), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--group-batch', default=32, type=int)
    parser.add_argument('--compare-saved', type=Path)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    started = time.perf_counter()
    inputs, original, paths = prior.load_inputs(args.saved)
    candidate, candidate_report = load_candidate(args.candidate, args.saved, paths)
    loaded = time.perf_counter()
    report, payload = run(*inputs, candidate, bz=original['nominal_axial_bz_T'],
                         mass=original['mass_hypothesis_MeV'], group_batch=args.group_batch)
    report.update(load_and_oracle_seconds=loaded-started, processing_seconds=time.perf_counter()-loaded,
        processing_scope='Read-only original and polished-root derivative probes; no optimizer or baseline rerun; output excluded.',
        historical_candidate_implementation=candidate_report['current_implementation'])
    paths = dict(paths, baseline_report=args.saved/'report.json', baseline_payload=args.saved/'diagnostics.npz',
                 candidate_report=args.candidate/'report.json', candidate_payload=args.candidate/'diagnostics.npz')
    report['inputs'] = {k: dict(path=str(p.resolve()), sha256=_sha256(p)) for k, p in paths.items()}
    from trkfastsim.reco import driftfailure, driftprobecompare, driftjacobian, stabledriftfit, driftfitdiagnostic, helixfit, helixtiming, fielddrift, helixdriftfit
    from . import run1a_timing, run1a_wirefit, run1a_drift
    modules = (driftfailure, driftprobecompare, driftjacobian, stabledriftfit, driftfitdiagnostic, helixfit, helixtiming, fielddrift,
               helixdriftfit, prior, stable, run1a_timing, run1a_wirefit, run1a_drift)
    sources = {m.__name__: Path(m.__file__) for m in modules}
    sources[__name__] = Path(__file__)
    report['current_implementation'] = {k: dict(path=str(p.resolve()), sha256=_sha256(p)) for k, p in sources.items()}
    if args.compare_saved:
        old = prior.read_npz(args.compare_saved)
        if set(old) != set(payload) or any(not _equal(a, old[k]) for k, a in payload.items()):
            raise ValueError('diagnostic repeat differs')
        report['saved_comparison'] = dict(exact=True, path=str(args.compare_saved.resolve()), sha256=_sha256(args.compare_saved))
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir/'diagnostics.npz', **payload)
    report['output_payload_sha256'] = _sha256(args.outdir/'diagnostics.npz')
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    (args.outdir/'report.json').write_text(text)
    print(text, end='')
    return report


if __name__ == '__main__':
    main()
