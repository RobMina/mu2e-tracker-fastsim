"""Opt-in measured LR continuation of saved analytic-candidate final states.

This is NOT an independent original-seed comparison: it adds bounded LR/refit
work to saved finals, including numerical failures. No support pruning, stock
membership, truth, direction winner, material or momentum output is introduced.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from trkfastsim.reco.stereoseed import _integer
from . import run1a_drift_diagnostic as prior
from . import run1a_stabledrift as stable
from . import run1a_drift_failure as failure
from .run1a_timing import prepare_saved, measurement_arrays
from .run1a_wirefit import _quantiles, _sha256


def replay(ev, parameters, candidate, groups, idx, valid):
    """Check recorded final state and original criterion before any LR update."""
    from trkfastsim.reco.driftfitdiagnostic import _empty_state
    from trkfastsim.reco.driftjacobian import linearize
    from trkfastsim.reco.stabledriftfit import certify
    state = ev(parameters) if ev.batch and ev.hits else _empty_state(ev)
    lin = linearize(ev, parameters, state)
    cert = certify(ev, parameters, state, lin)
    old_ok = candidate['candidate_objective_valid'][groups].reshape(-1)
    checks = dict(objective_valid=state['objective_valid'], final_objective=state['objective'],
                  analytic_differentiable=lin['differentiable'], success=cert['certified'], **cert)
    for key, a in checks.items():
        b = candidate['candidate_'+key][groups].reshape(a.shape)
        if not failure._equal(a, b):
            raise ValueError(f'candidate final replay differs: {key}')
    for key, field in (('gradient_norm', 'gradient_norm'), ('rank', 'rank'), ('condition', 'condition'),
                       ('g', 'equilibrated_gradient')):
        a = lin[key][old_ok]
        b = candidate['candidate_'+field][groups].reshape(lin[key].shape)[old_ok]
        if key == 'rank' and b.dtype.kind in 'iu':
            a = a.astype(b.dtype)
        if not failure._equal(a, b):
            raise ValueError(f'candidate linear replay differs: {key}')
    hit_ok = old_ok[:, None] & ev.mask
    from trkfastsim.reco.helixfit import _at_z
    with np.errstate(all='ignore'):
        root = _at_z(parameters[:, :5], ev.zref, state['poca_z'], ev.mid, ev.direction)
    hit_state = dict(state, **{'final_poca_'+k: root[k] for k in ('correction', 'g', 'h')})
    for key, value in hit_state.items():
        if value.ndim == 1:
            continue
        old = candidate['source_candidate_'+key]
        a = np.zeros(valid.shape+(2,)+old.shape[2:], old.dtype)
        a[valid] = old[idx[valid]]
        a = np.moveaxis(a, 2, 1).reshape(value.shape)
        if not failure._equal(a[hit_ok], value[hit_ok]):
            raise ValueError(f'candidate hit replay differs: {key}')
    preferred = np.sign(state['r_drift'])*np.sign(state['signed_doca'])
    flip = hit_ok & (preferred != 0) & (preferred != ev.lr)
    return state, dict(initial_lr_stationary=old_ok & ~np.any(flip, axis=1),
                       initial_pending_lr_flips=flip.sum(axis=1),
                       initial_combined_checked=cert['certified'] & ~np.any(flip, axis=1))


def summarize(candidate, payload):
    """Compare complete census under both continuous-only and combined criteria."""
    old_status, new_status = candidate['candidate_status'], payload['lr_status']
    valid = candidate['candidate_objective_valid'] & payload['lr_objective_valid']
    delta = payload['lr_final_objective']-candidate['candidate_final_objective']
    selected = candidate['source_selected']
    group = candidate['hit_group'][selected]
    hit_valid = valid[group]
    changed = payload['source_lr_final_lr'][selected] != payload['source_lr_initial_lr'][selected]
    original_checked = candidate['candidate_success']
    before_combined = payload['lr_initial_combined_checked']
    final_checked = payload['lr_success']
    statuses = np.unique(new_status)

    def successes(a):
        eligible = np.any(candidate['candidate_objective_valid'], axis=1)
        return dict(hypotheses=int(a.sum()), negative_z=int(a[:, 0].sum()), positive_z=int(a[:, 1].sum()),
            any_direction_groups=int(np.any(a, axis=1).sum()), both_direction_groups=int(np.all(a, axis=1).sum()),
            neither_eligible_groups=int(np.sum(eligible & ~np.any(a, axis=1))))

    out = dict(saved_candidate_status_counts=stable._counts(old_status), lr_status_counts=stable._counts(new_status),
        termination_status_counts=stable._counts(payload['lr_termination_status']),
        certificate_status_counts=stable._counts(payload['lr_certificate_status']),
        last_refit_status_counts=stable._counts(payload['lr_last_refit_status']),
        saved_continuous_checked=successes(original_checked), saved_combined_checked=successes(before_combined),
        final_continuous_checked=successes(payload['lr_certified']), final_combined_checked=successes(final_checked),
        combined_criterion='Unchanged analytic/adjacent-stencil criterion AND no pointwise beneficial LR flip '
            'AND no proposal/contract error or sticky refit numerical failure; '
            'not global optimality, calibrated ambiguity probability, direction identification or tracking efficiency.',
        status_transition_table={str(label): {str(target): int(np.sum((old_status == label) & (new_status == target)))
            for target in statuses} for label in np.unique(old_status)},
        by_saved_status={}, accepted_lr_flip_operations=int(payload['lr_accepted_lr_flips'].sum()),
        accepted_lr_rounds=int(payload['lr_accepted_lr_rounds'].sum()),
        net_changed_hit_hypothesis_entries=int(np.sum(changed & hit_valid)),
        net_changed_distinct_source_hits=int(np.sum(np.any(changed & hit_valid, axis=1))),
        continuous_refits=int(payload['lr_continuous_refits'].sum()),
        refit_failure_count=int(payload['lr_refit_failure_count'].sum()),
        original_arrays_preserved=True, assigned_directions=0,
        comparison_scope='Explicit continuation from every saved candidate final, including numerical failures, '
            'not original seeds. Additional LR/continuous budgets are not equal compute to the fixed-LR baseline.',
        input_provenance_caveat='Current hashes/replay cannot restore original association input hashes, '
            'baseline output-NPZ binding or historical candidate payload authentication.')
    final_lr_stationary = payload['lr_lr_stationary']
    out['final_lr_stationary_hypotheses'] = int(final_lr_stationary.sum())
    out['initial_pending_lr_hypotheses'] = int(np.sum(payload['lr_initial_pending_lr_flips'] > 0))
    out['final_pending_lr_hypotheses'] = int(np.sum(payload['lr_pending_lr_flips'] > 0))
    out['final_pending_lr_hit_hypothesis_entries'] = int(payload['lr_pending_lr_flips'].sum())
    for label in np.unique(old_status):
        rows = old_status == label
        use = rows & valid
        huse = rows[group] & hit_valid
        out['by_saved_status'][str(label)] = dict(hypotheses=int(rows.sum()), comparable_valid=int(use.sum()),
            combined_checked_before=int(np.sum(rows & before_combined)), combined_checked_after=int(np.sum(rows & final_checked)),
            continuous_checked_after=int(np.sum(rows & payload['lr_certified'])),
            initially_pending_lr=int(np.sum(rows & (payload['lr_initial_pending_lr_flips'] > 0))),
            final_pending_lr=int(np.sum(rows & (payload['lr_pending_lr_flips'] > 0))),
            accepted_lr_flip_operations=int(payload['lr_accepted_lr_flips'][rows].sum()),
            net_changed_hit_hypothesis_entries=int(np.sum(changed & huse)),
            lower_objective=int(np.sum(use & (delta < 0))), equal_objective=int(np.sum(use & (delta == 0))),
            higher_objective=int(np.sum(use & (delta > 0))), final_minus_saved_objective=_quantiles(delta[use]),
            saved_abs_drift_residual_mm=_quantiles(np.abs(candidate['source_candidate_drift_residual'][selected][huse])),
            final_abs_drift_residual_mm=_quantiles(np.abs(payload['source_lr_drift_residual'][selected][huse])))
    hit_changes = np.zeros(old_status.shape, bool)
    np.logical_or.at(hit_changes, group, changed & hit_valid)
    out['changed_lr_hypotheses'] = int(hit_changes.sum())
    out['objective_comparison_all_valid'] = dict(hypotheses=int(valid.sum()),
        lower=int(np.sum(valid & (delta < 0))), equal=int(np.sum(valid & (delta == 0))),
        higher=int(np.sum(valid & (delta > 0))), final_minus_saved=_quantiles(delta[valid]))
    out['same_hit_residuals_all_valid'] = dict(hit_hypothesis_entries=int(hit_valid.sum()),
        saved_abs_drift_mm=_quantiles(np.abs(candidate['source_candidate_drift_residual'][selected][hit_valid])),
        final_abs_drift_mm=_quantiles(np.abs(payload['source_lr_drift_residual'][selected][hit_valid])),
        scope='Identical valid hit-hypothesis identities including numerical failures; objective reduction '
            'is not an independent resolution validation.')
    for key in ('continuous_iterations', 'continuous_accepted_steps', 'fit_trial_evaluations', 'certificate_evaluations'):
        out[key] = int(payload['lr_'+key].sum())
    out['work_counter_scope'] = 'Logical row work; certificate_evaluations counts final checks, not inner-refit checks or replay. '
    out['work_counter_scope'] += 'Extra LR/refit work is not equal compute or end-to-end throughput.'
    for key in ('active', 'flip_accepted', 'refit_attempted', 'refit_accepted', 'invalid', 'roundoff', 'refit_error'):
        out[key+'_rows_by_round'] = np.sum(payload['lr_trace_'+key], axis=(0, 1)).tolist()
    accepted = payload['lr_trace_flip_accepted']
    lr_gain = payload['lr_trace_before_objective']-payload['lr_trace_postflip_objective']
    refit_gain = payload['lr_trace_postflip_objective']-payload['lr_trace_postrefit_objective']
    out['accepted_lr_objective_gain'] = _quantiles(lr_gain[accepted])
    out['accepted_refit_objective_gain'] = _quantiles(refit_gain[payload['lr_trace_refit_accepted']])
    out['total_accepted_lr_objective_gain'] = float(lr_gain[accepted].sum())
    out['total_accepted_refit_objective_gain'] = float(refit_gain[payload['lr_trace_refit_accepted']].sum())
    out['continuous_refit_raw_terminations'] = stable._counts(payload['lr_trace_refit_termination_status'][accepted])
    out['continuous_refit_final_statuses'] = stable._counts(payload['lr_trace_refit_status'][accepted])
    out['final_failure_flags'] = dict(
        contract_or_proposal_error=int(payload['lr_refit_error'].sum()),
        sticky_numerical_failure=int(payload['lr_refit_numerical_failure'].sum()),
        sticky_failure_but_continuous_checked=int(np.sum(payload['lr_refit_numerical_failure'] & payload['lr_certified'])),
        pending_lr_but_continuous_checked=int(np.sum((payload['lr_pending_lr_flips'] > 0) & payload['lr_certified'])),
        pending_lr_and_sticky_failure=int(np.sum((payload['lr_pending_lr_flips'] > 0) & payload['lr_refit_numerical_failure'])))
    out['combined_pass_transition'] = dict(retained=int(np.sum(before_combined & final_checked)),
        gained=int(np.sum(~before_combined & final_checked)), lost=int(np.sum(before_combined & ~final_checked)))
    return out


def run(keys, sh, cond, association, wirefit, geom, response, model, saved, candidate, *,
        bz, mass=105.6583755, group_batch=32, max_rounds=3, max_iterations=40):
    """Fresh bounded continuation; no baseline fit/search or diagnostic-polishing rerun."""
    from trkfastsim.reco.driftfitdiagnostic import _Evaluator
    from trkfastsim.reco.lrdriftfit import fit_lr_drift_helix
    group_batch = _integer('group_batch', group_batch, 1)
    max_rounds = _integer('max_rounds', max_rounds, 0)
    max_iterations = _integer('max_iterations', max_iterations, 0)
    if max_rounds > 8 or max_iterations > 40:
        raise ValueError('LR round/refit budget bound exceeded')
    if any(k.startswith(('lr_', 'source_lr_')) for k in candidate):
        raise ValueError('input must be fixed-LR candidate, not an LR continuation')
    data = prepare_saved(keys, sh, cond, association, wirefit)
    prior.validate_saved(data, wirefit, saved)
    failure.validate_candidate(saved, candidate)
    original_seeds = stable.original_seeds(wirefit, saved)
    for name, expected in original_seeds.items():
        if not failure._equal(candidate['candidate_initial_'+name], expected):
            raise ValueError(f'candidate original seed differs: {name}')
    if not failure._equal(candidate['candidate_initial_objective'], saved['drift_initial_objective']):
        raise ValueError('candidate original objective differs')
    measurements = measurement_arrays(data, geom, response)
    ng, nh = len(data['gev']), len(data['hg'])
    count, selected = data['count'], data['selected']
    width = int(count.max(initial=0))
    if width > 256:
        raise ValueError('LR support bound exceeded')
    batch = min(group_batch, 512, max(1, 65536//(2*max(1, width))))
    starts = np.r_[0, np.cumsum(count)]
    rows, hits = {}, {}
    replay_seconds = continuation_seconds = 0.
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
        kwargs = dict(local_parameters=candidate['candidate_local_parameters'][groups].reshape(-1, 5),
            z_ref=saved['drift_z_ref'][groups].reshape(-1), t0=candidate['candidate_t0'][groups].reshape(-1),
            flight_direction=saved['drift_flight_direction'][groups].reshape(-1), **padded,
            measured_u=np.repeat(u, 2, axis=0), uvar=np.repeat(variance, 2, axis=0),
            valid=np.repeat(valid, 2, axis=0), model=model, bz=bz, mass=mass, initial_lr=lr,
            parameter_scale=saved['drift_parameter_scale'][groups].reshape(-1, 6),
            seed_valid=saved['profile_profile_valid'][groups].reshape(-1))
        started = time.perf_counter()
        ev = _Evaluator(**kwargs)
        state, initial = replay(ev, ev.parameters, candidate, groups, idx, valid)
        replay_seconds += time.perf_counter()-started
        started = time.perf_counter()
        result = fit_lr_drift_helix(**kwargs, max_rounds=max_rounds, max_iterations=max_iterations)
        continuation_seconds += time.perf_counter()-started
        if not failure._equal(result['initial_objective'], state['objective']):
            raise ValueError('LR stage changed saved starting objective')
        result.update(initial)
        for key, value in result.items():
            a = np.asarray(value)
            if a.shape[:1] != (2*len(groups),):
                raise RuntimeError(f'invalid LR row shape: {key}')
            if key.startswith('hit_trace_'):
                a = np.moveaxis(a, 2, 1)
            a = a.reshape((len(groups), 2)+a.shape[1:])
            if key.startswith('hit_'):
                if a.shape[2:3] != (width,):
                    raise RuntimeError(f'invalid LR hit shape: {key}')
                name = key[4:]
                missing = np.nan if a.dtype.kind == 'f' else (-1 if '_bin' in name else 0)
                hits.setdefault(name, np.full((nh, 2)+a.shape[3:], missing, a.dtype))
                hits[name][idx[valid]] = np.moveaxis(a, 1, 2)[valid]
            else:
                rows.setdefault(key, []).append(a)
    payload = {k: v.copy() for k, v in candidate.items()}
    payload.update({'lr_'+k: np.concatenate(v) for k, v in rows.items()})
    payload.update({'source_lr_'+k: v for k, v in hits.items()})
    if not failure._equal(payload['lr_initial_objective_valid'], candidate['candidate_objective_valid']):
        raise ValueError('LR initial objective validity changed')
    if not failure._equal(payload['source_lr_mask'], candidate['source_candidate_mask']):
        raise ValueError('LR stage changed measured support')
    if not failure._equal(payload['source_lr_initial_lr'][selected], candidate['source_candidate_initial_lr'][selected]):
        raise ValueError('LR stage changed initial assignments')
    use = payload['lr_objective_valid'] & candidate['candidate_objective_valid']
    if np.any(payload['lr_final_objective'][use] > candidate['candidate_final_objective'][use]):
        raise ValueError('LR continuation increased starting objective')
    report = summarize(candidate, payload)
    report.update(events=len(keys), groups=ng, hits=nh, fixed_selected_hits=len(selected),
        zero_hit_events=int(np.sum(np.bincount(data['sh']['ev'], minlength=len(keys)) == 0)),
        group_batch=group_batch, effective_group_batch=batch, fixed_full_width=width,
        nominal_axial_bz_T=float(bz), mass_hypothesis_MeV=float(mass), max_rounds=max_rounds,
        continuous_max_iterations=max_iterations, continuous_max_backtracks=8,
        saved_final_replay_seconds=replay_seconds, lr_continuation_seconds=continuation_seconds,
        replay_scope='Full saved final evaluator state on objective-valid selected hits, raw linear diagnostics, '
            'all certificate fields including invalid lanes; optimizer history copied, not rerun.',
        numerical_policy='Unchanged evaluator/POCA/analytic derivatives/certificate thresholds; simultaneous '
            'pointwise LR proposals, strict actual-objective acceptance, unchanged bounded continuous refits.',
        caveat=__doc__)
    return report, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--group-batch', default=32, type=int)
    parser.add_argument('--max-rounds', default=3, type=int)
    parser.add_argument('--max-iterations', default=40, type=int)
    parser.add_argument('--compare-saved', type=Path)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    started = time.perf_counter()
    inputs, original, paths = prior.load_inputs(args.saved)
    candidate, candidate_report = failure.load_candidate(args.candidate, args.saved, paths)
    loaded = time.perf_counter()
    report, payload = run(*inputs, candidate, bz=original['nominal_axial_bz_T'], mass=original['mass_hypothesis_MeV'],
        group_batch=args.group_batch, max_rounds=args.max_rounds, max_iterations=args.max_iterations)
    report.update(load_and_oracle_seconds=loaded-started, processing_seconds=time.perf_counter()-loaded,
        historical_candidate_implementation=candidate_report['current_implementation'],
        processing_scope='Saved-final replay plus explicit LR continuation; no baseline, association or wire-fit rerun; output excluded.')
    paths = dict(paths, baseline_report=args.saved/'report.json', baseline_payload=args.saved/'diagnostics.npz',
                 candidate_report=args.candidate/'report.json', candidate_payload=args.candidate/'diagnostics.npz')
    report['inputs'] = {k: dict(path=str(p.resolve()), sha256=_sha256(p)) for k, p in paths.items()}
    from trkfastsim.reco import lrdriftfit, driftjacobian, stabledriftfit, driftfitdiagnostic, helixfit, helixtiming, fielddrift, helixdriftfit
    from . import run1a_timing, run1a_wirefit, run1a_drift
    modules = (lrdriftfit, driftjacobian, stabledriftfit, driftfitdiagnostic, helixfit, helixtiming, fielddrift,
               helixdriftfit, prior, stable, failure, run1a_timing, run1a_wirefit, run1a_drift)
    sources = {m.__name__: Path(m.__file__) for m in modules}
    sources[__name__] = Path(__file__)
    report['current_implementation'] = {k: dict(path=str(p.resolve()), sha256=_sha256(p)) for k, p in sources.items()}
    if args.compare_saved:
        old = prior.read_npz(args.compare_saved)
        if set(old) != set(payload) or any(not failure._equal(a, old[k]) for k, a in payload.items()):
            raise ValueError('LR continuation repeat differs')
        report['saved_comparison'] = dict(exact=True, path=str(args.compare_saved.resolve()), sha256=_sha256(args.compare_saved))
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
