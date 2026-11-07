"""Bounded measured-only LR continuation from supplied analytic-candidate finals.

All objective-valid rows are eligible, not only previously successful rows. This
is not a same-original-seed comparison or an equal-compute speedup. Support,
scales, direction, POCA, calibration and the current-branch certificate stay
unchanged; no momentum, covariance, material or smoothing is introduced.
"""
import numpy as np

from .driftfitdiagnostic import _Evaluator, _empty_state, _center_output, FD_STEPS
from .driftjacobian import linearize
from .helixfit import _at_z
from .helixtiming import _integer, _scalar
from .stabledriftfit import certify, fit_stable_drift_helix


def _full_valid(evaluator, state):
    return (state['objective_valid'] & np.isfinite(state['objective'])
            & np.all(~evaluator.mask | state['valid'], axis=1))


def _preferred(state, lr, mask):
    """Sign factors avoid overflow; exact zero in either factor retains LR."""
    r, d, s = (state[k] for k in ('r_drift', 'signed_doca', 'signed_error'))
    use = mask & state['objective_valid'][:, None]
    sign = np.sign(r)*np.sign(d)
    preferred = np.where(use & (sign != 0), sign, lr)
    with np.errstate(all='ignore'):
        improvement = -2*((lr*r/s)*(d/s))
    improvement = np.where((r == 0) | (d == 0), 0., improvement)
    gain = np.where(use, np.maximum(improvement, 0.), np.nan)
    return preferred, gain


def fit_lr_drift_helix(local_parameters, z_ref, t0, flight_direction, wire_cal,
                       wire_hv, early_end, early_time, signal_speed, measured_u,
                       uvar, valid, model, *, bz, initial_lr, parameter_scale,
                       seed_valid, mass=105.6583755, max_rows=1024, max_cells=65536,
                       max_iterations=40, condition_limit=1e12, gradient_tolerance=1e-6,
                       initial_damping=1e-3, max_parameter_step=10., max_backtracks=8,
                       max_rounds=3):
    """Continue saved finals with simultaneous separable LR flips then stable refits.

    Inputs match fit_stable_drift_helix, plus max_rounds (0..8, default3).
    The caller supplies both directions and original seed_valid, NOT fit success.
    Iterations are bounded0..40 and backtracks1..8. Zero rounds only checks the
    supplied state, and can pass if already LR-stable and fully certified.

    Each proposed sign change must strictly reduce the finite full objective
    .5*sum(((LR*r-signed_doca)/sigma)**2 + long_pull**2). Negative radii are
    never absolutized. Exact sign ties keep LR. A new evaluator validates each
    proposal; a predicted gain alone cannot authorize an update. Only accepted
    LR-changed rows call the unchanged stable fitter, with original support and
    scales. Its last finite state is independently replayed; a decrease is kept,
    identical parameters/objective retain the flip, and all other outcomes are
    explicit failures. Validated improvements survive numerical refit failures.

    Final success requires LR stationarity AND the unchanged analytic/adjacent
    stencil certificate, without a prior numerical or contract failure. Lack of
    certification does not prevent a later LR round. Invalid/increasing/contract
    failures stop the affected row. Arrays all have first dimension B; per-hit
    arrays have hit_ prefixes (including hit_trace_*). Per-round accepted-state
    traces use [B,R,...]; hit traces use [B,R,H]. Inner trial histories are not
    retained across rounds: storage O(B*H*R+B*R*40), bounded fixed loops only.
    """
    max_rounds = _integer('max_rounds', max_rounds)
    max_iterations = _integer('max_iterations', max_iterations)
    max_backtracks = _integer('max_backtracks', max_backtracks, 1)
    max_rows, max_cells = _integer('max_rows', max_rows), _integer('max_cells', max_cells)
    if max_rounds > 8 or max_iterations > 40 or max_backtracks > 8:
        raise ValueError('LR continuation round/iteration/backtrack bound exceeded')
    if max_rows > 1024 or max_cells > 65536:
        raise ValueError('LR continuation hard row/cell bound exceeded')
    condition_limit = _scalar('condition_limit', condition_limit, True)
    gradient_tolerance = _scalar('gradient_tolerance', gradient_tolerance, True)
    initial_damping = _scalar('initial_damping', initial_damping, True)
    max_parameter_step = _scalar('max_parameter_step', max_parameter_step, True)
    if condition_limit != 1e12 or gradient_tolerance != 1e-6:
        raise ValueError('LR continuation uses unchanged certificate thresholds')
    if initial_damping > 1e12:
        raise ValueError('initial_damping must not exceed1e12')
    rows = dict(local_parameters=local_parameters, z_ref=z_ref, t0=t0,
                flight_direction=flight_direction, wire_cal=wire_cal, wire_hv=wire_hv,
                early_end=early_end, early_time=early_time, signal_speed=signal_speed,
                measured_u=measured_u, uvar=uvar, valid=valid, initial_lr=initial_lr,
                parameter_scale=parameter_scale, seed_valid=seed_valid)
    common = dict(model=model, bz=bz, mass=mass, max_rows=max_rows, max_cells=max_cells)
    evaluator = _Evaluator(**rows, **common)
    rows = {k: np.asarray(v).copy() for k, v in rows.items()}
    b, h = evaluator.batch, evaluator.hits
    parameters, lr = evaluator.candidate.copy(), evaluator.initial_lr.copy()
    state = evaluator(parameters) if b and h else _empty_state(evaluator)
    initial, initial_valid = state['objective'].copy(), _full_valid(evaluator, state)
    active = initial_valid.copy()
    termination = np.full(b, 'max_rounds', dtype='<U40')
    termination[~initial_valid] = 'invalid_initial'
    termination[~rows['seed_valid']] = 'invalid_seed'
    termination[~rows['valid'].any(axis=1)] = 'empty'
    counters = {k: np.zeros(b, np.int32) for k in
                ('count_rounds', 'accepted_lr_rounds', 'accepted_lr_flips',
                 'continuous_refits', 'refit_failure_count', 'continuous_iterations',
                 'continuous_accepted_steps', 'fit_trial_evaluations')}
    errors, numerical_failure = np.zeros(b, bool), np.zeros(b, bool)
    last_status = np.full(b, 'not_attempted', dtype='<U40')
    last_termination = last_status.copy()
    shape = (b, max_rounds)
    trace = {k: np.zeros(shape, bool) for k in
             ('active', 'flip_accepted', 'refit_attempted', 'refit_accepted',
              'budget', 'lr_budget', 'invalid', 'roundoff', 'refit_error', 'refit_numerical_failure',
              'refit_certified', 'proposal_valid')}
    trace.update({k: np.full(shape, np.nan) for k in
                  ('before_objective', 'postflip_objective', 'postrefit_objective',
                   'predicted_lr_gain', 'proposed_objective', 'refit_actual_objective',
                   'refit_reported_objective', 't0_before', 't0_after')})
    trace.update({k: np.zeros(shape, np.int32) for k in
                  ('attempted_flip_count', 'accepted_flip_count', 'refit_iterations', 'refit_accepted_steps')})
    trace.update({k: np.full(shape, 'not_attempted', dtype='<U40') for k in
                  ('refit_status', 'refit_termination_status', 'certificate_status')})
    for key in ('parameters_before', 'parameters_after'):
        trace[key] = np.full(shape+(5,), np.nan)
    hit_trace = {k: np.zeros(shape+(h,), bool) for k in ('attempted_flip', 'accepted_flip')}
    for key in ('lr_before', 'lr_after'):
        hit_trace[key] = np.full(shape+(h,), np.nan)
    with np.errstate(all='ignore'):
        for round_index in range(max_rounds):
            if not active.any():
                break
            j, running = round_index, active.copy()
            counters['count_rounds'] += running
            trace['active'][:, j] = running
            trace['parameters_before'][:, j] = np.where(running[:, None], parameters[:, :5], np.nan)
            trace['t0_before'][:, j] = np.where(running, parameters[:, 5], np.nan)
            trace['before_objective'][:, j] = np.where(running, state['objective'], np.nan)
            hit_trace['lr_before'][:, j] = np.where(running[:, None], lr, np.nan)
            preferred, gain = _preferred(state, lr, evaluator.mask)
            flips = running[:, None] & evaluator.mask & (preferred != lr)
            attempted = flips.any(axis=1)
            predicted = np.sum(np.where(flips, gain, 0.), axis=1)
            trace['predicted_lr_gain'][:, j] = np.where(running, predicted, np.nan)
            trace['attempted_flip_count'][:, j] = flips.sum(axis=1)
            hit_trace['attempted_flip'][:, j] = flips
            termination[running & ~attempted] = 'lr_stationary'
            active &= attempted
            accepted = np.zeros(b, bool)
            if attempted.any():
                proposed_lr = np.where(flips, preferred, lr)
                proposal_ev = _Evaluator(**dict(rows, initial_lr=proposed_lr), **common)
                proposal = proposal_ev(parameters)
                valid_proposal = _full_valid(proposal_ev, proposal)
                trace['proposal_valid'][:, j] = attempted & valid_proposal
                trace['proposed_objective'][:, j] = np.where(attempted, proposal['objective'], np.nan)
                accepted = (attempted & _full_valid(evaluator, state) & valid_proposal
                            & (proposal['objective'] < state['objective']))
                rejected = attempted & ~accepted
                trace['invalid'][:, j] = attempted & ~valid_proposal
                trace['roundoff'][:, j] = rejected & valid_proposal
                termination[rejected] = 'lr_no_strict_decrease'
                termination[trace['invalid'][:, j]] = 'lr_invalid_proposal'
                errors |= rejected
                active &= ~rejected
                lr = np.where(accepted[:, None], proposed_lr, lr)
                evaluator = _Evaluator(**dict(rows, initial_lr=lr), **common)
                state = evaluator(parameters)
            trace['flip_accepted'][:, j] = accepted
            hit_trace['accepted_flip'][:, j] = flips & accepted[:, None]
            counts = hit_trace['accepted_flip'][:, j].sum(axis=1)
            trace['accepted_flip_count'][:, j] = counts
            counters['accepted_lr_rounds'] += accepted
            counters['accepted_lr_flips'] += counts.astype(np.int32)
            trace['postflip_objective'][:, j] = np.where(running, state['objective'], np.nan)
            if accepted.any():
                selected = np.flatnonzero(accepted)
                subset = {k: v[selected] for k, v in rows.items()}
                subset.update(local_parameters=parameters[selected, :5].copy(),
                              t0=parameters[selected, 5].copy(), initial_lr=lr[selected].copy())
                fitted = fit_stable_drift_helix(**subset, **common, max_iterations=max_iterations,
                            max_backtracks=max_backtracks, condition_limit=condition_limit,
                            gradient_tolerance=gradient_tolerance, initial_damping=initial_damping,
                            max_parameter_step=max_parameter_step)
                counters['continuous_refits'] += accepted
                trace['refit_attempted'][:, j] = accepted
                last_status[selected], last_termination[selected] = fitted['status'], fitted['termination_status']
                trace['refit_status'][selected, j] = fitted['status']
                trace['refit_termination_status'][selected, j] = fitted['termination_status']
                trace['certificate_status'][selected, j] = fitted['certificate_status']
                trace['refit_certified'][selected, j] = fitted['certified']
                trace['refit_reported_objective'][selected, j] = fitted['final_objective']
                for dest, src in (('continuous_iterations', 'iterations'),
                                  ('continuous_accepted_steps', 'accepted_steps'),
                                  ('fit_trial_evaluations', 'fit_trial_evaluations')):
                    counters[dest][selected] += fitted[src]
                trace['refit_iterations'][selected, j] = fitted['iterations']
                trace['refit_accepted_steps'][selected, j] = fitted['accepted_steps']
                trial_parameters = parameters.copy()
                result_parameters = np.column_stack((fitted['local_parameters'], fitted['t0']))
                finite = np.isfinite(result_parameters).all(axis=1)
                trial_parameters[selected] = np.where(finite[:, None], result_parameters, parameters[selected])
                replay = evaluator(trial_parameters)
                actual, postflip = replay['objective'][selected], state['objective'][selected]
                valid_refit = finite & _full_valid(evaluator, replay)[selected] & fitted['objective_valid']
                contract = actual == fitted['final_objective']
                contract &= fitted['initial_objective'] == postflip
                for name, expected in (('mask', subset['valid']), ('initial_lr', subset['initial_lr']),
                                       ('parameter_scale', subset['parameter_scale']),
                                       ('flight_direction', subset['flight_direction']), ('z_ref', subset['z_ref']),
                                       ('initial_local_parameters', subset['local_parameters']), ('initial_t0', subset['t0'])):
                    equal = (fitted[name] == expected) | (np.isnan(fitted[name]) & np.isnan(expected))
                    contract &= equal if equal.ndim == 1 else np.all(equal, axis=tuple(range(1, equal.ndim)))
                identical = np.all(result_parameters == parameters[selected], axis=1) & (actual == postflip)
                take = valid_refit & contract & (actual < postflip)
                bad = ~(valid_refit & contract & ((actual < postflip) | identical))
                numerical = np.isin(fitted['termination_status'],
                                   ['linearization_failed', 'ill_conditioned', 'stalled',
                                    'invalid_initial', 'invalid_seed', 'empty'])
                counters['refit_failure_count'][selected] += (bad | numerical | ~fitted['success']).astype(np.int32)
                numerical_failure[selected] |= numerical
                errors[selected] |= bad
                active[selected] &= ~bad
                termination[selected[bad]] = 'refit_no_strict_decrease'
                termination[selected[~contract]] = 'refit_contract_failure'
                termination[selected[~valid_refit]] = 'refit_invalid'
                trace['refit_error'][selected, j] = bad
                trace['refit_numerical_failure'][selected, j] = numerical
                trace['invalid'][selected, j] |= ~valid_refit
                trace['roundoff'][selected, j] |= valid_refit & contract & ~take & ~identical
                trace['refit_actual_objective'][selected, j] = np.where(valid_refit, actual, np.nan)
                trace['refit_accepted'][selected, j] = take
                trace['budget'][selected, j] = fitted['termination_status'] == 'max_iterations'
                parameters[selected[take]] = result_parameters[take]
                state = evaluator(parameters)
                del fitted
            trace['parameters_after'][:, j] = np.where(running[:, None], parameters[:, :5], np.nan)
            trace['t0_after'][:, j] = np.where(running, parameters[:, 5], np.nan)
            trace['postrefit_objective'][:, j] = np.where(running, state['objective'], np.nan)
            hit_trace['lr_after'][:, j] = np.where(running[:, None], lr, np.nan)
        final_valid = _full_valid(evaluator, state)
        preferred, gain = _preferred(state, lr, evaluator.mask)
        pending = np.sum(evaluator.mask & final_valid[:, None] & (preferred != lr), axis=1).astype(np.int32)
        lr_stationary = final_valid & (pending == 0)
        if max_rounds:
            trace['lr_budget'][:, -1] = trace['active'][:, -1] & (pending > 0) & ~errors
        termination[lr_stationary & ~errors] = 'lr_stationary'
        center = _center_output(evaluator, state)
        hit_keys = {'mask', 'initial_lr', 'residuals'} | {k for k, v in state.items() if v.ndim == 2}
        out = {('hit_'+k if k in hit_keys else k): v for k, v in center.items()}
        out.update(local_parameters=np.where(initial_valid[:, None], parameters[:, :5], rows['local_parameters']),
                   t0=np.where(initial_valid, parameters[:, 5], rows['t0']),
                   initial_local_parameters=rows['local_parameters'].copy(), initial_t0=rows['t0'].copy(),
                   initial_objective=initial, initial_objective_valid=initial_valid,
                   objective_valid=final_valid, final_objective=state['objective'].copy(),
                   hit_initial_lr=rows['initial_lr'].copy(), hit_final_lr=lr.copy(),
                   hit_lr_changed=initial_valid[:, None] & evaluator.mask & (lr != rows['initial_lr']),
                   hit_preferred_lr=preferred, hit_lr_gain=gain,
                   lr_stationary=lr_stationary, pending_lr_flips=pending,
                   last_refit_status=last_status, last_refit_termination_status=last_termination,
                   refit_error=errors, refit_numerical_failure=numerical_failure,
                   termination_status=termination, **counters)
        lin = dict(good=np.zeros(b, bool), differentiable=np.zeros(b, bool))
        out.update(analytic_stationary=np.zeros(b, bool), analytic_differentiable=np.zeros(b, bool),
                   analytic_numeric=np.zeros(b, bool), analytic_good=np.zeros(b, bool),
                   rank=np.zeros(b, np.int32), condition=np.full(b, np.nan),
                   gradient_norm=np.full(b, np.nan), equilibrated_gradient=np.full((b, 6), np.nan),
                   column_norm=np.full((b, 6), np.nan))
        status = termination.copy()
        if b and h and final_valid.any():
            lin = linearize(evaluator, parameters, state, condition_limit=condition_limit)
            stationary = final_valid & lin['good'] & lin['differentiable'] & (
                lin['gradient_norm'] <= gradient_tolerance*(1+np.sqrt(2*state['objective'])))
            out['analytic_stationary'] = stationary
            for dest, src in (('analytic_differentiable', 'differentiable'),
                              ('analytic_numeric', 'numeric'), ('analytic_good', 'good')):
                out[dest] = final_valid & lin[src]
            for key in ('rank', 'condition', 'gradient_norm'):
                out[key] = np.where(final_valid, lin[key], out[key])
            out['equilibrated_gradient'] = np.where(final_valid[:, None], lin['g'], np.nan)
            out['column_norm'] = np.where(final_valid[:, None], lin['norm'], np.nan)
            status[final_valid & ~stationary] = 'analytic_nonstationary'
            status[final_valid & ~lin['differentiable']] = 'nondifferentiable'
            status[final_valid & ~lin['good']] = 'ill_conditioned'
            status[final_valid & ~lin['numeric']] = 'linearization_failed'
        cert = certify(evaluator, parameters, state, lin, condition_limit=condition_limit,
                       gradient_tolerance=gradient_tolerance)
        out.update(cert)
        status[out['analytic_stationary'] & ~cert['certified']] = 'derivative_unstable'
        status[final_valid & (pending > 0)] = 'pending_lr_flips'
        status[numerical_failure] = 'refit_numerical_failure'
        status[errors] = termination[errors]
        out['success'] = lr_stationary & out['analytic_stationary'] & cert['certified'] & ~errors & ~numerical_failure
        status[out['success']] = 'converged_certified'
        out['status'] = status
        if b and h:
            root = _at_z(parameters[:, :5], evaluator.zref, state['poca_z'], evaluator.mid, evaluator.direction)
        for name in ('correction', 'g', 'h'):
            out['hit_final_poca_'+name] = (np.where(state['valid'] & final_valid[:, None], root[name], np.nan)
                                          if b and h else np.full((b, h), np.nan))
    out['certificate_evaluations'] = np.where(final_valid & out['analytic_differentiable']
                                             & out['analytic_good'], 12*len(FD_STEPS), 0)
    out['lr_budget_exhausted'] = final_valid & (pending > 0) & ~errors
    out.update({'trace_'+k: v for k, v in trace.items()})
    out.update({'hit_trace_'+k: v for k, v in hit_trace.items()})
    return out
