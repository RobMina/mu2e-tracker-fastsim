"""Opt-in analytic-derivative candidate for the fixed-LR conditional drift fit.

The baseline fitter is unchanged. Exact stock response values, local POCA policy,
measured hit support, both caller-supplied directions and initial LR are retained.
Full-rank analytic stationarity alone is insufficient: final success also requires
two adjacent same-branch numerical stencils with matching normalized gradients.
No covariance, ambiguity update, material or physics-performance claim.
"""
import numpy as np

from .driftfitdiagnostic import _Evaluator, _empty_state, _center_output, FD_STEPS
from .helixdriftfit import _linearize
from .helixfit import _at_z, _trajectory
from .helixtiming import _integer, _scalar

_CERT_FRACTION = .25
_CERT_JAC_RTOL = 1e-3
_EPS = np.finfo(float).eps


def _same_branch(center, trial, mask, model):
    """Conservative native-cell test; identical local-regression coefficients match."""
    same = np.ones(mask.shape, bool)
    for key in ('time_bin', 'phi_bin_lower', 'phi_bin_upper'):
        same &= center[key] == trial[key]
    same &= np.sign(center['signed_doca']) == np.sign(trial['signed_doca'])
    for key, calibration in (('offset_bin', model.offset), ('signed_error_bin', model.signed)):
        a, b = np.maximum(center[key], 0), np.maximum(trial[key], 0)
        same &= ((calibration.intercept[a] == calibration.intercept[b])
                 & (calibration.slope[a] == calibration.slope[b]))
    return np.all(~mask | same, axis=1)


def _angle_branch(evaluator, parameters, state):
    """Unfolded geometric signs are needed in addition to folded phi cell IDs."""
    point, _, _ = _trajectory(parameters[:, :5], evaluator.zref, state['poca_z'])
    s = np.where((evaluator.end == 0)[..., None], evaluator.direction, -evaluator.direction)
    late = np.where((evaluator.end == 0)[..., None], evaluator.hv, evaluator.cal)
    sensor = late+np.sum((point-late)*s, axis=-1)[..., None]*s
    q = point-sensor
    b = np.array([0., 0., np.sign(evaluator.bz)])
    return np.stack((np.sign(np.sum(q*np.cross(s, b), axis=-1)),
                     np.sign(np.sum(q*b, axis=-1))), axis=-1)


def certify(evaluator, parameters, state, analytic, *, condition_limit=1e12, gradient_tolerance=1e-6):
    """A bounded local derivative check, never a global-minimum certificate.

    All normalized gradients use the analytic Jacobian column norms, so a
    finite-difference norm change cannot conceal disagreement. Numerical normal
    matrices must independently pass rank/condition checks. Both adjacent FD
    stencils must remain in the center's native interpolation cells and identical
    calibration regressions, pass the same stationarity threshold, and disagree
    with the analytic gradient by at most one quarter of that threshold. This
    quarter is a declared engineering consistency policy, not an uncertainty.
    Each FD Jacobian column must also agree within relative L2 error1e-3;
    gradient agreement alone can hide errors orthogonal to the residual. These
    checks do not establish a local minimum or an independent exact derivative.
    """
    batch = len(parameters)
    result = dict(certified=np.zeros(batch, bool),
                  certificate_same_branch=np.zeros((batch, len(FD_STEPS)), bool),
                  certificate_numeric_good=np.zeros((batch, len(FD_STEPS)), bool),
                  certificate_gradient_norm=np.full((batch, len(FD_STEPS)), np.nan),
                  certificate_gradient_difference=np.full((batch, len(FD_STEPS)), np.nan),
                  certificate_column_relative_error=np.full((batch, len(FD_STEPS), 6), np.nan),
                  certificate_column_norm_ratio=np.full((batch, len(FD_STEPS), 6), np.nan),
                  certificate_status=np.full(batch, 'not_evaluated', dtype='<U40'),
                  certificate_scale_pass=np.zeros((batch, len(FD_STEPS)), bool),
                  certificate_pair_index=np.full(batch, -1, np.int32))
    ok = state['objective_valid'] & analytic['good'] & analytic['differentiable']
    result['certificate_status'][state['objective_valid'] & ~analytic['good']] = 'analytic_linearization_failed'
    result['certificate_status'][state['objective_valid'] & analytic['good'] & ~analytic['differentiable']] = 'analytic_nondifferentiable'
    if not np.any(ok):
        return result
    tolerance = gradient_tolerance*(1+np.sqrt(2*np.where(state['objective_valid'], state['objective'], 0.)))
    angle_branch = _angle_branch(evaluator, parameters, state)
    for index, fd in enumerate(FD_STEPS):
        same = ok.copy()
        jac = np.empty_like(analytic['jacobian'])
        call_index, plus_residual = 0, None

        def observe(candidate):
            nonlocal same, call_index, plus_residual
            trial = evaluator(candidate)
            same &= trial['objective_valid'] & _same_branch(state, trial, evaluator.mask, evaluator.model)
            same &= np.all(~evaluator.mask[..., None] | (_angle_branch(evaluator, candidate, trial) == angle_branch), axis=(1, 2))
            if call_index % 2 == 0:
                plus_residual = trial['residuals']
            else:
                jac[..., call_index//2] = (plus_residual-trial['residuals'])/(2*fd)
            call_index += 1
            return trial

        lin = _linearize(parameters, evaluator.scale, observe, state['residuals'], fd, condition_limit)
        common = lin['g']*lin['norm']/analytic['norm']
        numeric = ok & lin['good'] & np.isfinite(common).all(axis=1)
        norm = np.max(np.abs(common), axis=1)
        difference = np.max(np.abs(common-analytic['g']), axis=1)
        column_error = np.sqrt(np.sum((jac-analytic['jacobian'])**2, axis=(1, 2)))/analytic['norm']
        result['certificate_column_relative_error'][:, index] = np.where(numeric[:, None], column_error, np.nan)
        result['certificate_column_norm_ratio'][:, index] = np.where(numeric[:, None], lin['norm']/analytic['norm'], np.nan)
        passed = (numeric & same & (norm <= tolerance) & (lin['gradient_norm'] <= tolerance)
                  & (difference <= _CERT_FRACTION*tolerance) & np.all(column_error <= _CERT_JAC_RTOL, axis=1))
        result['certificate_same_branch'][:, index] = same
        result['certificate_numeric_good'][:, index] = numeric
        result['certificate_gradient_norm'][:, index] = np.where(numeric, norm, np.nan)
        result['certificate_gradient_difference'][:, index] = np.where(numeric, difference, np.nan)
        result['certificate_scale_pass'][:, index] = passed
    pairs = result['certificate_scale_pass'][:, :-1] & result['certificate_scale_pass'][:, 1:]
    result['certified'] = np.any(pairs, axis=1) & (analytic['gradient_norm'] <= tolerance)
    result['certificate_pair_index'] = np.where(result['certified'], np.argmax(pairs, axis=1), -1).astype(np.int32)
    compatible = result['certificate_same_branch'] & result['certificate_numeric_good']
    result['certificate_status'][ok] = 'fd_check_failed'
    result['certificate_status'][ok & ~np.any(compatible[:, :-1] & compatible[:, 1:], axis=1)] = 'no_admissible_stencil_pair'
    result['certificate_status'][ok & (analytic['gradient_norm'] > tolerance)] = 'analytic_nonstationary'
    result['certificate_status'][result['certified']] = 'adjacent_stencil_checked'
    return result


def fit_stable_drift_helix(local_parameters, z_ref, t0, flight_direction, wire_cal,
                          wire_hv, early_end, early_time, signal_speed, measured_u,
                          uvar, valid, model, *, bz, initial_lr, parameter_scale,
                          seed_valid, mass=105.6583755, max_rows=1024, max_cells=65536,
                          max_iterations=40, condition_limit=1e12, gradient_tolerance=1e-6,
                          initial_damping=1e-3, max_parameter_step=10., max_backtracks=8):
    """Fit the actual objective with analytic within-branch derivatives.

    Inputs match driftfitdiagnostic.diagnose, with explicit frozen initial LR and
    scale. Start from the original measured-wire/profile seeds for baseline
    comparisons, not silently from successful baseline final states. Full event
    census and direction expansion belong to the caller. Memory O(B*H+B*I*6)
    with B<=1024, cells<=65536, I<=200, backtracks<=16; no hit/event loops.

    Every accepted trial has all-hit validity and strictly lower actual stock
    objective; first decreasing trial in a fixed half-step search is accepted.
    Native calibration boundaries may be crossed by trials, never smoothed.
    Within-branch derivatives at nondifferentiable states cannot certify success.
    Damping decreases by three after acceptance and grows by ten after rejection;
    damping above1e12, failed linearization, exhausted budgets and derivative
    inconsistency are explicit failures. No objective-only success or hit pruning.

    Success requires full-rank analytic stationarity at the unchanged1e-6 default
    threshold, differentiability, plus certify(). Failed rows retain last accepted
    finite state. Trace records active iterations, parameters, objective, damping,
    gradients and all bounded trial validity/deltas/parameter stagnation. The
    iterations field counts attempted update rounds; linearization_checks also
    includes stationarity-only checks and the final check. Certificate evaluation
    counts are logical eligible-row attempts, not physical batched work. Physical
    momentum outputs remain absent: this candidate tests numerical fitting only.
    """
    from .driftjacobian import linearize
    max_iterations = _integer('max_iterations', max_iterations)
    max_backtracks = _integer('max_backtracks', max_backtracks, 1)
    if max_iterations > 200 or max_backtracks > 16:
        raise ValueError('candidate iteration/backtrack bound exceeded')
    condition_limit = _scalar('condition_limit', condition_limit, True)
    gradient_tolerance = _scalar('gradient_tolerance', gradient_tolerance, True)
    initial_damping = _scalar('initial_damping', initial_damping, True)
    max_parameter_step = _scalar('max_parameter_step', max_parameter_step, True)
    if condition_limit <= 1 or initial_damping > 1e12:
        raise ValueError('condition_limit must exceed one; damping must not exceed1e12')
    evaluator = _Evaluator(local_parameters, z_ref, t0, flight_direction, wire_cal,
                           wire_hv, early_end, early_time, signal_speed, measured_u,
                           uvar, valid, model, bz=bz, initial_lr=initial_lr,
                           parameter_scale=parameter_scale, seed_valid=seed_valid,
                           mass=mass, max_rows=max_rows, max_cells=max_cells)
    batch, hits = evaluator.batch, evaluator.hits
    parameters = evaluator.candidate.copy()
    state = evaluator(parameters) if batch and hits else _empty_state(evaluator)
    initial = state['objective'].copy()
    initial_valid = state['objective_valid'].copy()
    status = np.full(batch, 'max_iterations', dtype='<U32')
    status[~initial_valid] = 'invalid_initial'
    status[~np.asarray(seed_valid)] = 'invalid_seed'
    status[evaluator.valid.sum(axis=1) == 0] = 'empty'
    active = initial_valid.copy()
    damping = np.full(batch, initial_damping)
    iterations, accepted = np.zeros(batch, np.int32), np.zeros(batch, np.int32)
    trace = dict(active=np.zeros((batch, max_iterations), bool),
                 parameters=np.full((batch, max_iterations, 6), np.nan),
                 objective=np.full((batch, max_iterations), np.nan),
                 damping=np.full((batch, max_iterations), np.nan),
                 gradient_norm=np.full((batch, max_iterations), np.nan),
                 differentiable=np.zeros((batch, max_iterations), bool),
                 trial_valid=np.zeros((batch, max_iterations, max_backtracks), bool),
                 trial_evaluated=np.zeros((batch, max_iterations, max_backtracks), bool),
                 trial_delta=np.full((batch, max_iterations, max_backtracks), np.nan),
                 trial_predicted_delta=np.full((batch, max_iterations, max_backtracks), np.nan),
                 trial_same_parameters=np.zeros((batch, max_iterations, max_backtracks), bool),
                 trial_same_branch=np.zeros((batch, max_iterations, max_backtracks), bool),
                 accepted_backtrack=np.full((batch, max_iterations), -1, np.int32))
    with np.errstate(all='ignore'):
        for iteration in range(max_iterations):
            if not np.any(active):
                break
            lin = linearize(evaluator, parameters, state, condition_limit=condition_limit)
            trace['active'][:, iteration] = active
            for name, value in (('parameters', parameters), ('objective', state['objective']),
                                ('damping', damping), ('gradient_norm', lin['gradient_norm'])):
                trace[name][:, iteration] = np.where(active.reshape((batch,)+(1,)*(value.ndim-1)), value, np.nan)
            trace['differentiable'][:, iteration] = active & lin['differentiable']
            threshold = gradient_tolerance*(1+np.sqrt(2*state['objective']))
            stationary = lin['good'] & lin['differentiable'] & (lin['gradient_norm'] <= threshold)
            status[active & stationary] = 'analytic_stationary'
            status[active & ~lin['good']] = 'ill_conditioned'
            status[active & ~lin['numeric']] = 'linearization_failed'
            active &= lin['good'] & ~stationary
            if not np.any(active):
                break
            iterations += active
            v = lin['eigenvectors']
            projection = np.einsum('bji,bj->bi', v, lin['g'], optimize=False)
            denom = np.maximum(lin['eigenvalues'], 0.)+damping[:, None]
            step = -np.einsum('bij,bj->bi', v, projection/denom, optimize=False)/lin['norm']
            step /= np.maximum(1., np.max(np.abs(step), axis=1)/max_parameter_step)[:, None]
            took = np.zeros(batch, bool)
            angle_branch = _angle_branch(evaluator, parameters, state)
            next_parameters = parameters.copy()
            next_state = {key: value.copy() for key, value in state.items()}
            for backtrack in range(max_backtracks):
                pending = active & ~took
                if not np.any(pending):
                    break
                proposed = step*(.5**backtrack)
                candidate = parameters+np.where(pending[:, None], evaluator.scale*proposed, 0.)
                trial = evaluator(candidate)
                delta = trial['objective']-state['objective']
                dr = np.einsum('bhij,bj->bhi', lin['jacobian'], proposed, optimize=False)
                predicted = np.sum(state['residuals']*dr+.5*dr*dr, axis=(1, 2))
                trace['trial_evaluated'][:, iteration, backtrack] = pending
                trace['trial_valid'][:, iteration, backtrack] = pending & trial['objective_valid']
                trace['trial_delta'][:, iteration, backtrack] = np.where(pending & trial['objective_valid'], delta, np.nan)
                trace['trial_predicted_delta'][:, iteration, backtrack] = np.where(pending, predicted, np.nan)
                trace['trial_same_parameters'][:, iteration, backtrack] = pending & np.all(candidate == parameters, axis=1)
                trace['trial_same_branch'][:, iteration, backtrack] = pending & trial['objective_valid'] & _same_branch(
                    state, trial, evaluator.mask, model) & np.all(~evaluator.mask[..., None] | (
                        _angle_branch(evaluator, candidate, trial) == angle_branch), axis=(1, 2))
                take = pending & trial['objective_valid'] & (delta < 0)
                next_parameters = np.where(take[:, None], candidate, next_parameters)
                for key, value in state.items():
                    expand = take.reshape((batch,)+(1,)*(value.ndim-1))
                    next_state[key] = np.where(expand, trial[key], next_state[key])
                trace['accepted_backtrack'][take, iteration] = backtrack
                took |= take
            parameters, state = next_parameters, next_state
            accepted += took
            damping = np.where(took, np.maximum(1e-12, damping/3),
                               np.where(active, np.minimum(1e13, damping*10), damping))
            stalled = active & (damping > 1e12)
            status[stalled] = 'stalled'
            active &= ~stalled
        out = _center_output(evaluator, state)
        out.update(local_parameters=np.where(initial_valid[:, None], parameters[:, :5], evaluator.parameters[:, :5]),
                   t0=np.where(initial_valid, parameters[:, 5], evaluator.parameters[:, 5]),
                   initial_local_parameters=evaluator.parameters[:, :5].copy(), initial_t0=evaluator.parameters[:, 5].copy(),
                   initial_objective=initial, initial_objective_valid=initial_valid,
                   final_objective=state['objective'].copy(), status=status, termination_status=status.copy(), iterations=iterations,
                   accepted_steps=accepted, damping=damping, success=np.zeros(batch, bool),
                   analytic_stationary=np.zeros(batch, bool), analytic_differentiable=np.zeros(batch, bool),
                   rank=np.zeros(batch, np.int32), condition=np.full(batch, np.nan),
                   gradient_norm=np.full(batch, np.nan), equilibrated_gradient=np.full((batch, 6), np.nan))
        if batch and hits and np.any(initial_valid):
            lin = linearize(evaluator, parameters, state, condition_limit=condition_limit)
            cert = certify(evaluator, parameters, state, lin, condition_limit=condition_limit,
                           gradient_tolerance=gradient_tolerance)
            threshold = gradient_tolerance*(1+np.sqrt(2*state['objective']))
            stationary = initial_valid & lin['good'] & lin['differentiable'] & (lin['gradient_norm'] <= threshold)
            out['success'] = stationary & cert['certified']
            out['analytic_stationary'] = stationary
            out['analytic_differentiable'] = initial_valid & lin['differentiable']
            for key in ('rank', 'condition', 'gradient_norm'):
                out[key] = np.where(initial_valid, lin[key], out[key])
            out['equilibrated_gradient'] = np.where(initial_valid[:, None], lin['g'], out['equilibrated_gradient'])
            status[initial_valid & ~lin['good']] = 'ill_conditioned'
            status[initial_valid & ~lin['numeric']] = 'linearization_failed'
            status[initial_valid & lin['good'] & ~lin['differentiable']] = 'nondifferentiable'
            status[stationary & ~cert['certified']] = 'derivative_unstable'
            status[out['success']] = 'converged_certified'
        else:
            dummy = dict(good=np.zeros(batch, bool), differentiable=np.zeros(batch, bool))
            cert = certify(evaluator, parameters, state, dummy, condition_limit=condition_limit,
                           gradient_tolerance=gradient_tolerance)
        out.update(cert)
    if batch and hits:
        with np.errstate(all='ignore'):
            root = _at_z(parameters[:, :5], evaluator.zref, state['poca_z'], evaluator.mid, evaluator.direction)
        for name in ('correction', 'g', 'h'):
            out['final_poca_'+name] = np.where(state['valid'], root[name], np.nan)
    else:
        for name in ('correction', 'g', 'h'):
            out['final_poca_'+name] = np.full((batch, hits), np.nan)
    out['linearization_checks'] = trace['active'].sum(axis=1)+initial_valid.astype(np.int64)
    out['fit_trial_evaluations'] = trace['trial_evaluated'].sum(axis=(1, 2))
    out['certificate_evaluations'] = np.where(state['objective_valid'] & out['analytic_differentiable']
                                            & (out['rank'] == 6) & (out['condition'] <= condition_limit), 12*len(FD_STEPS), 0)
    out.update({'trace_'+key: value for key, value in trace.items()})
    return out
