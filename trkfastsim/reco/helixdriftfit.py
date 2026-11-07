"""Opt-in fixed-membership six-parameter drift/time + longitudinal refit.

An engineering conditional objective, NOT chi-square, a covariance, a global
helix/ambiguity search, or a validated physical reconstruction. No truth,
association, TOT/timeVar, material, fallback, outlier cuts or direction winner.
"""
import numpy as np

from .fielddrift import drift_info
from .helixfit import _poca, _real
from .helixtiming import _bool, _integer, _scalar, prepare_timing

_EPS = np.finfo(float).eps
_MAX = np.finfo(float).max
_POCA = dict(iterations=20, tolerance=1e-7, max_step=50., max_offset=500.)


def _linearize(parameters, scale, evaluate, residual, fd_step, condition_limit):
    """Central differences of the ACTUAL normalized residual, including POCA."""
    batch, hits, _ = residual.shape
    jac = np.empty((batch, hits, 2, 6))
    perturbations_ok = np.ones(batch, bool)
    for column in range(6):
        step = np.zeros_like(parameters)
        step[:, column] = fd_step*scale[:, column]
        plus, minus = evaluate(parameters+step), evaluate(parameters-step)
        perturbations_ok &= plus['objective_valid'] & minus['objective_valid']
        jac[..., column] = (plus['residuals']-minus['residuals'])/(2*fd_step)
    numeric = perturbations_ok & np.isfinite(jac).all(axis=(1, 2, 3))
    # Placeholders ONLY protect linear algebra; they NEVER certify a derivative.
    jac = np.where(numeric[:, None, None, None], jac, 0.)
    flat = jac.reshape(batch, hits*2, 6)
    normal = np.einsum('bhi,bhj->bij', flat, flat, optimize=False)
    gradient = np.einsum('bhi,bh->bi', flat, residual.reshape(batch, hits*2), optimize=False)
    norm = np.sqrt(np.maximum(np.diagonal(normal, axis1=1, axis2=2), 0.))
    nonzero = np.all(norm > np.finfo(float).tiny, axis=1)
    norm = np.where(norm > np.finfo(float).tiny, norm, 1.)
    correlation = normal/norm[:, :, None]/norm[:, None, :]
    g = gradient/norm
    numeric &= np.isfinite(correlation).all(axis=(1, 2)) & np.isfinite(g).all(axis=1)
    correlation = np.where(numeric[:, None, None], correlation, np.eye(6))
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    except np.linalg.LinAlgError:
        # No unsafe pseudoinverse, damping-only or objective-only success.
        eigenvalues = np.zeros((batch, 6))
        eigenvectors = np.broadcast_to(np.eye(6), (batch, 6, 6)).copy()
        numeric[:] = False
    numeric &= (np.isfinite(eigenvalues).all(axis=1)
                & np.isfinite(eigenvectors).all(axis=(1, 2)))
    lo, hi = eigenvalues[:, 0], eigenvalues[:, -1]
    condition = np.divide(hi, lo, out=np.full(batch, _MAX), where=lo > 0)
    condition = np.minimum(np.nan_to_num(condition, nan=_MAX, posinf=_MAX), _MAX)
    rank = np.sum(eigenvalues > 128*_EPS*np.maximum(hi, 0.)[:, None], axis=1)
    good = numeric & nonzero & (rank == 6) & (condition <= condition_limit)
    return dict(good=good, numeric=numeric,
                condition=np.where(numeric & nonzero, condition, _MAX),
                rank=np.where(numeric & nonzero, rank, 0),
                eigenvalues=np.where(numeric[:, None], eigenvalues, 0.),
                eigenvectors=eigenvectors, norm=np.where(numeric[:, None], norm, 1.),
                g=np.where(numeric[:, None], g, 0.),
                gradient_norm=np.where(numeric, np.max(np.abs(g), axis=1), _MAX))


def fit_drift_helix(local_parameters, z_ref, t0, flight_direction, wire_cal,
                    wire_hv, early_end, early_time, signal_speed, measured_u,
                    uvar, valid, model, *, bz, seed_valid=None, max_iterations=40,
                    condition_limit=1e12, gradient_tolerance=1e-6, fd_step=1e-3,
                    initial_damping=1e-3, max_parameter_step=10., max_rows=1024,
                    max_cells=65536, mass=105.6583755):
    """Joint local fit in mm/ns/MeV, separately for each supplied direction.

    local_parameters [B,5]=(x0,y0,tx,ty,omega), z_ref/t0/flight_direction [B];
    endpoints [B,H,3], all remaining hit inputs [B,H]. valid is boolean and
    frozen. measured_u is relative to (cal+hv)/2 along unit(cal-hv), uvar>0.
    Direction MUST be +/-1 (bad values fail the lane). To fit both directions
    the CALLER expands rows. seed_valid=None certifies no rows; explicitly
    certify a successful measured-wire fit AND valid t0 profile. No inference
    of that certification from numeric values is possible.

    Every candidate and every +/- finite-difference column recomputes LOCAL
    spatial _poca from the measured wire coordinate, then prepare_timing from
    those POCAs and ORIGINAL early-end measurements. All selected hits must
    be finite, converged, strictly inside finite wires, with well-defined
    angle/sign and current measured-curvature beta. Initial LR is frozen as
    sign(signed_doca_positive)*flight_direction. The CURRENT signed DOCA may
    change sign if geometry remains finite/stable (zero DOCA is invalid):
    this is NOT a fixed-spatial-side branch and LR is never reassigned.

    dt = sensor_time-t0-flight_direction*tof_positive;
    drift_residual = initial_LR*r_drift-flight_direction*signed_doca_positive;
    pull = drift_residual/current_signed_error; long_pull=(wire_u-measured_u)/
    sqrt(uvar). Objective=.5*sum(pull**2+long_pull**2). No abs(r_drift), sigma
    floor, smoothing of calibration switches, dt/radius cuts or extra errors.
    Outside [-50,200] ns is recorded as oracle COVERAGE, not a cut. Native
    model evaluation exceptions propagate (as in helixtiming.profile_t0);
    returned nonfinite/nonpositive signed error fails the whole candidate.

    Fixed scales from the INITIAL measured-z span L=max(1,span): (1,1,1/L,
    1/L,1/(L*L*max(transverse_slope,1/L)),10 ns). fd_step is dimensionless.
    Diagonally equilibrated full-rank-six LM, one bounded trial/iteration,
    accepted damping /3, rejected damping *10. Any invalid +/- perturbation
    is linearization_failed, NEVER a zero-Jacobian success. Unconstrained
    equilibrated gradient <= tolerance*(1+sqrt(2*objective)), full rank and
    condition, and all-hit validity are required at the FINAL iterate.
    max_iterations=0 checks stationarity only. Budget exhaustion, damping
    stalls and piecewise-model derivative stalls do NOT imply success.

    Returns per-row success/converged/status, n_hits, iterations/accepted_steps,
    initial/final_objective, initial/final_objective_valid (objective_valid is
    final), rank/condition/eigenvalues/gradient_norm/damping/parameter_scale,
    initial/local_parameters, initial/t0, z_ref, flight_direction, beta/p_abs.
    Finite last accepted parameters/t0 survive failures; beta/p_abs and
    physical_identifiable require success AND |omega|*max(1,measured_z_span)>1e-8,
    matching the wire initializer's numerical curvature-span guard. A converged
    nearly straight fit may expose no physical momentum. This is not a momentum
    uncertainty criterion. No charge or covariance is reported.
    Per-hit: mask (original membership), initial_LR, doca/signed_doca/phi,
    delta_t/r_drift/signed_error/drift_residual/pull/long_pull/wire_u/poca_z,
    residuals [B,H,2], valid (usable hit), timing_valid, drift_valid,
    poca_valid/converged/inside/boundary/iterations, timing geometry flags,
    outside_oracle_mask. outside_oracle is the per-row hit count. Unavailable
    diagnostics are zero with false validity; unavailable objectives are NaN.
    objective_label and status_counts are explicit non-array metadata.

    O(B*H) workspace, largest Jacobian [B,H,2,6]. Hard row/cell limits raise,
    never truncate. Only fixed-coordinate, bounded iteration and dict-key
    loops, not event/hit loops. Caller chunks rows without changing reductions.
    """
    valid = _bool('valid', valid)
    if valid.ndim != 2:
        raise ValueError('valid must be boolean [batch,hits]')
    batch, hits = valid.shape
    max_iterations = _integer('max_iterations', max_iterations)
    max_rows, max_cells = _integer('max_rows', max_rows), _integer('max_cells', max_cells)
    if batch > max_rows or batch*hits > max_cells:
        raise ValueError('joint refit row/cell bound exceeded; chunk rows explicitly')
    bz, mass = _scalar('bz', bz), _scalar('mass', mass, True)
    if bz == 0:
        raise ValueError('bz must be nonzero')
    condition_limit = _scalar('condition_limit', condition_limit, True)
    gradient_tolerance = _scalar('gradient_tolerance', gradient_tolerance, True)
    fd_step = _scalar('fd_step', fd_step, True)
    initial_damping = _scalar('initial_damping', initial_damping, True)
    max_parameter_step = _scalar('max_parameter_step', max_parameter_step, True)
    if condition_limit <= 1 or initial_damping > 1e12:
        raise ValueError('condition_limit must exceed one; initial_damping <= 1e12')
    p = _real('local_parameters', local_parameters, (batch, 5)).copy()
    zref = _real('z_ref', z_ref, (batch,)).copy()
    t0 = _real('t0', t0, (batch,)).copy()
    flight = _real('flight_direction', flight_direction, (batch,)).copy()
    cal = _real('wire_cal', wire_cal, valid.shape+(3,))
    hv = _real('wire_hv', wire_hv, cal.shape)
    end, te, speed, u, variance = (_real(n, a, valid.shape) for n, a in
        (('early_end', early_end), ('early_time', early_time),
         ('signal_speed', signal_speed), ('measured_u', measured_u), ('uvar', uvar)))
    certified = np.zeros(batch, bool) if seed_valid is None else _bool('seed_valid', seed_valid, (batch,))
    seed_finite = np.isfinite(p).all(axis=1) & np.isfinite(t0) & np.isfinite(zref)
    seed_ok = certified & seed_finite
    n_hits = valid.sum(axis=1)
    out = dict(success=np.zeros(batch, bool), converged=np.zeros(batch, bool),
               status=np.full(batch, 'max_iterations', dtype='<U32'), n_hits=n_hits,
               iterations=np.zeros(batch, np.int32), accepted_steps=np.zeros(batch, np.int32),
               initial_objective=np.full(batch, np.nan), final_objective=np.full(batch, np.nan),
               initial_objective_valid=np.zeros(batch, bool), final_objective_valid=np.zeros(batch, bool),
               objective_valid=np.zeros(batch, bool), condition=np.full(batch, _MAX),
               rank=np.zeros(batch, np.int32), eigenvalues=np.zeros((batch, 6)),
               gradient_norm=np.full(batch, _MAX), damping=np.full(batch, initial_damping),
               initial_local_parameters=p.copy(), local_parameters=p.copy(),
               initial_t0=t0.copy(), t0=t0.copy(), z_ref=zref.copy(), flight_direction=flight.copy(),
               parameter_scale=np.ones((batch, 6)), initial_LR=np.zeros(valid.shape),
               mask=valid.copy(), physical_identifiable=np.zeros(batch, bool),
               beta=np.full(batch, np.nan), p_abs=np.full(batch, np.nan),
               residuals=np.zeros(valid.shape+(2,)), outside_oracle=np.zeros(batch, np.int64),
               objective_label='engineering_conditional_objective')
    for name in ('doca', 'signed_doca', 'phi', 'delta_t', 'r_drift', 'signed_error',
                 'drift_residual', 'pull', 'long_pull', 'wire_u', 'poca_z'):
        out[name] = np.zeros(valid.shape)
    for name in ('valid', 'timing_valid', 'drift_valid', 'poca_valid', 'poca_converged',
                 'poca_inside', 'poca_boundary', 'inside', 'stationary', 'parallel',
                 'angle_valid', 'sign_valid', 'outside_oracle_mask'):
        out[name] = np.zeros(valid.shape, bool)
    out['poca_iterations'] = np.zeros(valid.shape, np.int32)
    out['status'][~seed_ok] = 'invalid_seed'
    out['status'][n_hits == 0] = 'empty'
    if batch == 0 or hits == 0:
        labels, counts = np.unique(out['status'], return_counts=True)
        out['status_counts'] = dict(zip(labels.tolist(), counts.tolist()))
        return out

    with np.errstate(all='ignore'):
        # Derived arithmetic failure invalidates the entire lane without dropping hits.
        mid = .5*cal+.5*hv
        wire = cal-hv
        length = np.sqrt(np.sum(wire*wire, axis=-1))
        hit_ok = (np.isfinite(cal).all(axis=-1) & np.isfinite(hv).all(axis=-1)
                  & np.isfinite(length) & (length > 0) & np.isfinite(u)
                  & np.isfinite(variance) & (variance > 0)
                  & ((end == 0) | (end == 1)) & np.isfinite(te)
                  & np.isfinite(speed) & (speed > 0))
        input_ok = np.all(~valid | hit_ok, axis=1) & ((flight == 1) | (flight == -1))
        eligible = seed_ok & input_ok & (n_hits > 0)
        mask = valid & eligible[:, None]
        out['status'][~input_ok & (n_hits > 0)] = 'invalid_input'
        cal = np.where(mask[..., None], cal, [1., 0., 0.])
        hv = np.where(mask[..., None], hv, [-1., 0., 0.])
        mid = np.where(mask[..., None], mid, 0.)
        direction = np.where(mask[..., None], wire, [1., 0., 0.])
        direction /= np.where(mask, length, 1.)[..., None]
        hlen = np.where(mask, length/2, 1.)
        u = np.where(mask, u, 0.)
        sigma_u = np.sqrt(np.where(mask, variance, 1.))
        te, speed, end = np.where(mask, te, 0.), np.where(mask, speed, 1.), np.where(mask, end, 0.)
        parameters = np.column_stack((np.where(seed_finite[:, None], p, 0.),
                                      np.where(seed_finite, t0, 0.)))
        zref = np.where(seed_finite, zref, 0.)
        flight = np.where((flight == 1) | (flight == -1), flight, 1.)
        measured_z = mid[..., 2]+u*direction[..., 2]
        span = np.max(np.where(mask, measured_z, -np.inf), axis=1)-np.min(
            np.where(mask, measured_z, np.inf), axis=1)
        span_ok = np.isfinite(span)
        extent = np.where(eligible & span_ok, np.maximum(1., span), 1.)
        slope = np.hypot(parameters[:, 2], parameters[:, 3])
        scale = np.column_stack((np.ones(batch), np.ones(batch), 1/extent, 1/extent,
                                 1/extent/extent/np.maximum(slope, 1/extent), np.full(batch, 10.)))
        scale_ok = np.isfinite(scale).all(axis=1) & (scale > 0).all(axis=1) & span_ok
        out['parameter_scale'] = np.where(scale_ok[:, None], scale, 1.)
        scale = out['parameter_scale']
        eligible &= scale_ok
        mask = valid & eligible[:, None]
        out['status'][seed_ok & input_ok & (n_hits > 0) & ~scale_ok] = 'invalid_input'
        initial_lr = None

        def evaluate(candidate):
            spatial = _poca(candidate[:, :5], zref, mid, direction, u, hlen, mask, **_POCA)
            timing = prepare_timing(candidate[:, :5], zref, spatial['z'], cal, hv,
                                    end, te, speed, mask, bz=bz, mass=mass,
                                    seed_valid=eligible, max_rows=max_rows, max_cells=max_cells)
            poca_good = spatial['valid'] & spatial['converged'] & spatial['inside'] & ~spatial['boundary']
            geometry_row = eligible & timing['row_valid'] & np.all(~mask | poca_good, axis=1)
            dt = timing['sensor_time']-candidate[:, 5, None]-flight[:, None]*timing['tof_positive']
            signed = flight[:, None]*timing['signed_doca_positive']
            lr = np.sign(signed) if initial_lr is None else initial_lr
            query = mask & geometry_row[:, None] & np.isfinite(dt)
            radius, sigma = np.zeros_like(u), np.zeros_like(u)
            if np.any(query):
                info = drift_info(model, dt[query], timing['phi'][query])
                radius[query], sigma[query] = info['r_drift'], info['signed_error']
            drift_good = query & np.isfinite(radius) & np.isfinite(sigma) & (sigma > 0)
            dr = lr*radius-signed
            pull = dr/np.where(drift_good, sigma, 1.)
            long_pull = (spatial['u']-u)/sigma_u
            residual = np.where(mask[..., None], np.stack((pull, long_pull), axis=-1), 0.)
            objective = .5*np.sum(residual**2, axis=(1, 2))
            row_ok = (geometry_row & np.isfinite(candidate).all(axis=1)
                      & np.all(~mask | drift_good, axis=1) & np.isfinite(objective))
            status = timing['status'].astype('<U32')
            status[geometry_row & ~row_ok] = 'drift_failed'
            status[eligible & ~np.all(~mask | poca_good, axis=1)] = 'poca_failed'
            status[eligible & np.any(spatial['boundary'], axis=1)] = 'poca_boundary'
            status[row_ok] = 'ready'
            hit_finite = mask & spatial['valid'] & np.isfinite(residual).all(axis=-1)
            result = dict(objective_valid=row_ok, objective=np.where(row_ok, objective, np.nan),
                          status=status, residuals=np.where(hit_finite[..., None], residual, 0.),
                          valid=hit_finite & poca_good & timing['valid'] & drift_good,
                          timing_valid=timing['valid'], drift_valid=drift_good,
                          beta=timing['beta'], p_abs=timing['p_abs'])
            for name, value in (('doca', timing['doca']), ('signed_doca', signed),
                                ('phi', timing['phi']), ('delta_t', dt), ('r_drift', radius),
                                ('signed_error', sigma), ('drift_residual', dr), ('pull', pull),
                                ('long_pull', long_pull), ('wire_u', spatial['u']), ('poca_z', spatial['z'])):
                result[name] = np.where(hit_finite & np.isfinite(value), value, 0.)
            for name in ('valid', 'converged', 'inside', 'boundary'):
                result['poca_'+name] = mask & spatial[name]
            result['poca_iterations'] = np.where(mask, spatial['iterations'], 0)
            for name in ('inside', 'stationary', 'parallel', 'angle_valid', 'sign_valid'):
                result[name] = timing[name]
            result['outside_oracle_mask'] = query & ((dt < -50.) | (dt > 200.))
            return result

        state = evaluate(parameters)
        good = state['objective_valid'].copy()
        initial_lr = np.where(mask & good[:, None], np.sign(state['signed_doca']), 0.)
        out['initial_LR'] = initial_lr.copy()
        out['initial_objective'] = state['objective'].copy()
        out['initial_objective_valid'] = good.copy()
        out['status'][eligible & ~good] = state['status'][eligible & ~good]
        active = good.copy()
        # Derivative/trial calls cannot rehabilitate an invalid initial lane.
        eligible &= good
        mask = valid & eligible[:, None]
        for _ in range(max_iterations):
            if not np.any(active):
                break
            lin = _linearize(parameters, scale, evaluate, state['residuals'], fd_step, condition_limit)
            stationary = lin['gradient_norm'] <= gradient_tolerance*(1+np.sqrt(2*state['objective']))
            out['status'][active & ~lin['good']] = 'ill_conditioned'
            out['status'][active & ~lin['numeric']] = 'linearization_failed'
            active &= lin['good'] & ~stationary
            if not np.any(active):
                break
            out['iterations'] += active
            v = lin['eigenvectors']
            projection = np.einsum('bji,bj->bi', v, lin['g'], optimize=False)
            denom = np.maximum(lin['eigenvalues'], 0.)+out['damping'][:, None]
            step = -np.einsum('bij,bj->bi', v, projection/denom, optimize=False)/lin['norm']
            step /= np.maximum(1., np.max(np.abs(step), axis=1)/max_parameter_step)[:, None]
            trial_p = parameters+np.where(active[:, None], scale*step, 0.)
            trial = evaluate(trial_p)
            take = active & trial['objective_valid'] & (trial['objective'] < state['objective'])
            parameters = np.where(take[:, None], trial_p, parameters)
            for key in state:
                expand = take.reshape((batch,)+(1,)*(state[key].ndim-1))
                state[key] = np.where(expand, trial[key], state[key])
            out['accepted_steps'] += take
            out['damping'] = np.where(take, np.maximum(1e-12, out['damping']/3),
                                      np.where(active, np.minimum(1e13, out['damping']*10), out['damping']))
            stalled = active & (out['damping'] > 1e12)
            out['status'][stalled] = 'stalled'
            active &= ~stalled
        if np.any(good):
            lin = _linearize(parameters, scale, evaluate, state['residuals'], fd_step, condition_limit)
            stationary = lin['gradient_norm'] <= gradient_tolerance*(1+np.sqrt(2*state['objective']))
            out['success'] = good & state['objective_valid'] & lin['good'] & stationary
            out['converged'] = out['success'].copy()
            out['status'][good & ~lin['good']] = 'ill_conditioned'
            out['status'][good & ~lin['numeric']] = 'linearization_failed'
            out['status'][out['success']] = 'converged'
            for key in ('condition', 'rank', 'eigenvalues', 'gradient_norm'):
                expand = good.reshape((batch,)+(1,)*(lin[key].ndim-1))
                out[key] = np.where(expand, lin[key], out[key])
        out['final_objective'] = state['objective'].copy()
        out['objective_valid'] = state['objective_valid'].copy()
        out['final_objective_valid'] = out['objective_valid'].copy()
        out['local_parameters'] = np.where(seed_finite[:, None], parameters[:, :5], p)
        out['t0'] = np.where(seed_finite, parameters[:, 5], t0)
        for key in state:
            if key in out and key not in ('status', 'objective_valid', 'beta', 'p_abs'):
                out[key] = state[key].copy()
        out['outside_oracle'] = out['outside_oracle_mask'].sum(axis=1)
        out['physical_identifiable'] = (out['success'] & (np.abs(parameters[:, 4])*extent > 1e-8)
                                        & (np.hypot(parameters[:, 2], parameters[:, 3]) > 128*_EPS))
        for key in ('beta', 'p_abs'):
            out[key] = np.where(out['physical_identifiable'], state[key], np.nan)
    labels, counts = np.unique(out['status'], return_counts=True)
    out['status_counts'] = dict(zip(labels.tolist(), counts.tolist()))
    return out
