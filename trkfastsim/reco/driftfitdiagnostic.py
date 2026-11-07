"""Read-only final-state evidence for the conditional drift fit, NOT a refit.

No acceptance, convergence classification, support changes or LR updates occur.
The supplied LR and saved parameter scales are mandatory, including for failed
fits. Calibration regression-bin switches are potentially discontinuous;
time/phi interpolation knots are continuous (but need not be differentiable).
"""
import numpy as np

from .fielddrift import _index, drift_info
from .helixdriftfit import _POCA, _linearize
from .helixfit import _poca, _real
from .helixtiming import _bool, _integer, _scalar, prepare_timing

FD_STEPS = (1e-3, 1e-4, 1e-5, 1e-6)
DESCENT_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
DESCENT_FD_INDICES = (0, 3)
DESCENT_FD_STEPS = tuple(FD_STEPS[i] for i in DESCENT_FD_INDICES)
COORDINATES = ('x0', 'y0', 'tx', 'ty', 'omega', 't0')
SIDES = ('minus', 'plus')
BRANCH_LABELS = ('time', 'phi', 'offset', 'signed_error', 'signed_doca')
CONDITION_LIMIT = 1e12  # Exactly the fitter default, not a new success criterion.


class _Evaluator:
    """Validated fixed support and geometry inputs; candidates never mutate them."""

    def __init__(self, local_parameters, z_ref, t0, flight_direction, wire_cal,
                 wire_hv, early_end, early_time, signal_speed, measured_u,
                 uvar, valid, model, *, bz, initial_lr, parameter_scale,
                 seed_valid, mass=105.6583755, max_rows=1024, max_cells=65536):
        valid = _bool('valid', valid)
        if valid.ndim != 2:
            raise ValueError('valid must be boolean [batch,hits]')
        batch, hits = valid.shape
        self.max_rows = _integer('max_rows', max_rows)
        self.max_cells = _integer('max_cells', max_cells)
        if batch > self.max_rows or batch*hits > self.max_cells:
            raise ValueError('diagnostic row/cell bound exceeded; chunk rows explicitly')
        self.bz, self.mass = _scalar('bz', bz), _scalar('mass', mass, True)
        if self.bz == 0:
            raise ValueError('bz must be nonzero')
        p = _real('local_parameters', local_parameters, (batch, 5))
        zref = _real('z_ref', z_ref, (batch,))
        t0 = _real('t0', t0, (batch,))
        flight = _real('flight_direction', flight_direction, (batch,))
        cal = _real('wire_cal', wire_cal, valid.shape+(3,))
        hv = _real('wire_hv', wire_hv, cal.shape)
        end, te, speed, u, variance = (_real(n, a, valid.shape) for n, a in
            (('early_end', early_end), ('early_time', early_time),
             ('signal_speed', signal_speed), ('measured_u', measured_u), ('uvar', uvar)))
        lr = _real('initial_lr', initial_lr, valid.shape)
        scale = _real('parameter_scale', parameter_scale, (batch, 6))
        certified = _bool('seed_valid', seed_valid, (batch,))
        self.parameters = np.column_stack((p, t0))
        self.z_ref = zref.copy()
        self.flight_direction = flight.copy()
        self.initial_lr, self.parameter_scale = lr.copy(), scale.copy()
        self.valid, self.model = valid.copy(), model
        self.batch, self.hits = batch, hits
        with np.errstate(all='ignore'):
            # Keep the arithmetic order of fit_drift_helix, not algebraic rewrites.
            mid = .5*cal+.5*hv
            wire = cal-hv
            length = np.sqrt(np.sum(wire*wire, axis=-1))
            hit_ok = (np.isfinite(cal).all(axis=-1) & np.isfinite(hv).all(axis=-1)
                      & np.isfinite(length) & (length > 0) & np.isfinite(u)
                      & np.isfinite(variance) & (variance > 0)
                      & ((end == 0) | (end == 1)) & np.isfinite(te)
                      & np.isfinite(speed) & (speed > 0)
                      & ((lr == -1) | (lr == 1)))
            finite = np.isfinite(p).all(axis=1) & np.isfinite(t0) & np.isfinite(zref)
            self.eligible = (certified & finite & (valid.sum(axis=1) > 0)
                             & np.all(~valid | hit_ok, axis=1)
                             & ((flight == 1) | (flight == -1))
                             & np.isfinite(scale).all(axis=1) & (scale > 0).all(axis=1))
            mask = self.mask = valid & self.eligible[:, None]
            self.cal = np.where(mask[..., None], cal, [1., 0., 0.])
            self.hv = np.where(mask[..., None], hv, [-1., 0., 0.])
            self.mid = np.where(mask[..., None], mid, 0.)
            self.direction = np.where(mask[..., None], wire, [1., 0., 0.])
            self.direction /= np.where(mask, length, 1.)[..., None]
            self.hlen = np.where(mask, length/2, 1.)
            self.u = np.where(mask, u, 0.)
            self.sigma_u = np.sqrt(np.where(mask, variance, 1.))
            self.te, self.speed = np.where(mask, te, 0.), np.where(mask, speed, 1.)
            self.end = np.where(mask, end, 0.)
            self.zref = np.where(finite, zref, 0.)
            self.flight = np.where((flight == 1) | (flight == -1), flight, 1.)
            self.lr = np.where(mask, lr, 0.)
            self.scale = np.where(self.eligible[:, None], scale, 1.)
            self.candidate = np.where(finite[:, None], self.parameters, 0.)

    def __call__(self, candidate):
        """Kernel-format residuals (masked zero), plus branch identities."""
        mask, model = self.mask, self.model
        with np.errstate(all='ignore'):
            spatial = _poca(candidate[:, :5], self.zref, self.mid, self.direction,
                            self.u, self.hlen, mask, **_POCA)
            timing = prepare_timing(candidate[:, :5], self.zref, spatial['z'], self.cal,
                                    self.hv, self.end, self.te, self.speed, mask,
                                    bz=self.bz, mass=self.mass, seed_valid=self.eligible,
                                    max_rows=self.max_rows, max_cells=self.max_cells)
            poca_good = (spatial['valid'] & spatial['converged'] & spatial['inside']
                         & ~spatial['boundary'])
            geometry_row = (self.eligible & timing['row_valid']
                            & np.all(~mask | poca_good, axis=1))
            dt = timing['sensor_time']-candidate[:, 5, None]-self.flight[:, None]*timing['tof_positive']
            signed = self.flight[:, None]*timing['signed_doca_positive']
            query = mask & geometry_row[:, None] & np.isfinite(dt)
            radius, sigma, cdrift = (np.zeros_like(self.u) for _ in range(3))
            bins = {name: np.full(mask.shape, -1, np.int64) for name in
                    ('time_bin', 'phi_bin_lower', 'phi_bin_upper', 'offset_bin', 'signed_error_bin')}
            if np.any(query):
                info = drift_info(model, dt[query], timing['phi'][query])
                radius[query], sigma[query], cdrift[query] = (info['r_drift'],
                                                           info['signed_error'], info['c_drift'])
                folded = np.abs(np.fmod(info['used_phi'], np.pi))
                folded = np.where(folded > np.pi/2, np.pi-folded, folded)
                bins['time_bin'][query] = _index(dt[query], 0., model.delta_t, model.times.size-2)
                bins['phi_bin_lower'][query] = np.floor(folded/model.delta_phi).astype(np.intp)
                bins['phi_bin_upper'][query] = np.minimum(
                    np.ceil(folded/model.delta_phi).astype(np.intp), model.times_dbins.shape[1]-1)
                for name, calibration, value in (('offset', model.offset, info['c_drift']),
                                                ('signed_error', model.signed, info['r_drift'])):
                    bins[name+'_bin'][query] = _index(value, calibration.low, calibration.width,
                                                    calibration.slope.size-1)
            drift_good = query & np.isfinite(radius) & np.isfinite(sigma) & (sigma > 0)
            dr = self.lr*radius-signed
            pull = dr/np.where(drift_good, sigma, 1.)
            long_pull = (spatial['u']-self.u)/self.sigma_u
            residual = np.where(mask[..., None], np.stack((pull, long_pull), axis=-1), 0.)
            objective = .5*np.sum(residual**2, axis=(1, 2))
            row_ok = (geometry_row & np.isfinite(candidate).all(axis=1)
                      & np.all(~mask | drift_good, axis=1) & np.isfinite(objective))
            hit_finite = mask & spatial['valid'] & np.isfinite(residual).all(axis=-1)
            result = dict(objective_valid=row_ok, objective=np.where(row_ok, objective, np.nan),
                          residuals=np.where(hit_finite[..., None], residual, 0.),
                          valid=hit_finite & poca_good & timing['valid'] & drift_good,
                          **bins)
            for name, value in (('doca', timing['doca']), ('signed_doca', signed),
                                ('phi', timing['phi']), ('delta_t', dt), ('r_drift', radius),
                                ('signed_error', sigma), ('c_drift', cdrift),
                                ('drift_residual', dr), ('pull', pull), ('long_pull', long_pull),
                                ('wire_u', spatial['u']), ('poca_z', spatial['z'])):
                result[name] = np.where(hit_finite & np.isfinite(value), value, 0.)
            return result


def _center_output(evaluator, state):
    """Public missing-value policy: false/NaN rows; valid-row masked residuals zero."""
    out = dict(mask=evaluator.valid.copy(), n_hits=evaluator.valid.sum(axis=1),
               initial_lr=evaluator.initial_lr.copy(),
               parameter_scale=evaluator.parameter_scale.copy(),
               local_parameters=evaluator.parameters[:, :5].copy(),
               t0=evaluator.parameters[:, 5].copy(), z_ref=evaluator.z_ref.copy(),
               flight_direction=evaluator.flight_direction.copy())
    row = state['objective_valid']
    hit = state['valid'] & row[:, None]
    for key, value in state.items():
        if key == 'residuals':
            out[key] = np.where(row[:, None, None], value, np.nan)
        elif key == 'valid':
            out[key] = hit
        elif value.ndim == 2:
            out[key] = np.where(hit, value, -1 if key.endswith('_bin') or '_bin_' in key else np.nan)
        else:
            out[key] = value.copy()
    return out


def _empty_state(evaluator):
    shape = evaluator.valid.shape
    out = dict(objective=np.full(evaluator.batch, np.nan),
               objective_valid=np.zeros(evaluator.batch, bool), valid=np.zeros(shape, bool),
               residuals=np.zeros(shape+(2,)))
    for name in ('doca', 'signed_doca', 'phi', 'delta_t', 'r_drift', 'signed_error',
                 'c_drift', 'drift_residual', 'pull', 'long_pull', 'wire_u', 'poca_z'):
        out[name] = np.zeros(shape)
    for name in ('time_bin', 'phi_bin_lower', 'phi_bin_upper', 'offset_bin', 'signed_error_bin'):
        out[name] = np.full(shape, -1, np.int64)
    return out


def evaluate(local_parameters, z_ref, t0, flight_direction, wire_cal, wire_hv,
             early_end, early_time, signal_speed, measured_u, uvar, valid, model,
             *, bz, initial_lr, parameter_scale, seed_valid, mass=105.6583755,
             max_rows=1024, max_cells=65536):
    """Re-evaluate one supplied state using the same contract as :func:`diagnose`.

    This performs no finite differences or LR choice. It is useful for independent
    replay of slopes/trials. No objective is available unless the entire fixed
    support is valid. Model evaluation exceptions propagate, as in the fitter.
    """
    evaluator = _Evaluator(local_parameters, z_ref, t0, flight_direction, wire_cal,
                           wire_hv, early_end, early_time, signal_speed, measured_u,
                           uvar, valid, model, bz=bz, initial_lr=initial_lr,
                           parameter_scale=parameter_scale, seed_valid=seed_valid,
                           mass=mass, max_rows=max_rows, max_cells=max_cells)
    state = evaluator(evaluator.candidate) if evaluator.batch and evaluator.hits else _empty_state(evaluator)
    return _center_output(evaluator, state)


def _branch_counts(center, trial, valid, model):
    """Count hit/perturbation incidences, not unique hits across perturbations."""
    mask = center['valid'] & trial['valid'] & valid[:, None]
    changes = [center['time_bin'] != trial['time_bin'],
               (center['phi_bin_lower'] != trial['phi_bin_lower'])
               | (center['phi_bin_upper'] != trial['phi_bin_upper']),
               center['offset_bin'] != trial['offset_bin'],
               center['signed_error_bin'] != trial['signed_error_bin'],
               np.sign(center['signed_doca']) != np.sign(trial['signed_doca'])]
    counts = np.stack([np.sum(mask & change, axis=1) for change in changes], axis=-1)
    regression = []
    for name, calibration in (('offset', model.offset), ('signed_error', model.signed)):
        a, b = np.maximum(center[name+'_bin'], 0), np.maximum(trial[name+'_bin'], 0)
        changed = ((calibration.intercept[a] != calibration.intercept[b])
                   | (calibration.slope[a] != calibration.slope[b]))
        regression.append(np.sum(mask & changed, axis=1))
    return counts, np.stack(regression, axis=-1)


def diagnose(local_parameters, z_ref, t0, flight_direction, wire_cal, wire_hv,
             early_end, early_time, signal_speed, measured_u, uvar, valid, model,
             *, bz, initial_lr, parameter_scale, seed_valid, mass=105.6583755,
             max_rows=1024, max_cells=65536):
    """Diagnose a SAVED final state, including unsuccessful fits, without fitting.

    Shapes match fit_drift_helix: parameters [B,5], row inputs [B], endpoints
    [B,H,3], other hit inputs [B,H]. initial_lr [B,H] must be +/-1 on selected
    hits; parameter_scale [B,6] must be the saved positive finite scales.
    seed_valid [B] certifies the original seed, NOT final fit success. Invalid
    data fail the entire row; invalid scales/LR never get inferred or repaired.
    Hard row/cell bounds raise, never truncate. Memory is O(B*H); only fixed
    coordinate/scale loops are used. Caller row batching is array-exact.

    numeric/linearization_good/gradient_norm/condition/rank are [B,4], using
    FD_STEPS and the exact default _linearize implementation. equilibrated_gradient
    and column_norm [B,4,6] expose alternate-gradient evidence. No new gradient
    changes any fit status. Invalid numeric diagnostics are NaN/false (rank too).

    objective_slopes [B,4,6,2] are positive-coordinate one-sided derivatives,
    (F0-Fminus)/h and (Fplus-F0)/h in dimensionless saved-scale coordinates.
    perturbation_valid has the same shape. branch_counts [B,4,6,2,5] uses
    BRANCH_LABELS; *_branch_count [B,4] sums hit/perturbation incidences over
    all 12 evaluations. branch_valid equals perturbation_valid: zero counts
    with false validity are unavailable, NOT evidence of smoothness. Time and
    phi knots are continuous interpolation; offset/signed-error bins select
    regressions, which can be discontinuous. *_regression_change_count excludes
    bin changes with identical coefficients, but is NOT proof of a nonzero jump.

    descent_delta/valid [B,2,7] uses DESCENT_FD_STEPS and DESCENT_STEPS. The
    dimensionless direction is -g/column_norm, normalized to maxabs=1. These
    are signed Ftrial-Fcenter values, not accepted steps. A zero or nonnumeric
    direction is unavailable; full rank is not needed to probe a direction.

    lr_preferred = sign(r_drift*signed_doca), retaining supplied LR on a tie.
    lr_improvement is the SIGNED old-minus-flipped per-hit objective,
    -2*initial_lr*r_drift*signed_doca/signed_error**2; long pulls cancel.
    lr_gain/count sum positive benefits/counts per row. Negative radii are
    preserved. This fixed-geometry counterfactual is not ambiguity resolution.
    Unavailable hit values are NaN (bin indices -1); masked residuals are zero
    in valid rows for exact objective reduction. Invalid rows have NaN
    residuals, false validity and NaN gains. Original mask/LR/parameters/scales
    are copied unchanged, including invalid supplied values. No status or
    success classification is returned.
    """
    evaluator = _Evaluator(local_parameters, z_ref, t0, flight_direction, wire_cal,
                           wire_hv, early_end, early_time, signal_speed, measured_u,
                           uvar, valid, model, bz=bz, initial_lr=initial_lr,
                           parameter_scale=parameter_scale, seed_valid=seed_valid,
                           mass=mass, max_rows=max_rows, max_cells=max_cells)
    batch = evaluator.batch
    center = evaluator(evaluator.candidate) if batch and evaluator.hits else _empty_state(evaluator)
    out = _center_output(evaluator, center)
    for key in ('gradient_norm', 'condition', 'rank'):
        out[key] = np.full((batch, len(FD_STEPS)), np.nan)
    for key in ('numeric', 'linearization_good'):
        out[key] = np.zeros((batch, len(FD_STEPS)), bool)
    for key in ('equilibrated_gradient', 'column_norm'):
        out[key] = np.full((batch, len(FD_STEPS), 6), np.nan)
    out['objective_slopes'] = np.full((batch, len(FD_STEPS), 6, 2), np.nan)
    out['perturbation_valid'] = np.zeros((batch, len(FD_STEPS), 6, 2), bool)
    out['branch_counts'] = np.zeros((batch, len(FD_STEPS), 6, 2, len(BRANCH_LABELS)), np.int64)
    out['regression_change_counts'] = np.zeros((batch, len(FD_STEPS), 6, 2, 2), np.int64)
    out['descent_delta'] = np.full((batch, len(DESCENT_FD_INDICES), len(DESCENT_STEPS)), np.nan)
    out['descent_valid'] = np.zeros(out['descent_delta'].shape, bool)
    out['descent_direction'] = np.full((batch, len(DESCENT_FD_INDICES), 6), np.nan)
    row = center['objective_valid']
    with np.errstate(all='ignore'):
        if np.any(row):
            for scale_index, fd in enumerate(FD_STEPS):
                call_index = 0

                def observe(candidate):
                    nonlocal call_index
                    column, side = call_index//2, 1-call_index % 2  # fitter calls + then -
                    call_index += 1
                    trial = evaluator(candidate)
                    good = row & trial['objective_valid']
                    delta = trial['objective']-center['objective']
                    out['objective_slopes'][:, scale_index, column, side] = np.where(
                        good, delta/(fd if side else -fd), np.nan)
                    out['perturbation_valid'][:, scale_index, column, side] = good
                    counts, regression = _branch_counts(center, trial, good, model)
                    out['branch_counts'][:, scale_index, column, side] = counts
                    out['regression_change_counts'][:, scale_index, column, side] = regression
                    return trial

                lin = _linearize(evaluator.candidate, evaluator.scale, observe,
                                 center['residuals'], fd, CONDITION_LIMIT)
                numeric = row & lin['numeric']
                out['numeric'][:, scale_index] = numeric
                out['linearization_good'][:, scale_index] = row & lin['good']
                for key in ('gradient_norm', 'condition', 'rank'):
                    out[key][:, scale_index] = np.where(numeric, lin[key], np.nan)
                out['equilibrated_gradient'][:, scale_index] = np.where(numeric[:, None], lin['g'], np.nan)
                out['column_norm'][:, scale_index] = np.where(numeric[:, None], lin['norm'], np.nan)
                if scale_index in DESCENT_FD_INDICES:
                    descent_index = DESCENT_FD_INDICES.index(scale_index)
                    direction = -lin['g']/lin['norm']
                    maximum = np.max(np.abs(direction), axis=1)
                    direction_ok = numeric & np.isfinite(direction).all(axis=1) & (maximum > 0)
                    direction /= np.where(direction_ok, maximum, 1.)[:, None]
                    direction = np.where(direction_ok[:, None], direction, 0.)
                    out['descent_direction'][:, descent_index] = np.where(direction_ok[:, None], direction, np.nan)
                    for step_index, step in enumerate(DESCENT_STEPS):
                        trial = evaluator(evaluator.candidate+evaluator.scale*direction*step)
                        good = direction_ok & trial['objective_valid']
                        out['descent_valid'][:, descent_index, step_index] = good
                        out['descent_delta'][:, descent_index, step_index] = np.where(
                            good, trial['objective']-center['objective'], np.nan)
        for index, label in enumerate(BRANCH_LABELS):
            out[label+'_branch_count'] = out['branch_counts'][..., index].sum(axis=(2, 3))
        for index, label in enumerate(('offset', 'signed_error')):
            out[label+'_regression_change_count'] = out['regression_change_counts'][..., index].sum(axis=(2, 3))
        out['branch_valid'] = out['perturbation_valid'].copy()
        hit = evaluator.valid & row[:, None]
        product = center['r_drift']*center['signed_doca']
        preferred = np.where(product == 0, evaluator.initial_lr, np.sign(product))
        improvement = -2*evaluator.initial_lr*product/center['signed_error']**2
        out['lr_preferred'] = np.where(hit, preferred, np.nan)
        out['lr_improvement'] = np.where(hit, improvement, np.nan)
        out['lr_gain'] = np.where(row, np.sum(np.where(hit, np.maximum(improvement, 0.), 0.), axis=1), np.nan)
        out['lr_flip_count'] = np.where(row, np.sum(hit & (improvement > 0), axis=1), np.nan)
    return out
