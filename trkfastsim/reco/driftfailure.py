"""Read-only failure evidence, NOT a fit, repair, or new success criterion.

The counterfactual changes only numerical POCA stopping: three bounded Newton
rounds start from the ORIGINAL evaluator root at every center/probe evaluation.
Stock timing/response and the unchanged analytic/adjacent-stencil check follow.
No LR, support, parameter, calibration, sigma or certificate threshold changes.
"""
import numpy as np

from . import driftjacobian as analytic
from .driftfitdiagnostic import _empty_state
from .fielddrift import _index, drift_info
from .helixdriftfit import _POCA
from .helixfit import _at_z, _real, _trajectory
from .helixtiming import prepare_timing
from .stabledriftfit import _angle_branch, _same_branch, certify

_EPS = np.finfo(float).eps
_BOUNDARIES = ('time_boundary', 'phi_boundary', 'offset_boundary', 'signed_error_boundary')


def _timing(ev, p, z):
    return prepare_timing(p[:, :5], ev.zref, z, ev.cal, ev.hv, ev.end, ev.te,
                          ev.speed, ev.mask, bz=ev.bz, mass=ev.mass,
                          seed_valid=ev.eligible, max_rows=ev.max_rows, max_cells=ev.max_cells)


def _root(ev, p, z):
    root = _at_z(p[:, :5], ev.zref, z, ev.mid, ev.direction)
    omega = np.abs(p[:, 4, None])
    inverse = 1./np.maximum(omega, np.finfo(float).tiny)
    basin = np.minimum(_POCA['max_offset'], (np.pi/2)*np.minimum(inverse, np.finfo(float).max/2))
    anchor = ev.mid[..., 2]+ev.u*ev.direction[..., 2]
    valid = (ev.mask & root['positive'] & (np.abs(z-anchor) < basin)
             & (np.abs(root['u']) < ev.hlen)
             & (np.abs(root['correction']) <= _POCA['tolerance'])
             & (np.abs(root['g']) <= _POCA['tolerance']*np.maximum(1., root['speed2'])))
    return root, valid


def _recompute(ev, p, original, z, root_valid):
    """Stock arithmetic at supplied z; original-invalid rows are never rescued."""
    state = {key: value.copy() for key, value in original.items()}
    timing = _timing(ev, p, z)
    root, _ = _root(ev, p, z)
    geometry = original['objective_valid'] & timing['row_valid'] & np.all(~ev.mask | root_valid, axis=1)
    dt = timing['sensor_time']-p[:, 5, None]-ev.flight[:, None]*timing['tof_positive']
    signed = ev.flight[:, None]*timing['signed_doca_positive']
    query = ev.mask & geometry[:, None] & np.isfinite(dt)
    radius, sigma, cdrift = (np.zeros_like(ev.u) for _ in range(3))
    bins = {key: np.full(ev.mask.shape, -1, np.int64) for key in
            ('time_bin', 'phi_bin_lower', 'phi_bin_upper', 'offset_bin', 'signed_error_bin')}
    if np.any(query):
        info = drift_info(ev.model, dt[query], timing['phi'][query])
        radius[query], sigma[query], cdrift[query] = info['r_drift'], info['signed_error'], info['c_drift']
        folded = np.abs(np.fmod(info['used_phi'], np.pi))
        folded = np.where(folded > np.pi/2, np.pi-folded, folded)
        bins['time_bin'][query] = _index(dt[query], 0., ev.model.delta_t, ev.model.times.size-2)
        bins['phi_bin_lower'][query] = np.floor(folded/ev.model.delta_phi).astype(np.intp)
        bins['phi_bin_upper'][query] = np.minimum(np.ceil(folded/ev.model.delta_phi).astype(np.intp),
                                                ev.model.times_dbins.shape[1]-1)
        for name, cal, value in (('offset', ev.model.offset, info['c_drift']),
                                 ('signed_error', ev.model.signed, info['r_drift'])):
            bins[name+'_bin'][query] = _index(value, cal.low, cal.width, cal.slope.size-1)
    good = query & np.isfinite(radius) & np.isfinite(sigma) & (sigma > 0)
    dr = ev.lr*radius-signed
    pull = dr/np.where(good, sigma, 1.)
    long_pull = (root['u']-ev.u)/ev.sigma_u
    residual = np.where(ev.mask[..., None], np.stack((pull, long_pull), axis=-1), 0.)
    objective = .5*np.sum(residual**2, axis=(1, 2))
    row = geometry & np.isfinite(p).all(axis=1) & np.all(~ev.mask | good, axis=1) & np.isfinite(objective)
    finite = ev.mask & root['finite'] & np.isfinite(residual).all(axis=-1)
    updated = dict(objective_valid=row, objective=np.where(row, objective, np.nan),
                   residuals=np.where(finite[..., None], residual, 0.),
                   valid=finite & root_valid & timing['valid'] & good, **bins)
    for key, value in (('doca', timing['doca']), ('signed_doca', signed), ('phi', timing['phi']),
                       ('delta_t', dt), ('r_drift', radius), ('signed_error', sigma),
                       ('c_drift', cdrift), ('drift_residual', dr), ('pull', pull),
                       ('long_pull', long_pull), ('wire_u', root['u']), ('poca_z', z)):
        updated[key] = np.where(finite & np.isfinite(value), value, 0.)
    for key, value in updated.items():
        use = original['objective_valid'].reshape((ev.batch,)+(1,)*(value.ndim-1))
        state[key] = np.where(use, value, state[key])
    return state


class _PolishedEvaluator:
    """Delegating certificate evaluator; rounds=0 is a reconstruction identity test.

    Only 0 (test) and 3 (diagnostic) rounds are allowed. Each accepted root has
    finite positive Hessian, stock stationarity, strict |g/h| improvement, valid
    timing and finite-wire interior. Displacement stays within the stock basin
    AND four original corrections plus coordinate-roundoff slack. This is local
    root evidence, not a proof of basin uniqueness or arbitrarily accurate roots.
    """
    def __init__(self, evaluator, *, rounds=3):
        if rounds not in (0, 3):
            raise ValueError('polishing rounds must be 0 or 3')
        self.original, self.rounds = evaluator, rounds

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, parameters):
        return self.evaluate(parameters)[0]

    def evaluate(self, parameters, original=None):
        ev = self.original
        p = _real('parameters', parameters, (ev.batch, 6))
        if original is None:
            original = ev(p) if ev.batch and ev.hits else _empty_state(ev)
        hit = ev.mask & original['valid'] & original['objective_valid'][:, None]
        with np.errstate(all='ignore'):
            z0 = original['poca_z']
            z = z0.copy()
            root, valid = _root(ev, p, z)
            correction = root['correction'].copy()
            coord = np.maximum(1., np.maximum(np.abs(z0), np.max(np.abs(ev.mid), axis=-1)))
            neighborhood = 4*(np.abs(correction)+64*_EPS*coord)
            count = np.zeros(ev.mask.shape, np.int32)
            for _ in range(self.rounds):
                proposed = np.where(hit & valid, z-root['correction'], z)
                trial, trial_valid = _root(ev, p, proposed)
                take = (hit & valid & trial_valid & (proposed != z)
                        & (np.abs(proposed-z0) <= neighborhood)
                        & (np.abs(trial['correction']) < np.abs(root['correction']))
                        & _timing(ev, p, proposed)['valid'])
                z = np.where(take, proposed, z)
                count += take
                root, valid = _root(ev, p, z)
            state = _recompute(ev, p, original, z, valid) if ev.batch and ev.hits else original.copy()
            root_valid = hit & valid & _timing(ev, p, z)['valid']
            roundoff = 64*_EPS*coord*np.maximum(1., np.sqrt(root['speed2']))/np.where(root['positive'], root['h'], 1.)
            evidence = dict(poca_shift=np.where(hit, z-z0, np.nan),
                            original_correction=np.where(hit, correction, np.nan),
                            polished_correction=np.where(root_valid, root['correction'], np.nan),
                            polished_root_valid=root_valid, polish_changed=hit & (z != z0),
                            polish_rounds_accepted=count,
                            polish_roundoff_scale=np.where(root_valid, roundoff, np.nan),
                            polish_arithmetic_floor=root_valid & ((z-root['correction'] == z)
                                                                 | (np.abs(root['correction']) <= roundoff)))
        return state, evidence


def _inventory(ev, parameters, state, lin):
    """Mirror linearize's arithmetic guards, including absolute time/phi scales."""
    hit = ev.mask & state['valid'] & state['objective_valid'][:, None]
    z = np.where(hit, state['poca_z'], ev.zref[:, None])
    p = np.where(np.isfinite(parameters), parameters, 0.)
    with np.errstate(all='ignore'):
        point, v, second = _trajectory(p[:, :5], ev.zref, z)
        w = ev.direction
        s = np.where((ev.end == 0)[..., None], w, -w)
        delta = point-ev.mid
        r = delta-np.sum(delta*w, axis=-1)[..., None]*w
        tp = v-np.sum(v*w, axis=-1)[..., None]*w
        curvature = np.sum(r*second, axis=-1)
        h = np.sum(tp*tp, axis=-1)+curvature
        positive = h > 128*_EPS*(np.sum(v*v, axis=-1)+np.abs(curvature))
        late = np.where((ev.end == 0)[..., None], ev.hv, ev.cal)
        sensor = late+np.sum((point-late)*s, axis=-1)[..., None]*s
        q = point-sensor
        b = np.array([0., 0., np.sign(ev.bz)])
        qr, qb = np.sum(q*np.cross(s, b), axis=-1), np.sum(q*b, axis=-1)
        coord = np.maximum(1., np.maximum(np.max(np.abs(point), axis=-1), np.max(np.abs(sensor), axis=-1)))
        angle = analytic._near(qr, 0., coord) | analytic._near(qb, 0., coord)
        doca = state['doca'] <= np.maximum(1e-9, 64*_EPS*coord)
        endpoint = np.maximum(np.max(np.abs(ev.cal), axis=-1), np.max(np.abs(ev.hv), axis=-1))
        k = ev.mass/(.299792458*abs(ev.bz))
        path = np.sqrt(1+p[:, 2]**2+p[:, 3]**2+(k*p[:, 4])**2)
        time_scale = np.maximum(np.abs(ev.te), np.abs(p[:, 5, None]))
        time_scale = np.maximum(time_scale, np.maximum(coord, endpoint)/ev.speed)
        time_scale = np.maximum(time_scale, np.abs(z-ev.zref[:, None])*path[:, None]/299.792458)
        size = np.sqrt(qr*qr+qb*qb)
        flags = {name: np.zeros(ev.mask.shape, bool) for name in _BOUNDARIES}
        finite = np.zeros(ev.mask.shape, bool)
        if np.any(hit):
            cp = analytic.calibration_partials(ev.model, state['delta_t'][hit], state['phi'][hit],
                                              time_scale=time_scale[hit], phi_scale=(coord/np.where(size > 0, size, 1.))[hit])
            for name in _BOUNDARIES:
                flags[name][hit] = cp[name]
            finite[hit] = np.logical_and.reduce([np.isfinite(cp[key]) for key in
                                                ('dc_dt', 'dc_dphi', 'dr_dt', 'dr_dphi', 'dsigma_dt', 'dsigma_dphi')])
        explained = hit & positive & finite & ~angle & ~doca & lin['numeric'][:, None]
        for flag in flags.values():
            explained &= ~flag
        if not np.array_equal(explained, lin['differentiable_hit']):
            raise AssertionError('boundary inventory disagrees with analytic differentiability')
        flags.update(boundary_valid=hit, positive_hessian=hit & positive,
                     angle_branch=hit & angle, doca_branch=hit & doca,
                     calibration_finite=finite, differentiable=explained)
        cal = ev.model.offset
        for name in ('offset_edge', 'offset_edge_distance', 'offset_edge_jump', 'offset_edge_slope_jump'):
            flags[name] = np.full(ev.mask.shape, np.nan)
        flags['offset_edge_nonidentical'] = np.zeros(ev.mask.shape, bool)
        if cal.slope.size > 1:
            index = np.clip(np.rint((state['c_drift']-cal.low)/cal.width), 1, cal.slope.size-1).astype(np.intp)
            edge = cal.low+index*cal.width
            jump = (cal.intercept[index]+cal.slope[index]*edge)-(cal.intercept[index-1]+cal.slope[index-1]*edge)
            for name, value in (('offset_edge', edge), ('offset_edge_distance', np.abs(state['c_drift']-edge)),
                                ('offset_edge_jump', jump), ('offset_edge_slope_jump', cal.slope[index]-cal.slope[index-1])):
                flags[name] = np.where(hit, value, np.nan)
            flags['offset_edge_nonidentical'] = hit & ((cal.intercept[index] != cal.intercept[index-1])
                                                      | (cal.slope[index] != cal.slope[index-1]))
    return flags


def _outputs(ev, p, state, prefix):
    lin = analytic.linearize(ev, p, state)
    out = {prefix+'_'+key: value for key, value in certify(ev, p, state, lin).items()}
    row = state['objective_valid']
    hit = ev.mask & state['valid'] & row[:, None]
    for key in ('gradient_norm', 'rank', 'condition'):
        out[prefix+'_'+key] = np.where(row & lin['numeric'], lin[key], np.nan)
        out[prefix+'_raw_'+key] = lin[key].copy()
    for key in ('numeric', 'differentiable'):
        out[prefix+'_'+key] = lin[key].copy()
    out[prefix+'_linearization_good'] = lin['good'].copy()
    for key, value in state.items():
        if value.ndim == 1:
            out[prefix+'_'+key] = value.copy()
        elif key == 'valid':
            out['hit_'+prefix+'_'+key] = hit.copy()
        elif key == 'residuals':
            out['hit_'+prefix+'_'+key] = np.where(row[:, None, None], value, np.nan)
        else:
            out['hit_'+prefix+'_'+key] = np.where(hit, value, -1 if value.dtype.kind in 'iu' else np.nan)
    return out, lin


def diagnose(evaluator, parameters):
    """Return bounded, array-exact batch evidence for saved parameters [B,6].

    All arrays have leading dimension B; hit arrays start ``hit_``. Original and
    polished row outputs use ``original_``/``polished_``: objective/validity,
    analytic gradient_norm/rank/condition/numeric/differentiable/linearization_good,
    and EVERY unchanged certify() field (including [B,4,6] column evidence).
    State hit arrays use ``hit_original_``/``hit_polished_``. Original inventory
    uses ``hit_time_boundary`` etc.; polished inventory adds ``polished_``.
    ``center_same_branch`` requires native/regression/signed-DOCA AND unfolded
    angle identity. Offset jump is right-minus-left OFFSET at the nearest
    internal edge (radius jump has opposite sign); equal coefficients are not
    boundaries. No internal edge means NaN distance/jump.

    Missing floats are NaN, bins -1 and booleans false. Masked residuals are
    zero in valid rows; all residuals are NaN in invalid rows. Original mask,
    LR, scales and supplied parameters are copied unchanged, even if invalid.
    ``hit_polish_roundoff_scale`` is an arithmetic scale, NOT an error bound;
    arithmetic-floor flags do not assert exact roots. Polished certificates
    describe only this counterfactual, never a repaired production fit.
    """
    ev = evaluator
    p = _real('parameters', parameters, (ev.batch, 6))
    with np.errstate(all='ignore'):
        original = ev(p) if ev.batch and ev.hits else _empty_state(ev)
        wrapper = _PolishedEvaluator(ev)
        polished, evidence = wrapper.evaluate(p, original)
        out, lin = _outputs(ev, p, original, 'original')
        changed, plin = _outputs(wrapper, p, polished, 'polished')
        out.update(changed)
        out.update({'hit_'+key: value for key, value in _inventory(ev, p, original, lin).items()})
        out.update({'hit_polished_'+key: value for key, value in _inventory(ev, p, polished, plin).items()})
        out.update({'hit_'+key: value for key, value in evidence.items()})
        same = _same_branch(original, polished, ev.mask, ev.model)
        same &= np.all(~ev.mask[..., None] | (_angle_branch(ev, p, original) == _angle_branch(ev, p, polished)), axis=(1, 2))
        out['center_same_branch'] = original['objective_valid'] & polished['objective_valid'] & same
    out.update(original_parameters=p.copy(), parameter_scale=ev.parameter_scale.copy(),
               hit_mask=ev.valid.copy(), hit_initial_lr=ev.initial_lr.copy())
    return out
