"""Bounded NumPy wire-only helix initializer; NOT a drift fit or KinKal.

Fixed original 3D wires and measured along-wire coordinates are the only hit
inputs. No association, TOT radius, phi=0 drift table, truth, material, timing,
charge, covariance or statistical chi-square is used here.
"""
import numpy as np


_EPS = np.finfo(float).eps
_MAX = np.finfo(float).max


def _real(name, value, shape=None):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu" or (shape is not None and a.shape != shape):
        raise ValueError(f"{name} must be real numeric with shape {shape}")
    return a.astype(float, copy=False)


def _trajectory(p, z_ref, z):
    """Position, dz tangent and dz curvature, stable at exactly omega=0."""
    dz = z-z_ref[:, None]
    w = p[:, 4, None]
    a = w*dz
    sinc = np.sinc(a/np.pi)
    cosc = .5*a*np.sinc(a/(2*np.pi))**2
    tx, ty = p[:, 2, None], p[:, 3, None]
    x = p[:, 0, None] + dz*(tx*sinc-ty*cosc)
    y = p[:, 1, None] + dz*(ty*sinc+tx*cosc)
    vx, vy = tx*np.cos(a)-ty*np.sin(a), tx*np.sin(a)+ty*np.cos(a)
    one = np.ones_like(z)
    return (np.stack((x, y, z), axis=-1),
            np.stack((vx, vy, one), axis=-1),
            np.stack((-w*vy, w*vx, np.zeros_like(z)), axis=-1))


def _at_z(p, z_ref, z, mid, direction):
    point, tangent, second = _trajectory(p, z_ref, z)
    delta = point-mid
    u = np.sum(delta*direction, axis=-1)
    r = delta-u[..., None]*direction
    udot = np.sum(tangent*direction, axis=-1)
    tp = tangent-udot[..., None]*direction
    speed2 = np.sum(tp*tp, axis=-1)
    curvature = np.sum(r*second, axis=-1)
    h = speed2+curvature
    g = np.sum(r*tp, axis=-1)
    f = .5*np.sum(r*r, axis=-1)
    hscale = np.sum(tangent*tangent, axis=-1)+np.abs(curvature)
    finite = (np.isfinite(f) & np.isfinite(g) & np.isfinite(h)
              & np.isfinite(hscale) & np.isfinite(u) & np.isfinite(z))
    positive = finite & (h > 128*_EPS*hscale)
    correction = g/np.where(positive, h, 1.)
    # Include absolute coordinate arithmetic, without a large-radius subtraction.
    arithmetic = np.maximum(1., np.maximum(np.max(np.abs(point), axis=-1),
                                          np.max(np.abs(mid), axis=-1)))
    slack = 64*_EPS*arithmetic*(np.sqrt(2*f)+_EPS*arithmetic)
    return dict(r=r, u=u, tp=tp, udot=udot, speed2=speed2, h=h, g=g, f=f,
                finite=finite, positive=positive, correction=correction, slack=slack)


def _poca(p, z_ref, mid, direction, measured_u, hlen, mask, *,
          iterations, tolerance, max_step, max_offset):
    z0 = mid[..., 2]+measured_u*direction[..., 2]
    omega = np.abs(p[:, 4, None])
    inverse = 1./np.maximum(omega, np.finfo(float).tiny)
    basin = np.minimum(max_offset, (np.pi/2)*np.minimum(inverse, _MAX/2))
    limit = np.minimum(max_step, (np.pi/8)*inverse)
    offset = np.zeros_like(z0)
    count = np.zeros(z0.shape, np.int32)

    def evaluate(t):
        s = _at_z(p, z_ref, z0+t, mid, direction)
        s['stationary'] = (s['positive'] & (np.abs(s['correction']) <= tolerance)
                           & (np.abs(s['g']) <= tolerance*np.maximum(1., s['speed2'])))
        return s

    state = evaluate(offset)
    active = mask & state['positive'] & ~state['stationary']
    for _ in range(iterations):
        if not np.any(active):
            break
        count += active
        step = np.clip(-state['correction'], -limit, limit)
        step = np.clip(step, -basin-offset, basin-offset)
        accepted = np.zeros(mask.shape, bool)
        next_offset = offset.copy()
        for _ in range(10):
            trial_offset = offset+np.where(active, step, 0.)
            trial = evaluate(trial_offset)
            take = (active & ~accepted & trial['positive']
                    & (trial['f'] <= state['f']+1e-4*state['g']*step
                       + state['slack']+trial['slack'])
                    & (trial_offset != offset) & (state['g']*step < 0))
            next_offset = np.where(take, trial_offset, next_offset)
            accepted |= take
            if np.all(~active | accepted):
                break
            step *= .5
        offset = next_offset
        state = evaluate(offset)
        active &= accepted & state['positive'] & ~state['stationary']
    state.update(z=z0+offset, iterations=count,
                 valid=mask & state['finite'],
                 boundary=mask & (np.abs(offset) >= basin),
                 converged=mask & state['stationary'] & (np.abs(offset) < basin),
                 inside=mask & state['finite'] & (np.abs(state['u']) <= hlen))
    return state


def _residual(state, measured_u, sigma, transverse_scale, mask):
    r = np.concatenate((state['r']/transverse_scale,
                        ((state['u']-measured_u)/sigma)[..., None]), axis=-1)
    return np.where(mask[..., None], r, 0.)


def _linearize(p, scale, z_ref, state, mid, direction, sigma, mask,
               residual, transverse_scale, fd_step, condition_limit):
    # Implicit-POCA derivatives use central differences in five fixed parameter directions.
    jac = np.empty(residual.shape+(5,))
    for j in range(5):
        step = np.zeros_like(p)
        step[:, j] = fd_step*scale[:, j]
        plus = _at_z(p+step, z_ref, state['z'], mid, direction)
        minus = _at_z(p-step, z_ref, state['z'], mid, direction)
        dz = -(plus['g']-minus['g'])/(2*fd_step*np.where(state['positive'], state['h'], 1.))
        dr = (plus['r']-minus['r'])/(2*fd_step)+state['tp']*dz[..., None]
        du = (plus['u']-minus['u'])/(2*fd_step)+state['udot']*dz
        jac[..., :3, j] = dr/transverse_scale
        jac[..., 3, j] = du/sigma
    jac = np.where(mask[..., None, None], jac, 0.)
    numeric = np.isfinite(jac).all(axis=(1, 2, 3))
    jac = np.where(numeric[:, None, None, None], jac, 0.)
    # Flattening each row fixes the hit reduction order across batch sizes.
    flat = jac.reshape(len(p), residual.shape[1]*4, 5)
    normal = np.einsum('bhi,bhj->bij', flat, flat, optimize=False)
    gradient = np.einsum('bhi,bh->bi', flat, residual.reshape(len(p), -1), optimize=False)
    norm = np.sqrt(np.maximum(np.diagonal(normal, axis1=1, axis2=2), 0.))
    nonzero = np.all(norm > np.finfo(float).tiny, axis=1)
    norm = np.where(norm > np.finfo(float).tiny, norm, 1.)
    correlation = normal/norm[:, :, None]/norm[:, None, :]
    scaled_gradient = gradient/norm
    numeric &= (np.isfinite(correlation).all(axis=(1, 2))
                & np.isfinite(scaled_gradient).all(axis=1))
    correlation = np.where(numeric[:, None, None], correlation, np.eye(5))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    lo, hi = eigenvalues[:, 0], eigenvalues[:, -1]
    condition = np.divide(hi, lo, out=np.full(len(p), _MAX), where=lo > 0)
    condition = np.minimum(np.nan_to_num(condition, nan=_MAX, posinf=_MAX), _MAX)
    rank = np.sum(eigenvalues > 128*_EPS*np.maximum(hi, 0.)[:, None], axis=1)
    good = numeric & nonzero & (rank == 5) & (condition <= condition_limit)
    condition = np.where(numeric & nonzero, condition, _MAX)
    gn = np.where(numeric[:, None], scaled_gradient, 0.)
    return dict(good=good, condition=condition, rank=np.where(nonzero & numeric, rank, 0),
                eigenvalues=np.where(numeric[:, None], eigenvalues, 0.),
                eigenvectors=eigenvectors, norm=np.where(numeric[:, None], norm, 1.), g=gn,
                gradient_norm=np.where(numeric, np.max(np.abs(gn), axis=1), _MAX))


def fit_wire_helix(seed, mid, direction, measured_u, uvar, hlen, valid, *, bz,
                   max_iterations=40, transverse_scale=2.5, condition_limit=1e12,
                   gradient_tolerance=1e-7, fd_step=1e-4, initial_damping=1e-3,
                   max_parameter_step=10., poca_max_iterations=20,
                   poca_tolerance=1e-7, poca_max_step=50., poca_max_offset=500.,
                   max_rows=2048, max_cells=262144):
    """Refine fixed-membership wire geometry, preserving every batch lane.

    ``mid,direction`` are real [batch,hits,3]; ``measured_u,uvar,hlen`` are
    [batch,hits], in mm/mm2. ``valid`` is a boolean mask of exactly that shape.
    Directions are normalized in full 3D. Masked payloads are ignored, even
    NaNs; ANY selected nonfinite/nonphysical hit fails its entire lane. There
    is no dynamic mask, reassociation, robust weighting or outlier rejection.
    Finite lengths require |predicted wire_u| <= hlen; endpoints are NOT fitted.

    ``seed`` accepts fit_helix_seed's arrays [batch], center_xy [batch,2], and
    its success flag. Alternatively pass ``local_parameters`` [batch,5],
    ``z_ref`` [batch], and optional boolean ``success`` [batch]. The latter
    permits exactly straight initializers. Local parameters are
    (x0,y0,tx,ty,omega=dphase/dz) at FIXED seed z_ref, with
    x=x0+dz*(tx*sinc(omega*dz)-ty*cosc(omega*dz)),
    y=y0+dz*(ty*sinc(omega*dz)+tx*cosc(omega*dz)); sinc(t)=sin(t)/t,
    cosc(t)=(1-cos(t))/t. These are well defined at zero curvature.

    For each hit a bounded positive-Hessian Newton search finds a LOCAL
    infinite-line POCA starting at the measured wire point's z. Its basin is
    min(poca_max_offset, pi/(2*abs(omega))); no global POCA claim is made.
    Every selected hit must have finite, converged, interior POCA and lie
    within the finite wire, initially and on EVERY accepted LM trial.
    A failed trial retains the last accepted iterate, never silently marks
    the original seed successful. At least four selected hits are required.

    Objective = 0.5*sum(|perpendicular_3D_residual|**2/transverse_scale**2
    + ((predicted_u-measured_u)/sqrt(uvar))**2). Three perpendicular components
    are redundant (projection and optimized z remove dimensions); this score
    is NOT chi-square, a physical drift fit, or a source of covariance/errors.
    The default 2.5 mm scale and all thresholds are fixed engineering choices.

    Batched diagonally equilibrated LM uses an implicit-POCA finite-difference
    Jacobian, safe eigenvalue solves, and bounded scaled parameter updates.
    Only an unconstrained small gradient plus full numerical rank/conditioning
    and all-hit POCA validity constitutes convergence. Clipped/boundary stalls,
    damping exhaustion and iteration exhaustion are failures. max_iterations=0
    checks stationarity only, and can succeed for an already stationary seed.

    Work is O(batch*hits) storage (largest array [batch,hits,4,5], 160 bytes per
    cell), conservatively <2 KiB/cell of array workspace plus O(batch) matrices,
    excluding caller inputs/outputs and NumPy allocator overhead. Hard limits
    max_rows/max_cells raise rather than silently truncate; caller chunks rows.
    No per-event/per-wire loops: at most max_iterations LM trials, each one
    POCA (<=poca_max_iterations*10 line-search trials) and ten trajectory
    evaluations for the five-column Jacobian. One final Jacobian is evaluated.

    Returns arrays [batch] unless specified:
      success/converged (identical), status (string), n_hits, iterations;
      initial_objective, final_objective, objective_valid; finite diagnostics
      condition (equilibrated normal eigenvalue ratio; max-float if undefined),
      rank, eigenvalues [batch,5], gradient_norm, damping, accepted_steps;
      initial_local_parameters/local_parameters [batch,5], z_ref (last finite
      iterate even on fit failure; NaN for invalid seed); parameter_scale
      [batch,5]; physical_identifiable, center_xy [batch,2], radius, pitch,
      phase_ref, pt_abs, p_abs (NaN unless SUCCESS and identifiable).
      residuals [batch,hits,4] (normalized xyz perpendicular + longitudinal),
      doca, wire_u, wire_pull, poca_z [batch,hits], poca_valid/converged/inside/
      boundary [batch,hits], poca_iterations. Unusable/masked hit diagnostics
      are zero with false flags; initial/final objective are zero if unavailable
      (objective_valid=False), not a fabricated successful fit.

    Physical conversion requires |omega|*max(1, measured_z_span)>1e-8 and
    nonzero transverse slope. This is numerical identifiability, NOT a
    momentum-resolution criterion. Straight/near-straight local fits may
    succeed but report NO momentum. bz must be finite nonzero tesla; momenta
    are MeV/c magnitudes. Charge and flight direction remain unknown.
    """
    valid = np.asarray(valid)
    if valid.ndim != 2 or valid.dtype.kind != 'b':
        raise ValueError('valid must be boolean [batch,hits]')
    batch, hits = valid.shape
    for name, value in (("max_iterations", max_iterations),
                        ("poca_max_iterations", poca_max_iterations),
                        ("max_rows", max_rows), ("max_cells", max_cells)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not 0 <= value <= 2**31-1):
            raise ValueError(f'{name} must be a nonnegative bounded integer')
    if batch > max_rows or batch*hits > max_cells:
        raise ValueError('refit row/cell bound exceeded; chunk rows explicitly')
    for name, value in (("bz", bz), ("transverse_scale", transverse_scale),
                        ("condition_limit", condition_limit),
                        ("gradient_tolerance", gradient_tolerance), ("fd_step", fd_step),
                        ("initial_damping", initial_damping),
                        ("max_parameter_step", max_parameter_step),
                        ("poca_tolerance", poca_tolerance), ("poca_max_step", poca_max_step),
                        ("poca_max_offset", poca_max_offset)):
        a = _real(name, value)
        if a.ndim != 0 or not np.isfinite(a) or (a == 0 if name == 'bz' else a <= 0):
            raise ValueError(f'{name} must be a finite {"nonzero" if name == "bz" else "positive"} scalar')
    if condition_limit <= 1 or initial_damping > 1e12:
        raise ValueError('condition_limit must exceed one; initial_damping <= 1e12')
    mid = _real('mid', mid, (batch, hits, 3))
    direction = _real('direction', direction, mid.shape)
    measured_u, uvar, hlen = (_real(n, a, valid.shape) for n, a in
                             (('measured_u', measured_u), ('uvar', uvar), ('hlen', hlen)))
    try:
        seed_success = np.asarray(seed.get('success', np.ones(batch, bool)))
        if seed_success.shape != (batch,) or seed_success.dtype.kind != 'b':
            raise ValueError('seed success must be boolean [batch]')
        z_ref = _real('z_ref', seed['z_ref'], (batch,)).copy()
        if 'local_parameters' in seed:
            p = _real('local_parameters', seed['local_parameters'], (batch, 5)).copy()
        else:
            center = _real('center_xy', seed['center_xy'], (batch, 2))
            radius, pitch, phase = (_real(n, seed[n], (batch,))
                                    for n in ('radius', 'pitch', 'phase_ref'))
            with np.errstate(all='ignore'):
                p = np.column_stack((center[:, 0]+radius*np.cos(phase),
                                     center[:, 1]+radius*np.sin(phase),
                                     -radius/pitch*np.sin(phase),
                                     radius/pitch*np.cos(phase), 1./pitch))
            seed_success = seed_success & (radius > 0) & np.isfinite(pitch) & (pitch != 0)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('seed must contain helix seed or local parameter arrays') from exc
    seed_ok = seed_success & np.isfinite(p).all(axis=1) & np.isfinite(z_ref)
    initial_p = np.where(seed_ok[:, None], p, np.nan)
    out = dict(success=np.zeros(batch, bool), converged=np.zeros(batch, bool),
               status=np.full(batch, 'max_iterations', dtype='<U24'), n_hits=valid.sum(axis=1),
               iterations=np.zeros(batch, np.int32), accepted_steps=np.zeros(batch, np.int32),
               initial_objective=np.zeros(batch), final_objective=np.zeros(batch),
               objective_valid=np.zeros(batch, bool), condition=np.full(batch, _MAX),
               rank=np.zeros(batch, np.int32), eigenvalues=np.zeros((batch, 5)),
               gradient_norm=np.zeros(batch), damping=np.full(batch, float(initial_damping)),
               initial_local_parameters=initial_p, local_parameters=initial_p.copy(),
               z_ref=np.where(seed_ok, z_ref, np.nan), parameter_scale=np.ones((batch, 5)),
               physical_identifiable=np.zeros(batch, bool), center_xy=np.full((batch, 2), np.nan))
    for name in ('radius', 'pitch', 'phase_ref', 'pt_abs', 'p_abs'):
        out[name] = np.full(batch, np.nan)
    out['residuals'] = np.zeros((batch, hits, 4))
    for name in ('doca', 'wire_u', 'wire_pull', 'poca_z'):
        out[name] = np.zeros(valid.shape)
    for name in ('poca_valid', 'poca_converged', 'poca_inside', 'poca_boundary'):
        out[name] = np.zeros(valid.shape, bool)
    out['poca_iterations'] = np.zeros(valid.shape, np.int32)
    out['status'][~seed_ok] = 'invalid_seed'
    out['status'][out['n_hits'] < 4] = 'insufficient_hits'
    out['status'][out['n_hits'] == 0] = 'empty'
    if batch == 0 or hits == 0:
        return out
    # Keep malformed selected hits as lane failures, not a new membership mask.
    dscale = np.max(np.abs(direction), axis=-1)
    hit_ok = (np.isfinite(mid).all(axis=-1) & np.isfinite(direction).all(axis=-1)
              & (dscale > 0) & np.isfinite(measured_u) & np.isfinite(uvar) & (uvar > 0)
              & np.isfinite(hlen) & (hlen >= 0))
    input_ok = np.all(~valid | hit_ok, axis=1)
    eligible = seed_ok & input_ok & (out['n_hits'] >= 4)
    mask = valid & eligible[:, None]
    p = np.where(seed_ok[:, None], p, 0.)
    z_ref = np.where(seed_ok, z_ref, 0.)
    mid = np.where(mask[..., None], mid, 0.)
    direction = np.where(mask[..., None], direction, np.array([1., 0., 0.]))
    dscale = np.max(np.abs(direction), axis=-1)
    direction = direction/dscale[..., None]
    direction /= np.sqrt(np.sum(direction**2, axis=-1))[..., None]
    measured_u = np.where(mask, measured_u, 0.)
    sigma = np.sqrt(np.where(mask, uvar, 1.))
    hlen = np.where(mask, hlen, 0.)
    out['status'][~input_ok & (out['n_hits'] > 0)] = 'invalid_input'
    options = dict(iterations=poca_max_iterations, tolerance=poca_tolerance,
                   max_step=poca_max_step, max_offset=poca_max_offset)
    # Derived overflow is also a lane failure. No NaN matrix reaches eigh.
    with np.errstate(all='ignore'):
        measured_z = mid[..., 2]+measured_u*direction[..., 2]
        span = np.max(np.where(mask, measured_z, -np.inf), axis=1)-np.min(
            np.where(mask, measured_z, np.inf), axis=1)
        length = np.where(eligible & np.isfinite(span), np.maximum(1., span), 1.)
        slope = np.hypot(p[:, 2], p[:, 3])
        scale = np.column_stack((np.full(batch, transverse_scale), np.full(batch, transverse_scale),
                                 transverse_scale/length, transverse_scale/length,
                                 transverse_scale/length/length/np.maximum(slope, transverse_scale/length)))
        scale_ok = np.isfinite(scale).all(axis=1) & (scale > 0).all(axis=1)
        scale = np.where(scale_ok[:, None], scale, 1.)
        out['parameter_scale'] = scale
        state = _poca(p, z_ref, mid, direction, measured_u, hlen, mask, **options)
        residual = _residual(state, measured_u, sigma, transverse_scale, mask)
        objective = .5*np.sum(residual**2, axis=(1, 2))
        all_poca = np.all(~valid | (state['valid'] & state['converged'] & state['inside']), axis=1)
        good = eligible & scale_ok & all_poca & np.isfinite(objective)
        out['objective_valid'] = good.copy()
        out['initial_objective'] = np.where(good, objective, 0.)
        out['status'][eligible & ~good] = 'initial_poca_failed'
        active = good.copy()
        for _ in range(max_iterations):
            if not np.any(active):
                break
            lin = _linearize(p, scale, z_ref, state, mid, direction, sigma, mask,
                             residual, transverse_scale, fd_step, condition_limit)
            stationary = lin['gradient_norm'] <= gradient_tolerance*(1+np.sqrt(2*objective))
            out['status'][active & ~lin['good']] = 'ill_conditioned'
            out['converged'] |= active & lin['good'] & stationary
            active &= lin['good'] & ~stationary
            if not np.any(active):
                break
            out['iterations'] += active
            v = lin['eigenvectors']
            projection = np.einsum('bji,bj->bi', v, lin['g'], optimize=False)
            denom = np.maximum(lin['eigenvalues'], 0.)+out['damping'][:, None]
            step = -np.einsum('bij,bj->bi', v, projection/denom, optimize=False)/lin['norm']
            step /= np.maximum(1., np.max(np.abs(step), axis=1)/max_parameter_step)[:, None]
            trial_p = p+np.where(active[:, None], scale*step, 0.)
            trial = _poca(trial_p, z_ref, mid, direction, measured_u, hlen, mask, **options)
            trial_r = _residual(trial, measured_u, sigma, transverse_scale, mask)
            trial_f = .5*np.sum(trial_r**2, axis=(1, 2))
            trial_ok = np.all(~valid | (trial['valid'] & trial['converged'] & trial['inside']), axis=1)
            take = active & trial_ok & np.isfinite(trial_f) & (trial_f < objective)
            p = np.where(take[:, None], trial_p, p)
            residual = np.where(take[:, None, None], trial_r, residual)
            objective = np.where(take, trial_f, objective)
            for key in state:
                expand = take.reshape((batch,)+(1,)*(state[key].ndim-1))
                state[key] = np.where(expand, trial[key], state[key])
            out['accepted_steps'] += take
            out['damping'] = np.where(take, np.maximum(1e-12, out['damping']/3),
                                      np.where(active, np.minimum(1e13, out['damping']*10), out['damping']))
            stalled = active & (out['damping'] > 1e12)
            out['status'][stalled] = 'stalled'
            active &= ~stalled
        lin = _linearize(p, scale, z_ref, state, mid, direction, sigma, mask,
                         residual, transverse_scale, fd_step, condition_limit)
        stationary = lin['gradient_norm'] <= gradient_tolerance*(1+np.sqrt(2*objective))
        # Check the final unconstrained gradient, including a last-budget improving trial.
        out['converged'] = good & lin['good'] & stationary
        out['success'] = out['converged'].copy()
        out['status'][good & ~lin['good']] = 'ill_conditioned'
        out['status'][out['success']] = 'converged'
        out['condition'] = np.where(good, lin['condition'], _MAX)
        out['rank'] = np.where(good, lin['rank'], 0)
        out['eigenvalues'] = np.where(good[:, None], lin['eigenvalues'], 0.)
        out['gradient_norm'] = np.where(good, lin['gradient_norm'], 0.)
        out['final_objective'] = np.where(good, objective, 0.)
        out['local_parameters'] = np.where(seed_ok[:, None], p, np.nan)
        hit_finite = state['valid'] & np.isfinite(residual).all(axis=-1)
        out['residuals'] = np.where(hit_finite[..., None], residual, 0.)
        for name, value in (('doca', np.sqrt(2*state['f'])), ('wire_u', state['u']),
                            ('wire_pull', residual[..., 3]), ('poca_z', state['z'])):
            out[name] = np.where(hit_finite, value, 0.)
        for name in ('valid', 'converged', 'inside', 'boundary'):
            out['poca_'+name] = state[name] & hit_finite
        out['poca_iterations'] = state['iterations']
        omega = p[:, 4]
        transverse = np.hypot(p[:, 2], p[:, 3])
        physical = out['success'] & (np.abs(omega)*length > 1e-8) & (transverse > 128*_EPS)
        inverse = 1./np.where(physical, omega, 1.)
        center = p[:, :2]+np.column_stack((-p[:, 3]*inverse, p[:, 2]*inverse))
        radius = transverse*np.abs(inverse)
        pitch = inverse
        phase = np.arctan2(-p[:, 2]*inverse, p[:, 3]*inverse)
        values = dict(radius=radius, pitch=pitch, phase_ref=phase,
                      pt_abs=.299792458*abs(bz)*radius,
                      p_abs=.299792458*abs(bz)*np.hypot(radius, pitch))
        physical &= np.isfinite(center).all(axis=1)
        for value in values.values():
            physical &= np.isfinite(value)
        out['physical_identifiable'] = physical
        out['center_xy'] = np.where(physical[:, None], center, np.nan)
        for name, value in values.items():
            out[name] = np.where(physical, value, np.nan)
    return out
