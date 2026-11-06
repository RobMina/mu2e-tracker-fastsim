"""Device-only fixed-membership wire helix initializer, NOT CentralHelix/KinKal.

No drift, timing, charge, flight direction, momentum, covariance or material
inference. The fixed engineering objective is not statistical chi-square.
"""
import math
import numbers

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)
_EPS, _TINY, _MAX = 2.220446049250313e-16, 2.2250738585072014e-308, 1.7976931348623157e308
STATUS_NAMES = ("converged", "max_iterations", "invalid_seed", "insufficient_hits",
                "empty", "invalid_input", "initial_poca_failed", "ill_conditioned", "stalled")
CONVERGED, MAX_ITERATIONS, INVALID_SEED, INSUFFICIENT_HITS, EMPTY = range(5)
INVALID_INPUT, INITIAL_POCA_FAILED, ILL_CONDITIONED, STALLED = range(5, 9)


def _real(name, value, shape):
    a = jnp.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric with shape {shape}")
    return a.astype(jnp.float64)


def _trajectory(p, z_ref, z):
    dz, w = z-z_ref[:, None], p[:, 4, None]
    a = w*dz
    sinc, cosc = jnp.sinc(a/jnp.pi), .5*a*jnp.sinc(a/(2*jnp.pi))**2
    tx, ty = p[:, 2, None], p[:, 3, None]
    x = p[:, 0, None]+dz*(tx*sinc-ty*cosc)
    y = p[:, 1, None]+dz*(ty*sinc+tx*cosc)
    vx, vy = tx*jnp.cos(a)-ty*jnp.sin(a), tx*jnp.sin(a)+ty*jnp.cos(a)
    return (jnp.stack((x, y, z), -1), jnp.stack((vx, vy, jnp.ones_like(z)), -1),
            jnp.stack((-w*vy, w*vx, jnp.zeros_like(z)), -1))


def _at_z(p, z_ref, z, mid, direction):
    point, tangent, second = _trajectory(p, z_ref, z)
    delta = point-mid
    u = jnp.sum(delta*direction, -1)
    r = delta-u[..., None]*direction
    udot = jnp.sum(tangent*direction, -1)
    tp = tangent-udot[..., None]*direction
    speed2, curvature = jnp.sum(tp*tp, -1), jnp.sum(r*second, -1)
    h, g, f = speed2+curvature, jnp.sum(r*tp, -1), .5*jnp.sum(r*r, -1)
    hscale = jnp.sum(tangent*tangent, -1)+jnp.abs(curvature)
    finite = (jnp.isfinite(f) & jnp.isfinite(g) & jnp.isfinite(h)
              & jnp.isfinite(hscale) & jnp.isfinite(u) & jnp.isfinite(z))
    arithmetic = jnp.maximum(1., jnp.maximum(jnp.max(jnp.abs(point), -1), jnp.max(jnp.abs(mid), -1)))
    slack = 64*_EPS*arithmetic*(jnp.sqrt(2.)*jnp.sqrt(f)+_EPS*arithmetic)
    finite &= jnp.isfinite(slack)
    positive = finite & (h > 128*_EPS*hscale)
    return dict(r=r, u=u, tp=tp, udot=udot, speed2=speed2, h=h, g=g, f=f,
                finite=finite, positive=positive, correction=g/jnp.where(positive, h, 1.), slack=slack)


def _poca(p, z_ref, mid, direction, measured_u, hlen, mask, options):
    iterations, tolerance, max_step, max_offset = options
    z0 = mid[..., 2]+measured_u*direction[..., 2]
    inverse = 1./jnp.maximum(jnp.abs(p[:, 4, None]), _TINY)
    basin = jnp.minimum(max_offset, (jnp.pi/2)*jnp.minimum(inverse, _MAX/2))
    limit = jnp.minimum(max_step, (jnp.pi/8)*inverse)

    def evaluate(offset):
        s = _at_z(p, z_ref, z0+offset, mid, direction)
        s['stationary'] = (s['positive'] & (jnp.abs(s['correction']) <= tolerance)
                           & (jnp.abs(s['g']) <= tolerance*jnp.maximum(1., s['speed2'])))
        return s

    def update(carry):
        i, offset, count, state, active = carry
        step = jnp.clip(-state['correction'], -limit, limit)
        step = jnp.clip(step, -basin-offset, basin-offset)

        def backtrack(c):
            k, step, accepted, next_offset = c
            trial_offset = offset+jnp.where(active, step, 0.)
            trial = evaluate(trial_offset)
            take = (active & ~accepted & trial['positive'] & (trial_offset != offset)
                    & (state['g']*step < 0) & (trial['f'] <= state['f']+1e-4*state['g']*step
                                               +state['slack']+trial['slack']))
            return k+1, step*.5, accepted | take, jnp.where(take, trial_offset, next_offset)

        _, _, accepted, next_offset = lax.while_loop(
            lambda c: (c[0] < 10) & jnp.any(active & ~c[2]), backtrack,
            (jnp.int32(0), step, jnp.zeros_like(mask), offset))
        next_state = evaluate(next_offset)
        return (i+1, next_offset, count+active.astype(jnp.int32), next_state,
                active & accepted & next_state['positive'] & ~next_state['stationary'])

    offset = jnp.zeros_like(z0)
    state = evaluate(offset)
    active = mask & state['positive'] & ~state['stationary']
    _, offset, count, state, _ = lax.while_loop(
        lambda c: (c[0] < iterations) & jnp.any(c[4]), update,
        (jnp.int32(0), offset, jnp.zeros(mask.shape, jnp.int32), state, active))
    return dict(state, z=z0+offset, iterations=count, valid=mask & state['finite'],
                boundary=mask & (jnp.abs(offset) >= basin),
                converged=mask & state['stationary'] & (jnp.abs(offset) < basin),
                inside=mask & state['finite'] & (jnp.abs(state['u']) <= hlen))


def _residual(state, measured_u, sigma, transverse_scale, mask):
    r = jnp.concatenate((state['r']/transverse_scale,
                         ((state['u']-measured_u)/sigma)[..., None]), -1)
    return jnp.where(mask[..., None], r, 0.)


def _all_poca(state, valid):
    return jnp.all(~valid | (state['valid'] & state['converged'] & state['inside']), axis=1)


def _linearize(p, scale, z_ref, state, mid, direction, sigma, mask,
               residual, transverse_scale, fd_step, condition_limit):
    def column(j, jac):
        step = jnp.zeros_like(p).at[:, j].set(fd_step*scale[:, j])
        plus = _at_z(p+step, z_ref, state['z'], mid, direction)
        minus = _at_z(p-step, z_ref, state['z'], mid, direction)
        dz = -(plus['g']-minus['g'])/(2*fd_step*jnp.where(state['positive'], state['h'], 1.))
        dr = (plus['r']-minus['r'])/(2*fd_step)+state['tp']*dz[..., None]
        du = (plus['u']-minus['u'])/(2*fd_step)+state['udot']*dz
        value = jnp.concatenate((dr/transverse_scale, (du/sigma)[..., None]), -1)
        return jac.at[..., j].set(jnp.where(mask[..., None], value, 0.))

    jac = lax.fori_loop(0, 5, column, jnp.zeros(residual.shape+(5,), p.dtype))
    numeric = jnp.all(jnp.isfinite(jac), axis=(1, 2, 3)) & jnp.all(jnp.isfinite(residual), axis=(1, 2))
    flat = jnp.where(numeric[:, None, None, None], jac, 0.).reshape(p.shape[0], -1, 5)
    r = jnp.where(numeric[:, None, None], residual, 0.).reshape(p.shape[0], -1)
    normal, gradient = jnp.einsum('bhi,bhj->bij', flat, flat), jnp.einsum('bhi,bh->bi', flat, r)
    numeric &= jnp.all(jnp.isfinite(normal), axis=(1, 2)) & jnp.all(jnp.isfinite(gradient), axis=1)
    norm = jnp.sqrt(jnp.maximum(jnp.diagonal(normal, axis1=1, axis2=2), 0.))
    nonzero = jnp.all(norm > _TINY, axis=1)
    norm = jnp.where((norm > _TINY) & jnp.isfinite(norm), norm, 1.)
    correlation, g = normal/norm[:, :, None]/norm[:, None, :], gradient/norm
    numeric &= jnp.all(jnp.isfinite(correlation), axis=(1, 2)) & jnp.all(jnp.isfinite(g), axis=1)
    # Reject nonfinite normal matrices before any batched eigensolve.
    eigenvalues, eigenvectors = jnp.linalg.eigh(jnp.where(numeric[:, None, None], correlation, jnp.eye(5)))
    numeric &= jnp.all(jnp.isfinite(eigenvalues), axis=1) & jnp.all(jnp.isfinite(eigenvectors), axis=(1, 2))
    lo, hi = eigenvalues[:, 0], eigenvalues[:, -1]
    condition = jnp.minimum(jnp.where(lo > 0, hi/jnp.where(lo > 0, lo, 1.), _MAX), _MAX)
    rank = jnp.sum(eigenvalues > 128*_EPS*jnp.maximum(hi, 0.)[:, None], axis=1, dtype=jnp.int32)
    good = numeric & nonzero & (rank == 5) & (condition <= condition_limit)
    return dict(good=good, condition=jnp.where(numeric & nonzero, condition, _MAX),
                rank=jnp.where(numeric & nonzero, rank, 0),
                eigenvalues=jnp.where(numeric[:, None], eigenvalues, 0.),
                eigenvectors=jnp.where(numeric[:, None, None], eigenvectors, jnp.eye(5)),
                norm=jnp.where(numeric[:, None], norm, 1.), g=jnp.where(numeric[:, None], g, 0.),
                gradient_norm=jnp.where(numeric, jnp.max(jnp.abs(g), axis=1), _MAX))


def _initial_output(p, z_ref, seed_ok, valid, damping):
    batch, hits = valid.shape
    n = jnp.sum(valid, axis=1, dtype=jnp.int32)
    status = jnp.where(seed_ok, MAX_ITERATIONS, INVALID_SEED)
    status = jnp.where(n < 4, INSUFFICIENT_HITS, status)
    status = jnp.where(n == 0, EMPTY, status).astype(jnp.int32)
    initial = jnp.where(seed_ok[:, None], p, jnp.nan)
    out = dict(status=status, n_hits=n, initial_local_parameters=initial, local_parameters=initial,
               z_ref=jnp.where(seed_ok, z_ref, jnp.nan), parameter_scale=jnp.ones((batch, 5)),
               condition=jnp.full(batch, _MAX), eigenvalues=jnp.zeros((batch, 5)),
               damping=jnp.full(batch, float(damping)), residuals=jnp.zeros((batch, hits, 4)))
    out.update({k: jnp.zeros(batch, bool) for k in ('success', 'converged', 'objective_valid')})
    out.update({k: jnp.zeros(batch, jnp.int32) for k in ('rank', 'iterations', 'accepted_steps')})
    out.update({k: jnp.zeros(batch) for k in ('initial_objective', 'final_objective', 'gradient_norm')})
    out.update({k: jnp.zeros(valid.shape) for k in ('doca', 'wire_u', 'wire_pull', 'poca_z')})
    out.update({'poca_'+k: jnp.zeros(valid.shape, bool) for k in ('valid', 'converged', 'inside', 'boundary')})
    out['poca_iterations'] = jnp.zeros(valid.shape, jnp.int32)
    return out


def fit_wire_helix(local_parameters, z_ref, mid, direction, measured_u, uvar, hlen, valid, *,
                   seed_valid=None, max_iterations=40, transverse_scale=2.5, condition_limit=1e12,
                   gradient_tolerance=1e-7, fd_step=1e-4, initial_damping=1e-3,
                   max_parameter_step=10., poca_max_iterations=20, poca_tolerance=1e-7,
                   poca_max_step=50., poca_max_offset=500., max_rows=2048, max_cells=262144):
    """Bounded float64 diagonal LM with implicit-POCA central finite differences.

    Inputs: local_parameters [B,5]=(x0,y0,tx,ty,omega=dphase/dz) at fixed
    z_ref [B]; original mid/direction [B,H,3]; measured_u/uvar/hlen [B,H]
    in mm/mm2; boolean valid [B,H], optional boolean seed_valid [B]. Wire
    directions are normalized in 3D. Masked payloads, including NaNs, are
    ignored; ANY malformed selected hit fails its entire lane. At least four
    selected hits and full numerical rank are required. No membership changes.

    The trajectory is stable at omega=0. Every accepted step requires ALL
    selected local positive-Hessian POCAs to converge inside the bounded z
    search basin and finite wire (|wire_u|<=hlen); endpoints are not fitted.
    Objective = .5*sum(|perpendicular residual|^2/transverse_scale^2 +
    (predicted_u-measured_u)^2/uvar). Convergence uses the final unconstrained
    gradient, rank, conditioning and full POCA, not small accepted updates.
    max_iterations=0 only checks the initial stationarity; failures preserve
    the last accepted local state. No field value or momentum is needed.

    Returns device arrays: success/converged bool, status int32 (STATUS_NAMES),
    n_hits, iterations, accepted_steps, objective_valid, initial/final_objective,
    condition/rank/gradient_norm, damping, eigenvalues [B,5], parameter_scale
    [B,5], initial_local_parameters/local_parameters [B,5], z_ref; residuals
    [B,H,4] (scaled xyz and along-wire pull), doca/wire_u/wire_pull/poca_z,
    poca_valid/converged/inside/boundary and poca_iterations [B,H]. Unavailable
    objectives/diagnostics are zero, undefined condition is max-float; invalid
    seeds have NaN parameters/z_ref. No physical estimates or covariance.

    Configuration is static under jit; only shape/dtype/config checks are host
    operations. Work/storage are bounded O(B*H), with a [B,H,4,5] Jacobian and
    fixed-budget device loops. Row/cell limits raise; callers must chunk.
    """
    valid = jnp.asarray(valid)
    if valid.ndim != 2 or valid.dtype.kind != 'b':
        raise ValueError('valid must be boolean [batch,hits]')
    batch, hits = valid.shape
    for name, value in (('max_iterations', max_iterations), ('poca_max_iterations', poca_max_iterations),
                        ('max_rows', max_rows), ('max_cells', max_cells)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value <= 2**31-1:
            raise ValueError(f'{name} must be a nonnegative bounded integer')
    if batch > max_rows or batch*hits > max_cells:
        raise ValueError('refit row/cell bound exceeded; chunk rows explicitly')
    for name, value in (('transverse_scale', transverse_scale), ('condition_limit', condition_limit),
                        ('gradient_tolerance', gradient_tolerance), ('fd_step', fd_step),
                        ('initial_damping', initial_damping), ('max_parameter_step', max_parameter_step),
                        ('poca_tolerance', poca_tolerance), ('poca_max_step', poca_max_step),
                        ('poca_max_offset', poca_max_offset)):
        if not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive static scalar')
    if condition_limit <= 1 or initial_damping > 1e12:
        raise ValueError('condition_limit must exceed one; initial_damping <= 1e12')
    p, z_ref = _real('local_parameters', local_parameters, (batch, 5)), _real('z_ref', z_ref, (batch,))
    mid, direction = _real('mid', mid, (batch, hits, 3)), _real('direction', direction, (batch, hits, 3))
    measured_u, uvar, hlen = (_real(n, a, valid.shape) for n, a in
                             (('measured_u', measured_u), ('uvar', uvar), ('hlen', hlen)))
    seed_valid = jnp.ones(batch, bool) if seed_valid is None else jnp.asarray(seed_valid)
    if seed_valid.shape != (batch,) or seed_valid.dtype.kind != 'b':
        raise ValueError('seed_valid must be boolean [batch]')
    seed_ok = seed_valid & jnp.all(jnp.isfinite(p), axis=1) & jnp.isfinite(z_ref)
    out = _initial_output(p, z_ref, seed_ok, valid, initial_damping)
    if batch == 0 or hits == 0:
        return out
    hit_ok = (jnp.all(jnp.isfinite(mid), axis=-1) & jnp.all(jnp.isfinite(direction), axis=-1)
              & (jnp.max(jnp.abs(direction), axis=-1) > 0) & jnp.isfinite(measured_u)
              & jnp.isfinite(uvar) & (uvar > 0) & jnp.isfinite(hlen) & (hlen >= 0))
    input_ok = jnp.all(~valid | hit_ok, axis=1)
    eligible = seed_ok & input_ok & (out['n_hits'] >= 4)
    mask = valid & eligible[:, None]
    p, z_ref = jnp.where(seed_ok[:, None], p, 0.), jnp.where(seed_ok, z_ref, 0.)
    mid = jnp.where(mask[..., None], mid, 0.)
    direction = jnp.where(mask[..., None], direction, jnp.array([1., 0., 0.]))
    direction /= jnp.max(jnp.abs(direction), axis=-1)[..., None]
    direction /= jnp.sqrt(jnp.sum(direction**2, axis=-1))[..., None]
    measured_u, sigma, hlen = jnp.where(mask, measured_u, 0.), jnp.sqrt(jnp.where(mask, uvar, 1.)), jnp.where(mask, hlen, 0.)
    out['status'] = jnp.where(~input_ok & (out['n_hits'] > 0), INVALID_INPUT, out['status'])
    measured_z = mid[..., 2]+measured_u*direction[..., 2]
    span = jnp.max(jnp.where(mask, measured_z, -jnp.inf), axis=1)-jnp.min(jnp.where(mask, measured_z, jnp.inf), axis=1)
    length = jnp.where(eligible & jnp.isfinite(span), jnp.maximum(1., span), 1.)
    slope = jnp.hypot(p[:, 2], p[:, 3])
    scale = jnp.stack((jnp.full(batch, transverse_scale), jnp.full(batch, transverse_scale),
                       transverse_scale/length, transverse_scale/length,
                       transverse_scale/length/length/jnp.maximum(slope, transverse_scale/length)), axis=1)
    scale_ok = jnp.all(jnp.isfinite(scale) & (scale > 0), axis=1) & jnp.isfinite(span)
    scale = jnp.where(scale_ok[:, None], scale, 1.)
    out['parameter_scale'] = scale
    options = (poca_max_iterations, poca_tolerance, poca_max_step, poca_max_offset)

    def evaluate(q):
        state = _poca(q, z_ref, mid, direction, measured_u, hlen, mask, options)
        residual = _residual(state, measured_u, sigma, transverse_scale, mask)
        return state, residual, .5*jnp.sum(residual**2, axis=(1, 2))

    def linearize(q, state, residual):
        return _linearize(q, scale, z_ref, state, mid, direction, sigma, mask,
                          residual, transverse_scale, fd_step, condition_limit)

    def stationary(lin, objective):
        return lin['gradient_norm'] <= gradient_tolerance*(1+jnp.sqrt(2.)*jnp.sqrt(objective))

    state, residual, objective = evaluate(p)
    good = eligible & scale_ok & _all_poca(state, valid) & jnp.isfinite(objective)
    out['objective_valid'], out['initial_objective'] = good, jnp.where(good, objective, 0.)
    out['status'] = jnp.where(eligible & ~good, INITIAL_POCA_FAILED, out['status'])

    def update(c):
        i, q, state, residual, objective, active, status, iterations, accepted, damping = c
        lin = linearize(q, state, residual)
        status = jnp.where(active & ~lin['good'], ILL_CONDITIONED, status)
        active &= lin['good'] & ~stationary(lin, objective)
        v = lin['eigenvectors']
        projection = jnp.einsum('bji,bj->bi', v, lin['g'])
        denom = jnp.maximum(lin['eigenvalues'], 0.)+damping[:, None]
        step = -jnp.einsum('bij,bj->bi', v, projection/denom)/lin['norm']
        step /= jnp.maximum(1., jnp.max(jnp.abs(step), axis=1)/max_parameter_step)[:, None]
        trial_q = q+jnp.where(active[:, None], scale*step, 0.)
        trial, trial_r, trial_f = evaluate(trial_q)
        take = (active & _all_poca(trial, valid) & jnp.isfinite(trial_f)
                & jnp.all(jnp.isfinite(trial_q), axis=1) & (trial_f < objective))
        def select(a, b):
            return jnp.where(take.reshape((batch,)+(1,)*(a.ndim-1)), a, b)
        q, state, residual, objective = jax.tree_util.tree_map(select, (trial_q, trial, trial_r, trial_f), (q, state, residual, objective))
        damping = jnp.where(take, jnp.maximum(1e-12, damping/3),
                            jnp.where(active, jnp.minimum(1e13, damping*10), damping))
        stalled = active & (damping > 1e12)
        return (i+1, q, state, residual, objective, active & ~stalled,
                jnp.where(stalled, STALLED, status), iterations+active.astype(jnp.int32),
                accepted+take.astype(jnp.int32), damping)

    _, p, state, residual, objective, _, status, iterations, accepted, damping = lax.while_loop(
        lambda c: (c[0] < max_iterations) & jnp.any(c[5]), update,
        (jnp.int32(0), p, state, residual, objective, good, out['status'], out['iterations'], out['accepted_steps'], out['damping']))
    lin = linearize(p, state, residual)
    success = good & lin['good'] & stationary(lin, objective) & _all_poca(state, valid)
    status = jnp.where(good & ~lin['good'], ILL_CONDITIONED, status)
    out.update(success=success, converged=success, status=jnp.where(success, CONVERGED, status),
               iterations=iterations, accepted_steps=accepted, damping=damping,
               condition=jnp.where(good, lin['condition'], _MAX), rank=jnp.where(good, lin['rank'], 0),
               eigenvalues=jnp.where(good[:, None], lin['eigenvalues'], 0.),
               gradient_norm=jnp.where(good, lin['gradient_norm'], 0.), final_objective=jnp.where(good, objective, 0.),
               local_parameters=jnp.where(seed_ok[:, None], p, jnp.nan))
    hit_finite = state['valid'] & jnp.all(jnp.isfinite(residual), axis=-1)
    out['residuals'] = jnp.where(hit_finite[..., None], residual, 0.)
    for name, value in (('doca', jnp.sqrt(2.)*jnp.sqrt(state['f'])), ('wire_u', state['u']),
                        ('wire_pull', residual[..., 3]), ('poca_z', state['z'])):
        out[name] = jnp.where(hit_finite, value, 0.)
    out.update({'poca_'+name: state[name] & hit_finite for name in ('valid', 'converged', 'inside', 'boundary')})
    out['poca_iterations'] = state['iterations']
    return out
