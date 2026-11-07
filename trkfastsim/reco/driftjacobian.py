"""Analytic local residual derivatives for the unchanged conditional drift objective.

No finite differences, LR choices, POCA refits or modified response values occur
here. ``linearize`` consumes a validated driftfitdiagnostic._Evaluator and its
state at the supplied parameters. Derivatives describe the implicit stationary
local POCA, not the discrete stopping decisions of its numerical search. The
existing 1e-7 mm POCA tolerance consequently limits finite-difference agreement.
The fixed axial field, measured-curvature mass hypothesis, original support,
LR and saved dimensionless parameter scales are all retained.

At nonsmooth branches a selected one-sided derivative can support a search but
is NOT a stationarity certificate. ``good`` is only the numeric/rank/condition
check; callers must separately require ``differentiable``. Native interpolation
knots are conservatively flagged even if their adjacent slopes happen to agree.
Regression boundaries with identical coefficients are not artificial branches.
All boundary proximity tests use arithmetic-scaled roundoff, not physics buffers.
Workspace is O(B*H), with no loops over hits, rows or calibration queries.
"""
import numpy as np

from .fielddrift import _index, drift_info
from .helixfit import _real, _trajectory
from .helixtiming import _scalar

_EPS = np.finfo(float).eps
_MAX = np.finfo(float).max
_C = 299.792458


def _near(a, b, scale=1.):
    return np.abs(a-b) <= 64*_EPS*np.maximum(
        1., np.maximum(np.abs(a), np.maximum(np.abs(b), np.abs(scale))))


def _regression_branch(calibration, value, scale=1.):
    """Only internal index switches count; first/last bins extrapolate forever."""
    n = calibration.slope.size
    k = np.clip(np.rint((value-calibration.low)/calibration.width), 1, max(1, n-1))
    edge = calibration.low+k*calibration.width
    right = np.minimum(k.astype(np.intp), n-1)
    left = np.maximum(right-1, 0)
    same = ((calibration.intercept[left] == calibration.intercept[right])
            & (calibration.slope[left] == calibration.slope[right]))
    # Exact coefficient equality excludes both kinks and jumps.
    near = (n > 1) & _near(value, edge, np.maximum(abs(calibration.low), scale))
    return near & ~same


def calibration_partials(model, dt, phi, *, time_scale=1., phi_scale=1.):
    """Stock values and analytic partials, broadcasting exactly as drift_info.

    Returns the unmodified drift_info values plus ``dc_dt``, ``dc_dphi``,
    ``dr_dt``, ``dr_dphi``, ``dsigma_dt``, ``dsigma_dphi``, local regression
    slopes, four branch flags, and ``differentiable``. Angle partials are with
    respect to the *supplied* phi, including stock folding (or zero if ignored).
    Exact angle knots use the right-hand cell, except at pi/2 (left-hand cell).
    Tails extrapolate the native end cells; neither values nor slopes are clipped.
    This is not stock drift_velocity, which is a different calibrated quantity.
    ``time_scale`` optionally supplies the arithmetic scale of the subtraction
    producing dt (e.g. absolute sensor time/t0); ``phi_scale`` similarly supplies
    the coordinate/DOCA scale for an angle from projected geometry. Both affect
    only arithmetic roundoff guards, never response values or partials.
    Nonfinite/unsafe value queries raise under the stock contract.
    """
    info = drift_info(model, dt, phi)
    dt, phi = np.broadcast_arrays(np.asarray(dt, float), np.asarray(phi, float))
    with np.errstate(all='ignore'):
        used = info['used_phi']
        remainder = np.fmod(used, np.pi)
        absolute = np.abs(remainder)
        folded = np.where(absolute > np.pi/2, np.pi-absolute, absolute)
        fold_sign = np.where(absolute > np.pi/2, -1., 1.)*np.sign(remainder)
        # A one-sided derivative at zero is more useful than a fabricated zero.
        fold_sign = np.where(remainder == 0, 1., fold_sign)
        p = np.clip(np.floor(folded/model.delta_phi), 0,
                    model.distances_tbins.shape[1]-2).astype(np.intp)
        t = _index(dt, 0., model.delta_t, model.times.size-2)
        ft = (dt-model.times[t])/model.delta_t
        fp = (folded-p*model.delta_phi)/model.delta_phi
        grid = model.distances_tbins
        a, b = grid[t, p], grid[t+1, p]
        c, d = grid[t, p+1], grid[t+1, p+1]
        dc_dt = ((1-fp)*(b-a)+fp*(d-c))/model.delta_t
        dc_dphi = ((c+ft*(d-c))-(a+ft*(b-a)))/model.delta_phi*fold_sign
        if model.ignore_phi:
            dc_dphi = np.zeros_like(dc_dphi)
        oi = _index(info['c_drift'], model.offset.low, model.offset.width,
                    model.offset.slope.size-1)
        ei = _index(info['r_drift'], model.signed.low, model.signed.width,
                    model.signed.slope.size-1)
        offset_slope, error_slope = model.offset.slope[oi], model.signed.slope[ei]
        dr_dt, dr_dphi = (1-offset_slope)*dc_dt, (1-offset_slope)*dc_dphi
        dsigma_dt, dsigma_dphi = error_slope*dr_dt, error_slope*dr_dphi
        k = np.rint(dt/model.delta_t)
        time_boundary = ((k >= 1) & (k <= model.times.size-2)
                         & _near(dt, k*model.delta_t, time_scale))
        angle_edge = np.rint(folded/model.delta_phi)*model.delta_phi
        phi_boundary = (np.zeros(dt.shape, bool) if model.ignore_phi else
                        (_near(folded, angle_edge, phi_scale) | _near(folded, 0., phi_scale)
                         | _near(folded, np.pi/2, phi_scale)))
        offset_boundary = _regression_branch(model.offset, info['c_drift'],
                                               np.abs(dc_dt)*time_scale+np.abs(dc_dphi)*phi_scale)
        signed_error_boundary = _regression_branch(model.signed, info['r_drift'],
                                                     np.abs(dr_dt)*time_scale+np.abs(dr_dphi)*phi_scale)
        finite = (np.isfinite(dc_dt) & np.isfinite(dc_dphi)
                  & np.isfinite(dr_dt) & np.isfinite(dr_dphi)
                  & np.isfinite(dsigma_dt) & np.isfinite(dsigma_dphi))
    return dict(info, dc_dt=dc_dt, dc_dphi=dc_dphi, dr_dt=dr_dt, dr_dphi=dr_dphi,
                dsigma_dt=dsigma_dt, dsigma_dphi=dsigma_dphi,
                offset_slope=offset_slope, signed_error_slope=error_slope,
                time_boundary=time_boundary, phi_boundary=phi_boundary,
                offset_boundary=offset_boundary, signed_error_boundary=signed_error_boundary,
                differentiable=finite & ~time_boundary & ~phi_boundary
                & ~offset_boundary & ~signed_error_boundary)


def trajectory_partials(parameters, z_ref, z):
    """Fixed-z P_theta,V_theta [B,H,3,5], stable including omega == 0.

    The position/tangent *values* remain helixfit._trajectory's responsibility.
    Analytic sinc/cosc derivatives use Taylor expansions for |omega*dz|<0.01;
    the first omitted terms there are below double-precision roundoff.
    """
    p = _real('parameters', parameters)
    if p.ndim != 2 or p.shape[1] != 5:
        raise ValueError('parameters must have shape [batch,5]')
    zref = _real('z_ref', z_ref, (len(p),))
    z = _real('z', z)
    if z.ndim != 2 or z.shape[0] != len(p):
        raise ValueError('z must have shape [batch,hits]')
    with np.errstate(all='ignore'):
        dz = z-zref[:, None]
        a = p[:, 4, None]*dz
        a2 = a*a
        sinc = np.sinc(a/np.pi)
        cosc = .5*a*np.sinc(a/(2*np.pi))**2
        small = np.abs(a) < .01
        safe = np.where(small, 1., a)
        dsinc = np.where(small, a*(-1/3+a2*(1/30+a2*(-1/840+a2/45360))),
                         (a*np.cos(a)-np.sin(a))/(safe*safe))
        dcosc = np.where(small, .5+a2*(-1/8+a2*(1/144+a2*(-1/5760+a2/403200))),
                         (a*np.sin(a)-2*np.sin(a/2)**2)/(safe*safe))
        tx, ty = p[:, 2, None], p[:, 3, None]
        pt = np.zeros(z.shape+(3, 5))
        vt = np.zeros_like(pt)
        pt[..., 0, 0], pt[..., 1, 1] = 1., 1.
        pt[..., 0, 2], pt[..., 0, 3] = dz*sinc, -dz*cosc
        pt[..., 1, 2], pt[..., 1, 3] = dz*cosc, dz*sinc
        pt[..., 0, 4] = dz*dz*(tx*dsinc-ty*dcosc)
        pt[..., 1, 4] = dz*dz*(ty*dsinc+tx*dcosc)
        co, si = np.cos(a), np.sin(a)
        vt[..., 0, 2], vt[..., 0, 3] = co, -si
        vt[..., 1, 2], vt[..., 1, 3] = si, co
        vt[..., 0, 4] = -dz*(tx*si+ty*co)
        vt[..., 1, 4] = dz*(tx*co-ty*si)
    return pt, vt


def _normal_equations(jac, residual, numeric, condition_limit):
    """Same six-column equilibration/rank contract as helixdriftfit._linearize."""
    batch, hits = jac.shape[:2]
    jac = np.where(numeric[:, None, None, None], jac, 0.)
    flat = jac.reshape(batch, hits*2, 6)
    with np.errstate(all='ignore'):
        normal = np.einsum('bhi,bhj->bij', flat, flat, optimize=False)
        gradient = np.einsum('bhi,bh->bi', flat, residual.reshape(batch, hits*2), optimize=False)
        norm = np.sqrt(np.maximum(np.diagonal(normal, axis1=1, axis2=2), 0.))
        nonzero = np.all(norm > np.finfo(float).tiny, axis=1)
        norm = np.where(norm > np.finfo(float).tiny, norm, 1.)
        correlation = normal/norm[:, :, None]/norm[:, None, :]
        g = gradient/norm
        numeric = numeric & np.isfinite(correlation).all(axis=(1, 2)) & np.isfinite(g).all(axis=1)
        correlation = np.where(numeric[:, None, None], correlation, np.eye(6))
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        except np.linalg.LinAlgError:
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
                gradient_norm=np.where(numeric, np.max(np.abs(g), axis=1), _MAX),
                jacobian=np.where(numeric[:, None, None, None], jac, 0.))


def linearize(evaluator, parameters, state, *, condition_limit=1e12):
    """Return analytic scaled residual Jacobian [B,H,2,6] and LM diagnostics.

    ``state`` MUST be evaluator(parameters), with its unchanged residuals and
    POCA. ``numeric``/``good`` do not imply differentiability. Masked derivatives
    are zero. Invalid rows have zero placeholder Jacobians, rank=0, numeric and
    differentiable false; no placeholder can certify a failed or empty lane.
    Additional *_jacobian arrays are in the same saved scaled coordinates.
    No candidate evaluator calls are made, and inputs are never mutated.
    """
    condition_limit = _scalar('condition_limit', condition_limit, True)
    p = _real('parameters', parameters, (evaluator.batch, 6))
    mask = evaluator.mask
    residual = _real('residuals', state['residuals'], mask.shape+(2,))
    hit = mask & state['valid'] & state['objective_valid'][:, None]
    z = np.where(hit, state['poca_z'], evaluator.zref[:, None])
    p = np.where(np.isfinite(p), p, 0.)
    with np.errstate(all='ignore'):
        point, v, second = _trajectory(p[:, :5], evaluator.zref, z)
        pt, vt = trajectory_partials(p[:, :5], evaluator.zref, z)
        w = evaluator.direction
        s = np.where((evaluator.end == 0)[..., None], w, -w)
        delta = point-evaluator.mid
        u = np.sum(delta*w, axis=-1)
        r = delta-u[..., None]*w
        tp = v-np.sum(v*w, axis=-1)[..., None]*w
        curvature = np.sum(r*second, axis=-1)
        h = np.sum(tp*tp, axis=-1)+curvature
        positive = h > 128*_EPS*(np.sum(v*v, axis=-1)+np.abs(curvature))
        pr = pt-w[..., None]*np.sum(w[..., None]*pt, axis=-2)[..., None, :]
        pv = vt-w[..., None]*np.sum(w[..., None]*vt, axis=-2)[..., None, :]
        dg = np.sum(pr*tp[..., None], axis=-2)+np.sum(r[..., None]*pv, axis=-2)
        dz5 = -dg/np.where(positive, h, 1.)[..., None]
        dp = pt+v[..., None]*dz5[..., None, :]
        du5 = np.sum(w[..., None]*dp, axis=-2)
        dq = dp-w[..., None]*du5[..., None, :]
        # Match prepare_timing's projected-point arithmetic.
        late = np.where((evaluator.end == 0)[..., None], evaluator.hv, evaluator.cal)
        sensor = late+np.sum((point-late)*s, axis=-1)[..., None]*s
        q = point-sensor
        doca = state['doca']
        signed5 = (np.sign(state['signed_doca'])[..., None]
                   *np.sum(q[..., None]*dq, axis=-2)/np.where(doca > 0, doca, 1.)[..., None])
        b = np.array([0., 0., np.sign(evaluator.bz)])
        radial = np.cross(s, b)  # deliberately NOT normalized
        qr, qb = np.sum(q*radial, axis=-1), np.sum(q*b, axis=-1)
        dqr = np.sum(dq*radial[..., None], axis=-2)
        dqb = np.sum(dq*b[None, None, :, None], axis=-2)
        denominator = qr*qr+qb*qb
        phi5 = (np.abs(qb)[..., None]*np.sign(qr)[..., None]*dqr
                -np.abs(qr)[..., None]*np.sign(qb)[..., None]*dqb)/np.where(
                    denominator > 0, denominator, 1.)[..., None]
        coord = np.maximum(1., np.maximum(np.max(np.abs(point), axis=-1),
                                         np.max(np.abs(sensor), axis=-1)))
        angle_branch = _near(qr, 0., coord) | _near(qb, 0., coord)
        doca_branch = doca <= np.maximum(1e-9, 64*_EPS*coord)
        k = evaluator.mass/(.299792458*abs(evaluator.bz))
        tx, ty, omega = p[:, 2], p[:, 3], p[:, 4]
        path = np.sqrt(1+tx*tx+ty*ty+(k*omega)**2)
        path_derivative = np.zeros((evaluator.batch, 5))
        path_derivative[:, 2] = tx/path
        path_derivative[:, 3] = ty/path
        path_derivative[:, 4] = k*(k*omega)/path
        tof5 = (path[:, None, None]*dz5
                +(z-evaluator.zref[:, None])[..., None]*path_derivative[:, None, :])/_C
        sensor5 = np.sum(s[..., None]*dp, axis=-2)/evaluator.speed[..., None]
        dt5 = sensor5-evaluator.flight[:, None, None]*tof5
        dz, du, signed, phi, dt, tof, sensor_jac = (np.zeros(mask.shape+(6,)) for _ in range(7))
        for dest, src in ((dz, dz5), (du, du5), (signed, signed5), (phi, phi5),
                          (dt, dt5), (tof, tof5), (sensor_jac, sensor5)):
            dest[..., :5] = src
        dt[..., 5] = -1.
        dc, dr, dsigma = (np.zeros_like(dt) for _ in range(3))
        calibration_good = np.zeros(mask.shape, bool)
        if np.any(hit):
            # Boundary slack accounts for large propagation/TOF subtraction operands.
            endpoint_scale = np.maximum(np.max(np.abs(evaluator.cal), axis=-1),
                                        np.max(np.abs(evaluator.hv), axis=-1))
            time_scale = np.maximum(np.abs(evaluator.te), np.abs(p[:, 5, None]))
            time_scale = np.maximum(time_scale, np.maximum(coord, endpoint_scale)/evaluator.speed)
            time_scale = np.maximum(time_scale, np.abs(z-evaluator.zref[:, None])*path[:, None]/_C)
            angle_size = np.sqrt(denominator)
            cp = calibration_partials(evaluator.model, state['delta_t'][hit], state['phi'][hit],
                                      time_scale=time_scale[hit],
                                      phi_scale=(coord/np.where(angle_size > 0, angle_size, 1.))[hit])
            dc[hit] = cp['dc_dt'][:, None]*dt[hit]+cp['dc_dphi'][:, None]*phi[hit]
            dr[hit] = (1-cp['offset_slope'])[:, None]*dc[hit]
            dsigma[hit] = cp['signed_error_slope'][:, None]*dr[hit]
            calibration_good[hit] = cp['differentiable']
        sigma = np.where(hit, state['signed_error'], 1.)
        drift_residual = evaluator.lr*state['r_drift']-state['signed_doca']
        pull_jac = ((evaluator.lr[..., None]*dr-signed)
                    -drift_residual[..., None]/sigma[..., None]*dsigma)/sigma[..., None]
        jac = np.stack((pull_jac, du/evaluator.sigma_u[..., None]), axis=-2)
        jac = np.where(mask[..., None, None], jac*evaluator.scale[:, None, None, :], 0.)
        numeric = (state['objective_valid'] & evaluator.eligible
                   & np.all(~mask | (hit & positive), axis=1)
                   & np.isfinite(parameters).all(axis=1)
                   & np.isfinite(residual).all(axis=(1, 2))
                   & np.isfinite(jac).all(axis=(1, 2, 3)))
    out = _normal_equations(jac, residual, numeric, condition_limit)
    differentiable_hit = (hit & positive & calibration_good & ~angle_branch & ~doca_branch
                          & out['numeric'][:, None])
    out.update(differentiable_hit=differentiable_hit,
               differentiable=out['numeric'] & np.all(~mask | differentiable_hit, axis=1))
    for name, value in (('poca_z', dz), ('wire_u', du), ('signed_doca', signed),
                        ('phi', phi), ('delta_t', dt), ('tof_positive', tof),
                        ('sensor_time', sensor_jac), ('c_drift', dc), ('r_drift', dr),
                        ('signed_error', dsigma)):
        with np.errstate(all='ignore'):
            out[name+'_jacobian'] = np.where(
                (hit & out['numeric'][:, None])[..., None], value*evaluator.scale[:, None, :], 0.)
    return out
