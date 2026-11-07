"""Opt-in fixed-geometry timing diagnostics, not a joint fit or chi-square.

Geometry is the measured-wire local helix, not truth. No association, TOT,
ComboHit timeVar (a TOT error), corrected s_time, covariance or material enters.
The nominal trajectory field is the explicit constant axial approximation
(0,0,sign(bz)); it is NOT the drift model's detector-origin field vector.
"""
import numpy as np

from .fielddrift import drift_info
from .helixfit import _real, _trajectory

_C = 299.792458  # mm/ns
_EPS = np.finfo(float).eps
_DOCA_MIN = 1e-9  # mm; fixed engineering threshold, not detector resolution
_PARALLEL_MIN = 1e-5  # sine of particle/sensor crossing angle
_SIGN_REL = 1e-9  # relative triple product after normalizing the cross product
_POCA_TOL = 1e-7  # mm in z, matching the initializer's default local POCA test


def _integer(name, value, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= value <= 2**31-1):
        raise ValueError(f'{name} must be a bounded integer >= {minimum}')
    return int(value)


def _scalar(name, value, positive=False):
    a = _real(name, value)
    if a.ndim != 0 or not np.isfinite(a) or (positive and a <= 0):
        raise ValueError(f'{name} must be a finite scalar' + (' > 0' if positive else ''))
    return float(a)


def _bool(name, value, shape=None):
    a = np.asarray(value)
    if a.dtype.kind != 'b' or (shape is not None and a.shape != shape):
        raise ValueError(f'{name} must be boolean with shape {shape}')
    return a


def prepare_timing(local_parameters, z_ref, poca_z, wire_cal, wire_hv,
                   early_end, early_time, signal_speed, valid, *, bz,
                   mass=105.6583755, seed_valid=None, max_rows=2048,
                   max_cells=262144):
    """Check an existing POCA and construct fixed-geometry timing, in mm/ns/MeV.

    Shapes: local_parameters [B,5]=(x0,y0,tx,ty,omega), z_ref [B],
    poca_z/early_end/early_time/signal_speed/valid [B,H], endpoints [B,H,3].
    valid must be boolean; masked payloads, including NaNs, are ignored.
    early_end is 0=cal, 1=hv. Supply calibrated early-end time, not s_time.
    Signal speed is positive mm/ns (stock 2*halfPropV).

    Caller MUST pass seed_valid=seed_success & physical_identifiable [B].
    None deliberately certifies no rows. This gate cannot be inferred from
    numeric parameters. Zero omega or zero transverse slope also fails:
    a straight track has no measured curvature, not beta=1. Unit-charge
    p=.299792458*abs(bz)*sqrt(1+tx**2+ty**2)/abs(omega), beta=p/hypot(p,mass).
    bz is a finite nonzero scalar in tesla; mass is a positive hypothesis.

    _trajectory reconstructs position/tangent at supplied poca_z. Projection
    is onto the actual endpoint line, never a reoptimized POCA. The analytic
    gradient and positive Hessian must pass the fixed 1e-7 mm stationary test;
    every selected POCA must be STRICTLY inside the finite wire (endpoints
    rejected with 64-epsilon coordinate-scaled arithmetic slack). Sensor
    direction s=unit(E-L) points late->early, as in
    SensorLine(endpos,mpos): ts=te-dot(E-Xsensor,s)/signal_speed.
    tof_positive=(z-z_ref)*sqrt(1+tx**2+ty**2)/(beta*c); multiply by +/-1
    for the two flight directions. Neither direction is identified here.

    With q=Xp-Xs and b=(0,0,sign(bz)), r=s cross b is NOT normalized;
    flip r when q.r<0, then phi=atan2(q.r,q.b), folded above pi/2.
    DOCA <=1e-9 mm has undefined angle/sign. The positive-z signed DOCA is
    sign((s cross pdir).(-q))*|q|; cross norm <=1e-5 is parallel and the
    normalized triple product must exceed 1e-9*|q| in absolute value. This is
    an engineering sine-angle guard on supplied POCAs, not KinKal's different
    sin(theta)**2 < 1e-5 iterative closest-approach denominator guard.

    Returns row_valid/status [B], mask (frozen membership) and valid (usable
    hit) [B,H]; particle_poca/sensor_poca [B,H,3]; doca, phi,
    propagation_distance/time, sensor_time, tof_positive, signed_doca_positive
    [B,H]; angle_valid/sign_valid/inside/stationary/parallel/input_valid and
    hit_status [B,H]; beta/p_abs [B]. Momentum is NaN unless caller-certified
    and numerically defined. Failed rows retain finite available diagnostics;
    unavailable/masked hit diagnostics are zero/false, not fabricated success.
    Statuses: ready, empty, invalid_seed, undefined_momentum, invalid_input,
    numerical_failure, outside_wire, parallel, nonstationary, undefined_angle,
    undefined_sign. hit_status uses masked instead of empty/invalid_seed.
    O(B*H) workspace; hard row/cell bounds raise, never truncate.
    """
    valid = _bool('valid', valid)
    if valid.ndim != 2:
        raise ValueError('valid must be boolean [batch,hits]')
    batch, hits = valid.shape
    max_rows, max_cells = _integer('max_rows', max_rows), _integer('max_cells', max_cells)
    if batch > max_rows or batch*hits > max_cells:
        raise ValueError('timing row/cell bound exceeded; chunk rows explicitly')
    p = _real('local_parameters', local_parameters, (batch, 5))
    zref = _real('z_ref', z_ref, (batch,))
    z = _real('poca_z', poca_z, valid.shape)
    cal = _real('wire_cal', wire_cal, valid.shape+(3,))
    hv = _real('wire_hv', wire_hv, cal.shape)
    end, te, speed = (_real(n, a, valid.shape) for n, a in
                      (('early_end', early_end), ('early_time', early_time),
                       ('signal_speed', signal_speed)))
    bz, mass = _scalar('bz', bz), _scalar('mass', mass, True)
    if bz == 0:
        raise ValueError('bz must be nonzero')
    certified = (np.zeros(batch, bool) if seed_valid is None else
                 _bool('seed_valid', seed_valid, (batch,)))
    seed_finite = np.isfinite(p).all(axis=1) & np.isfinite(zref)
    p = np.where(seed_finite[:, None], p, 0.)
    zref = np.where(seed_finite, zref, 0.)
    input_ok = (np.isfinite(z) & np.isfinite(cal).all(axis=-1)
                & np.isfinite(hv).all(axis=-1) & ((end == 0) | (end == 1))
                & np.isfinite(te) & np.isfinite(speed) & (speed > 0))
    # Safe placeholders permit diagnostics in other hits of a failed row.
    use = valid & input_ok & seed_finite[:, None]
    cal = np.where(use[..., None], cal, 0.)
    hv = np.where(use[..., None], hv, np.array([1., 0., 0.]))
    z = np.where(use, z, zref[:, None])
    te, speed = np.where(use, te, 0.), np.where(use, speed, 1.)
    with np.errstate(all='ignore'):
        slope = np.hypot(p[:, 2], p[:, 3])
        path_scale = np.hypot(1., slope)
        p_abs = .299792458*abs(bz)*path_scale/np.abs(p[:, 4])
        beta = p_abs/np.hypot(p_abs, mass)
        momentum_ok = (certified & seed_finite & (slope > 128*_EPS)
                       & (p[:, 4] != 0) & np.isfinite(p_abs) & (p_abs > 0)
                       & np.isfinite(beta) & (beta > 0))
        early = np.where((end == 0)[..., None], cal, hv)
        late = np.where((end == 0)[..., None], hv, cal)
        wire = early-late
        length = np.sqrt(np.sum(wire*wire, axis=-1))
        line_ok = np.isfinite(length) & (length > 0)
        s = wire/np.where(line_ok, length, 1.)[..., None]
        point, tangent, second = _trajectory(p, zref, z)
        along = np.sum((point-late)*s, axis=-1)
        sensor = late+along[..., None]*s
        q = point-sensor
        doca = np.sqrt(np.sum(q*q, axis=-1))
        tp = tangent-np.sum(tangent*s, axis=-1)[..., None]*s
        speed2 = np.sum(tp*tp, axis=-1)
        curvature = np.sum(q*second, axis=-1)
        h = speed2+curvature
        g = np.sum(q*tp, axis=-1)
        positive = h > 128*_EPS*(np.sum(tangent*tangent, axis=-1)+np.abs(curvature))
        stationary = (positive & (np.abs(g/np.where(positive, h, 1.)) <= _POCA_TOL)
                      & (np.abs(g) <= _POCA_TOL*np.maximum(1., speed2)))
        # Reject endpoint roundoff using arithmetic slack, not a physical wire-length buffer.
        endpoint_slack = 64*_EPS*np.maximum(1., np.maximum(length, np.maximum(
            np.max(np.abs(early), axis=-1), np.max(np.abs(late), axis=-1))))
        inside = (along > endpoint_slack) & (along < length-endpoint_slack)
        pdir = tangent/path_scale[:, None, None]
        cross = np.cross(s, pdir)
        cross_norm = np.sqrt(np.sum(cross*cross, axis=-1))
        parallel = cross_norm <= _PARALLEL_MIN
        triple = np.sum(cross*(-q), axis=-1)/np.where(cross_norm > 0, cross_norm, 1.)
        sign_valid = (~parallel & (doca > _DOCA_MIN) & (np.abs(triple) > _SIGN_REL*doca))
        signed_doca = np.where(sign_valid, np.sign(triple)*doca, 0.)
        b = np.array([0., 0., np.sign(bz)])
        radial = np.cross(s, b)
        qr = np.sum(q*radial, axis=-1)
        radial = np.where((qr < 0)[..., None], -radial, radial)
        qr, qb = np.sum(q*radial, axis=-1), np.sum(q*b, axis=-1)
        angle_valid = (doca > _DOCA_MIN) & (np.hypot(qr, qb) > 128*_EPS*doca)
        phi = np.arctan2(qr, qb)
        phi = np.where(phi > np.pi/2, np.pi-phi, phi)
        propagation_distance = np.sum((early-sensor)*s, axis=-1)
        propagation_time = propagation_distance/speed
        sensor_time = te-propagation_time
        tof = (z-zref[:, None])*path_scale[:, None]/(np.where(momentum_ok, beta, 1.)[:, None]*_C)
    finite = (np.isfinite(point).all(axis=-1) & np.isfinite(sensor).all(axis=-1)
              & np.isfinite(doca) & np.isfinite(g) & np.isfinite(h)
              & np.isfinite(phi) & np.isfinite(propagation_distance)
              & np.isfinite(propagation_time) & np.isfinite(sensor_time)
              & np.isfinite(tof) & np.isfinite(cross_norm) & np.isfinite(triple))
    input_ok &= line_ok
    diagnostic = use & line_ok & finite
    flags = dict(angle_valid=angle_valid, sign_valid=sign_valid, inside=inside,
                 stationary=stationary, parallel=parallel)
    out = dict(mask=valid.copy(), input_valid=valid & input_ok,
               p_abs=np.where(momentum_ok, p_abs, np.nan),
               beta=np.where(momentum_ok, beta, np.nan))
    for name, value in flags.items():
        out[name] = diagnostic & value
    for name, value in (('particle_poca', point), ('sensor_poca', sensor)):
        out[name] = np.where(diagnostic[..., None], value, 0.)
    for name, value in (('doca', doca), ('phi', np.where(angle_valid, phi, 0.)),
                        ('propagation_distance', propagation_distance),
                        ('propagation_time', propagation_time), ('sensor_time', sensor_time),
                        ('tof_positive', np.where(momentum_ok[:, None], tof, 0.)),
                        ('signed_doca_positive', signed_doca)):
        out[name] = np.where(diagnostic, value, 0.)
    hit_status = np.full(valid.shape, 'ready', dtype='<U24')
    # Later entries have precedence, while independent boolean flags are retained.
    for condition, label in ((~sign_valid, 'undefined_sign'), (~angle_valid, 'undefined_angle'),
                             (~stationary, 'nonstationary'), (parallel, 'parallel'),
                             (~inside, 'outside_wire'), (~finite, 'numerical_failure'),
                             (~input_ok, 'invalid_input')):
        hit_status[valid & condition] = label
    hit_status[~valid] = 'masked'
    geometry_ok = diagnostic & inside & stationary & ~parallel & angle_valid & sign_valid
    out['valid'] = geometry_ok & momentum_ok[:, None]
    n_hits = np.sum(valid, axis=1)
    out['row_valid'] = momentum_ok & (n_hits > 0) & np.all(~valid | out['valid'], axis=1)
    status = np.full(batch, 'ready', dtype='<U24')
    for label in ('undefined_sign', 'undefined_angle', 'nonstationary', 'parallel',
                  'outside_wire', 'numerical_failure', 'invalid_input'):
        status[np.any(hit_status == label, axis=1)] = label
    status[~momentum_ok] = 'undefined_momentum'
    status[~certified | ~seed_finite] = 'invalid_seed'
    status[n_hits == 0] = 'empty'
    out.update(status=status, hit_status=hit_status)
    return out


def profile_t0(model, timing, valid, *, offset_min=-200., offset_max=50.,
               grid_step=.5, refinements=2, max_query_cells=262144):
    """Profile t0 on FROZEN geometry/membership, separately for directions [-1,+1].

    For each row/direction, anchor=median(ts-direction*tof_positive) over ALL
    selected hits. Search t0=anchor+offset in the closed offset domain. The
    default coarse grid has all 501 points. Each refinement evaluates 21 points
    in best_offset +/- previous_step, spaced previous_step/10; retain the best
    of ALL previous and new candidates. Default final resolution is .005 ns.
    Out-of-domain refinement points are not evaluated. This is a global coarse
    grid plus LOCAL refinement, NOT a mathematical global minimum guarantee or
    optimizer convergence claim, particularly at calibration switches.

    Score = .5*sum(((doca-r_drift)/signed_error)**2), not a chi-square or full
    fit likelihood. drift_info is queried at dt=ts-direction*tof_positive-t0
    and the prepared phi. No dt/radius clipping, sigma floor, timeVar addition,
    sigma derivative, outlier removal, or unsigned error is used. A selected
    nonfinite or nonpositive signed error invalidates the entire candidate.
    Native evaluator exceptions (e.g. unsafe extrapolation) propagate; returned
    nonfinite radii/errors invalidate candidates. Membership must exactly equal
    timing['mask']; any selected unusable geometry invalidates the whole row.

    Results: direction [2]=[-1,+1]; status/t0/anchor/offset/score/profile_valid/
    boundary/outside_oracle [B,2]; scalar resolution; best delta_t/r_drift/
    drift_residual/signed_error/pull [B,2,H]. status is empty, invalid_geometry,
    unavailable (no finite candidate), profile_boundary (best at search edge,
    NOT valid), or profiled_grid. Invalid rows have NaN scalar profile outputs
    and zero per-hit diagnostics; boundary solutions retain diagnostics.
    Masked hit diagnostics are zero. outside_oracle counts best dt outside
    [-50,200] ns: COVERAGE ONLY, never a reconstruction cut. direction_resolved
    is always false [B]; do not interpret score ordering as validated physics.

    Workspace is bounded by max_query_cells for every model query/candidate
    tile, plus O(B*H) inputs/outputs/median workspace. Hard failure if even one
    full-H reduction cannot fit. No event/hit loops; only bounded row/grid
    tiles and refinement settings. Reductions always use the same full H axis,
    making results exact under query chunk changes. Coarse grid count is also
    hard-bounded at 1,000,001 and refinements at 12, rather than truncated.
    """
    valid = _bool('valid', valid)
    if valid.ndim != 2:
        raise ValueError('valid must be boolean [batch,hits]')
    batch, hits = valid.shape
    max_query_cells = _integer('max_query_cells', max_query_cells, 1)
    refinements = _integer('refinements', refinements)
    if refinements > 12:
        raise ValueError('refinements exceeds 12')
    lo, hi = _scalar('offset_min', offset_min), _scalar('offset_max', offset_max)
    step = _scalar('grid_step', grid_step, True)
    with np.errstate(all='ignore'):
        intervals = np.float64(hi-lo)/step
    if hi <= lo or not np.isfinite(intervals) or intervals > 1_000_000:
        raise ValueError('offset domain must increase and have <= 1,000,001 grid points')
    grid_count = int(np.ceil(intervals))+1
    resolution = step/10**refinements
    if resolution == 0 or hits > max_query_cells:
        raise ValueError('resolution underflow or full-hit query cell bound exceeded')
    mask = _bool('timing mask', timing['mask'], valid.shape)
    if not np.array_equal(mask, valid):
        raise ValueError('profile membership must equal the frozen timing mask')
    row_valid = _bool('timing row_valid', timing['row_valid'], (batch,))
    hit_valid = _bool('timing valid', timing['valid'], valid.shape)
    ts, tof, doca, phi = (_real(n, timing[n], valid.shape) for n in
                         ('sensor_time', 'tof_positive', 'doca', 'phi'))
    finite = (np.isfinite(ts) & np.isfinite(tof) & np.isfinite(doca) & (doca >= 0)
              & np.isfinite(phi))
    count = np.sum(valid, axis=1)
    eligible = row_valid & (count > 0) & np.all(~valid | (hit_valid & finite), axis=1)
    direction = np.array([-1, 1], dtype=np.int8)
    shape = (batch, 2)
    out = dict(direction=direction, profile_valid=np.zeros(shape, bool),
               boundary=np.zeros(shape, bool), outside_oracle=np.zeros(shape, np.int64),
               resolution=resolution, direction_resolved=np.zeros(batch, bool),
               status=np.full(shape, 'invalid_geometry', dtype='<U24'))
    out['status'][count == 0] = 'empty'
    out['status'][eligible] = 'unavailable'
    for name in ('t0', 'anchor', 'offset', 'score'):
        out[name] = np.full(shape, np.nan)
    for name in ('delta_t', 'r_drift', 'drift_residual', 'signed_error', 'pull'):
        out[name] = np.zeros(shape+(hits,))
    if not np.any(eligible):
        return out
    # Flatten only row/direction; the complete hit axis is NEVER chunked/reordered.
    rows = np.flatnonzero(eligible)
    ts = np.where(valid[rows], ts[rows], 0.)
    tof = np.where(valid[rows], tof[rows], 0.)
    with np.errstate(all='ignore'):
        shifted = ts[:, None, :]-direction[None, :, None]*tof[:, None, :]
    numeric_row = np.all(~valid[rows, None, :] | np.isfinite(shifted), axis=(1, 2))
    rows, shifted = rows[numeric_row], shifted[numeric_row]
    if rows.size == 0:
        return out
    selected = np.repeat(valid[rows], 2, axis=0)
    shifted = shifted.reshape(-1, hits)
    nd = np.repeat(count[rows], 2)
    ordered = np.sort(np.where(selected, shifted, np.inf), axis=-1)
    lane = np.arange(len(shifted))
    anchor = .5*ordered[lane, (nd-1)//2]+.5*ordered[lane, nd//2]
    out['anchor'][rows] = anchor.reshape(-1, 2)
    target = np.repeat(doca[rows], 2, axis=0)
    angles = np.repeat(phi[rows], 2, axis=0)
    best_score = np.full(len(lane), np.inf)
    best_offset = np.zeros(len(lane))
    best_hit = {name: np.zeros_like(shifted) for name in
                ('delta_t', 'r_drift', 'drift_residual', 'signed_error', 'pull')}
    # Small row tiles allow useful grid vectorization without unbounded G*B*H.
    lane_chunk = min(len(lane), 64, max_query_cells//hits)
    for stage in range(refinements+1):
        center = best_offset.copy()  # fixed bracket for the ENTIRE stage
        spacing = step/10**stage
        ngrid = grid_count if stage == 0 else 21
        for start in range(0, len(lane), lane_chunk):
            stop = min(start+lane_chunk, len(lane))
            sl = slice(start, stop)
            grid_chunk = max_query_cells//((stop-start)*hits)
            for begin in range(0, ngrid, grid_chunk):
                indices = np.arange(begin, min(begin+grid_chunk, ngrid))
                if stage == 0:
                    offsets = np.broadcast_to(np.minimum(lo+indices*step, hi),
                                              (stop-start, len(indices)))
                else:
                    offsets = center[sl, None]+(indices[None, :]-10)*spacing
                in_domain = (offsets >= lo) & (offsets <= hi)
                if stage > 0:
                    # An unavailable coarse profile cannot supply a refinement bracket.
                    in_domain &= np.isfinite(best_score[sl, None])
                with np.errstate(all='ignore'):
                    t0 = anchor[sl, None]+offsets
                    dt = shifted[sl, None, :]-t0[..., None]
                member = np.broadcast_to(selected[sl, None, :], dt.shape)
                query = member & in_domain[..., None] & np.isfinite(dt)
                radii, sigma = np.zeros_like(dt), np.zeros_like(dt)
                if np.any(query):
                    info = drift_info(model, dt[query],
                                      np.broadcast_to(angles[sl, None, :], dt.shape)[query])
                    radii[query], sigma[query] = info['r_drift'], info['signed_error']
                good_hit = query & np.isfinite(radii) & np.isfinite(sigma) & (sigma > 0)
                with np.errstate(all='ignore'):
                    residual = target[sl, None, :]-radii
                    pull = np.where(member, residual/np.where(good_hit, sigma, 1.), 0.)
                    score = .5*np.sum(pull*pull, axis=-1)
                good = in_domain & np.all(~member | good_hit, axis=-1) & np.isfinite(score)
                score = np.where(good, score, np.inf)
                index = np.argmin(score, axis=1)
                local = np.arange(stop-start)
                candidate_score = score[local, index]
                take = candidate_score < best_score[sl]  # deterministic first tie wins
                best_score[sl] = np.where(take, candidate_score, best_score[sl])
                best_offset[sl] = np.where(take, offsets[local, index], best_offset[sl])
                for name, value in (('delta_t', dt), ('r_drift', radii),
                                    ('drift_residual', residual), ('signed_error', sigma),
                                    ('pull', pull)):
                    values = np.where(selected[sl], value[local, index], 0.)
                    best_hit[name][sl] = np.where(take[:, None], values, best_hit[name][sl])
    available = np.isfinite(best_score)
    edge_slack = 32*_EPS*max(1., abs(lo), abs(hi))
    boundary = available & ((best_offset <= lo+edge_slack) | (best_offset >= hi-edge_slack))
    out['offset'][rows] = np.where(available, best_offset, np.nan).reshape(-1, 2)
    out['score'][rows] = np.where(available, best_score, np.nan).reshape(-1, 2)
    out['t0'][rows] = np.where(available, anchor+best_offset, np.nan).reshape(-1, 2)
    out['boundary'][rows] = boundary.reshape(-1, 2)
    out['profile_valid'][rows] = (available & ~boundary).reshape(-1, 2)
    out['status'][rows] = np.where(~available, 'unavailable',
                                  np.where(boundary, 'profile_boundary', 'profiled_grid')).reshape(-1, 2)
    for name, values in best_hit.items():
        out[name][rows] = values.reshape(-1, 2, hits)
    dt = best_hit['delta_t']
    out['outside_oracle'][rows] = np.sum(selected & available[:, None] &
                                        ((dt < -50.) | (dt > 200.)), axis=-1).reshape(-1, 2)
    return out
