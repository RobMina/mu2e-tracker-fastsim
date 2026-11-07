"""Local geometric helix/wire residuals, not a drift or time-to-distance fit.

Only NumPy is required. All lanes are evaluated together; loops are bounded
Newton/backtracking iterations, never loops over events or wires.
"""

import numpy as np


_EPS = np.finfo(np.float64).eps


def _real(name, value):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric")
    return a.astype(np.float64, copy=False)


def _seed_column(seed, name, vector=False):
    try:
        a = _real(name, seed[name])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing seed parameter {name}") from exc
    if vector:
        if a.ndim < 1 or a.shape[-1] != 2:
            raise ValueError("center_xy must end in dimension 2")
        if a.ndim == 1:
            a = a[None, None, :]
        elif a.ndim == 2:
            a = a[:, None, :]
        if a.ndim != 3 or a.shape[1] != 1:
            raise ValueError("center_xy must have shape (2,), (batch,2) or (batch,1,2)")
    else:
        if a.ndim == 0:
            a = a[None, None]
        elif a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2 or a.shape[1] != 1:
            raise ValueError(f"{name} must be scalar, (batch,) or (batch,1)")
    return a


def wire_residuals(seed, mid, direction, measured_u, uvar, hlen, *,
                   max_iterations=12, max_step=50., tolerance=1e-6):
    """Find a bounded LOCAL minimum of distance to an infinite straight wire.

    Units are mm and radians. ``seed`` is a mapping with ``center_xy``, positive
    ``radius``, nonzero signed ``pitch = dz/dphase``, ``phase_ref`` and ``z_ref``:
    h(z) = (cx + R*cos(phi), cy + R*sin(phi), z),
    phi = phase_ref + (z-z_ref)/pitch. Scalar parameters may be scalars,
    [batch] or [batch,1]; center_xy is [2], [batch,2] or [batch,1,2]. Optional
    boolean ``success`` has the scalar-parameter shapes. Thus fit_helix_seed's
    dictionary can be passed directly; other entries are ignored. Invalid or
    failed seeds are masked per lane, not allowed to contaminate other lanes.

    ``mid`` and ``direction`` end in dimension 3, with zero, one (hits), or two
    (batch,hits) leading axes. Directions are normalized in FULL 3D (tilts are
    retained), so measured_u and hlen are physical along-wire distances, NOT
    coefficients of the supplied possibly nonunit direction. ``measured_u``,
    positive ``uvar`` and nonnegative ``hlen`` are scalar, [hits] or [batch,hits].
    All inputs broadcast to [batch,hits]; per-batch wire data need a singleton
    hit axis. Empty batches/hit axes are supported. Bad shapes/types/config,
    nonfinite wire inputs and zero directions raise ValueError even on empty
    outputs. Nonfinite/nonphysical seed values instead produce valid=False.

    Start at z0 = mid.z + measured_u*unit_direction.z. Eliminate the infinite
    line coordinate analytically: u(z) = dot(h(z)-mid, unit_direction). Newton
    minimizes f(z) = |h(z)-mid-u(z)*unit_direction|**2 / 2. Its trust region is
    |z-z0| <= pi*abs(pitch)/2; each step is also <= min(max_step, pi*abs(pitch)/8).
    This measured-point local basin prevents arbitrary turn hopping, but is NOT
    a proof of a unique minimum or a universal/global POCA. Starting in a
    nonpositive-curvature region is conservatively rejected, not rescued by a
    distant search. No more than max_iterations Newton steps, each with at most
    ten halved Armijo trials, are attempted. Only positive-curvature, decreasing
    trials (up to floating-point roundoff) are accepted.

    A strict minimum requires hessian > 128*eps*(|h'(z)|**2 + |r.h''(z)|).
    Convergence ALSO requires the UNCLIPPED Newton correction <= tolerance and
    |gradient| <= tolerance*max(1, |projected h'(z)|**2). Thus a stationary
    maximum, numerically flat direction, clipped/boundary stall, failed line
    search, or exhausted iteration budget is NOT automatically convergence.
    max_iterations=0 only checks the starting point. tolerance is in z-mm.

    Returns a dict of [batch,hits] arrays:
      doca: unsigned distance to the infinite wire;
      signed_doca: doca signed by dot(r, cross(h'(z), unit_direction)), with
        h'(z) pointing toward increasing z, NOT a flight/charge convention;
      sign_valid: the above normal is numerically defined (otherwise signed_doca
        is zero; unsigned doca remains usable);
      wire_u, wire_pull: predicted along-wire mm and (wire_u-measured_u)/sqrt(uvar);
      z, converged, inside: final z, strict local-minimum status, |wire_u| <= hlen;
      valid: seed and final numeric payload are usable;
      iterations: number of attempted Newton updates (not backtracking trials).

    Finite-wire ``inside`` is a separate geometric diagnostic: this routine does
    NOT minimize distance to endpoints. Association must require valid &
    converged & inside, and apply its own geometric/pull cuts. Unconverged lanes
    retain the last finite iterate where possible; invalid lanes have zero
    numeric payload and false flags. No drift radius, T2D, covariance, or fit
    quality is inferred from these purely geometric residuals.
    """
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer))
            or not 0 <= max_iterations <= 2**31 - 1):
        raise ValueError("max_iterations must be a nonnegative integer <= 2**31-1")
    for name, value in (("max_step", max_step), ("tolerance", tolerance)):
        a = _real(name, value)
        if a.ndim != 0 or not np.isfinite(a) or a <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
    max_step, tolerance = float(max_step), float(tolerance)

    center = _seed_column(seed, "center_xy", vector=True)
    params = {name: _seed_column(seed, name)
              for name in ("radius", "pitch", "phase_ref", "z_ref")}
    try:
        success = np.asarray(seed.get("success", True))
    except AttributeError as exc:
        raise ValueError("seed must be a mapping") from exc
    if success.dtype.kind != "b":
        raise ValueError("seed success must be boolean")
    if success.ndim == 0:
        success = success[None, None]
    elif success.ndim == 1:
        success = success[:, None]
    if success.ndim != 2 or success.shape[1] != 1:
        raise ValueError("seed success must be scalar, (batch,) or (batch,1)")

    wires = {}
    for name, value in (("mid", mid), ("direction", direction),
                        ("measured_u", measured_u), ("uvar", uvar), ("hlen", hlen)):
        a = _real(name, value)
        if name in ("mid", "direction"):
            if a.ndim not in (1, 2, 3) or a.shape[-1] != 3:
                raise ValueError(f"{name} must have shape (...,3), at most two leading axes")
        elif a.ndim > 2:
            raise ValueError(f"{name} must have at most two axes")
        if not np.isfinite(a).all():
            raise ValueError(f"{name} must be finite")
        wires[name] = a
    if np.any(wires["uvar"] <= 0) or np.any(wires["hlen"] < 0):
        raise ValueError("uvar must be positive and hlen nonnegative")
    scale = np.max(np.abs(wires["direction"]), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("direction must be nonzero")
    # Scaling first avoids overflow/underflow even for extreme direction magnitudes.
    d = wires["direction"] / scale
    d /= np.sqrt(np.sum(d*d, axis=-1, keepdims=True))
    try:
        shape = np.broadcast_shapes(center.shape[:-1], success.shape,
                                    *(p.shape for p in params.values()),
                                    wires["mid"].shape[:-1], d.shape[:-1],
                                    *(wires[n].shape for n in ("measured_u", "uvar", "hlen")))
    except ValueError as exc:
        raise ValueError("seed and wire shapes do not broadcast to [batch,hits]") from exc
    center = np.broadcast_to(center, shape + (2,))
    params = {name: np.broadcast_to(a, shape) for name, a in params.items()}
    mid = np.broadcast_to(wires["mid"], shape + (3,))
    d = np.broadcast_to(d, shape + (3,))
    measured_u, uvar, hlen = (np.broadcast_to(wires[n], shape)
                             for n in ("measured_u", "uvar", "hlen"))
    valid = np.broadcast_to(success, shape).copy() & np.isfinite(center).all(axis=-1)
    for a in params.values():
        valid &= np.isfinite(a)
    valid &= (params["radius"] > 0) & (params["pitch"] != 0)
    center = np.where(valid[..., None], center, 0.)
    radius = np.where(valid, params["radius"], 1.)
    pitch = np.where(valid, params["pitch"], 1.)
    phase_ref = np.where(valid, params["phase_ref"], 0.)
    z_ref = np.where(valid, params["z_ref"], 0.)

    # Mask overflowed geometry and retain the last finite iterate on trial failure.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        z0 = mid[..., 2] + measured_u*d[..., 2]
        phi0 = phase_ref + (z0-z_ref)/pitch
        slope = radius/pitch
        accel = slope/pitch
        basin = (np.pi/2)*np.abs(pitch)
        step_limit = np.minimum(max_step, (np.pi/8)*np.abs(pitch))

        def evaluate(offset):
            phi = phi0 + offset/pitch
            c, s = np.cos(phi), np.sin(phi)
            delta = np.stack((center[..., 0]-mid[..., 0] + radius*c,
                              center[..., 1]-mid[..., 1] + radius*s,
                              z0-mid[..., 2] + offset), axis=-1)
            tangent = np.stack((-slope*s, slope*c, np.ones(shape)), axis=-1)
            second = np.stack((-accel*c, -accel*s, np.zeros(shape)), axis=-1)
            u = np.sum(delta*d, axis=-1)
            r = delta-u[..., None]*d
            tp = tangent-np.sum(tangent*d, axis=-1)[..., None]*d
            f = .5*np.sum(r*r, axis=-1)
            # Coordinate arithmetic sets the descent slack near a nonzero residual minimum.
            arithmetic_scale = np.maximum(radius, np.max(np.abs(delta), axis=-1))
            f_roundoff = (64*_EPS*arithmetic_scale)*(np.sqrt(2*f)
                                                     + _EPS*arithmetic_scale)
            g = np.sum(r*tp, axis=-1)
            speed2 = np.sum(tp*tp, axis=-1)
            curvature = np.sum(r*second, axis=-1)
            h = speed2 + curvature
            hscale = np.sum(tangent*tangent, axis=-1) + np.abs(curvature)
            finite = (np.isfinite(f) & np.isfinite(g) & np.isfinite(h)
                      & np.isfinite(hscale) & np.isfinite(u)
                      & np.isfinite(f_roundoff) & np.isfinite(z0+offset))
            positive = finite & (h > 128*_EPS*hscale)
            correction = g/np.where(positive, h, 1.)
            stationary = (positive & (np.abs(correction) <= tolerance)
                          & (np.abs(g) <= tolerance*np.maximum(1., speed2)))
            return dict(f=f, g=g, h=h, correction=correction, positive=positive,
                        stationary=stationary, finite=finite, u=u, r=r, tangent=tangent,
                        f_roundoff=f_roundoff)

        offset = np.zeros(shape)
        state = evaluate(offset)
        valid &= state["finite"]
        active = valid & state["positive"] & ~state["stationary"]
        iterations = np.zeros(shape, dtype=np.int32)
        for _ in range(max_iterations):
            if not np.any(active):
                break
            iterations += active
            step = np.clip(-state["correction"], -step_limit, step_limit)
            step = np.clip(step, -basin-offset, basin-offset)
            accepted = np.zeros(shape, bool)
            next_offset = offset.copy()
            for _ in range(10):
                trial_offset = offset + np.where(active, step, 0.)
                trial = evaluate(trial_offset)
                armijo = state["f"] + 1e-4*state["g"]*step
                descent = trial["f"] <= armijo + state["f_roundoff"] + trial["f_roundoff"]
                take = (active & ~accepted & trial["positive"] & descent
                        & (trial_offset != offset) & (state["g"]*step < 0))
                next_offset = np.where(take, trial_offset, next_offset)
                accepted |= take
                if np.all(~active | accepted):
                    break
                step *= .5
            offset = next_offset
            state = evaluate(offset)
            active &= accepted & state["positive"] & ~state["stationary"]

        normal = np.cross(state["tangent"], d)
        normal2 = np.sum(normal*normal, axis=-1)
        sign_valid = valid & (normal2 > 128*_EPS*np.sum(state["tangent"]**2, axis=-1))
        orientation = np.sum(state["r"]*normal, axis=-1)
        doca = np.sqrt(2*state["f"])
        pull = (state["u"]-measured_u)/np.sqrt(uvar)
        valid &= np.isfinite(doca) & np.isfinite(pull)
        sign_valid &= valid & np.isfinite(orientation)
        values = dict(doca=doca,
                      signed_doca=np.where(sign_valid, np.where(orientation < 0, -doca, doca), 0.),
                      wire_u=state["u"], wire_pull=pull, z=z0+offset)
    out = {name: np.where(valid, a, 0.) for name, a in values.items()}
    out.update(valid=valid, converged=valid & state["stationary"],
               inside=valid & (np.abs(state["u"]) <= hlen),
               sign_valid=sign_valid, iterations=iterations)
    return out
