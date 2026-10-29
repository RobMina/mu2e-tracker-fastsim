"""Source-derived KL measurement derivatives, for saved-operand diagnostics only.

Reproduce the formulas in KinematicLine::{dXdPar,dMdPar}, ClosestApproach
and KKStrawHit::setResiduals at an analytic line-line CA. This is not the
native iterative solver or its retained-state/hint policy. Coordinates are
(d0, phi0, z0, theta, t0, momentum), with unrotated KinematicLine geometry.
Returned H uses F1 residual convention: (-doca or lr*rDrift-doca, TOT, long).
The drift radius is frozen (dDdT=0); no hit-state or variance policy is changed.
"""
import numpy as np

LIGHT = 299.792458
MASS = 105.6583755


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(name + ": invalid shape or nonfinite values")
    return value


def kinematics(parameters, time, mass=MASS):
    """Return position, direction, speed, dXdPar and dMdPar/momentum at fixed time."""
    q = np.asarray(parameters, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 6 or not np.all(np.isfinite(q)):
        raise ValueError("native parameters: invalid shape or nonfinite values")
    shape = q.shape[:-1]
    mass = np.asarray(mass, dtype=float)
    if mass.shape not in ((), shape) or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("invalid particle mass")
    t = _finite(time, shape, "particle time")
    d0, phi, z0, theta, t0, mom = np.moveaxis(q, -1, 0)
    if np.any(mom <= 0):
        raise ValueError("native momentum must be positive")
    sf, cf, st, ct = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)
    zero = np.zeros(shape)
    axis = np.stack((-sf, cf, zero), -1)
    direction = np.stack((st*cf, st*sf, ct), -1)
    theta_axis = np.stack((ct*cf, ct*sf, -st), -1)
    energy = np.hypot(mom, mass)
    speed = LIGHT * mom / energy
    dt = t - t0
    length = speed * dt
    origin = np.stack((-d0*sf, d0*cf, z0), -1)
    position = origin + length[..., None] * direction
    dx = np.zeros(shape + (3, 6))
    dx[..., :, 0] = axis
    dx[..., :, 1] = (st*length)[..., None]*axis - d0[..., None]*np.stack((cf, sf, zero), -1)
    dx[..., 2, 2] = 1.0
    dx[..., :, 3] = length[..., None]*theta_axis
    dx[..., :, 4] = -speed[..., None]*direction
    dx[..., :, 5] = -(dt*mass**2/(energy**2*mom))[..., None]*dx[..., :, 4]
    dm = np.zeros_like(dx)
    dm[..., :, 1] = st[..., None]*axis
    dm[..., :, 3] = theta_axis
    dm[..., :, 5] = direction/mom[..., None]
    if not all(np.all(np.isfinite(v)) for v in (position, direction, speed, dx, dm)):
        raise ValueError("nonfinite native kinematics")
    return position, direction, speed, dx, dm


def rows(parameters, midpoint, wire_direction, end_sign, signal_speed):
    """Return source-policy H and CA diagnostics for matching leading batch axes.

    No floor, normalization repair, or regularization: require unit wire vectors,
    end signs +/-1, positive signal speed and native nonparallel denominator.
    The exact full time derivative's omitted direction term is returned separately.
    Residual values and weights must remain the saved F1 operands in an isolation test.
    """
    q = np.asarray(parameters, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 6:
        raise ValueError("native parameters: invalid shape")
    shape = q.shape[:-1]
    mid = _finite(midpoint, shape + (3,), "wire midpoint")
    wire = _finite(wire_direction, shape + (3,), "wire direction")
    sign = _finite(end_sign, shape, "end sign")
    vs = _finite(signal_speed, shape, "signal speed")
    if np.any(np.abs(np.sum(wire*wire, -1)-1.0) > 1e-10):
        raise ValueError("wire direction must be unit length")
    if np.any(np.abs(sign) != 1) or np.any(vs <= 0):
        raise ValueError("invalid end sign or nonpositive signal speed")
    origin, pdir, speed, _, _ = kinematics(q, q[..., 4])
    sdir = wire*sign[..., None]
    dot = np.sum(sdir*pdir, -1)
    den = 1-dot*dot
    if np.any(den < 1e-5):
        raise ValueError("parallel native CA denominator")
    dl = origin-mid
    du, dw = np.sum(dl*pdir, -1), np.sum(dl*sdir, -1)
    particle_length = (dw*dot-du)/den
    sensor_length = (dw-du*dot)/den
    time = q[..., 4]+particle_length/speed
    pp, _, _, dx, dm = kinematics(q, time)
    sp = mid+sdir*sensor_length[..., None]
    delta = sp-pp
    distance = np.linalg.norm(delta, axis=-1)
    if np.any(distance == 0):
        raise ValueError("zero-distance CA has undefined DOCA direction")
    lsign = np.copysign(1.0, np.sum(np.cross(sdir, pdir)*delta, -1))
    doca_prediction = np.einsum("...i,...ij->...j", (lsign/distance)[..., None]*delta, dx)
    beta = (sdir*dot[..., None]-pdir)/(speed*den)[..., None]
    gamma = (sdir-pdir*dot[..., None])/(vs*den)[..., None]
    time_prediction = np.einsum("...i,...ij->...j", beta-gamma, dx)
    ds = np.einsum("...i,...ij->...j", sdir, dx)
    dp = np.einsum("...i,...ij->...j", pdir, dx)
    ddm = np.einsum("...i,...ij->...j", delta, dm)
    long_prediction = -(ds+(ds*dot[..., None]-dp-ddm)/den[..., None]*dot[..., None])
    H = np.stack((doca_prediction, -time_prediction, -long_prediction), -2)
    normal = np.cross(sdir, pdir)/np.sqrt(den)[..., None]
    orthogonal_doca_H = np.einsum("...i,...ij->...j", normal, dx)
    du_dp = dm.copy()
    du_dp[..., :, 5] = 0.0
    delta_du = np.einsum("...i,...ij->...j", delta, du_dp)
    omitted_time = ((dot/vs-1/speed)/den)[..., None]*delta_du
    full_minus_native_long = (2*dot/den)[..., None]*delta_du
    if not all(np.all(np.isfinite(v)) for v in (H, omitted_time, full_minus_native_long)):
        raise ValueError("nonfinite source-policy derivatives")
    return {"H": H, "omitted_time_direction": omitted_time, "full_minus_native_long": full_minus_native_long,
            "particle_time": time,
            "particle_poca": pp, "sensor_poca": sp, "delta": delta,
            "doca": lsign*distance, "sensor_length": sensor_length, "direction_dot": dot,
            "orthogonal_doca_H": orthogonal_doca_H,
            "delta_dot_particle": np.sum(delta*pdir, -1), "delta_dot_sensor": np.sum(delta*sdir, -1)}


def information(parameters, H, residual, weight):
    """Form W and b in one native chart for residual r+H*(q-qref)."""
    q = np.asarray(parameters, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 6 or not np.all(np.isfinite(q)):
        raise ValueError("native parameters: invalid shape or nonfinite values")
    H = _finite(H, q.shape[:-1]+(3, 6), "H")
    r = _finite(residual, q.shape[:-1]+(3,), "residual")
    w = _finite(weight, q.shape[:-1]+(3,), "weight")
    if np.any(w < 0):
        raise ValueError("negative measurement weight")
    ht = H.swapaxes(-1, -2)*w[..., None, :]
    W = ht@H
    b = np.einsum("...ij,...j->...i", ht, np.einsum("...ij,...j->...i", H, q)-r)
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
        raise ValueError("nonfinite measurement information")
    return W, b
