"""Fixed source-grouped arithmetic diagnostics; not ROOT-exact reconstruction.

This module is isolated from the fitter and from the CA solver. It evaluates the
retained state supplied by the caller, without recomputing CA or selecting an
arithmetic variant against native results. KinematicLine scalar expressions keep
the source grouping. Three-component contractions are explicitly left-to-right.

The captured build's ROOT headers implement Cartesian3D Unit as reciprocal
multiplication: ``delta * (1 / sqrt((x*x + y*y) + z*z))``. Header provenance is
bound by the operand driver. ROOT SMatrix contractions, Similarity, compiler
fusion and transcendental implementations are not certified. Consequently these
are source-grouping arithmetic diagnostics, not bitwise native claims.
"""
from functools import wraps

import numpy as np

from . import nativemeasurement as N

LIGHT = N.LIGHT
MASS = N.MASS
ARITHMETIC_DIAGNOSTIC = "fixed source grouping including ROOT Cartesian3D Unit; SMatrix/compiler parity uncertified"


def _checked(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                return function(*args, **kwargs)
        except FloatingPointError as exc:
            raise ValueError("nonfinite arithmetic diagnostic") from exc
    return call


def _parameters(parameters):
    q = np.asarray(parameters, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 6 or not np.all(np.isfinite(q)):
        raise ValueError("native parameters: invalid shape or nonfinite values")
    return q


def _dot3(a, b):
    """One fixed scalar reduction, not a claim about ROOT Dot."""
    return (a[..., 0]*b[..., 0] + a[..., 1]*b[..., 1]) + a[..., 2]*b[..., 2]


def _contract3(a, matrix):
    return ((a[..., 0, None]*matrix[..., 0, :]
             + a[..., 1, None]*matrix[..., 1, :])
            + a[..., 2, None]*matrix[..., 2, :])


def _matvec6(matrix, vector):
    result = matrix[..., :, 0]*vector[..., 0, None]
    for j in range(1, 6):  # fixed parameter components, never hits
        result = result + matrix[..., :, j]*vector[..., j, None]
    return result


@_checked
def sensor(early, late, measurement_time, signal_speed):
    """Endpoint SensorLine with ROOT Cartesian3D Unit source grouping, not compiler parity."""
    from . import nativeca as C
    line = C.sensor(early, late, measurement_time, signal_speed)
    delta = np.asarray(early, dtype=float)-np.asarray(late, dtype=float)
    length = np.sqrt(_dot3(delta, delta))
    if np.any(length <= 0):
        raise ValueError("sensor length must be positive")
    return dict(line, length=length, direction=delta*(1./length)[..., None])


@_checked
def kinematics(parameters, time, mass=MASS):
    """Return position, direction, speed, dXdPar, dMdPar/p (N.kinematics API).

    Do not simplify the momentum/gamma or dMdPar-then-divide expressions: their
    rounding is the purpose of this diagnostic. Scalar or exact-batch mass only.
    """
    q = _parameters(parameters)
    shape = q.shape[:-1]
    t = N._finite(time, shape, "particle time")
    mass = np.asarray(mass, dtype=np.float64)
    if mass.shape not in ((), shape) or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("invalid particle mass")
    d0, phi, z0, theta, t0, p = np.moveaxis(q, -1, 0)
    if np.any(p <= 0):
        raise ValueError("native momentum must be positive")
    sf, cf, st, ct = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)
    zero = np.zeros(shape)
    axis = np.stack((-sf, cf, zero), -1)
    direction = np.stack((st*cf, st*sf, ct), -1)
    thetaaxis = np.stack((ct*cf, ct*sf, -st), -1)
    energy = np.sqrt(p*p + mass*mass)
    speed = (p/energy)*LIGHT
    gamma = energy/mass
    dt = t-t0
    origin = np.stack((-d0*sf, d0*cf, z0), -1)
    position = origin + (dt*speed)[..., None]*direction
    dx = np.zeros(shape+(3, 6))
    dx[..., :, 0] = axis
    dx[..., :, 1] = ((st*speed)*dt)[..., None]*axis - d0[..., None]*np.stack((cf, sf, zero), -1)
    dx[..., 2, 2] = 1.
    dx[..., :, 3] = (speed*dt)[..., None]*thetaaxis
    dx[..., :, 4] = (-speed)[..., None]*direction
    dx[..., :, 5] = (-(dt/((gamma*gamma)*p)))[..., None]*dx[..., :, 4]
    dm = np.zeros_like(dx)
    dm[..., :, 1] = ((p*st)[..., None]*axis)/p[..., None]
    dm[..., :, 3] = (p[..., None]*thetaaxis)/p[..., None]
    dm[..., :, 5] = direction/p[..., None]
    return position, direction, speed, dx, dm


@_checked
def rows_at_state(q, line, state, mass=MASS):
    """Evaluate fixed arithmetic at supplied retained CA (nativeca API).

    H order is DOCA, time, longitudinal in the F1 residual convention. No CA
    iteration, delta repair, orthogonal-normal substitution, or tolerance tuning.
    """
    q = _parameters(q)
    shape = q.shape[:-1]
    usable = N._finite(state["usable"], shape, "usable state")
    if np.any(usable != 1):
        raise ValueError("unusable local CA state")
    _, u, speed, dx, dm = kinematics(q, state["particle_time"], mass)
    s = N._finite(line["direction"], shape+(3,), "sensor direction")
    vs = N._finite(line["speed"], shape, "sensor speed")
    mid = N._finite(line["mid"], shape+(3,), "sensor midpoint")
    delta = N._finite(state["delta"], shape+(3,), "retained delta")
    sp = N._finite(state["sensor_poca"], shape+(3,), "sensor POCA")
    if np.any(np.abs(_dot3(s, s)-1.) > 1e-10):
        raise ValueError("sensor direction must be unit length")
    if np.any(vs <= 0):
        raise ValueError("sensor speed must be positive")
    dot = _dot3(s, u)
    den = 1.-dot*dot
    norm = np.sqrt(_dot3(delta, delta))
    if np.any(norm == 0) or np.any(den < 1e-5):
        raise ValueError("undefined zero-distance or parallel CA derivative")
    sign = np.copysign(1., _dot3(np.cross(s, u), delta))
    unit = delta*(1./norm)[..., None]
    Hdoca = _contract3(sign[..., None]*unit, dx)
    pfactor = 1./(speed*den)
    sfactor = 1./(vs*den)
    beta = pfactor[..., None]*(s*dot[..., None]-u)
    gamma = sfactor[..., None]*(s-dot[..., None]*u)
    Htime = -_contract3(beta-gamma, dx)
    ds, dp, ddm = _contract3(s, dx), _contract3(u, dx), _contract3(delta, dm)
    Hlong = ds + ((dot[..., None]*ds-dp-ddm)/den[..., None])*dot[..., None]
    return {"H": np.stack((Hdoca, Htime, Hlong), -2), "doca": sign*norm,
            "delta_dot_particle": _dot3(delta, u), "delta_dot_sensor": _dot3(delta, s),
            "longitudinal": _dot3(sp-mid, s)}


@_checked
def information(parameters, H, residual, weight):
    """Separate row-wise source-grouping diagnostic; return W,b as in N.

    Fix Wrow[i,j] = (Hrow[i]*w)*Hrow[j], then
    brow = Wrow @ q - (Hrow*r)*w, then ((row0+row1)+row2).
    The six-term matvec is left-to-right; ROOT Similarity and SMatrix ordering
    are NOT certified. Saved weights are used as supplied (no variance recovery).
    This intentionally does not factor b as H.T*w*(H*q-r).
    """
    q = _parameters(parameters)
    shape = q.shape[:-1]
    H = N._finite(H, shape+(3, 6), "H")
    r = N._finite(residual, shape+(3,), "residual")
    w = N._finite(weight, shape+(3,), "weight")
    if np.any(w < 0):
        raise ValueError("negative measurement weight")
    Wrow = (H*w[..., :, None])[..., :, :, None]*H[..., :, None, :]
    brow = _matvec6(Wrow, q[..., None, :]) - (H*r[..., :, None])*w[..., :, None]
    W = (Wrow[..., 0, :, :]+Wrow[..., 1, :, :])+Wrow[..., 2, :, :]
    b = (brow[..., 0, :]+brow[..., 1, :])+brow[..., 2, :]
    return W, b
