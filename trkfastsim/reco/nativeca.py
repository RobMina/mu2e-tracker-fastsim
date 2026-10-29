"""Vectorized local KL/SensorLine CA reconstruction, not captured native state.

Source control flow follows ClosestApproach::findTCA with 100 masked time updates.
No piece search, hit-z retry, prior usable hint recovery, or ROOT arithmetic parity.
"""
import numpy as np

from . import nativemeasurement as N


def _vector(value, shape, name):
    return N._finite(value, shape, name)


def particle(parameters, time, mass=N.MASS):
    """KinematicLine position/direction/speed with native expression ordering."""
    q = np.asarray(parameters, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 6 or not np.all(np.isfinite(q)) or np.any(q[..., 5] <= 0):
        raise ValueError("invalid native particle parameters")
    t = _vector(time, q.shape[:-1], "particle time")
    d, phi, z, theta, t0, p = np.moveaxis(q, -1, 0)
    u = np.stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)), -1)
    origin = np.stack((-d*np.sin(phi), d*np.cos(phi), z), -1)
    mass = np.asarray(mass, dtype=float)
    if mass.shape not in ((), q.shape[:-1]) or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("invalid particle mass")
    v = p/np.sqrt(p*p+mass*mass)*N.LIGHT
    with np.errstate(over="ignore", invalid="ignore"):
        position = origin+((t-t0)*v)[..., None]*u
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(v)):
        raise ValueError("nonfinite particle arithmetic; failure-state propagation unsupported")
    return position, u, v


def sensor(early, late, measurement_time, signal_speed):
    """Directed line formed from the dumped wire endpoints, not inferred straw axes."""
    early, late = np.asarray(early), np.asarray(late)
    if early.ndim < 1 or early.shape[-1] != 3:
        raise ValueError("invalid sensor endpoint shape")
    shape = early.shape[:-1]
    early = _vector(early, shape+(3,), "early endpoint")
    late = _vector(late, shape+(3,), "late endpoint")
    mt = _vector(measurement_time, shape, "measurement time")
    vs = _vector(signal_speed, shape, "sensor speed")
    length = np.linalg.norm(early-late, axis=-1)
    if np.any(length <= 0) or np.any(vs <= 0):
        raise ValueError("sensor length and speed must be positive")
    return {"end": early, "mid": .5*(early+late), "direction": (early-late)/length[..., None],
            "length": length, "measurement_time": mt, "speed": vs}


def sensor_position(line, time):
    time = _vector(time, line["measurement_time"].shape, "sensor time")
    with np.errstate(over="ignore", invalid="ignore"):
        position = line["end"]-((line["measurement_time"]-time)*line["speed"])[..., None]*line["direction"]
    if not np.all(np.isfinite(position)):
        raise ValueError("nonfinite sensor arithmetic; failure-state propagation unsupported")
    return position


def constructor_hint(q, line, corrected_time, mass=N.MASS):
    """KKFit::makeStrawHits z-adjustment from the measured ComboHit corrected time."""
    time = _vector(corrected_time, np.asarray(q).shape[:-1], "corrected time")
    pp, u, v = particle(q, time, mass)
    sp = sensor_position(line, time)
    vz = u[..., 2]*v
    dztime = np.divide(sp[..., 2]-pp[..., 2], vz, out=np.zeros_like(time), where=vz != 0)
    return time+dztime, time.copy()


def solve(q, line, particle_hint, sensor_hint, precision=1e-4, max_updates=100, mass=N.MASS):
    """Reconstruct a local CA; return retained times/positions, status and update counts.

    Status is converged=0, unconverged=1, parallel=4, failed=5. Input arrays must be
    finite; arithmetic overflow aborts the audit, not native failure propagation.
    Precision is a time-update threshold as used in native code, despite
    the legacy FCL comment labelling it mm. No extra convergence test is introduced.
    """
    if not np.isfinite(precision) or precision <= 0 or isinstance(max_updates, bool) or not isinstance(max_updates, int) or not 1 <= max_updates <= 100:
        raise ValueError("invalid CA precision/update cap")
    q = np.asarray(q, dtype=np.float64)
    shape = q.shape[:-1]
    pt = _vector(particle_hint, shape, "particle hint").copy()
    st = _vector(sensor_hint, shape, "sensor hint").copy()
    pp, u, v = particle(q, pt, mass)
    sp = sensor_position(line, st)
    s = line["direction"]
    if s.shape != u.shape:
        raise ValueError("particle/sensor batch shapes differ")
    dot = np.sum(s*u, -1)
    den = 1-dot*dot
    parallel = den < 1e-5
    status = np.where(parallel, 4, 1).astype(np.int32)
    active = ~parallel
    updates = np.zeros(shape, np.int32)
    dpt = np.full(shape, np.inf)
    dst = np.full(shape, np.inf)
    safe_den = np.where(parallel, 1., den)
    for _ in range(max_updates):
        if not np.any(active):
            break
        delta = sp-pp
        pdd, sdd = np.sum(delta*u, -1), np.sum(delta*s, -1)
        a = (pdd-sdd*dot)/(safe_den*v)
        b = (pdd*dot-sdd)/(safe_den*line["speed"])
        dpt, dst = np.where(active, a, dpt), np.where(active, b, dst)
        pt, st = np.where(active, pt+a, pt), np.where(active, st+b, st)
        pp, _, _ = particle(q, pt, mass)
        sp = sensor_position(line, st)
        updates += active
        finite = np.isfinite(pp).all(-1) & np.isfinite(sp).all(-1) & np.isfinite(a) & np.isfinite(b)
        status = np.where(active & ~finite, 5, status)
        active &= finite & ((np.abs(dpt) > precision) | (np.abs(dst) > precision))
    converged = (status < 4) & (np.abs(dpt) < precision) & (np.abs(dst) < precision)
    status = np.where(converged, 0, status)
    return {"particle_time": pt, "sensor_time": st, "particle_poca": pp, "sensor_poca": sp,
            "delta": sp-pp, "direction": u, "sensor_direction": s, "speed": v,
            "status": status, "updates": updates, "particle_update": dpt, "sensor_update": dst,
            "usable": status < 3}


def rows_at_state(q, line, state, mass=N.MASS):
    """Native source rows evaluated on the reconstructed retained CA vector."""
    if not np.all(state["usable"]):
        raise ValueError("unusable local CA state")
    _, u, v, dx, dm = N.kinematics(q, state["particle_time"], mass)
    s, delta = line["direction"], state["delta"]
    dot = np.sum(u*s, -1)
    den = 1-dot*dot
    norm = np.linalg.norm(delta, axis=-1)
    if np.any(norm == 0) or np.any(den < 1e-5):
        raise ValueError("undefined zero-distance or parallel CA derivative")
    sign = np.copysign(1., np.sum(np.cross(s, u)*delta, -1))
    Hdoca = np.einsum("...i,...ij->...j", sign[..., None]*delta/norm[..., None], dx)
    beta = (s*dot[..., None]-u)/(v*den)[..., None]
    gamma = (s-u*dot[..., None])/(line["speed"]*den)[..., None]
    Htime = -np.einsum("...i,...ij->...j", beta-gamma, dx)
    ds, dp = np.einsum("...i,...ij->...j", s, dx), np.einsum("...i,...ij->...j", u, dx)
    ddm = np.einsum("...i,...ij->...j", delta, dm)
    Hlong = ds+(dot[..., None]*ds-dp-ddm)/den[..., None]*dot[..., None]
    H = np.stack((Hdoca, Htime, Hlong), -2)
    if not np.all(np.isfinite(H)):
        raise ValueError("nonfinite retained-state derivative")
    return {"H": H, "doca": sign*norm,
            "delta_dot_particle": np.sum(delta*u, -1), "delta_dot_sensor": np.sum(delta*s, -1),
            "longitudinal": np.sum((state["sensor_poca"]-line["mid"])*s, -1)}
