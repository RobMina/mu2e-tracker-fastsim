"""Opt-in source-level Run1A StrawResponse::driftInfo evaluator.

This ports the pinned 20261106 StrawDriftMaker/StrawDrift/StrawResponse
semantics, not a fit to oracle responses. Supply resolved FHiCL sub-mappings,
nominal geometry in mm and the exact detector-origin Bz in tesla. The origin
field is not the trajectory field used to construct a hit's Lorentz angle.
No digitiser, reconstruction default or geometric wire fit imports this module.

Native grids are bounded, immutable NumPy arrays. Queries broadcast, including
scalars and empty arrays; output values are float64 arrays of the broadcast
shape. Only interval indices are clamped: negative times, radii and errors and
large tails are not clipped. Nonfinite inputs/results fail rather than being
silently repaired. This implements source semantics, not cross-libm bitwise
identity or validation of a physical fit, field-map tracking or multiple IOVs.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

MAX_NODES = 100_000
MAX_CELLS = 2_000_000


@dataclass(frozen=True)
class _Calibration:
    low: float
    width: float
    intercept: np.ndarray
    slope: np.ndarray


@dataclass(frozen=True)
class DriftModel:
    """Native stock grids and local-regression coefficients, constructed by build_model.

    ``cc`` is the public stock GetCC value, after float Bz and float CC rounding
    and the final double multiplication by 1000. ``bz`` retains the supplied
    exact origin Bz; ``bz_float`` records the maker's rounded value.
    """

    cc: float
    bz: float
    bz_float: float
    wire_radius: float
    outer_radius: float
    delta_d: float
    delta_t: float
    delta_phi: float
    distances: np.ndarray
    instant_speeds: np.ndarray
    times_dbins: np.ndarray
    times: np.ndarray
    distances_tbins: np.ndarray
    ignore_phi: bool
    drdt_scale: float
    offset: _Calibration
    signed: _Calibration
    unsigned: _Calibration


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable as float64") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _array(value, name, *, bounded=False):
    if bounded:
        try:
            if len(value) > MAX_NODES:
                raise ValueError(f"{name} exceeds {MAX_NODES} nodes")
        except TypeError as exc:
            raise ValueError(f"{name} must be a one-dimensional numeric array") from exc
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must contain real numeric values")
        if bounded and (raw.ndim != 1 or raw.size > MAX_NODES):
            raise ValueError(f"{name} must be a bounded one-dimensional array")
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if bounded and (array.ndim != 1 or array.size > MAX_NODES):
        raise ValueError(f"{name} must be one-dimensional with at most {MAX_NODES} nodes")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _readonly(array):
    array.setflags(write=False)
    return array


def _calibration(bins, values, name):
    """Precompute every possible GSL local fit, retaining its two online-mean passes.

    The five iterations here are over fit positions, never over queries or
    calibration bins. A sum/polyfit formulation changes GSL's rounding order.
    The final evaluation remains c0+c1*x, including extrapolation past the edges.
    """
    bins = _array(bins, name + " bins", bounded=True)
    values = _array(values, name, bounded=True)
    if bins.size != 2 or bins[1] <= bins[0] or values.size < 2:
        raise ValueError(f"{name} needs two increasing edges and at least two values")
    n = values.size
    width = (bins[1] - bins[0]) / n
    centers = bins[0] + width * (np.arange(n, dtype=np.float64) + 0.5)
    if width <= 0 or not np.all(np.diff(centers) > 0):
        raise ValueError(f"{name} has degenerate bin centers")
    indices = np.arange(n)
    start, stop = np.maximum(0, indices - 2), np.minimum(n - 1, indices + 2)
    count = stop - start + 1
    mx, my = np.zeros(n), np.zeros(n)
    for j in range(5):
        k = np.minimum(start + j, n - 1)
        active = j < count
        mx = np.where(active, mx + (centers[k] - mx) / (j + 1), mx)
        my = np.where(active, my + (values[k] - my) / (j + 1), my)
    mxx, mxy = np.zeros(n), np.zeros(n)
    for j in range(5):
        k = np.minimum(start + j, n - 1)
        active = j < count
        dx, dy = centers[k] - mx, values[k] - my
        mxx = np.where(active, mxx + (dx * dx - mxx) / (j + 1), mxx)
        mxy = np.where(active, mxy + (dx * dy - mxy) / (j + 1), mxy)
    if not np.all(mxx > 0):
        raise ValueError(f"{name} has singular local fits")
    slope = mxy / mxx
    intercept = my - mx * slope
    if not np.all(np.isfinite(slope)) or not np.all(np.isfinite(intercept)):
        raise ValueError(f"{name} has nonfinite local fits")
    return _Calibration(float(bins[0]), float(width), _readonly(intercept), _readonly(slope))


def _inverse_grid(distances, times_dbins, times, delta_d):
    """Vectorized exact maker indexing, without floating offsets between phi columns.

    Structured search keys provide independent searches of all monotone phi
    columns. The maker selects the first time STRICTLY GREATER than the target,
    then interpolates/extrapolates from that upper index and its successor, not
    from the conventional lower bracketing index. It caps the index at ndist-2.
    The first inverse row is initialized to zero instead of being searched.
    """
    nd, nphi = times_dbins.shape
    dtype = np.dtype([("phi", np.int32), ("time", np.float64)])
    native = np.empty((nphi, nd), dtype=dtype)
    native["phi"] = np.arange(nphi)[:, None]
    native["time"] = times_dbins.T
    query = np.empty((nphi, times.size - 1), dtype=dtype)
    query["phi"] = np.arange(nphi)[:, None]
    query["time"] = times[None, 1:]
    index = np.searchsorted(native.ravel(), query.ravel(), side="right").reshape(query.shape)
    index = np.minimum(index - np.arange(nphi)[:, None] * nd, nd - 2).T
    p = np.arange(nphi)[None, :]
    inverse = np.zeros((times.size, nphi), dtype=np.float64)
    inverse[1:] = distances[index] + (times[1:, None] - times_dbins[index, p]) / (
        times_dbins[index + 1, p] - times_dbins[index, p]) * delta_d
    return inverse


def build_model(drift_cfg, response_cfg, *, wire_radius, outer_radius, bz):
    """Build only from resolved stock config, nominal radii and exact origin Bz.

    Required drift keys: wireVoltage, kVcm, cmus, phiBins, deltaDistance,
    deltaTime. Required response keys: driftIgnorePhi, dRdTScale, driftOffBins,
    driftOffset, driftRMSBins, signedDriftRMS, unsignedDriftRMS. Additional keys
    are ignored; in particular useNonLinearDrift affects other stock APIs, not
    driftInfo. This is not validation of unrelated response/electronics fields.

    Fail closed on malformed, nonfinite, degenerate or nonmonotone models, or
    more than MAX_NODES nodes per axis/config vector and MAX_CELLS combined
    native distance/angle and time/angle cells. There is no grid truncation.
    """
    if not isinstance(drift_cfg, Mapping) or not isinstance(response_cfg, Mapping):
        raise ValueError("drift_cfg and response_cfg must be resolved config mappings")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return _build_model(drift_cfg, response_cfg, wire_radius, outer_radius, bz)
    except KeyError as exc:
        raise ValueError(f"missing resolved config field: {exc.args[0]}") from exc
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("unsafe numerical range in drift model configuration") from exc


def _build_model(drift, response, wire_radius, outer_radius, bz):
    wire_radius = _scalar(wire_radius, "wire_radius", positive=True)
    outer_radius = _scalar(outer_radius, "outer_radius", positive=True)
    bz = _scalar(bz, "bz")
    if wire_radius >= outer_radius:
        raise ValueError("wire_radius must be smaller than outer_radius")
    voltage = _scalar(drift["wireVoltage"], "wireVoltage", positive=True)
    delta_d = _scalar(drift["deltaDistance"], "deltaDistance", positive=True)
    delta_t = _scalar(drift["deltaTime"], "deltaTime", positive=True)
    nphi = drift["phiBins"]
    if isinstance(nphi, (bool, np.bool_)) or not isinstance(nphi, Integral) or not 2 <= nphi <= MAX_NODES:
        raise ValueError(f"phiBins must be an integer in [2, {MAX_NODES}]")
    nphi = int(nphi)
    efield = _array(drift["kVcm"], "kVcm", bounded=True) * 100.0
    speed_data = _array(drift["cmus"], "cmus", bounded=True) * 0.01
    if (efield.size < 2 or efield.size != speed_data.size or not np.all(efield > 0)
            or not np.all(np.diff(efield) > 0) or not np.all(speed_data > 0)):
        raise ValueError("kVcm must increase strictly; matching cmus must be positive, with >=2 nodes")
    ignore_phi = response["driftIgnorePhi"]
    if not isinstance(ignore_phi, (bool, np.bool_)):
        raise ValueError("driftIgnorePhi must be boolean")
    scale = _scalar(response["dRdTScale"], "dRdTScale")
    offset = _calibration(response["driftOffBins"], response["driftOffset"], "driftOffset")
    signed = _calibration(response["driftRMSBins"], response["signedDriftRMS"], "signedDriftRMS")
    unsigned = _calibration(response["driftRMSBins"], response["unsignedDriftRMS"], "unsignedDriftRMS")
    if signed.slope.size != unsigned.slope.size:
        raise ValueError("signedDriftRMS and unsignedDriftRMS sizes differ")
    log_radius = math.log(outer_radius / wire_radius)
    if not math.isfinite(log_radius) or log_radius <= 0:
        raise ValueError("unsafe radius ratio")
    bz_float = float(np.float32(bz))
    raw_cc = float(np.float32(bz_float * log_radius / voltage))
    data_distances = voltage / (efield * log_radius)
    if not np.all(data_distances > 0) or not np.all(np.diff(data_distances) < 0):
        raise ValueError("degenerate electric-field distance grid")
    if not delta_d < outer_radius or outer_radius / delta_d > MAX_NODES + 1:
        raise ValueError("distance grid is too short or exceeds node bound")
    nodes = [0.0]
    distance = delta_d
    while distance < outer_radius:
        if len(nodes) >= MAX_NODES or (len(nodes) + 1) * nphi > MAX_CELLS:
            raise ValueError("distance grid exceeds native node/cell bound")
        nodes.append(distance)
        next_distance = distance + delta_d
        if not math.isfinite(next_distance) or next_distance <= distance:
            raise ValueError("distance recurrence does not advance safely")
        distance = next_distance
    distances = np.asarray(nodes, dtype=np.float64)
    index = np.clip(np.searchsorted(-data_distances, -distances[1:], side="right") - 1,
                    0, data_distances.size - 2)
    dist0, dist1 = data_distances[index + 1], data_distances[index]
    speed0, speed1 = speed_data[index + 1], speed_data[index]
    speeds = np.empty_like(distances)
    speeds[1:] = speed0 + (speed1 - speed0) * (distances[1:] - dist0) / (dist1 - dist0)
    speeds[0] = speeds[1]
    if not np.all(speeds > 0) or not np.all(np.isfinite(speeds)):
        raise ValueError("interpolated/extrapolated instantaneous speed must stay positive and finite")
    base_time = np.add.accumulate(delta_d / speeds[1:])
    if not np.all(base_time > 0) or not np.all(np.diff(base_time) > 0):
        raise ValueError("distance-time integration is degenerate")
    delta_phi = (math.pi / 2.0) / (nphi - 1)
    vavg = distances[1:] / base_time
    c = raw_cc * vavg * 1_000_000.0
    zetta = c * distances[1:] * 0.001
    projected = zetta[:, None] * np.cos(delta_phi * np.arange(nphi))[None, :]
    gamma = (1.0 + (zetta * zetta)[:, None] / 3.0) / (1.0 + projected * projected / 3.0)
    times_dbins = np.zeros((distances.size, nphi), dtype=np.float64)
    times_dbins[1:] = base_time[:, None] * gamma
    if not np.all(np.isfinite(times_dbins)) or not np.all(np.diff(times_dbins, axis=0) > 0):
        raise ValueError("native distance-time columns must be finite and strictly increasing")
    count = float(np.max(times_dbins)) / delta_t
    if not math.isfinite(count) or count > MAX_NODES:
        raise ValueError("time grid exceeds node bound")
    nt = math.ceil(count)
    if nt < 2 or (nt + distances.size) * nphi > MAX_CELLS:
        raise ValueError("time grid is too short or exceeds native cell bound")
    times = delta_t * np.arange(nt, dtype=np.float64)
    if not np.all(np.diff(times) > 0):
        raise ValueError("degenerate time grid")
    inverse = _inverse_grid(distances, times_dbins, times, delta_d)
    if not np.all(np.isfinite(inverse)):
        raise ValueError("nonfinite inverse grid")
    return DriftModel(raw_cc * 1000.0, bz, bz_float, wire_radius, outer_radius,
                      delta_d, delta_t, delta_phi, _readonly(distances), _readonly(speeds),
                      _readonly(times_dbins), _readonly(times), _readonly(inverse), bool(ignore_phi),
                      scale, offset, signed, unsigned)


def _index(value, low, width, maximum):
    with np.errstate(over="ignore"):
        coordinate = np.floor((value - low) / width)
    return np.clip(coordinate, 0, maximum).astype(np.intp)


def _evaluate_calibration(calibration, value):
    index = _index(value, calibration.low, calibration.width, calibration.slope.size - 1)
    slope = calibration.slope[index]
    return calibration.intercept[index] + slope * value, slope


def drift_info(model, dt, phi):
    """Evaluate stock driftInfo in a fully vectorized broadcast, without query loops.

    used_phi is the original supplied angle, or zero for driftIgnorePhi, not the
    internally folded angle. The nonlinear native inverse is always used. No
    interpolation of final response values or numerical velocity derivative is
    substituted for the stock composition.
    """
    if not isinstance(model, DriftModel):
        raise ValueError("model must be returned by build_model")
    dt = _array(dt, "dt")
    phi = _array(phi, "phi")
    try:
        dt, phi = np.broadcast_arrays(dt, phi)
    except ValueError as exc:
        raise ValueError("dt and phi must broadcast") from exc
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            used = np.zeros_like(phi) if model.ignore_phi else phi.copy()
            folded = np.abs(np.fmod(used, math.pi))
            folded = np.where(folded > math.pi / 2, math.pi - folded, folded)
            p0 = np.floor(folded / model.delta_phi).astype(np.intp)
            p1 = np.minimum(np.ceil(folded / model.delta_phi).astype(np.intp),
                            model.times_dbins.shape[1] - 1)
            t = _index(dt, 0.0, model.delta_t, model.times.size - 2)
            fraction = (dt - model.times[t]) / model.delta_t
            inverse = model.distances_tbins
            lower = inverse[t, p0] + fraction * (inverse[t + 1, p0] - inverse[t, p0])
            upper = inverse[t, p1] + fraction * (inverse[t + 1, p1] - inverse[t, p1])
            cdrift = np.where(folded == 0, lower, lower + (folded - p0 * model.delta_phi) /
                             model.delta_phi * (upper - lower))
            correction, slope = _evaluate_calibration(model.offset, cdrift)
            rdrift = cdrift - correction
            d = _index(cdrift, 0.0, model.delta_d, model.distances.size - 2)
            speed = model.instant_speeds[d] + (cdrift - model.distances[d]) / model.delta_d * (
                model.instant_speeds[d + 1] - model.instant_speeds[d])
            velocity = speed * (1.0 - slope) * model.drdt_scale
            signed, _ = _evaluate_calibration(model.signed, rdrift)
            unsigned, _ = _evaluate_calibration(model.unsigned, rdrift)
            result = dict(used_phi=used, c_drift=cdrift, r_drift=rdrift,
                          drift_velocity=velocity, signed_error=signed, unsigned_error=unsigned)
    except FloatingPointError as exc:
        raise ValueError("query extrapolation produced unsafe/nonfinite arithmetic") from exc
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("query extrapolation produced nonfinite results")
    return {key: np.asarray(value) for key, value in result.items()}
