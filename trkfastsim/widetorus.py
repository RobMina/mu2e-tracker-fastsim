"""Independent wider-spatial torus diagnostic; never a production navigator.

The preserved torusprobe subdivision/certificate algorithm is evaluated with
NumPy longdouble spatial arithmetic, but every cell endpoint, midpoint, stall
check and width termination remains on the binary64 ray-parameter lattice.
Radii and angle arguments retain their binary64 conversion and full-sector
sentinel. The sector end angle is added in binary64 before promotion; trig is
then evaluated in longdouble on these frozen angles, not on reparsed geometry
or a different pi. Callers supply local rays transformed in wider arithmetic
from exact casts of the stored binary64 placement matrices/translations.

The 128*eps spatial guard, guarded actual direction speed, sector-plane guards
and outward nextafter operations are engineering precautions, not verified
interval arithmetic. They describe the stored-binary64-placement model only;
no GDML/G4 equivalence or transform-accuracy claim follows. This is a genuine
wider-arithmetic engineering sensitivity probe, not a transport admission.
Platforms without extra longdouble mantissa precision are refused explicitly.
AABB bounds are computed in longdouble and projected outward to binary64 once;
sector cuts are binary64 subdivision hints, never boundary certificates.
Final binary64 length aggregation retains the preserved binary64 error factor.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


_LD = np.longdouble
_EPS = _LD(np.finfo(np.longdouble).eps)
_F64_EPS = np.finfo(float).eps
_TAU = 2.0 * np.pi
_ZERO, _ONE, _INF = _LD(0), _LD(1), _LD(np.inf)


@dataclass(frozen=True)
class TorusProbeResult:
    """Ordered disjoint binary64 cells and guarded per-lane length bounds.

    Intervals have lane, s0_mm, s1_mm, status and reason columns. Outside cells
    are omitted; inside/unresolved cells retain the preserved reason vocabulary.
    No result, even a wholly certified one, authorizes geometry or transport.
    """

    lower_mm: np.ndarray
    upper_mm: np.ndarray
    intervals: pd.DataFrame
    inside_count: np.ndarray
    unresolved_count: np.ndarray
    counts: dict


def require_wider():
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        raise RuntimeError("widetorus requires longdouble mantissa precision greater than binary64")


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _project(value, *, upper):
    """Directed projection of finite longdouble brackets onto binary64."""
    result = value.astype(float)
    inward = result.astype(_LD) < value if upper else result.astype(_LD) > value
    return np.where(inward, np.nextafter(result, np.inf if upper else -np.inf), result)


def _clip(o, d, length, extent, guard):
    """Wider outward slabs, followed by one outward binary64 projection."""
    bound = np.nextafter(extent[None, :] + guard[:, None], _INF)
    parallel = d == _ZERO
    safe = np.where(parallel, _ONE, d)
    a = np.nextafter(-bound - o, -_INF)
    b = np.nextafter(bound - o, _INF)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        enter = np.nextafter(np.where(d >= _ZERO, a, b) / safe, -_INF)
        leave = np.nextafter(np.where(d >= _ZERO, b, a) / safe, _INF)
    lo = np.maximum(_ZERO, np.where(parallel, -_INF, enter).max(axis=1))
    hi = np.minimum(length.astype(_LD), np.where(parallel, _INF, leave).min(axis=1))
    outside = (parallel & ((o < -bound) | (o > bound))).any(axis=1)
    hit = ~outside & (hi > lo)
    # Project only live finite brackets, avoiding overflow in discarded quotients.
    return _project(lo[hit], upper=False), _project(hi[hit], upper=True), hit


def torus_intervals(o, d, length, *, rtor, rmin, rmax, startphi=0.0,
                    deltaphi=_TAU, min_width_mm=1e-7, max_nodes=1_000_000,
                    max_active=131072, max_rounds=64, max_lanes=100_000, coordinate_guard_mm=0.):
    """Guard actual ring-torus occupancy on finite local rays o + s*d.

    API, cap behavior and result attributes match the preserved torus probe.
    Inputs have matching (N, 3) shapes; scalar/(N,) length is binary64 in mm.
    Directions must be unit within the preserved 1e-12 tolerance, without
    renormalization. Radii require rtor > rmax >= rmin >= 0; only exactly the
    binary64 2*pi sentinel denotes a full sector. coordinate_guard_mm adds
    caller uncertainty, not a replacement for the wider arithmetic guard.
    min_width_mm stops binary64 subdivision, not numerical uncertainty.
    All pending cells survive caps as unresolved rows; roundoff/stall and
    min_width reasons stay distinct. Empty/zero-length lanes retain zero bounds.
    """
    require_wider()
    max_nodes = _integer(max_nodes, "max_nodes", 0)
    max_active = _integer(max_active, "max_active", 1)
    max_rounds = _integer(max_rounds, "max_rounds", 0)
    max_lanes = _integer(max_lanes, "max_lanes", 0)
    geometry = np.asarray([rtor, rmin, rmax, startphi, deltaphi, min_width_mm], dtype=float)
    coordinate_guard_mm = np.asarray(coordinate_guard_mm, dtype=_LD)
    if not np.isfinite(geometry).all() or coordinate_guard_mm.ndim != 0 or not np.isfinite(coordinate_guard_mm):
        raise ValueError("geometry, angles, guard and minimum width must be finite scalars")
    coordinate_guard_mm = coordinate_guard_mm[()]
    rtor, rmin, rmax, startphi, deltaphi, min_width_mm = geometry
    if coordinate_guard_mm < _ZERO or coordinate_guard_mm > _LD(1e150):
        raise ValueError("coordinate guard must be nonnegative and <= 1e150 mm")
    if not (rtor > rmax >= rmin >= 0):
        raise ValueError("ring torus requires rtor > rmax >= rmin >= 0")
    if not (0 < deltaphi <= _TAU) or abs(startphi) > 1e6:
        raise ValueError("require 0 < deltaphi <= 2*pi and abs(startphi) <= 1e6 radians")
    if min_width_mm <= 0 or max(rtor, min_width_mm) > 1e150:
        raise ValueError("invalid minimum width or geometry magnitude > 1e150")
    full = deltaphi == _TAU
    short_sector = deltaphi <= np.pi
    end = startphi + deltaphi
    rtor, rmin, rmax = _LD(rtor), _LD(rmin), _LD(rmax)
    start, end, delta = _LD(startphi), _LD(end), _LD(deltaphi)
    o, d = np.asarray(o, dtype=_LD), np.asarray(d, dtype=_LD)
    if o.ndim != 2 or o.shape[1:] != (3,) or d.shape != o.shape:
        raise ValueError("origins and directions must have matching (N, 3) shapes")
    n = len(o)
    if n > max_lanes:
        raise ValueError("ray batch exceeds max_lanes")
    if not np.isfinite(o).all() or not np.isfinite(d).all() or np.any(np.abs(o) > _LD(1e150)):
        raise ValueError("rays must be finite and origin magnitudes <= 1e150")
    speed = np.hypot(np.hypot(d[:, 0], d[:, 1]), d[:, 2])
    if np.any(np.abs(speed - _ONE) > _LD(1e-12)):
        raise ValueError("directions must be unit vectors (absolute tolerance 1e-12)")
    speed = np.nextafter(speed + _LD(8) * _EPS, _INF)
    length = np.asarray(length, dtype=float)
    if length.ndim != 0 and length.shape != (n,):
        raise ValueError("length must be scalar or shape (N,)")
    if not np.isfinite(length).all() or np.any(length < 0) or np.any(length > 1e150):
        raise ValueError("length must be finite, nonnegative and <= 1e150 mm")
    length = np.broadcast_to(length, (n,))
    scale = _ONE + np.max(np.abs(o), axis=1) + length.astype(_LD) + rtor + rmax
    guard = np.nextafter(_LD(128) * _EPS * scale + coordinate_guard_mm, _INF)
    extent = np.array([rtor + rmax, rtor + rmax, rmax], dtype=_LD)
    lo, hi, hit = _clip(o, d, length, extent, guard)
    lane = np.flatnonzero(hit)
    planes = None
    if not full:
        planes = np.array([[-np.sin(start), np.cos(start), _ZERO],
                           [np.sin(end), -np.cos(end), _ZERO]], dtype=_LD)
    angle_factor = _LD(8) * (_ONE + abs(start) + delta)
    counts = dict(lanes=n, positive_length_lanes=int(np.count_nonzero(length)),
                  aabb_candidates=len(lane), aabb_excluded_lanes=int(np.count_nonzero(length) - len(lane)),
                  evaluated_nodes=0, certified_outside_cells=0, rounds=0,
                  peak_active_cells=0, cap_reached=False, coordinate_guard_mm=float(coordinate_guard_mm))
    retained = []

    def retain(ids, a, b, status, reason):
        if len(ids):
            retained.append(pd.DataFrame(dict(lane=ids, s0_mm=a, s1_mm=b,
                                              status=status, reason=reason)))

    # Preserve clipped brackets under memory pressure, including initial sector hints.
    if len(lane) > max_active or (not full and 3 * len(lane) > max_active):
        counts["peak_active_cells"] = len(lane)
        retain(lane, lo, hi, "unresolved", "max_active")
        counts["cap_reached"] = True
        lane, lo, hi = lane[:0], lo[:0], hi[:0]
    elif not full and len(lane):
        den = d[lane] @ planes.T
        num = -(o[lane] @ planes.T)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            cuts = num / np.where(den == _ZERO, _ONE, den)
        cuts = np.where(den == _ZERO, lo.astype(_LD)[:, None], cuts)
        cuts = np.clip(cuts, lo.astype(_LD)[:, None], hi.astype(_LD)[:, None]).astype(float)
        edges = np.sort(np.column_stack((lo, cuts, hi)), axis=1)
        a, b = edges[:, :-1].ravel(), edges[:, 1:].ravel()
        ids = np.repeat(lane, 3)
        keep = b > a
        lane, lo, hi = ids[keep], a[keep], b[keep]

    while len(lane):
        counts["peak_active_cells"] = max(counts["peak_active_cells"], len(lane))
        reason = None
        if counts["evaluated_nodes"] + len(lane) > max_nodes:
            reason = "max_nodes"
        elif counts["rounds"] >= max_rounds:
            reason = "max_rounds"
        if reason is not None:
            retain(lane, lo, hi, "unresolved", reason)
            counts["cap_reached"] = True
            break
        counts["evaluated_nodes"] += len(lane)
        counts["rounds"] += 1
        mid = lo + (hi - lo) * 0.5
        m, a, b = mid.astype(_LD), lo.astype(_LD), hi.astype(_LD)
        half = np.nextafter(np.maximum(m - a, b - m), _INF)
        point = o[lane] + m[:, None] * d[lane]
        g = np.hypot(np.hypot(point[:, 0], point[:, 1]) - rtor, point[:, 2])
        reach = np.nextafter(half * speed[lane] + guard[lane], _INF)
        low = np.maximum(_ZERO, np.nextafter(g - reach, -_INF))
        high = np.nextafter(g + reach, _INF)
        outside = (low > rmax) | (high < rmin)
        inside = (low >= rmin) & (high <= rmax)
        if not full:
            value = point @ planes.T
            slope = np.abs(d[lane] @ planes.T)
            pad = np.nextafter(half[:, None] * slope + (guard[lane] * angle_factor)[:, None], _INF)
            plow = np.nextafter(value - pad, -_INF)
            phigh = np.nextafter(value + pad, _INF)
            if short_sector:
                sector_in, sector_out = (plow >= _ZERO).all(axis=1), (phigh < _ZERO).any(axis=1)
            else:
                sector_in, sector_out = (plow >= _ZERO).any(axis=1), (phigh < _ZERO).all(axis=1)
            inside &= sector_in
            outside |= sector_out
        counts["certified_outside_cells"] += int(np.count_nonzero(outside))
        retain(lane[inside], lo[inside], hi[inside], "inside", "certificate")
        unknown = ~(inside | outside)
        stalled = (mid <= lo) | (mid >= hi)
        small = (hi - lo) <= min_width_mm
        terminal = unknown & (small | stalled)
        retain(lane[terminal & ~stalled], lo[terminal & ~stalled], hi[terminal & ~stalled],
               "unresolved", "min_width")
        retain(lane[terminal & stalled], lo[terminal & stalled], hi[terminal & stalled],
               "unresolved", "roundoff")
        split = unknown & ~terminal
        lane, lo, hi, mid = lane[split], lo[split], hi[split], mid[split]
        if 2 * len(lane) > max_active:
            retain(lane, lo, hi, "unresolved", "max_active")
            counts["cap_reached"] = True
            break
        lane = np.repeat(lane, 2)
        lo, hi = np.column_stack((lo, mid)).ravel(), np.column_stack((mid, hi)).ravel()

    if retained:
        rows = pd.concat(retained, ignore_index=True).sort_values(["lane", "s0_mm"], kind="stable").reset_index(drop=True)
    else:
        rows = pd.DataFrame({"lane": pd.Series(dtype="int64"), "s0_mm": pd.Series(dtype=float),
                             "s1_mm": pd.Series(dtype=float), "status": pd.Series(dtype=str),
                             "reason": pd.Series(dtype=str)})
    ids = rows.lane.to_numpy(dtype=np.int64)
    inside = rows.status.to_numpy() == "inside"
    widths = rows.s1_mm.to_numpy() - rows.s0_mm.to_numpy()
    ni = np.bincount(ids[inside], minlength=n)
    nu = np.bincount(ids[~inside], minlength=n)
    # bincount and endpoint subtraction are binary64, regardless of spatial dtype.
    gl, gu = (ni + 2) * _F64_EPS, (ni + nu + 2) * _F64_EPS
    gl, gu = gl / (1.0 - gl), gu / (1.0 - gu)
    lower = np.bincount(ids[inside], weights=np.nextafter(widths[inside], 0.0), minlength=n)
    upper = np.bincount(ids, weights=np.nextafter(widths, np.inf), minlength=n)
    lower = np.maximum(0.0, np.nextafter(lower / (1.0 + gl), -np.inf))
    upper = np.where(upper > 0, np.nextafter(upper / (1.0 - gu), np.inf), 0.0)
    upper = np.minimum(length, upper)
    counts.update(inside_cells=int(ni.sum()), unresolved_cells=int(nu.sum()),
                  unresolved_lanes=int(np.count_nonzero(nu)),
                  unresolved_reasons={str(k): int(v) for k, v in rows.loc[~inside, "reason"].value_counts().items()})
    return TorusProbeResult(lower, upper, rows, ni, nu, counts)
