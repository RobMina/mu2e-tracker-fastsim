"""Bounded actual-solid torus occupancy diagnostic, never a production navigator.

Geometry is a local-coordinate ring torus, R > rmax >= rmin >= 0, with a
counterclockwise closed azimuthal sector in radians. The caller owns GDML unit
validation and rigid world-to-local transforms. No material precedence, Boolean
operations, hierarchy, exact crossings, or transport admission are provided.

The distance to the circular centreline is g = hypot(hypot(x, y)-R, z).
It is 1-Lipschitz in space. Midpoint distance intervals therefore certify entire
ray cells inside/outside the radial shell. Sector planes have analogous linear
interval bounds (AND for sectors <= pi, OR for reflex sectors). Approximate plane
crossings split the initial cells, but are *only subdivision hints*: all cells
still require interval certificates. Parallel/on-plane rays and angular-axis
ambiguities are not guessed inside; they remain unresolved unless radial bounds
exclude them. This deliberately can leave positive-length boundary rays unknown.

Numerics assume ordinary IEEE binary64 NumPy arithmetic/hypot/sin/cos, with no
fast-math. nextafter expands slab quotients, distance/plane bounds, and length
sums. A deliberately loose 128*eps*(1 + max(abs(origin)) + requested_length
+ R + rmax) spatial guard covers coordinate cancellation and elementary-function
roundoff; plane guards additionally cover angle reduction and trig errors.
This is a guarded floating-point diagnostic, not a machine-verified interval
library or an exact-boundary claim. Extreme magnitudes (> 1e150), angles
(|startphi| > 1e6 rad), and non-unit directions are refused. Very far origins
can make the guard dominate resolution: unresolved output is then intentional.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


_EPS = np.finfo(float).eps
_TAU = 2.0 * np.pi


@dataclass(frozen=True)
class TorusProbeResult:
    """Length bounds and disjoint, ordered positive-width interval rows.

    ``intervals`` columns: lane, s0_mm, s1_mm, status (inside/unresolved), reason.
    Certified-outside cells are omitted. Endpoints bound cells, not known
    crossings. ``lower_mm`` sums certified-inside lengths; ``upper_mm`` also
    includes unresolved lengths, with conservative summation roundoff margins.
    ``inside_count`` and ``unresolved_count`` are per-lane row counts; ``counts``
    contains global diagnostics. No successful computation implies navigation
    admission, including when there are no unresolved rows.
    """

    lower_mm: np.ndarray
    upper_mm: np.ndarray
    intervals: pd.DataFrame
    inside_count: np.ndarray
    unresolved_count: np.ndarray
    counts: dict


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _clip(o, d, length, extent, guard):
    """Outward-rounded full-torus slabs; parallel directions never divide by zero."""
    bound = np.nextafter(extent[None, :] + guard[:, None], np.inf)
    parallel = d == 0
    safe = np.where(parallel, 1.0, d)
    a = np.nextafter(-bound - o, -np.inf)
    b = np.nextafter(bound - o, np.inf)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        enter = np.nextafter(np.where(d >= 0, a, b) / safe, -np.inf)
        leave = np.nextafter(np.where(d >= 0, b, a) / safe, np.inf)
    lo = np.maximum(0.0, np.where(parallel, -np.inf, enter).max(axis=1))
    hi = np.minimum(length, np.where(parallel, np.inf, leave).min(axis=1))
    outside = (parallel & ((o < -bound) | (o > bound))).any(axis=1)
    return lo, hi, ~outside & (hi > lo)


def torus_intervals(o, d, length, *, rtor, rmin, rmax, startphi=0.0,
                    deltaphi=_TAU, min_width_mm=1e-7, max_nodes=1_000_000,
                    max_active=131072, max_rounds=64, max_lanes=100_000, coordinate_guard_mm=0.):
    """Bound actual torus occupancy on finite local rays ``o + s*d``, 0 <= s <= L.

    Origins/directions have shape (N, 3); length is a nonnegative scalar or (N,)
    in mm. Directions must be unit within 1e-12 (no silent renormalization).
    Bounds use their guarded actual speed, and reported lengths are in the ray
    parameter s. Radii are mm; 0 < deltaphi <= 2*pi, including wrapped/reflex
    sectors. Only exactly 2*pi requests a full torus; a near-full sector is not
    silently widened. Empty/zero-length lanes have zero bounds and no rows.

    Breadth-first, vectorized cells have no per-ray Python loop. Initial cells
    are AABB-clipped before adaptive work, so distant origins do not consume
    subdivision depth. coordinate_guard_mm adds an explicit nonnegative caller
    coordinate-transform uncertainty to all spatial certificates. max_nodes bounds evaluated cells, max_active bounds the
    adaptive frontier, and max_rounds bounds depth/iterations. All pending cells
    become explicit unresolved rows on a cap; no lane is dropped or declared
    transmitted. max_nodes/max_rounds may be zero. max_lanes rejects oversized
    batches before working-array allocation. Memory is O(max_lanes + max_nodes
    + max_active), including retained output. No cap exception discards partial
    geometry evidence. min_width_mm stops subdivision, not numerical uncertainty.
    """
    max_nodes = _integer(max_nodes, "max_nodes", 0)
    max_active = _integer(max_active, "max_active", 1)
    max_rounds = _integer(max_rounds, "max_rounds", 0)
    max_lanes = _integer(max_lanes, "max_lanes", 0)
    geometry = np.asarray([rtor, rmin, rmax, startphi, deltaphi, min_width_mm, coordinate_guard_mm], dtype=float)
    if not np.isfinite(geometry).all():
        raise ValueError("geometry, angles, guard and minimum width must be finite")
    rtor, rmin, rmax, startphi, deltaphi, min_width_mm, coordinate_guard_mm = geometry
    if coordinate_guard_mm < 0 or coordinate_guard_mm > 1e150:
        raise ValueError("coordinate guard must be nonnegative and <= 1e150 mm")
    if not (rtor > rmax >= rmin >= 0):
        raise ValueError("ring torus requires rtor > rmax >= rmin >= 0")
    if not (0 < deltaphi <= _TAU) or abs(startphi) > 1e6:
        raise ValueError("require 0 < deltaphi <= 2*pi and abs(startphi) <= 1e6 radians")
    if min_width_mm <= 0 or max(rtor, min_width_mm) > 1e150:
        raise ValueError("invalid minimum width or geometry magnitude > 1e150")
    o, d = np.asarray(o, dtype=float), np.asarray(d, dtype=float)
    if o.ndim != 2 or o.shape[1:] != (3,) or d.shape != o.shape:
        raise ValueError("origins and directions must have matching (N, 3) shapes")
    n = len(o)
    if n > max_lanes:
        raise ValueError("ray batch exceeds max_lanes")
    if not np.isfinite(o).all() or not np.isfinite(d).all() or np.any(np.abs(o) > 1e150):
        raise ValueError("rays must be finite and origin magnitudes <= 1e150")
    speed = np.hypot(np.hypot(d[:, 0], d[:, 1]), d[:, 2])
    if np.any(np.abs(speed - 1.0) > 1e-12):
        raise ValueError("directions must be unit vectors (absolute tolerance 1e-12)")
    speed = np.nextafter(speed + 8 * _EPS, np.inf)
    length = np.asarray(length, dtype=float)
    if length.ndim != 0 and length.shape != (n,):
        raise ValueError("length must be scalar or shape (N,)")
    if not np.isfinite(length).all() or np.any(length < 0) or np.any(length > 1e150):
        raise ValueError("length must be finite, nonnegative and <= 1e150 mm")
    length = np.broadcast_to(length, (n,))
    scale = 1.0 + np.max(np.abs(o), axis=1) + length + rtor + rmax
    guard = np.nextafter(128 * _EPS * scale + coordinate_guard_mm, np.inf)
    lo, hi, hit = _clip(o, d, length, np.array([rtor + rmax, rtor + rmax, rmax]), guard)
    lane = np.flatnonzero(hit)
    lo, hi = lo[hit], hi[hit]
    full = deltaphi == _TAU
    planes = None
    # Inward normals: cross(start, p), cross(p, end). No angle wrap branch.
    if not full:
        end = startphi + deltaphi
        planes = np.array([[-np.sin(startphi), np.cos(startphi), 0.0],
                           [np.sin(end), -np.cos(end), 0.0]])
    angle_factor = 8.0 * (1.0 + abs(startphi) + deltaphi)
    counts = dict(lanes=n, positive_length_lanes=int(np.count_nonzero(length)),
                  aabb_candidates=len(lane), aabb_excluded_lanes=int(np.count_nonzero(length) - len(lane)),
                  evaluated_nodes=0, certified_outside_cells=0, rounds=0,
                  peak_active_cells=0, cap_reached=False, coordinate_guard_mm=float(coordinate_guard_mm))
    retained = []

    def retain(ids, a, b, status, reason):
        if len(ids):
            retained.append(pd.DataFrame(dict(lane=ids, s0_mm=a, s1_mm=b,
                                              status=status, reason=reason)))

    # Preserve clipped brackets under memory pressure; plane splits use at most three cells per lane.
    if len(lane) > max_active or (not full and 3 * len(lane) > max_active):
        retain(lane, lo, hi, "unresolved", "max_active")
        counts["cap_reached"] = True
        lane, lo, hi = lane[:0], lo[:0], hi[:0]
    elif not full and len(lane):
        den = d[lane] @ planes.T
        num = -(o[lane] @ planes.T)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            cuts = num / np.where(den == 0, 1.0, den)
        cuts = np.where(den == 0, lo[:, None], cuts)
        cuts = np.clip(cuts, lo[:, None], hi[:, None])
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
        half = np.nextafter(np.maximum(mid - lo, hi - mid), np.inf)
        point = o[lane] + mid[:, None] * d[lane]
        g = np.hypot(np.hypot(point[:, 0], point[:, 1]) - rtor, point[:, 2])
        reach = np.nextafter(half * speed[lane] + guard[lane], np.inf)
        low = np.maximum(0.0, np.nextafter(g - reach, -np.inf))
        high = np.nextafter(g + reach, np.inf)
        outside = (low > rmax) | (high < rmin)
        inside = (low >= rmin) & (high <= rmax)
        if not full:
            value = point @ planes.T
            slope = np.abs(d[lane] @ planes.T)
            pad = np.nextafter(half[:, None] * slope + (guard[lane] * angle_factor)[:, None], np.inf)
            plow = np.nextafter(value - pad, -np.inf)
            phigh = np.nextafter(value + pad, np.inf)
            if deltaphi <= np.pi:
                sector_in = (plow >= 0).all(axis=1)
                sector_out = (phigh < 0).any(axis=1)
            else:
                sector_in = (plow >= 0).any(axis=1)
                sector_out = (phigh < 0).all(axis=1)
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
    # Bound summation by gamma_m and endpoint subtraction by directed width rounding.
    gl = (ni + 2) * _EPS
    gu = (ni + nu + 2) * _EPS
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
