"""Analytic straight-ray support navigation, not a production transport navigator."""
import numpy as np
import pandas as pd

from mu2efastsim.world import CoaxialShells


def _supported(shells):
    if hasattr(shells, "coax"):
        if getattr(shells, "calo", None) is not None:
            raise ValueError("calorimeter geometry is not supported")
        shells = shells.coax
    if not isinstance(shells, CoaxialShells):
        raise TypeError("expected CoaxialShells or an InteriorColumn with coax shells")
    if shells.other:
        raise ValueError("noncoaxial support volumes are not supported")
    volumes = []
    for g in shells.groups:
        if not np.all(np.asarray(g["dp"]) >= 2 * np.pi - 1e-9):
            raise ValueError("phi sectors are not supported; full rings are required")
        ri, ro = g["rmin"], g["rmax"]
        for z0, z1 in zip(g["z0"], g["z1"]):
            if not np.isfinite([ri, ro, z0, z1]).all() or not (0 <= ri < ro and z0 < z1):
                raise ValueError("support bounds must be finite with positive radial and axial extent")
            for ai, ao, a0, a1 in volumes:
                if max(ri, ai) < min(ro, ao) and max(z0, a0) < min(z1, a1):
                    raise ValueError("overlapping support volumes are not supported, even for the same material")
            volumes.append((ri, ro, z0, z1))
    return shells


def support_intervals(shells, o, d, length):
    """Return a DataFrame of lane, s0_mm, s1_mm, material, ordered by lane then distance.

    Rays are o + s*d with finite (N, 3) origins and unit directions (absolute norm tolerance 1e-10).
    Length is a finite nonnegative scalar or (N,) vector in mm. Positive-length lanes partition
    [0, length] into support and explicit material='vacuum' gaps; zero-length lanes have no rows.
    Empty (0, 3) rays return an empty table with the same schema. Adjacent shell rows are not merged.

    Only CoaxialShells or InteriorColumn.coax with calo=None, no other tubes, and full phi rings
    are supported. Positive-volume overlaps are rejected globally, even for equal materials.
    Bounds are those stored by CoaxialShells (including its radius rounding and axis tolerance).
    Tangencies have zero measure; rays lying in axial end faces are excluded, as in column().
    This is analytic straight-ray navigation for a bounded first candidate, not a production
    navigator for the full extracted geometry, straw materials, scattering, or transport.
    """
    shells = _supported(shells)
    o, d, length = _rays(o, d, length)
    w = o - shells.c
    ox, oy, oz = w @ shells.e1, w @ shells.e2, w @ shells.a
    dx, dy, dz = d @ shells.e1, d @ shells.e2, d @ shells.a
    radial_speed2 = dx * dx + dy * dy
    radial_parallel = radial_speed2 == 0
    safe_speed2 = np.where(radial_parallel, 1.0, radial_speed2)
    center = -(ox * dx + oy * dy) / safe_speed2
    closest2 = (ox * dy - oy * dx) ** 2 / safe_speed2

    def chord(radius):
        inside = ox * ox + oy * oy <= radius * radius
        delta = radius * radius - closest2
        hit = np.where(radial_parallel, inside, delta > 0)
        half = np.sqrt(np.maximum(delta, 0) / safe_speed2)
        lo = np.where(radial_parallel, -np.inf, center - half)
        hi = np.where(radial_parallel, np.inf, center + half)
        return lo, hi, hit

    intervals = []
    z_parallel = dz == 0
    safe_dz = np.where(z_parallel, 1.0, dz)
    for g in shells.groups:
        a0, a1, hit = chord(g["rmax"])
        lo, hi = np.maximum(a0, 0), np.minimum(a1, length)
        hi = np.where(hit, hi, lo)
        radial = [(lo, hi)]
        if g["rmin"] > 0:
            b0, b1, hole = chord(g["rmin"])
            b0, b1 = np.where(hole, b0, hi), np.where(hole, b1, hi)
            radial = [(lo, np.minimum(hi, b0)), (np.maximum(lo, b1), hi)]
        for z0, z1, material in zip(g["z0"], g["z1"], g["mat"]):
            t0, t1 = (z0 - oz) / safe_dz, (z1 - oz) / safe_dz
            zlo = np.where(z_parallel, -np.inf, np.minimum(t0, t1))
            zhi = np.where(z_parallel, np.inf, np.maximum(t0, t1))
            zhit = ~z_parallel | ((oz > z0) & (oz < z1))
            for r0, r1 in radial:
                s0, s1 = np.maximum(r0, zlo), np.minimum(r1, zhi)
                lane = np.flatnonzero(zhit & (s1 > s0))
                intervals.append(pd.DataFrame(dict(lane=lane, s0_mm=s0[lane], s1_mm=s1[lane], material=str(material))))
    return _partition(intervals, length)


def _rays(o, d, length):
    o, d = np.asarray(o, dtype=float), np.asarray(d, dtype=float)
    if o.ndim != 2 or o.shape[1:] != (3,) or d.shape != o.shape:
        raise ValueError("origins and directions must have matching (N, 3) shapes")
    if not np.isfinite(o).all() or not np.isfinite(d).all():
        raise ValueError("origins and directions must be finite")
    if not np.all(np.isclose(np.linalg.norm(d, axis=1), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("directions must be unit vectors")
    length = np.asarray(length, dtype=float)
    if length.ndim != 0 and length.shape != (len(o),):
        raise ValueError("length must be a scalar or an (N,) vector")
    if not np.isfinite(length).all() or np.any(length < 0):
        raise ValueError("length must be finite and nonnegative")
    length = np.broadcast_to(length, (len(o),))
    return o, d, length


def _partition(intervals, length):
    """Sort actual intersections, reject overlaps, and fill vacuum (no shell-pair scan)."""
    dtype = {"lane": "int64", "s0_mm": "float64", "s1_mm": "float64", "material": "object"}
    solid = (pd.concat(intervals, ignore_index=True).sort_values(["lane", "s0_mm"], kind="stable")
             if intervals else pd.DataFrame({k: np.empty(0, dtype=v) for k, v in dtype.items()}))
    lane = solid.lane.to_numpy(dtype=np.int64)
    s0, s1 = solid.s0_mm.to_numpy(), solid.s1_mm.to_numpy()
    first = np.r_[True, lane[1:] != lane[:-1]] if len(lane) else np.empty(0, bool)
    previous = np.where(first, 0., np.r_[0., s1[:-1]]) if len(lane) else np.empty(0)
    if (s0 < previous).any():
        raise ValueError("overlapping support intervals")
    gap = s0 > previous
    gaps = pd.DataFrame(dict(lane=lane[gap], s0_mm=previous[gap], s1_mm=s0[gap], material="vacuum"))
    last_end = np.zeros(len(length))
    np.maximum.at(last_end, lane, s1)
    tail = np.flatnonzero(last_end < length)
    tails = pd.DataFrame(dict(lane=tail, s0_mm=last_end[tail], s1_mm=length[tail], material="vacuum"))
    return pd.concat([solid, gaps, tails], ignore_index=True).sort_values(
        ["lane", "s0_mm"], kind="stable").reset_index(drop=True).astype(dtype)


def _annular_intervals(ol, dl, length, ri, ro):
    """At most two radial intervals for every lane, using a stable closest-approach chord."""
    ox, oy = ol[:, 0], ol[:, 1]
    dx, dy = dl[:, 0], dl[:, 1]
    speed2 = dx * dx + dy * dy
    parallel = speed2 == 0
    safe = np.where(parallel, 1., speed2)
    center = -(ox * dx + oy * dy) / safe
    closest2 = (ox * dy - oy * dx) ** 2 / safe

    def chord(radius):
        delta = radius * radius - closest2
        hit = np.where(parallel, ox * ox + oy * oy <= radius * radius, delta > 0)
        half = np.sqrt(np.maximum(delta, 0) / safe)
        return (np.where(parallel, -np.inf, center - half),
                np.where(parallel, np.inf, center + half), hit)

    a, b, hit = chord(ro)
    lo, hi = np.maximum(a, 0.), np.minimum(b, length)
    hi = np.where(hit, hi, lo)
    if ri == 0:
        return [(lo, hi)]
    a, b, hole = chord(ri)
    a, b = np.where(hole, a, hi), np.where(hole, b, hi)
    return [(lo, np.minimum(hi, a)), (np.maximum(lo, b), hi)]


def _tube_bounds(ri, ro, z0, z1, p0, dp):
    if not np.isfinite([ri, ro, z0, z1, p0, dp]).all():
        raise ValueError("tube bounds and angles must be finite")
    if not (0 <= ri < ro and z0 < z1 and 0 < dp <= 2 * np.pi + 1e-9):
        raise ValueError("tube bounds require positive radial, axial and phi extent (dphi <= 2*pi)")


def _tube_spans(radial, ol, dl, z0, z1, p0, dp, material):
    """Geometry-sized generator; each yielded table contains all intersecting lanes."""
    parallel = dl[:, 2] == 0
    dz = np.where(parallel, 1., dl[:, 2])
    a, b = (z0 - ol[:, 2]) / dz, (z1 - ol[:, 2]) / dz
    zlo = np.where(parallel, -np.inf, np.minimum(a, b))
    zhi = np.where(parallel, np.inf, np.maximum(a, b))
    zhit = ~parallel | ((ol[:, 2] > z0) & (ol[:, 2] < z1))
    for r0, r1 in radial:
        lo, hi = np.maximum(r0, zlo), np.minimum(r1, zhi)
        lane = np.flatnonzero(zhit & (hi > lo))
        lo, hi = lo[lane], hi[lane]
        if dp >= 2 * np.pi - 1e-9:
            yield pd.DataFrame(dict(lane=lane, s0_mm=lo, s1_mm=hi, material=str(material)))
            continue
        ox, oy = ol[lane, 0], ol[lane, 1]
        dx, dy = dl[lane, 0], dl[lane, 1]
        # A boundary ray lies in a plane through the axis. Cutting at the entire plane
        # is harmless: midpoint angular classification rejects its opposite half-ray.
        p0 = np.mod(p0, 2 * np.pi)
        angles = np.array([p0, p0 + dp])
        nx, ny = -np.sin(angles), np.cos(angles)
        den = dx[:, None] * nx + dy[:, None] * ny
        num = -(ox[:, None] * nx + oy[:, None] * ny)
        speed2 = dx * dx + dy * dy
        axis = -(ox * dx + oy * dy) / np.where(speed2 == 0, 1., speed2)
        crosses_axis = (speed2 > 0) & (ox * dy - oy * dx == 0)
        # For a ray contained in a boundary plane, the axis can still change phi by pi.
        fallback = np.where(crosses_axis, axis, hi)
        cuts = np.divide(num, den, out=np.broadcast_to(fallback[:, None], den.shape).copy(), where=den != 0)
        cuts = np.sort(np.clip(cuts, lo[:, None], hi[:, None]), axis=1)
        edges = np.column_stack([lo, cuts, hi])
        starts, ends = edges[:, :-1], edges[:, 1:]
        mid = starts + .5 * (ends - starts)
        phi = np.mod(np.arctan2(oy[:, None] + dy[:, None] * mid,
                               ox[:, None] + dx[:, None] * mid) - p0, 2 * np.pi)
        face_tol = 32 * np.finfo(float).eps
        inside = (ends > starts) & ((phi <= dp + face_tol) | (phi >= 2 * np.pi - face_tol))
        # Drop duplicate cuts before merging: zero-width slots must not break a run.
        # There are only three slots, so this loop is independent of lane count.
        for j in range(2):
            bridge = inside[:, j] & (ends[:, j+1] == starts[:, j+1])
            inside[:, j+1] |= bridge
        begin = inside & ~np.column_stack([np.zeros(len(lane), bool), inside[:, :-1]])
        end = inside & ~np.column_stack([inside[:, 1:], np.zeros(len(lane), bool)])
        rows, cols = np.nonzero(begin)
        s0, s1 = starts[rows, cols], ends[end]
        keep = s1 > s0
        yield pd.DataFrame(dict(lane=lane[rows[keep]], s0_mm=s0[keep], s1_mm=s1[keep], material=str(material)))


def tube_intervals(shells, o, d, length):
    """Ordered exact straight-ray intervals through full/sector coaxial and other tubes.

    Same ray inputs, table schema, vacuum gaps and zero-length behavior as
    :func:`support_intervals`. InteriorColumn is accepted only with calo=None.
    Coaxial tubes use the stored rounded radii, axis tolerance, z and phi bounds;
    noncoaxial ``other`` tuples use their original local-to-global rotations and centers.
    Finite annuli are clipped analytically to z, then to both phi boundary planes,
    with angular midpoint membership handling wrap-around and reflex sectors.

    Positive-length overlaps on the supplied rays raise ValueError, even for equal
    materials. Unlike support_intervals, unseen geometry overlaps are not checked.
    Tangencies have zero measure; axial end-face rays are excluded. Phi boundaries
    are inclusive within 32 float64 eps radians on either face, including wrap at zero;
    shared-face rays can therefore be rejected as overlapping.
    dp within 1e-9 of 2*pi is full, matching core's convention. Adjacent distinct
    tubes are not merged. No sampling, convex/calo solids, straw navigation, curved
    paths or transport is implemented. Work is vectorized over lanes, with loops
    over geometry and at most three phi subintervals; memory is bounded by batch
    size and returned intersections. Use bounded batches for large geometries.
    """
    o, d, length = _rays(o, d, length)
    hits = tube_hits(shells, o, d, length)
    return _partition([hits], length)


def tube_hits(shells, o, d, length):
    """Unpartitioned stored-tube intersections for diagnostics; overlaps are retained, never repaired.

    Same inputs and solid-row schema as tube_intervals, but no vacuum rows or overlap check.
    Label material keys before CoaxialShells packing to retain source-volume identities.
    This helper is not a physical-history or transport admission gate.
    """
    if hasattr(shells, "coax"):
        if getattr(shells, "calo", None) is not None:
            raise ValueError("calorimeter geometry is not supported")
        shells = shells.coax
    if not isinstance(shells, CoaxialShells):
        raise TypeError("expected CoaxialShells or an InteriorColumn with coax shells")
    o, d, length = _rays(o, d, length)
    frame = np.column_stack([shells.e1, shells.e2, shells.a])
    if not np.isfinite(shells.c).all() or not np.isfinite(frame).all():
        raise ValueError("tube frame must be finite")
    ol, dl = (o - shells.c) @ frame, d @ frame
    intervals = []
    for g in shells.groups:
        ri, ro = g["rmin"], g["rmax"]
        for z0, z1, p0, dp in zip(g["z0"], g["z1"], g["p0"], g["dp"]):
            _tube_bounds(ri, ro, z0, z1, p0, dp)
        radial = _annular_intervals(ol, dl, length, ri, ro)
        for z0, z1, p0, dp, mat in zip(g["z0"], g["z1"], g["p0"], g["dp"], g["mat"]):
            intervals.extend(_tube_spans(radial, ol, dl, z0, z1, p0, dp, mat))
    for mat, rotation, center, ri, ro, hz, p0, dp in shells.other:
        _tube_bounds(ri, ro, -hz, hz, p0, dp)
        rotation, center = np.asarray(rotation, float), np.asarray(center, float)
        if (rotation.shape != (3, 3) or center.shape != (3,) or
                not np.isfinite(rotation).all() or not np.isfinite(center).all()):
            raise ValueError("tube rotation and center must be finite with shapes (3,3) and (3,)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10):
            raise ValueError("tube rotation must be orthogonal")
        local_o, local_d = (o - center) @ rotation, d @ rotation
        radial = _annular_intervals(local_o, local_d, length, ri, ro)
        intervals.extend(_tube_spans(radial, local_o, local_d, -hz, hz, p0, dp, mat))
    dtype = {"lane": "int64", "s0_mm": "float64", "s1_mm": "float64", "material": "object"}
    if not intervals:
        return pd.DataFrame({k: np.empty(0, dtype=v) for k, v in dtype.items()})
    return pd.concat(intervals, ignore_index=True).astype(dtype)
