"""Diagnostic straight-ray partitions of stored convex pieces, not an independent GDML navigator."""
import numpy as np
import pandas as pd

from mu2efastsim.world import PieceSet, SignedPieceSet


_DTYPES = {"lane": "int64", "s0_mm": "float64", "s1_mm": "float64", "material": "object"}
_PAIR_CHUNK = 65536


def _empty():
    return pd.DataFrame({k: np.empty(0, dtype=v) for k, v in _DTYPES.items()})


def _sets(pieces):
    if isinstance(pieces, PieceSet):
        sets = [(1, pieces)]
    elif isinstance(pieces, SignedPieceSet):
        sets = pieces.sets
        if sum(len(ps) for _, ps in sets) != len(pieces):
            raise ValueError("signed pieces omitted by packing; expected signs +1 or -1 and nonempty planes")
    else:
        raise TypeError("expected PieceSet or SignedPieceSet")
    for sign, ps in sets:
        if sign not in (-1, 1) or not isinstance(ps, PieceSet):
            raise ValueError("signed sets require signs +1 or -1 and PieceSet geometry")
        if (ps.lo.shape != (len(ps), 3) or ps.hi.shape != ps.lo.shape
                or ps.P.ndim != 3 or ps.P.shape[0] != len(ps) or ps.P.shape[2] != 4
                or ps.P.shape[1] == 0):
            raise ValueError("invalid packed convex geometry shapes")
        if not all(np.isfinite(a).all() for a in (ps.P, ps.lo, ps.hi)) or np.any(ps.lo > ps.hi):
            raise ValueError("convex planes and bounding boxes must be finite with ordered bounds")
    return sets


def _pairs(ps, o, d, length):
    if getattr(ps, "grid", None):
        chunk = max(1, min(1024, _PAIR_CHUNK // max(len(ps), 1)))
        for start in range(0, len(o), chunk):
            stop = start + chunk
            lane, piece = ps._candidates(o[start:stop], d[start:stop], length[start:stop])
            for first in range(0, len(lane), _PAIR_CHUNK):
                yield lane[first:first + _PAIR_CHUNK] + start, piece[first:first + _PAIR_CHUNK]
    elif len(ps):
        for start in range(0, len(o) * len(ps), _PAIR_CHUNK):
            flat = np.arange(start, min(start + _PAIR_CHUNK, len(o) * len(ps)))
            yield flat // len(ps), flat % len(ps)


def _hits(ps, o, d, length):
    for lane, piece in _pairs(ps, o, d, length):
        origin, direction = o[lane], d[lane]
        parallel = direction == 0
        safe = np.where(parallel, 1., direction)
        t0, t1 = (ps.lo[piece] - origin) / safe, (ps.hi[piece] - origin) / safe
        lo = np.maximum(0., np.where(parallel, -np.inf, np.minimum(t0, t1)).max(axis=1))
        hi = np.minimum(length[lane], np.where(parallel, np.inf, np.maximum(t0, t1)).min(axis=1))
        outside = (parallel & ((origin < ps.lo[piece]) | (origin > ps.hi[piece]))).any(axis=1)
        keep = ~outside & (hi > lo)
        lane, piece = lane[keep], piece[keep]
        # Bound plane work too, including wide polygons in a packed PieceSet.
        chunk = max(1, _PAIR_CHUNK // ps.P.shape[1])
        for start in range(0, len(lane), chunk):
            it, ip = lane[start:start + chunk], piece[start:start + chunk]
            planes = ps.P[ip]
            den = np.einsum("ij,ikj->ik", d[it], planes[:, :, :3])
            num = planes[:, :, 3] - np.einsum("ij,ikj->ik", o[it], planes[:, :, :3])
            with np.errstate(divide="ignore", invalid="ignore"):
                crossing = num / den
            enter = np.maximum(0., np.where(den < 0, crossing, -np.inf).max(axis=1))
            leave = np.minimum(length[it], np.where(den > 0, crossing, np.inf).min(axis=1))
            outside = ((den == 0) & (num < 0)).any(axis=1)
            keep = ~outside & (leave > enter)
            yield it[keep], ip[keep], enter[keep], leave[keep]


def convex_intervals(pieces, o, d, length):
    """Return ordered lane/s0_mm/s1_mm/material rows partitioning each positive-length ray.

    Accept only existing PieceSet or SignedPieceSet objects. Origins and unit directions have
    finite (N, 3) shape (unit norm absolute tolerance 1e-10); length is a finite nonnegative
    scalar or (N,) array in mm. Empty and zero-length lanes have no rows. Unoccupied spans
    use the literal material key 'vacuum'; other stored material keys are preserved verbatim,
    not interpreted via density, names, or the core materials registry.

    Intersect the stored halfspaces P with [0, length], using lo/hi as conservative broad-phase
    bounds and the existing grid when present. Parallel rays on closed faces are included;
    isolated tangencies have no length. Exact coincident boundaries aggregate before summing
    signed events per (lane, material). Counts must be 0 or 1 on every positive-length span;
    negative counts, same-material multiplicity and cross-material overlaps raise ValueError.
    Touching pieces are allowed; adjacent rows need not be merged. Validation is on requested
    rays, not a global proof that the geometry has no overlaps elsewhere.

    Signed pieces can represent mother-minus-daughter and hollow/cut solids only to the extent
    already encoded in these arrays. No global clipping of summed columns is performed. Core
    loaders may omit solids, approximate curves with polygons, or retain only a Boolean's first
    constituent; missing geometry and incorrect conservative bounds cannot be reconstructed.
    This is not GDML truth, curved-path navigation, straw transport, or production integration.
    Pair/plane calculations are bounded chunks, with no per-ray Python loop; grid candidate
    construction and output/event storage retain the existing geometry-dependent memory cost.
    """
    sets = _sets(pieces)
    o, d = np.asarray(o, dtype=float), np.asarray(d, dtype=float)
    if o.ndim != 2 or o.shape[1:] != (3,) or d.shape != o.shape:
        raise ValueError("origins and directions must have matching (N, 3) shapes")
    if not np.isfinite(o).all() or not np.isfinite(d).all():
        raise ValueError("origins and directions must be finite")
    if not np.all(np.isclose(np.linalg.norm(d, axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError("directions must be unit vectors")
    length = np.asarray(length, dtype=float)
    if length.ndim != 0 and length.shape != (len(o),):
        raise ValueError("length must be a scalar or an (N,) vector")
    if not np.isfinite(length).all() or np.any(length < 0):
        raise ValueError("length must be finite and nonnegative")
    length = np.broadcast_to(length, (len(o),))
    events = []
    for sign, ps in sets:
        for lane, piece, enter, leave in _hits(ps, o, d, length):
            if len(lane):
                material = np.asarray(ps.mats, dtype=object)[ps.mat[piece]]
                events.append(pd.DataFrame(dict(lane=np.tile(lane, 2), s=np.r_[enter, leave],
                                                material=np.tile(material, 2),
                                                delta=np.repeat([sign, -sign], len(lane)))))
    solid = _empty()
    if events:
        events = pd.concat(events, ignore_index=True).groupby(
            ["lane", "material", "s"], as_index=False, sort=True).delta.sum()
        grouped = events.groupby(["lane", "material"], sort=False)
        count = grouped.delta.cumsum()
        end = grouped.s.shift(-1)
        span = end > events.s
        if ((count[span] < 0) | (count[span] > 1)).any():
            raise ValueError("invalid signed occupancy: material multiplicity must be 0 or 1")
        active = span & (count == 1)
        solid = pd.DataFrame(dict(lane=events.lane[active], s0_mm=events.s[active],
                                  s1_mm=end[active], material=events.material[active]))
        solid = solid.sort_values(["lane", "s0_mm"], kind="stable").reset_index(drop=True)
    lane = solid.lane.to_numpy(dtype=np.int64)
    s0, s1 = solid.s0_mm.to_numpy(), solid.s1_mm.to_numpy()
    first = np.r_[True, lane[1:] != lane[:-1]] if len(lane) else np.empty(0, bool)
    previous = np.where(first, 0., np.r_[0., s1[:-1]]) if len(lane) else np.empty(0)
    if np.any(s0 < previous):
        raise ValueError("cross-material overlapping intervals")
    gap = s0 > previous
    gaps = pd.DataFrame(dict(lane=lane[gap], s0_mm=previous[gap], s1_mm=s0[gap], material="vacuum"))
    last_end = np.zeros(len(o))
    np.maximum.at(last_end, lane, s1)
    tail = np.flatnonzero(last_end < length)
    tails = pd.DataFrame(dict(lane=tail, s0_mm=last_end[tail], s1_mm=length[tail], material="vacuum"))
    return pd.concat([solid, gaps, tails], ignore_index=True).sort_values(
        ["lane", "s0_mm"], kind="stable").reset_index(drop=True).astype(_DTYPES)
