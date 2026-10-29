"""Vectorized straight-line reference search on captured KinematicLine owners.

This diagnostic reproduces piece-index iteration, not the full native CA solver,
hit retry policy, construction contract or numerical fit algebra.
"""
import numpy as np

from .linefit import C, MMU


def match_rows(wanted, available):
    """Many-to-one integer-key join with missing rows marked -1 and duplicate targets rejected."""
    wanted, available = np.asarray(wanted), np.asarray(available)
    if wanted.ndim != 2 or available.ndim != 2 or wanted.shape[1] != available.shape[1]:
        raise ValueError("incompatible key columns")
    if wanted.dtype.kind not in "iu" or available.dtype.kind not in "iu":
        raise ValueError("identities must be integers")
    dtype = np.dtype([(f"k{i}", np.int64) for i in range(wanted.shape[1])])
    w, a = (np.ascontiguousarray(x, dtype=np.int64).view(dtype).ravel() for x in (wanted, available))
    if len(np.unique(a)) != len(a):
        raise ValueError("duplicate available identities")
    if not len(a):
        return np.full(len(w), -1, dtype=np.int64)
    order = np.argsort(a)
    pos = np.searchsorted(a[order], w)
    safe = np.minimum(pos, len(a) - 1)
    return np.where((pos < len(a)) & (a[order[safe]] == w), order[safe], -1)


def nearest(ranges, counts, parent, times):
    """Native earlier-piece lower_bound, with begin taking precedence over end."""
    ends = ranges[parent, :, 1]
    valid = np.arange(ends.shape[1])[None] < counts[parent, None]
    index = np.minimum(np.sum(valid & (ends < times[:, None]), axis=1), counts[parent] - 1)
    index = np.where(times >= ranges[parent, counts[parent] - 1, 1], counts[parent] - 1, index)
    return np.where(times <= ranges[parent, 0, 0], 0, index).astype(np.int64)


def toca(parameters, mid, wdir):
    """Analytic line/line TOCA in native (d0,phi0,z0,theta,t0,momentum) coordinates."""
    d0, phi, z0, theta, t0, mom = parameters.T
    p = np.column_stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0])
    u = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    c = np.sum(u * wdir, axis=1)
    den = 1 - c * c
    delta = mid - p
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (np.sum(delta * u, axis=1) - c * np.sum(delta * wdir, axis=1)) / den
        time = t0 + s / (mom / np.hypot(mom, MMU) * C)
    usable = np.isfinite(time) & (den >= 1e-5)
    return time, usable


def search(parameters, ranges, counts, parent, mid, wdir, hint, max_moves=10):
    """Configured p109 piece search, including the reference/solution split at the move cap.

The local CA is an analytic line solution. At exhaustion native code changes the
reference pointer once more without recomputing its solution; both indices are returned.
"""
    if max_moves < 0:
        raise ValueError("max_moves must be nonnegative")
    index = nearest(ranges, counts, parent, hint)
    solution_index = index.copy()
    time, usable = toca(parameters[parent, index], mid, wdir)
    moves = np.zeros(len(index), dtype=np.int64)
    active = np.ones(len(index), dtype=bool)
    exhausted = np.zeros(len(index), dtype=bool)
    if max_moves:
        for attempt in range(max_moves + 1):
            target = nearest(ranges, counts, parent, time)
            changed = active & (target != index)
            index = np.where(active, target, index)
            step = changed & usable
            if attempt == max_moves:
                exhausted = step
                break
            active = step
            solution_index = np.where(step, index, solution_index)
            new_time, new_usable = toca(parameters[parent, solution_index], mid, wdir)
            time = np.where(step, new_time, time)
            usable = np.where(step, new_usable, usable)
            moves += step
    return {"index": index, "solution_index": solution_index, "time": time,
            "usable": usable, "moves": moves, "exhausted": exhausted}
