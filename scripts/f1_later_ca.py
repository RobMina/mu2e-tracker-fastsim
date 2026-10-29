"""Standalone later-reference CA diagnostics; no fitter integration or native parity claim.

Only piece search and retained-time/midpoint hint policies are implemented here.
The local arithmetic remains nativeca.solve (100 updates); no hit-z retry/zTime,
ROOT arithmetic substitution, or native failure-state propagation is introduced.
"""
from functools import wraps

import numpy as np

from trkfastsim.reco import nativeca as C

NATIVE_CA_PRECISION = float(np.float32(1e-4))
MAX_MOVES = 1000


def _checked(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("nonfinite later CA arithmetic; failure propagation unsupported") from exc
    return call


def _array(value, name):
    # Inspect masks BEFORE np.asarray, which silently discards MaskedArray masks.
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"missing {name}")
    out = np.asarray(value)
    if out.dtype.kind not in "biuf":
        raise ValueError(f"invalid {name} type")
    return out


def _real(value, shape, name):
    out = _array(value, name)
    if out.shape != shape or out.dtype.kind not in "iuf":
        raise ValueError(f"invalid {name} shape/type")
    out = out.astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"nonfinite {name}")
    return out


def _integer(value, shape, name, low, high):
    out = _array(value, name)
    if out.shape != shape or out.dtype.kind not in "iu":
        raise ValueError(f"invalid {name} shape/type")
    # Check bounds before conversion, including uint64 values beyond int64.
    if np.any(out < low) or np.any(out > high):
        raise ValueError(f"invalid {name} bounds")
    return out.astype(np.int64)


def _usable(value, n):
    out = _array(value, "usable")
    if out.shape != (n,) or out.dtype.kind not in "biu" or np.any((out != 0) & (out != 1)):
        raise ValueError("invalid usable shape/type/value")
    return out.astype(bool)


def _line(line, n=None):
    if not isinstance(line, dict):
        raise ValueError("line must be a dictionary")
    required = ("end", "mid", "direction", "length", "measurement_time", "speed")
    if not all(k in line for k in required):
        raise ValueError("missing sensor line fields")
    mt = _array(line["measurement_time"], "measurement time")
    if mt.ndim != 1 or (n is not None and mt.shape != (n,)):
        raise ValueError("invalid measurement time shape")
    n = len(mt)
    result = {k: _real(line[k], (n, 3) if k in required[:3] else (n,), k)
              for k in required}
    if np.any(result["length"] <= 0) or np.any(result["speed"] <= 0):
        raise ValueError("sensor length and speed must be positive")
    if np.any(np.abs(np.sum(result["direction"]**2, axis=-1)-1.) > 1e-10):
        raise ValueError("sensor direction must be unit length")
    return result


def _state_times(state, n):
    """None denotes no previous CA; supplied states must be complete and finite."""
    if state is None:
        return np.zeros(n, bool), np.zeros(n), np.zeros(n)
    if not isinstance(state, dict) or not all(k in state for k in ("usable", "particle_time", "sensor_time")):
        raise ValueError("missing retained state fields")
    return (_usable(state["usable"], n),
            _real(state["particle_time"], (n,), "retained particle time"),
            _real(state["sensor_time"], (n,), "retained sensor time"))


def _midpoint(line):
    # SensorLine::timeAtMidpoint; measurement_time is the early endpoint time.
    return line["measurement_time"]-0.5*line["length"]/line["speed"]


@_checked
def hit_hint(previous_state, line, drift_time):
    """Return (particle_hint, sensor_hint): own usable CA, else (mid-drift, mid).

    drift_time is an exact [N] finite array, not a scalar broadcast. Supplied
    retained times are checked even on unusable lanes. No zTime policy is applied.
    """
    line = _line(line)
    n = len(line["speed"])
    usable, pt, st = _state_times(previous_state, n)
    drift = _real(drift_time, (n,), "drift time")
    mid = _midpoint(line)
    return np.where(usable, pt, mid-drift), np.where(usable, st, mid)


@_checked
def crossing_hint(own, associated, line):
    """Return own usable times, else associated usable times, else (mid, mid).

    Either state may be None. Flags must be boolean or binary integers; all
    supplied retained times are finite exact [N] arrays, even if not selected.
    """
    line = _line(line)
    n = len(line["speed"])
    ou, op, os = _state_times(own, n)
    au, ap, ass = _state_times(associated, n)
    mid = _midpoint(line)
    return np.where(ou, op, np.where(au, ap, mid)), np.where(ou, os, np.where(au, ass, mid))


def _pieces(parameters, ranges, counts):
    # Validate only counted pieces, preserving masks until padding is excluded.
    q, r = np.asanyarray(parameters), np.asanyarray(ranges)
    if q.ndim != 3 or q.shape[-1] != 6 or q.shape[1] < 1:
        raise ValueError("invalid parameters shape")
    ni, npiece = q.shape[:2]
    if r.shape != (ni, npiece, 2):
        raise ValueError("invalid ranges shape")
    count = _integer(counts, (ni,), "counts", 1, npiece)
    valid = np.arange(npiece)[None, :] < count[:, None]
    qq = _real(q[valid], (int(valid.sum()), 6), "valid parameters")
    rr = _real(r[valid], (int(valid.sum()), 2), "valid ranges")
    if np.any(qq[:, 5] <= 0):
        raise ValueError("particle momentum must be positive")
    if np.any(rr[:, 0] > rr[:, 1]):
        raise ValueError("reversed piece ranges")
    qout, rout = np.zeros(q.shape), np.zeros(r.shape)
    qout[valid], rout[valid] = qq, rr
    adjacent = valid[:, 1:]
    if np.any(rout[:, 1:, 0][adjacent] != rout[:, :-1, 1][adjacent]):
        raise ValueError("piece ranges must be sorted and contiguous")
    return qout, rout, count


def _nearest(ranges, counts, parent, time):
    ends = ranges[parent, :, 1]
    valid = np.arange(ranges.shape[1])[None, :] < counts[parent, None]
    index = np.minimum(np.sum(valid & (ends < time[:, None]), axis=1), counts[parent]-1)
    index = np.where(time >= ranges[parent, counts[parent]-1, 1], counts[parent]-1, index)
    # Begin precedence matters when the entire trajectory has zero duration.
    return np.where(time <= ranges[parent, 0, 0], 0, index).astype(np.int64)


def _solve(q, line, pt, st, precision, mass):
    state = C.solve(q, line, pt, st, precision=precision, mass=mass)
    # Parallel CA legitimately retains +inf update sentinels without an update.
    for key, value in state.items():
        if key in ("particle_update", "sensor_update"):
            sentinel = (state["status"] == 4) & (state["updates"] == 0) & np.isposinf(value)
            finite = np.isfinite(value) | sentinel
        else:
            finite = np.isfinite(value)
        if not np.all(finite):
            raise ValueError(f"nonfinite local CA output: {key}")
    return state


@_checked
def piecewise(parameters, ranges, counts, parent, line, particle_hint, sensor_hint,
              mass, precision=NATIVE_CA_PRECISION, max_moves=10):
    """Vectorized native piece-search control flow using unchanged nativeca.solve.

    Shapes: parameters [I,P,6], ranges [I,P,2], counts [I], parent [N],
    line fields [N] / [N,3], particle_hint/sensor_hint/mass [N]. Counted pieces
    must have sorted, exactly contiguous ranges (zero durations are permitted).
    Piece identities/order are the caller's assembly contract; padding is ignored.

    Return every nativeca.solve field plus index (reference pointer),
    solution_index (retained state's piece), moves (number of re-solves),
    exhausted, and solution_history [N,max_moves+1] (initial and re-solved piece
    indices, -1 for unused entries). The cap can advance index to an UNSOLVED
    piece; only converged status becomes unconverged on exhaustion. With cap 0,
    return the initial solve without even advancing the pointer.

    All lane operations are vectorized; only bounded move/local-update loops and
    fixed field loops are used. Invalid input or overflow aborts the whole call.
    max_moves is a nonboolean integer in [0,1000]. Inputs are not modified.
    """
    if isinstance(max_moves, (bool, np.bool_)) or not isinstance(max_moves, (int, np.integer)) or not 0 <= max_moves <= MAX_MOVES:
        raise ValueError("invalid max_moves; require nonboolean integer in [0,1000]")
    precision = float(_real(precision, (), "precision"))
    if precision <= 0:
        raise ValueError("precision must be positive")
    q, r, count = _pieces(parameters, ranges, counts)
    raw_parent = _array(parent, "parent")
    if raw_parent.ndim != 1:
        raise ValueError("invalid parent shape")
    n = len(raw_parent)
    parent = _integer(raw_parent, (n,), "parent", 0, len(count)-1)
    line = _line(line, n)
    pt = _real(particle_hint, (n,), "particle hint")
    st = _real(sensor_hint, (n,), "sensor hint")
    mass = _real(mass, (n,), "mass")
    if np.any(mass <= 0):
        raise ValueError("particle mass must be positive")
    index = _nearest(r, count, parent, pt)
    solution_index = index.copy()
    state = _solve(q[parent, index], line, pt, st, precision, mass)
    moves, exhausted = np.zeros(n, np.int64), np.zeros(n, bool)
    history = np.full((n, int(max_moves)+1), -1, np.int64)
    history[:, 0] = index
    active = np.ones(n, bool)
    if max_moves:
        for attempt in range(int(max_moves)+1):
            target = _nearest(r, count, parent, state["particle_time"])
            changed = active & (target != index)
            index = np.where(active, target, index)  # Native pointer update FIRST.
            step = changed & state["usable"]
            if attempt == max_moves:
                exhausted = step
                state["status"] = np.where(exhausted & (state["status"] == 0), 1, state["status"])
                break
            active = step
            if not np.any(step):
                break
            # Subset lanes: stopped/unusable states must never be re-solved.
            new = _solve(q[parent[step], index[step]], {k: v[step] for k, v in line.items()},
                         state["particle_time"][step], state["sensor_time"][step], precision, mass[step])
            for key in state:
                state[key][step] = new[key]
            solution_index[step] = index[step]
            moves += step
            history[step, attempt+1] = index[step]
    return dict(state, index=index, solution_index=solution_index, moves=moves,
                exhausted=exhausted, solution_history=history)
