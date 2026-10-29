"""Read-only state queries for supplied straight, linear-in-path histories, not production transport."""
import numpy as np
import pandas as pd

from mu2efastsim.eloss import MMU
from .flighttime import linear_kinetic_dt_ns
from .transportledger import KEY, SegmentHistory


def path_coordinates(history):
    """Return row-aligned (s0, s1, t0, t1), in mm/ns, reset per full muon key.

    This only accumulates supplied path and known time; it does not certify straight
    geometry or an energy-evolution law. Positive increments use grouped compensated
    sums. Zero increments are excluded from that sum and forward-filled per muon:
    even a compensation correction must not give a displacement path or flight time.
    Positive paths must advance the representable coordinate, and clocks must not
    decrease; otherwise the history cannot support unambiguous float path queries.
    Arrays are independent of the immutable history, including for empty histories.
    """
    if not isinstance(history, SegmentHistory):
        raise TypeError("history must be a SegmentHistory")
    if (history["dt_status"] != "known").any():
        raise ValueError("only histories with known integrated times can be queried")
    if not len(history):
        return tuple(np.empty(0, dtype=float) for _ in range(4))
    hkey = np.column_stack([history[name] for name in KEY[:-1]])
    first = np.r_[True, np.any(hkey[1:] != hkey[:-1], axis=1)]
    group = np.cumsum(first) - 1
    coordinates = []
    for name in ("path_mm", "dt_ns"):
        increments = history[name]
        positive = increments > 0
        end = np.full(len(history), np.nan)
        summed = pd.Series(increments[positive]).groupby(
            group[positive], sort=False).cumsum().to_numpy()
        if not np.isfinite(summed).all():
            raise ValueError("accumulated path and flight time must be finite")
        end[positive] = summed
        end = pd.Series(end).groupby(group, sort=False).ffill().fillna(0.).to_numpy(copy=True)
        start = np.r_[0., end[:-1]]
        start[first] = 0.
        if (end < start).any() or (name == "path_mm" and (positive & (end <= start)).any()):
            raise ValueError("unrepresentable accumulated path or flight time coordinate")
        coordinates.extend((start, end))
    return tuple(coordinates)


def _search_keys(identity, distance):
    out = np.empty(len(distance), dtype=[(name, np.int64) for name in KEY[:-1]] + [("s", float)])
    for i, name in enumerate(KEY[:-1]):
        out[name] = identity[:, i]
    out["s"] = distance
    return out


def at_path(history, identity, path_mm, *, energy_evolution):
    """Query full-key muon states at recorded path coordinates without extrapolation.

    Identities are integer (N,4) run/subrun/event/sp; finite distances are (N,) in mm
    from each muon's first recorded point. The explicit linear-in-path KE declaration
    applies to all physical rows, whose known times must agree with that law. Curved
    physical segments are unsupported; recorded displacements have zero path and time.
    At shared coordinates the last outgoing row wins, including the exit state of a
    trailing displacement. Jumps never interpolate positions or add flight time.
    Interior states use the representable prefix span, evaluated from the nearer
    endpoint; supplied physical length sets the integrated time. Unrepresentable
    positive path advances or positive interior KE are rejected, never silently stopped.
    Terminal endpoints are available but not alive, including positive-KE model stops;
    nonterminal final endpoints remain alive. Missing identities and paths outside the
    closed recorded domain are unavailable, with NaN state and segment -1. No boundary
    tolerance extends that domain. dt_from_start_ns is per-muon relative flight time,
    without generator, bunch or world offsets. Inputs and provenance are not changed.
    """
    if not isinstance(history, SegmentHistory) or energy_evolution != "linear-in-path":
        raise ValueError("SegmentHistory and explicit linear-in-path energy evolution required")
    keys, distance = np.asarray(identity), np.asarray(path_mm, float)
    if (keys.ndim != 2 or keys.shape[1] != 4 or keys.dtype.kind not in "iu" or (keys < 0).any()
            or (keys > np.iinfo(np.int64).max).any() or distance.shape != (len(keys),)
            or not np.isfinite(distance).all()):
        raise ValueError("query needs int64-compatible nonnegative (N,4) identities and finite (N,) paths")
    if (history["dt_status"] != "known").any():
        raise ValueError("only histories with known integrated times can be queried")
    physical = history["kind"] == "physical"
    chord = np.linalg.norm(history["exit_mm"][physical] - history["entry_mm"][physical], axis=1)
    if not np.allclose(chord, history["path_mm"][physical], rtol=1e-10, atol=1e-8):
        raise ValueError("position interpolation requires straight physical segments")
    expected = linear_kinetic_dt_ns(history["path_mm"], history["ke_in_MeV"], history["ke_out_MeV"])
    if not np.allclose(expected, history["dt_ns"], rtol=1e-10, atol=1e-12):
        raise ValueError("history times disagree with declared energy evolution")
    n = len(keys)
    result = dict(available=np.zeros(n, bool), alive=np.zeros(n, bool), ke_MeV=np.full(n, np.nan),
                  p_MeV_c=np.full(n, np.nan), position_mm=np.full((n, 3), np.nan),
                  dt_from_start_ns=np.full(n, np.nan), segment=np.full(n, -1, np.int64))
    s0, s1, t0, t1 = path_coordinates(history)
    if not len(history) or not n:
        return result
    hkey = np.column_stack([history[name] for name in KEY[:-1]])
    index = np.searchsorted(_search_keys(hkey, s0), _search_keys(keys, distance), side="right") - 1
    safe = np.clip(index, 0, len(history) - 1)
    valid = ((index >= 0) & np.all(hkey[safe] == keys, axis=1)
             & (distance >= s0[safe]) & (distance <= s1[safe]))
    q, row = np.flatnonzero(valid), safe[valid]
    length = history["path_mm"][row]
    endpoint = distance[q] == s1[row]
    # Nearest-end interpolation preserves shared boundaries; physical length sets flight time.
    left, right = distance[q] - s0[row], s1[row] - distance[q]
    from_exit = right < left
    span = s1[row] - s0[row]
    fraction = np.divide(np.minimum(left, right), span,
                         out=np.zeros_like(left), where=span > 0)
    kin, kout = history["ke_in_MeV"][row], history["ke_out_MeV"][row]
    change = (kin - kout) * fraction
    energy = np.where(from_exit, kout + change, kin - change)
    # Zero-path rows expose their exit state; right-sided lookup skips earlier jumps.
    energy = np.where(endpoint, kout, energy)
    if ((~endpoint) & (energy <= 0)).any():
        raise ValueError("unrepresentable positive interior kinetic energy")
    result["available"][q] = True
    result["alive"][q] = ~(endpoint & (history["terminal"][row] != "none"))
    result["ke_MeV"][q] = energy
    result["p_MeV_c"][q] = np.sqrt(energy) * np.sqrt(energy + 2 * MMU)
    delta = history["exit_mm"][row] - history["entry_mm"][row]
    offset = fraction[:, None] * delta
    position = np.where(from_exit[:, None], history["exit_mm"][row] - offset,
                        history["entry_mm"][row] + offset)
    result["position_mm"][q] = np.where(endpoint[:, None], history["exit_mm"][row], position)
    near_length = length * fraction
    near_time = linear_kinetic_dt_ns(near_length, np.where(from_exit, energy, kin),
                                    np.where(from_exit, kout, energy))
    time = np.where(from_exit, t1[row] - near_time, t0[row] + near_time)
    result["dt_from_start_ns"][q] = np.where(endpoint, t1[row], np.clip(time, t0[row], t1[row]))
    result["segment"][q] = history["segment"][row]
    return result
