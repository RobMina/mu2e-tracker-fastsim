"""Vectorized shared-crossing state queries on explicitly linear-in-path physical histories."""
import numpy as np
import pandas as pd

from mu2efastsim.eloss import MMU
from .flighttime import linear_kinetic_dt_ns
from .transportledger import KEY, SegmentHistory


def _search_keys(identity, distance):
    out = np.empty(len(distance), dtype=[(name, np.int64) for name in KEY[:-1]] + [("s", float)])
    for i, name in enumerate(KEY[:-1]):
        out[name] = identity[:, i]
    out["s"] = distance
    return out


def at_path(history, identity, path_mm, *, energy_evolution):
    """Query one state convention at arbitrary path crossings, retaining unavailable/stopped masks.

    Identities are (N,4) run/subrun/event/sp; distances are (N,) from that muon's first recorded point.
    Caller must explicitly declare linear-in-path KE; every history time must already be known and agree
    with that law. Only physical straight segments are supported; displacement/curved paths are rejected.
    Shared boundaries select the outgoing segment; the last endpoint is valid but a terminal stop is not
    alive. Past a stop/end, missing identities and negative paths are unavailable (NaN), not extrapolated.
    dt_from_start_ns is relative flight time only, with no generator/bunch/world offset added.
    """
    if not isinstance(history, SegmentHistory) or energy_evolution != "linear-in-path":
        raise ValueError("SegmentHistory and explicit linear-in-path energy evolution required")
    keys, distance = np.asarray(identity), np.asarray(path_mm, float)
    if (keys.ndim != 2 or keys.shape[1] != 4 or keys.dtype.kind not in "iu" or (keys < 0).any()
            or (keys > np.iinfo(np.int64).max).any() or distance.shape != (len(keys),) or not np.isfinite(distance).all()):
        raise ValueError("query needs int64-compatible nonnegative (N,4) identities and finite (N,) paths")
    if (history["kind"] != "physical").any() or (history["dt_status"] != "known").any():
        raise ValueError("only physical histories with known integrated times can be queried")
    chord = np.linalg.norm(history["exit_mm"] - history["entry_mm"], axis=1)
    if not np.allclose(chord, history["path_mm"], rtol=1e-10, atol=1e-8):
        raise ValueError("position interpolation requires straight segments")
    expected = linear_kinetic_dt_ns(history["path_mm"], history["ke_in_MeV"], history["ke_out_MeV"])
    if not np.allclose(expected, history["dt_ns"], rtol=1e-10, atol=1e-12):
        raise ValueError("history times disagree with declared energy evolution")
    n = len(keys)
    result = dict(available=np.zeros(n, bool), alive=np.zeros(n, bool), ke_MeV=np.full(n, np.nan),
                  p_MeV_c=np.full(n, np.nan), position_mm=np.full((n, 3), np.nan),
                  dt_from_start_ns=np.full(n, np.nan), segment=np.full(n, -1, np.int64))
    if not len(history) or not n:
        return result
    hkey = np.column_stack([history[name] for name in KEY[:-1]])
    first = np.r_[True, np.any(hkey[1:] != hkey[:-1], axis=1)]
    group = np.cumsum(first) - 1
    s1 = pd.Series(history["path_mm"]).groupby(group, sort=False).cumsum().to_numpy()
    t1 = pd.Series(history["dt_ns"]).groupby(group, sort=False).cumsum().to_numpy()
    s0, t0 = s1 - history["path_mm"], t1 - history["dt_ns"]
    index = np.searchsorted(_search_keys(hkey, s0), _search_keys(keys, distance), side="right") - 1
    safe = np.clip(index, 0, len(history)-1)
    valid = ((index >= 0) & np.all(hkey[safe] == keys, axis=1) & (distance >= s0[safe]) & (distance <= s1[safe]))
    q, row = np.flatnonzero(valid), safe[valid]
    length = history["path_mm"][row]
    partial = distance[q] - s0[row]
    fraction = np.divide(partial, length, out=np.zeros_like(partial), where=length > 0)
    kin, kout = history["ke_in_MeV"][row], history["ke_out_MeV"][row]
    energy = kin * (1 - fraction) + kout * fraction
    result["available"][q] = True
    result["alive"][q] = ~((distance[q] == s1[row]) & (history["terminal"][row] != "none"))
    result["ke_MeV"][q] = energy
    result["p_MeV_c"][q] = np.sqrt(energy) * np.sqrt(energy + 2 * MMU)
    result["position_mm"][q] = history["entry_mm"][row] + fraction[:, None] * (history["exit_mm"][row] - history["entry_mm"][row])
    result["dt_from_start_ns"][q] = t0[row] + linear_kinetic_dt_ns(partial, kin, energy)
    result["segment"][q] = history["segment"][row]
    return result
