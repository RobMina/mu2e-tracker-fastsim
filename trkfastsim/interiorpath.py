"""Ordered support intervals from the stored tube and signed-convex interior representations."""
import numpy as np
import pandas as pd

from .materialpath import tube_intervals
from .convexpath import convex_intervals


def interior_intervals(shells, origin, direction, length):
    """Combine tube/calo partitions, rejecting any positive support overlap on requested rays.

    No material precedence is invented. Vacuum spans are reconstructed after merging solids.
    This navigates the stored approximate support geometry, not straw material or GDML truth.
    """
    coax = shells.coax if hasattr(shells, "coax") else shells
    tubes = tube_intervals(coax, origin, direction, length)
    calo = getattr(shells, "calo", None)
    if calo is None:
        return tubes
    convex = convex_intervals(calo, origin, direction, length)
    solid = pd.concat([tubes[tubes.material != "vacuum"], convex[convex.material != "vacuum"]], ignore_index=True)
    solid = solid.sort_values(["lane", "s0_mm"], kind="stable").reset_index(drop=True)
    lane = solid.lane.to_numpy(dtype=np.int64)
    s0, s1 = solid.s0_mm.to_numpy(float), solid.s1_mm.to_numpy(float)
    first = np.r_[True, lane[1:] != lane[:-1]] if len(lane) else np.empty(0, bool)
    previous = np.where(first, 0., np.r_[0., s1[:-1]]) if len(lane) else np.empty(0)
    if (s0 < previous).any():
        raise ValueError("overlap between tube and calorimeter support intervals")
    gap = s0 > previous
    gaps = pd.DataFrame(dict(lane=lane[gap], s0_mm=previous[gap], s1_mm=s0[gap], material="vacuum"))
    lengths = np.broadcast_to(np.asarray(length, float), (len(origin),))
    last = np.zeros(len(origin))
    np.maximum.at(last, lane, s1)
    tail = np.flatnonzero(last < lengths)
    tails = pd.DataFrame(dict(lane=tail, s0_mm=last[tail], s1_mm=lengths[tail], material="vacuum"))
    return pd.concat([solid, gaps, tails], ignore_index=True).sort_values(
        ["lane", "s0_mm"], kind="stable").reset_index(drop=True).astype(tubes.dtypes.to_dict())
