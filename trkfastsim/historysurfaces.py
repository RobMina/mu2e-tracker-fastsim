"""Explicit physical-history surface projection; production surface_steps is unchanged."""
import numpy as np
import pandas as pd

from . import historystate, surfaces
from .transportledger import KEY


def surface_steps(history, starts, *, energy_evolution):
    """Project established straight physical segments using the shared KE/time query.

    History positions must explicitly be in the detector frame. starts is a DataFrame
    with exactly run/subrun/event/sp/t_start_ns, one finite anchor per recorded muon.
    No generator or straw-hit selection is inferred. Unknown times, curved physical
    segments and unsupported evolution are refused by the shared query. Zero-path
    displacements do not produce physical surface crossings. A final terminal endpoint
    is not transmitted, and no extrapolated approach/exit segment is manufactured.
    Surface rows use the standard VD thickness convention, not energy deposition in it.
    """
    keys = list(KEY[:-1])
    if history.metadata["frame"] != "detector":
        raise ValueError("history surface projection requires frame='detector'")
    if not isinstance(starts, pd.DataFrame) or set(starts.columns) != set(keys + ["t_start_ns"]):
        raise ValueError("starts must contain full identities and t_start_ns only")
    if any(starts[k].dtype.kind not in "iu" for k in keys):
        raise ValueError("start identities must be nonnegative int64-compatible integers")
    if any(((starts[k] < 0) | (starts[k] > np.iinfo(np.int64).max)).any() for k in keys):
        raise ValueError("start identities must be nonnegative int64-compatible integers")
    if (starts.sp > np.iinfo(np.int32).max).any():
        raise ValueError("SurfaceStep output requires sp representable as int32")
    if starts.duplicated(keys).any() or not np.isfinite(starts.t_start_ns.to_numpy(float)).all():
        raise ValueError("start identities must be unique with finite time anchors")
    identity = np.column_stack([history[k] for k in keys])
    historystate.at_path(history, np.empty((0, 4), np.int64), [], energy_evolution=energy_evolution)
    anchors = pd.DataFrame(identity, columns=keys).drop_duplicates().merge(
        starts, how="outer", on=keys, indicator=True, validate="one_to_one")
    if not (anchors._merge == "both").all():
        raise ValueError("start identities must exactly cover recorded histories")
    physical = np.flatnonzero((history["kind"] == "physical") & (history["path_mm"] > 0))
    if not len(physical):
        return pd.DataFrame({c: np.zeros(0) for c in surfaces.COLS})
    s0, s1, _, _ = historystate.path_coordinates(history)
    a, b = history["entry_mm"][physical], history["exit_mm"][physical]
    row, fraction, sid = surfaces._crossings(a, b)
    direction_all = (b - a) / history["path_mm"][physical, None]
    contact = a[row] + fraction[:, None] * (b[row] - a[row])
    transverse = np.einsum("ij,ij->i", contact[:, :2], direction_all[row, :2])
    crossing = (sid <= 2) | (transverse != 0.)
    row, fraction, sid = row[crossing], fraction[crossing], sid[crossing]
    segment = physical[row]
    span = s1[segment] - s0[segment]
    distance = np.where(fraction <= .5, s0[segment] + fraction * span,
                        s1[segment] - (1. - fraction) * span)
    state = historystate.at_path(history, identity[segment], distance, energy_evolution=energy_evolution)
    if not (state["available"] & state["alive"]
            & (state["segment"] == history["segment"][segment])).all():
        raise ValueError("physical crossing is not representable in its live history segment")
    src = pd.DataFrame(identity[segment], columns=keys).merge(
        starts, on=keys, how="left", sort=False, validate="many_to_one")
    direction = (b[row] - a[row]) / history["path_mm"][segment, None]
    with np.errstate(over="ignore", invalid="ignore"):
        absolute_time = src.t_start_ns.to_numpy(float) + state["dt_from_start_ns"]
    if not np.isfinite(absolute_time).all():
        raise ValueError("absolute crossing time must be finite")
    return surfaces._rows(src, sid, state["position_mm"], direction, state["p_MeV_c"], absolute_time)
