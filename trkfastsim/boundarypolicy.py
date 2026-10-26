"""Opt-in stored-endpoint diagnostics, never a production navigation tolerance.

A caller must name an absolute millimetre budget. Genuine geometric gaps and
overlaps below that resolution cannot be distinguished from endpoint roundoff.
Passing this policy is therefore not physical proof or transport admission; it
cannot repair omitted geometry, stored-shape approximations or hierarchy errors.
No production defaults or existing strict navigators are changed by this module.
"""
import math

import numpy as np
import pandas as pd


_DTYPES = {"lane": "int64", "s0_mm": "float64", "s1_mm": "float64", "material": "object"}
_REQUIRED = [*_DTYPES, "sign"]


def _real(values, name):
    """Reject coercion of strings, booleans, complex values and missing numbers."""
    array = np.asarray(values)
    if array.size and array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers, not strings or booleans")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validate(hits, lengths):
    """Require a lane-length vector and finite ordered intervals within its exact bounds."""
    if not isinstance(hits, pd.DataFrame):
        raise ValueError("hits must be a DataFrame")
    if not hits.columns.is_unique or not set(_REQUIRED).issubset(hits.columns):
        raise ValueError(f"hits require unique columns including {_REQUIRED}")
    lengths = _real(lengths, "lengths")
    if lengths.ndim != 1 or np.any(lengths < 0):
        raise ValueError("lengths must be a one-dimensional finite nonnegative vector")
    lane = _real(hits.lane, "lane")
    if np.any((lane < 0) | (lane >= len(lengths)) | (lane != np.floor(lane))):
        raise ValueError("lane must be an integer index into lengths")
    lane = lane.astype(np.int64)
    sign = _real(hits.sign, "sign")
    if np.any((sign != 1) & (sign != -1)):
        raise ValueError("sign must be +1 or -1")
    s0, s1 = _real(hits.s0_mm, "s0_mm"), _real(hits.s1_mm, "s1_mm")
    if np.any((s0 < 0) | (s1 < s0) | (s1 > lengths[lane])):
        raise ValueError("intervals must satisfy 0 <= s0_mm <= s1_mm <= lengths[lane]")
    material = hits.material
    if len(material) and (pd.api.types.infer_dtype(material, skipna=False) != "string"
                          or not material.str.len().gt(0).all()):
        raise ValueError("material must contain nonempty string keys")
    return lengths, lane, s0, s1, sign.astype(np.int64)


def coalesce(hits, lengths, *, max_shift_mm):
    """Return ``(adjusted_hits, {"endpoints": table, "summary": dict})``.

    Required hit columns are lane, s0_mm, s1_mm, material and sign (+1/-1).
    Lengths is a finite nonnegative (N,) vector; lane is an integer index into it.
    Raw endpoints must already obey 0 <= s0 <= s1 <= length: this is not clipping.
    Zero-width source intervals are allowed. A positive-width source interval of
    either sign may never collapse. All extra hit columns, index and row order
    are preserved; only the two endpoint columns are replaced in a fresh copy.

    Sort endpoints within each lane, including exact 0 and length anchors for
    lanes with hits. Adjacent gaps <= max_shift_mm define connected groups, but
    reject any entire group whose diameter exceeds that same explicit budget.
    Use the midpoint of its minimum and maximum, or the exact anchor when present.
    Distinct 0/length anchors in one group are incompatible and rejected. Empty
    lanes have no endpoint groups; their vacuum is supplied by partition().
    The budget is finite, nonnegative, absolute mm, with no relative tolerance.
    Zero means only exactly coincident endpoints can group.

    Evidence has one row per raw source endpoint (anchors are not source rows).
    row_position is the original positional identity even for duplicate indexes;
    row_index preserves its label; endpoint is s0_mm or s1_mm; sign is unchanged.
    group_id is lane-local, and group_size includes artificial anchors. Bounds,
    source/anchor counts, raw/canonical coordinates and signed shifts are retained.
    summary reports the actual maximum shift, total unsigned-source length change,
    signed-source length change and sum of absolute source-length changes. These
    sums are not occupied physical lengths when raw sources overlap or subtract.

    This function does NOT admit occupancy: call partition(adjusted_hits, lengths)
    separately. Sub-budget real gaps/overlaps may disappear, so success is only a
    stored-model diagnostic, not proof that the original geometry was correct.
    Sorting/reductions operate on all lanes together, without a per-lane loop.
    """
    lengths, lane, s0, s1, sign = _validate(hits, lengths)
    budget = _real(max_shift_mm, "max_shift_mm")
    if budget.ndim != 0 or budget < 0:
        raise ValueError("max_shift_mm must be an explicit finite nonnegative scalar")
    budget = float(budget)
    n = len(hits)
    active = np.unique(lane)
    raw = np.r_[s0, s1, np.zeros(len(active)), lengths[active]]
    lanes = np.r_[lane, lane, active, active]
    is_anchor = np.arange(len(raw)) >= 2 * n
    order = np.lexsort((raw, lanes))
    x, sorted_lane = raw[order], lanes[order]
    begins = np.r_[True, (sorted_lane[1:] != sorted_lane[:-1]) | (np.diff(x) > budget)] if len(x) else np.empty(0, bool)
    starts = np.flatnonzero(begins)
    group = np.cumsum(begins) - 1
    sizes = np.diff(np.r_[starts, len(x)])
    lo, hi = x[starts], x[starts + sizes - 1]
    if np.any(hi - lo > budget):
        raise ValueError("endpoint group diameter exceeds max_shift_mm (transitive chain rejected)")
    anchor_lo = np.minimum.reduceat(np.where(is_anchor[order], x, np.inf), starts)
    anchor_hi = np.maximum.reduceat(np.where(is_anchor[order], x, -np.inf), starts)
    anchored = np.isfinite(anchor_lo)
    if np.any(anchored & (anchor_lo != anchor_hi)):
        raise ValueError("endpoint group contains distinct 0/length anchors")
    canonical = np.where(anchored, anchor_lo, lo + .5 * (hi - lo))
    unsorted_group = np.empty(len(x), dtype=np.int64)
    unsorted_group[order] = group
    source_group = unsorted_group[:2 * n]
    source_canonical = canonical[source_group]
    a0, a1 = source_canonical[:n], source_canonical[n:]
    if np.any((s1 > s0) & (a1 <= a0)):
        raise ValueError("coalescence would collapse a positive-width source interval")
    shift = source_canonical - raw[:2 * n]
    if np.any(np.abs(shift) > budget):
        raise ValueError("actual endpoint shift exceeds max_shift_mm")
    adjusted = hits.copy(deep=True)
    adjusted["s0_mm"], adjusted["s1_mm"] = a0, a1
    group_lanes = sorted_lane[starts]
    first_lane = np.r_[True, group_lanes[1:] != group_lanes[:-1]] if len(starts) else np.empty(0, bool)
    local_group = np.arange(len(starts)) - np.maximum.accumulate(np.where(first_lane, np.arange(len(starts)), 0))
    anchor_count = np.add.reduceat(is_anchor[order].astype(np.int64), starts)
    endpoints = hits[["lane", "material", "sign"]].iloc[np.tile(np.arange(n), 2)].reset_index(drop=True).copy()
    endpoints["row_position"] = np.tile(np.arange(n), 2)
    endpoints["row_index"] = np.tile(hits.index.to_numpy(), 2)
    endpoints["endpoint"] = np.repeat(["s0_mm", "s1_mm"], n)
    endpoints["raw_endpoint_mm"] = raw[:2 * n]
    endpoints["canonical_endpoint_mm"] = source_canonical
    endpoints["shift_mm"] = shift
    endpoints["group_id"] = local_group[source_group]
    endpoints["group_min_mm"], endpoints["group_max_mm"] = lo[source_group], hi[source_group]
    endpoints["group_size"] = sizes[source_group]
    endpoints["group_anchor_size"] = anchor_count[source_group]
    endpoints["group_source_size"] = (sizes - anchor_count)[source_group]
    length_change = (a1 - a0) - (s1 - s0)
    try:
        total_change = math.fsum(length_change)
        signed_change = math.fsum(sign * length_change)
        absolute_change = math.fsum(np.abs(length_change))
    except OverflowError as error:
        raise ValueError("nonfinite total source length change") from error
    summary = dict(budget_mm=budget, actual_max_shift_mm=float(np.max(np.abs(shift), initial=0.)),
                   total_length_change_mm=total_change, signed_total_length_change_mm=signed_change,
                   total_absolute_length_change_mm=absolute_change, changed_endpoints=int(np.count_nonzero(shift)),
                   endpoint_groups=len(starts), source_endpoints=2 * n, anchor_endpoints=2 * len(active))
    return adjusted, dict(endpoints=endpoints, summary=summary)


def partition(hits, lengths):
    """Strict signed sweep returning lane/s0_mm/s1_mm/material, including vacuum.

    No tolerance, coalescence, union, negative clipping or material precedence is
    implicit here. Validate as in coalesce(), aggregate exactly coincident events,
    and require each (lane, material) occupancy to be 0 or 1 on every positive
    span. Then reject any cross-material overlap. Opposite signs cancel only
    within the same material; a positive net column cannot conceal negative spans.
    Zero-width hits contribute no occupancy. Material keys are literal strings,
    including 'vacuum' if explicitly supplied, not density-based classifications.

    Fill every remaining gap with material='vacuum', exactly covering [0, length]
    in each positive-length lane. Empty/zero-length lanes have respectively vacuum
    or no rows; an empty length vector returns the same empty four-column schema.
    Adjacent equal-material rows need not be merged. This partition does not retain
    source extras because signed cancellation can involve many source rows; use
    the untouched raw hits and coalesce evidence for provenance instead.
    """
    lengths, lane, s0, s1, sign = _validate(hits, lengths)
    solid = pd.DataFrame({key: np.empty(0, dtype=dtype) for key, dtype in _DTYPES.items()})
    if len(hits):
        events = pd.DataFrame(dict(lane=np.tile(lane, 2), material=np.tile(hits.material.to_numpy(), 2),
                                   s=np.r_[s0, s1], delta=np.r_[sign, -sign]))
        events = events.groupby(["lane", "material", "s"], as_index=False, sort=True).delta.sum()
        grouped = events.groupby(["lane", "material"], sort=False)
        count, end = grouped.delta.cumsum(), grouped.s.shift(-1)
        span = end > events.s
        if ((count[span] < 0) | (count[span] > 1)).any():
            raise ValueError("invalid signed occupancy: per-material multiplicity must be 0 or 1")
        keep = span & (count == 1)
        solid = pd.DataFrame(dict(lane=events.lane[keep], s0_mm=events.s[keep],
                                  s1_mm=end[keep], material=events.material[keep]))
        solid = solid.sort_values(["lane", "s0_mm"], kind="stable").reset_index(drop=True)
    lane = solid.lane.to_numpy(dtype=np.int64)
    s0, s1 = solid.s0_mm.to_numpy(), solid.s1_mm.to_numpy()
    first = np.r_[True, lane[1:] != lane[:-1]] if len(lane) else np.empty(0, bool)
    previous = np.where(first, 0., np.r_[0., s1[:-1]]) if len(lane) else np.empty(0)
    if np.any(s0 < previous):
        raise ValueError("cross-material overlapping intervals")
    gap = s0 > previous
    gaps = pd.DataFrame(dict(lane=lane[gap], s0_mm=previous[gap], s1_mm=s0[gap], material="vacuum"))
    last_end = np.zeros(len(lengths))
    np.maximum.at(last_end, lane, s1)
    tail = np.flatnonzero(last_end < lengths)
    tails = pd.DataFrame(dict(lane=tail, s0_mm=last_end[tail], s1_mm=lengths[tail], material="vacuum"))
    return pd.concat([solid, gaps, tails], ignore_index=True).sort_values(
        ["lane", "s0_mm"], kind="stable").reset_index(drop=True).astype(_DTYPES)
