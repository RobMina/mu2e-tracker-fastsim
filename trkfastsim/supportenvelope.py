"""Quantified torus occupancy envelopes, not resolved crossings or transport admission.

An explicit geometric budget admits a PAIR of complete stored-model partitions:
inner uses certified leaf cells only; outer also uses eligible possible cells.
Neither is a guessed physical boundary. Per-material bounds use atomic label
alternatives, not min/max of whole-arm columns, which can cancel uncertainty.
The unchanged refusal-first candidate remains the control.
"""
import numpy as np
import pandas as pd

from . import boundarypolicy, supportnavigation as strict

MODE = "support-navigation-torus-envelope-v1"
POLICY = "paired-inner-outer-min-width-torus-bands"
_EPS = np.finfo(float).eps


def _budget(value, name):
    value = boundarypolicy._real(value, name)
    if value.ndim != 0 or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(value)


def grouped_bounds(spans, keys):
    """Outward per-group sums of nonnegative binary64 endpoint differences, without row loops."""
    if spans.empty:
        return pd.DataFrame(columns=keys + ["lower_mm", "upper_mm"])
    width = spans.s1_mm.to_numpy(float) - spans.s0_mm.to_numpy(float)
    if not np.isfinite(width).all() or (width < 0).any():
        raise ValueError("finite ordered spans required for length bounds")
    terms = spans[keys].copy()
    terms["lo"] = np.where(width > 0, np.nextafter(width, 0.), 0.)
    terms["hi"] = np.where(width > 0, np.nextafter(width, np.inf), 0.)
    groups = terms.groupby(keys, as_index=False, sort=True).agg(lo=("lo", "sum"), hi=("hi", "sum"), n=("lo", "size"))
    gamma = (groups.n.to_numpy(float) + 2.) * _EPS
    if (gamma >= .5).any():
        raise ValueError("length reduction exceeds supported floating-point bound")
    gamma = gamma / (1. - gamma)
    groups["lower_mm"] = np.maximum(0., np.nextafter(groups.lo / (1. + gamma), -np.inf))
    groups["upper_mm"] = np.where(groups.hi > 0, np.nextafter(groups.hi / (1. - gamma), np.inf), 0.)
    if not np.isfinite(groups[["lower_mm", "upper_mm"]].to_numpy(float)).all():
        raise ValueError("nonfinite aggregate length bound")
    return groups[keys + ["lower_mm", "upper_mm"]]


def unknown_components(cells, lengths):
    """Exact-touch union per lane across placements; keep original rows and component membership."""
    unknown = cells[cells.state == "unresolved"].sort_values(["lane", "s0_mm", "s1_mm", "source"], kind="stable").copy()
    if unknown.empty:
        unknown["component"] = pd.Series(dtype="int64")
        components = pd.DataFrame({"component": pd.Series(dtype="int64"), "lane": pd.Series(dtype="int64"),
                                  "s0_mm": pd.Series(dtype=float), "s1_mm": pd.Series(dtype=float),
                                  "cells": pd.Series(dtype="int64"), "sources": pd.Series(dtype="int64"),
                                  "lower_mm": pd.Series(dtype=float), "upper_mm": pd.Series(dtype=float)})
    else:
        previous = unknown.groupby("lane").s1_mm.cummax().groupby(unknown.lane).shift(1)
        unknown["component"] = (previous.isna() | (unknown.s0_mm > previous)).cumsum().to_numpy() - 1
        components = unknown.groupby("component", as_index=False, sort=True).agg(lane=("lane", "first"),
            s0_mm=("s0_mm", "min"), s1_mm=("s1_mm", "max"), cells=("lane", "size"), sources=("source", "nunique"))
        components = components.merge(grouped_bounds(components, ["component"]), on="component", validate="one_to_one")
    rays = pd.DataFrame(dict(lane=np.arange(len(lengths), dtype=np.int64)))
    rays = rays.merge(grouped_bounds(components, ["lane"]), on="lane", how="left").fillna(0.)
    rays = rays.rename(columns={"lower_mm": "unknown_lower_mm", "upper_mm": "unknown_upper_mm"})
    return unknown, components, rays


def _eligibility(cells, reports, lengths, min_cell_mm):
    """No cap/stall or untyped uncertainty may enter the quantified boundary policy."""
    refused = []
    for report in reports:
        if not report["resolved"]:
            continue
        subset = cells[cells.source == report["source"]]
        unknown = subset[subset.state == "unresolved"]
        method = report.get("method", {})
        if report["solid_tag"] == "torus":
            if type(method.get("cap_reached")) is not bool:
                raise ValueError("torus cap_reached must be an explicit boolean")
            if "capped" in method and (type(method["capped"]) is not bool or method["capped"] != method["cap_reached"]):
                raise ValueError("conflicting cap metadata aliases")
            if "reason" not in subset or not subset.reason.isin(["certificate", "min_width", "roundoff", "max_nodes", "max_active", "max_rounds"]).all():
                raise ValueError("torus cells require recognized explicit reasons")
            if not (subset.loc[subset.state == "inside", "reason"] == "certificate").all():
                raise ValueError("inside torus cells require certificate reasons")
            metrics = dict(lanes=len(lengths), positive_length_lanes=int(np.count_nonzero(lengths)),
                inside_cells=int((subset.state == "inside").sum()), unresolved_cells=len(unknown),
                unresolved_lanes=int(unknown.lane.nunique()),
                unresolved_reasons={str(k): int(v) for k, v in unknown.reason.value_counts().items()})
            if any(method.get(key) != value for key, value in metrics.items()):
                raise ValueError("torus reason/count evidence disagrees with probe metadata")
            if unknown.reason.isin(["max_nodes", "max_active", "max_rounds"]).any() and not method["cap_reached"]:
                raise ValueError("cap reason contradicts uncapped placement metadata")
            if ((unknown.reason == "min_width") & ((unknown.s1_mm - unknown.s0_mm) > min_cell_mm)).any():
                raise ValueError("min_width cell exceeds the declared probe width")
            midpoint = unknown.s0_mm + (unknown.s1_mm - unknown.s0_mm) * .5
            if ((unknown.reason == "min_width") & ((midpoint <= unknown.s0_mm) | (midpoint >= unknown.s1_mm))).any():
                raise ValueError("stalled midpoint contradicts min_width reason; roundoff is required")
            bad = unknown[unknown.reason != "min_width"]
            if len(bad):
                refused.append(pd.DataFrame(dict(lane=bad.lane.unique(), reason="ineligible-torus-boundary-reason")))
            capped = method["cap_reached"]
        else:
            if type(method.get("capped")) is not bool:
                raise ValueError("non-torus probe capped must be an explicit boolean")
            if "cap_reached" in method and (type(method["cap_reached"]) is not bool or method["cap_reached"] != method["capped"]):
                raise ValueError("conflicting cap metadata aliases")
            capped = method["capped"]
            if len(unknown):
                refused.append(pd.DataFrame(dict(lane=unknown.lane.unique(), reason="non-torus-unresolved-boundary")))
        if capped:
            refused.append(pd.DataFrame(dict(lane=np.arange(len(lengths)), reason="capped-placement-probe")))
    return refused


def _parent_checks(cells, stored, adjusted, bad_stored, max_pairs):
    """Check ALL possible cells, including unknown bands, against the coalesced stored source terms."""
    checked = cells[~cells.lane.isin(bad_stored)].copy()
    checked["cell_id"] = np.arange(len(checked), dtype=np.int64)
    for table in (stored, adjusted):
        sizes = checked.groupby("lane").size().to_frame("n").join(table.groupby("lane").size().rename("m")).fillna(0)
        if (sizes.n * sizes.m).sum() > max_pairs:
            raise ValueError("possible-cell comparison exceeds explicit pair bound")
    old = stored.rename(columns={"s0_mm": "old_s0_mm", "s1_mm": "old_s1_mm", "material": "old_material"})
    pairs = checked.merge(old, on="lane")
    pairs["s0_mm"] = np.maximum(pairs.s0_mm, pairs.old_s0_mm)
    pairs["s1_mm"] = np.minimum(pairs.s1_mm, pairs.old_s1_mm)
    pairs = pairs[pairs.s1_mm > pairs.s0_mm].copy()
    pairs["expected_parent"] = pairs.old_material == pairs.replaces_material
    missing = checked[~checked.cell_id.isin(pairs.cell_id)]
    mismatch = np.union1d(pairs.loc[~pairs.expected_parent, "lane"], missing.lane).astype(np.int64)
    raw = adjusted.rename(columns={"s0_mm": "raw_s0_mm", "s1_mm": "raw_s1_mm", "material": "raw_material", "source": "raw_source"})
    raw = raw[[k for k in ("lane", "raw_s0_mm", "raw_s1_mm", "raw_material", "raw_source", "sign") if k in raw]]
    ownership = checked.merge(raw, on="lane")
    ownership["s0_mm"] = np.maximum(ownership.s0_mm, ownership.raw_s0_mm)
    ownership["s1_mm"] = np.minimum(ownership.s1_mm, ownership.raw_s1_mm)
    ownership = ownership[ownership.s1_mm > ownership.s0_mm].astype({"s0_mm": "float64", "s1_mm": "float64"})
    positive = ownership[ownership.sign == 1].sort_values(["cell_id", "s0_mm", "s1_mm"], kind="stable")
    previous = positive.groupby("cell_id").s1_mm.cummax().groupby(positive.cell_id).shift(1)
    ambiguous = np.union1d(positive.loc[positive.s0_mm < previous, "lane"],
        ownership.loc[(ownership.sign == -1) | (ownership.raw_material != ownership.replaces_material), "lane"]).astype(np.int64)
    return pairs, ownership, mismatch, ambiguous


def _terms(cells, arm):
    eligible = cells if arm == "outer" else cells[cells.state == "inside"]
    columns = strict.COLS + ["source", "replaces_material", "state"]
    additions = eligible[columns].assign(sign=1, operation="add-leaf", envelope=arm)
    subtraction = eligible[eligible.replaces_material != "vacuum"][columns].copy()
    subtraction["material"] = subtraction.replaces_material
    subtraction = subtraction.assign(sign=-1, operation="subtract-parent", envelope=arm)
    return pd.concat([additions, subtraction], ignore_index=True)


def _lookup(intervals, lanes, starts):
    """Vectorized lexicographic point lookup on a validated disjoint partition or union."""
    rows = intervals.sort_values(["lane", "s0_mm"], kind="stable")
    source = np.rec.fromarrays([rows.lane.to_numpy(np.int64), rows.s0_mm.to_numpy(float)], names="lane,s")
    query = np.rec.fromarrays([lanes, starts], names="lane,s")
    index = np.searchsorted(source, query, side="right") - 1
    valid = index >= 0
    if len(rows):
        safe = np.maximum(index, 0)
        valid &= (rows.lane.to_numpy()[safe] == lanes) & (rows.s1_mm.to_numpy()[safe] > starts)
    else:
        valid[:] = False
    return rows, index, valid


def _atomic_pair(inner, outer, components):
    endpoints = []
    for rows in (inner, outer, components):
        for key in ("s0_mm", "s1_mm"):
            endpoints.append(rows[["lane", key]].rename(columns={key: "s0_mm"}))
    edges = pd.concat(endpoints, ignore_index=True).drop_duplicates().sort_values(["lane", "s0_mm"], kind="stable")
    edges["s1_mm"] = edges.groupby("lane").s0_mm.shift(-1)
    atoms = edges[edges.s1_mm > edges.s0_mm].reset_index(drop=True).astype({"lane": "int64", "s0_mm": "float64", "s1_mm": "float64"})
    lane, start = atoms.lane.to_numpy(np.int64), atoms.s0_mm.to_numpy(float)
    for arm, intervals in (("inner", inner), ("outer", outer)):
        rows, index, valid = _lookup(intervals, lane, start)
        if not valid.all() or (len(index) and (atoms.s1_mm.to_numpy() > rows.s1_mm.to_numpy()[index]).any()):
            raise ValueError("envelope arms do not share complete atomic coverage")
        atoms[arm + "_material"] = rows.material.to_numpy()[index]
    _, _, disputed = _lookup(components, lane, start)
    atoms["disputed_geometry"] = disputed
    if ((atoms.inner_material != atoms.outer_material) & ~atoms.disputed_geometry).any():
        raise ValueError("envelopes differ outside explicit uncertainty components")
    return atoms


def _material_bounds(atoms, admitted, lengths):
    same = atoms.inner_material == atoms.outer_material
    certain = atoms[same].rename(columns={"inner_material": "material"})
    possible = pd.concat([atoms.rename(columns={"inner_material": "material"}),
                          atoms[~same].rename(columns={"outer_material": "material"})], ignore_index=True)
    low = grouped_bounds(certain, ["lane", "material"])[["lane", "material", "lower_mm"]]
    high = grouped_bounds(possible, ["lane", "material"])[["lane", "material", "upper_mm"]]
    bounds = high.merge(low, on=["lane", "material"], how="left").fillna({"lower_mm": 0.})
    bounds["upper_mm"] = np.minimum(bounds.upper_mm.to_numpy(float), lengths[bounds.lane.to_numpy(np.int64)])
    lo = grouped_bounds(atoms[(atoms.inner_material != "vacuum") & (atoms.outer_material != "vacuum")], ["lane"])
    hi = grouped_bounds(atoms[(atoms.inner_material != "vacuum") | (atoms.outer_material != "vacuum")], ["lane"])
    aggregate = pd.DataFrame(dict(lane=np.flatnonzero(admitted)))
    aggregate = aggregate.merge(lo[["lane", "lower_mm"]], on="lane", how="left").merge(
        hi[["lane", "upper_mm"]], on="lane", how="left").fillna(0.)
    aggregate["upper_mm"] = np.minimum(aggregate.upper_mm.to_numpy(float), lengths[aggregate.lane.to_numpy(np.int64)])
    aggregate["quantity"] = "all-nonvacuum-path"
    return bounds[["lane", "material", "lower_mm", "upper_mm"]], aggregate


def compose(hits, lengths, cells, reports, records, *, endpoint_budget_mm, min_cell_mm,
            max_component_mm, max_ray_unknown_mm, max_pairs=2000000):
    """Admit only paired geometric envelopes, retaining the unchanged refusal control.

    Budgets bound connected unknown UNION components and each ray's union-length
    sum. They are caller-declared geometric resolutions, not physics tolerances.
    Parent and ownership checks apply to the full possible footprint. A failure
    of either arm removes BOTH arms for the lane. Output atoms retain geometric
    dispute even for equal parent/leaf labels. Sources and original cell states
    are never rewritten as certified. No energy or timing model is called.
    """
    component_budget = _budget(max_component_mm, "max_component_mm")
    ray_budget = _budget(max_ray_unknown_mm, "max_ray_unknown_mm")
    cell_width = _budget(min_cell_mm, "min_cell_mm")
    if cell_width <= 0:
        raise ValueError("min_cell_mm must be positive")
    control = strict.compose(hits, lengths, cells, reports, records,
                             endpoint_budget_mm=endpoint_budget_mm, max_pairs=max_pairs)
    lengths = np.asarray(lengths, float)
    adjusted = control["adjusted_hits"]
    unknown, components, uncertainty = unknown_components(cells, lengths)
    issues = [control["refusals"][control["refusals"].reason != "unresolved-solid-boundary"].copy()]
    issues.extend(_eligibility(cells, reports, lengths, cell_width))

    def refuse(lanes, reason):
        lanes = np.unique(np.asarray(lanes, dtype=np.int64))
        if len(lanes):
            issues.append(pd.DataFrame(dict(lane=lanes, reason=reason)))

    refuse(components.loc[components.upper_mm > component_budget, "lane"], "unknown-component-budget-exceeded")
    refuse(uncertainty.loc[uncertainty.unknown_upper_mm > ray_budget, "lane"], "unknown-ray-budget-exceeded")
    bad_stored = strict._bad_occupancy(adjusted)
    safe_lengths = lengths.copy()
    safe_lengths[bad_stored] = 0.
    stored = boundarypolicy.partition(adjusted[~adjusted.lane.isin(bad_stored)], safe_lengths)
    pairs, ownership, mismatch, ambiguous = _parent_checks(cells, stored, adjusted, bad_stored, max_pairs)
    refuse(mismatch, "possible-cell-parent-mismatch")
    refuse(ambiguous, "possible-cell-ambiguous-ownership")
    refusals = pd.concat(issues, ignore_index=True).drop_duplicates().sort_values(["lane", "reason"], kind="stable").reset_index(drop=True)
    accepted = np.ones(len(lengths), bool)
    accepted[refusals.lane.to_numpy(np.int64)] = False
    accepted_lengths = lengths.copy()
    accepted_lengths[~accepted] = 0.
    selected = cells[cells.lane.isin(np.flatnonzero(accepted))]
    base = stored[stored.lane.isin(np.flatnonzero(accepted)) & (stored.material != "vacuum")].assign(sign=1)
    products, terms = {}, {}
    for arm in ("inner", "outer"):
        terms[arm] = _terms(selected, arm)
        trial = pd.concat([base, terms[arm]], ignore_index=True)
        bad = strict._bad_occupancy(trial)
        if len(bad):
            raise ValueError("envelope composition violated prechecked strict occupancy; no pair returned")
        products[arm] = boundarypolicy.partition(trial, accepted_lengths)
    selected_components = components[components.lane.isin(np.flatnonzero(accepted))]
    atoms = _atomic_pair(products["inner"], products["outer"], selected_components)
    bounds, aggregate = _material_bounds(atoms, accepted, lengths)
    decisions = uncertainty.assign(requested_length_mm=lengths, bounded_pair_admitted=accepted,
        strict_candidate_resolved=control["decisions"].candidate_resolved.to_numpy(), admitted_for_transport=False)
    components["within_component_budget"] = components.upper_mm <= component_budget
    summary = dict(mode=MODE, unresolved_policy=POLICY, endpoint_budget_mm=float(endpoint_budget_mm),
        min_cell_mm=cell_width, max_component_mm=component_budget, max_ray_unknown_mm=ray_budget,
        rays=len(lengths), bounded_pair_rays=int(accepted.sum()), refused_rays=int((~accepted).sum()),
        strict_resolved_rays=control["summary"]["resolved_rays"], unknown_cells=len(unknown),
        unknown_components=len(components), max_component_upper_mm=float(components.upper_mm.max()) if len(components) else 0.,
        max_ray_unknown_upper_mm=float(uncertainty.unknown_upper_mm.max()) if len(uncertainty) else 0.,
        parent_comparisons=len(pairs), unexpected_parent_comparisons=int((~pairs.expected_parent).sum()),
        ownership_comparisons=len(ownership), ambiguous_ownership_rays=len(ambiguous),
        inner_interval_rows=len(products["inner"]), outer_interval_rows=len(products["outer"]), atom_rows=len(atoms),
        all_pairs_admitted=bool(accepted.all() and not control["summary"]["global_refusals"]),
        admitted_for_transport=False, guard_contract="engineering float64 envelopes, not machine-verified intervals or G4 truth")
    return dict(inner_intervals=products["inner"], outer_intervals=products["outer"], atoms=atoms,
        material_bounds=bounds, nonvacuum_bounds=aggregate, decisions=decisions, refusals=refusals,
        unknown_cells=unknown, unknown_components=components, parent_comparisons=pairs, ownership_comparisons=ownership,
        replacement_terms=pd.concat([terms["inner"], terms["outer"]], ignore_index=True),
        control=control, summary=summary)
