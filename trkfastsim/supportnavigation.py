"""Separate support-navigation candidate: guarded leaf replacement and whole-lane refusal.

Stored endpoint coalescence is an explicit numerical arm, not shared-face proof.
Omitted-solid brackets are NEVER coalesced, midpoint-filled, or assigned parent
material. No result here is approved for energy/time transport or GDML truth.
"""
import numpy as np
import pandas as pd

from . import boundarypolicy, materialpath, omittedpath
from .navigationcensus import _VACUUM

MODE = "support-navigation-torus-replace-refuse-v1"
POLICY = "refuse-whole-lane-on-any-unresolved-cell"
COLS = ["lane", "s0_mm", "s1_mm", "material"]
CELL_COLS = COLS + ["state", "ancestors_verified", "leaf_verified", "replaces_material", "source", "scope"]


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _bad_occupancy(hits):
    """Vectorized signed sweeps identify bad lanes before the independent strict partition."""
    if hits.empty:
        return np.empty(0, dtype=np.int64)
    bad = []
    sign = hits.sign.to_numpy(dtype=np.int64)
    for keys in (["lane", "material"], ["lane"]):
        events = pd.concat([hits[keys].assign(s=hits.s0_mm, delta=sign),
                            hits[keys].assign(s=hits.s1_mm, delta=-sign)], ignore_index=True)
        events = events.groupby(keys + ["s"], as_index=False).delta.sum()
        group = events.groupby(keys, sort=False)
        count = group.delta.cumsum()
        bad.append(events.loc[(group.s.shift(-1) > events.s) & ((count < 0) | (count > 1)), "lane"].to_numpy())
    return np.unique(np.concatenate(bad)).astype(np.int64)


def _overlap_lanes(cells):
    """Refuse overlapping possible leaf cells, independently of material/sign cancellation."""
    if cells.empty:
        return np.empty(0, dtype=np.int64)
    rows = cells.sort_values(["lane", "s0_mm", "s1_mm"], kind="stable")
    previous = rows.groupby("lane").s1_mm.cummax().groupby(rows.lane).shift(1)
    return rows.loc[rows.s0_mm < previous, "lane"].unique().astype(np.int64)


def _validate_evidence(cells, reports, records, lengths):
    """Require complete evaluated-placement accounting; an absent report is never a geometric miss."""
    if not isinstance(cells, pd.DataFrame) or not cells.columns.is_unique or not set(CELL_COLS) <= set(cells.columns):
        raise ValueError(f"omission cells require unique columns including {CELL_COLS}")
    boundarypolicy._validate(cells.assign(sign=1), lengths)
    if (cells.s1_mm <= cells.s0_mm).any() or not cells.state.isin(["inside", "unresolved"]).all():
        raise ValueError("omission cells must have positive width and recognized state")
    for key in ("ancestors_verified", "leaf_verified"):
        if len(cells) and pd.api.types.infer_dtype(cells[key], skipna=False) != "boolean":
            raise ValueError(f"{key} must contain actual booleans")
    for key in ("replaces_material", "source", "scope"):
        if len(cells) and (pd.api.types.infer_dtype(cells[key], skipna=False) != "string" or not cells[key].str.len().gt(0).all()):
            raise ValueError(f"{key} must contain nonempty strings")
    expected = [r for r in records if r["classification"] == "unsupported"]
    paths = [r["placement_path"] for r in expected]
    sources = [r["source"] for r in reports]
    if len(set(paths)) != len(paths) or len(set(sources)) != len(sources) or set(paths) != set(sources):
        raise ValueError("exactly one report per unsupported census placement required")
    if not set(cells.source) <= set(sources):
        raise ValueError("orphan omission cell source")
    lookup = {r["placement_path"]: r for r in expected}
    global_reasons = []
    for report in reports:
        record = lookup[report["source"]]
        for key in ("material", "solid_tag", "scope"):
            if report[key] != record[key]:
                raise ValueError(f"report {key} disagrees with census")
        if type(report.get("resolved")) is not bool or type(report.get("leaf")) is not bool:
            raise ValueError("report resolved/leaf must be actual booleans")
        if report["leaf"] != record["is_leaf"]:
            raise ValueError("report leaf disagrees with census")
        subset = cells[cells.source == report["source"]]
        if not report["resolved"]:
            if len(subset):
                raise ValueError("unevaluated placement has cells")
            global_reasons.append("unevaluated-placement:" + report["source"])
            continue
        if not report["leaf"]:
            global_reasons.append("non-leaf-placement:" + report["source"])
        parent = "vacuum" if _VACUUM.search(report["parent_material"]) else report["parent_material"]
        if not ((subset.material == report["material"]) & (subset.scope == report["scope"])
                & (subset.replaces_material == parent) & (subset.leaf_verified == report["leaf"])).all():
            raise ValueError("omission cell metadata disagrees with placement report")
        inside = subset.state == "inside"
        known = inside & subset.ancestors_verified & subset.leaf_verified
        metrics = dict(solid_lower_mm=omittedpath.length_bound(subset[inside], upper=False),
                       solid_upper_mm=omittedpath.length_bound(subset, upper=True),
                       hierarchy_verified_lower_mm=omittedpath.length_bound(subset[known], upper=False),
                       hit_lanes=int(subset.loc[inside, "lane"].nunique()), possible_lanes=int(subset.lane.nunique()))
        if any(report.get(key) != value for key, value in metrics.items()):
            raise ValueError("placement report bounds/counts disagree with retained cells")
        method = report.get("method", {})
        for key, value in (("inside_cells", int(inside.sum())), ("unresolved_cells", int((~inside).sum()))):
            if key in method and method[key] != value:
                raise ValueError("placement method counts disagree with retained cells")
        if (method.get("cap_reached", False) or method.get("capped", False)) and not (~inside).any():
            global_reasons.append("cap-without-unresolved-evidence:" + report["source"])
    return global_reasons


def compose(hits, lengths, cells, reports, records, *, endpoint_budget_mm, max_pairs=2000000):
    """Compose trusted probe evidence, retaining refusals and signed parent-replacement terms.

    Complete intervals are emitted ONLY for candidate-resolved whole lanes.
    ``replacement_terms`` are partial evidence, never a complete material path.
    Any unevaluated placement blocks all lanes. Invalid evidence raises. Stored
    occupancy, parent mismatch, leaf overlap, ancestry and unresolved-cell faults
    refuse individual lanes. Coalescence failure refuses the batch by raising.
    This low-level API assumes evidence came from omittedpath.probe on these rays;
    use navigate() to bind evaluation to the GDML and census in one call.
    """
    _positive_integer(max_pairs, "max_pairs")
    lengths, _, _, _, _ = boundarypolicy._validate(hits, lengths)
    global_reasons = _validate_evidence(cells, reports, records, lengths)
    adjusted, evidence = boundarypolicy.coalesce(hits, lengths, max_shift_mm=endpoint_budget_mm)
    n = len(lengths)
    issues = []

    def refuse(lanes, reason):
        lanes = np.unique(np.asarray(lanes, dtype=np.int64))
        if len(lanes):
            issues.append(pd.DataFrame(dict(lane=lanes, reason=reason)))

    bad_stored = _bad_occupancy(adjusted)
    refuse(bad_stored, "invalid-stored-occupancy")
    safe_lengths = lengths.copy()
    safe_lengths[bad_stored] = 0.
    stored = boundarypolicy.partition(adjusted[~adjusted.lane.isin(bad_stored)], safe_lengths)
    refuse(cells.loc[cells.state == "unresolved", "lane"], "unresolved-solid-boundary")
    refuse(cells.loc[~cells.ancestors_verified.astype(bool), "lane"], "unverified-ancestor")
    refuse(cells.loc[~cells.leaf_verified.astype(bool), "lane"], "unverified-leaf")
    overlaps = _overlap_lanes(cells)
    refuse(overlaps, "overlapping-possible-leaf-cells")
    known = cells[(cells.state == "inside") & cells.ancestors_verified.astype(bool)
                  & cells.leaf_verified.astype(bool) & ~cells.lane.isin(bad_stored)].copy()
    known["cell_id"] = np.arange(len(known), dtype=np.int64)
    sizes = known.groupby("lane").size().to_frame("n").join(stored.groupby("lane").size().rename("m")).fillna(0)
    if (sizes.n * sizes.m).sum() > max_pairs:
        raise ValueError("parent comparison exceeds explicit pair bound")
    old = stored.rename(columns={"s0_mm": "old_s0_mm", "s1_mm": "old_s1_mm", "material": "old_material"})
    pairs = known.merge(old, on="lane")
    pairs["s0_mm"] = np.maximum(pairs.s0_mm, pairs.old_s0_mm)
    pairs["s1_mm"] = np.minimum(pairs.s1_mm, pairs.old_s1_mm)
    pairs = pairs[pairs.s1_mm > pairs.s0_mm].copy()
    pairs["expected_parent"] = pairs.old_material == pairs.replaces_material
    missing = known[~known.cell_id.isin(pairs.cell_id)]
    mismatch = np.union1d(pairs.loc[~pairs.expected_parent, "lane"], missing.lane).astype(np.int64)
    refuse(mismatch, "stored-parent-mismatch")
    sizes = known.groupby("lane").size().to_frame("n").join(adjusted.groupby("lane").size().rename("m")).fillna(0)
    if (sizes.n * sizes.m).sum() > max_pairs:
        raise ValueError("raw ownership comparison exceeds explicit pair bound")
    raw = adjusted.rename(columns={"s0_mm": "raw_s0_mm", "s1_mm": "raw_s1_mm", "material": "raw_material", "source": "raw_source"})
    ownership = known.merge(raw.drop(columns=[k for k in raw if k not in
        ["lane", "raw_s0_mm", "raw_s1_mm", "raw_material", "raw_source", "sign"]]), on="lane")
    ownership["s0_mm"] = np.maximum(ownership.s0_mm, ownership.raw_s0_mm)
    ownership["s1_mm"] = np.minimum(ownership.s1_mm, ownership.raw_s1_mm)
    ownership = ownership[ownership.s1_mm > ownership.s0_mm].copy()
    ownership = ownership.astype({"s0_mm": "float64", "s1_mm": "float64"})
    positive = ownership[ownership.sign == 1].sort_values(["cell_id", "s0_mm", "s1_mm"], kind="stable")
    previous = positive.groupby("cell_id").s1_mm.cummax().groupby(positive.cell_id).shift(1)
    ambiguous = np.union1d(positive.loc[positive.s0_mm < previous, "lane"],
                          ownership.loc[(ownership.sign == -1) | (ownership.raw_material != ownership.replaces_material), "lane"]).astype(np.int64)
    refuse(ambiguous, "ambiguous-stored-parent-ownership")
    # Negative terms or multiple positive owners may hide same-material siblings; refuse rather than guess hierarchy.
    eligible = known[~known.lane.isin(np.union1d(np.union1d(overlaps, mismatch), ambiguous))]
    additions = eligible[COLS + ["source", "replaces_material", "cell_id"]].assign(sign=1, operation="add-leaf")
    subtraction = eligible[eligible.replaces_material != "vacuum"][COLS + ["source", "replaces_material", "cell_id"]].copy()
    subtraction["material"] = subtraction.replaces_material
    subtraction = subtraction.assign(sign=-1, operation="subtract-parent")
    terms = pd.concat([additions, subtraction], ignore_index=True)
    trial = pd.concat([stored[stored.material != "vacuum"].assign(sign=1), terms], ignore_index=True)
    refuse(_bad_occupancy(trial), "invalid-replacement-occupancy")
    for reason in global_reasons:
        refuse(np.arange(n), reason)
    refusals = (pd.concat(issues, ignore_index=True).sort_values(["lane", "reason"]).reset_index(drop=True)
                if issues else pd.DataFrame({"lane": pd.Series(dtype="int64"), "reason": pd.Series(dtype="str")}))
    refused = np.zeros(n, bool)
    refused[refusals.lane.to_numpy(dtype=np.int64)] = True
    admitted_lengths = lengths.copy()
    admitted_lengths[refused] = 0.
    intervals = boundarypolicy.partition(trial[~trial.lane.isin(np.flatnonzero(refused))], admitted_lengths)
    decisions = pd.DataFrame(dict(lane=np.arange(n), requested_length_mm=lengths,
                                  candidate_resolved=~refused, admitted_for_transport=False))
    summary = dict(mode=MODE, unresolved_policy=POLICY, endpoint_policy=evidence["summary"],
        rays=n, resolved_rays=int((~refused).sum()), refused_rays=int(refused.sum()), interval_rows=len(intervals),
        unresolved_cells=int((cells.state == "unresolved").sum()),
        unresolved_cell_width_upper_mm=omittedpath.length_bound(cells[cells.state == "unresolved"], upper=True),
        replacement_terms=len(terms), parent_comparisons=len(pairs), unexpected_parent_comparisons=int((~pairs.expected_parent).sum()),
        raw_ownership_comparisons=len(ownership), ambiguous_ownership_rays=len(ambiguous),
        global_refusals=global_reasons, all_rays_resolved=bool((~refused).all() and not global_reasons),
        admitted_for_transport=False)
    return dict(intervals=intervals, decisions=decisions, refusals=refusals, adjusted_hits=adjusted,
                endpoints=evidence["endpoints"], replacement_terms=terms, parent_comparisons=pairs,
                ownership_comparisons=ownership, summary=summary)


def navigate(gdml, census, origin, direction, lengths, hits, *, endpoint_budget_mm,
             min_cell_mm=1e-7, max_cells=1000000, max_pairs=2000000):
    """Evaluate all known omitted placements and compose a bounded stored-model candidate.

    The caller binds raw stored hits/census to these rays and geometry (the replay
    runner hashes and checks that provenance). Census bounds only select geometry;
    no AABB extent is ever assigned material. Source/model approximations persist.
    """
    origin, direction, lengths = materialpath._rays(origin, direction, lengths)
    _positive_integer(max_cells, "max_cells")
    _positive_integer(max_pairs, "max_pairs")
    if not np.isfinite(min_cell_mm) or min_cell_mm <= 0:
        raise ValueError("positive finite min_cell_mm required")
    if census.get("schema") != "tracker-navigation-census-v1":
        raise ValueError("recognized navigation census required")
    bbox = np.asarray(census["bbox_mm"], float)
    if bbox.shape != (2, 3) or not np.isfinite(bbox).all() or np.any(bbox[0] > bbox[1]):
        raise ValueError("finite ordered census bbox required")
    end = origin + direction * lengths[:, None]
    guard = omittedpath._roundoff(origin, end, bbox)
    if ((origin < bbox[0]-guard) | (origin > bbox[1]+guard) | (end < bbox[0]-guard) | (end > bbox[1]+guard)).any():
        raise ValueError("rays outside census selection domain")
    records = census["records"]
    paths = [r["placement_path"] for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate census placement")
    if census.get("placement_warnings") or any(r.get("placement_assumptions") for r in records):
        raise ValueError("unresolved census placement assumptions")
    classes = pd.Series([r["classification"] for r in records], dtype="str").value_counts().to_dict()
    allowed = {"unsupported", "retained", "approximate", "intentional_exclusion", "bbox_excluded"}
    if not set(classes) <= allowed:
        raise ValueError("unrecognized census classification")
    if any(classes.get(key, 0) != census.get("counts", {}).get(key, 0) for key in allowed):
        raise ValueError("census classification counts disagree")
    cells, reports = omittedpath.probe(gdml, records, origin, direction, lengths,
                                      min_cell_mm=min_cell_mm, max_cells=max_cells)
    result = compose(hits, lengths, cells, reports, records, endpoint_budget_mm=endpoint_budget_mm, max_pairs=max_pairs)
    result.update(omitted_cells=cells, placement_reports=reports)
    result["summary"].update(min_cell_mm=min_cell_mm, max_cells=max_cells, max_pairs=max_pairs,
        census_bbox_roundoff_guard_mm=guard, placements=len(reports))
    return result
