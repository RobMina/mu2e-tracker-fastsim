"""Conditional guarded-envelope evolution policy, not complete GDML or upstream ownership."""
import numpy as np
import pandas as pd

from . import boundarypolicy, materialpath, strawpath

POLICY = "ordered-guarded-torus-sensitivity-v1"


def refine(support, thin, lengths):
    """Refine strict support with actual straw media; never resolve overlap by priority."""
    solids = pd.concat([support[support.material != "vacuum"].assign(kind="support"),
                        thin[thin.material != "vacuum"].assign(kind="thin")], ignore_index=True)
    result = materialpath._partition([solids[["lane", "s0_mm", "s1_mm", "material"]]], lengths)
    result = result.merge(solids, on=["lane", "s0_mm", "s1_mm", "material"], how="left",
                          validate="one_to_one").fillna({"kind": "vacuum"})
    previous = result.shift()
    new = ((result.lane != previous.lane) | (result.material != previous.material) |
           (result.kind != previous.kind) | (result.s0_mm != previous.s1_mm) | (result.kind == "thin"))
    return result.assign(block=new.cumsum()).groupby("block", sort=False, as_index=False).agg(
        lane=("lane", "first"), s0_mm=("s0_mm", "min"), s1_mm=("s1_mm", "max"),
        material=("material", "first"), kind=("kind", "first")).drop(columns="block")


def replacement_domain(cells, origin, direction, plane, *, allow_start_contact=False):
    """Closed forward split permits only an exact s=0 downward contact, never an upstream span."""
    lane = cells.lane.to_numpy(np.int64)
    start, end = cells.s0_mm.to_numpy(float), cells.s1_mm.to_numpy(float)
    y0 = origin[lane, 1] + direction[lane, 1] * start
    y1 = origin[lane, 1] + direction[lane, 1] * end
    strict = np.maximum(y0, y1) < plane
    contact = (start == 0.) & (y0 == plane) & (y1 < plane) & (direction[lane, 1] < 0)
    admitted = strict | (contact & allow_start_contact)
    evidence = cells.loc[~strict].copy().assign(y0_mm=y0[~strict], y1_mm=y1[~strict],
        exact_start_contact=contact[~strict], domain_admitted=admitted[~strict])
    return bool(admitted.all()), bool(strict.all()), evidence


class GuardedNavigator:
    """Recompute both geometric arms on each fresh ray; evolve only the explicitly named arm.

    The fixed upstream handoff is a conditional input, not repaired or certified.
    Every possible replacement cell must lie strictly below the world endpoint
    plane. This prevents this policy adding leaf material above that plane, but
    does not certify legacy backprojection or aggregate world-source ownership.
    Both envelopes retain their engineering guards, caps and raw-parent refusals.
    Their independently evolved trajectories are sensitivity arms, not bounds.
    """

    def __init__(self, *, arm, gdml, census, modules, raw_hits, world_plane_y_mm,
                 endpoint_budget_mm, min_cell_mm, max_component_mm, max_ray_unknown_mm,
                 max_cells=1000000, max_pairs=2000000, forward_only=False, ownership_normalizer=None, census_provider=None, allow_start_contact=False):
        if arm not in ("inner", "outer"):
            raise ValueError("explicit inner or outer sensitivity arm required")
        values = (world_plane_y_mm, endpoint_budget_mm, min_cell_mm, max_component_mm, max_ray_unknown_mm)
        if not np.isfinite(values).all() or min(values[1:]) < 0 or min_cell_mm == 0:
            raise ValueError("finite plane and explicit nonnegative engineering budgets required")
        for value in (max_cells, max_pairs):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError("positive integer geometry caps required")
        if census.get("schema") != "tracker-navigation-census-v1" or census.get("placement_warnings") or any(
                r.get("placement_assumptions") for r in census["records"]):
            raise ValueError("validated census without unresolved placement assumptions required")
        self.arm, self.gdml, self.census, self.modules, self.raw_hits = arm, gdml, census, modules, raw_hits
        self.plane, self.endpoint_budget = float(world_plane_y_mm), endpoint_budget_mm
        self.min_cell, self.component_budget, self.ray_budget = min_cell_mm, max_component_mm, max_ray_unknown_mm
        self.max_cells, self.max_pairs = max_cells, max_pairs
        self.policy = POLICY + ":" + arm
        self.calls = []
        if type(forward_only) is not bool:
            raise ValueError("forward_only must be explicit bool")
        self.forward_only = forward_only
        if type(allow_start_contact) is not bool or (allow_start_contact and not forward_only):
            raise ValueError("exact start-plane contact requires explicit forward-only policy")
        self.allow_start_contact = allow_start_contact
        if allow_start_contact:
            self.policy += ":exact-downward-zero-path-plane-contact-v1"
        self.primary_keys, self.query_lanes, self.query_iteration = None, None, None
        if forward_only:
            self.policy += ":forward-domain"
        self.ownership_normalizer = ownership_normalizer
        self.census_provider = census_provider
        if ownership_normalizer is not None:
            from .placementownership import POLICY as ownership_policy
            self.policy += ":" + ownership_policy

    def bind_primary_keys(self, keys):
        keys = np.asarray(keys)
        if keys.ndim != 2 or keys.shape[1] != 4 or keys.dtype.kind not in "iu" or len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError("unique full integer primary keys required")
        self.primary_keys = keys.copy()

    def bind_query_lanes(self, lanes, iteration):
        lanes = np.asarray(lanes)
        if lanes.ndim != 1 or lanes.dtype.kind not in "iu" or (lanes < 0).any() or len(np.unique(lanes)) != len(lanes):
            raise ValueError("unique nonnegative query-lane map required")
        if self.primary_keys is not None and (lanes >= len(self.primary_keys)).any():
            raise ValueError("query-lane map outside primary identities")
        self.query_lanes, self.query_iteration = lanes.copy(), int(iteration)

    def __call__(self, shells, geom, origin, direction, lengths, *, endpoint_budget_mm=None,
                 endpoint_evidence=None, leading_resolution_mm=None):
        origin, direction, lengths = materialpath._rays(origin, direction, lengths)
        if endpoint_budget_mm != self.endpoint_budget:
            raise ValueError("transport and guarded stored-endpoint budgets disagree")
        call = dict(origin=origin.copy(), direction=direction.copy(), lengths=lengths.copy(), stage="probe", status="running")
        self.calls.append(call)
        if self.query_lanes is not None:
            if len(self.query_lanes) != len(origin):
                raise ValueError("query lanes do not cover all fresh rays")
            call["interior_lanes"], call["iteration"] = self.query_lanes.copy(), self.query_iteration
            if self.primary_keys is not None:
                call["primary_keys"] = self.primary_keys[self.query_lanes].copy()
        try:
            if self.forward_only:
                call["stage"] = "forward-domain"
                from .orderedhandoff import validate_forward_query
                validate_forward_query(origin, direction, lengths, self.plane)
            call["stage"] = "probe"
            return self._query(geom, origin, direction, lengths, endpoint_evidence, leading_resolution_mm, call)
        except (ValueError, RuntimeError, AttributeError, TypeError) as error:
            call.update(status="refused", reason=str(error), details=getattr(error, "details", None))
            raise

    def _query(self, geom, origin, direction, lengths, endpoint_evidence, leading_resolution_mm, call):
        if self.census_provider is not None:
            raw = self.raw_hits(origin, direction, lengths)
            self.census = self.census_provider(np.asarray(self.raw_hits.bbox))
            if self.census.get("placement_warnings") or any(r.get("placement_assumptions") for r in self.census["records"]):
                raise ValueError("expanded census has unresolved placement assumptions")
            call["expanded_census"] = self.census
        else:
            raw = None
        cells, reports = self.modules["omittedpath"].probe(self.gdml, self.census["records"], origin, direction, lengths,
            min_cell_mm=self.min_cell, max_cells=self.max_cells)
        call.update(cells=cells.copy(), reports=reports, stage="raw-stored")
        if raw is None:
            raw = self.raw_hits(origin, direction, lengths)
        call.update(raw_hits=raw.copy(), stage="stored-endpoints")
        if hasattr(self.raw_hits, "bbox"):
            call["source_selection_bbox_mm"] = np.asarray(self.raw_hits.bbox).copy()
            call["physical_source_terms"] = self.raw_hits.relationships.copy()
        adjusted, evidence = boundarypolicy.coalesce(raw, lengths, max_shift_mm=self.endpoint_budget,
                                                     leading_resolution_mm=leading_resolution_mm)
        call.update(adjusted_hits=adjusted.copy(), endpoints=evidence, stage="endpoint-callback")
        if endpoint_evidence is not None:
            endpoint_evidence(evidence)
        if self.ownership_normalizer is not None:
            call["stage"] = "physical-owner-normalization"
            normalized, comparison = self.ownership_normalizer(adjusted, lengths)
            call["ownership_normalized_hits"], call["ownership_material_comparison"] = normalized.copy(), comparison
            from .placementownership import parent_owners
            call["physical_parent_comparisons"] = parent_owners(cells, normalized, max_pairs=self.max_pairs)
            adjusted = normalized
        call["stage"] = "paired-composition"
        result = self.modules["supportenvelope"].compose(adjusted, lengths, cells, reports, self.census["records"],
            endpoint_budget_mm=0., min_cell_mm=self.min_cell, max_component_mm=self.component_budget,
            max_ray_unknown_mm=self.ray_budget, max_pairs=self.max_pairs)
        allowed, strict, plane_evidence = replacement_domain(cells, origin, direction, self.plane,
                                                            allow_start_contact=self.allow_start_contact)
        call.update(result=result, replacements_strictly_below_plane=strict, replacements_within_forward_domain=allowed,
            plane_contacts=plane_evidence, allow_start_contact=self.allow_start_contact, stage="world-plane")
        if not allowed:
            raise ValueError("guarded replacement crosses or lies above world endpoint plane; no evolution")
        if not result["summary"]["all_pairs_admitted"]:
            raise ValueError("guarded geometric pair refused; no partial-lane evolution")
        call["stage"] = "paired-straw-refinement"
        thin = strawpath.straw_intervals(geom, origin, direction, lengths)
        call["straw_intervals"] = thin.copy()
        refined = {arm: refine(result[arm + "_intervals"], thin, lengths) for arm in ("inner", "outer")}
        call.update(refined=refined, stage="complete", status="complete")
        return refined[self.arm]
