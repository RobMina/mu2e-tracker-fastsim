"""Separate mixed torus/xtru geometric pairs; never transport admission.

Reviewed helpers and the refusal-first control are used without mutation. Xtru
plane-offset evidence is authenticated separately, never relabelled as torus.
"""
import numpy as np
import pandas as pd

MODE = "support-navigation-mixed-torus-xtru-envelope-v1"
POLICY = "paired-min-width-torus-and-plane-offset-xtru-bands"
_XTRU_METHOD = "core prism triangulation with explicit inner/outer plane-offset bands"


class MixedEnvelope:
    """Compose trusted retained evidence with the unchanged reviewed control."""

    def __init__(self, reviewedEnvelope, reviewedStrict):
        self.reviewed = reviewedEnvelope
        self.strict = reviewedStrict

    def _xtru_eligibility(self, cells, reports, lengths):
        """Authenticate only the explicit uncapped plane-offset certificate scheme."""
        authenticated = set()
        for report in reports:
            if report["solid_tag"] != "xtru" or not report["resolved"]:
                continue
            subset = cells[cells.source == report["source"]]
            unknown = subset[subset.state == "unresolved"]
            method = report.get("method", {})
            if not isinstance(method, dict) or method.get("certificate_scheme") != "xtru-plane-offset-v1":
                raise ValueError("xtru requires authenticated plane-offset certificate metadata")
            if method.get("method") != _XTRU_METHOD:
                raise ValueError("unrecognized xtru plane-offset method")
            if type(method.get("capped")) is not bool or method["capped"]:
                raise ValueError("xtru plane-offset certificate requires explicit capped=False")
            if "cap_reached" in method and (type(method["cap_reached"]) is not bool or method["cap_reached"] != method["capped"]):
                raise ValueError("conflicting cap metadata aliases")
            self.reviewed._budget(method.get("guard_mm"), "xtru guard_mm")
            if "reason" not in subset or not subset.reason.isin(["certificate", "plane_offset"]).all():
                raise ValueError("xtru cells require recognized explicit reasons")
            if not (subset.loc[subset.state == "inside", "reason"] == "certificate").all() or not (unknown.reason == "plane_offset").all():
                raise ValueError("xtru cell state disagrees with certificate reason")
            metrics = dict(lanes=len(lengths), positive_length_lanes=int(np.count_nonzero(lengths)),
                inside_cells=int((subset.state == "inside").sum()), unresolved_cells=len(unknown),
                unresolved_lanes=int(unknown.lane.nunique()))
            for key, value in metrics.items():
                actual = method.get(key)
                if isinstance(actual, (bool, np.bool_)) or not isinstance(actual, (int, np.integer)) or actual != value:
                    raise ValueError("xtru count evidence disagrees with retained cells")
            reasons = method.get("unresolved_reasons")
            expected = {str(k): int(v) for k, v in unknown.reason.value_counts().items()}
            if not isinstance(reasons, dict) or reasons != expected or any(
                    isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in reasons.values()):
                raise ValueError("xtru reason/count evidence disagrees with retained cells")
            authenticated.add(report["source"])
        return authenticated

    def _control(self, hits, lengths, cells, reports, records, **kwargs):
        return self.strict.compose(hits, lengths, cells, reports, records, **kwargs)

    def _authenticated(self, cells, reports, lengths):
        return self._xtru_eligibility(cells, reports, lengths)

    def compose(self, hits, lengths, cells, reports, records, *, endpoint_budget_mm, min_cell_mm,
                max_component_mm, max_ray_unknown_mm, max_pairs=2000000):
        """Emit both arms atomically with explicit cross-family union budgets.

        Full-footprint parent/ownership, hierarchy, overlap and raw occupancy
        checks are unchanged. No uncertainty cell becomes certified evidence.
        """
        reviewed, strict = self.reviewed, self.strict
        boundarypolicy = reviewed.boundarypolicy
        component_budget = reviewed._budget(max_component_mm, "max_component_mm")
        ray_budget = reviewed._budget(max_ray_unknown_mm, "max_ray_unknown_mm")
        cell_width = reviewed._budget(min_cell_mm, "min_cell_mm")
        if cell_width <= 0:
            raise ValueError("min_cell_mm must be positive")
        control = self._control(hits, lengths, cells, reports, records,
                                endpoint_budget_mm=endpoint_budget_mm, max_pairs=max_pairs)
        lengths = np.asarray(lengths, float)
        authenticated = self._authenticated(cells, reports, lengths)
        adjusted = control["adjusted_hits"]
        unknown, components, uncertainty = reviewed.unknown_components(cells, lengths)
        issues = [control["refusals"][control["refusals"].reason != "unresolved-solid-boundary"].copy()]
        # Only authenticated xtru certificates bypass the old blanket boundary refusal.
        issues.extend(reviewed._eligibility(cells,
            [r for r in reports if r["source"] not in authenticated], lengths, cell_width))

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
        pairs, ownership, mismatch, ambiguous = reviewed._parent_checks(cells, stored, adjusted, bad_stored, max_pairs)
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
            terms[arm] = reviewed._terms(selected, arm)
            trial = pd.concat([base, terms[arm]], ignore_index=True)
            if len(strict._bad_occupancy(trial)):
                raise ValueError("envelope composition violated prechecked strict occupancy; no pair returned")
            products[arm] = boundarypolicy.partition(trial, accepted_lengths)
        selected_components = components[components.lane.isin(np.flatnonzero(accepted))]
        atoms = reviewed._atomic_pair(products["inner"], products["outer"], selected_components)
        bounds, aggregate = reviewed._material_bounds(atoms, accepted, lengths)
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
            authenticated_xtru_placements=len(authenticated), admitted_for_transport=False,
            guard_contract="engineering float64 torus min-width and authenticated xtru plane-offset envelopes; not machine-verified intervals or G4 truth")
        return dict(inner_intervals=products["inner"], outer_intervals=products["outer"], atoms=atoms,
            material_bounds=bounds, nonvacuum_bounds=aggregate, decisions=decisions, refusals=refusals,
            unknown_cells=unknown, unknown_components=components, parent_comparisons=pairs, ownership_comparisons=ownership,
            replacement_terms=pd.concat([terms["inner"], terms["outer"]], ignore_index=True),
            control=control, summary=summary)
