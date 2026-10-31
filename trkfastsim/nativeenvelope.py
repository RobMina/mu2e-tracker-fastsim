"""Separate native planar model with joint directed root projections, never old-model certification."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import exactplanar, inclusiveenvelope, nativeplanar
from .inclusiveenvelope import _array_digest, _cells_digest, _json_digest

POLICY = "native-planar-joint-root-projection-sensitivity-v1"
SCHEME = "native-planar-exact-topology-directed-roots-v1"


class _Evidence:
    def __init__(self, cells, reports, records, lengths, saved):
        self.latest = dict(saved, cells_sha256=_cells_digest(cells), reports_sha256=_json_digest(reports),
            records_sha256=_json_digest(records), lengths_sha256=_array_digest(lengths))

    def validate(self, cells, reports, records, lengths):
        saved = self.latest
        if (_cells_digest(cells), _json_digest(reports), _json_digest(records), _array_digest(lengths)) != (
                saved["cells_sha256"], saved["reports_sha256"], saved["records_sha256"], saved["lengths_sha256"]):
            raise ValueError("unbound native projection evidence")
        return saved


class _Arm(inclusiveenvelope.InclusiveEnvelope):
    def _authenticated(self, cells, reports, lengths):
        native = self.probe.latest["native"]
        old = [r for r in reports if r["source"] not in native]
        authenticated = self._xtru_eligibility(cells, old, lengths)
        for report in reports:
            if report["source"] in native:
                if report["method"]["certificate_scheme"] != SCHEME or report["method"]["capped"] is not False:
                    raise ValueError("native root certificate metadata")
                authenticated.add(report["source"])
        return authenticated


class NativeOmission:
    """Evaluate every eligible planar leaf on every ray; retain complete old-model controls."""

    def __init__(self, base, reviewed):
        self.base, self.reviewed, self.latest = base, reviewed, None

    def probe(self, gdml, records, origin, direction, lengths, *, min_cell_mm=1e-7, max_cells=1000000):
        self.latest = None
        original_cells, original_reports = self.base.probe(gdml, records, origin, direction, lengths,
                                                          min_cell_mm=min_cell_mm, max_cells=max_cells)
        saved = self.base.validate(original_cells, original_reports, records, lengths)
        paths = [r["source"] for r in original_reports if r["leaf"] and r["resolved"] and
                 r["solid_tag"] in ("xtru", "subtraction", "union", "intersection")]
        if not paths:
            raise ValueError("native model applicability requires at least one eligible planar leaf; no implicit fallback")
        index = self.reviewed.PlacementIndex(gdml, paths)
        literal = nativeplanar.LiteralIndex(gdml, paths)
        reports, projected = {}, {"lower": [], "upper": []}
        footprint, unknown, exact, root_rows, methods = [], [], [], [], []
        for path in paths:
            placement = index.placements[path]
            solid = index.solids[placement["solid"]]
            rows, faults, method = nativeplanar.intervals(solid, index.solids, literal.placements[path], origin, direction, lengths,
                                                        reviewed=self.reviewed, max_work=2000000)
            if len(faults) or method["coplanar_lanes"]:
                raise ValueError("native topology has invalid occupancy or positive-length boundary contact")
            if len(rows) > max_cells:
                raise ValueError("native retained cell cap")
            methods.append(dict(source=path, method=method))
            q0, q1 = rows.q0.to_numpy(object), rows.q1.to_numpy(object)
            lo0, hi0 = exactplanar.export(q0, False), exactplanar.export(q0, True)
            lo1, hi1 = exactplanar.export(q1, False), exactplanar.export(q1, True)
            if (hi0 >= lo1).any():
                raise ValueError("distinct exact native roots not separated on binary64 lattice")
            guard = float(max((hi0 - lo0).max(initial=0.), (hi1 - lo1).max(initial=0.)))
            if guard > 1e-10:
                raise ValueError("directed root enclosure exceeds original 1e-10mm endpoint budget")
            context = next(r for r in original_reports if r["source"] == path)
            exact.append(rows.assign(source=path, material=context["material"]))
            root_rows.append(exactplanar.serialize_roots(rows.assign(source=path)))
            possible = pd.DataFrame(dict(lane=rows.lane.to_numpy(np.int64), s0_mm=lo0, s1_mm=hi1, state="inside"))
            possible, ancestors = self.reviewed.hierarchy_check(index, placement, possible, origin, direction)
            possible["source"], possible["material"], possible["scope"] = path, context["material"], context["scope"]
            footprint.append(possible)
            bracket = pd.concat([possible.assign(s0_mm=lo0, s1_mm=hi0), possible.assign(s0_mm=lo1, s1_mm=hi1)], ignore_index=True)
            bracket = bracket[bracket.s1_mm > bracket.s0_mm].assign(state="unresolved", reason="directed_native_root")
            unknown.append(bracket)
            for arm, a, b in (("lower", lo0, lo1), ("upper", hi0, hi1)):
                spans = possible.assign(s0_mm=a, s1_mm=b, state="inside", reason="native_projected_root")
                inside = spans.state == "inside"
                known = inside & spans.ancestors_verified & spans.leaf_verified
                report = dict(context, resolved=True, evaluation_status="native-model-projection", method=dict(method,
                    certificate_scheme=SCHEME, projection=arm, guard_mm=guard, endpoint_budget_mm=1e-10,
                    inside_cells=len(spans), unresolved_cells=0, unresolved_lanes=0, unresolved_reasons={}),
                    solid_lower_mm=self.reviewed.length_bound(spans, upper=False), solid_upper_mm=self.reviewed.length_bound(spans, upper=True),
                    hierarchy_verified_lower_mm=self.reviewed.length_bound(spans[known], upper=False),
                    hit_lanes=int(spans.lane.nunique()), possible_lanes=int(spans.lane.nunique()), unresolved_ancestors=ancestors)
                projected[arm].append(spans)
                reports[(arm, path)] = report
        exact = pd.concat(exact, ignore_index=True) if exact else pd.DataFrame(columns=["lane", "q0", "q1", "source", "material"])
        sizes = exact.groupby("lane").size()
        if int((sizes * sizes).sum()) > 2000000:
            raise ValueError("native exact leaf comparison cap")
        pairs = exact.merge(exact, on="lane", suffixes=("_a", "_b"))
        positive = (pairs.source_a < pairs.source_b) & (np.minimum(pairs.q1_a, pairs.q1_b) > np.maximum(pairs.q0_a, pairs.q0_b))
        if positive.any():
            raise ValueError("positive exact overlap between native physical leaves; no projection")
        cells_by_arm, reports_by_arm, evidences = {}, {}, {}
        remaining = original_cells[~original_cells.source.isin(paths)]
        for arm in ("lower", "upper"):
            cells = pd.concat(([remaining] if len(remaining) else []) + projected[arm], ignore_index=True)
            r = [reports[(arm, row["source"])] if row["source"] in paths else deepcopy(row) for row in original_reports]
            state = dict(saved, native=set(paths), csg=set(), original_cells=original_cells.copy(), original_reports=deepcopy(original_reports))
            cells_by_arm[arm], reports_by_arm[arm] = cells, r
            evidences[arm] = _Evidence(cells, r, records, lengths, state)
        if hashlib.sha256(Path(gdml).read_bytes()).hexdigest() != saved["binding"]["gdml_sha256"]:
            raise ValueError("native geometry changed during query")
        self.latest = dict(cells=cells_by_arm, reports=reports_by_arm, evidence=evidences,
            original_cells=original_cells, original_reports=original_reports, records=deepcopy(records),
            footprint=pd.concat(footprint, ignore_index=True), root_unknown=pd.concat(unknown, ignore_index=True),
            roots=pd.concat(root_rows, ignore_index=True), methods=methods, binding=saved["binding"],
            original_model_binding=saved["binding"], native_model=nativeplanar.POLICY,
            footprint_sha256=_cells_digest(pd.concat(footprint, ignore_index=True)),
            roots_sha256=_cells_digest(pd.concat(root_rows, ignore_index=True)),
            unknown_sha256=_cells_digest(pd.concat(unknown, ignore_index=True)), methods_sha256=_json_digest(methods))
        return cells_by_arm["lower"], reports_by_arm["lower"]


class NativeEnvelope:
    """Two correlated root-projection sensitivity paths, NOT inner/outer material enclosures."""

    def __init__(self, reviewedEnvelope, reviewedStrict, probe):
        self.reviewed, self.strict, self.probe = reviewedEnvelope, reviewedStrict, probe

    def compose(self, hits, lengths, cells, reports, records, *, endpoint_budget_mm, min_cell_mm,
                max_component_mm, max_ray_unknown_mm, max_pairs=2000000):
        saved = self.probe.latest
        if saved is None:
            raise ValueError("native projection query absent")
        saved["evidence"]["lower"].validate(cells, reports, records, lengths)
        if any(_cells_digest(saved[key]) != saved[hash_key] for key, hash_key in (
                ("footprint", "footprint_sha256"), ("roots", "roots_sha256"), ("root_unknown", "unknown_sha256"))):
            raise ValueError("mutated native full-footprint/root evidence")
        if _json_digest(saved["methods"]) != saved["methods_sha256"]:
            raise ValueError("mutated native source selection/method evidence")
        kwargs = dict(endpoint_budget_mm=endpoint_budget_mm, min_cell_mm=min_cell_mm,
            max_component_mm=max_component_mm, max_ray_unknown_mm=max_ray_unknown_mm, max_pairs=max_pairs)
        old_gate = inclusiveenvelope.InclusiveEnvelope(self.reviewed, self.strict, self.probe.base).compose(
            hits, lengths, saved["original_cells"], saved["original_reports"], records, **kwargs)
        results = {arm: _Arm(self.reviewed, self.strict, saved["evidence"][arm]).compose(hits, lengths,
            saved["cells"][arm], saved["reports"][arm], records, **kwargs) for arm in ("lower", "upper")}
        boundary = self.reviewed.boundarypolicy
        adjusted = results["lower"]["control"]["adjusted_hits"]
        bad = self.strict._bad_occupancy(adjusted)
        safe_lengths = np.asarray(lengths, float).copy()
        safe_lengths[bad] = 0.
        stored = boundary.partition(adjusted[~adjusted.lane.isin(bad)], safe_lengths)
        pairs, ownership, mismatch, ambiguous = self.reviewed._parent_checks(saved["footprint"], stored, adjusted, bad, max_pairs)
        from .placementownership import parent_owners
        physical = parent_owners(saved["footprint"], adjusted, max_pairs=max_pairs)
        unknown_input = pd.concat([saved["original_cells"][~saved["original_cells"].source.isin({r["source"] for r in saved["methods"]})],
                                   saved["root_unknown"]], ignore_index=True)
        unknown, components, uncertainty = self.reviewed.unknown_components(unknown_input, lengths)
        issues = [results[a]["refusals"] for a in ("lower", "upper")]
        for lanes, reason in ((mismatch, "native-full-footprint-parent-mismatch"), (ambiguous, "native-full-footprint-ambiguous-owner"),
                (saved["footprint"].loc[~saved["footprint"].ancestors_verified, "lane"].unique(), "native-unverified-footprint-ancestor"),
                (components.loc[components.upper_mm > max_component_mm, "lane"], "native-unknown-component-budget-exceeded"),
                (uncertainty.loc[uncertainty.unknown_upper_mm > max_ray_unknown_mm, "lane"], "native-unknown-ray-budget-exceeded")):
            if len(lanes):
                issues.append(pd.DataFrame(dict(lane=np.unique(lanes).astype(np.int64), reason=reason)))
        refusals = pd.concat(issues, ignore_index=True).drop_duplicates().sort_values(["lane", "reason"])
        accepted = ~np.isin(np.arange(len(lengths)), refusals.lane.to_numpy(np.int64))
        inner = results["lower"]["inner_intervals"]
        outer = results["upper"]["outer_intervals"]
        inner, outer = inner[inner.lane.isin(np.flatnonzero(accepted))], outer[outer.lane.isin(np.flatnonzero(accepted))]
        atoms = self.reviewed._atomic_pair(inner, outer, components[components.lane.isin(np.flatnonzero(accepted))])
        bounds, aggregate = self.reviewed._material_bounds(atoms, accepted, np.asarray(lengths, float))
        summary = dict(mode=POLICY, geometry_model=nativeplanar.POLICY, rays=len(lengths), bounded_pair_rays=int(accepted.sum()),
            refused_rays=int((~accepted).sum()), all_pairs_admitted=bool(accepted.all()), native_placements=len(saved["methods"]),
            native_positive_exact_leaf_overlaps=0, native_root_rows=len(saved["roots"]), unknown_cells=len(unknown),
            unknown_components=len(components), max_component_upper_mm=float(components.upper_mm.max()) if len(components) else 0.,
            max_ray_unknown_upper_mm=float(uncertainty.unknown_upper_mm.max()) if len(uncertainty) else 0.,
            max_component_mm=max_component_mm, max_ray_unknown_mm=max_ray_unknown_mm, endpoint_budget_mm=1e-10, native_root_projection_budget_mm=1e-10,
            stored_composition_endpoint_budget_mm=endpoint_budget_mm,
            parent_comparisons=len(pairs), unexpected_parent_comparisons=int((~pairs.expected_parent).sum()),
            ownership_comparisons=len(ownership), ambiguous_ownership_rays=len(ambiguous), certificate_binding=saved["binding"],
            admitted_for_transport=False, scope="literal native model and jointly directed roots; arms are sensitivities, NOT geometric/energy bounds or G4 truth")
        return dict(inner_intervals=inner, outer_intervals=outer, atoms=atoms, material_bounds=bounds, nonvacuum_bounds=aggregate,
            decisions=uncertainty.assign(bounded_pair_admitted=accepted, admitted_for_transport=False), refusals=refusals,
            unknown_cells=unknown, unknown_components=components, parent_comparisons=pairs, ownership_comparisons=ownership,
            physical_parent_comparisons=physical, native_root_endpoints=saved["roots"].copy(deep=True), native_possible_footprint=saved["footprint"].copy(deep=True),
            original_inclusive_control=old_gate, lower_projection_control=results["lower"], upper_projection_control=results["upper"], summary=summary)
