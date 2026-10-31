"""Ray-bound evaluated box misses and guarded leaf CSG, separate from reviewed controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import csgenvelope, materialpath
from .mixedenvelope import MixedEnvelope

POLICY = "support-navigation-ray-bound-miss-planar-csg-v1"
MISS = "stored-placement-box-all-ray-miss-v1"


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _array_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value, dtype="<f8")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _cells_digest(cells):
    h = hashlib.sha256(str([(c, str(cells[c].dtype)) for c in cells]).encode())
    h.update(pd.util.hash_pandas_object(cells, index=True).to_numpy().tobytes())
    return h.hexdigest()


def bounds_exposure(origin, direction, lengths, bounds):
    """Outward padded slabs, retaining tangencies as possible intersections."""
    bounds = np.asarray(bounds, float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] < bounds[0]).any():
        raise ValueError("finite ordered exclusion bounds required")
    scale = max(1., float(np.abs(origin).max(initial=0.)), float(np.abs(bounds).max(initial=0.)),
                float(lengths.max(initial=0.)))
    guard = 1024 * np.finfo(float).eps * scale
    lo, hi = np.nextafter(bounds[0] - guard, -np.inf), np.nextafter(bounds[1] + guard, np.inf)
    parallel = direction == 0
    safe = np.where(parallel, 1., direction)
    with np.errstate(over="ignore", invalid="ignore"):
        left, right = (lo - origin) / safe, (hi - origin) / safe
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("exclusion arithmetic overflow")
    entry = np.maximum(0., np.where(parallel, -np.inf, np.nextafter(np.minimum(left, right), -np.inf)).max(axis=1))
    exit_ = np.minimum(lengths, np.where(parallel, np.inf, np.nextafter(np.maximum(left, right), np.inf)).min(axis=1))
    outside = (parallel & ((origin < lo) | (origin > hi))).any(axis=1)
    return ~outside & (exit_ >= entry) & (lengths > 0), guard


def _box_bounds(solid, placement):
    csgenvelope.supported(solid, {})
    wide = np.longdouble
    if np.finfo(wide).nmant <= np.finfo(float).nmant:
        raise ValueError("box exclusion requires wider spatial arithmetic")
    r, t = np.asarray(placement["rotation"], wide), np.asarray(placement["position"], wide)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("finite stored box transform required")
    half = np.array([float(solid.get(k)) for k in "xyz"], dtype=wide) * wide(.5)
    extent = np.abs(r) @ half
    scale = max(wide(1), np.abs(t).max(), extent.max())
    guard = wide(1024) * np.finfo(float).eps * scale * (len(placement["ancestors"]) + 1)
    bounds = np.stack([csgenvelope._project(t - extent - guard, False), csgenvelope._project(t + extent + guard, True)])
    if not np.isfinite(bounds).all():
        raise ValueError("finite stored box bounds required")
    return bounds


class CertifiedOmission:
    """Bind every generated certificate to immutable GDML, full census and actual query rays."""

    def __init__(self, base, reviewed):
        self.base, self.reviewed = base, reviewed
        self.latest = None

    def probe(self, gdml, records, origin, direction, lengths, *, min_cell_mm=1e-7, max_cells=1000000):
        self.latest = None
        origin, direction, lengths = materialpath._rays(origin, direction, lengths)
        if isinstance(max_cells, bool) or not isinstance(max_cells, (int, np.integer)) or max_cells <= 0:
            raise ValueError("positive integer certificate cell cap required")
        gdml_hash = hashlib.sha256(Path(gdml).read_bytes()).hexdigest()
        binding = dict(gdml_sha256=gdml_hash, records_sha256=_json_digest(records),
                       rays_sha256=_array_digest(origin, direction, lengths))
        cells, original = self.base.probe(gdml, records, origin, direction, lengths,
                                         min_cell_mm=min_cell_mm, max_cells=max_cells)
        reports = deepcopy(original)
        original_cells = cells.copy()
        wanted = [r["source"] for r in reports if not r["resolved"]]
        index = self.reviewed.PlacementIndex(gdml, wanted)
        lookup = {r["placement_path"]: r for r in records if r["classification"] == "unsupported"}
        additions, misses, csg = [], set(), set()
        for report in reports:
            if report["resolved"]:
                continue
            path = report["source"]
            placement, record = index.placements[path], lookup[path]
            solid = index.solids[placement["solid"]]
            if (placement["material"] != record["material"] or solid.tag != record["solid_tag"] or
                    placement["solid"] != record["solid"] or placement["is_leaf"] != record["is_leaf"]):
                raise ValueError("certificate placement disagrees with full census")
            if placement["assumptions"] or record.get("placement_assumptions"):
                continue
            try:
                if solid.tag == "box":
                    bounds = _box_bounds(solid, placement)
                    possible, guard = bounds_exposure(origin, direction, lengths, bounds)
                    if possible.any():
                        continue
                    method = dict(certificate_scheme=MISS, method="outward stored-transform box AABB all-ray exclusion",
                        binding=binding, bounds_mm=bounds.tolist(), guard_mm=guard, capped=False, lanes=len(lengths),
                        positive_length_lanes=int(np.count_nonzero(lengths)), possible_lanes=0, inside_cells=0,
                        unresolved_cells=0, unresolved_lanes=0, unresolved_reasons={})
                    report.update(evaluation_status="evaluated-miss", miss_certificate=method)
                    misses.add(path)
                    continue
                if solid.tag not in ("subtraction", "union", "intersection") or not placement["is_leaf"]:
                    continue
                csgenvelope.construction_bound(solid, index.solids, max_work=2000000)
                self._validate_xtru(solid, index.solids)
                local_o, local_d = index.local_rays(placement, origin, direction)
                guard = (len(placement["ancestors"]) + 1) * self.reviewed._roundoff(
                    origin, origin + direction * lengths[:, None], placement["position"],
                    np.array([a["position"] for a in placement["ancestors"]]))
                spans, method = csgenvelope.probe(solid, index.solids, local_o, local_d, lengths,
                                                 coordinate_guard_mm=guard, max_pairs=2000000)
                if len(spans) > max_cells:
                    raise ValueError("CSG retained cell cap")
                spans, unresolved = self.reviewed.hierarchy_check(index, placement, spans, origin, direction)
                spans["source"], spans["material"], spans["scope"] = path, placement["material"], record["scope"]
                inside = spans.state == "inside"
                known = inside & spans.ancestors_verified & spans.leaf_verified
                report.update(resolved=True, reason="guarded planar CSG evaluated", evaluation_status="evaluated-leaf-csg",
                    method=dict(method, binding=binding), solid_lower_mm=self.reviewed.length_bound(spans[inside], upper=False),
                    solid_upper_mm=self.reviewed.length_bound(spans, upper=True),
                    hierarchy_verified_lower_mm=self.reviewed.length_bound(spans[known], upper=False),
                    hit_lanes=int(spans.loc[inside, "lane"].nunique()), possible_lanes=int(spans.lane.nunique()),
                    unresolved_ancestors=unresolved)
                additions.append(spans)
                csg.add(path)
            except ValueError as error:
                report["certificate_refusal"] = str(error)
        if additions:
            cells = pd.concat(([cells] if len(cells) else []) + additions, ignore_index=True)
        if hashlib.sha256(Path(gdml).read_bytes()).hexdigest() != gdml_hash or _json_digest(records) != binding["records_sha256"]:
            raise ValueError("certificate geometry/census changed during query")
        self.latest = dict(binding=binding, lengths_sha256=_array_digest(lengths), cells_sha256=_cells_digest(cells),
            reports_sha256=_json_digest(reports), records_sha256=_json_digest(records), misses=misses, csg=csg,
            original_reports=deepcopy(original), original_cells=original_cells)
        return cells, reports

    def _validate_xtru(self, solid, solids):
        if solid.tag == "xtru":
            self.reviewed._xtru_pieces(solid)
        elif solid.tag in ("subtraction", "union", "intersection"):
            for tag in ("first", "second"):
                self._validate_xtru(solids[solid.find(tag).get("ref")], solids)

    def validate(self, cells, reports, records, lengths):
        saved = self.latest
        if saved is None or (_cells_digest(cells), _json_digest(reports), _json_digest(records), _array_digest(lengths)) != (
                saved["cells_sha256"], saved["reports_sha256"], saved["records_sha256"], saved["lengths_sha256"]):
            raise ValueError("unbound, mutated or stale certificate evidence")
        return saved


class InclusiveEnvelope(MixedEnvelope):
    """Explicit certificate-conditioned gate; the full original strict result remains retained."""

    def __init__(self, reviewedEnvelope, reviewedStrict, probe):
        super().__init__(reviewedEnvelope, reviewedStrict)
        self.probe = probe

    def _control(self, hits, lengths, cells, reports, records, **kwargs):
        saved = self.probe.validate(cells, reports, records, lengths)
        result = self.strict.compose(hits, lengths, cells, reports, records, **kwargs)
        self.original_control = result
        removable = {"unevaluated-placement:" + p for p in saved["misses"]}
        refusals = result["refusals"][~result["refusals"].reason.isin(removable)].copy()
        accepted = ~np.isin(np.arange(len(lengths)), refusals.lane.to_numpy(np.int64))
        summary = dict(result["summary"], global_refusals=[r for r in result["summary"]["global_refusals"] if r not in removable],
            resolved_rays=int(accepted.sum()), refused_rays=int((~accepted).sum()),
            all_rays_resolved=bool(accepted.all()), mode=POLICY + ":miss-conditioned-strict-refusals",
            products_scope="original strict products unchanged; only conditional refusal view is used by pair compositor")
        return dict(result, refusals=refusals, summary=summary,
                    decisions=result["decisions"].assign(candidate_resolved=accepted))

    def _authenticated(self, cells, reports, lengths):
        saved = self.probe.latest
        authenticated = super()._authenticated(cells, reports, lengths)
        for report in reports:
            if report["source"] not in saved["csg"]:
                continue
            subset = cells[cells.source == report["source"]]
            method = report["method"]
            if (method["certificate_scheme"] != csgenvelope.POLICY or method["capped"] is not False or
                    not report["leaf"] or not subset.reason.isin(["certificate", "csg_plane_offset"]).all()):
                raise ValueError("unrecognized planar CSG certificate")
            authenticated.add(report["source"])
        return authenticated

    def compose(self, hits, lengths, cells, reports, records, **kwargs):
        saved = self.probe.validate(cells, reports, records, lengths)
        result = super().compose(hits, lengths, cells, reports, records, **kwargs)
        result["original_strict_control"] = self.original_control
        result["original_probe_strict_control"] = self.strict.compose(hits, lengths, saved["original_cells"],
            saved["original_reports"], records, endpoint_budget_mm=kwargs["endpoint_budget_mm"],
            max_pairs=kwargs.get("max_pairs", 2000000))
        result["summary"].update(mode=POLICY, unresolved_policy="ray-bound-box-miss-and-guarded-leaf-CSG-with-original-overlap-refusals",
            authenticated_xtru_placements=len({r["source"] for r in reports
            if r["solid_tag"] == "xtru" and r["resolved"]}), evaluated_box_misses=len(saved["misses"]),
            authenticated_csg_placements=len(saved["csg"]), certificate_binding=saved["binding"],
            guard_contract="engineering stored-model torus/xtru/planar-CSG plus ray-bound box misses; not G4/GDML truth")
        return result
