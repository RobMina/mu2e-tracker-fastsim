"""Wider torus spatial arithmetic on the unchanged stored-binary64 placement model."""
import numpy as np
import pandas as pd

from . import materialpath, widetorus

POLICY = "stored-placement-wide-torus-v1"


class WideOmission:
    """Use unchanged xtru/hierarchy code; only torus local arithmetic is wider.

    Already accumulated binary64 placement R,t define this numerical model.
    Their discrepancy from unrounded GDML transforms is not eliminated here.
    The reviewed binary64 probe remains a distinct unchanged refusal control.
    """

    def __init__(self, reviewed, *, authenticate_xtru=False):
        self.reviewed = reviewed
        self.policy = POLICY
        self.authenticate_xtru = authenticate_xtru

    def probe(self, gdml, records, origin, direction, lengths, *, min_cell_mm=1e-7, max_cells=1000000):
        widetorus.require_wider()
        origin, direction, lengths = materialpath._rays(origin, direction, lengths)
        records = [r for r in records if r["classification"] == "unsupported"]
        tori = [r for r in records if r["solid_tag"] == "torus"]
        other = [r for r in records if r["solid_tag"] != "torus"]
        cells, reports = self.reviewed.probe(gdml, other, origin, direction, lengths,
            min_cell_mm=min_cell_mm, max_cells=max_cells)
        reports = list(reports)
        if self.authenticate_xtru:
            cells = cells.copy()
            if "reason" not in cells:
                cells["reason"] = pd.Series(dtype=str)
            for report in reports:
                if report["solid_tag"] != "xtru" or not report["resolved"]:
                    continue
                selected = cells.source == report["source"]
                if cells.loc[selected, "reason"].notna().any() or any(key in report["method"] for key in (
                        "certificate_scheme", "lanes", "positive_length_lanes", "inside_cells", "unresolved_cells", "unresolved_lanes", "unresolved_reasons")):
                    raise ValueError("xtru adapter only augments original reason-free reviewed probe evidence")
                cells.loc[selected, "reason"] = np.where(cells.loc[selected, "state"] == "inside", "certificate", "plane_offset")
                subset = cells.loc[selected]
                unknown = subset[subset.state == "unresolved"]
                report["method"] = dict(report["method"], certificate_scheme="xtru-plane-offset-v1", lanes=len(lengths),
                    positive_length_lanes=int(np.count_nonzero(lengths)), inside_cells=int((subset.state == "inside").sum()),
                    unresolved_cells=len(unknown), unresolved_lanes=int(unknown.lane.nunique()),
                    unresolved_reasons={str(k): int(v) for k, v in unknown.reason.value_counts().items()})
        intervals = [cells]
        index = self.reviewed.PlacementIndex(gdml, [r["placement_path"] for r in tori])
        wide = np.longdouble
        eps = np.finfo(wide).eps
        for record in tori:
            path = record["placement_path"]
            placement = index.placements[path]
            solid = index.solids[placement["solid"]]
            row = dict(source=path, material=placement["material"], solid_tag=solid.tag, scope=record["scope"],
                leaf=placement["is_leaf"], parent_material=placement["ancestors"][-1]["material"])
            if placement["assumptions"] or not placement["is_leaf"]:
                reports.append(dict(row, resolved=False, reason="unresolved transform or non-leaf material hierarchy"))
                continue
            try:
                if solid.get("lunit", "mm") != "mm" or solid.get("aunit", "deg") not in ("deg", "rad"):
                    raise ValueError("unsupported torus units")
                rotation, position = np.asarray(placement["rotation"], wide), np.asarray(placement["position"], wide)
                wo, wd = np.asarray(origin, wide), np.asarray(direction, wide)
                local_o, local_d = (wo - position) @ rotation, wd @ rotation
                scale = max([wide(1)] + [np.max(np.abs(value)) for value in (wo, wo + wd * lengths[:, None], position) if value.size])
                transform_guard = np.nextafter(wide(128) * eps * scale * wide(4), wide(np.inf))
                angle_scale = np.pi / 180 if solid.get("aunit", "deg") == "deg" else 1.
                result = widetorus.torus_intervals(local_o, local_d, lengths, rtor=float(solid.get("rtor")),
                    rmin=float(solid.get("rmin", 0)), rmax=float(solid.get("rmax")),
                    startphi=float(solid.get("startphi", 0)) * angle_scale,
                    deltaphi=float(solid.get("deltaphi")) * angle_scale, min_width_mm=min_cell_mm,
                    max_nodes=max_cells, coordinate_guard_mm=transform_guard)
                spans = result.intervals.rename(columns={"status": "state"})
                extra = dict(result.counts, arithmetic_policy=POLICY, transform_guard_mm=float(transform_guard),
                    stored_rotation_model="exact promoted already-accumulated binary64 R,t; no reorthogonalization")
            except ValueError as error:
                reports.append(dict(row, resolved=False, reason=str(error)))
                continue
            spans, unresolved = self.reviewed.hierarchy_check(index, placement, spans, origin, direction)
            spans["source"], spans["material"], spans["scope"] = path, placement["material"], record["scope"]
            inside = spans.state == "inside"
            known = inside & spans.ancestors_verified & spans.leaf_verified
            row.update(resolved=True, method=extra, solid_lower_mm=self.reviewed.length_bound(spans[inside], upper=False),
                solid_upper_mm=self.reviewed.length_bound(spans, upper=True),
                hierarchy_verified_lower_mm=self.reviewed.length_bound(spans[known], upper=False),
                hit_lanes=int(spans.loc[inside, "lane"].nunique()), possible_lanes=int(spans.lane.nunique()),
                unresolved_ancestors=unresolved)
            reports.append(row)
            intervals.append(spans)
        return pd.concat(intervals, ignore_index=True), reports
