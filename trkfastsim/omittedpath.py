"""Bounded leaf-solid omission probes; never interpret a bounding box as material or alter loaders."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from mu2efastsim import world
from . import convexpath, materialpath
from .navigationcensus import _placement_notes, _VACUUM


class PlacementIndex:
    """Verified inline GDML placements indexed by census physical path, with explicit ancestry."""

    def __init__(self, path, wanted):
        root = ET.parse(path).getroot()
        self.solids = {s.get("name"): s for s in root.find("solids")}
        volumes = {v.get("name"): v for v in root.find("structure").findall("volume")}
        self.placements = {}
        wanted = set(wanted)

        def node(lv, rotation, position, physical, notes):
            v = volumes[lv]
            return dict(path=physical, logical=lv, solid=v.find("solidref").get("ref"),
                        material=world._strip(v.find("materialref").get("ref")),
                        rotation=rotation, position=position, assumptions=notes,
                        is_leaf=not any(v.find(tag) is not None for tag in ("physvol", "replicavol", "divisionvol", "paramvol")))

        def walk(lv, rotation, position, physical, ancestors, notes, lineage):
            if lv in lineage:
                raise ValueError("cyclic GDML placement")
            v = volumes[lv]
            if any(v.find(tag) is not None for tag in ("replicavol", "divisionvol", "paramvol")):
                raise ValueError("unsupported GDML placement construct")
            for index, pv in enumerate(v.findall("physvol")):
                child = pv.find("volumeref").get("ref")
                leaf = world._strip(child)
                childpath = physical + f"{pv.get('name', leaf)}[{index}]"
                if not any(path == childpath or path.startswith(childpath + "/") for path in wanted):
                    continue
                assumptions = notes + _placement_notes(pv)
                p = pv.find("position")
                local = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
                r, t = rotation @ world._rot(pv.find("rotation")), rotation @ local + position
                here = node(child, r, t, childpath, assumptions)
                if childpath in wanted:
                    self.placements[childpath] = dict(here, ancestors=ancestors)
                walk(child, r, t, childpath + "/", ancestors + [here], assumptions, lineage + (lv,))

        root_lv = root.find("setup/world").get("ref")
        top = node(root_lv, np.eye(3), np.zeros(3), "world", [])
        walk(root_lv, np.eye(3), np.zeros(3), "", [top], [], ())
        if set(self.placements) != wanted:
            raise ValueError("census placement paths do not match this GDML")

    def local_rays(self, placement, origin, direction):
        if placement["assumptions"]:
            raise ValueError(f"unresolved placement assumptions: {placement['assumptions']}")
        r, t = placement["rotation"], placement["position"]
        return (origin - t) @ r, direction @ r


def _roundoff(*arrays):
    scale = max([1.] + [float(np.max(np.abs(a))) for a in arrays if np.size(a)])
    return 128 * np.finfo(float).eps * scale


def contains_segments(solid, solids, a, b, coordinate_guard_mm=0.):
    """Conservative whole-segment containment for verified box/full-tube/stair-polycone/Boolean parents.

    False means unverified, not outside. Full-circle tube radial extrema are analytic; union
    containment by either operand is sufficient but not necessary. No tapered-envelope substitution.
    """
    if solid.get("lunit", "mm") != "mm" or solid.get("aunit", "deg") not in ("deg", "rad"):
        return np.zeros(len(a), bool)
    tag = solid.tag
    if not np.isfinite(coordinate_guard_mm) or coordinate_guard_mm < 0:
        raise ValueError("finite nonnegative coordinate guard required")
    eps = coordinate_guard_mm + _roundoff(a, b)
    if tag == "box":
        half = .5 * np.array([float(solid.get(k)) for k in "xyz"])
        return (np.maximum(np.abs(a), np.abs(b)) < half - eps).all(axis=1)
    if tag in ("tube", "polycone"):
        scale = np.pi / 180 if solid.get("aunit", "deg") == "deg" else 1.
        if float(solid.get("deltaphi", 360 if scale != 1 else 2*np.pi)) * scale != 2*np.pi:
            return np.zeros(len(a), bool)
        if tag == "tube":
            hz = .5 * float(solid.get("z"))
            sections = [(-hz, hz, float(solid.get("rmin", 0)), float(solid.get("rmax")))]
        else:
            zp = [(float(z.get("z")), float(z.get("rmin", 0)), float(z.get("rmax")))
                  for z in solid.findall("zplane")]
            if len(zp) < 2 or not np.isfinite(zp).all() or np.any(np.diff(np.asarray(zp)[:, 0]) < 0):
                return np.zeros(len(a), bool)
            sections = [(z0, z1, ri, ro) for (z0, ri, ro), (z1, ai, ao) in zip(zp[:-1], zp[1:])
                        if z1 > z0 and ri == ai and ro == ao]
        v = b[:, :2] - a[:, :2]
        speed2 = np.einsum("ij,ij->i", v, v)
        t = np.clip(-np.einsum("ij,ij->i", a[:, :2], v) / np.where(speed2 == 0, 1., speed2), 0., 1.)
        rmin = np.linalg.norm(a[:, :2] + t[:, None] * v, axis=1)
        rmax = np.maximum(np.linalg.norm(a[:, :2], axis=1), np.linalg.norm(b[:, :2], axis=1))
        answer = np.zeros(len(a), bool)
        for z0, z1, ri, ro in sections:
            answer |= ((np.minimum(a[:, 2], b[:, 2]) > z0 + eps) & (np.maximum(a[:, 2], b[:, 2]) < z1 - eps)
                       & (rmax < ro - eps) & ((ri == 0) | (rmin > ri + eps)))
        return answer
    if tag in ("union", "intersection") and not _placement_notes(solid):
        p = solid.find("position")
        t = np.array([float(p.get(k, 0)) for k in "xyz"]) if p is not None else np.zeros(3)
        r = world._rot(solid.find("rotation"))
        first = contains_segments(solids[solid.find("first").get("ref")], solids, a, b, eps)
        second = contains_segments(solids[solid.find("second").get("ref")], solids, (a - t) @ r, (b - t) @ r,
                                   eps + _roundoff(a, b, t))
        return (first | second) if tag == "union" else (first & second)
    return np.zeros(len(a), bool)


def _xtru_pieces(solid):
    """Reuse core polygon decomposition only when it is an exact unscaled, unshifted prism."""
    if solid.tag != "xtru" or solid.get("lunit", "mm") != "mm":
        raise ValueError("only mm xtru prisms supported")
    sections = solid.findall("section")
    if len(sections) != 2 or any(float(s.get("scalingFactor", 1)) != 1 or
            float(s.get("xOffset", 0)) != 0 or float(s.get("yOffset", 0)) != 0 for s in sections):
        raise ValueError("only two unscaled/unshifted xtru sections supported")
    if float(sections[0].get("zPosition")) >= float(sections[1].get("zPosition")):
        raise ValueError("xtru sections must be ordered")
    vertices = np.array([[float(p.get(k)) for k in "xy"] for p in solid.findall("twoDimVertex")])
    if len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("invalid xtru polygon")
    edge = np.roll(vertices, -1, axis=0) - vertices
    if (np.linalg.norm(edge, axis=1) == 0).any():
        raise ValueError("xtru polygon has a zero-length edge")
    n = len(vertices)
    ii, jj = np.triu_indices(n, 1)
    adjacent = (jj == ii + 1) | ((ii == 0) & (jj == n-1))
    ii, jj = ii[~adjacent], jj[~adjacent]
    a, b = vertices[ii], vertices[(ii+1) % n]
    c, d = vertices[jj], vertices[(jj+1) % n]
    def cross(u, v):
        return u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]
    s0, s1 = cross(b-a, c-a), cross(b-a, d-a)
    t0, t1 = cross(d-c, a-c), cross(d-c, b-c)
    boxes_overlap = (np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                     np.minimum(np.maximum(a, b), np.maximum(c, d))).all(axis=1)
    if (((s0*s1 <= 0) & (t0*t1 <= 0)) & boxes_overlap).any():
        raise ValueError("self-intersecting xtru polygon")
    triangles = world._convex_pieces_2d(vertices)
    def area(v):
        return .5 * abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))
    target = area(vertices)
    if target <= 0 or len(triangles) != len(vertices) - 2 or not np.isclose(sum(area(t) for t in triangles), target,
            rtol=1e-12, atol=1e-10):
        raise ValueError("incomplete xtru triangulation")
    z0, z1 = (float(s.get("zPosition")) for s in sections)
    return [("solid", *world._prism(tri, z0, z1)) for tri in triangles]


def _prism_union(pieces, origin, direction, lengths):
    pieces = world.PieceSet(pieces)
    rows = []
    for lane, _, enter, leave in convexpath._hits(pieces, origin, direction, lengths):
        if len(lane):
            rows.append(pd.DataFrame(dict(lane=lane, s0_mm=enter, s1_mm=leave)))
    if not rows:
        return pd.DataFrame({k: np.empty(0, dtype=v) for k, v in
                            dict(lane="int64", s0_mm="float64", s1_mm="float64").items()})
    raw = pd.concat(rows, ignore_index=True).sort_values(["lane", "s0_mm", "s1_mm"], kind="stable").reset_index(drop=True)
    previous_max = raw.groupby("lane").s1_mm.cummax().groupby(raw.lane).shift(1)
    new = previous_max.isna() | (raw.s0_mm > previous_max)
    group = new.cumsum()
    return raw.groupby(group, sort=False).agg(lane=("lane", "first"), s0_mm=("s0_mm", "min"),
                                              s1_mm=("s1_mm", "max")).reset_index(drop=True)


def xtru_intervals(solid, origin, direction, lengths):
    """Union of the core's exact prism triangles, not union of distinct material volumes."""
    origin, direction, lengths = materialpath._rays(origin, direction, lengths)
    return _prism_union(_xtru_pieces(solid), origin, direction, lengths)


def xtru_probe(solid, origin, direction, lengths, *, coordinate_guard_mm=0.):
    """Numerically guarded inner/outer prism envelopes; return unknown boundary bands explicitly."""
    from .boundarypolicy import partition
    origin, direction, lengths = materialpath._rays(origin, direction, lengths)
    pieces = _xtru_pieces(solid)
    if not np.isfinite(coordinate_guard_mm) or coordinate_guard_mm < 0:
        raise ValueError("finite nonnegative coordinate guard required")
    eps = coordinate_guard_mm + _roundoff(origin, origin + direction * lengths[:, None],
                    np.concatenate([np.r_[lo, hi, planes[:, 3]] for _, planes, lo, hi in pieces]))
    inner, outer = [], []
    for mat, planes, lo, hi in pieces:
        expanded, contracted = planes.copy(), planes.copy()
        expanded[:, 3] += eps
        contracted[:, 3] -= eps
        normals = planes[:-2, :2]
        corners = np.stack([normals, np.roll(normals, -1, axis=0)], axis=1)
        det = np.linalg.det(corners)
        if (np.abs(det) < 1e-12).any():
            raise ValueError("ill-conditioned xtru triangle corner")
        displacement = np.linalg.solve(corners, np.full((len(corners), 2, 1), eps))[:, :, 0]
        margin = max(eps, float(np.max(np.abs(displacement)))) + eps
        outer.append((mat, expanded, lo - margin, hi + margin))
        inner.append((mat, contracted, lo - margin, hi + margin))
    inside = _prism_union(inner, origin, direction, lengths).assign(material="solid", sign=1)
    possible = _prism_union(outer, origin, direction, lengths).assign(material="solid", sign=1)
    unknown = partition(pd.concat([possible, inside.assign(sign=-1)], ignore_index=True), lengths)
    unknown = unknown[unknown.material != "vacuum"][["lane", "s0_mm", "s1_mm"]].assign(state="unresolved")
    spans = pd.concat([inside[["lane", "s0_mm", "s1_mm"]].assign(state="inside"), unknown], ignore_index=True)
    return spans.sort_values(["lane", "s0_mm"], kind="stable"), dict(
        method="core prism triangulation with explicit inner/outer plane-offset bands", guard_mm=eps,
        guard_contract="engineering float64 guard, not interval-arithmetic proof", capped=False)


def hierarchy_check(index, placement, spans, origin, direction):
    """Flag intervals contained in every represented ancestor; unresolved ancestry is never assumed valid."""
    result = spans.copy()
    lane = result.lane.to_numpy(dtype=np.int64)
    a = origin[lane] + direction[lane] * result.s0_mm.to_numpy(float)[:, None]
    b = origin[lane] + direction[lane] * result.s1_mm.to_numpy(float)[:, None]
    contained = np.ones(len(result), bool)
    unresolved = []
    for ancestor in placement["ancestors"]:
        if ancestor["assumptions"]:
            ok = np.zeros(len(result), bool)
        else:
            r, t = ancestor["rotation"], ancestor["position"]
            guard = (len(placement["ancestors"])+1) * _roundoff(a, b, t)
            ok = contains_segments(index.solids[ancestor["solid"]], index.solids, (a - t) @ r, (b - t) @ r, guard)
        if (~ok).any():
            unresolved.append(dict(path=ancestor["path"], solid=ancestor["solid"], intervals=int((~ok).sum())))
        contained &= ok
    result["ancestors_verified"] = contained
    result["leaf_verified"] = bool(placement["is_leaf"])
    parent = placement["ancestors"][-1]
    result["replaces_material"] = "vacuum" if _VACUUM.search(parent["material"]) else parent["material"]
    return result, unresolved


def length_bound(spans, *, upper):
    """Outward sum of positive cell widths, including endpoint-subtraction rounding."""
    if spans.empty:
        return 0.
    width = spans.s1_mm.to_numpy(float) - spans.s0_mm.to_numpy(float)
    terms = np.nextafter(width, np.inf if upper else 0.)
    result = math.fsum(terms)
    return float(np.nextafter(result, np.inf if upper else 0.))


def probe(gdml, records, origin, direction, lengths, *, min_cell_mm=1e-7, max_cells=1000000):
    """Probe every unsupported census placement on all rays; return bounds, signed-context rows and unresolved cases."""
    from .torusprobe import torus_intervals
    origin, direction, lengths = materialpath._rays(origin, direction, lengths)
    records = [r for r in records if r["classification"] == "unsupported"]
    index = PlacementIndex(gdml, [r["placement_path"] for r in records])
    intervals, report = [], []
    for record in records:
        path = record["placement_path"]
        placement = index.placements[path]
        solid = index.solids[placement["solid"]]
        row = dict(source=path, material=placement["material"], solid_tag=solid.tag, scope=record["scope"],
                   leaf=placement["is_leaf"], parent_material=placement["ancestors"][-1]["material"])
        if placement["assumptions"] or not placement["is_leaf"]:
            report.append(dict(row, resolved=False, reason="unresolved transform or non-leaf material hierarchy"))
            continue
        try:
            local_o, local_d = index.local_rays(placement, origin, direction)
            transform_guard = (len(placement["ancestors"])+1) * _roundoff(
                origin, origin + direction * lengths[:, None], placement["position"],
                np.array([a["position"] for a in placement["ancestors"]]))
            if solid.tag == "xtru":
                spans, extra = xtru_probe(solid, local_o, local_d, lengths, coordinate_guard_mm=transform_guard)
            elif solid.tag == "torus":
                if solid.get("lunit", "mm") != "mm" or solid.get("aunit", "deg") not in ("deg", "rad"):
                    raise ValueError("unsupported torus units")
                scale = np.pi / 180 if solid.get("aunit", "deg") == "deg" else 1.
                result = torus_intervals(local_o, local_d, lengths, rtor=float(solid.get("rtor")),
                    rmin=float(solid.get("rmin", 0)), rmax=float(solid.get("rmax")),
                    startphi=float(solid.get("startphi", 0))*scale, deltaphi=float(solid.get("deltaphi"))*scale,
                    min_width_mm=min_cell_mm, max_nodes=max_cells, coordinate_guard_mm=transform_guard)
                spans, extra = result.intervals.rename(columns={"status": "state"}), result.counts
            else:
                raise ValueError("actual solid probe not implemented")
        except ValueError as error:
            report.append(dict(row, resolved=False, reason=str(error)))
            continue
        spans, unresolved = hierarchy_check(index, placement, spans, origin, direction)
        spans["source"], spans["material"], spans["scope"] = path, placement["material"], record["scope"]
        inside = spans.state == "inside"
        known = inside & spans.ancestors_verified & spans.leaf_verified
        row.update(resolved=True, method=extra, solid_lower_mm=length_bound(spans[inside], upper=False),
            solid_upper_mm=length_bound(spans, upper=True), hierarchy_verified_lower_mm=length_bound(spans[known], upper=False),
            hit_lanes=int(spans.loc[inside, "lane"].nunique()), possible_lanes=int(spans.lane.nunique()),
            unresolved_ancestors=unresolved)
        report.append(row)
        intervals.append(spans)
    columns = ["lane", "s0_mm", "s1_mm", "state", "ancestors_verified", "leaf_verified", "replaces_material", "source", "material", "scope"]
    return (pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame(columns=columns)), report
