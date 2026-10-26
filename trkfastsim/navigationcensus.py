"""Read-only inventory of the stored TrackerInterior navigation representation.

``census(gdml_path, bbox)`` returns a JSON-serializable dict with ``records``,
``counts``, ``pruned`` and ``assumptions``; bbox is (lo, hi) in world-frame mm.
All TrackerMother descendants and external placements with overlapping or unknown
bounds are inventoried. Straw/wire subtrees are counted, not expanded. Vacuum is
counted separately. Calorimeter descendants use the default calo=True route.
No trajectories, material losses, exact Boolean volumes or truth navigation are
computed. AABB overlap means possible overlap, never a solid intersection claim.
"""
from collections import Counter
from functools import lru_cache
import re
import xml.etree.ElementTree as ET

import numpy as np

from mu2efastsim import world
from mu2efastsim.worldcensus import TUBE_LIKE, _first
from .steps import CALO_MOTHER, INTERIOR_SKIP

_VACUUM = re.compile(r"AIR|Vacuum|VACUUM|Galactic")
_BOOLEAN = ("subtraction", "union", "intersection")


def _placed(bounds, rotation, position):
    if bounds is None:
        return None
    lo, hi = bounds
    center = rotation @ ((lo + hi) / 2) + position
    extent = np.abs(rotation) @ ((hi - lo) / 2)
    return center - extent, center + extent


def _overlaps(bounds, bbox):
    return not (np.any(bounds[1] < bbox[0]) or np.any(bounds[0] > bbox[1]))


def _geometry_bounds(solid, solids):
    """Conservative bounds only; deliberately unknown for unimplemented shapes."""
    if solid.get("lunit", "mm") != "mm" or solid.get("aunit", "deg") not in ("deg", "rad"):
        return None
    tag = solid.tag
    if tag in _BOOLEAN:
        if _placement_notes(solid) or any(child.tag.startswith("first") and child.tag != "first" for child in solid):
            return None
        first = _geometry_bounds(solids[solid.find("first").get("ref")], solids)
        if tag != "union" or first is None:
            return first
        second = _geometry_bounds(solids[solid.find("second").get("ref")], solids)
        p = solid.find("position")
        position = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
        second = _placed(second, world._rot(solid.find("rotation")), position)
        if second is None:
            return None
        return np.minimum(first[0], second[0]), np.maximum(first[1], second[1])
    if tag == "box":
        h = np.array([float(solid.get(a)) for a in "xyz"]) / 2
        return -h, h
    if tag in ("tube", "cone", "torus"):
        if tag == "torus":
            h = float(solid.get("rmax"))
            r = float(solid.get("rtor")) + h
        else:
            r = (max(float(solid.get(a)) for a in ("rmax1", "rmax2")) if tag == "cone"
                 else float(solid.get("rmax")))
            h = float(solid.get("z")) / 2
        return np.array([-r, -r, -h]), np.array([r, r, h])
    if tag in ("polycone", "polyhedra"):
        planes = solid.findall("zplane")
        if not planes:
            return None
        r = max(float(p.get("rmax")) for p in planes)
        z = [float(p.get("z")) for p in planes]
        if tag == "polyhedra":
            sides = int(solid.get("numsides", 0))
            angle = float(solid.get("deltaphi", 0)) * (np.pi / 180 if solid.get("aunit", "deg") == "deg" else 1)
            if sides <= 0 or not 0 < angle <= 2 * np.pi or angle / sides >= np.pi:
                return None
            # GDML polyhedra radii are apothems; bound the circumradius, not just rmax.
            r /= np.cos(angle / (2 * sides))
        return np.array([-r, -r, min(z)]), np.array([r, r, max(z)])
    if tag == "xtru":
        vertices, sections = solid.findall("twoDimVertex"), solid.findall("section")
        if len(vertices) < 3 or len(sections) < 2:
            return None
        xy = np.array([[float(p.get(a)) for a in "xy"] for p in vertices])
        points = []
        for section in sections:
            offset = np.array([float(section.get(a, 0)) for a in ("xOffset", "yOffset")])
            scaled = xy * float(section.get("scalingFactor", 1)) + offset
            points.extend([[x, y, float(section.get("zPosition"))] for x, y in scaled])
        # Interpolated sections stay inside the envelope of all transformed section vertices.
        points = np.array(points)
        return points.min(0), points.max(0)
    if tag == "arb8":
        xy = np.array([[float(solid.get(f"v{i}{a}")) for a in "xy"] for i in range(1, 9)])
        h = float(solid.get("dz"))
        return np.r_[xy.min(0), -h], np.r_[xy.max(0), h]
    return None


def _tube_bounds(solid):
    """Local full-circle envelopes matching load_tubes, including its approximations."""
    if solid.tag in ("tube", "cutTube"):
        r, h = float(solid.get("rmax")), float(solid.get("z")) / 2
        return [(np.array([-r, -r, -h]), np.array([r, r, h]))]
    if solid.tag == "polycone":
        planes = sorted((float(p.get("z")), float(p.get("rmin", 0)), float(p.get("rmax")))
                        for p in solid.findall("zplane"))
        return [(np.array([-r, -r, z0]), np.array([r, r, z1]))
                for (z0, _, r0), (z1, _, r1) in zip(planes[:-1], planes[1:])
                if z1 > z0 for r in [max(r0, r1)]]
    return []


def _solid_notes(solid, solids, signed=False):
    notes = []
    if solid.get("lunit", "mm") != "mm":
        notes.append("solid_length_unit_ignored:" + solid.get("lunit"))
    if solid.get("aunit", "deg") not in ("deg", "rad"):
        notes.append("solid_angle_unit_treated_as_radians:" + solid.get("aunit"))
    if solid.tag in _BOOLEAN:
        if not signed:
            notes.append("boolean_first_only:" + solid.tag)
        for tag in (("first", "second") if signed else ("first",)):
            notes.extend(_solid_notes(solids[solid.find(tag).get("ref")], solids, signed))
        if signed:
            notes.extend(_placement_notes(solid))
    if solid.tag == "cutTube":
        notes.append("cutTube_planes_ignored")
    if solid.tag == "polycone":
        planes = sorted((float(p.get("z")), float(p.get("rmin", 0)), float(p.get("rmax")))
                        for p in solid.findall("zplane"))
        if any(z1 > z0 and (a0 != a1 or b0 != b1)
               for (z0, a0, b0), (z1, a1, b1) in zip(planes[:-1], planes[1:])):
            notes.append("tapered_polycone_section_envelopes")
    if signed and solid.tag in TUBE_LIKE:
        notes.append("equal_area_polygonal_tube")
    if signed and solid.tag == "xtru":
        if any(float(p.get(a, 0)) != 0 for p in solid.findall("section") for a in ("xOffset", "yOffset")):
            notes.append("xtru_section_offsets_ignored")
    return notes


def _placement_notes(placement):
    notes = []
    for kind in ("position", "rotation", "scale"):
        ref = placement.find(kind + "ref")
        if ref is not None:
            notes.append(kind + "ref_ignored:" + ref.get("ref", ""))
    p, r = placement.find("position"), placement.find("rotation")
    if p is not None and p.get("unit", "mm") != "mm":
        notes.append("position_unit_ignored:" + p.get("unit"))
    if r is not None and r.get("unit", "deg") not in ("deg", "rad"):
        notes.append("rotation_unit_treated_as_radians:" + r.get("unit"))
    scale = placement.find("scale")
    if scale is not None and any(float(scale.get(a, 1)) != 1 for a in "xyz"):
        notes.append("placement_scale_ignored")
    return notes


def census(gdml_path, bbox):
    """Return a JSON-safe placement inventory, not a geometry qualification.

    ``tube_loader_selected`` and piece counts describe loader numeric conventions,
    even when ``geometry_bbox_relation`` is unknown. ``classification`` is one of
    retained, approximate, unsupported, intentional_exclusion, bbox_excluded.
    External known-disjoint rows are summarized rather than stored. Physical paths
    include sibling indices to distinguish repeated logical-volume placements.
    """
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape != (2, 3) or not np.isfinite(bbox).all() or np.any(bbox[0] > bbox[1]):
        raise ValueError("bbox must be finite ordered (lo, hi) world-frame 3-vectors")
    root = ET.parse(gdml_path).getroot()
    volumes = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    records, counts, pruned = [], Counter(), Counter()
    warnings = set()

    @lru_cache(None)
    def solid_info(ref, signed):
        s = solids[ref]
        return (_geometry_bounds(s, solids), _tube_bounds(_first(s, solids)),
                _solid_notes(s, solids, signed), world.signed_local(s, solids) if signed else None)

    def walk(lv, rotation, position, path, physical, tracker, calo, tube_skip, inherited, ancestors):
        if lv in ancestors:
            raise ValueError("cyclic GDML placement: " + lv)
        volume = volumes[lv]
        for tag in ("replicavol", "divisionvol", "paramvol"):
            if volume.find(tag) is not None:
                raise ValueError("unsupported GDML placement construct: " + tag)
        for index, pv in enumerate(volume.findall("physvol")):
            ref = pv.find("volumeref")
            if ref is None or ref.get("ref") not in volumes:
                raise ValueError("physvol must reference a known volume")
            child = ref.get("ref")
            leaf = world._strip(child)
            child_path = path + leaf
            child_physical = physical + f"{pv.get('name', leaf)}[{index}]"
            straw = next((prefix for prefix in INTERIOR_SKIP if prefix != CALO_MOTHER and leaf.startswith(prefix)), None)
            if straw:
                pruned[straw] += 1
                continue
            placement_notes = inherited + _placement_notes(pv)
            warnings.update(placement_notes)
            p = pv.find("position")
            local = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
            r2, t2 = rotation @ world._rot(pv.find("rotation")), rotation @ local + position
            in_tracker, in_calo = tracker or leaf == "TrackerMother", calo or leaf == CALO_MOTHER
            skip = tube_skip or leaf.startswith(tuple(INTERIOR_SKIP))
            v = volumes[child]
            material = world._strip(v.find("materialref").get("ref"))
            solid_ref = v.find("solidref").get("ref")
            s = solids[solid_ref]
            signed = in_calo and leaf != CALO_MOTHER
            bounds, tubes, notes, pieces = solid_info(solid_ref, signed)
            global_bounds = None if placement_notes else _placed(bounds, r2, t2)
            relation = "unknown" if global_bounds is None else ("overlap" if _overlaps(global_bounds, bbox) else "disjoint")
            counts["visited_placements"] += 1
            vacuum = bool(_VACUUM.search(material))
            if vacuum:
                counts["vacuum_placements"] += 1
            elif not in_tracker and not in_calo and relation == "disjoint":
                counts["external_known_disjoint"] += 1
            else:
                n_tubes = sum(_overlaps(_placed(b, r2, t2), bbox) for b in tubes) if not skip else 0
                n_pieces = sum(_overlaps((lo, hi), bbox) for _, _, lo, hi in world._move(pieces, r2, t2)) if pieces else 0
                reasons = list(dict.fromkeys(notes + placement_notes))
                if material == "StrawGas":
                    classification, reason = "intentional_exclusion", "StrawGas_filtered_by_TrackerInterior"
                elif leaf == CALO_MOTHER:
                    classification, reason = "intentional_exclusion", "calorimeter_mother_not_emitted"
                elif signed:
                    classification = "unsupported" if pieces is None else ("approximate" if reasons else "retained")
                    reason = "signed_local_unsupported" if pieces is None else "calorimeter_signed_route"
                    if pieces is not None and n_pieces == 0:
                        classification, reason = "bbox_excluded", "no_signed_pieces_selected"
                elif not tubes:
                    classification, reason = "unsupported", "non_tube_support_omitted"
                elif not n_tubes:
                    classification, reason = "bbox_excluded", "no_tube_sections_selected"
                else:
                    classification, reason = ("approximate" if reasons else "retained"), "tube_route"
                counts[classification] += 1
                records.append({
                    "leaf": leaf, "path": child_path, "placement_path": child_physical,
                    "material": material, "solid": solid_ref, "solid_tag": s.tag,
                    "loader_base_tag": _first(s, solids).tag,
                    "scope": "calorimeter" if in_calo else ("tracker" if in_tracker else "external"),
                    "is_leaf": not bool(v.findall("physvol")),
                    "classification": classification, "reason": reason,
                    "tube_loader_selected": bool(n_tubes), "tube_loader_excluded_by_name": bool(skip),
                    "tube_sections_total": len(tubes), "tube_sections_selected": int(n_tubes),
                    "signed_pieces_selected": int(n_pieces),
                    "interior_selected": bool((n_pieces if signed else n_tubes) and material != "StrawGas"),
                    "geometry_bbox_relation": relation,
                    "geometry_bounds_mm": None if global_bounds is None else [a.tolist() for a in global_bounds],
                    "approximations": reasons,
                    "placement_assumptions": placement_notes,
                })
            walk(child, r2, t2, child_path + "/", child_physical + "/", in_tracker, in_calo,
                 skip, placement_notes, ancestors + (lv,))

    walk(root.find("setup/world").get("ref"), np.eye(3), np.zeros(3), "", "", False, False, False, [], ())
    return {
        "schema": "tracker-navigation-census-v1", "bbox_mm": bbox.tolist(), "records": records,
        "counts": dict(sorted(counts.items())), "pruned": dict(sorted(pruned.items())),
        "placement_warnings": sorted(warnings),
        "assumptions": [
            "Default TrackerInterior calo=True selection; no production behavior changed.",
            "Literal unnamespaced GDML volumes/physvols; unsupported placement constructs raise.",
            "Loader numeric lengths assume mm; inline rotations use inverse Rz Ry Rx, deg or otherwise rad.",
            "Placement refs/nonidentity scales are ignored by loaders; affected descendant bounds are unknown.",
            "Known mm bounds: box/tube/cone/torus/polycone/polyhedra/xtru/arb8; angular sectors use full envelopes.",
            "Boolean unions envelope both transformed operands; subtraction/intersection use first bounds.",
            "Unresolved transforms/units, cutTube and other unimplemented bounds are unknown, never absent.",
            "AABB overlap is only a candidate; disjoint exclusions apply to conservative bounds, not unknowns.",
            "Tube selection mirrors load_tubes section AABBs; Booleans use first only and cut planes are ignored.",
            "Polycone tapered sections use smaller inner/larger outer radii; calo tubes use equal-area polygons.",
            "Signed piece counts are own-solid pieces, not mother-material subtraction terms or overlap checks.",
            "Vacuum-name filtering matches loaders; StrawGas is separately excluded by TrackerInterior.",
            "Straw/wire counts cover pruned roots only; their descendants are intentionally not enumerated.",
            "No material constants, shell rounding, axis tolerance, loss or ray-navigation qualification is made.",
        ],
    }
