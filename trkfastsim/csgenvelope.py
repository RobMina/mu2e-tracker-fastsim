"""Guarded signed planar CSG occupancy on a fixed stored local-plane model, not GDML certification."""
import numpy as np
import pandas as pd

from mu2efastsim import world
from . import materialpath

POLICY = "signed-planar-csg-envelope-v1"


def supported(solid, solids, seen=()):
    name = solid.get("name")
    if name in seen or solid.get("lunit", "mm") != "mm" or solid.get("aunit", "deg") not in ("deg", "rad"):
        raise ValueError("unsupported/cyclic CSG solid or units")
    if solid.tag == "box":
        if not np.isfinite([float(solid.get(a)) for a in "xyz"]).all() or min(float(solid.get(a)) for a in "xyz") <= 0:
            raise ValueError("finite positive CSG box dimensions required")
    elif solid.tag == "xtru":
        sections = solid.findall("section")
        if len(sections) != 2 or any(float(s.get("scalingFactor", 1)) != 1 or float(s.get("xOffset", 0)) != 0 or float(s.get("yOffset", 0)) != 0 for s in sections):
            raise ValueError("CSG xtru requires two unshifted unscaled sections")
        vertices = np.array([[float(v.get(a)) for a in "xy"] for v in solid.findall("twoDimVertex")])
        z = np.array([float(s.get("zPosition")) for s in sections])
        if not np.isfinite(vertices).all() or not np.isfinite(z).all() or z[1] <= z[0]:
            raise ValueError("finite ordered xtru coordinates and positive extent required")
        triangles = world._convex_pieces_2d(vertices)
        area = lambda v: .5 * abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))
        if len(vertices) < 3 or len(triangles) != len(vertices) - 2 or not np.isclose(sum(area(t) for t in triangles), area(vertices), rtol=1e-12, atol=1e-10):
            raise ValueError("CSG xtru triangulation incomplete")
    elif solid.tag in ("subtraction", "union", "intersection"):
        if any(solid.find(tag) is not None for tag in ("positionref", "rotationref", "scale", "scaleref")):
            raise ValueError("unsupported CSG transform references or scale")
        p, r = solid.find("position"), solid.find("rotation")
        if (p is not None and p.get("unit", "mm") != "mm") or (r is not None and r.get("unit", "deg") not in ("deg", "rad")):
            raise ValueError("unsupported CSG transform units")
        if any(not np.isfinite([float(node.get(a, 0)) for a in "xyz"]).all() for node in (p, r) if node is not None):
            raise ValueError("finite CSG transform coordinates required")
        for tag in ("first", "second"):
            supported(solids[solid.find(tag).get("ref")], solids, seen + (name,))
    else:
        raise ValueError("only planar boxes/xtru/Boolean CSG supported")


def construction_bound(solid, solids, *, max_work, depth=0):
    if depth > 32:
        raise ValueError("CSG construction depth cap")
    if solid.tag in ("box", "xtru"):
        pieces = 1 if solid.tag == "box" else len(solid.findall("twoDimVertex")) - 2
        bound = (pieces, 6)
    else:
        a, ap = construction_bound(solids[solid.find("first").get("ref")], solids, max_work=max_work, depth=depth+1)
        b, bp = construction_bound(solids[solid.find("second").get("ref")], solids, max_work=max_work, depth=depth+1)
        pieces = a*b if solid.tag == "intersection" else a + (b if solid.tag == "union" else 0) + a*b
        bound = (pieces, ap+bp)
    if bound[0] * bound[1] > max_work:
        raise ValueError("CSG construction piece/plane cap")
    return bound


def _project(value, upward):
    result = np.asarray(value, float)
    wrong = result.astype(np.longdouble) < value if upward else result.astype(np.longdouble) > value
    return np.where(wrong, np.nextafter(result, np.inf if upward else -np.inf), result)


def probe(solid, solids, origin, direction, lengths, *, coordinate_guard_mm=0., max_pairs=2000000):
    """Piece membership enclosures yield signed lower/upper occupancy bounds on common atoms.

    Positive inner minus negative outer bounds exact signed occupancy from below;
    positive outer minus negative inner bounds it from above. Bounds outside 0/1
    are uncertainty, never clipped into admission. Coplanar uncertainty survives.
    """
    if not np.isfinite(coordinate_guard_mm) or coordinate_guard_mm < 0:
        raise ValueError("finite nonnegative CSG coordinate guard required")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)) or max_pairs <= 0:
        raise ValueError("positive integer CSG piece/plane cap required")
    construction_bound(solid, solids, max_work=max_pairs)
    supported(solid, solids)
    origin, direction, lengths = materialpath._rays(origin, direction, lengths)
    wide = np.longdouble
    if np.finfo(wide).nmant <= np.finfo(float).nmant:
        raise ValueError("CSG spatial arithmetic requires wider mantissa")
    pieces = world.signed_local(solid, solids)
    if pieces is None or len(pieces) * len(origin) > max_pairs:
        raise ValueError("CSG piece/ray cap or unsupported signed representation")
    o, d = np.asarray(origin, wide), np.asarray(direction, wide)
    eps = np.finfo(wide).eps
    events = []
    max_guard = wide(0)
    for sign, raw_planes, _, _ in pieces:
        planes = np.asarray(raw_planes, wide)
        if planes.ndim != 2 or planes.shape[1] != 4 or not np.isfinite(planes).all():
            raise ValueError("finite CSG plane coefficients required")
        norms = np.sqrt(np.sum(planes[:, :3] ** 2, axis=1))
        if (norms == 0).any() or not np.isfinite(norms).all():
            raise ValueError("finite nonzero CSG normals required")
        planes = planes / norms[:, None]
        scale = max(wide(1), np.abs(o).max(initial=wide(0)), np.abs(planes).max(initial=wide(0)), wide(lengths.max(initial=0.)))
        guard = np.nextafter(wide(coordinate_guard_mm) + wide(512) * eps * scale, wide(np.inf))
        max_guard = max(max_guard, guard)
        den = np.sum(d[:, None, :] * planes[None, :, :3], axis=2)
        num = planes[None, :, 3] - np.sum(o[:, None, :] * planes[None, :, :3], axis=2)
        for inner in (True, False):
            adjusted = num - guard if inner else num + guard
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = adjusted / den
            low = np.maximum(wide(0), np.where(den < 0, cross, -np.inf).max(axis=1))
            high = np.minimum(np.asarray(lengths, wide), np.where(den > 0, cross, np.inf).min(axis=1))
            valid = ~((den == 0) & (adjusted < 0)).any(axis=1) & (high > low)
            a = np.maximum(0., _project(low[valid], upward=inner))
            b = np.minimum(lengths[valid], _project(high[valid], upward=not inner))
            ids = np.flatnonzero(valid)
            keep = b > a
            ids, a, b = ids[keep], a[keep], b[keep]
            lower_term = (sign == 1 and inner) or (sign == -1 and not inner)
            name = "lower" if lower_term else "upper"
            events.extend([pd.DataFrame(dict(lane=ids, s=a, bound=name, delta=sign)),
                           pd.DataFrame(dict(lane=ids, s=b, bound=name, delta=-sign))])
    anchors = pd.concat([pd.DataFrame(dict(lane=np.arange(len(lengths)), s=np.zeros(len(lengths)))),
                         pd.DataFrame(dict(lane=np.arange(len(lengths)), s=lengths))])
    if events:
        rows = pd.concat(events, ignore_index=True).groupby(["lane", "s", "bound"], as_index=False).delta.sum()
        wide_events = rows.pivot(index=["lane", "s"], columns="bound", values="delta").fillna(0).reset_index()
    else:
        wide_events = anchors.iloc[:0].copy()
    edges = anchors.merge(wide_events, on=["lane", "s"], how="outer").fillna(0).drop_duplicates(["lane", "s"]).sort_values(["lane", "s"])
    for name in ("lower", "upper"):
        if name not in edges:
            edges[name] = 0
        edges[name] = edges.groupby("lane")[name].cumsum()
    edges["s1_mm"] = edges.groupby("lane").s.shift(-1)
    atoms = edges[edges.s1_mm > edges.s].rename(columns={"s": "s0_mm"}).copy()
    if (atoms.lower > atoms.upper).any():
        raise ValueError("CSG signed lower/upper enclosure inverted")
    atoms["state"] = np.where(atoms.lower >= 1, "inside", np.where(atoms.upper < 1, "outside", "unresolved"))
    kept = atoms[atoms.state != "outside"][["lane", "s0_mm", "s1_mm", "state"]].copy()
    kept["reason"] = np.where(kept.state == "inside", "certificate", "csg_plane_offset")
    return kept, dict(certificate_scheme=POLICY, method="signed planar primitive inner/outer occupancy bounds", guard_mm=float(max_guard),
        capped=False, lanes=len(lengths), positive_length_lanes=int(np.count_nonzero(lengths)), pieces=len(pieces),
        inside_cells=int((kept.state == "inside").sum()), unresolved_cells=int((kept.state == "unresolved").sum()),
        unresolved_lanes=int(kept.loc[kept.state == "unresolved", "lane"].nunique()),
        unresolved_reasons={str(k): int(v) for k, v in kept.loc[kept.state == "unresolved", "reason"].value_counts().items()})
