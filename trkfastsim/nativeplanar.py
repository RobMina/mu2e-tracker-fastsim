"""Separate literal-coordinate/native-polygon model; not the frozen binary64 stored-plane model."""
from fractions import Fraction
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from mu2efastsim import world
from . import csgenvelope, exactplanar, materialpath

POLICY = "literal-gdml-native-planar-quarterturn-v1"


def _number(value):
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("finite literal rational coordinate required") from error


def _transform(node):
    if any(node.find(tag) is not None for tag in ("positionref", "rotationref", "scale", "scaleref")):
        raise ValueError("native model transform references/scales unsupported")
    p, rotation = node.find("position"), node.find("rotation")
    if p is not None and p.get("unit", "mm") != "mm":
        raise ValueError("native model requires mm positions")
    position = np.array([_number(p.get(a, "0")) if p is not None else Fraction(0) for a in "xyz"], object)
    if rotation is None:
        return np.eye(3, dtype=int).astype(object), position, 0
    if rotation.get("unit", "deg") not in ("deg", "rad"):
        raise ValueError("native rotation units")
    angles = [_number(rotation.get(a, "0")) for a in "xyz"]
    if rotation.get("unit", "deg") == "deg" and all(a.denominator == 1 and a % 90 == 0 for a in angles):
        cs = [((1, 0), (0, 1), (-1, 0), (0, -1))[int(a / 90) % 4] for a in angles]
        (cx, sx), (cy, sy), (cz, sz) = cs
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], object)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], object)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], object)
        return (rz @ ry @ rx).T, position, 0
    return exactplanar.rational(world._rot(rotation)), position, 1


class LiteralIndex:
    """Accumulate literal inline positions and exact quarter turns along authenticated physical paths."""

    def __init__(self, gdml, wanted):
        root = ET.parse(gdml).getroot()
        volumes = {v.get("name"): v for v in root.find("structure").findall("volume")}
        wanted = set(wanted)
        self.placements = {}
        def walk(lv, rotation, position, path, count, lineage):
            if lv in lineage:
                raise ValueError("native cyclic placement")
            volume = volumes[lv]
            if any(volume.find(tag) is not None for tag in ("replicavol", "divisionvol", "paramvol")):
                raise ValueError("native unimplemented placement")
            for i, pv in enumerate(volume.findall("physvol")):
                child = pv.find("volumeref").get("ref")
                here = path + f"{pv.get('name', world._strip(child))}[{i}]"
                if not any(p == here or p.startswith(here + "/") for p in wanted):
                    continue
                r, t, n = _transform(pv)
                r, t, n = rotation @ r, rotation @ t + position, count + n
                if here in wanted:
                    self.placements[here] = dict(rotation=r, position=t, nonquarter_rotations=n)
                walk(child, r, t, here + "/", n, lineage + (lv,))
        walk(root.find("setup/world").get("ref"), np.eye(3, dtype=int).astype(object), np.zeros(3, dtype=int).astype(object), "", 0, ())
        if set(self.placements) != wanted:
            raise ValueError("native model physical path coverage mismatch")


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _polygon(solid, reviewed):
    if solid.tag == "box":
        h = [_number(solid.get(a)) / 2 for a in "xyz"]
        return np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]], object), -h[2], h[2]
    sections = solid.findall("section")
    if len(sections) != 2 or any(_number(s.get("scalingFactor", "1")) != 1 or
            _number(s.get("xOffset", "0")) != 0 or _number(s.get("yOffset", "0")) != 0 for s in sections):
        raise ValueError("native literal section scale/offset must be exactly identity")
    reviewed._xtru_pieces(solid)
    vertices = np.array([[_number(v.get(a)) for a in "xy"] for v in solid.findall("twoDimVertex")], object)
    if len(vertices) * len(vertices) > 2000000:
        raise ValueError("native exact polygon validation pair cap")
    edges = np.roll(vertices, -1, axis=0) - vertices
    if ((edges[:, 0] == 0) & (edges[:, 1] == 0)).any() or sum(_cross(vertices, np.roll(vertices, -1, axis=0))) == 0:
        raise ValueError("native exact polygon must have positive area and nonzero edges")
    ii, jj = np.triu_indices(len(vertices), 1)
    keep = (jj != ii + 1) & ~((ii == 0) & (jj == len(vertices) - 1))
    ii, jj = ii[keep], jj[keep]
    a, b, c, d = vertices[ii], vertices[(ii+1) % len(vertices)], vertices[jj], vertices[(jj+1) % len(vertices)]
    s0, s1 = _cross(b-a, c-a), _cross(b-a, d-a)
    t0, t1 = _cross(d-c, a-c), _cross(d-c, b-c)
    overlap = (np.maximum(np.minimum(a, b), np.minimum(c, d)) <= np.minimum(np.maximum(a, b), np.maximum(c, d))).all(axis=1)
    if ((s0*s1 <= 0) & (t0*t1 <= 0) & overlap).any():
        raise ValueError("native exact polygon self-intersection")
    z = [_number(s.get("zPosition")) for s in solid.findall("section")]
    if z[1] <= z[0]:
        raise ValueError("native literal positive axial extent required")
    return vertices, z[0], z[1]


def intervals(solid, solids, placement, origin, direction, lengths, *, reviewed, max_work=2000000):
    """Exact roots and winding on literal native primitives, with recursive Boolean masks.

    Quarter-turn placement rotations are exact integers. Other rotations are
    explicitly the core's frozen binary64 coefficients, promoted as rationals.
    This is a different geometry model, not reduced uncertainty on the old one.
    """
    if isinstance(max_work, (bool, np.bool_)) or not isinstance(max_work, (int, np.integer)) or max_work <= 0:
        raise ValueError("positive integer native model work cap")
    csgenvelope.construction_bound(solid, solids, max_work=max_work)
    csgenvelope.supported(solid, solids)
    origin, direction, lengths = materialpath._rays(origin, direction, lengths)
    qo, qd, end = exactplanar.rational(origin), exactplanar.rational(direction), exactplanar.rational(lengths)
    o = (qo - placement["position"]) @ placement["rotation"]
    d = qd @ placement["rotation"]
    primitives, nodes, roots, nonquarter = [], [], [], placement["nonquarter_rotations"]
    work = 0
    def build(shape, local_o, local_d):
        nonlocal work, nonquarter
        node_id = len(nodes)
        nodes.append(None)
        if shape.tag in ("box", "xtru"):
            vertices, z0, z1 = _polygon(shape, reviewed)
            work += (len(vertices) + 2) * len(origin)
            if work > max_work:
                raise ValueError("native primitive ray/edge cap")
            edges = np.roll(vertices, -1, axis=0) - vertices
            num = _cross(edges[None, :, :], vertices[None, :, :] - local_o[:, None, :2])
            den = _cross(edges[None, :, :], local_d[:, None, :2])
            nums = np.concatenate([num, (z0 - local_o[:, 2])[:, None], (z1 - local_o[:, 2])[:, None]], axis=1)
            dens = np.concatenate([den, local_d[:, 2, None], local_d[:, 2, None]], axis=1)
            q = nums / np.where(dens == 0, Fraction(1), dens)
            valid = (dens != 0) & (q > 0) & (q < end[:, None])
            lane, plane = np.nonzero(valid)
            roots.append(pd.DataFrame(dict(lane=lane, q=q[lane, plane])))
            nodes[node_id] = ("primitive", len(primitives))
            primitives.append((vertices, z0, z1, local_o, local_d))
        else:
            a = build(solids[shape.find("first").get("ref")], local_o, local_d)
            r, t, n = _transform(shape)
            nonquarter += n
            b = build(solids[shape.find("second").get("ref")], (local_o - t) @ r, local_d @ r)
            nodes[node_id] = (shape.tag, a, b)
        return node_id
    root_node = build(solid, o, d)
    anchors = [pd.DataFrame(dict(lane=np.arange(len(end)), q=np.full(len(end), Fraction(0), object))),
               pd.DataFrame(dict(lane=np.arange(len(end)), q=end))]
    edges = pd.concat([*anchors, *roots], ignore_index=True).drop_duplicates(["lane", "q"]).sort_values(["lane", "q"])
    edges["q1"] = edges.groupby("lane").q.shift(-1)
    atoms = edges[edges.q1.notna() & (edges.q1 > edges.q)].rename(columns={"q": "q0"}).reset_index(drop=True)
    if sum(len(v[0]) + 2 for v in primitives) * len(atoms) > max_work:
        raise ValueError("native atom/edge evaluation cap")
    lane = atoms.lane.to_numpy(np.int64)
    mid = (atoms.q0.to_numpy(object) + atoms.q1.to_numpy(object)) / 2
    membership, boundaries = [], []
    for vertices, z0, z1, local_o, local_d in primitives:
        point = local_o[lane] + local_d[lane] * mid[:, None]
        a, b = vertices, np.roll(vertices, -1, axis=0)
        x, y = point[:, 0, None], point[:, 1, None]
        cross = _cross((b - a)[None, :, :], point[:, None, :2] - a[None, :, :])
        on_edge = (cross == 0) & (x >= np.minimum(a[:, 0], b[:, 0])) & (x <= np.maximum(a[:, 0], b[:, 0])) & (y >= np.minimum(a[:, 1], b[:, 1])) & (y <= np.maximum(a[:, 1], b[:, 1]))
        upward = (a[:, 1] <= y) & (b[:, 1] > y) & (cross > 0)
        downward = (a[:, 1] > y) & (b[:, 1] <= y) & (cross < 0)
        inside_xy = (upward.sum(axis=1) - downward.sum(axis=1)) != 0
        edge_xy = on_edge.any(axis=1)
        z = point[:, 2]
        inside = inside_xy & ~edge_xy & (z > z0) & (z < z1)
        boundary = (edge_xy & (z >= z0) & (z <= z1)) | ((z == z0) | (z == z1)) & (inside_xy | edge_xy)
        membership.append(inside)
        boundaries.append(boundary)
    def evaluate(node):
        kind, *args = nodes[node]
        if kind == "primitive":
            k = args[0]
            return membership[k], membership[k] | boundaries[k]
        al, au = evaluate(args[0])
        bl, bu = evaluate(args[1])
        if kind == "subtraction":
            return al & ~bu, au & ~bl
        return (al | bl, au | bu) if kind == "union" else (al & bl, au & bu)
    lower, upper = evaluate(root_node)
    occupied = atoms.loc[lower, ["lane", "q0", "q1"]].copy()
    if len(occupied):
        previous = occupied.shift()
        blocks = ((occupied.lane != previous.lane) | (occupied.q0 != previous.q1)).cumsum()
        occupied = occupied.groupby(blocks, sort=False).agg(lane=("lane", "first"), q0=("q0", "first"), q1=("q1", "last")).reset_index(drop=True)
    contact_lanes = atoms.loc[lower != upper, "lane"].unique().astype(np.int64)
    faults = atoms.iloc[:0][["lane", "q0", "q1"]].assign(occupancy=pd.Series(dtype="int64"))
    payload = ET.tostring(solid) + str((placement["rotation"].tolist(), placement["position"].tolist())).encode()
    return occupied, faults, dict(policy=POLICY, lanes=len(lengths), native_primitives=len(primitives),
        atom_rows=len(atoms), primitive_edge_ray_work=work, max_work=max_work, capped=False,
        coplanar_lanes=contact_lanes.tolist(), invalid_occupancy_atoms=0, occupied_atoms=len(occupied),
        operands_sha256=hashlib.sha256(payload).hexdigest(), nonquarter_frozen_binary64_rotations=nonquarter,
        model="literal rational positions/vertices, exact quarter turns, frozen binary64 nonquarter rotations, native polygon winding and recursive Boolean membership")
