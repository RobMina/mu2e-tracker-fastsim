"""Explicit signed-source hierarchy normalization, not a new material or geometry priority."""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from mu2efastsim import world
from . import boundarypolicy, convexpath, geometry, materialpath, steps

POLICY = "calo-physical-placement-signed-parent-v1"
VACUUM = re.compile(r"AIR|Vacuum|VACUUM|Galactic")
COLS = ["lane", "s0_mm", "s1_mm", "material", "source", "sign"]


class RaySelectedSources:
    """Cache source selections that enclose actual fresh-ray extents, never truncate a ray to a loader box."""

    def __init__(self, gdml, initial_bbox, *, wide_intersections=False, local_planes=False):
        self.gdml, self.bbox = gdml, np.asarray(initial_bbox, float)
        self.wide_intersections, self.local_planes = wide_intersections, local_planes
        self.loaded = PlacementSources(gdml, self.bbox, wide_intersections=wide_intersections, local_planes=local_planes)
        self.selections = []

    def __call__(self, origin, direction, lengths):
        end = origin + direction * lengths[:, None]
        lo, hi = np.minimum(origin, end).min(axis=0), np.maximum(origin, end).max(axis=0)
        if np.any(lo < self.bbox[0]) or np.any(hi > self.bbox[1]):
            self.bbox = np.stack([np.nextafter(np.minimum(lo, self.bbox[0]), -np.inf),
                                  np.nextafter(np.maximum(hi, self.bbox[1]), np.inf)])
            self.loaded = PlacementSources(self.gdml, self.bbox, wide_intersections=self.wide_intersections, local_planes=self.local_planes)
        self.selections.append(self.bbox.copy())
        return self.loaded(origin, direction, lengths)

    @property
    def relationships(self):
        return self.loaded.relationships

    def normalize(self, adjusted, lengths):
        return self.loaded.normalize(adjusted, lengths)


class PlacementSources:
    """Preserve physical owners and direct-daughter subtraction through stored calorimeter CSG."""

    def __init__(self, gdml, bbox, *, wide_intersections=False, local_planes=False):
        self.wide_intersections, self.local_planes = wide_intersections, local_planes
        self.local_terms = []
        root = ET.parse(gdml).getroot()
        volumes = {v.get("name"): v for v in root.find("structure").findall("volume")}
        solids = {v.get("name"): v for v in root.find("solids")}
        self.terms, self.relationships = [], []
        def validate_transform(node):
            if any(node.find(tag) is not None for tag in ("positionref", "rotationref", "scaleref", "scale")):
                raise ValueError("placement references/scales require explicit ownership implementation")
            p, r = node.find("position"), node.find("rotation")
            if (p is not None and p.get("unit", "mm") != "mm") or (r is not None and r.get("unit", "deg") not in ("deg", "rad")):
                raise ValueError("ownership requires millimetre positions and deg/rad rotations")
        from functools import lru_cache
        @lru_cache(None)
        def validate_solid(solid):
            if solid.get("lunit", "mm") != "mm" or solid.get("aunit", "deg") not in ("deg", "rad"):
                raise ValueError("ownership requires supported solid length/angle units")
            if solid.tag in ("subtraction", "union", "intersection"):
                validate_transform(solid)
                for tag in ("first", "second"):
                    validate_solid(solids[solid.find(tag).get("ref")])
        @lru_cache(None)
        def contains_calo(lv):
            return world._strip(lv) == steps.CALO_MOTHER or any(contains_calo(p.find("volumeref").get("ref"))
                for p in volumes[lv].findall("physvol"))
        @lru_cache(None)
        def local_shape(solid):
            return world.signed_local(solid, solids)
        def emit(local, rotation, position, owner, material, operation, daughter):
            for original, transformed in zip(local, world._move(local, rotation, position)):
                sign, planes, lo, hi = transformed
                if np.any(hi < bbox[0]) or np.any(lo > bbox[1]):
                    continue
                label = str(len(self.relationships))
                self.relationships.append(dict(owner=owner, material=material, operation=operation, daughter=daughter,
                                               primitive_sign=int(sign)))
                self.terms.append((sign * (-1 if operation == "subtract-direct-daughter" else 1), label, planes, lo, hi))
                self.local_terms.append((label, np.asarray(original[1]).copy(), np.asarray(rotation).copy(), np.asarray(position).copy()))
        def walk(lv, rotation, position, path, inside, parent_key):
            for i, pv in enumerate(volumes[lv].findall("physvol")):
                child = pv.find("volumeref").get("ref")
                if not inside and not contains_calo(child):
                    continue
                leaf = world._strip(child)
                physical = path + f"{pv.get('name', leaf)}[{i}]"
                validate_transform(pv)
                p = pv.find("position")
                local_pos = np.zeros(3) if p is None else np.array([float(p.get(a, 0)) for a in "xyz"])
                r, t = rotation @ world._rot(pv.find("rotation")), rotation @ local_pos + position
                active = inside or leaf == steps.CALO_MOTHER
                key = None
                if active and leaf != steps.CALO_MOTHER:
                    material = world._strip(volumes[child].find("materialref").get("ref"))
                    solid = solids[volumes[child].find("solidref").get("ref")]
                    validate_solid(solid)
                    local = local_shape(solid)
                    key = None if VACUUM.search(material) else (physical, material)
                    if local is not None:
                        if key is not None:
                            emit(local, r, t, physical, material, "add-own-shape", physical)
                        if parent_key is not None:
                            emit(local, r, t, parent_key[0], parent_key[1], "subtract-direct-daughter", physical)
                walk(child, r, t, physical + "/", active, key)
        walk(root.find("setup/world").get("ref"), np.eye(3), np.zeros(3), "", False, None)
        legacy, _ = world.signed_subtree(gdml, steps.CALO_MOTHER, bbox=bbox, labels=False)
        own = [q for q in self.terms if self.relationships[int(q[1])]["material"] != "StrawGas"]
        legacy = [q for q in legacy if q[1] != "StrawGas"]
        def signature(q, material):
            return (q[0], material, np.asarray(q[2]).tobytes(), np.asarray(q[3]).tobytes(), np.asarray(q[4]).tobytes())
        from collections import Counter
        if Counter(signature(q, self.relationships[int(q[1])]["material"]) for q in own) != Counter(signature(q, q[1]) for q in legacy):
            raise ValueError("physical ownership loader differs from existing signed stored geometry")
        self.terms = own
        self.packed = world.SignedPieceSet(self.terms, cell=steps.CALO_CELL_MM) if self.terms else None
        self.relationships = pd.DataFrame(self.relationships)
        self.local_terms = {int(label): (planes, rotation, position) for label, planes, rotation, position in self.local_terms}
        tubes = world.load_tubes(gdml, bbox=bbox, skip_names=steps.INTERIOR_SKIP)
        labelled = [(f"tube:{i}|{t[0]}", *t[1:]) for i, t in enumerate(tubes) if t[0] != "StrawGas"]
        self.shells = world.CoaxialShells(labelled, geometry.DETECTOR_ORIGIN_EXTRACTED)

    def __call__(self, origin, direction, lengths):
        tube = materialpath.tube_hits(self.shells, origin, direction, lengths)
        label = tube.material.str.rsplit("|", n=1, expand=True).reindex(columns=[0, 1])
        tube["source"], tube["material"], tube["sign"] = label[0], label[1], 1
        tube["operation"], tube["daughter"] = "add-own-shape", tube.source
        rows = [tube]
        if self.packed is not None:
            intersect = convexpath._hits
            if self.wide_intersections:
                from .wideconvex import hits as intersect
            if self.local_planes:
                from .wideconvex import local_hits
                intersect = lambda ps, o, d, l: local_hits(ps, o, d, l, self.local_terms)
            for sign, packed in convexpath._sets(self.packed):
                for lane, piece, start, end in intersect(packed, origin, direction, lengths):
                    codes = np.asarray(packed.mats, object)[packed.mat[piece]].astype(int)
                    metadata = self.relationships.iloc[codes].reset_index(drop=True)
                    rows.append(pd.DataFrame(dict(lane=lane, s0_mm=start, s1_mm=end, material=metadata.material,
                        source=metadata.owner, sign=sign, operation=metadata.operation, daughter=metadata.daughter, term_id=codes)))
        return pd.concat(rows, ignore_index=True)

    def normalize(self, adjusted, lengths):
        if adjusted.empty:
            return normalize(adjusted, lengths)
        physical = adjusted[~adjusted.source.str.startswith("tube:")]
        if len(physical):
            identifiers = physical.term_id.to_numpy()
            if not np.isfinite(identifiers).all() or (identifiers != np.floor(identifiers)).any() or (identifiers < 0).any() or (identifiers >= len(self.relationships)).any():
                raise ValueError("physical source term identifier is not authenticated")
            expected = self.relationships.iloc[identifiers.astype(np.int64)]
            for field, column in (("source", "owner"), ("material", "material"), ("operation", "operation"), ("daughter", "daughter")):
                if not np.array_equal(physical[field].to_numpy(), expected[column].to_numpy()):
                    raise ValueError("physical source provenance differs from loaded geometry terms")
            signs = expected.primitive_sign.to_numpy() * np.where(expected.operation == "subtract-direct-daughter", -1, 1)
            if not np.array_equal(physical.sign.to_numpy(), signs):
                raise ValueError("physical source sign differs from loaded geometry term")
        return normalize(adjusted, lengths)


def parent_owners(cells, normalized, *, max_pairs):
    """Require any surviving physical calorimeter owner to be the leaf's direct actual parent."""
    selected = cells[cells.scope == "calorimeter"].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["expected_owner"] = selected.source.str.rsplit("/", n=1).str[0]
    selected["cell_id"] = np.arange(len(selected), dtype=np.int64)
    sizes = selected.groupby("lane").size().to_frame("n").join(normalized.groupby("lane").size().rename("m")).fillna(0)
    if (sizes.n * sizes.m).sum() > max_pairs:
        raise ValueError("physical parent comparison exceeds explicit pair cap")
    raw = normalized.rename(columns={"source": "owner", "s0_mm": "owner_s0_mm", "s1_mm": "owner_s1_mm", "material": "owner_material"})
    comparison = selected.merge(raw, on="lane")
    comparison["s0_mm"] = np.maximum(comparison.s0_mm, comparison.owner_s0_mm)
    comparison["s1_mm"] = np.minimum(comparison.s1_mm, comparison.owner_s1_mm)
    comparison = comparison[comparison.s1_mm > comparison.s0_mm].copy()
    if ((comparison.owner != comparison.expected_owner) | (comparison.owner_material != comparison.replaces_material)).any():
        raise ValueError("surviving physical owner is not the actual omitted leaf parent")
    nonvacuum = selected[selected.replaces_material != "vacuum"]
    coverage = comparison.groupby("cell_id", as_index=False).agg(start_mm=("s0_mm", "min"), end_mm=("s1_mm", "max"), spans=("lane", "size"))
    checked = nonvacuum.merge(coverage, on="cell_id", how="left", validate="one_to_one")
    previous = comparison.sort_values(["cell_id", "s0_mm"]).groupby("cell_id").s1_mm.shift()
    ordered = comparison.sort_values(["cell_id", "s0_mm"])
    if (checked.start_mm.isna().any() or (checked.start_mm != checked.s0_mm).any() or (checked.end_mm != checked.s1_mm).any()
            or (ordered.loc[previous.notna(), "s0_mm"] != previous[previous.notna()]).any()):
        raise ValueError("actual physical parent does not cover complete possible leaf footprint")
    return comparison


def normalize(adjusted, lengths):
    """Cancel only explicitly tagged own-shape CSG and direct-daughter terms per physical owner."""
    boundarypolicy._validate(adjusted, lengths)
    if adjusted.empty:
        return adjusted.copy(), pd.DataFrame()
    if not set(["operation", "daughter"]) <= set(adjusted) or not adjusted.operation.isin(
            ["add-own-shape", "subtract-direct-daughter"]).all():
        raise ValueError("ownership normalization requires explicit shape/direct-daughter provenance")
    tube = adjusted[adjusted.source.str.startswith("tube:")].copy()
    physical = adjusted[~adjusted.source.str.startswith("tube:")].copy()
    if (tube.sign != 1).any() or not (tube.operation == "add-own-shape").all():
        raise ValueError("stored tube approximations never enter physical cancellation")
    owners = physical[["lane", "source", "material"]].drop_duplicates().reset_index(drop=True).assign(owner_lane=lambda f: f.index)
    local = physical.merge(owners, on=["lane", "source", "material"], validate="many_to_one")
    if local[["source", "daughter"]].isna().any().any():
        raise ValueError("complete physical ownership labels required")
    daughter = local.operation == "subtract-direct-daughter"
    if not (local.loc[~daughter, "source"] == local.loc[~daughter, "daughter"]).all():
        raise ValueError("own-shape provenance must name its own physical owner")
    if not all(child.startswith(parent + "/") and "/" not in child[len(parent)+1:] for parent, child in
               local.loc[daughter, ["source", "daughter"]].drop_duplicates().itertuples(index=False, name=None)):
        raise ValueError("subtraction must name a direct physical daughter, never a sibling")
    local["lane"] = local.owner_lane
    local_lengths = np.asarray(lengths)[owners.lane.to_numpy(np.int64)]
    # Strict per-owner sweeps refuse sibling overlap or daughter subtraction outside its parent shape.
    partition = boundarypolicy.partition(local, local_lengths)
    occupied = partition[partition.material != "vacuum"].merge(owners[["owner_lane", "lane", "source"]],
        left_on="lane", right_on="owner_lane", suffixes=("_local", ""), validate="many_to_one")
    normalized = pd.concat([occupied[["lane", "s0_mm", "s1_mm", "material", "source"]].assign(sign=1), tube[COLS]], ignore_index=True)
    raw_partition = boundarypolicy.partition(adjusted, lengths)
    owned_partition = boundarypolicy.partition(normalized, lengths)
    edges = pd.concat([raw_partition[["lane", "s0_mm", "s1_mm"]], owned_partition[["lane", "s0_mm", "s1_mm"]]], ignore_index=True)
    atoms = pd.concat([edges[["lane", "s0_mm"]], edges[["lane", "s1_mm"]].rename(columns={"s1_mm": "s0_mm"})]).drop_duplicates().sort_values(["lane", "s0_mm"])
    atoms["s1_mm"] = atoms.groupby("lane").s0_mm.shift(-1)
    atoms = atoms[atoms.s1_mm > atoms.s0_mm].copy()
    for name, table in (("original", raw_partition), ("normalized", owned_partition)):
        table = table.sort_values(["lane", "s0_mm"])
        keys = np.rec.fromarrays([table.lane.to_numpy(np.int64), table.s0_mm.to_numpy()], names="lane,s")
        query = np.rec.fromarrays([atoms.lane.to_numpy(np.int64), atoms.s0_mm.to_numpy()], names="lane,s")
        index = np.searchsorted(keys, query, side="right") - 1
        if ((index < 0).any() or (table.lane.to_numpy()[index] != atoms.lane.to_numpy()).any()
                or (table.s0_mm.to_numpy()[index] > atoms.s0_mm.to_numpy()).any()
                or (atoms.s1_mm.to_numpy() > table.s1_mm.to_numpy()[index]).any()):
            raise ValueError("ownership comparison lacks complete partition coverage")
        atoms[name] = table.material.to_numpy()[index]
    if not (atoms.original == atoms.normalized).all():
        raise ValueError("physical-owner normalization changed stored material labels")
    return normalized.reset_index(drop=True), atoms.reset_index(drop=True)
