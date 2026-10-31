"""Exact dyadic affine predicates on frozen binary64 planar operands; not GDML truth."""
from fractions import Fraction
import hashlib

import numpy as np
import pandas as pd

from . import csgenvelope, materialpath
from mu2efastsim import world

POLICY = "stored-binary64-exact-dyadic-planar-predicates-v1"
_FRACTION = np.frompyfunc(Fraction, 2, 1)
_NUMERATOR = np.frompyfunc(lambda x: str(x.numerator), 1, 1)
_DENOMINATOR = np.frompyfunc(lambda x: str(x.denominator), 1, 1)


def rational(values):
    """Decode finite binary64 arrays exactly, with array-wide integer arithmetic."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("finite binary64 rational operands required")
    mantissa, exponent = np.frexp(values)
    numerator = (mantissa * 2.**53).astype(np.int64).astype(object)
    exponent = exponent.astype(np.int64) - 53
    numerator = np.left_shift(numerator, np.maximum(exponent, 0).astype(object))
    denominator = np.left_shift(np.ones(values.shape, dtype=object), np.maximum(-exponent, 0).astype(object))
    return _FRACTION(numerator, denominator)


def export(values, upward):
    """Directed binary64 projection of rational roots, never equality by a tolerance."""
    values = np.asarray(values, dtype=object)
    result = values.astype(float)
    if not np.isfinite(result).all():
        raise ValueError("rational endpoint export overflow")
    rounded = rational(result)
    wrong = rounded < values if upward else rounded > values
    return np.where(wrong, np.nextafter(result, np.inf if upward else -np.inf), result)


def serialize_roots(rows):
    """Retain exact endpoints as decimal numerator/denominator text and directed float brackets."""
    result = rows.copy()
    for name in ("q0", "q1"):
        values = result.pop(name).to_numpy(object)
        result[name + "_numerator"] = _NUMERATOR(values).astype(str)
        result[name + "_denominator"] = _DENOMINATOR(values).astype(str)
        result[name + "_lower_mm"] = export(values, False)
        result[name + "_upper_mm"] = export(values, True)
    return result


def _validate_xtru(solid, solids, reviewed):
    if solid.tag == "xtru":
        reviewed._xtru_pieces(solid)
    elif solid.tag in ("subtraction", "union", "intersection"):
        for tag in ("first", "second"):
            _validate_xtru(solids[solid.find(tag).get("ref")], solids, reviewed)


def intervals(solid, solids, placement, origin, direction, lengths, *, reviewed, max_work=2000000):
    """Evaluate exact open-atom signed occupancy; preserve positive-width multi-occupancy faults.

    The model is local binary64 planes with exact real (o-t)@R and d@R from
    stored binary64 R,t and world rays. No inverse or rounded local ray is used.
    Boolean pieces/triangulation are explicitly the stored core representation.
    """
    if isinstance(max_work, (bool, np.bool_)) or not isinstance(max_work, (int, np.integer)) or max_work <= 0:
        raise ValueError("positive integer exact predicate work cap required")
    if placement["assumptions"] or not placement["is_leaf"]:
        raise ValueError("exact leaf predicates require verified leaf placement")
    csgenvelope.construction_bound(solid, solids, max_work=max_work)
    csgenvelope.supported(solid, solids)
    _validate_xtru(solid, solids, reviewed)
    origin, direction, lengths = materialpath._rays(origin, direction, lengths)
    pieces = world.signed_local(solid, solids)
    if pieces is None or sum(len(p[1]) for p in pieces) * len(origin) > max_work:
        raise ValueError("exact planar plane/ray work cap")
    rotation, position = np.asarray(placement["rotation"], float), np.asarray(placement["position"], float)
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("stored transform shape")
    h = hashlib.sha256(rotation.tobytes() + position.tobytes())
    qr, qt = rational(rotation), rational(position)
    local_o = (rational(origin) - qt) @ qr
    local_d = rational(direction) @ qr
    end = rational(lengths)
    rows, coplanar = [], []
    for piece_id, (sign, planes, _, _) in enumerate(pieces):
        planes = np.asarray(planes, float)
        if sign not in (-1, 1) or planes.ndim != 2 or planes.shape[1] != 4 or not len(planes):
            raise ValueError("signed finite planar representation required")
        if (np.linalg.norm(planes[:, :3], axis=1) == 0).any():
            raise ValueError("nonzero exact plane normal required")
        h.update(np.asarray([sign], np.int64).tobytes())
        h.update(planes.tobytes())
        qp = rational(planes)
        den = local_d @ qp[:, :3].T
        num = qp[:, 3] - local_o @ qp[:, :3].T
        root = num / np.where(den == 0, Fraction(1), den)
        lo = np.maximum(Fraction(0), np.where(den < 0, root, Fraction(0)).max(axis=1))
        hi = np.minimum(end, np.where(den > 0, root, end[:, None]).min(axis=1))
        valid = ~((den == 0) & (num < 0)).any(axis=1) & (hi > lo)
        contact = valid & ((den == 0) & (num == 0)).any(axis=1)
        lanes = np.flatnonzero(valid)
        rows.append(pd.DataFrame(dict(lane=lanes, q0=lo[lanes], q1=hi[lanes], sign=sign, piece=piece_id)))
        coplanar.append(np.flatnonzero(contact))
    raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lane", "q0", "q1", "sign", "piece"])
    raw = raw.astype({"lane": "int64", "sign": "int64", "piece": "int64"})
    if len(raw):
        events = pd.concat([raw[["lane", "q0"]].rename(columns={"q0": "q"}).assign(delta=raw.sign),
                            raw[["lane", "q1"]].rename(columns={"q1": "q"}).assign(delta=-raw.sign)], ignore_index=True)
        events = events.groupby(["lane", "q"], as_index=False).delta.sum().sort_values(["lane", "q"])
        events["occupancy"] = events.groupby("lane").delta.cumsum()
        events["q1"] = events.groupby("lane").q.shift(-1)
        atoms = events[(events.q1.notna()) & (events.q1 > events.q)].rename(columns={"q": "q0"})
        faults = atoms[(atoms.occupancy < 0) | (atoms.occupancy > 1)].copy()
        occupied = atoms[atoms.occupancy == 1][["lane", "q0", "q1"]].reset_index(drop=True)
    else:
        occupied = pd.DataFrame({"lane": pd.Series(dtype="int64"), "q0": pd.Series(dtype=object), "q1": pd.Series(dtype=object)})
        faults = occupied.assign(occupancy=pd.Series(dtype="int64"))
    contact_lanes = np.unique(np.concatenate(coplanar)) if coplanar else np.array([], dtype=np.int64)
    return occupied, faults, dict(policy=POLICY, lanes=len(lengths), pieces=len(pieces),
        plane_ray_work=sum(len(p[1]) for p in pieces) * len(origin), max_work=max_work,
        operands_sha256=h.hexdigest(), coplanar_lanes=contact_lanes.tolist(),
        invalid_occupancy_atoms=len(faults), occupied_atoms=len(occupied), capped=False,
        model="exact rational affine predicates on frozen local binary64 planes, accumulated R,t and world rays")
