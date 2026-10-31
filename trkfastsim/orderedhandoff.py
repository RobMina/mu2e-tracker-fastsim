"""Explicit forward-only computational world/interior split, not exact GDML ownership."""
import numpy as np
import pandas as pd

from . import crossings, geometry, ledger

POLICY = "raised-plane-forward-v1"


def plan(geom, interior, census, *, clearance_mm):
    """Put the split above all known tracker/calo bounds and loaded support envelopes."""
    if not np.isfinite(clearance_mm) or clearance_mm <= 0:
        raise ValueError("explicit positive finite handoff clearance required")
    if census.get("schema") != "tracker-navigation-census-v1" or census.get("placement_warnings"):
        raise ValueError("validated placement census required for forward handoff")
    records = [r for r in census["records"] if r["scope"] in ("tracker", "calorimeter")
               and r["classification"] != "intentional_exclusion"]
    if any(r.get("geometry_bounds_mm") is None or r.get("placement_assumptions") for r in records):
        raise ValueError("unknown tracker/calo bound prevents forward handoff plane")
    def box(value):
        value = np.asarray(value, float)
        if value.shape != (2, 3) or not np.isfinite(value).all() or (value[1] < value[0]).any():
            raise ValueError("finite ordered 2x3 handoff bounds required")
        return value
    bounds = []
    if records:
        bounds.append(max(float(box(r["geometry_bounds_mm"])[1, 1]) for r in records))
    shells = interior.shells
    center = np.asarray(shells.c, float)
    radii = np.asarray([g["rmax"] for g in shells.groups], float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radii).all() or (radii < 0).any():
        raise ValueError("finite stored support center/radii required")
    if len(radii):
        bounds.append(float(radii.max() + center[1]))
    other = getattr(shells, "other", [])
    if other:
        rotation = np.asarray([tube[1] for tube in other])
        center = np.asarray([tube[2] for tube in other])
        radii = np.asarray([tube[4] for tube in other])
        half = np.asarray([tube[5] for tube in other])
        tops = center[:, 1] + np.linalg.norm(rotation[:, 1, :2], axis=1) * radii + np.abs(rotation[:, 1, 2]) * half
        if not np.isfinite(tops).all():
            raise ValueError("nonfinite stored support envelope prevents handoff")
        bounds.append(float(tops.max()))
    if getattr(shells, "calo", None) is not None:
        bounds.append(float(box([shells.calo_lo, shells.calo_hi])[1, 1]))
    lo, hi = crossings.panel_boxes(geom)
    if lo.ndim != 2 or lo.shape[1:] != (3,) or hi.shape != lo.shape or not len(lo) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
        raise ValueError("finite ordered straw panel bounds required")
    bounds.append(float(hi[:, 1].max()))
    top = max(bounds)
    plane = float(np.nextafter(top + clearance_mm, np.inf))
    if not np.isfinite(plane) or plane <= top:
        raise ValueError("handoff plane must be representably above all support bounds")
    return dict(policy=POLICY, plane_y_mm=plane, known_interior_top_y_mm=top, clearance_mm=float(clearance_mm),
        tracker_calo_bound_records=len(records), approximate_records=sum(r["classification"] == "approximate" for r in records),
        world_domain="generated-ray queries y>=plane; aggregated stored world model unchanged",
        interior_domain="actual forward rays y<=plane; upward re-crossing refuses, no backprojection",
        unresolved=["upstream world omissions/Boolean ownership", "retained interior shape approximations",
                    "aggregate MSC endpoint is not an actual world path", "finite interior arc is not world exit"],
        admitted_for_physics=False)


def validate_world(world, plane):
    from .steps import BAND_EDGES_Y
    if world is None or not hasattr(world, "edges") or len(world.edges) < 2:
        raise ValueError("forward handoff requires a world with explicit band edges")
    edges = np.asarray(world.edges, float)
    expected = np.r_[BAND_EDGES_Y[BAND_EDGES_Y > plane], plane]
    if not np.isfinite(edges).all() or not (np.diff(edges) < 0).all() or not np.array_equal(edges, expected):
        raise ValueError("world endpoint or internal bands differ from forward handoff")


def validate_forward_query(origin, direction, lengths, plane):
    """Refuse a requested forward leg entering the upstream domain, never clip a leg."""
    end = origin + direction * lengths[:, None]
    guard = 64 * np.finfo(float).eps * np.maximum(1., np.maximum(np.abs(origin[:, 1]), abs(plane)))
    if ((origin[:, 1] > plane + guard) | (end[:, 1] > plane + guard)).any():
        raise ValueError("forward interior query crosses world handoff plane; no continuation")


def capture(mu, origin, direction, momentum, offset, alive, world_time, *, plane):
    """Retain every primary identity and disposition before the interior stage."""
    from .steps import KEY
    origin, direction, momentum = np.asarray(origin), np.asarray(direction), np.asarray(momentum)
    alive, offset, world_time = np.asarray(alive, bool), np.asarray(offset, float), np.asarray(world_time, float)
    if alive.shape != (len(mu),) or offset.shape != (len(mu),) or world_time.shape != (len(mu),):
        raise ValueError("complete one-dimensional handoff disposition/time/path arrays required")
    if (origin.shape != (len(mu), 3) or direction.shape != origin.shape or momentum.shape != (len(mu),)
            or not np.isfinite(origin).all() or not np.isfinite(direction).all() or not np.isfinite(momentum).all()):
        raise ValueError("finite complete handoff states required")
    unit = np.isclose(np.linalg.norm(direction, axis=1), 1., rtol=0, atol=1e-10)
    residual = origin[:, 1] - plane
    plane_guard = 64 * np.finfo(float).eps * np.maximum(1., np.maximum(np.abs(origin[:, 1]), abs(plane)))
    accepted = alive & unit & (direction[:, 1] < 0) & (np.abs(residual) <= plane_guard)
    accepted &= np.isfinite(world_time) & (world_time >= 0) & (momentum > 0) & np.isfinite(offset) & (offset >= 0)
    out = mu[KEY].copy().assign(sp=mu.mukey.to_numpy(), alive_world=alive, accepted_forward=accepted,
        p_handoff_MeV_c=momentum, path_offset_mm=offset, world_dt_ns=world_time,
        y_plane_residual_mm=residual, plane_roundoff_guard_mm=plane_guard)
    if out.duplicated(KEY + ["sp"]).any():
        raise ValueError("duplicate full handoff primary identity")
    for j, axis in enumerate("xyz"):
        out[axis], out["u" + axis] = origin[:, j], direction[:, j]
    out["disposition"] = np.where(~alive, "not-delivered-by-world", np.where(accepted, "forward-handoff", "refused-handoff"))
    return out
