"""Field-on transport from the cosmic stage-2 start states to the tracker (NumPy).

The Mu2e cosmic chain splits Geant4 in two: stage 1 (CRY to the DS) stops every particle on the outer surface of the DS
cryostat (r = 1328 mm about the DS axis, and the TS-DS end at z = 2929 mm), and stage 2 restarts them there in the
full detector. This module replaces stage 2 for the primary muon up to the tracker: from that boundary through the DS
cryostat (shells, thermal shields, coils, end walls), the DS vacuum and everything in it (target, absorbers, supports,
calorimeter, MBS, cable runs) to the tracker mother volume, then `fieldsteps.transport` inside the tracker.

  * Material: every non-vacuum volume of the GDML inside the DS region, as signed convex pieces
    (mu2efastsim.world.signed_subtree, cones included), without the tracker mother volume and its daughters; materials
    without core constants come from the GDML (mu2efastsim.materials.from_gdml). The opt-in core parser represents
    cones and polyhedra; unsupported solids such as tori (calorimeter cooling pipes) are listed in `skipped`.
  * Field: the Offline map stack, first inner map holding the point, else the outer map (BFieldManager), trilinear,
    with the y reflection of maps marked `extendY` (WorldMap).
  * Stepping, loss and scattering: those of fieldsteps.transport (chords of the local helix, muon slice sampler,
    Highland kink per chord with the single-scattering tail), without straws.
  * Hand-off: the closed cylinder of the tracker mother (r <= R_TRACKER, Z_FRONT <= z <= Z_BACK), the surfaces on which
    the Geant4 sample records the tracker entry (TT_FrontHollow / TT_Front, TT_Back, TT_Outer).

This is an unvalidated development prototype; its delivery pilot failed, so it is not a validated world replacement.
"""
import re

import numpy as np

from mu2efastsim import ehelix, materials, world as world_mod

DS_AXIS_XY = (-3904.0, 0.0)
R_START = 1328.0                     # DSOuterCryoShell rmax: the stage-1 stopping surface
R_TRACKER = 850.12                   # TrackerMother rmax (TT_Outer SurfaceStep radius)
Z_FRONT, Z_BACK = 8539.88, 11810.12  # TT_FrontHollow (VD 13) and TT_Back (VD 15) planes
REGION_R, REGION_Z = 1335.0, (2850.0, 17600.0)


class MapStack:
    """Callable B(x) [T] as Offline BFieldManager: the first inner map whose grid holds the point, else the first outer
    map that does, else zero. Maps are G4BL headers (+ .bin); `extendY` maps hold y >= 0 and are reflected (By flips)."""

    def __init__(self, inner, outer=()):
        self.maps = [(self._load(h), False) for h in inner] + [(self._load(h), True) for h in outer]

    @staticmethod
    def _load(header):
        lo, d, n, grid = ehelix.read_g4bl(header)
        flip = bool(re.search(r"^extendY", open(header).read(), re.M))
        return lo, d, n, grid, flip

    def __call__(self, x):
        x = np.asarray(x, float)
        out = np.zeros(x.shape)
        todo = np.ones(len(x), bool)
        for (lo, d, n, grid, flip), _outer in self.maps:
            if not todo.any():
                break
            xx = x[todo].copy()
            neg = xx[:, 1] < 0
            if flip:
                xx[:, 1] = np.abs(xx[:, 1])
            u = (xx - lo) / d
            i = np.floor(u).astype(int)
            ok = np.all((i >= 0) & (i < n - 1), axis=1)
            if not ok.any():
                continue
            ii = np.clip(i[ok], 0, n - 2)
            f = u[ok] - ii
            b = np.zeros((ok.sum(), 3))
            for dx in (0, 1):
                wx = f[:, 0] if dx else 1 - f[:, 0]
                for dy in (0, 1):
                    wy = f[:, 1] if dy else 1 - f[:, 1]
                    for dz in (0, 1):
                        wz = f[:, 2] if dz else 1 - f[:, 2]
                        b += (wx * wy * wz)[:, None] * grid[ii[:, 0] + dx, ii[:, 1] + dy, ii[:, 2] + dz]
            if flip:
                b[neg[ok], 1] *= -1.0
            idx = np.flatnonzero(todo)[ok]
            out[idx] = b
            todo[idx] = False
        return out


def map_stack(maps_dir, inner=("DSMap", "TSdMap", "DSExtension"), outer=("WorldMap",)):
    """MapStack of the Mau13 maps in maps_dir (bfgeom_no_tsu_ps_v01: DS, TSd, DS extension inner; world outer)."""
    import os
    return MapStack([os.path.join(maps_dir, f"{m}.header") for m in inner],
                    [os.path.join(maps_dir, f"{m}.header") for m in outer])


class Region:
    """The transport region about the DS axis: tracks leaving it (outwards) stop. Envelope-like (axis, z0, z1,
    leaving)."""

    def __init__(self, r=REGION_R, z=REGION_Z, axis_xy=DS_AXIS_XY):
        self.axis = np.asarray(axis_xy, float)
        self.r, (self.z0, self.z1) = float(r), (float(z[0]), float(z[1]))

    def leaving(self, x, u):
        rv = x[:, :2] - self.axis
        r = np.hypot(rv[:, 0], rv[:, 1])
        ur = (rv * u[:, :2]).sum(1)
        return ((r > self.r) & (ur > 0)) | ((x[:, 2] < self.z0) & (u[:, 2] < 0)) | ((x[:, 2] > self.z1) & (u[:, 2] > 0))


class Handoff:
    """Closed cylinder (r <= rmax about the axis, z0 <= z <= z1): cut(x0, x1) gives the fraction of the chord x0 -> x1
    at which it enters (NaN when x0 is inside or the chord misses it)."""

    def __init__(self, rmax=R_TRACKER, z0=Z_FRONT, z1=Z_BACK, axis_xy=DS_AXIS_XY):
        self.axis = np.asarray(axis_xy, float)
        self.rmax, self.z0, self.z1 = float(rmax), float(z0), float(z1)

    def inside(self, x, tol=1e-6):
        r = np.hypot(x[:, 0] - self.axis[0], x[:, 1] - self.axis[1])
        return (r <= self.rmax + tol) & (x[:, 2] >= self.z0 - tol) & (x[:, 2] <= self.z1 + tol)

    def cut(self, x0, x1):
        d = x1 - x0
        o = x0[:, :2] - self.axis
        A = d[:, 0] ** 2 + d[:, 1] ** 2
        B = 2 * (o * d[:, :2]).sum(1)
        C = (o * o).sum(1) - self.rmax ** 2
        disc = B * B - 4 * A * C
        s = np.full(len(x0), np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            sq = np.sqrt(np.where(disc > 0, disc, np.nan))
            s_side = (-B - sq) / (2 * A)
            z_side = x0[:, 2] + s_side * d[:, 2]
            ok = (A > 0) & (s_side >= 0) & (s_side <= 1) & (z_side >= self.z0) & (z_side <= self.z1)
            s = np.where(ok, s_side, s)
            for zp in (self.z0, self.z1):
                sp = (zp - x0[:, 2]) / d[:, 2]
                pp = x0[:, :2] + sp[:, None] * d[:, :2] - self.axis
                okp = (d[:, 2] != 0) & (sp >= 0) & (sp <= 1) & ((pp * pp).sum(1) <= self.rmax ** 2)
                s = np.where(okp & (sp < s), sp, s)
        s = np.where(self.inside(x0) | ~np.isfinite(s), np.nan, s)
        return s

    def face(self, x, tol=0.05):
        """Entry face of points on the surface: 0 front, 2 back, 4 outer (the SurfaceId numbers), -1 otherwise."""
        r = np.hypot(x[:, 0] - self.axis[0], x[:, 1] - self.axis[1])
        return np.where(np.abs(x[:, 2] - self.z0) < tol, 0, np.where(np.abs(x[:, 2] - self.z1) < tol, 2,
                        np.where(np.abs(r - self.rmax) < tol, 4, -1)))


class WorldSupport:
    """The DS-region material as a fieldsteps interior: _support(x, c, L, moments) -> (lengths, moments) per material,
    and `materials` for the names without core constants."""

    def __init__(self, gdml, region=None, skip=("TrackerMother",), cell=250.0):
        region = Region() if region is None else region
        lo = np.array([region.axis[0] - region.r, region.axis[1] - region.r, region.z0])
        hi = np.array([region.axis[0] + region.r, region.axis[1] + region.r, region.z1])
        pieces, self.skipped = world_mod.signed_subtree(gdml, "HallAir", bbox=(lo, hi), skip=skip, cones=True)
        vac = re.compile(r"AIR|Vacuum|VACUUM|Galactic")
        pieces = [q for q in pieces if not vac.search(q[1])]
        self.pieces = world_mod.SignedPieceSet(pieces, cell=cell)
        names = sorted({q[1] for q in pieces})
        self.materials = {k: v for k, v in materials.from_gdml(gdml, names).items() if k not in materials.BY_NAME}
        self.material_names = names

    def _support(self, x, c, L, moments):
        col = self.pieces.column(x, c, L, moments=moments)
        if moments:
            mom = {m: v for m, v in col.items() if np.asarray(v[0]).any()}
            return {m: v[0] for m, v in mom.items()}, mom
        return {m: v for m, v in col.items() if np.asarray(v).any()}, {}
