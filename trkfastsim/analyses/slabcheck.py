"""slabcheck -- the stage C world transport on one concrete slab, against Geant4 (art/run_slab.sh, g4study ConstructEnvBox).

The fast arm sends mu- at normal incidence through one G4_CONCRETE box (the study geometry has NIST materials only) with the exact stage C functions
(transport.world_loss for the sliced energy loss and the band inputs, transport.msc_to_plane for the correlated
scattering with or without the per-band |p|, the single-scattering tail on or off). The slab is split into `--nbands`
equal bands, as the height bands of the real world. Muons travel along -y (the transport's convention); G4 muons
travel along +z and are rotated onto the same frame. Per (thickness, |p|): survival, exit |p| (median, q10, q90, binned
ratio), exit space angle (q50, q90, q99), lateral displacement at the exit face (q50, q90) and its correlation with the
angle in one transverse plane.

    slabcheck G4_SLAB.root [...] --outdir DIR [--nbands 4] [--nmu 100000]
"""
import json
import os
import re

import numpy as np

from mu2efastsim import eloss, materials, stats
from mu2efastsim import transport as tr
from .. import tables as T

# the G4 study geometry gets NIST materials only, so the slab is G4_CONCRETE (G4NistMaterialBuilder mass fractions)
G4_CONCRETE = materials.Material("G4_CONCRETE", 2.3, 135.2, {"H": 0.01, "C": 0.001, "O": 0.529107, "Na": 0.016, "Mg": 0.002,
                                                       "Al": 0.033872, "Si": 0.337021, "K": 0.013, "Ca": 0.044, "Fe": 0.014})
RHO = G4_CONCRETE.rho
_TAG = re.compile(r"t(\d+)_p(\d+)")


def _slab_world(thick_mm, half_xz=2.0e5):
    """One CONCRETE_MARS box spanning y in [-thick, 0], as world.load pieces (material, planes [nx, ny, nz, d], lo, hi)."""
    P = np.array([[1, 0, 0, half_xz], [-1, 0, 0, half_xz], [0, 0, 1, half_xz], [0, 0, -1, half_xz],
                  [0, 1, 0, 0.0], [0, -1, 0, thick_mm]], float)
    return [("G4_CONCRETE", P, np.array([-half_xz, -thick_mm, -half_xz]), np.array([half_xz, 0.0, half_xz]))]


def fast(rng, p_mev, thick_mm, n, nbands=4, sliced=True, tail=True):
    """Exit state of n muons: alive, |p|, direction [n,3], position on the exit face y = -thick [n,3]."""
    materials.BY_NAME.setdefault("G4_CONCRETE", G4_CONCRETE)
    o = np.tile([0.0, 1e-3, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    p = np.full(n, float(p_mev))
    edges = np.linspace(0.0, -thick_mm, nbands + 1)
    edges[0] = 1e-3
    res = tr.world_loss(rng, o, d, p, _slab_world(thick_mm), edges, return_band_p=True, return_tail=tail)
    pw, xx0, ok, bxx0, bh, bp = res[:6]
    chic2, thn = (res[6], res[7]) if tail else (None, None)
    # as stage C (to_plane): only muons that survive the column are scattered
    s = np.flatnonzero(ok)
    p_msc = np.maximum(np.sqrt(p[s] * np.maximum(pw[s], 1.0)), 1.0)
    on, dn = o.copy(), d.copy()
    on[s], dn[s], _ = tr.msc_to_plane(rng, o[s], d[s], p_msc, -thick_mm, eloss.radiation_length_mm(materials.AIR), xx0[s],
                                      bxx0[s], bh[s], band_p=bp[s] if sliced else None,
                                      band_chic2=chic2[s] if tail else None, band_thn=thn[s] if tail else None)
    return ok, pw, dn, on


def g4(path):
    """Primary muon end state from the slab dump: alive (left through the exit face), |p|, direction, position, in the
    fast frame (G4 +z -> -y, G4 x -> x, G4 y -> z), positions relative to the gun point (1 um before the entry face)."""
    pa = T.read(path, "particles")
    m = pa[(pa.isprim == 1) & (pa.pdg.abs() == 13)]
    pe = m[["epx", "epy", "epz"]].to_numpy(float)
    xe = m[["ex", "ey", "ez"]].to_numpy(float)
    p = np.linalg.norm(pe, axis=1)
    alive = pe[:, 2] > 0                                    # still moving forward when it left the world
    rot = lambda v: np.c_[v[:, 0], -v[:, 2], v[:, 1]]
    x0 = m[["sx", "sy", "sz"]].to_numpy(float)
    return alive, p, rot(pe) / np.maximum(p, 1e-30)[:, None], rot(xe - x0)   # entry face (gun point) at y = 0


def _project(pos, d, thick_mm):
    """Point where each track crosses the exit face y = -thick (G4 end points lie beyond it, in vacuum)."""
    s = (-thick_mm - pos[:, 1]) / np.where(d[:, 1] == 0, -1e-30, d[:, 1])
    return pos + s[:, None] * d


def _summ(alive, p, d, pos):
    a = alive & (p > 0)
    ang = np.arccos(np.clip(-d[a, 1], -1, 1)) * 1000.0                         # mrad from the -y axis
    lat = np.hypot(pos[a, 0], pos[a, 2])
    tx = np.arctan2(d[a, 0], -d[a, 1])
    rho = float(np.corrcoef(pos[a, 0], tx)[0, 1]) if a.sum() > 10 else float("nan")
    return {"survival": float(a.mean()), "n_alive": int(a.sum()),
            "p_q10_q50_q90": [float(x) for x in np.quantile(p[a], [0.1, 0.5, 0.9])] if a.any() else [],
            "angle_mrad_q50_q90_q99": [float(x) for x in np.quantile(ang, [0.5, 0.9, 0.99])] if a.any() else [],
            "lateral_mm_q50_q90": [float(x) for x in np.quantile(lat, [0.5, 0.9])] if a.any() else [],
            "rho_x_thetax": rho, "_p": p[a], "_ang": ang, "_lat": lat}


def run(inputs, outdir, nbands=4, nmu=100000, seed=1, **kw):
    outdir = outdir or "plots/slabcheck"
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for path in T.expand(inputs):
        mt = _TAG.search(os.path.basename(path))
        if not mt:
            continue
        tg, pm = int(mt.group(1)), int(mt.group(2))
        thick = tg / RHO * 10.0
        ga, gp, gd, gx = g4(path)
        gx = _project(gx, gd, thick)
        ref = _summ(ga, gp, gd, gx)
        row = {"g_cm2": tg, "p_MeV": pm, "thick_mm": thick, "g4": ref}
        for name, sl, tl in (("fast", True, True), ("fast_noslice", False, True), ("fast_notail", True, False)):
            rng = np.random.default_rng([int(seed), tg, pm])
            ok, p, d, pos = fast(rng, pm, thick, int(nmu), nbands=int(nbands), sliced=sl, tail=tl)
            f = _summ(ok, p, d, pos)
            for k in ("_p", "_ang", "_lat"):
                if len(f[k]) > 100 and len(ref[k]) > 100:
                    f[k[1:] + "_band"] = stats.ratio_band(f[k], ref[k], min_ref=100)["max_abs_dev_minus_2sigma"]
            row[name] = f
        rows.append(row)
    rows.sort(key=lambda r: (r["g_cm2"], r["p_MeV"]))
    clean = [{k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
              for k, v in r.items()} for r in rows]
    with open(os.path.join(outdir, "slabcheck.json"), "w") as fh:
        json.dump(clean, fh, indent=1)
    print(f"{'g/cm2':>5} {'p':>5} | {'surv G4':>7} {'fast':>6} | {'p50 G4':>7} {'fast':>7} | "
          f"{'ang q50/q90/q99 G4 [mrad]':>26} {'fast':>20} {'noslice':>20} {'notail':>20} | {'lat q90 G4':>10} {'fast':>6}")
    for r in clean:
        g, f, ns, nt = r["g4"], r["fast"], r["fast_noslice"], r["fast_notail"]
        fa = lambda x: "/".join(f"{v:.0f}" for v in x.get("angle_mrad_q50_q90_q99", [])) or "-"
        print(f"{r['g_cm2']:>5} {r['p_MeV']:>5} | {g['survival']:>7.3f} {f['survival']:>6.3f} | "
              f"{(g['p_q10_q50_q90'] or [0, 0])[1]:>7.0f} {(f['p_q10_q50_q90'] or [0, 0])[1]:>7.0f} | "
              f"{fa(g):>26} {fa(f):>20} {fa(ns):>20} {fa(nt):>20} | "
              f"{(g['lateral_mm_q50_q90'] or [0, 0])[1]:>10.1f} {(f['lateral_mm_q50_q90'] or [0, 0])[1]:>6.1f}")
    print(f"wrote {os.path.join(outdir, 'slabcheck.json')}")
