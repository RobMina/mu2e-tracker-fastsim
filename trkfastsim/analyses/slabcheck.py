"""slabcheck -- the stage C world transport on one concrete slab, against Geant4 (art/run_slab.sh, g4study ConstructEnvBox).

The fast arm sends mu- at normal incidence through one G4_CONCRETE box (the study geometry has NIST materials only) with the exact stage C functions
(transport.world_loss for the sliced energy loss and the band inputs, transport.msc_to_plane for the correlated
scattering with or without the per-band |p|, the single-scattering tail on or off). The slab is split into `--nbands`
equal bands, as the height bands of the real world. Muons travel along -y (the transport's convention); G4 muons
travel along +z and are rotated onto the same frame. Per (thickness, |p|): survival, exit |p| (median, q10, q90, binned
ratio), exit space angle (q50, q90, q99), lateral displacement at the exit face (q50, q90) and its correlation with the
angle in one transverse plane.

    slabcheck G4_SLAB.root [...] --outdir DIR [--nbands 4] [--nmu 100000]

--model interior: the fast arm is the tracker-interior kink (steps.slab_group: the code path of TrackerInterior.transport
for one support segment: Highland increment on the group's x/X0, lever arm at mid-segment, tail), on a slab of the NIST
material of the G4 run (--material, default: the input's parent directory name, e.g. slab/G4_Cu/). Per point: the
plane-projected exit angle (both planes pooled), width = half the 16-84 % range, q90 and q99 of |angle|, G4/fast ratios,
with and without the calibration (--cal JSON: steps.INTERIOR_MSC_CAL form; default the built-in one). --fit writes the
per-material fit s = c0 + c1 ln(x/X0) + c2 ln(p / 1 GeV) of the width ratio (interior_cal.json).

    slabcheck "slab/G4_Al/slab.*.root" "slab/G4_Cu/slab.*.root" --model interior --fit --outdir DIR

--kink-p-mean uses mean entry/exit momentum for the interior core and tail, keeping the calibration evaluated at
exit momentum as in TrackerInterior.transport. The default remains exit momentum. This is a single unsplit group;
it does not validate support-segment splitting, straw crossings, or production transport. Effective settings are
recorded in slabcheck_interior.json. Reuse the same G4 inputs and seed with separate output directories for each arm.
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
_TAG = re.compile(r"t(\d+(?:\.\d+)?)_p(\d+)")
# (name, per-band |p|, single-scattering tail, tail threshold in theta0 (None = where tail and Gaussian cross), core width factor)
VARIANTS = (("fast", True, True, None, 1.0), ("fast_noslice", False, True, None, 1.0), ("fast_notail", True, False, None, 1.0),
            ("k2", True, True, 2.0, 1.0), ("k2_c96", True, True, 2.0, 0.96), ("k2_c94", True, True, 2.0, 0.94),
            ("k1.7_c94", True, True, 1.7, 0.94), ("k1.7_c92", True, True, 1.7, 0.92))


def _slab_world(thick_mm, half_xz=2.0e5):
    """One CONCRETE_MARS box spanning y in [-thick, 0], as world.load pieces (material, planes [nx, ny, nz, d], lo, hi)."""
    P = np.array([[1, 0, 0, half_xz], [-1, 0, 0, half_xz], [0, 0, 1, half_xz], [0, 0, -1, half_xz],
                  [0, 1, 0, 0.0], [0, -1, 0, thick_mm]], float)
    return [("G4_CONCRETE", P, np.array([-half_xz, -thick_mm, -half_xz]), np.array([half_xz, 0.0, half_xz]))]


def fast(rng, p_mev, thick_mm, n, nbands=4, sliced=True, tail=True, tail_k=None, core_scale=1.0):
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
                                      band_chic2=chic2[s] if tail else None, band_thn=thn[s] if tail else None,
                                      tail_k=tail_k, core_scale=core_scale)
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


def _proj_g4(path):
    """G4 primary end state, projected exit angles [n, 2] (rad) of the muons that left through the exit face, and |p|."""
    pa = T.read(path, "particles")
    m = pa[(pa.isprim == 1) & (pa.pdg.abs() == 13)]
    pe = m[["epx", "epy", "epz"]].to_numpy(float)
    ok = pe[:, 2] > 0
    pe = pe[ok]
    return np.c_[np.arctan2(pe[:, 0], pe[:, 2]), np.arctan2(pe[:, 1], pe[:, 2])], np.linalg.norm(pe, axis=1), int(len(m))


def _wq(th):
    """Width (half the 16-84 % range of both projections pooled) and q90, q99 of |projected angle| [mrad]."""
    a = np.asarray(th, float).ravel() * 1000.0
    q16, q84 = np.quantile(a, [0.16, 0.84])
    q90, q99 = np.quantile(np.abs(a), [0.9, 0.99])
    return [float(0.5 * (q84 - q16)), float(q90), float(q99)]


FIT_MIN_PEXIT = 0.6   # fit range: median exit |p| at least this fraction of the entry |p| (one kink at the exit |p| fails below)


def fit_cal(rows, clip=(0.5, 1.2)):
    """Least squares per material of the width ratio r = G4/fast (uncalibrated) on (1, ln x/X0, ln p/GeV), weights 1/err^2,
    over the points with p_exit / p >= FIT_MIN_PEXIT."""
    from mu2efastsim import eloss
    out, fits = [], {}
    for name in sorted({r["material"] for r in rows}):
        rr = [r for r in rows if r["material"] == name and r["p_exit_frac"] >= FIT_MIN_PEXIT]
        A = np.array([[1.0, np.log(r["xx0"]), np.log(r["p_MeV"] / 1000.0)] for r in rr])
        y = np.array([r["ratio"][0] for r in rr])
        e = np.array([r["ratio_err"] for r in rr])
        c, *_ = np.linalg.lstsq(A / e[:, None], y / e, rcond=None)
        res = y - A @ c
        mat = materials.BY_NAME[name]
        x0g = eloss.radiation_length_mm(mat) * mat.rho / 10.0
        fits[name] = {"X0_g_cm2": x0g, "coef": [float(v) for v in c], "chi2": float(((res / e) ** 2).sum()),
                      "ndf": int(len(y) - 3), "max_abs_resid": float(np.abs(res).max()), "rms_resid": float(np.sqrt((res ** 2).mean()))}
        out.append((x0g, [float(v) for v in c]))
    out.sort()
    return {"points": out, "clip": list(clip)}, fits


def run_interior(inputs, outdir, nmu=30000, seed=1, material=None, cal=None, fit=False, tail_k=None, kink_p_mean=False):
    from .. import steps as st
    calib = st.INTERIOR_MSC_CAL if cal is None else json.load(open(cal))
    rows = []
    for path in T.expand(inputs):
        mt = _TAG.search(os.path.basename(path))
        if not mt:
            continue
        name = material or os.path.basename(os.path.dirname(os.path.abspath(path)))
        mat = materials.BY_NAME[name]
        tg, pm = float(mt.group(1)), int(mt.group(2))
        L = tg / mat.rho * 10.0
        from mu2efastsim import eloss
        xx0 = L / eloss.radiation_length_mm(mat)
        gth, gp, ng = _proj_g4(path)
        if len(gp) < 1000:                      # muons stop in the slab: no width to compare
            print(f"# {name} t{tg} p{pm}: {len(gp)} of {ng} G4 muons leave the slab, skipped")
            continue
        row = {"material": name, "g_cm2": tg, "xx0": float(xx0), "p_MeV": pm, "n_g4": ng, "surv_g4": len(gp) / max(ng, 1)}
        row["g4"] = _wq(gth)
        for lab, cc in (("fast", None), ("fast_cal", calib)):
            if lab == "fast_cal" and cc is None:
                continue
            rng = np.random.default_rng([int(seed), int(round(tg * 1000)), pm])
            ok, pk, dth, _ = st.slab_group(rng, mat, L, np.full(int(nmu), float(pm)), tail=True, tail_k=tail_k,
                                          msc_cal=cc, kink_p_mean=kink_p_mean)
            if ok.sum() < 1000:
                break
            row[lab] = _wq(dth[ok])
            row[lab + "_surv"] = float(ok.mean())
            row["p_exit_frac"] = float(np.median(pk[ok]) / pm)
            row[lab + "_over_g4"] = [f / g for f, g in zip(row[lab], row["g4"])]
        if "fast" not in row:
            print(f"# {name} t{tg} p{pm}: fast muons stop in the slab, skipped")
            continue
        row["ratio"] = [g / f for g, f in zip(row["g4"], row["fast"])]
        # width error: sigma/sqrt(2n) per sample (both projections pooled, n = 2 per muon), both arms
        row["ratio_err"] = float(row["ratio"][0] * np.sqrt(1.0 / (4 * max(len(gp), 1)) + 1.0 / (4 * int(nmu))) * 1.25)
        rows.append(row)
    rows.sort(key=lambda r: (r["material"], r["xx0"], r["p_MeV"]))
    os.makedirs(outdir, exist_ok=True)
    out = {"rows": rows, "settings": {"model": "interior", "kink_p_mean": bool(kink_p_mean),
            "scattering_momentum": "entry_exit_mean" if kink_p_mean else "exit",
            "calibration_momentum": "exit", "segment_model": "single_unsplit_group",
            "nmu": int(nmu), "seed": int(seed), "tail_k": tail_k, "calibration": calib}}
    if fit:
        c, fits = fit_cal(rows)
        out["fit"], out["cal"] = fits, c
        with open(os.path.join(outdir, "interior_cal.json"), "w") as fh:
            json.dump(c, fh, indent=1)
    with open(os.path.join(outdir, "slabcheck_interior.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("material x/X0 p | width q90 q99 G4 [mrad] | G4/fast width q90 q99 | fast_cal/G4 width q90 q99 | surv G4 fast")
    for r in rows:
        fc = " ".join(f"{v:.3f}" for v in r.get("fast_cal_over_g4", [])) or "-"
        print(f"{r['material']:6s} {r['xx0']:5.2f} {r['p_MeV']:5d} | " + " ".join(f"{v:6.2f}" for v in r["g4"])
              + " | " + " ".join(f"{v:.3f}" for v in r["ratio"]) + f" | {fc} | {r['surv_g4']:.3f} {r['fast_surv']:.3f} | pexit {r['p_exit_frac']:.2f}")
    if fit:
        for k, f in out["fit"].items():
            print(f"fit {k}: X0 {f['X0_g_cm2']:.2f} g/cm2, s = {f['coef'][0]:.4f} + {f['coef'][1]:.4f} ln x + {f['coef'][2]:.4f} ln(p/GeV); "
                  f"chi2/ndf {f['chi2']:.1f}/{f['ndf']}, rms resid {f['rms_resid']:.4f}, max {f['max_abs_resid']:.4f}")
    print(f"wrote {os.path.join(outdir, 'slabcheck_interior.json')}")
    return out


def run(inputs, outdir, nbands=4, nmu=100000, seed=1, model="world", material=None, cal=None, fit=False,
        kink_p_mean=False, **kw):
    outdir = outdir or "plots/slabcheck"
    if kink_p_mean and model != "interior":
        raise ValueError("--kink-p-mean requires --model interior")
    if model == "interior":
        return run_interior(inputs, outdir, nmu=nmu, seed=seed, material=material, cal=cal, fit=fit,
                            kink_p_mean=kink_p_mean)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for path in T.expand(inputs):
        mt = _TAG.search(os.path.basename(path))
        if not mt:
            continue
        tg, pm = int(mt.group(1)), int(mt.group(2))   # integer g/cm2 of the concrete runs
        thick = tg / RHO * 10.0
        ga, gp, gd, gx = g4(path)
        gx = _project(gx, gd, thick)
        ref = _summ(ga, gp, gd, gx)
        row = {"g_cm2": tg, "p_MeV": pm, "thick_mm": thick, "g4": ref}
        for name, sl, tl, tk, cs in VARIANTS:
            rng = np.random.default_rng([int(seed), tg, pm])
            ok, p, d, pos = fast(rng, pm, thick, int(nmu), nbands=int(nbands), sliced=sl, tail=tl, tail_k=tk, core_scale=cs)
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
    print("angle quantiles relative to G4 (fast / G4 - 1) for each variant: q50 q90 q99")
    for r in clean:
        g = r["g4"]["angle_mrad_q50_q90_q99"]
        if not g:
            continue
        cells = []
        for name, *_ in VARIANTS:
            f = r[name]["angle_mrad_q50_q90_q99"]
            cells.append(f"{name}: " + " ".join(f"{a / b - 1:+.2f}" for a, b in zip(f, g)) if f else f"{name}: -")
        print(f"{r['g_cm2']:>5} {r['p_MeV']:>5} | " + " | ".join(cells))
    for r in clean:
        g, f, ns, nt = r["g4"], r["fast"], r["fast_noslice"], r["fast_notail"]
        fa = lambda x: "/".join(f"{v:.0f}" for v in x.get("angle_mrad_q50_q90_q99", [])) or "-"
        print(f"{r['g_cm2']:>5} {r['p_MeV']:>5} | {g['survival']:>7.3f} {f['survival']:>6.3f} | "
              f"{(g['p_q10_q50_q90'] or [0, 0])[1]:>7.0f} {(f['p_q10_q50_q90'] or [0, 0])[1]:>7.0f} | "
              f"{fa(g):>26} {fa(f):>20} {fa(ns):>20} {fa(nt):>20} | "
              f"{(g['lateral_mm_q50_q90'] or [0, 0])[1]:>10.1f} {(f['lateral_mm_q50_q90'] or [0, 0])[1]:>6.1f}")
    print(f"wrote {os.path.join(outdir, 'slabcheck.json')}")
