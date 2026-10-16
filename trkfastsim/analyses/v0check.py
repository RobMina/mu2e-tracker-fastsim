"""v0check -- T-V0 straw geometry from the GDML against the Geant4 reference steps (no extra G4 job).

Reference step positions are in the detector frame and are moved to the Mu2e frame (geometry.to_mu2e) first.
Every G4 StrawGasStep of the reference was produced inside the gas volume of its StrawId. For each step this checks,
with the straws built by geometry.from_gdml: the entry and exit points lie within the straw radius of the straw axis
and within its half length, and (for primary-muon steps, which are straight) both lie on the cylinder surface. A wrong
StrawId mapping, rotation convention or straw position shows up as distances far beyond the radius.
"""
import numpy as np

from mu2efastsim import plots
from .. import geometry, tables as T


def _axis_dist(geom, js, x):
    w = x - geom.mid[js]
    a = geom.dir[js]
    wa = np.einsum("ij,ij->i", w, a)
    return np.linalg.norm(w - wa[:, None] * a, axis=1), np.abs(wa)


def run(inputs, outdir, gdml=None, **kw):
    from .steps import default_gdml
    g = geometry.from_gdml(gdml or default_gdml())
    s = T.read_many(inputs, "sgsteps")
    p = T.read_many(inputs, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="left")
    js = (s.upanel.to_numpy() * geometry.NSTRAWS + s.straw.to_numpy()).astype(np.int64)
    if not np.array_equal(g.sid[js], s.sid.to_numpy()):
        raise SystemExit("StrawId <-> straw index mapping disagrees with the reference sid column")
    out = {"straws": int(g.nstraws), "radius_mm": g.radius, "steps": int(len(s))}
    for end in ("0", "1"):
        x = geometry.to_mu2e(s[[f"x{end}", f"y{end}", f"z{end}"]].to_numpy())
        r, z = _axis_dist(g, js, x)
        out[f"end{end}"] = {"r_max_mm": float(r.max()), "r_over_radius_p999": float(np.quantile(r, 0.999) / g.radius),
                            "frac_r_gt_radius_plus_1um": float((r > g.radius + 1e-3).mean()),
                            "frac_beyond_halflen_plus_1um": float((z > g.halflen[js] + 1e-3).mean())}
    mu = (s.ismcprim == 1) & (s.pdg.abs() == 13) & (s["shape"] == 0)
    r0, _ = _axis_dist(g, js[mu], geometry.to_mu2e(s.loc[mu, ["x0", "y0", "z0"]].to_numpy()))
    r1, _ = _axis_dist(g, js[mu], geometry.to_mu2e(s.loc[mu, ["x1", "y1", "z1"]].to_numpy()))
    out["primary_muon_surface"] = {"steps": int(mu.sum()),
                                   "entry_r_minus_radius_um_p01_p50_p99": (1e3 * np.quantile(r0 - g.radius, [0.01, 0.5, 0.99])).tolist(),
                                   "exit_r_minus_radius_um_p01_p50_p99": (1e3 * np.quantile(r1 - g.radius, [0.01, 0.5, 0.99])).tolist()}
    tol = 1e-3   # mm: entry/exit on the inner wall surface within 1 um
    out["pass"] = bool(all(out[f"end{e}"]["frac_r_gt_radius_plus_1um"] == 0 and
                           out[f"end{e}"]["frac_beyond_halflen_plus_1um"] == 0 for e in "01") and
                       abs(np.median(r0 - g.radius)) < tol)
    for k, v in out.items():
        print(f"{k:24s} {v}")
    plots.write_json(outdir, "v0check.json", out)
