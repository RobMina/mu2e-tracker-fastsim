"""libplacecheck -- placement of non-muon StrawGasSteps inside their straws, fast (one or more arms) vs Geant4.

Events: the v1compare selection (primary muon passes the PrimaryFilter tracker term), muon = MC primary by identity.
Non-muon steps are split into those in a muon straw and off-axis ones (any other straw). Per class, in the frame of
library format v2 (library.straw_frame, same code as the library build):
  muon straw  |along-wire offset| of each end from the muon chord's point nearest the wire; distance from the wire
              (segment minimum and start); share of shared straws where a secondary's first cluster is nearer the wire
              than the muon (the StrawDigiMC attribution proxy, secdiag's rclu rule);
  off-axis    |along-wire offset| from where the anchor muon step's line (nearest in plane, then time) passes closest to
              the straw axis; |position along the wire| from the straw centre; distance from the wire; the plane-offset
              profile (steps per event by dpl) with chi2/ndf against G4.

    libplacecheck FAST_V2 [...] --compare G4_SGSTEPS [...] [--fast-ref FAST_V1 ...] [--gdml G] --outdir DIR
"""
import numpy as np
import pandas as pd

from mu2efastsim import plots
from .. import geometry, library as L, tables as T
from .secdiag import POINT_LEN
from .v1compare import _all_steps, _select

QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def _wire_dist(s, js, geom):
    """Segment minimum and start-point distance of each step from its straw's wire."""
    a, b = geometry.to_mu2e(L._xyz(s, "0")), geometry.to_mu2e(L._xyz(s, "1"))
    d, m = geom.dir[js], geom.mid[js]
    w0 = a - m
    w0 -= L._dot(w0, d)[:, None] * d
    dw = (b - m) - L._dot(b - m, d)[:, None] * d - w0
    tt = np.clip(-L._dot(w0, dw) / np.maximum(L._dot(dw, dw), 1e-12), 0, 1)
    return np.linalg.norm(w0 + tt[:, None] * dw, axis=1), np.linalg.norm(w0, axis=1)


def _arm(paths, geom):
    s = _all_steps(paths)
    s["is_mu"] = ((s.ismcprim == 1) & (s.pdg.abs() == 13)).to_numpy()
    sel = _select(s[s.is_mu])[T.KEY].drop_duplicates()
    s = s.merge(sel, on=T.KEY).reset_index(drop=True)
    js = (s.upanel.to_numpy() * geometry.NSTRAWS + s.straw.to_numpy()).astype(np.int64)
    s["dmin"], s["rstart"] = _wire_dist(s, js, geom)
    single = (s["shape"].to_numpy() == 3) | (s.len.to_numpy() < POINT_LEN)
    s["rclu"] = np.where(single, s.rstart, s.dmin)
    mu = s[s.is_mu].reset_index(drop=True)
    sec = s[~s.is_mu].reset_index(drop=True)
    jm = (mu.upanel.to_numpy() * geometry.NSTRAWS + mu.straw.to_numpy()).astype(np.int64)
    m, ax = geom.mid[jm], geom.dir[jm]
    q0, q1 = geometry.to_mu2e(L._xyz(mu, "0")), geometry.to_mu2e(L._xyz(mu, "1"))
    mu["uc"] = L.chord_u(m, ax, q0, q1)
    mu["nxs"] = mu.groupby(T.KEY + ["sid"]).cumcount()
    mu1 = mu[mu.nxs.to_numpy() == 0]
    ins = sec.merge(mu1[T.KEY + ["sid", "uc", "dmin", "px", "py", "pz", "x0", "y0", "z0", "x1", "y1", "z1"]]
                    .rename(columns={"dmin": "mu_dmin", **{c: "m" + c for c in ["px", "py", "pz", "x0", "y0", "z0", "x1", "y1", "z1"]}}),
                    on=T.KEY + ["sid"], how="inner")
    out = {"events": int(len(sel)), "muon_crossings": int(len(mu)), "nonmuon_steps": int(len(sec)),
           "instraw_steps": int(len(ins))}
    blk = {}
    if len(ins):
        ji = (ins.upanel.to_numpy() * geometry.NSTRAWS + ins.straw.to_numpy()).astype(np.int64)
        m, ax = geom.mid[ji], geom.dir[ji]
        mq0 = geometry.to_mu2e(L._xyz(ins, "0", "m"))
        dm = L.muon_dir(ins[["mpx", "mpy", "mpz"]].to_numpy(float), mq0, geometry.to_mu2e(L._xyz(ins, "1", "m")))
        sg, e1, e2 = L.straw_frame(ax, dm)
        du = np.abs(np.r_[[L.to_local(geometry.to_mu2e(L._xyz(ins, e)), m, ax, sg, e1, e2, ins.uc.to_numpy())[0] for e in "01"]].ravel())
        sh = ins.groupby(T.KEY + ["sid"]).agg(rsec=("rclu", "min"), rmu=("mu_dmin", "first"))
        blk["instraw"] = {"n": int(len(ins)), "absdu_mm_q": np.quantile(du, QS).round(2).tolist(),
                          "dmin_mm_q": np.quantile(ins.dmin, QS).round(3).tolist(),
                          "rstart_mm_q": np.quantile(ins.rstart, QS).round(3).tolist(),
                          "frac_dmin_lt_1mm": float((ins.dmin < 1.0).mean()),
                          "shared_straws": int(len(sh)), "frac_sec_nearer_wire": float((sh.rsec < sh.rmu).mean())}
    mst = mu[T.KEY + ["sid"]].drop_duplicates().assign(_m=1)
    off = sec.merge(mst, on=T.KEY + ["sid"], how="left")
    off = sec[off._m.isna().to_numpy()].reset_index(drop=True)
    if len(off):
        mk = mu.sort_values(T.KEY + ["t"]).reset_index(drop=True)
        mk["k"] = mk.groupby(T.KEY).cumcount()
        loc = L.anchor_offsets(off, mk, geom)
        joff = (off.upanel.to_numpy() * geometry.NSTRAWS + off.straw.to_numpy()).astype(np.int64)
        u = np.abs(np.r_[[L._dot(geometry.to_mu2e(L._xyz(off, e)) - geom.mid[joff], geom.dir[joff]) for e in "01"]].ravel())
        dpl = loc.dpl.to_numpy()
        blk["offaxis"] = {"n": int(len(off)), "absdu_from_anchor_line_mm_q": np.quantile(np.abs(np.r_[loc.du0, loc.du1]), QS).round(1).tolist(),
                          "abs_u_from_centre_mm_q": np.quantile(u, QS).round(1).tolist(),
                          "dmin_mm_q": np.quantile(off.dmin, QS).round(3).tolist(),
                          "rstart_mm_q": np.quantile(off.rstart, QS).round(3).tolist(),
                          "frac_dmin_lt_1mm": float((off.dmin < 1.0).mean()),
                          "_dpl": np.bincount(np.clip(dpl + 36, 0, 72), minlength=73) / max(len(sel), 1),
                          "_dpl_n": np.bincount(np.clip(dpl + 36, 0, 72), minlength=73)}
    out.update(blk)
    return out


def _profile_chi2(f, g, nf, ng):
    """chi2/ndf of two per-event dpl profiles (Poisson errors on the step counts)."""
    var = f["_dpl_n"] / max(nf, 1) ** 2 + g["_dpl_n"] / max(ng, 1) ** 2
    ok = (f["_dpl_n"] + g["_dpl_n"]) >= 10
    return float(((f["_dpl"] - g["_dpl"])[ok] ** 2 / var[ok]).sum() / max(ok.sum(), 1))


def run(inputs, outdir, compare=None, fast_ref=None, gdml=None, **kw):
    if not compare:
        raise SystemExit("libplacecheck FAST --compare G4 [--fast-ref FAST_V1]")
    from .steps import default_gdml
    geom = geometry.from_gdml(gdml or default_gdml())
    arms = {"fast": _arm(inputs, geom), "g4": _arm(compare, geom)}
    if fast_ref:
        arms["fast_ref"] = _arm(fast_ref, geom)
    g = arms["g4"]
    for lab, a in arms.items():
        if "offaxis" in a and "offaxis" in g:
            a["offaxis"]["plane_profile_chi2ndf_vs_g4"] = _profile_chi2(a["offaxis"], g["offaxis"], a["events"], g["events"])
    for a in arms.values():
        if "offaxis" in a:
            a["offaxis"]["dpl_share_abs_le_2"] = float(a["offaxis"]["_dpl_n"][34:39].sum() / max(a["offaxis"]["_dpl_n"].sum(), 1))
            a["offaxis"]["steps_per_event"] = float(a["offaxis"]["n"] / max(a["events"], 1))
            del a["offaxis"]["_dpl"], a["offaxis"]["_dpl_n"]
    for lab, a in arms.items():
        print(f"== {lab}")
        for k, v in a.items():
            print(f"  {k:14s} {v}")
    plots.write_json(outdir, "libplacecheck.json", arms)
