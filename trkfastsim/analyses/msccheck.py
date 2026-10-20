"""msccheck -- in-tracker scattering, step pair by step pair: fast against Geant4, for tuning the single-scattering tail.

For every selected primary muon (PrimaryFilter tracker term, as v1compare), consecutive StrawGasSteps in time order form
a pair. Per pair: the space angle between the two step momenta, p * theta [MeV] (scattering scales as 1/p, so p theta
pools momenta; p = the mean of the two |p|), the path between the step entries, and the material the straight line
between the two entries crosses in the tracker support structure (TrackerInterior shells, x/X0), which splits pairs into
straw-only pairs (support x/X0 < 1e-5) and support pairs.

Reported per class and |p| bin, fast against G4: pair counts, median and q90/q99/q99.9 of p theta, and the share of
pairs above fixed p theta thresholds (the tail). The straw-only class isolates the straw walls and gas (Gaussian core
plus single scattering); the support class tests the support model. The G4 pairs with the largest p theta are also
located (plane, radius from the tracker axis, support x/X0 between them) in msccheck_g4_tail.csv, to find material the
fast model lacks.

    msccheck FAST.root [...] --compare G4.root [...] --gdml mu2e_extracted_v04.gdml --outdir DIR
"""
import os

import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials, plots
from .. import geometry, steps as st, tables as T

P_EDGES = np.array([50.0, 300.0, 500.0, 1000.0, 2000.0, 1e9])       # MeV/c, mean |p| of the pair
PT_THRESH = np.array([0.5, 1.0, 2.0, 4.0, 8.0])                       # MeV: p theta thresholds for the tail shares
SUPPORT_XX0_MIN = 1e-5


def _pairs(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="inner")
    s = s[(s.ismcprim == 1) & (s.pdg.abs() == 13)]
    f = T.trk_filter(s)
    s = s.merge(f[f.pass_trk].reset_index()[T.KEY], on=T.KEY).sort_values(T.KEY + ["t"]).reset_index(drop=True)
    key = s[T.KEY].to_numpy(np.int64)
    same = np.all(key[1:] == key[:-1], axis=1)
    a, b = np.flatnonzero(same), np.flatnonzero(same) + 1
    pa = s[["px", "py", "pz"]].to_numpy(float)
    na, nb = np.linalg.norm(pa[a], axis=1), np.linalg.norm(pa[b], axis=1)
    c = np.clip(np.einsum("ij,ij->i", pa[a], pa[b]) / np.maximum(na * nb, 1e-12), -1, 1)
    th = np.arccos(c)
    x0 = geometry.to_mu2e(s[["x0", "y0", "z0"]].to_numpy(float))
    seg = x0[b] - x0[a]
    L = np.linalg.norm(seg, axis=1)
    return pd.DataFrame({"run": key[a, 0], "subrun": key[a, 1], "event": key[a, 2], "p": 0.5 * (na + nb), "theta": th,
                         "L": L, "plane_a": s.plane.to_numpy()[a], "plane_b": s.plane.to_numpy()[b],
                         "ox": x0[a, 0], "oy": x0[a, 1], "oz": x0[a, 2],
                         "dx": seg[:, 0] / np.maximum(L, 1e-9), "dy": seg[:, 1] / np.maximum(L, 1e-9),
                         "dz": seg[:, 2] / np.maximum(L, 1e-9)})


def _support_xx0(df, interior):
    o = df[["ox", "oy", "oz"]].to_numpy()
    d = df[["dx", "dy", "dz"]].to_numpy()
    col = interior.shells.column(o, d, df.L.to_numpy(), nsub=interior.nsub)
    x = np.zeros(len(df))
    for m, c in col.items():
        x += c / eloss.radiation_length_mm(materials.BY_NAME.get(m, materials.ALUMINIUM))
    return x


def _table(f, g):
    out = {}
    for cls in ("straw", "support"):
        for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
            row = {}
            for lab, df in (("fast", f), ("g4", g)):
                m = (df.cls == cls) & (df.p >= lo) & (df.p < hi)
                pt = (df.p * df.theta)[m].to_numpy()
                row[lab] = {"n": int(m.sum())}
                if m.sum() >= 50:
                    row[lab].update({"median": float(np.median(pt)),
                                     **{f"q{q}": float(np.quantile(pt, q / 100)) for q in (90, 99, 99.9)},
                                     **{f"share_gt_{t:g}MeV": float(np.mean(pt > t)) for t in PT_THRESH}})
            out[f"{cls} {int(lo)}-{int(min(hi, 1e6))}"] = row
    return out


def run(inputs, outdir, compare=None, gdml=None, **kw):
    if not compare:
        raise SystemExit("msccheck FAST --compare G4REF")
    from .steps import default_gdml
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    interior = st.TrackerInterior(gdml, geom)
    f, g = _pairs(T.expand(inputs)), _pairs(T.expand(compare))
    for df in (f, g):
        df["xsup"] = _support_xx0(df, interior)
        df["cls"] = np.where(df.xsup >= SUPPORT_XX0_MIN, "support", "straw")
    out = {"pairs": {"fast": int(len(f)), "g4": int(len(g))},
           "support_pair_share": {"fast": float(np.mean(f.cls == "support")), "g4": float(np.mean(g.cls == "support"))},
           "p_theta_MeV": _table(f, g)}
    for k, v in out["p_theta_MeV"].items():
        a, b = v["fast"], v["g4"]
        if "median" in a and "median" in b:
            print(f"{k:16s} n {a['n']:6d}/{b['n']:6d}  med {a['median']:.3f}/{b['median']:.3f}  q99 {a['q99']:.2f}/{b['q99']:.2f}"
                  f"  q99.9 {a['q99.9']:.1f}/{b['q99.9']:.1f}  >2MeV {a['share_gt_2MeV']:.4f}/{b['share_gt_2MeV']:.4f}"
                  f"  >4MeV {a['share_gt_4MeV']:.4f}/{b['share_gt_4MeV']:.4f}  >8MeV {a['share_gt_8MeV']:.4f}/{b['share_gt_8MeV']:.4f}")
    os.makedirs(outdir, exist_ok=True)
    g["pt"] = g.p * g.theta
    g["r"] = np.hypot(g.ox - geometry.DETECTOR_ORIGIN_EXTRACTED[0], g.oy - geometry.DETECTOR_ORIGIN_EXTRACTED[1])
    g.nlargest(500, "pt")[T.KEY + ["p", "theta", "pt", "L", "plane_a", "plane_b", "r", "oy", "oz", "xsup", "cls"]] \
        .to_csv(os.path.join(outdir, "msccheck_g4_tail.csv"), index=False)
    plots.write_json(outdir, "msccheck.json", out)
