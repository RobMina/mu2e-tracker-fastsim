"""survivalcheck -- selected-muon rate by generated |p| and by the material column above the tracker, fast vs G4.

Unpaired, per second of live time, so the reference can be the MDC2025as production (its gen tree holds the primaries
of every accepted event, which includes every selected one). For each selected event (primary muon with >= 12 good
StrawGasSteps) it takes the generated muon, integrates the GDML world column along the straight generated line down to
the plane above the tracker (as worldcheck: g/cm2, all box/xtru materials), and the energy margin
m = p_gen / (2.1 MeV cm2/g x column). It then compares the rates in (p_gen, column) and in margin bins. A deficit near
m ~ 1-2 only (muons that barely make it through) points at range-out or the loss fluctuations in thick material; a
deficit in thin columns at the same p_gen points at the acceptance (scattering), not at survival.

    survivalcheck FAST.root [...] --compare G4.root [...] --gdml mu2e_extracted_v04.gdml --outdir DIR
"""
import numpy as np
import pandas as pd

from mu2efastsim import plots
from mu2efastsim.tables import KEY, primary_muons
from .. import geometry, steps as st, tables as T
from .worldcheck import _column

P_EDGES = np.array([0.0, 500.0, 750.0, 1000.0, 1500.0, 3000.0, 1e9])
COL_EDGES = np.array([0.0, 50.0, 150.0, 300.0, 450.0, 1e9])
MARGIN_EDGES = np.array([0.0, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 1e9])
DEDX_REF = 2.1      # MeV cm2/g, for the energy margin only


def _selected(paths, world, y_pl):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles")
    pm = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)]
    s = s.merge(pm[KEY + ["key"]].rename(columns={"key": "sp"}), on=KEY + ["sp"])
    f = T.trk_filter(s)
    sel = f[f.pass_trk].reset_index()[KEY]
    g = primary_muons(T.read_many(paths, "gen")).merge(sel, on=KEY)
    g["p_gen"] = np.sqrt(g.px ** 2 + g.py ** 2 + g.pz ** 2)
    g["gcm2"], _ = _column(g, world, y_pl)
    g["margin"] = g.p_gen / (DEDX_REF * np.maximum(g.gcm2, 1e-3))
    return g


def _ratio(na, nb, lt_a, lt_b):
    r = (na / lt_a) / max(nb / lt_b, 1e-30)
    return float(r - 1), float(r * np.sqrt(1 / max(na, 1) + 1 / max(nb, 1)))


def run(inputs, outdir, compare=None, gdml=None, **kw):
    if not compare:
        raise SystemExit("survivalcheck FAST --compare G4REF")
    from .steps import default_gdml
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    world = st.load_world(gdml, geom)
    y_pl = st.plane_y(geom)
    fp, rp = T.expand(inputs), T.expand(compare)
    lt_f, lt_r = T.livetime(fp), T.livetime(rp)
    f, r = _selected(fp, world, y_pl), _selected(rp, world, y_pl)
    out = {"livetime_s": [lt_f, lt_r], "selected": [int(len(f)), int(len(r))], "by_p_gen_column": [], "by_margin": []}
    for i in range(len(P_EDGES) - 1):
        for j in range(len(COL_EDGES) - 1):
            sel = lambda x: (x.p_gen >= P_EDGES[i]) & (x.p_gen < P_EDGES[i + 1]) & (x.gcm2 >= COL_EDGES[j]) & (x.gcm2 < COL_EDGES[j + 1])
            na, nb = int(sel(f).sum()), int(sel(r).sum())
            if nb < 10:
                continue
            d, e = _ratio(na, nb, lt_f, lt_r)
            out["by_p_gen_column"].append({"p_gen": [float(P_EDGES[i]), float(min(P_EDGES[i + 1], 1e6))],
                                           "gcm2": [float(COL_EDGES[j]), float(min(COL_EDGES[j + 1], 1e5))],
                                           "n": [na, nb], "ratio_minus_1": d, "err": e})
    for lab, sub_f, sub_r in (("all", f, r), ("p_gen < 1 GeV/c", f[f.p_gen < 1000], r[r.p_gen < 1000]),
                              ("p_gen 1-3 GeV/c", f[(f.p_gen >= 1000) & (f.p_gen < 3000)], r[(r.p_gen >= 1000) & (r.p_gen < 3000)])):
        ha, _ = np.histogram(sub_f.margin, MARGIN_EDGES)
        hb, _ = np.histogram(sub_r.margin, MARGIN_EDGES)
        rows = []
        for k in range(len(MARGIN_EDGES) - 1):
            if hb[k] < 10:
                continue
            d, e = _ratio(int(ha[k]), int(hb[k]), lt_f, lt_r)
            rows.append({"margin": [float(MARGIN_EDGES[k]), float(min(MARGIN_EDGES[k + 1], 1e6))],
                         "n": [int(ha[k]), int(hb[k])], "ratio_minus_1": d, "err": e})
        out["by_margin"].append({"sample": lab, "rows": rows})
    print(f"live time fast {lt_f:.2f} s, G4 {lt_r:.2f} s; selected muons fast {len(f):,}, G4 {len(r):,}")
    print("selected rate fast/G4 - 1 by generated |p| (rows) and world column above the plane [g/cm2] (columns):")
    for i in range(len(P_EDGES) - 1):
        cells = [x for x in out["by_p_gen_column"] if x["p_gen"][0] == P_EDGES[i]]
        if cells:
            print(f"  {P_EDGES[i]:5.0f}-{min(P_EDGES[i + 1], 1e6):7.0f}  " + "  ".join(
                f"[{x['gcm2'][0]:.0f},{x['gcm2'][1]:.0f}) {x['ratio_minus_1']:+.2f}+-{x['err']:.2f} ({x['n'][0]}/{x['n'][1]})" for x in cells))
    print(f"selected rate fast/G4 - 1 by energy margin p_gen / ({DEDX_REF} MeV cm2/g x column):")
    for blk in out["by_margin"]:
        print(f"  {blk['sample']:16s} " + "  ".join(
            f"[{x['margin'][0]:g},{x['margin'][1]:g}) {x['ratio_minus_1']:+.2f}+-{x['err']:.2f} ({x['n'][0]}/{x['n'][1]})" for x in blk["rows"]))
    plots.write_json(outdir, "survivalcheck.json", out)
