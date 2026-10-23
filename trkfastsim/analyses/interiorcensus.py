"""interiorcensus -- per-volume census of the tracker-interior material: fast shells against Geant4, plane to first straw.

For every G4 primary muon (reference tables with `trajpts`) that crosses the plane above the tracker (steps.plane_y) and
has a StrawGasStep: the G4 trajectory polyline from its first downward plane crossing to the entry of its first straw
(time order), and per trajectory segment
  - the G4 kinetic-energy loss (trajectory point ke; the last point is the straw entry, ke from the step momentum);
  - the fast interior column (the TrackerInterior shells, labelled by GDML volume group and material: leaf volume name
    without its copy indices), in mm, g/cm2 and x/X0, and its mean (Bethe, unrestricted) loss at the segment's G4 |p|.
Reported: per muon, G4 vs fast loss per bin of fast x/X0 (along the G4 path, and on the straight line from the plane
state to the first straight-line straw, as TrackerInterior sees it); per volume group, the column and fast loss summed
over muons and the G4/fast loss ratio on segments where that group carries >= 90 % of the fast loss (where the excess
sits); per group, shells and mass of the fast model against the GDML placements (exact solid volume x density); the
share of the straight-line column above the plane; geomcheck's seeded-from-G4 count started from the shell top (as
geomcheck) and from the plane; fast vs GDML densities.

    interiorcensus "g4ref_v1_tables/sgsteps.*.root" --outdir DIR [--gdml GDML] [--nsub 8]
"""
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials
from mu2efastsim import world as world_mod
from .. import crossings, geometry, steps as st, tables as T

XX0_EDGES = np.array([0.0, 0.05, 0.3, 1.0, 2.0, 5.0, 1e9])
DOMINANT = 0.9                     # share of a segment's fast loss for the segment to be attributed to one group


def _group(path):
    """Volume group of a shell: leaf volume name without copy indices (TrackerSupportService_11_03_c -> ..._c)."""
    leaf = path.rsplit("/", 1)[-1]
    suf = "_c" if leaf.endswith("_c") else ""
    return re.sub(r"(_\d+)+$", "", leaf[:-2] if suf else leaf) + suf


def _mat(name):
    return materials.BY_NAME.get(name, materials.ALUMINIUM)


def _ke(p):
    return np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU


def _p(ke):
    ke = np.maximum(ke, 1e-3)
    return np.sqrt((ke + eloss.MMU) ** 2 - eloss.MMU ** 2)


def labelled_shells(gdml, geom, nsub_pad=300.0):
    """The TrackerInterior shells (same selection) with material keys 'group|material', and the plain interior."""
    interior = st.TrackerInterior(gdml, geom)
    lo, hi = crossings.panel_boxes(geom)
    names = []
    tubes = world_mod.load_tubes(gdml, bbox=(lo.min(0) - nsub_pad, hi.max(0) + nsub_pad), skip_names=st.INTERIOR_SKIP,
                                 names=names)
    keep = [i for i, t in enumerate(tubes) if t[0] != "StrawGas"]
    lab = [(f"{_group(names[i])}|{tubes[i][0]}",) + tuple(tubes[i][1:]) for i in keep]
    shells = world_mod.CoaxialShells(lab, axis_point=geometry.DETECTOR_ORIGIN_EXTRACTED)
    return interior, shells, [tubes[i] for i in keep], [names[i] for i in keep]


def _shell_volume(t):
    _, _, _, rmin, rmax, hz, _, dphi = t
    return 0.5 * dphi * (rmax ** 2 - rmin ** 2) * 2 * hz


def gdml_mass(gdml, skip=st.INTERIOR_SKIP, mother="TrackerMother"):
    """Exact solid volume [mm3] and mass [kg] per (group, material) of the non-vacuum tube/polycone placements inside
    `mother` (straws skipped), from the GDML: tubes exactly, polycone sections as frusta, boolean solids not counted."""
    root = ET.parse(gdml).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    rho = {}
    for m in root.find("materials").findall("material"):
        d = m.find("D")
        if d is not None:
            rho[world_mod._strip(m.get("name"))] = float(d.get("value"))
    out = {}

    def vol(s):
        ang = np.pi / 180 if s.get("aunit", "deg") == "deg" else 1.0
        dphi = float(s.get("deltaphi", 360)) * ang
        if s.tag == "tube":
            return 0.5 * dphi * (float(s.get("rmax")) ** 2 - float(s.get("rmin", 0)) ** 2) * float(s.get("z"))
        if s.tag == "polycone":
            zp = sorted((float(z.get("z")), float(z.get("rmin", 0)), float(z.get("rmax"))) for z in s.findall("zplane"))
            v = 0.0
            for (z0, a0, b0), (z1, a1, b1) in zip(zp[:-1], zp[1:]):
                v += 0.5 * dphi * (z1 - z0) * ((b0 * b0 + b0 * b1 + b1 * b1) - (a0 * a0 + a0 * a1 + a1 * a1)) / 3.0
            return v
        return None

    def walk(lv, inside, path):
        for pv in vols[lv].findall("physvol"):
            c = pv.find("volumeref").get("ref")
            n = world_mod._strip(c)
            if n.startswith(tuple(skip)):
                continue
            ins = inside or n == mother
            if ins and n != mother:
                mat = world_mod._strip(vols[c].find("materialref").get("ref"))
                if not re.search(r"AIR|Vacuum|VACUUM|Galactic", mat) and mat != "StrawGas":
                    v = vol(solids[vols[c].find("solidref").get("ref")])
                    k = (_group(path + n), mat)
                    e = out.setdefault(k, [0, 0.0, 0.0])
                    e[0] += 1
                    if v is not None:
                        e[1] += v
                        e[2] += v * 1e-3 * rho.get(mat, np.nan) * 1e-3
            walk(c, ins, path + n + "/")

    walk(root.find("setup/world").get("ref"), False, "")
    return out


def g4_polylines(paths, y_pl):
    """Per G4 primary muon crossing the plane and reaching a straw: trajectory segments from the plane crossing to the
    first straw entry (flat arrays sorted by muon), plus per-muon plane state (point, direction, |p|) and entry state."""
    tp = T.read_many(paths, "trajpts")
    pa = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    pm = pa[(pa.ismcprim == 1) & (pa.pdg.abs() == 13)].drop_duplicates(T.KEY)[T.KEY + ["sp"]]
    s = T.read_many(paths, "sgsteps").merge(pm, on=T.KEY + ["sp"])
    s = s.sort_values(T.KEY + ["t"]).drop_duplicates(T.KEY).reset_index(drop=True)
    s["iev"] = np.arange(len(s))
    tp = tp.merge(pm, on=T.KEY + ["sp"]).merge(s[T.KEY + ["iev"]], on=T.KEY)
    tp = tp.sort_values(["iev", "ipt"]).reset_index(drop=True)
    iev = tp.iev.to_numpy()
    X = tp[["x", "y", "z"]].to_numpy(float)
    ke, tt = tp.ke.to_numpy(float), tp.t.to_numpy(float)
    same = iev[1:] == iev[:-1]
    cross = same & (X[:-1, 1] >= y_pl) & (X[1:, 1] < y_pl)
    jc_all = np.flatnonzero(cross)
    ev_c, first = np.unique(iev[jc_all], return_index=True)
    jc = jc_all[first]
    f = (X[jc, 1] - y_pl) / np.maximum(X[jc, 1] - X[jc + 1, 1], 1e-12)
    n = len(s)
    P0 = np.full((n, 3), np.nan)
    P0[ev_c] = X[jc] + (X[jc + 1] - X[jc]) * f[:, None]
    ke0, t0 = np.full(n, np.nan), np.full(n, np.nan)
    ke0[ev_c] = ke[jc] + (ke[jc + 1] - ke[jc]) * f
    t0[ev_c] = tt[jc] + (tt[jc + 1] - tt[jc]) * f
    u0 = np.full((n, 3), np.nan)
    du = X[jc + 1] - X[jc]
    u0[ev_c] = du / np.linalg.norm(du, axis=1)[:, None]
    E = geometry.to_mu2e(s[["x0", "y0", "z0"]].to_numpy(float))
    ke1 = _ke(np.sqrt(s.px ** 2 + s.py ** 2 + s.pz ** 2).to_numpy())
    t1 = s.t.to_numpy(float)
    ok = np.isfinite(t0) & (t1 > t0) & (E[:, 1] < y_pl)
    # polyline points: plane crossing, trajectory points strictly between, straw entry
    mid = ok[iev] & (tt > t0[iev]) & (tt < t1[iev]) & (X[:, 1] < y_pl)
    ids = np.flatnonzero(ok)
    pe = np.r_[ids, iev[mid], ids]
    pt = np.r_[t0[ids], tt[mid], t1[ids]]
    pX = np.r_[P0[ids], X[mid], E[ids]]
    pk = np.r_[ke0[ids], ke[mid], ke1[ids]]
    o = np.lexsort((pt, pe))
    pe, pX, pk = pe[o], pX[o], pk[o]
    sm = pe[1:] == pe[:-1]
    a, b = pX[:-1][sm], pX[1:][sm]
    L = np.linalg.norm(b - a, axis=1)
    g = L > 1e-6
    seg = {"iev": pe[:-1][sm][g], "a": a[g], "u": (b - a)[g] / L[g, None], "L": L[g], "ka": pk[:-1][sm][g],
           "kb": pk[1:][sm][g]}
    mu = s[T.KEY + ["iev"]].copy()
    mu["ok"], mu["ke0"], mu["ke1"] = ok, ke0, ke1
    return seg, mu, P0, u0


def density_audit(gdml, mats):
    """Fast density (materials.BY_NAME, else the Al default) against the GDML density for each interior material."""
    root = ET.parse(gdml).getroot()
    rho = {world_mod._strip(m.get("name")): float(m.find("D").get("value"))
           for m in root.find("materials").findall("material") if m.find("D") is not None}
    return {m: [round(_mat(m).rho, 4), rho.get(m)] for m in mats if abs(_mat(m).rho - rho.get(m, _mat(m).rho)) > 1e-3 * _mat(m).rho}


def seeded(paths, geom, interior, y_pl, seed=1):
    """geomcheck's seeded-from-G4 tracker-term count (G4 plane state -> interior transport + straws), started from the
    shell top (geomcheck: the G4 state already lost the material above the plane) and from the plane itself."""
    from .geomcheck import _g4_paths, _primary_pass
    gg = T.read_many(paths, "gen", T.KEY).drop_duplicates().reset_index(drop=True)
    gg["iev"] = np.arange(len(gg))
    seg = _g4_paths(paths, gg, y_pl)
    jl = np.flatnonzero(seg["last"])
    o = seg["b"][jl].copy()
    o[:, 1] = y_pl
    u = seg["b"][jl] - seg["a"][jl]
    u /= np.linalg.norm(u, axis=1)[:, None]
    p = _p(seg["kb"][jl])
    gf = _primary_pass(paths)
    df = gg.iloc[seg["iev"][jl]][T.KEY].reset_index(drop=True).merge(gf, on=T.KEY, how="left")
    alive = (u[:, 1] < 0) & (p > 1.0)
    sel = np.flatnonzero(alive)
    res = {"n": int(len(df)), "g4_pass": int(df.pass_trk.fillna(False).astype(bool).sum())}
    first_lost = df[T.KEY].copy()
    for tag, start in (("from_top", st._back_to_interior_top(o, u, np.zeros(len(o)), alive, interior, geom)[0]), ("from_plane", o)):
        tr_ = interior.transport(np.random.default_rng([int(seed), 99]), start[sel], u[sel], p[sel], geom)
        pk = _p(_ke(p[sel][tr_["it"]]) - tr_["lost"])
        n_sc = np.bincount(sel[tr_["it"][pk > T.FILTER_MIN_MOM]], minlength=len(o))
        res[f"seed_pass_{tag}"] = int((n_sc >= T.FILTER_MIN_TRK_STEPS).sum())
        fr = np.r_[True, tr_["it"][1:] != tr_["it"][:-1]] if len(tr_["it"]) else np.zeros(0, bool)
        v = np.full(len(o), np.nan)
        v[sel[tr_["it"][fr]]] = tr_["lost"][fr]                  # transport loss before the first straw (production quantity)
        first_lost[f"lost_{tag}"] = v
    return res, first_lost


def run(inputs, outdir, gdml=None, nsub=8, **kw):
    from .steps import default_gdml
    gdml = gdml or default_gdml()
    os.makedirs(outdir, exist_ok=True)
    geom = geometry.from_gdml(gdml)
    y_pl = st.plane_y(geom)
    paths = T.expand(inputs)
    interior, shells, tubes, names = labelled_shells(gdml, geom)
    keys_all = sorted({t[0] for t in tubes})
    seg, mu, P0, u0 = g4_polylines(paths, y_pl)
    nseg, nmu = len(seg["L"]), len(mu)
    # fast column per (group|material) on every G4 segment, and its Bethe loss at the segment's G4 |p|
    col = shells.column(seg["a"], seg["u"], seg["L"], nsub=nsub)
    col_plain = interior.shells.column(seg["a"], seg["u"], seg["L"], nsub=nsub)
    tot_l = sum(col.values(), np.zeros(nseg))
    tot_p = sum(col_plain.values(), np.zeros(nseg))
    pa = _p(seg["ka"])
    keys = sorted(col)
    loss = {k: eloss.dedx_restricted(_mat(k.split("|")[1]), np.maximum(pa, 1.0), 1e9) * col[k] for k in keys}
    xx0 = {k: col[k] / eloss.radiation_length_mm(_mat(k.split("|")[1])) for k in keys}
    gcm2 = {k: col[k] * _mat(k.split("|")[1]).rho / 10.0 for k in keys}
    fl = sum(loss.values(), np.zeros(nseg))
    g4l = seg["ka"] - seg["kb"]
    # per muon, along the G4 path
    iev = seg["iev"]
    m_fast = np.bincount(iev, fl, nmu)
    m_g4 = np.bincount(iev, g4l, nmu)
    m_xx0 = np.bincount(iev, sum(xx0.values(), np.zeros(nseg)), nmu)
    ok = mu.ok.to_numpy()
    # straight line from the plane state to the first straight-line straw (as TrackerInterior sees it)
    ids = np.flatnonzero(ok)
    oi, _ = st._back_to_interior_top(P0[ids], u0[ids], np.zeros(len(ids)), np.ones(len(ids), bool), interior, geom)
    xs = crossings.straw_crossings(oi, u0[ids], geom)
    s1 = np.full(len(ids), np.nan)
    if len(xs):
        np.fmin.at(s1, xs[:, 0].astype(np.int64), xs[:, 2])
    has = np.isfinite(s1)
    p0 = _p(mu.ke0.to_numpy()[ids])
    back = np.linalg.norm(oi - P0[ids], axis=1)                 # stretch above the plane added by the back-up

    def line(o_, t_):
        colS = shells.column(o_[has], u0[ids][has], t_[has], nsub=nsub)
        f_, x_ = np.zeros(len(ids)), np.zeros(len(ids))
        for k, c in colS.items():
            m = _mat(k.split("|")[1])
            f_[has] += eloss.dedx_restricted(m, np.maximum(p0[has], 1.0), 1e9) * c
            x_[has] += c / eloss.radiation_length_mm(m)
        return f_, x_

    sl_fast, sl_xx0 = line(oi, s1)                               # from the shell top (as geomcheck's seeded test)
    pl_fast, pl_xx0 = line(P0[ids], s1 - back)                   # from the plane only (same end point)

    def xbins(x, g, f, sel):
        rows = []
        b = np.digitize(x, XX0_EDGES) - 1
        for k in range(len(XX0_EDGES) - 1):
            m = sel & (b == k)
            if m.sum() < 20:
                continue
            rows.append({"xx0": [float(XX0_EDGES[k]), float(XX0_EDGES[k + 1])], "n": int(m.sum()),
                         "g4_median": round(float(np.median(g[m])), 2), "fast_median": round(float(np.median(f[m])), 2),
                         "g4_mean": round(float(g[m].mean()), 2), "fast_mean": round(float(f[m].mean()), 2),
                         "ratio_median": round(float(np.median(g[m]) / max(np.median(f[m]), 1e-9)), 3),
                         "ratio_mean": round(float(g[m].mean() / max(f[m].mean(), 1e-9)), 3)})
        return rows

    g4tot = (mu.ke0 - mu.ke1).to_numpy()
    seed_res, fl1 = seeded(paths, geom, interior, y_pl)
    j = mu[T.KEY].iloc[ids].reset_index(drop=True).merge(fl1, on=T.KEY, how="left")
    tr_rows = {}
    for tag in ("from_top", "from_plane"):
        lv = j[f"lost_{tag}"].to_numpy()
        tr_rows[tag] = xbins(pl_xx0, g4tot[ids], np.nan_to_num(lv), has & np.isfinite(lv))
    by_xx0_path = xbins(m_xx0, m_g4, m_fast, ok)
    by_xx0_line = xbins(sl_xx0, g4tot[ids], sl_fast, has)
    by_xx0_plane = xbins(pl_xx0, g4tot[ids], pl_fast, has)
    above = {"muons": int(has.sum()), "frac_xx0_above_plane": round(float((sl_xx0 - pl_xx0)[has].sum() / max(sl_xx0[has].sum(), 1e-9)), 3),
             "frac_loss_above_plane": round(float((sl_fast - pl_fast)[has].sum() / max(sl_fast[has].sum(), 1e-9)), 3)}
    # per group: column, fast loss, and the G4/fast ratio on segments the group dominates
    grp_rows = []
    nz = fl > 0
    for k in keys:
        dom = nz & (loss[k] >= DOMINANT * fl)
        grp_rows.append({"group": k.split("|")[0], "material": k.split("|")[1], "rho": _mat(k.split("|")[1]).rho,
                         "default_al": k.split("|")[1] not in materials.BY_NAME,
                         "muons_crossing": int(len(np.unique(iev[col[k] > 0]))),
                         "sum_mm": round(float(col[k].sum()), 1), "sum_gcm2": round(float(gcm2[k].sum()), 1),
                         "sum_xx0": round(float(xx0[k].sum()), 2), "fast_loss_MeV": round(float(loss[k].sum()), 1),
                         "dominated_segments": int(dom.sum()), "dom_g4_MeV": round(float(g4l[dom].sum()), 1),
                         "dom_fast_MeV": round(float(fl[dom].sum()), 1),
                         "dom_ratio_g4_fast": round(float(g4l[dom].sum() / max(fl[dom].sum(), 1e-9)), 3)})
    grp = pd.DataFrame(grp_rows).sort_values("fast_loss_MeV", ascending=False)
    empty = ~nz
    no_mat = {"segments": int(empty.sum()), "g4_MeV": round(float(g4l[empty].sum()), 1),
              "g4_MeV_per_m": round(float(g4l[empty].sum() / max(seg["L"][empty].sum(), 1e-9) * 1e3), 2),
              "segments_g4_gt_5MeV": int((empty & (g4l > 5)).sum())}
    # mass: fast shells against the GDML placements
    fm = {}
    for t, nm in zip(tubes, names):
        k = (_group(nm), t[0])
        e = fm.setdefault(k, [0, 0.0])
        e[0] += 1
        e[1] += _shell_volume(t) * 1e-3 * _mat(t[0]).rho * 1e-3
    gm = gdml_mass(gdml)
    mass_rows = []
    for k in sorted(set(fm) | set(gm)):
        f_, g_ = fm.get(k, [0, 0.0]), gm.get(k, [0, 0.0, 0.0])
        mass_rows.append({"group": k[0], "material": k[1], "fast_shells": f_[0], "fast_kg": round(f_[1], 2),
                          "gdml_placements": g_[0], "gdml_kg": round(g_[2], 2)})
    mass = pd.DataFrame(mass_rows)
    out = {"files": len(paths), "muons": int(nmu), "muons_plane_to_straw": int(ok.sum()), "segments": int(nseg),
           "plane_y": y_pl, "check_labelled_vs_interior_mm": [round(float(tot_l.sum()), 3), round(float(tot_p.sum()), 3)],
           "unknown_materials_as_al": interior.unknown, "by_xx0_along_g4_path": by_xx0_path,
           "by_xx0_straight_line": by_xx0_line, "by_xx0_straight_line_from_plane": by_xx0_plane,
           "straight_line_above_plane": above, "no_fast_material_segments": no_mat,
           "seeded": seed_res, "transport_first_lost_by_xx0": tr_rows, "density_audit": density_audit(gdml, keys_all),
           "groups": grp.to_dict("records"), "mass": mass.to_dict("records")}
    with open(os.path.join(outdir, "interiorcensus.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    grp.to_csv(os.path.join(outdir, "interiorcensus_groups.csv"), index=False)
    mass.to_csv(os.path.join(outdir, "interiorcensus_mass.csv"), index=False)
    pd.DataFrame({"iev": np.arange(nmu), "ok": ok, "g4_loss": m_g4, "fast_loss": m_fast, "fast_xx0": m_xx0}).to_csv(
        os.path.join(outdir, "interiorcensus_muons.csv"), index=False)
    _print(out, grp, mass)
    return out


def _print(out, grp, mass):
    print(f"files {out['files']}  muons with a straw {out['muons']}  plane->first straw {out['muons_plane_to_straw']}  "
          f"segments {out['segments']}  plane y {out['plane_y']:.1f}")
    print(f"labelled vs TrackerInterior column (mm, sum): {out['check_labelled_vs_interior_mm']}  "
          f"unknown materials (as Al): {out['unknown_materials_as_al']}")
    print(f"seeded from the G4 plane state: {out['seeded']}")
    for tag, rows in out["transport_first_lost_by_xx0"].items():
        print(f"\nG4 loss vs TrackerInterior.transport loss before the first straw, seeded {tag} (bins: plane straight-line x/X0):")
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"straight line, share above the plane (shell top -> plane): {out['straight_line_above_plane']}")
    print(f"density audit (fast vs GDML, g/cm3): {out['density_audit']}")
    for name in ("by_xx0_along_g4_path", "by_xx0_straight_line", "by_xx0_straight_line_from_plane"):
        print(f"\nG4 vs fast loss [MeV] per muon, {name}:")
        print(pd.DataFrame(out[name]).to_string(index=False))
    print(f"\nsegments with no fast material: {out['no_fast_material_segments']}")
    pd.set_option("display.width", 250)
    print("\nper volume group (fast column summed over muons; G4/fast on segments the group dominates):")
    print(grp.to_string(index=False))
    print("\nmass, fast shells vs GDML placements in TrackerMother:")
    print(mass.to_string(index=False))
