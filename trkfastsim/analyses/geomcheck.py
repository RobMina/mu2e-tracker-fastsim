"""geomcheck -- the true G4 path of the primary muon against the fast world, for the muons one arm delivers and the other not.

Paired by generated event on the same CRY primaries. Inputs: the gen tables the fast `steps` run used (same glob, so
file i reproduces seed (seed, i)), the fast step tables (--fast) and the G4 tables (--compare, with `trajpts`: the
MCTrajectory points of the primary muon, Mu2e frame, with kinetic energy). Per event that passes the PrimaryFilter
tracker term (primary muon, >= 12 good steps) in either arm:
  - the fast fate (stage C to_plane rerun with the steps seed: not transported (generated line outside the near_tracker
    footprint), stopped above the plane, missed the straws, too few straws, pass);
  - the G4 path above the plane (trajectory clipped at its first downward crossing of the plane): deflection from the
    generated direction at the plane, lateral offset of the plane crossing, the height and size of the largest kink
    between consecutive trajectory segments, the height where half the final deflection is reached, and whether the
    G4 state at the plane passes the near_tracker test;
  - the fast-world material column (world.PieceSet of the stage C bands, materials without constants as transport
    does and with their GDML density) along the generated line and along the G4 path, the tube/polycone column above
    the tracker interior (world.column_tubes; not in the fast world), the Bethe mean loss of the fast column along the
    G4 path at G4's own |p| per segment, and the G4 kinetic-energy loss from generation to the plane;
  - per G4 trajectory segment, G4 loss minus fast-column Bethe loss, summed per height band (where the fast world has
    too much or too little material), and segments with no fast material but a large G4 loss (missing material) or a
    large fast prediction and little G4 loss (extra material).
Also a census of the xtru solids: area of the ear-clipped convex pieces world.load makes against the polygon area.

    geomcheck "gen.001400_*.root" --fast "fast_c1geo/fasttrkC.*.root" --compare "g4ref_v1_tables/sgsteps.*.root" \
        --outdir DIR [--seed 1] [--no-tubes]
"""
import os

import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials, plots
from mu2efastsim import world as world_mod
from mu2efastsim.tables import KEY, primary_muons
from .. import geometry, steps as st, tables as T
from .tubecheck import _gdml_densities, interior_top_y

P_DEF = (500.0, 1000.0)          # MeV/c: generated |p| of the deficit region
C_DEF = (50.0, 300.0)            # g/cm2: generated-line world column of the deficit region
P_EDGES = np.array([0.0, 500.0, 1000.0, 3000.0, 1e9])
C_EDGES = np.array([0.0, 50.0, 300.0, 1e9])
SEG_MIN_MM = 1.0                 # trajectory segments shorter than this do not define a kink
MISSING_MEV = 5.0                # segment with no fast material but at least this G4 loss: missing material
EXTRA_FRAC, EXTRA_MEV = 0.25, 10.0   # segment where G4 loses < EXTRA_FRAC of a fast prediction >= EXTRA_MEV: extra material
FATES = ("not_near", "stopped_above", "missed", "few", "pass")


def _p_of_ke(ke):
    ke = np.maximum(ke, 0.0)
    return np.sqrt((ke + eloss.MMU) ** 2 - eloss.MMU ** 2)


def _ke_of_p(p):
    return np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU


def _mat(name):
    return materials.BY_NAME.get(name, materials.CONCRETE_MARS)


def _primary_pass(paths):
    """Per event: max good steps of the primary muon and the tracker-term decision (G4: ismcprim |pdg| 13; fast: sp 1)."""
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p[(p.ismcprim == 1) & (p.pdg.abs() == 13)][KEY + ["sp"]], on=KEY + ["sp"])
    f = T.trk_filter(s).reset_index()
    return f[KEY + ["nmax", "pass_trk"]]


def _fast_plane(gen_paths, geom, world, seed):
    """Fast state at the plane per generated muon, rerunning to_plane with the steps rng (first rng use in make_steps)."""
    rows = []
    for i, f in enumerate(gen_paths):
        gen = T.read(f, "gen")
        mu, o, d, p, L, alive = st.to_plane(gen, geom, np.random.default_rng([int(seed), i]), world, sliced_msc=True)
        p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
        d0 = mu[["px", "py", "pz"]].to_numpy() / p0[:, None]
        near = st.near_tracker(mu[["x", "y", "z"]].to_numpy(), d0, geom, st.plane_y(geom))
        df = mu[KEY].copy()
        df["near"], df["alive"], df["p_plane_fast"] = near, alive, np.where(alive, p, np.nan)
        for j, c in enumerate("xyz"):
            df[f"f{c}"], df[f"fu{c}"] = o[:, j], d[:, j]
        rows.append(df)
        print(f"# {os.path.basename(f)}: {len(df):,} muons, near {int(near.sum()):,}, alive {int(alive.sum()):,}", flush=True)
    return pd.concat(rows, ignore_index=True)


def _line_column(o, d, y_pl, world, rho_gdml, p0):
    """Fast-world column along the generated line, band by band as world_loss: g/cm2 (transport densities and GDML
    densities) and the Bethe mean loss with |p| stepped down band by band."""
    edges = world.edges
    L = (o[:, 1] - y_pl) / -d[:, 1]
    g_tr, g_gd, bethe = np.zeros(len(o)), np.zeros(len(o)), np.zeros(len(o))
    unknown = {}
    p = p0.copy()
    for k in range(len(edges) - 1):
        s_top = np.clip((o[:, 1] - edges[k]) / -d[:, 1], 0, L)
        s_bot = np.clip((o[:, 1] - edges[k + 1]) / -d[:, 1], 0, L)
        seg = s_bot - s_top
        act = np.flatnonzero(seg > 0)
        if not len(act):
            continue
        col = world.band(k).column((o + d * s_top[:, None])[act], d[act], seg[act])
        loss = np.zeros(len(act))
        for name, c in col.items():
            m = _mat(name)
            g_tr[act] += c * m.rho / 10.0
            g_gd[act] += c * rho_gdml.get(name, m.rho) / 10.0
            loss += eloss.dedx_restricted(m, np.maximum(p[act], 1.0), 1e9) * c
            if name not in materials.BY_NAME:
                unknown[name] = unknown.get(name, 0.0) + float(c.sum())
        p[act] = _p_of_ke(_ke_of_p(p[act]) - loss)
        bethe[act] += loss
    return g_tr, g_gd, bethe, unknown


def _g4_paths(paths, ev, y_pl):
    """G4 primary-muon trajectory segments above the plane for the events in `ev` (column iev): flat arrays, sorted by
    event then point, clipped at the first downward crossing of the plane; events without a crossing are dropped."""
    tp = T.read_many(paths, "trajpts")
    p = T.read_many(paths, "particles", KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    pm = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)].drop_duplicates(KEY)
    tp = tp.merge(pm[KEY + ["sp"]], on=KEY + ["sp"]).merge(ev[KEY + ["iev"]], on=KEY)
    tp = tp.sort_values(["iev", "ipt"]).reset_index(drop=True)
    iev = tp.iev.to_numpy()
    X = tp[["x", "y", "z"]].to_numpy(float)
    ke = tp.ke.to_numpy(float)
    same = iev[1:] == iev[:-1]
    a, b, ka, kb, si = X[:-1][same], X[1:][same], ke[:-1][same], ke[1:][same], iev[:-1][same]
    cross = (a[:, 1] >= y_pl) & (b[:, 1] < y_pl)
    ev_c, first = np.unique(si[cross], return_index=True)
    jc = np.flatnonzero(cross)[first]                       # segment index of each event's first crossing
    last = np.full(len(ev), -1)
    last[ev_c] = jc
    keep = (last[si] >= 0) & (np.arange(len(si)) <= last[si])
    f = np.clip((a[jc, 1] - y_pl) / np.maximum(a[jc, 1] - b[jc, 1], 1e-12), 0, 1)
    b, kb = b.copy(), kb.copy()
    b[jc] = a[jc] + (b[jc] - a[jc]) * f[:, None]
    kb[jc] = ka[jc] + (kb[jc] - ka[jc]) * f
    is_last = np.zeros(len(si), bool)
    is_last[jc] = True
    a, b, ka, kb, si, is_last = a[keep], b[keep], ka[keep], kb[keep], si[keep], is_last[keep]
    has_traj = np.zeros(len(ev), bool)
    has_traj[np.unique(iev)] = True
    return {"a": a, "b": b, "ka": ka, "kb": kb, "iev": si, "last": is_last, "has_traj": has_traj, "crossed": last >= 0}


def _path_quantities(seg, ev, d0, world, rho_gdml, tubes, geom, y_pl):
    """Per-event G4 path quantities and per-segment fast-column predictions (see module doc)."""
    a, b, iev = seg["a"], seg["b"], seg["iev"]
    n = len(ev)
    v = b - a
    L = np.linalg.norm(v, axis=1)
    u = v / np.maximum(L, 1e-12)[:, None]
    col = world.pieces().column(a, u, L)
    g_tr, g_gd, bethe = np.zeros(len(a)), np.zeros(len(a)), np.zeros(len(a))
    pa = _p_of_ke(seg["ka"])
    for name, c in col.items():
        m = _mat(name)
        g_tr += c * m.rho / 10.0
        g_gd += c * rho_gdml.get(name, m.rho) / 10.0
        bethe += eloss.dedx_restricted(m, np.maximum(pa, 1.0), 1e9) * c
    solid = sum(col.values()) if col else np.zeros(len(a))
    bethe += eloss.dedx_restricted(materials.AIR, np.maximum(pa, 1.0), 1e9) * np.clip(L - solid, 0, None)
    tg, tb = _tube_column(tubes, a, u, L, pa, rho_gdml) if tubes else (np.zeros(len(a)), np.zeros(len(a)))
    dke = seg["ka"] - seg["kb"]
    out = pd.DataFrame({"iev": np.arange(n)})
    for k, w in (("g4path_gcm2", g_tr), ("g4path_gcm2_gdml", g_gd), ("g4path_bethe", bethe), ("g4path_tube_gcm2", tg),
                 ("g4path_tube_bethe", tb), ("g4_loss_plane", dke), ("g4path_len", L)):
        out[k] = np.bincount(iev, weights=w, minlength=n)
    # plane crossing: position, direction, kinetic energy
    jl = np.flatnonzero(seg["last"])
    el = iev[jl]
    for j, c in enumerate("xyz"):
        out.loc[el, f"g{c}"] = b[jl, j]
        out.loc[el, f"gu{c}"] = u[jl, j]
    out.loc[el, "ke_plane_g4"] = seg["kb"][jl]
    # deflection from the generated direction per segment, at the plane, and the half-deflection height
    th = np.arccos(np.clip(np.einsum("ij,ij->i", u, d0[iev]), -1, 1))
    th_pl = np.full(n, np.nan)
    th_pl[el] = th[jl]
    out["g4_theta_plane"] = th_pl
    half = (L > SEG_MIN_MM) & (th >= 0.5 * th_pl[iev])
    e_h, ih = np.unique(iev[half], return_index=True)
    out.loc[e_h, "y_half_deflection"] = a[np.flatnonzero(half)[ih], 1]
    # kinks between consecutive segments of the same event (both longer than SEG_MIN_MM)
    ok = (iev[1:] == iev[:-1]) & (L[1:] > SEG_MIN_MM) & (L[:-1] > SEG_MIN_MM)
    kang = np.arccos(np.clip(np.einsum("ij,ij->i", u[1:], u[:-1]), -1, 1))[ok]
    kev, ky = iev[1:][ok], a[1:][ok, 1]
    if len(kev):
        order = np.lexsort((-kang, kev))
        e_k, ik = np.unique(kev[order], return_index=True)
        out.loc[e_k, "kink_max"] = kang[order][ik]
        out.loc[e_k, "y_kink_max"] = ky[order][ik]
    segdf = pd.DataFrame({"iev": iev, "y_mid": 0.5 * (a[:, 1] + b[:, 1]), "x_mid": 0.5 * (a[:, 0] + b[:, 0]),
                          "z_mid": 0.5 * (a[:, 2] + b[:, 2]), "len": L, "fast_gcm2": g_tr, "fast_bethe": bethe,
                          "tube_gcm2": tg, "g4_dke": dke, "solid_mm": solid})
    return out, segdf


def _tube_column(tubes, a, u, L, pa, rho_gdml):
    """Tube/polycone column (g/cm2, Bethe MeV) per segment; each shell only against the segments its box overlaps."""
    lo, hi = np.minimum(a, a + u * L[:, None]), np.maximum(a, a + u * L[:, None])
    g, bethe = np.zeros(len(a)), np.zeros(len(a))
    for t in tubes:
        mat, R, tc, rmin, rmax, hz = t[:6]
        ext = np.abs(R) @ np.array([rmax, rmax, hz])
        m = np.flatnonzero(np.all(hi >= tc - ext, axis=1) & np.all(lo <= tc + ext, axis=1))
        if not len(m):
            continue
        for name, c in world_mod.column_tubes([t], a[m], u[m], L[m], nsub=8).items():
            mm = materials.BY_NAME.get(name, materials.MILD_STEEL)
            r = rho_gdml.get(name, mm.rho)
            g[m] += c * r / 10.0
            bethe[m] += eloss.dedx_restricted(mm, np.maximum(pa[m], 1.0), 1e9) * c * (r / mm.rho)
    return g, bethe


def _xtru_census(gdml, y_pl):
    """xtru solids placed with material: polygon area vs the summed area of the ear-clipped pieces world.load keeps."""
    import re
    import xml.etree.ElementTree as ET
    root = ET.parse(gdml).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    rho = _gdml_densities(gdml)
    used = {}
    for v in vols.values():
        mr = v.find("materialref")
        mat = world_mod._strip(mr.get("ref")) if mr is not None else ""
        if re.search(r"AIR|Vacuum|VACUUM|Galactic", mat):
            continue
        s = solids[v.find("solidref").get("ref")]
        boolean = s.tag in ("subtraction", "union", "intersection")
        while s.tag in ("subtraction", "union", "intersection"):
            s = solids[s.find("first").get("ref")]
        if s.tag == "xtru":
            used[world_mod._strip(v.get("name"))] = (s, mat, boolean)
    rows = []
    for name, (s, mat, boolean) in used.items():
        vv = np.array([[float(p.get("x")), float(p.get("y"))] for p in s.findall("twoDimVertex")])
        sec = [float(x.get("zPosition")) for x in s.findall("section")]
        a_poly = 0.5 * abs(np.sum(vv[:, 0] * np.roll(vv[:, 1], -1) - np.roll(vv[:, 0], -1) * vv[:, 1]))
        tris = world_mod._convex_pieces_2d(vv)
        a_tri = sum(0.5 * abs((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0]))
                    for t in tris)
        dz = max(sec) - min(sec) if sec else 0.0
        dropped = len(sec) != 2 or any(float(x.get("scalingFactor", 1)) != 1 for x in s.findall("section"))
        if dropped:   # world.load skips these entirely
            a_tri = 0.0
        rows.append({"volume": name, "material": mat, "boolean": boolean, "dropped": dropped, "nvert": len(vv), "ntri": len(tris),
                     "area_poly_m2": a_poly * 1e-6, "area_pieces_m2": a_tri * 1e-6,
                     "excess_t": (a_tri - a_poly) * dz * 1e-9 * rho.get(mat, 0.0)})
    df = pd.DataFrame(rows)
    df["ratio"] = df.area_pieces_m2 / df.area_poly_m2.clip(lower=1e-12)
    return df


SEED_P_EDGES = np.array([0.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 3000.0, 1e9])   # MeV/c at the plane (G4)


def _seeded(g4_paths, geom, gdml, y_pl, seed):
    """Fast interior + straws started from the G4 muon state at the plane, for every G4-kept event whose primary muon
    crosses the plane: tracker-term pass rate fast-from-G4-state (scattering, and straight) against G4 itself."""
    from .. import crossings
    gg = T.read_many(g4_paths, "gen", KEY).drop_duplicates().reset_index(drop=True)
    gg["iev"] = np.arange(len(gg))
    seg = _g4_paths(g4_paths, gg, y_pl)
    jl = np.flatnonzero(seg["last"])
    e = seg["iev"][jl]
    o = seg["b"][jl].copy()
    o[:, 1] = y_pl
    u = seg["b"][jl] - seg["a"][jl]
    u /= np.linalg.norm(u, axis=1)[:, None]
    p = _p_of_ke(seg["kb"][jl])
    gf = _primary_pass(g4_paths)
    df = gg.iloc[e][KEY].reset_index(drop=True).merge(gf, on=KEY, how="left")
    df["g4_pass"] = df.pass_trk.fillna(False).astype(bool)
    df["p_plane"] = p
    interior = st.TrackerInterior(gdml, geom)
    alive = (u[:, 1] < 0) & (p > 1.0)
    oi, _ = st._back_to_interior_top(o, u, np.zeros(len(o)), alive, interior, geom)
    sel = np.flatnonzero(alive)
    tr_ = interior.transport(np.random.default_rng([int(seed), 99]), oi[sel], u[sel], p[sel], geom)
    pk = _p_of_ke(_ke_of_p(p[sel][tr_["it"]]) - tr_["lost"])
    good = pk > T.FILTER_MIN_MOM
    n_sc = np.zeros(len(o), int)
    np.add.at(n_sc, sel[tr_["it"][good]], 1)
    xs = crossings.straw_crossings(o, u, geom)
    n_st = np.bincount(xs[:, 0].astype(np.int64), minlength=len(o)) if len(xs) else np.zeros(len(o), int)
    df["seed_pass"] = n_sc >= T.FILTER_MIN_TRK_STEPS
    df["straight_pass"] = alive & (n_st >= T.FILTER_MIN_TRK_STEPS)
    df["n_seed"], df["n_straight"], df["ngood_g4"] = n_sc, n_st, df.nmax.fillna(0).astype(int)
    b = np.clip(np.digitize(df.p_plane, SEED_P_EDGES) - 1, 0, len(SEED_P_EDGES) - 2)
    rows = []
    for k in range(len(SEED_P_EDGES) - 1):
        x = df[b == k]
        if not len(x):
            continue
        rows.append({"p_plane": [float(SEED_P_EDGES[k]), float(SEED_P_EDGES[k + 1])], "n": int(len(x)),
                     "g4_pass": int(x.g4_pass.sum()), "seed_pass": int(x.seed_pass.sum()),
                     "straight_pass": int(x.straight_pass.sum()),
                     "g4_only": int((x.g4_pass & ~x.seed_pass).sum()), "seed_only": int((~x.g4_pass & x.seed_pass).sum())})
    tot = {"n": int(len(df)), "g4_pass": int(df.g4_pass.sum()), "seed_pass": int(df.seed_pass.sum()),
           "straight_pass": int(df.straight_pass.sum())}
    return {"total": tot, "by_p_plane": rows, "deflection_vs_highland": _defl_highland(g4_paths, gg, e, o, oi, u, p, sel, tr_,
                                                                                       interior, geom)}, df


def _defl_highland(g4_paths, gg, e, o, oi, u, p, sel, tr_, interior, geom):
    """Plane -> first straw: projected deflection over Highland theta0 of the straight-line support x/X0, G4 vs fast."""
    from .. import crossings
    n = len(o)
    xs = crossings.straw_crossings(oi, u, geom)
    s1 = np.full(n, np.inf)
    np.minimum.at(s1, xs[:, 0].astype(np.int64), xs[:, 2])
    col = interior.shells.column(oi, u, np.where(np.isfinite(s1), s1, 0.0), nsub=interior.nsub)
    xx0 = sum((c / eloss.radiation_length_mm(materials.BY_NAME.get(k, materials.ALUMINIUM)) for k, c in col.items()), np.zeros(n))
    s = T.read_many(g4_paths, "sgsteps")
    pp = T.read_many(g4_paths, "particles", KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(pp[(pp.ismcprim == 1) & (pp.pdg.abs() == 13)][KEY + ["sp"]], on=KEY + ["sp"])
    s = s.merge(gg[KEY + ["iev"]], on=KEY).sort_values(["iev", "t"])
    pos = pd.Series(np.full(len(gg), -1), dtype=int)
    pos[e] = np.arange(n)
    f = s.groupby("iev")[["px", "py", "pz"]].first()
    vg = np.full((n, 3), np.nan)
    k = pos[f.index].to_numpy()
    vg[k[k >= 0]] = f.to_numpy()[k >= 0]
    first = np.r_[True, tr_["it"][1:] != tr_["it"][:-1]] if len(tr_["it"]) else np.zeros(0, bool)
    vf = np.full((n, 3), np.nan)
    vf[sel[tr_["it"][first]]] = tr_["dir"][first]
    ref = np.where(np.abs(u[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(u, e1)
    beta = p / np.sqrt(p ** 2 + eloss.MMU ** 2)
    x = np.maximum(xx0, 1e-9)
    th0 = 13.6 / (beta * p) * np.sqrt(x) * (1 + 0.038 * np.log(x))
    def ratio(v):   # projected angles in both transverse planes over theta0
        v = v / np.linalg.norm(v, axis=1)[:, None]
        c = np.einsum("ij,ij->i", v, u)
        return np.c_[np.einsum("ij,ij->i", v, e1) / c / th0, np.einsum("ij,ij->i", v, e2) / c / th0]
    rg, rf = ratio(vg), ratio(vf)
    rows = []
    for a, b in ((0.05, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 10.0)):
        for plo, phi in ((0.0, 1000.0), (1000.0, 3000.0), (3000.0, 1e9)):
            m = (xx0 >= a) & (xx0 < b) & (p >= plo) & (p < phi) & np.isfinite(rg).all(1) & np.isfinite(rf).all(1)
            if m.sum() < 30:
                continue
            q = lambda r: [round(float(1.4826 * np.median(np.abs(r[m]))), 3), round(float(np.quantile(np.abs(r[m]), 0.9)), 3)]
            rows.append({"xx0": [a, b], "p_plane": [plo, phi], "n": int(m.sum()), "g4_sigma_q90": q(rg), "fast_sigma_q90": q(rf)})
    return rows


def _q(x, qs=(0.5, 0.9)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [round(float(v), 3) for v in np.quantile(x, qs)] if len(x) else None


def _class_table(ev, mask_region):
    """Per class (both, g4_only, fast_only) inside a region: fate, near tests, kinematics and material summaries."""
    out = {}
    for lab, m in (("both", ev.g4_pass & ev.fast_pass), ("g4_only", ev.g4_pass & ~ev.fast_pass),
                   ("fast_only", ev.fast_pass & ~ev.g4_pass)):
        x = ev[m & mask_region]
        if not len(x):
            continue
        g = x[x.g4_crossed]
        out[lab] = {
            "n": int(len(x)), "fast_fate": {f: int((x.fate == f).sum()) for f in FATES},
            "gen_line_near": int(x.near.sum()), "g4_plane_state_near": int(g.g4_near.sum()), "g4_crossed_plane": int(len(g)),
            "p_gen_q": _q(x.p_gen), "gen_gcm2_q": _q(x.gen_gcm2), "g4path_gcm2_q": _q(g.g4path_gcm2),
            "g4path_minus_gen_gcm2_q": _q(g.g4path_gcm2 - g.gen_gcm2, (0.1, 0.5, 0.9)),
            "gen_gcm2_gdml_minus_transport_q": _q(x.gen_gcm2_gdml - x.gen_gcm2, (0.5, 0.9, 0.99)),
            "tube_gcm2_g4path_q": _q(g.g4path_tube_gcm2, (0.5, 0.9, 0.99)),
            "g4_loss_plane_q": _q(g.g4_loss_plane), "g4path_bethe_q": _q(g.g4path_bethe),
            "g4_loss_over_bethe_median": float(np.median(g.g4_loss_plane / g.g4path_bethe.clip(lower=1e-3))) if len(g) else None,
            "fast_loss_plane_q": _q(x.fast_loss_plane), "gen_bethe_q": _q(x.gen_bethe),
            "g4_theta_plane_mrad_q": _q(1e3 * g.g4_theta_plane, (0.5, 0.9, 0.99)),
            "fast_theta_plane_mrad_q": _q(1e3 * x.fast_theta_plane, (0.5, 0.9, 0.99)),
            "g4_offset_plane_mm_q": _q(g.g4_offset_plane, (0.5, 0.9, 0.99)),
            "fast_offset_plane_mm_q": _q(x.fast_offset_plane, (0.5, 0.9, 0.99)),
            "g4_miss_plane_mm_q": _q(g.g4_miss_plane, (0.5, 0.9)), "gen_miss_plane_mm_q": _q(x.gen_miss_plane, (0.5, 0.9)),
            "fast_miss_plane_mm_q": _q(x.fast_miss_plane, (0.5, 0.9)),
            "kink_max_mrad_q": _q(1e3 * g.kink_max, (0.5, 0.9)), "y_kink_max_q": _q(g.y_kink_max, (0.1, 0.5, 0.9)),
            "y_half_deflection_q": _q(g.y_half_deflection, (0.1, 0.5, 0.9))}
    return out


def _band_residuals(segdf, ev, edges, sel):
    """G4 loss and fast-column Bethe loss per height band on the G4 paths of the selected events."""
    s = segdf[sel[segdf.iev.to_numpy()]]
    b = np.clip(np.searchsorted(-edges, -s.y_mid.to_numpy()) - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        x = s[b == k]
        if not len(x):
            continue
        miss = x[(x.solid_mm < 1e-6) & (x.g4_dke > MISSING_MEV)]
        extra = x[(x.fast_bethe >= EXTRA_MEV) & (x.g4_dke < EXTRA_FRAC * x.fast_bethe)]
        rows.append({"y": [float(edges[k]), float(edges[k + 1])], "nseg": int(len(x)),
                     "g4_loss_sum": float(x.g4_dke.sum()), "fast_bethe_sum": float(x.fast_bethe.sum()),
                     "ratio": float(x.g4_dke.sum() / max(x.fast_bethe.sum(), 1e-9)),
                     "fast_gcm2_sum": float(x.fast_gcm2.sum()), "tube_gcm2_sum": float(x.tube_gcm2.sum()),
                     "missing_nseg": int(len(miss)), "missing_g4_loss": float(miss.g4_dke.sum()),
                     "missing_xyz_median": [round(float(miss[c].median()), 0) for c in ("x_mid", "y_mid", "z_mid")] if len(miss) else None,
                     "extra_nseg": int(len(extra)), "extra_fast_bethe": float(extra.fast_bethe.sum()),
                     "extra_g4_loss": float(extra.g4_dke.sum()),
                     "extra_xyz_median": [round(float(extra[c].median()), 0) for c in ("x_mid", "y_mid", "z_mid")] if len(extra) else None})
    return rows


def run(inputs, outdir, fast=None, compare=None, gdml=None, seed=1, no_tubes=False, **kw):
    if not fast or not compare:
        raise SystemExit("geomcheck GEN --fast FASTSTEPS --compare G4REF")
    from .steps import default_gdml
    from .worldcheck import _miss_distance
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    world = st.load_world(gdml, geom)
    y_pl = st.plane_y(geom)
    rho_gdml = _gdml_densities(gdml)
    gen_paths, fast_paths, g4_paths = T.expand(inputs), T.expand(fast), T.expand(compare)
    # fast state at the plane (to_plane rerun with the steps seeds) and the tracker-term decisions of both arms
    fp = _fast_plane(gen_paths, geom, world, seed)
    gen = primary_muons(T.read_many(gen_paths, "gen"))
    gen["p_gen"] = np.sqrt(gen.px ** 2 + gen.py ** 2 + gen.pz ** 2)
    ff, gf = _primary_pass(fast_paths), _primary_pass(g4_paths)
    ev = gen.merge(fp, on=KEY).merge(ff.rename(columns={"nmax": "ngood_fast", "pass_trk": "fast_pass"}), on=KEY, how="left")
    ev = ev.merge(gf.rename(columns={"nmax": "ngood_g4", "pass_trk": "g4_pass"}), on=KEY, how="left")
    ev["fast_pass"] = ev.fast_pass.fillna(False).astype(bool)
    ev["g4_pass"] = ev.g4_pass.fillna(False).astype(bool)
    ev["ngood_fast"] = ev.ngood_fast.fillna(0).astype(int)
    n_gen = len(ev)
    fate = np.select([~ev.near, ~ev.alive, ev.ngood_fast == 0, ev.ngood_fast < T.FILTER_MIN_TRK_STEPS],
                     list(FATES[:4]), "pass")
    consistency = {"fast_pass_not_alive": int((ev.fast_pass & ~ev.alive).sum()),
                   "fate_pass_vs_tables": int(((fate == "pass") != ev.fast_pass).sum())}
    ev["fate"] = fate
    ev = ev[ev.fast_pass | ev.g4_pass].reset_index(drop=True)
    ev["iev"] = np.arange(len(ev))
    o = ev[["x", "y", "z"]].to_numpy(float)
    d0 = ev[["px", "py", "pz"]].to_numpy(float) / ev.p_gen.to_numpy()[:, None]
    # generated line: fast-world column and Bethe loss; fast loss and kinematics at the plane
    g_tr, g_gd, g_be, unknown = _line_column(o, d0, y_pl, world, rho_gdml, ev.p_gen.to_numpy())
    ev["gen_gcm2"], ev["gen_gcm2_gdml"], ev["gen_bethe"] = g_tr, g_gd, g_be
    ev["fast_loss_plane"] = _ke_of_p(ev.p_gen) - _ke_of_p(ev.p_plane_fast)
    Lg = (o[:, 1] - y_pl) / -d0[:, 1]
    xg = o + d0 * Lg[:, None]                                    # generated line at the plane
    fu = ev[["fux", "fuy", "fuz"]].to_numpy()
    ev["fast_theta_plane"] = np.where(ev.alive, np.arccos(np.clip(np.einsum("ij,ij->i", fu, d0), -1, 1)), np.nan)
    ev["fast_offset_plane"] = np.where(ev.alive, np.hypot(ev.fx - xg[:, 0], ev.fz - xg[:, 2]), np.nan)
    ev["gen_miss_plane"] = _miss_distance(o, d0, geom, y_pl)
    ev["fast_miss_plane"] = np.where(ev.alive, _miss_distance(ev[["fx", "fy", "fz"]].to_numpy(), fu, geom, y_pl), np.nan)
    # G4 path above the plane
    seg = _g4_paths(g4_paths, ev, y_pl)
    tubes = []
    if not no_tubes:
        from .. import steps as st_mod
        top = interior_top_y(st_mod.TrackerInterior(gdml, geom), y_pl)
        lo, hi = np.minimum(seg["a"], seg["b"]).min(0), np.maximum(seg["a"], seg["b"]).max(0)
        tubes = [t for t in world_mod.load_tubes(gdml, bbox=(lo, hi)) if t[0] != "StrawGas"
                 and t[2][1] + (np.abs(t[1]) @ np.array([t[4], t[4], t[5]]))[1] > top]
    pq, segdf = _path_quantities(seg, ev, d0, world, rho_gdml, tubes, geom, y_pl)
    ev = ev.merge(pq, on="iev", how="left")
    ev["g4_has_traj"], ev["g4_crossed"] = seg["has_traj"], seg["crossed"]
    gx = ev[["gx", "gy", "gz"]].to_numpy()
    gu = ev[["gux", "guy", "guz"]].to_numpy()
    okc = ev.g4_crossed.to_numpy()
    ev["g4_offset_plane"] = np.where(okc, np.hypot(gx[:, 0] - xg[:, 0], gx[:, 2] - xg[:, 2]), np.nan)
    gx0, gu0 = np.where(okc[:, None], gx, o), np.where(okc[:, None], gu, d0)
    ev["g4_near"] = okc & st.near_tracker(gx0 - gu0 * 1.0, gu0, geom, y_pl)
    ev["g4_miss_plane"] = np.where(okc, _miss_distance(gx0 - gu0 * 1.0, gu0, geom, y_pl), np.nan)
    region = ((ev.p_gen >= P_DEF[0]) & (ev.p_gen < P_DEF[1]) & (ev.gen_gcm2 >= C_DEF[0]) & (ev.gen_gcm2 < C_DEF[1]))
    out = {"generated": int(n_gen), "consistency": consistency, "plane_y": y_pl, "tubes_above_interior": len(tubes),
           "unknown_material_mm_on_gen_lines": unknown,
           "pass": {"fast": int(ev.fast_pass.sum()), "g4": int(ev.g4_pass.sum()), "both": int((ev.fast_pass & ev.g4_pass).sum())},
           "g4_traj_coverage": {"g4_pass": int(ev.g4_pass.sum()), "with_traj": int((ev.g4_pass & ev.g4_has_traj).sum()),
                                "crossed_plane": int((ev.g4_pass & ev.g4_crossed).sum())},
           "deficit_region": {"p_gen": P_DEF, "gen_gcm2": C_DEF, "classes": _class_table(ev, region)},
           "all": {"classes": _class_table(ev, np.ones(len(ev), bool))}, "cells": []}
    pb = np.clip(np.digitize(ev.p_gen, P_EDGES) - 1, 0, len(P_EDGES) - 2)
    cb = np.clip(np.digitize(ev.gen_gcm2, C_EDGES) - 1, 0, len(C_EDGES) - 2)
    for i in range(len(P_EDGES) - 1):
        for j in range(len(C_EDGES) - 1):
            m = (pb == i) & (cb == j)
            x = ev[m]
            if not len(x):
                continue
            g4o = x[x.g4_pass & ~x.fast_pass]
            fo = x[x.fast_pass & ~x.g4_pass]
            out["cells"].append({"p_gen": [float(P_EDGES[i]), float(P_EDGES[i + 1])],
                                 "gen_gcm2": [float(C_EDGES[j]), float(C_EDGES[j + 1])],
                                 "g4": int(x.g4_pass.sum()), "fast": int(x.fast_pass.sum()),
                                 "g4_only_fate": {f: int((g4o.fate == f).sum()) for f in FATES[:4]},
                                 "fast_only_n": int(len(fo)),
                                 "g4_only_g4near": int(g4o.g4_near.sum()), "g4_only_gennear": int(g4o.near.sum())})
    edges = world.edges
    sel_g4 = ev.g4_pass.to_numpy()
    out["band_residuals_g4pass"] = _band_residuals(segdf, ev, edges, sel_g4)
    out["band_residuals_g4pass_region"] = _band_residuals(segdf, ev, edges, (ev.g4_pass & region).to_numpy())
    # G4 loss vs the fast prediction along the G4 path, per event, by generated-line column
    g = ev[ev.g4_pass & ev.g4_crossed & (ev.ke_plane_g4 > 0)]
    r = g.g4_loss_plane / g.g4path_bethe.clip(lower=1e-3)
    cbg = np.clip(np.digitize(g.gen_gcm2, C_EDGES) - 1, 0, len(C_EDGES) - 2)
    out["g4_loss_over_fast_bethe_by_column"] = [
        {"gen_gcm2": [float(C_EDGES[j]), float(C_EDGES[j + 1])], "n": int((cbg == j).sum()),
         "ratio_q": _q(r[cbg == j], (0.1, 0.5, 0.9)),
         "g4path_minus_gen_gcm2_q": _q((g.g4path_gcm2 - g.gen_gcm2)[cbg == j], (0.1, 0.5, 0.9)),
         "share_g4_much_less": float(np.mean(r[cbg == j] < 0.5)), "share_g4_much_more": float(np.mean(r[cbg == j] > 2.0))}
        for j in range(len(C_EDGES) - 1) if (cbg == j).sum()]
    out["seeded_from_g4_plane_state"], sdf = _seeded(g4_paths, geom, gdml, y_pl, seed)
    os.makedirs(outdir, exist_ok=True)
    sdf.to_csv(os.path.join(outdir, "geomcheck_seeded.csv.gz"), index=False)
    xc = _xtru_census(gdml, y_pl)
    bad = xc[(xc.ratio - 1).abs() > 0.01].sort_values("excess_t", key=np.abs, ascending=False)
    out["xtru_census"] = {"n": int(len(xc)), "n_bad": int(len(bad)), "bad": bad.head(30).to_dict("records")}
    _print(out)
    os.makedirs(outdir, exist_ok=True)
    plots.write_json(outdir, "geomcheck.json", out)
    ev.to_csv(os.path.join(outdir, "geomcheck_events.csv.gz"), index=False)
    segdf[sel_g4[segdf.iev.to_numpy()]].to_csv(os.path.join(outdir, "geomcheck_g4segments.csv.gz"), index=False)
    xc.to_csv(os.path.join(outdir, "geomcheck_xtru.csv"), index=False)


def _print(out):
    print(f"generated {out['generated']:,}  passes {out['pass']}  consistency {out['consistency']}  "
          f"G4 trajectories {out['g4_traj_coverage']}  tubes above interior {out['tubes_above_interior']}")
    print(f"materials without constants on the generated lines (mm): {out['unknown_material_mm_on_gen_lines']}")
    for lab in ("deficit_region", "all"):
        print(f"== {lab} {out[lab].get('p_gen', '')} {out[lab].get('gen_gcm2', '')}")
        for cl, x in out[lab]["classes"].items():
            print(f"  [{cl}]")
            for k, v in x.items():
                print(f"     {k:34s} {v}")
    print("cells: p_gen x gen column: g4 fast | fast fate of G4-only (not_near/stopped/missed/few) | G4-only with G4 plane state near / gen line near")
    for c in out["cells"]:
        f = c["g4_only_fate"]
        print(f"  {c['p_gen'][0]:6.0f}-{min(c['p_gen'][1], 1e6):7.0f} {c['gen_gcm2'][0]:4.0f}-{min(c['gen_gcm2'][1], 1e5):6.0f}"
              f"  {c['g4']:5d} {c['fast']:5d} | {f['not_near']:4d} {f['stopped_above']:4d} {f['missed']:4d} {f['few']:4d}"
              f" | {c['g4_only_g4near']:4d} {c['g4_only_gennear']:4d}  (fast-only {c['fast_only_n']})")
    for lab in ("band_residuals_g4pass", "band_residuals_g4pass_region"):
        print(f"{lab}: G4 loss vs fast-column Bethe along the G4 path per height band")
        for r in out[lab]:
            print(f"  y {r['y'][0]:7.0f}-{r['y'][1]:7.0f} nseg {r['nseg']:7d}  G4 {r['g4_loss_sum']:10.0f}  fast {r['fast_bethe_sum']:10.0f}"
                  f"  ratio {r['ratio']:.3f}  tube {r['tube_gcm2_sum']:.0f} g/cm2 | missing {r['missing_nseg']} ({r['missing_g4_loss']:.0f} MeV)"
                  f" at {r['missing_xyz_median']} | extra {r['extra_nseg']} (fast {r['extra_fast_bethe']:.0f}, G4 {r['extra_g4_loss']:.0f} MeV)"
                  f" at {r['extra_xyz_median']}")
    print("G4 loss / fast-column Bethe along the G4 path, by generated-line column:")
    for r in out["g4_loss_over_fast_bethe_by_column"]:
        print(f"  {r}")
    s = out["seeded_from_g4_plane_state"]
    print(f"fast interior+straws from the G4 state at the plane (all G4-kept events crossing it): {s['total']}")
    for r in s["by_p_plane"]:
        print(f"  {r}")
    print("plane -> first straw projected deflection / Highland theta0 (straight-line support x/X0): robust sigma, |.| q90 (Gauss 1, 1.645)")
    for r in s["deflection_vs_highland"]:
        print(f"  {r}")
    xc = out["xtru_census"]
    print(f"xtru census: {xc['n']} volumes, {xc['n_bad']} with ear-clipped area off by > 1 %:")
    for r in xc["bad"][:15]:
        print(f"  {r['volume']:36s} {r['material']:16s} bool {r['boolean']!s:5s} nvert {r['nvert']:3d} ntri {r['ntri']:3d}"
              f"  area {r['area_poly_m2']:9.2f} -> {r['area_pieces_m2']:9.2f} m2  excess {r['excess_t']:9.1f} t")
