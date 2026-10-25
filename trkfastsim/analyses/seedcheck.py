"""seedcheck -- is the tracker-interior scattering excess of geomcheck's seeded comparison real or an artefact?

geomcheck (seeded_from_g4_plane_state) starts the fast interior from the G4 muon state at the plane above the tracker
and finds the fast deflection plane -> first straw 1.2-1.5x G4's (in units of Highland theta0). Three possible
artefacts of that diagnostic, each removed by one variant here (all on the same G4 events and the same fast code):
  - start: geomcheck moves the fast seed back up to the ring tops (steps._back_to_interior_top), so the fast arm crosses
    the ring material above the plane that G4's plane state has already crossed. Variants "top" (as geomcheck) and
    "plane" (fast starts at the plane).
  - direction: the G4 reference direction is the trajectory chord through the plane (points >= 15 mm apart), which can
    already hold part of the deflection. Variant "envelope": seed from G4's primary SurfaceStep where it first enters
    the tracker envelope (TT_OutSurf r 850 mm, TT_Front/TT_Back planes; above all interior material), position and
    momentum of a point state, so both arms cross the same material from the same state.
  - selection: the G4 reference holds only events that passed the production PrimaryFilter (tracker OR CRV term), so
    seeded events G4 failed but the fast arm would pass are missing unless the CRV term kept them. Subset "crv":
    events whose G4 CrvSteps pass the CRV term (visible edep > 0.1 MeV, 50 < |p| < 1e6 MeV/c), kept independently
    of the tracker.
Per variant and subset: tracker-term passes (G4, fast seeded, straight line from the seed), straws per muon, and the
projected deflection seed -> first straw over the Highland theta0 of the straight-line support x/X0 (robust sigma and
|.| q90; Gaussian 1, 1.645), plus the G4/fast width ratio. Also the angle between the G4 chord at the plane and the
G4 envelope direction (the chord artefact on the G4 side alone).

    seedcheck "trk_fastsim/g4ref_v1_tables/sgsteps.*.root" --crv "crv_fastsim/g4ref_v1_tables/crvsteps.*.root" \
        --outdir DIR [--gdml GDML] [--seed 1]
"""
import os

import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials, plots
from mu2efastsim.tables import KEY
from .. import crossings, geometry, steps as st, tables as T
from .geomcheck import _g4_paths, _ke_of_p, _p_of_ke, _primary_pass

ENVELOPE_SIDS = (0, 2, 4)          # TT_Front, TT_Back, TT_Outer: the outer boundary of the tracker VDs
XX0_BINS = ((0.05, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 10.0))
P_BINS = ((0.0, 1000.0), (1000.0, 3000.0), (3000.0, 1e9))
COUNT_P_EDGES = np.array([0.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 3000.0, 1e9])
MIN_N = 30                         # events per (x/X0, |p|) cell to report a width
ROBUST = 1.4826                    # median |x| -> Gaussian sigma


def crv_pass_from(steps):
    """Events whose CrvSteps pass the production CRV term (any particle, one step suffices): DataFrame KEY + crv_pass."""
    p = np.sqrt(steps.px0 ** 2 + steps.py0 ** 2 + steps.pz0 ** 2)
    ok = (steps.edep > T.FILTER_MIN_CRV_EDEP) & (p > T.FILTER_MIN_MOM) & (p < T.FILTER_MAX_MOM)
    return steps.loc[ok, KEY].drop_duplicates().assign(crv_pass=True)


def _primaries(paths):
    pa = T.read_many(paths, "particles", KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    return pa[(pa.ismcprim == 1) & (pa.pdg.abs() == 13)].drop_duplicates(KEY)[KEY + ["sp"]]


def envelope_from(ss, pm, ev):
    """First (in time) primary-muon SurfaceStep on the tracker envelope per event: (iev, origin Mu2e frame, unit
    direction, |p|, sid). ss: surfsteps (detector frame), pm: primary (KEY + sp), ev: events (KEY + iev)."""
    s = ss[ss.surf.isin(ENVELOPE_SIDS)].merge(pm, on=KEY + ["sp"]).merge(ev[KEY + ["iev"]], on=KEY)
    s = s.sort_values(["iev", "t"]).drop_duplicates("iev")
    mid = 0.5 * (s[["x0", "y0", "z0"]].to_numpy(float) + s[["x1", "y1", "z1"]].to_numpy(float))
    pv = s[["px", "py", "pz"]].to_numpy(float)
    p = np.linalg.norm(pv, axis=1)
    return s.iev.to_numpy(), geometry.to_mu2e(mid), pv / np.maximum(p, 1e-12)[:, None], p, s.surf.to_numpy()


def chord_seeds(g4_paths, ev, y_pl):
    """geomcheck's seed: G4 state at the first downward plane crossing, direction of the trajectory chord through it."""
    seg = _g4_paths(g4_paths, ev, y_pl)
    jl = np.flatnonzero(seg["last"])
    o = seg["b"][jl].copy()
    o[:, 1] = y_pl
    u = seg["b"][jl] - seg["a"][jl]
    u /= np.linalg.norm(u, axis=1)[:, None]
    return seg["iev"][jl], o, u, _p_of_ke(seg["kb"][jl])


def _g4_first_dir(g4_paths, pm, ev):
    """Unit momentum [n_ev, 3] and entry point (Mu2e frame) [n_ev, 3] of each event's first (in time) primary-muon
    StrawGasStep (NaN where none)."""
    s = T.read_many(g4_paths, "sgsteps").merge(pm, on=KEY + ["sp"]).merge(ev[KEY + ["iev"]], on=KEY)
    f = s.sort_values(["iev", "t"]).drop_duplicates("iev")
    out = np.full((len(ev), 3), np.nan)
    pos = np.full((len(ev), 3), np.nan)
    v = f[["px", "py", "pz"]].to_numpy(float)
    out[f.iev.to_numpy()] = v / np.linalg.norm(v, axis=1)[:, None]
    pos[f.iev.to_numpy()] = geometry.to_mu2e(f[["x0", "y0", "z0"]].to_numpy(float))
    return out, pos


def _g4_end(g4_paths, pm, ev, geom, pad=50.0):
    """Per event: G4 primary muon stopped (end kinetic energy < 1 MeV) inside the straw panels' box (+ pad)."""
    pa = T.read_many(g4_paths, "particles", KEY + ["key", "ex", "ey", "ez", "epx", "epy", "epz"]).rename(columns={"key": "sp"})
    e = pa.merge(pm, on=KEY + ["sp"]).merge(ev[KEY + ["iev"]], on=KEY)
    lo, hi = crossings.panel_boxes(geom)
    x = e[["ex", "ey", "ez"]].to_numpy(float)
    inside = np.all(x > lo.min(0) - pad, axis=1) & np.all(x < hi.max(0) + pad, axis=1)
    ke = _ke_of_p(np.linalg.norm(e[["epx", "epy", "epz"]].to_numpy(float), axis=1))
    out = np.zeros(len(ev), bool)
    out[e.iev.to_numpy()] = inside & (ke < 1.0)
    return out


def seeded(rng, interior, geom, o, u, p, start="as_is", **transport_kw):
    """Fast interior + straws from seeds (o, u, p). start "top": move plane seeds back to the ring tops (geomcheck).
    Returns per seed: fast good-step count, straight-line straw count, straight-line support x/X0 to the first straw,
    fast direction at the first straw (NaN if none), fast first-straw entry point (Mu2e frame, NaN if none), lowest
    fast |p| over the recorded crossings (NaN if none). transport_kw go to TrackerInterior.transport."""
    n = len(o)
    alive = p > 1.0
    oi = st._back_to_interior_top(o, u, np.zeros(n), alive, interior, geom)[0] if start == "top" else o
    sel = np.flatnonzero(alive)
    tr_ = interior.transport(rng, oi[sel], u[sel], p[sel], geom, **transport_kw)
    pk = _p_of_ke(_ke_of_p(p[sel][tr_["it"]]) - tr_["lost"])
    pmin = np.full(n, np.inf)
    np.minimum.at(pmin, sel[tr_["it"]], pk)
    pmin[~np.isfinite(pmin)] = np.nan
    n_seed = np.zeros(n, int)
    np.add.at(n_seed, sel[tr_["it"][pk > T.FILTER_MIN_MOM]], 1)
    xs = crossings.straw_crossings(oi, u, geom)
    xs = xs[alive[xs[:, 0].astype(np.int64)]] if len(xs) else xs
    it = xs[:, 0].astype(np.int64)
    n_st = np.bincount(it, minlength=n)
    s1 = np.full(n, np.inf)
    np.minimum.at(s1, it, xs[:, 2])
    col = interior.shells.column(oi, u, np.where(np.isfinite(s1), s1, 0.0), nsub=interior.nsub)
    xx0 = sum((c / eloss.radiation_length_mm(materials.get(k)) for k, c in col.items()),
              np.zeros(n))
    vf = np.full((n, 3), np.nan)
    xf = np.full((n, 3), np.nan)
    if len(tr_["it"]):
        first = np.r_[True, tr_["it"][1:] != tr_["it"][:-1]]
        vf[sel[tr_["it"][first]]] = tr_["dir"][first]
        xf[sel[tr_["it"][first]]] = tr_["pos0"][first]
    return n_seed, n_st, xx0, vf, xf, pmin


def projected_over_theta0(v, u, p, xx0):
    """Projected angles of v about u in two transverse planes, over Highland theta0(p, xx0): [n, 2] (NaN-safe)."""
    ref = np.where(np.abs(u[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(u, e1)
    th0 = eloss.highland_theta0(p, np.maximum(xx0, 1e-9))
    v = v / np.linalg.norm(v, axis=1)[:, None]
    c = np.einsum("ij,ij->i", v, u)
    return np.c_[np.einsum("ij,ij->i", v, e1) / c, np.einsum("ij,ij->i", v, e2) / c] / th0[:, None]


def transverse_offset(x, o, u):
    """Offset [n, 2] of points x from the seed lines (o, u) in the two transverse directions of projected_over_theta0."""
    ref = np.where(np.abs(u[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(u, e1)
    w = x - o
    return np.c_[np.einsum("ij,ij->i", w, e1), np.einsum("ij,ij->i", w, e2)]


def offset_table(df):
    """Robust width and q90 of the first-straw entry offset from the seed line [mm], per x/X0 and |p| bin, G4 vs fast."""
    rows = []
    for a, b in XX0_BINS:
        for plo, phi in P_BINS:
            m = ((df.xx0 >= a) & (df.xx0 < b) & (df.p_seed >= plo) & (df.p_seed < phi)
                 & np.isfinite(df[["dg1", "dg2", "df1", "df2"]]).all(axis=1)).to_numpy()
            if m.sum() < MIN_N:
                continue
            g, f = _rs(df.loc[m, ["dg1", "dg2"]].to_numpy()), _rs(df.loc[m, ["df1", "df2"]].to_numpy())
            rows.append({"xx0": [a, b], "p": [plo, phi], "n": int(m.sum()), "g4_sigma_q90": list(g),
                         "fast_sigma_q90": list(f), "g4_over_fast_sigma": round(g[0] / f[0], 3) if f[0] else None})
    return rows


def fate_table(df):
    """G4-only events (G4 passes, seeded fast fails) by cause: straight line also fails; fast reached < FILTER_MIN_MOM
    (ranged out, or its steps stop counting); G4's own muon stopped in the tracker; the rest (scattered off the straws)."""
    g = df[df.g4_pass & ~df.seed_pass]
    slow = g.fast_pmin < T.FILTER_MIN_MOM * 1.5
    return {"n": int(len(g)), "straight_fails": int((~g.straight_pass).sum()),
            "fast_slow_pmin_lt_75": int((g.straight_pass & slow).sum()),
            "fast_no_crossing": int((g.straight_pass & g.fast_pmin.isna()).sum()),
            "other": int((g.straight_pass & ~slow & g.fast_pmin.notna()).sum()),
            "g4_muon_stopped_in_tracker": int(g.g4_stop.sum()),
            "all_g4_pass_g4_stopped": int((df.g4_pass & df.g4_stop).sum()),
            "fast_only_n": int((df.seed_pass & ~df.g4_pass).sum())}


def _rs(r):
    a = np.abs(r[np.isfinite(r).all(1)])
    return (round(float(ROBUST * np.median(a)), 3), round(float(np.quantile(a, 0.9)), 3)) if len(a) else (None, None)


def width_table(df):
    rows = []
    for a, b in XX0_BINS:
        for plo, phi in P_BINS:
            m = ((df.xx0 >= a) & (df.xx0 < b) & (df.p_seed >= plo) & (df.p_seed < phi)
                 & np.isfinite(df[["rg1", "rg2", "rf1", "rf2"]]).all(axis=1)).to_numpy()
            if m.sum() < MIN_N:
                continue
            g, f = _rs(df.loc[m, ["rg1", "rg2"]].to_numpy()), _rs(df.loc[m, ["rf1", "rf2"]].to_numpy())
            rows.append({"xx0": [a, b], "p": [plo, phi], "n": int(m.sum()), "g4_sigma_q90": list(g),
                         "fast_sigma_q90": list(f), "g4_over_fast_sigma": round(g[0] / f[0], 3) if f[0] else None})
    return rows


def count_table(df):
    tot = {"n": int(len(df)), "g4_pass": int(df.g4_pass.sum()), "seed_pass": int(df.seed_pass.sum()),
           "straight_pass": int(df.straight_pass.sum()), "g4_only": int((df.g4_pass & ~df.seed_pass).sum()),
           "seed_only": int((~df.g4_pass & df.seed_pass).sum())}
    tot["seed_over_g4"] = round(tot["seed_pass"] / max(tot["g4_pass"], 1), 4)
    tot["straight_over_g4"] = round(tot["straight_pass"] / max(tot["g4_pass"], 1), 4)
    hit = df.ngood_g4 > 0
    tot["mean_straws_g4_seed_straight"] = [round(float(df.loc[hit, c].mean()), 3) for c in ("ngood_g4", "n_seed", "n_straight")]
    b = np.clip(np.digitize(df.p_seed, COUNT_P_EDGES) - 1, 0, len(COUNT_P_EDGES) - 2)
    rows = []
    for k in range(len(COUNT_P_EDGES) - 1):
        x = df[b == k]
        if len(x):
            rows.append({"p": [float(COUNT_P_EDGES[k]), float(COUNT_P_EDGES[k + 1])], "n": int(len(x)),
                         "g4_pass": int(x.g4_pass.sum()), "seed_pass": int(x.seed_pass.sum()),
                         "straight_pass": int(x.straight_pass.sum())})
    return {"total": tot, "by_p_seed": rows}


def run(inputs, outdir, crv=None, gdml=None, seed=1, lever_moments=False, kink_sup_xx0=st.KINK_SUP_XX0, kink_p_mean=False, straw_var_scale=1.0, **kw):
    if not crv:
        raise SystemExit("seedcheck G4SGSTEPS --crv G4CRVSTEPS")
    from .steps import default_gdml
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    y_pl = st.plane_y(geom)
    interior = st.TrackerInterior(gdml, geom)
    g4_paths = T.expand(inputs)
    ev = T.read_many(g4_paths, "gen", KEY).drop_duplicates().reset_index(drop=True)
    ev["iev"] = np.arange(len(ev))
    gf = _primary_pass(g4_paths)
    ev = ev.merge(gf, on=KEY, how="left").merge(crv_pass_from(T.read_many(T.expand(crv), "steps")), on=KEY, how="left")
    ev["g4_pass"] = ev.pass_trk.fillna(False).astype(bool)
    ev["crv_pass"] = ev.crv_pass.fillna(False).astype(bool)
    ev["ngood_g4"] = ev.nmax.fillna(0).astype(int)
    ev = ev.sort_values("iev").reset_index(drop=True)
    pm = _primaries(g4_paths)
    v_g4, x_g4 = _g4_first_dir(g4_paths, pm, ev)
    g4_stop = _g4_end(g4_paths, pm, ev, geom)
    filt = {"events": int(len(ev)), "trk": int(ev.g4_pass.sum()), "crv": int(ev.crv_pass.sum()),
            "trk_and_crv": int((ev.g4_pass & ev.crv_pass).sum()), "neither": int((~ev.g4_pass & ~ev.crv_pass).sum())}
    ec, oc, uc, pc = chord_seeds(g4_paths, ev, y_pl)
    ee, oe, ue, pe, sid = envelope_from(T.read_many(g4_paths, "surfsteps"), pm, ev)
    down = uc[:, 1] < 0
    variants = {"top": (ec[down], oc[down], uc[down], pc[down], "top"),
                "plane": (ec[down], oc[down], uc[down], pc[down], "as_is"),
                "envelope": (ee, oe, ue, pe, "as_is")}
    out = {"filter_terms_g4ref": filt, "envelope_sid_counts": {int(k): int(v) for k, v in zip(*np.unique(sid, return_counts=True))},
           "variants": {}}
    frames = []
    for i, (name, (e, o, u, p, start)) in enumerate(variants.items()):
        tkw = {"lever_moments": True} if lever_moments else {}
        tkw["kink_sup_xx0"] = kink_sup_xx0
        if kink_p_mean:
            tkw["kink_p_mean"] = True
        if straw_var_scale != 1.0:
            tkw["straw_var_scale"] = straw_var_scale
        n_seed, n_st, xx0, vf, xf, pmin = seeded(np.random.default_rng([int(seed), 77, i]), interior, geom, o, u, p, start, **tkw)
        df = ev.iloc[e][KEY + ["iev", "g4_pass", "crv_pass", "ngood_g4"]].reset_index(drop=True)
        df["variant"], df["p_seed"], df["xx0"], df["n_seed"], df["n_straight"] = name, p, xx0, n_seed, n_st
        df["seed_pass"] = n_seed >= T.FILTER_MIN_TRK_STEPS
        df["straight_pass"] = n_st >= T.FILTER_MIN_TRK_STEPS
        rg = projected_over_theta0(v_g4[e], u, p, xx0)
        rf = projected_over_theta0(vf, u, p, xx0)
        df[["rg1", "rg2"]], df[["rf1", "rf2"]] = rg, rf
        df[["dg1", "dg2"]], df[["df1", "df2"]] = transverse_offset(x_g4[e], o, u), transverse_offset(xf, o, u)
        df["g4_stop"], df["fast_pmin"] = g4_stop[e], pmin
        out["variants"][name] = {sub: {"counts": count_table(d), "deflection_over_theta0": width_table(d),
                                       "first_straw_offset_mm": offset_table(d), "g4_only_fate": fate_table(d)}
                                 for sub, d in (("all", df), ("crv", df[df.crv_pass]))}
        frames.append(df)
    # the chord artefact on the G4 side: angle between the plane chord and the envelope point direction (same events)
    m = pd.Series(np.arange(len(ec)), index=ec).reindex(ee).to_numpy()
    ok = np.isfinite(m)
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", ue[ok], uc[m[ok].astype(int)]), -1, 1)) * 1e3
    gc = np.arccos(np.clip(np.einsum("ij,ij->i", v_g4[ee[ok]], uc[m[ok].astype(int)]), -1, 1)) * 1e3
    ge = np.arccos(np.clip(np.einsum("ij,ij->i", v_g4[ee[ok]], ue[ok]), -1, 1)) * 1e3
    q = lambda x: [round(float(v), 2) for v in np.nanquantile(x, (0.5, 0.9))]
    out["g4_chord_vs_envelope_mrad"] = {"n": int(ok.sum()), "chord_to_envelope_q50_q90": q(ang),
                                        "g4_defl_from_chord_q50_q90": q(gc), "g4_defl_from_envelope_q50_q90": q(ge)}
    os.makedirs(outdir, exist_ok=True)
    pd.concat(frames).to_csv(os.path.join(outdir, "seedcheck_events.csv.gz"), index=False)
    plots.write_json(outdir, "seedcheck.json", out)
    _print(out)


def _print(out):
    print(f"G4 reference filter terms (primary muon tracker term, any-particle CRV term): {out['filter_terms_g4ref']}")
    print(f"envelope entry sid counts (0 front, 2 back, 4 outer): {out['envelope_sid_counts']}")
    print(f"G4 chord at the plane vs envelope direction [mrad]: {out['g4_chord_vs_envelope_mrad']}")
    for name, v in out["variants"].items():
        for sub, r in v.items():
            print(f"== {name} / {sub}: {r['counts']['total']}")
            for c in r["counts"]["by_p_seed"]:
                print(f"   {c}")
            print("   deflection seed -> first straw over Highland theta0: robust sigma, |.| q90 (Gauss 1, 1.645)")
            for w in r["deflection_over_theta0"]:
                print(f"   {w}")
            print("   first-straw entry offset from the seed line [mm]: robust sigma, |.| q90")
            for w in r["first_straw_offset_mm"]:
                print(f"   {w}")
            print(f"   G4-only fate: {r['g4_only_fate']}")
