"""arrivalcheck -- the muon state at the plane above the tracker, fast against G4, on the same CRY primaries.

geomcheck compares only events that pass the tracker term in one arm. Here the population is every generated primary
muon whose G4 trajectory crosses the plane (`trajpts`, i.e. every event the production filter kept, mostly through the
CRV term, which is independent of the tracker), so muons that arrive but miss the straws count too. For each such
event the fast stage C transport to the plane (steps.to_plane with the `steps` seed convention) is rerun on the same
primary, and per energy-margin bin (margin = p_gen / (2.1 MeV cm2/g x generated-line column), as survivalcheck) it
reports:
  - arrival: share of G4-arriving muons the fast transport delivers alive to the plane, among those it transports
    (generated line near the straw footprint, steps.near_tracker; the rest are not transported by design and are
    counted separately, with the G4 muons among them that are aimed at the footprint);
  - |p| at the plane: q10/q50/q90, G4 vs fast (fast alive only), and the share below 300 MeV/c;
  - the lateral offset from the generated line and the deflection from the generated direction at the plane;
  - the share whose state at the plane points into the straw footprint (worldcheck._miss_distance == 0), both arms.
G4 delivers some muons the fast transport stops (fluctuations, path), and some that it bends into the tracker; a margin
row with arrival ~1 but fewer fast muons aimed at the footprint points at scattering, arrival < 1 at range-out.
Variants (diagnostics): --no-sliced (one geometric-mean |p| for the column's scattering), --no-msc-tail, --band-mm
(uniform height bands of that size instead of steps.BAND_EDGES_Y); --crv CRVSTEPS keeps only the events whose G4
CrvSteps pass the production CRV term (free of the tracker-term selection). Selection caveat: the G4 events are those the production filter kept, so the population is biased towards muons that
reach a detector; the comparison is per event, so both arms see the same events.

    arrivalcheck "crv_fastsim/g4ref_v1_tables/gen.001400_*.root" --compare "trk_fastsim/g4ref_v1_tables/sgsteps.*.root" \
        --outdir DIR [--seed 1]
"""
import os

import numpy as np
import pandas as pd

from mu2efastsim import plots
from mu2efastsim.tables import KEY
from .. import geometry, steps as st, tables as T
from .geomcheck import _g4_paths, _p_of_ke
from .worldcheck import _column, _miss_distance

MARGIN_EDGES = np.array([0.0, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 1e9])
DEDX_REF = 2.1


def _q(x, qs=(0.1, 0.5, 0.9)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [round(float(v), 1) for v in np.quantile(x, qs)] if len(x) > 10 else None


def run(inputs, outdir, compare=None, gdml=None, seed=1, no_sliced=False, no_msc_tail=False, band_mm=None, crv=None, **kw):
    if not compare:
        raise SystemExit("arrivalcheck GEN --compare G4REF")
    from .steps import default_gdml
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    if band_mm:                       # diagnostic: uniform height bands of band_mm above the plane instead of BAND_EDGES_Y
        st.BAND_EDGES_Y = np.arange(st.BAND_EDGES_Y[0], st.plane_y(geom), -float(band_mm))
    world = st.load_world(gdml, geom)
    y_pl = st.plane_y(geom)
    g4_paths = T.expand(compare)
    rows = []
    for i, f in enumerate(T.expand(inputs)):
        gen = T.read(f, "gen")
        mu, o, d, p, L, alive = st.to_plane(gen, geom, np.random.default_rng([int(seed), i]), world, sliced_msc=not no_sliced,
                                            msc_tail=not no_msc_tail)
        df = mu[KEY + ["x", "y", "z", "px", "py", "pz"]].copy()
        df["f_alive"] = alive
        df["f_p"] = np.where(alive, p, np.nan)
        df[["f_x", "f_y", "f_z"]] = o
        df[["f_ux", "f_uy", "f_uz"]] = d
        df["f_miss"] = np.where(alive, _miss_distance(o, d, geom, y_pl), np.nan)
        rows.append(df)
    fe = pd.concat(rows, ignore_index=True)
    ev = T.read_many(g4_paths, "gen", KEY).drop_duplicates().merge(fe, on=KEY)
    ev = ev.reset_index(drop=True)
    ev["iev"] = np.arange(len(ev))
    seg = _g4_paths(g4_paths, ev, y_pl)
    jl = np.flatnonzero(seg["last"])
    e = seg["iev"][jl]
    ev["g4_arrive"] = False
    ev.loc[e, "g4_arrive"] = True
    u = seg["b"][jl] - seg["a"][jl]
    u /= np.linalg.norm(u, axis=1)[:, None]
    ev.loc[e, ["g_x", "g_y", "g_z"]] = seg["b"][jl]
    ev.loc[e, ["g_ux", "g_uy", "g_uz"]] = u
    ev.loc[e, "g_p"] = _p_of_ke(seg["kb"][jl])
    ev = ev[ev.g4_arrive].reset_index(drop=True)
    if crv:                           # keep the events the G4 CRV term passed (kept independently of the tracker)
        from .seedcheck import crv_pass_from
        ev = ev.merge(crv_pass_from(T.read_many(T.expand(crv), "steps")), on=KEY)
    p0 = np.sqrt(ev.px ** 2 + ev.py ** 2 + ev.pz ** 2).to_numpy()
    ev["f_near"] = st.near_tracker(ev[["x", "y", "z"]].to_numpy(), ev[["px", "py", "pz"]].to_numpy() / p0[:, None], geom, y_pl)
    ev["g_miss"] = _miss_distance(ev[["g_x", "g_y", "g_z"]].to_numpy(), ev[["g_ux", "g_uy", "g_uz"]].to_numpy(), geom, y_pl)
    ev["p_gen"] = np.sqrt(ev.px ** 2 + ev.py ** 2 + ev.pz ** 2)
    ev["gcm2"], _ = _column(ev, world, y_pl)
    ev["margin"] = ev.p_gen / (DEDX_REF * np.maximum(ev.gcm2, 1e-3))
    d0 = ev[["px", "py", "pz"]].to_numpy() / ev.p_gen.to_numpy()[:, None]
    o0 = ev[["x", "y", "z"]].to_numpy()
    xg = o0 + d0 * ((o0[:, 1] - y_pl) / -d0[:, 1])[:, None]
    for a in ("g", "f"):
        pos = ev[[f"{a}_x", f"{a}_y", f"{a}_z"]].to_numpy()
        dirs = ev[[f"{a}_ux", f"{a}_uy", f"{a}_uz"]].to_numpy()
        ev[f"{a}_offset"] = np.hypot(pos[:, 0] - xg[:, 0], pos[:, 2] - xg[:, 2])
        ev[f"{a}_defl"] = 1e3 * np.arccos(np.clip(np.einsum("ij,ij->i", dirs, d0), -1, 1))
    ev.loc[~ev.f_alive, ["f_offset", "f_defl"]] = np.nan
    out = {"events_g4_arriving": int(len(ev)), "share_not_near": round(float(np.mean(~ev.f_near)), 4), "rows": []}
    b = np.clip(np.digitize(ev.margin, MARGIN_EDGES) - 1, 0, len(MARGIN_EDGES) - 2)
    for k in range(len(MARGIN_EDGES) - 1):
        gall = ev[b == k]
        g = gall[gall.f_near]
        if len(g) < 20:
            continue
        fa = g[g.f_alive]
        out["rows"].append({
            "margin": [float(MARGIN_EDGES[k]), float(MARGIN_EDGES[k + 1])], "n": int(len(g)),
            "not_near_share": round(float(1 - len(g) / len(gall)), 4),
            "aimed_g4_not_near": int(((gall.g_miss <= 0) & ~gall.f_near).sum()),
            "fast_arrival": round(float(g.f_alive.mean()), 4),
            "p_plane_q_g4": _q(g.g_p), "p_plane_q_fast": _q(fa.f_p),
            "below300_g4": round(float(np.mean(g.g_p < 300)), 4),
            "below300_fast_of_all": round(float(np.mean(g.f_alive & (g.f_p < 300))), 4),
            "offset_q_g4": _q(g.g_offset), "offset_q_fast": _q(fa.f_offset),
            "defl_q_g4": _q(g.g_defl), "defl_q_fast": _q(fa.f_defl),
            "aimed_g4": round(float(np.mean(g.g_miss <= 0)), 4),
            "aimed_fast_of_all": round(float(np.mean(g.f_alive & (g.f_miss <= 0))), 4)})
    os.makedirs(outdir, exist_ok=True)
    ev.to_csv(os.path.join(outdir, "arrivalcheck_events.csv.gz"), index=False)
    plots.write_json(outdir, "arrivalcheck.json", out)
    print(f"G4 primary muons crossing the plane: {len(ev):,}; generated line not near the footprint: {out['share_not_near']:.3f}")
    print(f"{'margin':>12s} {'n':>6s} | arrive | |p| plane q10/q50/q90 G4 | fast | <300 G4/fast | "
          f"offset q10/q50/q90 G4 | fast | defl q50 G4/fast | aimed G4/fast")
    for r in out["rows"]:
        print(f"{str(r['margin']):>12s} {r['n']:6d} | {r['fast_arrival']:.3f} | {r['p_plane_q_g4']} | {r['p_plane_q_fast']} | "
              f"{r['below300_g4']:.3f}/{r['below300_fast_of_all']:.3f} | {r['offset_q_g4']} | {r['offset_q_fast']} | "
              f"{r['defl_q_g4'][1] if r['defl_q_g4'] else None}/{r['defl_q_fast'][1] if r['defl_q_fast'] else None} | "
              f"{r['aimed_g4']:.3f}/{r['aimed_fast_of_all']:.3f} | not near {r['not_near_share']:.3f}, of them aimed in G4 {r['aimed_g4_not_near']}")
