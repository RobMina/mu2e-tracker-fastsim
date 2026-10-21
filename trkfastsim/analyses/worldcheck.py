"""worldcheck -- the world model event by event: muon |p| at the first straw against |p| at generation, fast vs G4.

Both arms ran the same CRY primaries, so events are paired by (run, subrun, event); the denominator is every generated
event (the fast tables' gen tree; the G4 tables hold only PrimaryFilter-accepted events, a superset of the tracker-term
passes, so an event missing there did not pass in G4). Per event: generated muon |p| and
direction (gen tree, identical in both arms), the GDML material column along the generated straight line down to the
plane above the tracker (concrete/overburden g/cm2, total x/X0), and, where the muon passes the PrimaryFilter tracker
term (>= 12 good steps; every such event is in the G4 tables), |p| at its first straw.

Reported:
  1. delivery: P(tracker term passes) per generated-|p| bin and per column bin, fast against G4 (same generated
     events, so the ratio is a pure transport comparison);
  2. energy loss on events passing in both arms: loss = p_gen - p_first, fast against G4, per generated-|p| and
     column bin (median and mean, and the paired median difference);
  3. the unpaired events (pass in one arm only): their generated |p| and column, to locate what one arm delivers and
     the other does not;
  4. transverse state at the first straw per generated-|p| x column cell (each arm on its own passes, and paired):
     lateral offset of the first-straw entry from the generated straight line at the same height, deflection angle
     from the generated direction, and the per-plane offset-angle correlation (rho = sqrt(3)/2 for one uniform slab
     ending at the plane), to test how the world scattering is spread along the column.
"""
import os

import numpy as np
import pandas as pd

from mu2efastsim import materials, plots
from mu2efastsim import world as world_mod
from mu2efastsim.tables import primary_muons
from .. import crossings, geometry, steps as st, tables as T

P_EDGES = np.array([0.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0, 1e9])   # MeV/c, generated
COL_EDGES = np.array([0.0, 50.0, 150.0, 300.0, 450.0, 600.0, 1e9])                               # g/cm2 above the plane


def _arm(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="inner")
    s = s[(s.ismcprim == 1) & (s.pdg.abs() == 13)]
    f = T.trk_filter(s)
    s = s.merge(f[f.pass_trk].reset_index()[T.KEY], on=T.KEY)
    first = s.sort_values("t").groupby(T.KEY, sort=False).first().reset_index()
    first["p_first"] = np.sqrt(first.px ** 2 + first.py ** 2 + first.pz ** 2)
    pos = geometry.to_mu2e(first[["x0", "y0", "z0"]].to_numpy())   # step positions are detector frame, gen is Mu2e
    first["x_first"], first["y_first"], first["z_first"] = pos[:, 0], pos[:, 1], pos[:, 2]
    for c in ("px", "py", "pz"):
        first["u" + c[1]] = first[c] / first.p_first
    return first[T.KEY + ["p_first", "x_first", "y_first", "z_first", "ux", "uy", "uz"]]


TP_EDGES = np.array([0.0, 500.0, 750.0, 1000.0, 1500.0, 3000.0, 1e9])   # MeV/c, generated (as survivalcheck)
TC_EDGES = np.array([0.0, 50.0, 150.0, 300.0, 450.0, 1e9])               # g/cm2 (as survivalcheck)


def _transverse(ev, sfx):
    """Per event: lateral offset [mm] of the first-straw entry from the generated line at the same height, its two
    transverse components, the deflection components [rad] in the same basis, the space angle and the path length."""
    o = ev[["x", "y", "z"]].to_numpy()
    d0 = ev[["px", "py", "pz"]].to_numpy() / ev.p_gen.to_numpy()[:, None]
    xf = ev[[f"x_first{sfx}", f"y_first{sfx}", f"z_first{sfx}"]].to_numpy()
    uf = ev[[f"ux{sfx}", f"uy{sfx}", f"uz{sfx}"]].to_numpy()
    L = (o[:, 1] - xf[:, 1]) / -d0[:, 1]
    dl = xf - (o + d0 * L[:, None])
    dl -= (dl * d0).sum(1)[:, None] * d0
    ref = np.where(np.abs(d0[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    u = np.cross(d0, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d0, u)
    cth = np.clip((uf * d0).sum(1), -1, 1)
    return {"r": np.linalg.norm(dl, axis=1), "yu": (dl * u).sum(1), "yv": (dl * v).sum(1),
            "tu": (uf * u).sum(1) / cth, "tv": (uf * v).sum(1) / cth, "theta": np.arccos(cth), "L": L}


def _tstats(t, m):
    if m.sum() < 20:
        return None
    y = np.r_[t["yu"][m], t["yv"][m]]
    a = np.r_[t["tu"][m], t["tv"][m]]
    q = lambda x: np.quantile(x, [0.5, 0.9, 0.99]).round(2).tolist()
    return {"n": int(m.sum()), "r_mm_q": q(t["r"][m]), "theta_mrad_q": q(1e3 * t["theta"][m]),
            "rho": float(np.corrcoef(y, a)[0, 1]),
            "y_over_thetaL_rms": float(np.sqrt(np.mean(y ** 2)) / max(np.sqrt(np.mean((a * np.r_[t["L"][m], t["L"][m]]) ** 2)), 1e-9)),
            "L_median_mm": float(np.median(t["L"][m]))}


def _transverse_table(ev, pf, pg):
    """Transverse first-straw statistics per generated-|p| x column cell: fast on its passes, G4 on its passes, and
    both arms on the events passing in both (paired)."""
    tf = _transverse(ev[pf].reset_index(drop=True), "")
    tg = _transverse(ev[pg].reset_index(drop=True), "_g4")
    both = ev[pf & pg].reset_index(drop=True)
    tbf, tbg = _transverse(both, ""), _transverse(both, "_g4")
    rows = []
    for i in range(len(TP_EDGES) - 1):
        for j in range(len(TC_EDGES) - 1):
            cell = lambda x: ((x.p_gen >= TP_EDGES[i]) & (x.p_gen < TP_EDGES[i + 1]) & (x.gcm2 >= TC_EDGES[j])
                              & (x.gcm2 < TC_EDGES[j + 1])).to_numpy()
            r = {"p_gen": [float(TP_EDGES[i]), float(min(TP_EDGES[i + 1], 1e6))],
                 "gcm2": [float(TC_EDGES[j]), float(min(TC_EDGES[j + 1], 1e5))],
                 "fast": _tstats(tf, cell(ev[pf])), "g4": _tstats(tg, cell(ev[pg])),
                 "paired_fast": _tstats(tbf, cell(both)), "paired_g4": _tstats(tbg, cell(both))}
            if r["g4"] is not None:
                rows.append(r)
    return rows


MISS_EDGES = np.array([0.0, 1e-6, 300.0, 1000.0, 1500.0, 3000.0, 1e9])   # mm from the footprint at the plane


def _miss_distance(o, d, geom, y_plane):
    """Distance [mm] in x-z of the straight line's crossing of y = y_plane from the straws' footprint (0 inside);
    upward tracks get 1e9."""
    lo, hi = crossings.panel_boxes(geom)
    lo, hi = lo.min(0), hi.max(0)
    down = d[:, 1] < 0
    L = np.where(down, (o[:, 1] - y_plane) / -np.where(down, d[:, 1], -1.0), 0.0)
    x, z = o[:, 0] + d[:, 0] * L, o[:, 2] + d[:, 2] * L
    dx = np.maximum(np.maximum(lo[0] - x, x - hi[0]), 0.0)
    dz = np.maximum(np.maximum(lo[2] - z, z - hi[2]), 0.0)
    return np.where(down, np.hypot(dx, dz), 1e9)


def _column(gen_mu, world, y_plane):
    o = gen_mu[["x", "y", "z"]].to_numpy()
    p = np.sqrt(gen_mu.px ** 2 + gen_mu.py ** 2 + gen_mu.pz ** 2).to_numpy()
    d = gen_mu[["px", "py", "pz"]].to_numpy() / p[:, None]
    down = d[:, 1] < 0
    L = np.where(down, (o[:, 1] - y_plane) / -np.where(down, d[:, 1], -1.0), 0.0)
    col = world_mod.column(world.pieces() if hasattr(world, "pieces") else world, o, d, L)
    gcm2 = np.zeros(len(o))
    xx0 = np.zeros(len(o))
    from mu2efastsim import eloss
    for name, c in col.items():
        m = materials.BY_NAME.get(name, materials.CONCRETE_MARS)
        gcm2 += c * m.rho / 10.0
        xx0 += c / eloss.radiation_length_mm(m)
    return gcm2, xx0


def _rate_table(n_f, n_r, n_all, edges, key):
    out = []
    for k in range(len(edges) - 1):
        a, b, n = int(n_f[k]), int(n_r[k]), int(n_all[k])
        if n == 0:
            continue
        r = (a / max(b, 1)) if b else float("nan")
        err = r * np.sqrt(1 / max(a, 1) + 1 / max(b, 1)) if b else float("nan")
        out.append({key: [float(edges[k]), float(edges[k + 1])], "generated": n, "pass_fast": a, "pass_g4": b,
                    "ratio": float(r), "ratio_err": float(err)})
    return out


def run(inputs, outdir, compare=None, gdml=None, **kw):
    if not compare:
        raise SystemExit("worldcheck FAST --compare G4REF")
    from .steps import default_gdml
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    world = st.load_world(gdml, geom)
    y_pl = st.plane_y(geom)
    fast_paths, g4_paths = T.expand(inputs), T.expand(compare)
    # denominator: every generated event (the fast tables carry the whole gen stream; the G4 tables only the events
    # the PrimaryFilter accepted, a superset of the tracker-term passes, so an event missing there did not pass)
    gen = primary_muons(T.read_many(fast_paths, "gen")).reset_index(drop=True)
    gen_g4 = primary_muons(T.read_many(g4_paths, "gen"))[T.KEY]
    n_g4_not_in_fast = int(len(gen_g4.merge(gen[T.KEY], on=T.KEY, how="left", indicator=True).query("_merge == 'left_only'")))
    gen["p_gen"] = np.sqrt(gen.px ** 2 + gen.py ** 2 + gen.pz ** 2)
    o = gen[["x", "y", "z"]].to_numpy()
    d = gen[["px", "py", "pz"]].to_numpy() / gen.p_gen.to_numpy()[:, None]
    near = st.near_tracker(o, d, geom, y_pl).astype(bool)
    gen["miss_mm"] = _miss_distance(o, d, geom, y_pl)
    f, r = _arm(fast_paths), _arm(g4_paths)
    ev = gen[near].reset_index(drop=True)
    ev = ev.merge(f, on=T.KEY, how="left").merge(r.rename(columns={c: c + "_g4" for c in r.columns if c not in T.KEY}),
                                                 on=T.KEY, how="left")
    gcm2, xx0 = _column(ev, world, y_pl)
    ev["gcm2"], ev["xx0"] = gcm2, xx0
    pf, pg = ev.p_first.notna().to_numpy(), ev.p_first_g4.notna().to_numpy()
    out = {"generated_common": int(len(gen)), "g4_accepted_not_in_fast_gen": n_g4_not_in_fast,
           "near_tracker": int(near.sum()),
           "pass": {"fast": int(pf.sum()), "g4": int(pg.sum()), "both": int((pf & pg).sum()),
                    "fast_only": int((pf & ~pg).sum()), "g4_only": int((~pf & pg).sum())},
           "pass_outside_near_g4": int(r.merge(gen[~near][T.KEY], on=T.KEY).shape[0]),
           "pass_outside_near_fast": int(f.merge(gen[~near][T.KEY], on=T.KEY).shape[0])}
    # tracker-term passes by the distance of the generated straight line from the straws' x-z footprint at the
    # plane (0 = aimed at the tracker): muons G4 delivers from far outside were deflected into the acceptance
    allf = gen.merge(f[T.KEY].assign(_f=1), on=T.KEY, how="left").merge(r[T.KEY].assign(_g=1), on=T.KEY, how="left")
    mb = np.clip(np.digitize(allf.miss_mm, MISS_EDGES) - 1, 0, len(MISS_EDGES) - 2)
    out["pass_by_miss_mm"] = [{"miss_mm": [float(MISS_EDGES[k]), float(MISS_EDGES[k + 1])], "generated": int((mb == k).sum()),
                               "pass_fast": int(((mb == k) & allf._f.notna()).sum()),
                               "pass_g4": int(((mb == k) & allf._g.notna()).sum()),
                               "p_gen_median_g4_pass": float(allf.p_gen[(mb == k) & allf._g.notna()].median())
                               if ((mb == k) & allf._g.notna()).any() else None}
                              for k in range(len(MISS_EDGES) - 1)]
    pb = np.clip(np.digitize(ev.p_gen, P_EDGES) - 1, 0, len(P_EDGES) - 2)
    cb = np.clip(np.digitize(ev.gcm2, COL_EDGES) - 1, 0, len(COL_EDGES) - 2)
    nb_p, nb_c = len(P_EDGES) - 1, len(COL_EDGES) - 1
    out["delivery_by_pgen"] = _rate_table(np.bincount(pb[pf], minlength=nb_p), np.bincount(pb[pg], minlength=nb_p),
                                          np.bincount(pb, minlength=nb_p), P_EDGES, "p_gen")
    out["delivery_by_column"] = _rate_table(np.bincount(cb[pf], minlength=nb_c), np.bincount(cb[pg], minlength=nb_c),
                                            np.bincount(cb, minlength=nb_c), COL_EDGES, "gcm2")
    both = ev[pf & pg].copy()
    both["loss_fast"] = both.p_gen - both.p_first
    both["loss_g4"] = both.p_gen - both.p_first_g4
    rows = []
    for lab, edges, col in (("p_gen", P_EDGES, "p_gen"), ("gcm2", COL_EDGES, "gcm2")):
        b = np.clip(np.digitize(both[col], edges) - 1, 0, len(edges) - 2)
        for k in range(len(edges) - 1):
            x = both[b == k]
            if len(x) < 20:
                continue
            dl = (x.loss_fast - x.loss_g4).to_numpy()
            rows.append({"bin": lab, "range": [float(edges[k]), float(edges[k + 1])], "n": int(len(x)),
                         "loss_median_fast": float(x.loss_fast.median()), "loss_median_g4": float(x.loss_g4.median()),
                         "loss_mean_fast": float(x.loss_fast.mean()), "loss_mean_g4": float(x.loss_g4.mean()),
                         "paired_median_diff": float(np.median(dl)),
                         "paired_median_diff_err": float(1.2533 * dl.std() / np.sqrt(len(dl))),
                         "gcm2_median": float(x.gcm2.median())})
    out["loss_both"] = rows
    if len(both):
        k = both.gcm2 > 50
        dedx_f = (both.loss_fast[k] / both.gcm2[k]).median()
        dedx_g = (both.loss_g4[k] / both.gcm2[k]).median()
        out["loss_per_gcm2_median"] = {"fast": float(dedx_f), "g4": float(dedx_g), "ratio_minus_1": float(dedx_f / dedx_g - 1)}
    for lab, m in (("fast_only", pf & ~pg), ("g4_only", ~pf & pg), ("both", pf & pg)):
        x = ev[m]
        if len(x):
            out[f"{lab}_profile"] = {"n": int(len(x)), "p_gen_q": np.quantile(x.p_gen, [0.1, 0.5, 0.9]).round(0).tolist(),
                                     "gcm2_q": np.quantile(x.gcm2, [0.1, 0.5, 0.9]).round(1).tolist(),
                                     "share_p_gen_below_1GeV": float(np.mean(x.p_gen < 1000))}
    out["transverse_first_straw"] = _transverse_table(ev, pf, pg)
    _print(out)
    plots.write_json(outdir, "worldcheck.json", out)
    ev.to_csv(os.path.join(outdir, "worldcheck_events.csv"), index=False)


def _print(out):
    print(f"generated (common) {out['generated_common']:,}   heading for the tracker {out['near_tracker']:,}   "
          f"tracker term: {out['pass']}   G4 passes outside the 'near' cut: {out['pass_outside_near_g4']}")
    print(f"G4-accepted events missing from the fast gen tree: {out['g4_accepted_not_in_fast_gen']}   "
          f"passes outside the 'near' cut: G4 {out['pass_outside_near_g4']}, fast {out['pass_outside_near_fast']}")
    print("passes by distance of the generated line from the straw footprint at the plane [mm]:")
    for r in out["pass_by_miss_mm"]:
        print(f"  {r['miss_mm'][0]:7.0f}-{min(r['miss_mm'][1], 1e6):7.0f}  gen {r['generated']:8d}  fast {r['pass_fast']:5d}"
              f"  g4 {r['pass_g4']:5d}  (G4 passes: median p_gen {r['p_gen_median_g4_pass']})")
    print("delivery P(pass), fast/G4 by generated |p| [MeV/c]:")
    for r in out["delivery_by_pgen"]:
        print(f"  {r['p_gen'][0]:7.0f}-{min(r['p_gen'][1], 1e6):7.0f}  gen {r['generated']:7d}  fast {r['pass_fast']:5d}  g4 {r['pass_g4']:5d}"
              f"  ratio {r['ratio']:.3f} +- {r['ratio_err']:.3f}")
    print("delivery by material column above the plane [g/cm2]:")
    for r in out["delivery_by_column"]:
        print(f"  {r['gcm2'][0]:6.0f}-{min(r['gcm2'][1], 1e5):6.0f}  gen {r['generated']:7d}  fast {r['pass_fast']:5d}  g4 {r['pass_g4']:5d}"
              f"  ratio {r['ratio']:.3f} +- {r['ratio_err']:.3f}")
    print("energy loss gen -> first straw on events passing in both [MeV]:")
    for r in out["loss_both"]:
        print(f"  {r['bin']:5s} {r['range'][0]:7.0f}-{min(r['range'][1], 1e6):7.0f}  n {r['n']:5d}  median fast {r['loss_median_fast']:7.1f}  g4 {r['loss_median_g4']:7.1f}"
              f"  paired diff {r['paired_median_diff']:+7.1f} +- {r['paired_median_diff_err']:.1f}  (column median {r['gcm2_median']:.0f} g/cm2)")
    if "loss_per_gcm2_median" in out:
        print(f"median loss per g/cm2 (column > 50): {out['loss_per_gcm2_median']}")
    for lab in ("both", "fast_only", "g4_only"):
        if f"{lab}_profile" in out:
            print(f"  {lab:9s} {out[f'{lab}_profile']}")
    print("transverse state at the first straw (offset from the generated line r [mm] q50/q90/q99, angle [mrad] "
          "q50/q90/q99, rho(offset, angle), rms y/(theta L)); each arm on its passes, then paired:")
    for r in out.get("transverse_first_straw", []):
        for lab in ("fast", "g4", "paired_fast", "paired_g4"):
            x = r[lab]
            if x is None:
                continue
            print(f"  p {r['p_gen'][0]:5.0f}-{r['p_gen'][1]:7.0f} col {r['gcm2'][0]:4.0f}-{r['gcm2'][1]:6.0f} {lab:11s} n {x['n']:5d}"
                  f"  r {x['r_mm_q']}  th {x['theta_mrad_q']}  rho {x['rho']:.3f}  y/thL {x['y_over_thetaL_rms']:.3f}"
                  f"  L {x['L_median_mm']:.0f}")
