"""worldcheck -- the world model event by event: muon |p| at the first straw against |p| at generation, fast vs G4.

Both arms ran the same CRY primaries, so events are paired by (run, subrun, event). Per event: generated muon |p| and
direction (gen tree, identical in both arms), the GDML material column along the generated straight line down to the
plane above the tracker (concrete/overburden g/cm2, total x/X0), and, where the muon passes the PrimaryFilter tracker
term (>= 12 good steps; every such event is in the G4 tables), |p| at its first straw.

Reported:
  1. delivery: P(tracker term passes) per generated-|p| bin and per column bin, fast against G4 (same generated
     events, so the ratio is a pure transport comparison);
  2. energy loss on events passing in both arms: loss = p_gen - p_first, fast against G4, per generated-|p| and
     column bin (median and mean, and the paired median difference);
  3. the unpaired events (pass in one arm only): their generated |p| and column, to locate what one arm delivers and
     the other does not.
"""
import os

import numpy as np
import pandas as pd

from mu2efastsim import materials, plots
from mu2efastsim import world as world_mod
from mu2efastsim.tables import primary_muons
from .. import geometry, steps as st, tables as T

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
    return first[T.KEY + ["p_first"]]


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
    gen = primary_muons(T.read_many(fast_paths, "gen")).reset_index(drop=True)
    gen_g4 = primary_muons(T.read_many(g4_paths, "gen"))[T.KEY]
    gen = gen.merge(gen_g4, on=T.KEY)     # common generated events (identical primaries in both arms)
    gen["p_gen"] = np.sqrt(gen.px ** 2 + gen.py ** 2 + gen.pz ** 2)
    o = gen[["x", "y", "z"]].to_numpy()
    d = gen[["px", "py", "pz"]].to_numpy() / gen.p_gen.to_numpy()[:, None]
    near = st.near_tracker(o, d, geom, y_pl).astype(bool)
    f, r = _arm(fast_paths), _arm(g4_paths)
    ev = gen[near].reset_index(drop=True)
    ev = ev.merge(f, on=T.KEY, how="left").merge(r.rename(columns={"p_first": "p_first_g4"}), on=T.KEY, how="left")
    gcm2, xx0 = _column(ev, world, y_pl)
    ev["gcm2"], ev["xx0"] = gcm2, xx0
    pf, pg = ev.p_first.notna().to_numpy(), ev.p_first_g4.notna().to_numpy()
    out = {"generated_common": int(len(gen)), "near_tracker": int(near.sum()),
           "pass": {"fast": int(pf.sum()), "g4": int(pg.sum()), "both": int((pf & pg).sum()),
                    "fast_only": int((pf & ~pg).sum()), "g4_only": int((~pf & pg).sum())},
           "pass_outside_near_g4": int(r.merge(gen[~near][T.KEY], on=T.KEY).shape[0])}
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
    _print(out)
    plots.write_json(outdir, "worldcheck.json", out)
    ev.to_csv(os.path.join(outdir, "worldcheck_events.csv"), index=False)


def _print(out):
    print(f"generated (common) {out['generated_common']:,}   heading for the tracker {out['near_tracker']:,}   "
          f"tracker term: {out['pass']}   G4 passes outside the 'near' cut: {out['pass_outside_near_g4']}")
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
