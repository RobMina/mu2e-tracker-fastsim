"""fatecheck -- what happens to the muons the fast sim does not deliver, against G4 on the same CRY primaries.

Reruns the fast generator (stage C, current defaults) on the gen tables with the `steps` seed convention (file i uses
seed (seed, i)), so with the same inputs and seed the fast pass set is that of `steps`. Both rate sides count ANY
primary muon, with one row per event. Each fast muon is classified before event reduction:
  stopped_above  ranged out in the world column above the plane
  missed         alive at the plane, no straw crossed (scattered or aimed outside, or stopped in the support)
  few            1-11 good straw steps (PrimaryFilter tracker term not reached)
  pass           tracker term passes
and, from the G4 tables (PrimaryFilter-accepted events only), whether G4 passes, and for events G4 kept without a
tracker pass, where the G4 primary muon ended (stopped above the plane, ended below it). Tables per generated |p|:
counts of G4 passes, fast passes, both, G4-only, fast-only; the fast fate of the G4-only events; the G4 fate of the
fast-only events. Event fate is the most advanced fast-muon fate (pass > few > missed > stopped_above), with
fast_sp identifying the representative (largest good-step count, then lowest key). Generated |p| and G4 end-state
attribution are first-primary diagnostics, not necessarily the passing muon. fast_first_pass retains the old
fast-first versus G4-any rate convention. A deficit in missed can indicate scattering; stopped_above also includes
muons not transported because they are outside the tracker footprint.

    fatecheck gen.001400_*.root --compare "g4ref_v1_tables/sgsteps.*.root" --gdml ... --outdir DIR [--seed 1]
"""
import os

import numpy as np
import pandas as pd

from mu2efastsim import eloss, plots
from mu2efastsim.tables import KEY, all_primary_muons
from .. import geometry, steps as st, tables as T
from .worldcheck import _miss_distance

P_EDGES = np.array([0.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0, 1e9])
FATES = ("stopped_above", "missed", "few", "pass")


def _fast_events(gen, sgs, lp, geom, y_pl):
    """Classify each generated muon, then keep one representative event row and first-muon diagnostics."""
    mu = all_primary_muons(gen)
    if not np.array_equal(mu[KEY].to_numpy(), lp["key"]):
        raise ValueError("last_plane rows do not match all_primary_muons order")
    ids = KEY + ["sp"]
    df = mu[KEY + ["mukey"]].rename(columns={"mukey": "sp"})
    df["alive"] = lp["alive"]
    df["p_plane"] = lp["p_plane"]
    df["miss_plane_mm"] = _miss_distance(lp["o"], lp["d"], geom, y_pl)
    primary = sgs.merge(df[ids], on=ids, validate="many_to_one")
    good = primary[T.good_steps(primary)].groupby(ids).size().rename("ngood")
    p_last = primary.assign(p=np.sqrt(primary.px ** 2 + primary.py ** 2 + primary.pz ** 2)).groupby(
        ids).p.min().rename("p_last")
    df = df.merge(good.reset_index(), on=ids, how="left", validate="one_to_one")
    df = df.merge(p_last.reset_index(), on=ids, how="left", validate="one_to_one")
    df["ngood"] = df.ngood.fillna(0).astype(int)
    df["fate"] = np.select([~df.alive, df.ngood == 0, df.ngood < T.FILTER_MIN_TRK_STEPS], list(FATES[:3]), "pass")
    df["fast_pass"] = df.ngood >= T.FILTER_MIN_TRK_STEPS
    first = df[df.sp == 1][KEY + ["ngood", "fate", "fast_pass"]].rename(
        columns={"ngood": "ngood_first", "fate": "fate_first", "fast_pass": "fast_first_pass"})
    df["_rank"] = df.fate.map({f: i for i, f in enumerate(FATES)})
    ev = df.sort_values(["_rank", "ngood", "sp"], ascending=[False, False, True], kind="stable").drop_duplicates(KEY)
    ev = ev.drop(columns="_rank").rename(columns={"sp": "fast_sp"})
    return ev.merge(first, on=KEY, validate="one_to_one").sort_values(KEY).reset_index(drop=True)


def _fast(inputs, gdml, seed):
    geom = geometry.from_gdml(gdml)
    world = st.load_world(gdml, geom)
    interior = st.TrackerInterior(gdml, geom)
    y_pl = st.plane_y(geom)
    rows = []
    for i, f in enumerate(T.expand(inputs)):
        gen = T.read(f, "gen")
        sgs, _ = st.make_steps(gen, geom, np.random.default_rng([int(seed), i]), stage="C", world=world,
                               sliced_msc=True, interior=interior)
        df = _fast_events(gen, sgs, st.make_steps.last_plane, geom, y_pl)
        rows.append(df)
        print(f"# {f}: {len(df):,} events, {int((df.ngood >= T.FILTER_MIN_TRK_STEPS).sum()):,} fast passes", flush=True)
    return pd.concat(rows, ignore_index=True), geom, y_pl


def _g4(paths, y_pl):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles")
    pm = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)]
    s = s.merge(pm[KEY + ["key"]].rename(columns={"key": "sp"}), on=KEY + ["sp"])
    f = T.trk_filter(s)
    acc = pd.DataFrame(T.read_many(paths, "gen", KEY).drop_duplicates())
    acc["g4_pass"] = acc.set_index(KEY).index.isin(f[f.pass_trk].index)
    end = pm.drop_duplicates(KEY)[KEY + ["ey", "ee"]].copy()
    end["ey_mu2e"] = end.ey
    end["g4_stopped"] = (end.ee - eloss.MMU) < 1.0
    return acc.merge(end[KEY + ["ey_mu2e", "g4_stopped"]], on=KEY, how="left")


def run(inputs, outdir, compare=None, gdml=None, seed=1, **kw):
    if not compare:
        raise SystemExit("fatecheck GEN --compare G4REF")
    from .steps import default_gdml
    gdml = gdml or default_gdml()
    fa, geom, y_pl = _fast(inputs, gdml, seed)
    gen = T.read_many(T.expand(inputs), "gen")
    gm = all_primary_muons(gen).drop_duplicates(KEY)
    gm["p_gen"] = np.sqrt(gm.px ** 2 + gm.py ** 2 + gm.pz ** 2)
    ev = gm[KEY + ["p_gen"]].merge(fa, on=KEY, how="inner", validate="one_to_one").merge(
        _g4(T.expand(compare), y_pl), on=KEY, how="left", validate="one_to_one")
    ev["p_gen_first"] = ev.p_gen
    ev["in_g4"] = ev.g4_pass.notna()
    ev["g4_pass"] = ev.g4_pass.fillna(False).astype(bool)
    rel = ev[ev.fast_pass | ev.g4_pass].copy()
    b = np.clip(np.digitize(rel.p_gen, P_EDGES) - 1, 0, len(P_EDGES) - 2)
    out = {"generated": int(len(ev)), "generated_events": int(len(gen[KEY].drop_duplicates())),
           "generated_muon_events": int(len(ev)), "fast_pass": int(ev.fast_pass.sum()), "g4_pass": int(ev.g4_pass.sum()),
           "fast_first_pass": int(ev.fast_first_pass.sum()), "rate_convention": "any primary muon per event",
           "fast_fate_convention": "most advanced primary fate; most good steps then lowest fast key",
           "p_gen_attribution": "first generated primary muon diagnostic",
           "g4_end_attribution": "first primary muon diagnostic", "by_p_gen": []}
    for k in range(len(P_EDGES) - 1):
        x = rel[b == k]
        if not len(x):
            continue
        g4o = x[x.g4_pass & ~x.fast_pass]
        fo = x[x.fast_pass & ~x.g4_pass]
        fo_in = fo[fo.in_g4]
        out["by_p_gen"].append({
            "p_gen": [float(P_EDGES[k]), float(P_EDGES[k + 1])],
            "g4": int(x.g4_pass.sum()), "fast": int(x.fast_pass.sum()), "both": int((x.g4_pass & x.fast_pass).sum()),
            "g4_only": int(len(g4o)), "fast_only": int(len(fo)),
            "g4_only_fast_fate": {f: int((g4o.fate == f).sum()) for f in FATES[:3]},
            "fast_only_g4_fate": {"not_kept": int((~fo.in_g4).sum()),
                                  "stopped_above_plane": int((fo_in.g4_stopped & (fo_in.ey_mu2e > y_pl)).sum()),
                                  "stopped_below": int((fo_in.g4_stopped & (fo_in.ey_mu2e <= y_pl)).sum()),
                                  "not_stopped": int((~fo_in.g4_stopped.fillna(False).astype(bool)).sum())}})
    print(f"generated muon events {out['generated']:,}  fast passes {out['fast_pass']:,}  G4 passes {out['g4_pass']:,}")
    print(f"{'p_gen(first)':>13s} {'g4':>5s} {'fast':>5s} {'both':>5s} {'g4only':>6s} {'fonly':>5s}  fast fate of G4-only "
          f"(stop_above/missed/few)   G4 fate of fast-only (not_kept/stop_above/stop_below/through)")
    for r in out["by_p_gen"]:
        a, c = r["g4_only_fast_fate"], r["fast_only_g4_fate"]
        print(f"{r['p_gen'][0]:6.0f}-{min(r['p_gen'][1], 1e6):6.0f} {r['g4']:5d} {r['fast']:5d} {r['both']:5d} {r['g4_only']:6d}"
              f" {r['fast_only']:5d}  {a['stopped_above']:4d} {a['missed']:4d} {a['few']:4d}"
              f"                     {c['not_kept']:4d} {c['stopped_above_plane']:4d} {c['stopped_below']:4d} {c['not_stopped']:4d}")
    os.makedirs(outdir, exist_ok=True)
    ev.drop(columns=[]).to_csv(os.path.join(outdir, "fatecheck_events.csv.gz"), index=False)
    plots.write_json(outdir, "fatecheck.json", out)
