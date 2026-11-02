"""field-compare -- field-on primary StrawGasSteps, fast (sim.fieldsteps) against Geant4, paired on the G4 entry state.

Reference: StrawGasStepTableDumper tables of a field-on dts sample (sgsteps, vdsteps, particles). The MC primary
(ismcprim == 1) of the chosen species is restarted by the fast transport from its first crossing of the tracker
virtual detectors (`--entry-vds`, default every tracker VD; `13` = TT_FrontHollow for electrons from the target), with
the G4 position, momentum and time there, so both arms see the same track at the tracker envelope and differ only in
the transport inside it (field stepping, straw crossings, loss, scattering). The field is a G4BL map (`--map`, e.g. the
DS map) or a constant (`--bz`); `--no-field` is the straight-line control arm. Electrons use
`fieldsteps.electron_loss` and the Highland core without the single-scattering tail (the tail model is the muon one).

Compared (only primary steps after the entry time, events with at least one G4 primary step after it):
  * straws per track: paired mean difference with its error, and the fraction of G4 straws the fast track also crosses;
  * per step: path in the gas (mean, binned ratio chi2), eIon (mean, Moyal MPV, chi2), width (mean, chi2), shape and
    ionisation type tables;
  * per track at the last G4 straw: |p| difference (fast - G4) per |p| bin at the entry, the distance between the
    fast and G4 exit points of that straw (when both cross it), and the time from the entry to the first straw.
The same block is repeated in `first_pass` for the tracks the fast transport took out of the envelope, with the G4
steps up to the fast exit time (+ FIRST_PASS_NS): G4 tracks that loop back into the tracker in the field, or return
from the calorimeter, add later passages that the fast transport (which stops at the envelope) does not follow.
Errors: paired per-track differences, standard error of the mean; chi2 of binned distributions with Poisson errors on
both arms (unpaired), ndf = bins with entries.
"""
import json
import os

import numpy as np
import pandas as pd

from mu2efastsim import ehelix, stats
from trkfastsim import geometry, tables as T
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.sim import steps as st

TRACKER_VDS = (11, 12, 13, 14, 15, 16, 17)       # VirtualDetectorId TT_Mid .. TT_InSurf
FIRST_PASS_NS = 1.0     # first-passage selection: G4 steps up to the fast exit time plus this
P_EDGES = np.array([0.0, 80.0, 120.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 1e9])


def primaries(paths, species):
    """Primary particles of the species (13 or 11, either charge): the MC primary (ismcprim) or G4 primary (isprim),
    and their same-species daughters (the continuation of a stage-1 primary in a resampled stage).
    Returns (particles, their sgsteps, their vdsteps, census of the selection)."""
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "isprim", "ismcprim"]).rename(columns={"key": "sp"})
    census = {"events": int(len(p[T.KEY].drop_duplicates())), "ismcprim": int((p.ismcprim == 1).sum()),
              "isprim": int((p.isprim == 1).sum()),
              "isprim_species": int(((p.isprim == 1) & (p.pdg.abs() == species)).sum()),
              "ismcprim_species": int(((p.ismcprim == 1) & (p.pdg.abs() == species)).sum())}
    prim = p[((p.ismcprim == 1) | (p.isprim == 1)) & (p.pdg.abs() == species)]
    # a resampled stage (cosmic S2) continues each stage-1 primary as a daughter of the same species: take it too
    kids = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "parent"]).rename(columns={"key": "sp"})
    kids = kids.merge(prim[T.KEY + ["sp", "pdg"]].rename(columns={"sp": "parent"}), on=T.KEY + ["parent", "pdg"])
    census["continuations"] = int(len(kids))
    p = pd.concat([prim[T.KEY + ["sp", "pdg"]], kids[T.KEY + ["sp", "pdg"]]]).drop_duplicates(T.KEY + ["sp"])
    s_all = T.read_many(paths, "sgsteps")
    pa = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "isprim", "parent"]).rename(columns={"key": "sp"})
    sa = s_all[T.KEY + ["sp"]].merge(pa, on=T.KEY + ["sp"], how="left")
    census.update(sgsteps=int(len(s_all)), sgsteps_unknown_sp=int(sa.pdg.isna().sum()),
                  sgsteps_by_pdg=sa.pdg.value_counts().head(8).to_dict(),
                  sgsteps_isprim=int((sa.isprim == 1).sum()),
                  sgsteps_parent_isprim_muon=int(sa.merge(pa[(pa.isprim == 1) & (pa.pdg.abs() == species)][T.KEY + ["sp"]]
                                                          .rename(columns={"sp": "parent"}), on=T.KEY + ["parent"]).shape[0]),
                  sp_min=int(pa.sp.min()), prim_sp_values=pa[pa.isprim == 1].sp.value_counts().head(5).to_dict())
    s = s_all.merge(p, on=T.KEY + ["sp"], how="inner")
    v_all = T.read_many(paths, "vdsteps")
    v = v_all.merge(p, on=T.KEY + ["sp"], how="inner")
    u_all = T.read_many(paths, "surfsteps")
    u = u_all.merge(p, on=T.KEY + ["sp"], how="inner")
    census.update(vdsteps=int(len(v_all)), vdsteps_primary=int(len(v)), surfsteps=int(len(u_all)),
                  surfsteps_primary=int(len(u)), vd_ids_all=v_all.vdid.value_counts().sort_index().to_dict(),
                  surf_ids_primary=u.surf.value_counts().sort_index().to_dict())
    return p.reset_index(drop=True), s.reset_index(drop=True), v.reset_index(drop=True), u.reset_index(drop=True), census


def entry_states(vd, vds=TRACKER_VDS):
    """First crossing (in time) of each primary with the given VDs: KEY, sp, pdg, x, y, z, px, py, pz, t, vdid."""
    v = vd[vd.vdid.isin(vds)].sort_values("t")
    return v.groupby(T.KEY + ["sp"], sort=False).first().reset_index()


TRACKER_SURFS = (0, 1, 2, 3, 4)                  # SurfaceId TT_Front, TT_Mid, TT_Back, TT_Inner, TT_Outer


def entry_states_surface(surf, origin, surfs=TRACKER_SURFS):
    """First tracker SurfaceStep (in time) of each primary: its entry point (detector -> Mu2e frame), momentum, time.
    For samples whose tracker VD steps were not kept."""
    v = surf[surf.surf.isin(surfs)].sort_values("t")
    v = v.groupby(T.KEY + ["sp"], sort=False).first().reset_index()
    x = geometry.to_mu2e(v[["x0", "y0", "z0"]].to_numpy(), origin)
    v = v.assign(x=x[:, 0], y=x[:, 1], z=x[:, 2], vdid=-1 - v.surf.to_numpy())
    return v


def fit_origin_z(sgs, geom, x_det=-3904.0):
    """Detector-frame z origin in the Mu2e frame from the G4 steps: median of straw-mid z minus step-mid z."""
    j = _straw_index(sgs, geom)
    zm = 0.5 * (sgs.z0.to_numpy() + sgs.z1.to_numpy())
    return np.array([x_det, 0.0, float(np.median(geom.mid[j, 2] - zm))])


def geometry_check(sgs, geom, origin):
    """T-V0 on this sample: distance of each G4 step's entry and exit point from its own straw's axis against the gas
    radius (points inside the straw length), and the axial position against the half length."""
    j = _straw_index(sgs, geom)
    out = {}
    for end in "01":
        x = geometry.to_mu2e(sgs[[f"x{end}", f"y{end}", f"z{end}"]].to_numpy(), origin) - geom.mid[j]
        wa = (x * geom.dir[j]).sum(1)
        r = np.sqrt(np.clip((x * x).sum(1) - wa ** 2, 0, None))
        out[f"end{end}"] = {"steps": int(len(r)), "r_minus_radius_mm_q01_q50_q99":
                            [float(np.quantile(r - geom.radius, q)) for q in (0.01, 0.5, 0.99)] if len(r) else [],
                            "outside_gas_by_0p01mm": int((r > geom.radius + 0.01).sum()),
                            "beyond_half_length": int((np.abs(wa) > geom.halflen[j] + 0.01).sum())}
    return out


def _straw_index(sgs, geom):
    sid = geom.sid
    order = np.argsort(sid)
    return order[np.searchsorted(sid[order], sgs.sid.to_numpy())]


def _hist_chi2(a, b, edges):
    ha, _ = np.histogram(a, edges)
    hb, _ = np.histogram(b, edges)
    na, nb = ha.sum(), hb.sum()
    if na == 0 or nb == 0:
        return dict(chi2=float("nan"), ndf=0)
    fa, fb = ha / na, hb / nb
    var = ha / na ** 2 + hb / nb ** 2
    ok = var > 0
    return dict(chi2=float(((fa - fb)[ok] ** 2 / var[ok]).sum()), ndf=int(ok.sum()))


def _mean_err(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float("nan"), float("nan"), int(len(x))]
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def _mpv(x):
    try:
        r = stats.mpv_fit(np.asarray(x, float))
        return [float(r[0]), float(r[1])] if isinstance(r, (tuple, list)) else [float(r), float("nan")]
    except Exception:            # fit failure on a small sample: report the median instead
        return [float(np.median(x)), float("nan")]


def compare(g4, fast, ent, final, geom):
    """Summary dict of the paired comparison (g4/fast: primary sgsteps after the entry; ent: entry states)."""
    key = T.KEY + ["sp"]
    out = {}
    ng = g4.groupby(key).size().rename("ng")
    nf = fast.groupby(key).size().rename("nf")
    per = ent[key].merge(ng, on=key, how="left").merge(nf, on=key, how="left").fillna(0)
    per = per[per.ng > 0]
    out["tracks"] = int(len(per))
    out["straws_per_track"] = {"g4": _mean_err(per.ng), "fast": _mean_err(per.nf),
                               "fast_minus_g4": _mean_err(per.nf - per.ng),
                               "ratio": float(per.nf.sum() / max(per.ng.sum(), 1))}
    both = g4[key + ["sid"]].merge(fast[key + ["sid"]].drop_duplicates(), on=key + ["sid"], how="left", indicator=True)
    out["g4_straws_also_fast"] = float((both._merge == "both").mean()) if len(both) else float("nan")
    pe = ent[key + ["px", "py", "pz"]].assign(pent=lambda d: np.sqrt(d.px ** 2 + d.py ** 2 + d.pz ** 2))[key + ["pent"]]
    per = per.merge(pe, on=key)
    hit = both.assign(m=(both._merge == "both").astype(float)).groupby(key).m.mean().rename("match").reset_index()
    per = per.merge(hit, on=key, how="left")
    out["straws_by_entry_p"] = [
        {"p_lo": float(lo), "p_hi": float(hi), "tracks": int(k.sum()), "g4": _mean_err(per.ng[k]),
         "fast_minus_g4": _mean_err((per.nf - per.ng)[k]), "match": _mean_err(per.match[k])}
        for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]) for k in [(per.pent >= lo) & (per.pent < hi)] if k.sum() >= 5]
    out["len_mm"] = {"g4": _mean_err(g4.len), "fast": _mean_err(fast.len),
                     **_hist_chi2(g4.len, fast.len, np.linspace(0, 12, 49))}
    out["eion_keV"] = {"g4_mean": _mean_err(1e3 * g4.eion), "fast_mean": _mean_err(1e3 * fast.eion),
                       "g4_mpv": _mpv(1e3 * g4.eion), "fast_mpv": _mpv(1e3 * fast.eion),
                       **_hist_chi2(1e3 * g4.eion, 1e3 * fast.eion, np.linspace(0, 20, 81))}
    out["width_um"] = {"g4": _mean_err(1e3 * g4.width), "fast": _mean_err(1e3 * fast.width),
                       **_hist_chi2(1e3 * g4.width, 1e3 * fast.width, np.linspace(0, 300, 61))}
    out["shape_counts"] = {"g4": g4["shape"].value_counts().sort_index().to_dict(),
                           "fast": fast["shape"].value_counts().sort_index().to_dict()}
    out["ion_counts"] = {"g4": g4["ion"].value_counts().sort_index().to_dict(),
                         "fast": fast["ion"].value_counts().sort_index().to_dict()}
    # last G4 straw of each track: momentum and exit point, fast at the same straw when it crosses it
    gl = g4.sort_values("t").groupby(key).last().reset_index()
    fl = gl[key + ["sid"]].merge(fast, on=key + ["sid"], how="inner").sort_values("t").groupby(key).last().reset_index()
    m = gl.merge(fl, on=key + ["sid"], suffixes=("_g", "_f"))
    pg = np.sqrt(m.px_g ** 2 + m.py_g ** 2 + m.pz_g ** 2)
    pf = np.sqrt(m.px_f ** 2 + m.py_f ** 2 + m.pz_f ** 2)
    p0 = m.merge(ent[key + ["px", "py", "pz"]], on=key)
    pent = np.sqrt(p0.px ** 2 + p0.py ** 2 + p0.pz ** 2).to_numpy()
    dp = (pf - pg).to_numpy()
    dexit = np.sqrt((m.x1_f - m.x1_g) ** 2 + (m.y1_f - m.y1_g) ** 2 + (m.z1_f - m.z1_g) ** 2).to_numpy()
    out["last_straw"] = {"matched_tracks": int(len(m)), "dp_MeV": _mean_err(dp),
                         "dexit_mm_median": float(np.median(dexit)) if len(dexit) else float("nan"),
                         "dexit_mm_q90": float(np.quantile(dexit, 0.9)) if len(dexit) else float("nan")}
    bins = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        k = (pent >= lo) & (pent < hi)
        if k.sum() >= 5:
            bins.append({"p_lo": float(lo), "p_hi": float(hi), "dp_MeV": _mean_err(dp[k]),
                         "g4_p_drop_MeV": _mean_err(pent[k] - pg.to_numpy()[k]),
                         "dexit_mm_median": float(np.median(dexit[k]))})
    out["last_straw"]["by_entry_p"] = bins
    # entry-to-first-straw time
    g1 = g4.sort_values("t").groupby(key).first().reset_index()[key + ["sid", "t"]]
    f1 = fast.sort_values("t").groupby(key).first().reset_index()[key + ["sid", "t"]]
    t1 = g1.merge(f1, on=key + ["sid"], suffixes=("_g", "_f"))
    out["first_straw_dt_ns"] = _mean_err(t1.t_f - t1.t_g)
    out["final_status"] = final.status.value_counts().sort_index().to_dict()
    return out


def run(inputs, outdir, compare_=None, *, gdml, species=13, field_map=None, bz=None, no_field=False, seed=1,
        chord_mm=20.0, sagitta_mm=0.02, entry_vds=TRACKER_VDS, entry="vd", origin=None, interior=True, max_events=None, **kw):
    import time
    os.makedirs(outdir, exist_ok=True)
    geom = geometry.from_gdml(gdml)
    part, sgs, vd, surf, census = primaries(inputs, species)
    if max_events:
        ev = part[T.KEY].drop_duplicates().iloc[:max_events]
        part, sgs, vd, surf = (d.merge(ev, on=T.KEY) for d in (part, sgs, vd, surf))
    org = np.asarray(origin, float) if origin is not None else fit_origin_z(sgs, geom)
    geo = geometry_check(sgs, geom, org)
    if entry == "vd":
        ent = entry_states(vd, entry_vds)
    elif entry == "surface":
        ent = entry_states_surface(surf, org)
    else:                          # earliest of the two
        cols = T.KEY + ["sp", "pdg", "x", "y", "z", "px", "py", "pz", "t", "vdid"]
        ent = pd.concat([entry_states(vd, entry_vds)[cols], entry_states_surface(surf, org)[cols]]).sort_values("t")
        ent = ent.groupby(T.KEY + ["sp"], sort=False).first().reset_index()
    if no_field:
        field = None
    elif field_map:
        field = ehelix.grid_field(*ehelix.read_g4bl(field_map))
    else:
        field = np.array([0.0, 0.0, float(bz if bz is not None else 1.0)])
    env = fs.Envelope(geom, axis_xy=org[:2])
    inter = fs.interior(gdml, geom, env, origin=org) if interior else None
    bnom = ehelix.field_at(field if field is not None else np.zeros(3), org[None])[0]
    t0 = time.perf_counter()
    eopt = dict(loss=fs.electron_loss, tail=False) if species == 11 else {}
    fast, final = fs.transport(np.random.default_rng(seed), ent, geom, field, inter, chord_mm=chord_mm,
                               sagitta_mm=sagitta_mm, envelope=env, origin=org, bnom=bnom, **eopt)
    wall = time.perf_counter() - t0
    g4 = sgs.merge(ent[T.KEY + ["sp", "t"]].rename(columns={"t": "t_entry"}), on=T.KEY + ["sp"])
    g4 = g4[g4.t >= g4.t_entry - 1e-6]
    res = {"inputs": len(inputs), "species": species, "census": census, "geometry_check": geo,
           "vd_counts_primary": vd.vdid.value_counts().sort_index().to_dict(), "origin_mm": org.tolist(), "bnom_T": bnom.tolist(),
           "field": "none" if field is None else (os.path.basename(field_map) if field_map else f"bz={bz}"),
           "chord_mm": chord_mm, "sagitta_mm": sagitta_mm, "seed": seed, "entry_vds": list(entry_vds),
           "entries": int(len(ent)), "entry_vd_counts": ent.vdid.value_counts().sort_index().to_dict(),
           "transport_s": wall, "transport_timing_s": getattr(fs.transport, "timing", None), "envelope": {"r": env.r, "z0": env.z0, "z1": env.z1},
           "interior_materials": sorted({str(m) for g in (inter.shells.coax.groups if inter is not None else [])
                                         for m in g["mat"]}), **compare(g4, fast, ent, final, geom)}
    # first passage: G4 steps up to the time the fast track left the envelope (tracks that left it), both arms
    out_t = final[final.status == 0][T.KEY + ["sp", "t"]].rename(columns={"t": "t_exit"})
    g1 = g4.merge(out_t, on=T.KEY + ["sp"])
    g1 = g1[g1.t <= g1.t_exit + FIRST_PASS_NS].drop(columns="t_exit")
    f1 = fast.merge(out_t[T.KEY + ["sp"]], on=T.KEY + ["sp"])
    res["first_pass"] = compare(g1, f1, ent.merge(out_t[T.KEY + ["sp"]], on=T.KEY + ["sp"]), final, geom)
    res = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(os.path.join(outdir, "fieldcompare.json"), "w") as f:
        json.dump(res, f, indent=1)
    if kw.get("write_tables"):
        # primary-only step tables for the art injector (StrawGasStepsFromTable): fast, and G4 after the entry state
        from mu2efastsim import tables as core_tables
        parts = ent[T.KEY + ["sp", "pdg"]].rename(columns={"sp": "key"}).assign(parent=-1, isprim=1, ismcprim=1)
        cols = T.KEY + ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len",
                        "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]
        g4w = g4.assign(nsgs=g4.groupby(T.KEY).sid.transform("size"))
        for name, df in (("fasttrk_field.root", fast), ("g4prim_field.root", g4w)):
            core_tables.write(os.path.join(outdir, name), {"sgsteps": df[cols].reset_index(drop=True),
                                                           "particles": parts}, as_ttree=True)
    fast.to_parquet(os.path.join(outdir, "fast_sgsteps.parquet")) if kw.get("save_steps") else None
    final.to_parquet(os.path.join(outdir, "fast_final.parquet")) if kw.get("save_steps") else None
    return res
