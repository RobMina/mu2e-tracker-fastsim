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

from mu2efastsim import ehelix, eloss, stats
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


# CosmicSignal selection (CosmicMixingFilter SignalFilter / CalibFilter of the production job): per SimParticle, straw
# steps with |p| in the window, a minimum count and (signal) a minimum plane span; counted over every particle
SELF_FILTERS = {"signal": dict(p=(30.0, 250.0), n=15, span=16), "calib": dict(p=(1000.0, 1e6), n=20, span=0)}


def self_pass(sgs):
    """Per track (KEY, sp) of `sgs`: whether its own steps pass the sample's SignalFilter or CalibFilter windows (impact,
    time and edep cuts not applied). A track that fails both was kept through another particle's steps (a delta
    electron, a decay electron)."""
    key = T.KEY + ["sp"]
    p = np.sqrt(sgs.px ** 2 + sgs.py ** 2 + sgs.pz ** 2)
    out = sgs[key].drop_duplicates().reset_index(drop=True)
    for name, c in SELF_FILTERS.items():
        g = sgs[(p > c["p"][0]) & (p < c["p"][1])].groupby(key).plane.agg(["size", "min", "max"]).reset_index()
        g[name] = (g["size"] >= c["n"]) & ((g["max"] - g["min"]).abs() >= c["span"])
        out = out.merge(g[key + [name]], on=key, how="left")
        out[name] = out[name].fillna(False).astype(bool)
    out["self"] = out.signal | out.calib
    return out


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


def _entry_geometry(ent, origin):
    """Per entry type: quantiles (10, 50, 90 %) of the radius about the tracker axis, z, the radial direction cosine,
    and for SurfaceStep entries the step length |x1 - x0|."""
    out = {}
    for v, g in ent.groupby("vdid"):
        r = np.hypot(g.x - origin[0], g.y - origin[1])
        p = np.sqrt(g.px ** 2 + g.py ** 2 + g.pz ** 2)
        ur = ((g.x - origin[0]) * g.px + (g.y - origin[1]) * g.py) / np.maximum(r * p, 1e-9)
        q = lambda a: [float(np.quantile(a, k)) for k in (0.1, 0.5, 0.9)]
        d = {"n": int(len(g)), "r": q(r), "z": q(g.z), "u_r": q(ur)}
        if "x1" in g and g.x1.notna().any():
            d["step_mm"] = q(np.sqrt((g.x1 - g.x0) ** 2 + (g.y1 - g.y0) ** 2 + (g.z1 - g.z0) ** 2).dropna())
        out[int(v)] = d
    return out


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


DROP_MEV = 10.0                    # big_drops: a |p| drop between consecutive straws above this
DROP_R_EDGES = np.array([0.0, 380.0, 500.0, 600.0, 700.0, 800.0, 900.0])
DROP_Z_EDGES = np.array([-1700.0, -1600.0, -800.0, 0.0, 800.0, 1600.0, 1700.0])


def big_drops(g4, fast, ent, p_min=1000.0):
    """Where tracks above p_min lose more than DROP_MEV between two consecutive straws (time order), per arm: the
    fraction of tracks with such a drop, the summed drop per track, and counts of the drops in (r, z) cells of the
    midpoint between the two straws (detector frame: r about the tracker axis, z along it). Locates material that
    one arm crosses and the other does not."""
    key = T.KEY + ["sp"]
    pe = ent[key + ["px", "py", "pz"]].assign(pent=lambda d: np.sqrt(d.px ** 2 + d.py ** 2 + d.pz ** 2))
    sel = pe[pe.pent >= p_min][key]
    out = {"p_min_MeV": p_min, "drop_MeV": DROP_MEV, "r_edges": DROP_R_EDGES.tolist(), "z_edges": DROP_Z_EDGES.tolist()}
    for name, st_ in (("g4", g4), ("fast", fast)):
        s = st_.merge(sel, on=key).sort_values(key + ["t"])
        p = np.sqrt(s.px ** 2 + s.py ** 2 + s.pz ** 2).to_numpy()
        same = np.r_[False, (s[key].to_numpy()[1:] == s[key].to_numpy()[:-1]).all(1)]
        dp = np.r_[np.nan, p[:-1] - p[1:]]
        big = same & (dp > DROP_MEV)
        mx = 0.5 * (s.x1.to_numpy()[:-1] + s.x0.to_numpy()[1:])
        my = 0.5 * (s.y1.to_numpy()[:-1] + s.y0.to_numpy()[1:])
        mz = 0.5 * (s.z1.to_numpy()[:-1] + s.z0.to_numpy()[1:])
        r, z = np.r_[np.nan, np.hypot(mx, my)], np.r_[np.nan, mz]
        ntrk = int(len(s[key].drop_duplicates()))
        tb = s[big][key].drop_duplicates()
        h, _, _ = np.histogram2d(r[big], z[big], [DROP_R_EDGES, DROP_Z_EDGES])
        out[name] = {"tracks": ntrk, "tracks_with_drop": int(len(tb)), "fraction": float(len(tb) / max(ntrk, 1)),
                     "drops": int(big.sum()), "summed_drop_MeV_per_track": float(dp[big].sum() / max(ntrk, 1)),
                     "cells_r_by_z": h.astype(int).tolist()}
    return out


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
    if "vdid" in ent:
        pv = per.merge(ent[key + ["vdid"]], on=key)
        hp = pv.pent >= 2000.0
        out["match_by_entry_above_2GeV"] = {int(v): _mean_err(g.match) for v, g in pv[hp].groupby("vdid")}
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
            drop_g, drop_f = pent[k] - pg.to_numpy()[k], pent[k] - pf.to_numpy()[k]
            dtl = (m.t_f - m.t_g).to_numpy()[k]
            bins.append({"p_lo": float(lo), "p_hi": float(hi), "dp_MeV": _mean_err(dp[k]),
                         "dt_ns_mean_median": [float(np.mean(dtl)), float(np.median(dtl))],
                         "dexit_mm_q50_q90": [float(np.median(dexit[k])), float(np.quantile(dexit[k], 0.9))],
                         "g4_p_drop_MeV": _mean_err(drop_g), "fast_p_drop_MeV": _mean_err(drop_f),
                         "drop_q10_q50_q90_q99": {"g4": [float(np.quantile(drop_g, q)) for q in (0.1, 0.5, 0.9, 0.99)],
                                                  "fast": [float(np.quantile(drop_f, q)) for q in (0.1, 0.5, 0.9, 0.99)]},
                         "entry_p_median": float(np.median(pent[k])),
                         "dexit_mm_median": float(np.median(dexit[k]))})
    out["last_straw"]["by_entry_p"] = bins
    out["big_drops"] = big_drops(g4, fast, ent)
    # entry-to-first-straw time
    g1 = g4.sort_values("t").groupby(key).first().reset_index()[key + ["sid", "t"]]
    f1 = fast.sort_values("t").groupby(key).first().reset_index()[key + ["sid", "t"]]
    t1 = g1.merge(f1, on=key + ["sid"], suffixes=("_g", "_f"))
    out["first_straw_dt_ns"] = _mean_err(t1.t_f - t1.t_g)
    # entry point offset at the first G4 straw (both arms cross it first), per entry type, above 2 GeV/c: the spread of
    # the scattering between the entry state and the straws (support material), same field in both arms
    g1x = g4.sort_values("t").groupby(key).first().reset_index()[key + ["sid", "x0", "y0", "z0"]]
    f1x = fast.sort_values("t").groupby(key).first().reset_index()[key + ["sid", "x0", "y0", "z0"]]
    fx = g1x.merge(f1x, on=key + ["sid"], suffixes=("_g", "_f")).merge(pe, on=key)
    if "vdid" in ent:
        fx = fx.merge(ent[key + ["vdid"]], on=key)
        dd = np.sqrt((fx.x0_f - fx.x0_g) ** 2 + (fx.y0_f - fx.y0_g) ** 2 + (fx.z0_f - fx.z0_g) ** 2)
        hp = (fx.pent >= 2000.0).to_numpy()
        out["first_straw_offset_mm_above_2GeV"] = {
            int(v): [int(k.sum()), float(np.median(dd[k])), float(np.quantile(dd[k], 0.9))]
            for v in np.unique(fx.vdid) for k in [hp & (fx.vdid == v).to_numpy()] if k.sum() >= 5}
    out["final_status"] = final.status.value_counts().sort_index().to_dict()
    return out


def matched_straws(g4, fast, ent, geom, origin):
    """Per straw crossed by both arms (same track, same straw, first crossing in time): fast - G4 differences of the
    step time, of the distance of closest approach of the step chord to the wire (the drift distance the digitiser
    sees) and of the step length, per entry |p| bin: median and IQR/1.349, and the mean with its error."""
    key = T.KEY + ["sp"]
    cols = key + ["sid", "t", "x0", "y0", "z0", "x1", "y1", "z1", "len"]
    a = g4[cols].sort_values("t").groupby(key + ["sid"], sort=False).first().reset_index()
    b = fast[cols].sort_values("t").groupby(key + ["sid"], sort=False).first().reset_index()
    m = a.merge(b, on=key + ["sid"], suffixes=("_g", "_f"))
    pe = ent[key + ["px", "py", "pz"]].assign(pent=lambda d: np.sqrt(d.px ** 2 + d.py ** 2 + d.pz ** 2))[key + ["pent"]]
    m = m.merge(pe, on=key)
    si = _straw_index(m, geom)
    w0, wd = geom.mid[si], geom.dir[si]

    def doca(sfx):
        p0 = geometry.to_mu2e(m[[f"x0{sfx}", f"y0{sfx}", f"z0{sfx}"]].to_numpy(), origin)
        p1 = geometry.to_mu2e(m[[f"x1{sfx}", f"y1{sfx}", f"z1{sfx}"]].to_numpy(), origin)
        u = p1 - p0
        n = np.cross(u, wd)
        nn = np.linalg.norm(n, axis=1)
        return np.abs(((p0 - w0) * n).sum(1)) / np.maximum(nn, 1e-12)

    dd = doca("_f") - doca("_g")
    dt = (m.t_f - m.t_g).to_numpy()
    dl = (m.len_f - m.len_g).to_numpy()
    pent = m.pent.to_numpy()

    def summ(x):
        if len(x) < 5:
            return {}
        q = np.quantile(x, [0.25, 0.5, 0.75])
        return {"median": float(q[1]), "iqr_sigma": float((q[2] - q[0]) / 1.349), "mean": _mean_err(x)}
    out = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        k = (pent >= lo) & (pent < hi)
        if k.sum() >= 5:
            out.append({"p_lo": float(lo), "p_hi": float(hi), "straws": int(k.sum()), "dt_ns": summ(dt[k]),
                        "ddoca_mm": summ(dd[k]), "dlen_mm": summ(dl[k])})
    return out


def delta_census(paths, part, sgs, geom, t_min=10.0, chunk=2000):
    """Electrons and positrons with kinetic energy above t_min [MeV] whose parent is one of the primaries (`part`), by
    creation process code and by where they start: in a straw's gas, in its wall (r about the straw axis up to
    wall_rmax), or elsewhere; per primary straw crossing (`sgs` rows), with the kinetic-energy mean. A muon's hard
    knock-on rate in the straws is (K/2)(Z/A) rho x / t_min per crossing (about 1e-4 at 10 MeV for walls and gas)."""
    d = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "parent", "ccode", "se", "sx", "sy", "sz"])
    d = d.merge(part[T.KEY + ["sp"]].rename(columns={"sp": "parent"}), on=T.KEY + ["parent"])
    d = d[(d.pdg.abs() == 11) & (d.se - eloss.ME > t_min)].reset_index(drop=True)
    where = np.full(len(d), "other", dtype=object)
    x = d[["sx", "sy", "sz"]].to_numpy()
    r_out = geom.wall_rmax if geom.wall_rmax is not None else geom.radius + 0.015
    for a in range(0, len(d), chunk):
        xx = x[a:a + chunk]
        dv = xx[:, None, :] - geom.mid[None]
        t = (dv * geom.dir[None]).sum(2)
        perp = np.linalg.norm(dv - t[..., None] * geom.dir[None], axis=2)
        perp = np.where(np.abs(t) <= geom.halflen[None], perp, np.inf)
        pm = perp.min(1)
        where[a:a + chunk] = np.where(pm <= geom.radius, "gas", np.where(pm <= r_out, "wall", "other"))
    n_cross = int(len(sgs))
    out = {"t_min_MeV": t_min, "primary_straw_crossings": n_cross, "tracks": int(len(sgs[T.KEY + ["sp"]].drop_duplicates()))}
    for w in ("gas", "wall", "other"):
        k = where == w
        out[w] = {"n": int(k.sum()), "per_crossing": float(k.sum() / max(n_cross, 1)),
                  "tkin_mean_MeV": float((d.se[k] - eloss.ME).mean()) if k.any() else float("nan"),
                  "by_ccode": {int(c): int(n) for c, n in d[k].ccode.value_counts().head(6).items()}}
    return out


OFFSET_P_EDGES = np.array([2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 1e9])


def offset_bound(g4, fast, ent, inter, origin):
    """First-straw entry offset fast vs G4 per entry type and |p| bin (>= 2 GeV/c), against the spread that Geant4's
    own scattering in the support between the entry state and the first straw gives: Highland theta0 on the x/X0 of
    the support along the entry-to-first-straw line (inter._support), lever arm L = that line's length. Predicted
    median 3D offset = 1.177 sigma (Rayleigh) with sigma = theta0 L k, k = 1/sqrt(3) (material spread along the line)
    and k = 1 (material at the entry); a fast arm with its own support scattering doubles the variance (x sqrt 2).
    Also median(offset x p) per bin: constant across bins if the offset is scattering."""
    from mu2efastsim import eloss as _el, materials as _mt
    key = T.KEY + ["sp"]
    g1 = g4.sort_values("t").groupby(key).first().reset_index()[key + ["sid", "x0", "y0", "z0"]]
    f1 = fast.sort_values("t").groupby(key).first().reset_index()[key + ["sid", "x0", "y0", "z0"]]
    m = g1.merge(f1, on=key + ["sid"], suffixes=("_g", "_f")).merge(
        ent[key + ["x", "y", "z", "px", "py", "pz", "vdid"]], on=key)
    p = np.sqrt(m.px ** 2 + m.py ** 2 + m.pz ** 2).to_numpy()
    m, p = m[p >= OFFSET_P_EDGES[0]], p[p >= OFFSET_P_EDGES[0]]
    if not len(m):
        return {}
    d = np.sqrt((m.x0_f - m.x0_g) ** 2 + (m.y0_f - m.y0_g) ** 2 + (m.z0_f - m.z0_g) ** 2).to_numpy()
    a = m[["x", "y", "z"]].to_numpy()
    b = geometry.to_mu2e(m[["x0_g", "y0_g", "z0_g"]].to_numpy(), origin)
    L = np.linalg.norm(b - a, axis=1)
    col, _ = inter._support(a, (b - a) / np.maximum(L, 1e-9)[:, None], L, False)
    xx = sum(c / _el.radiation_length_mm(_mt.get(nm)) for nm, c in col.items()) if col else np.zeros(len(m))
    th0 = np.where(xx > 0, _el.highland_theta0(p, np.maximum(xx, 1e-12)), 0.0)
    out = {}
    for v in np.unique(m.vdid):
        rows = []
        for lo, hi in zip(OFFSET_P_EDGES[:-1], OFFSET_P_EDGES[1:]):
            k = ((m.vdid == v).to_numpy()) & (p >= lo) & (p < hi)
            if k.sum() < 5:
                continue
            sig = th0[k] * L[k]
            rows.append({"p_lo": float(lo), "p_hi": float(hi), "n": int(k.sum()), "median_mm": float(np.median(d[k])),
                         "q90_mm": float(np.quantile(d[k], 0.9)),
                         "median_mm_x_GeV": float(np.median(d[k] * p[k] / 1e3)),
                         "x_over_X0_median": float(np.median(xx[k])), "L_mm_median": float(np.median(L[k])),
                         "pred_median_mm_k1": float(1.177 * np.median(sig)),
                         "pred_median_mm_kuniform": float(1.177 * np.median(sig) / np.sqrt(3))})
        out[int(v)] = rows
    return out


def run(inputs, outdir, compare_=None, *, gdml, species=13, field_map=None, bz=None, no_field=False, seed=1,
        chord_mm=20.0, sagitta_mm=0.02, entry_vds=TRACKER_VDS, entry="vd", origin=None, interior=True, max_events=None,
        mu_radiative=False, gas_state="solid", **kw):
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
        cols += ["x0", "y0", "z0", "x1", "y1", "z1"]
        ent = pd.concat([entry_states(vd, entry_vds).reindex(columns=cols),
                         entry_states_surface(surf, org).reindex(columns=cols)]).sort_values("t")
        ent = ent.groupby(T.KEY + ["sp"], sort=False).first().reset_index()
    if kw.get("self_pass_only"):
        sp_ = self_pass(sgs)
        census["self_pass"] = {k: int(sp_[k].sum()) for k in ("signal", "calib", "self")} | {"tracks": int(len(sp_))}
        ent = ent.merge(sp_[sp_.self][T.KEY + ["sp"]], on=T.KEY + ["sp"])
    if no_field:
        field = None
    elif field_map:
        field = ehelix.grid_field(*ehelix.read_g4bl(field_map))
    else:
        field = np.array([0.0, 0.0, float(bz if bz is not None else 1.0)])
    env = fs.Envelope(geom, axis_xy=org[:2])
    census["entry_geometry"] = _entry_geometry(ent, org)
    inter = fs.interior(gdml, geom, env, origin=org) if interior else None
    bnom = ehelix.field_at(field if field is not None else np.zeros(3), org[None])[0]
    t0 = time.perf_counter()
    eopt = dict(loss=fs.electron_loss, tail=False) if species == 11 else dict(radiative=mu_radiative)
    eopt["gas"] = fs.STRAW_GAS_G4STATE if gas_state == "gas" else None
    fast, final = fs.transport(np.random.default_rng(seed), ent, geom, field, inter, chord_mm=chord_mm,
                               sagitta_mm=sagitta_mm, envelope=env, origin=org, bnom=bnom, **eopt)
    wall = time.perf_counter() - t0
    g4 = sgs.merge(ent[T.KEY + ["sp", "t"]].rename(columns={"t": "t_entry"}), on=T.KEY + ["sp"])
    g4 = g4[g4.t >= g4.t_entry - 1e-6]
    res = {"inputs": len(inputs), "species": species, "census": census, "geometry_check": geo,
           "vd_counts_primary": vd.vdid.value_counts().sort_index().to_dict(), "origin_mm": org.tolist(), "bnom_T": bnom.tolist(),
           "field": "none" if field is None else (os.path.basename(field_map) if field_map else f"bz={bz}"),
           "chord_mm": chord_mm, "sagitta_mm": sagitta_mm, "seed": seed, "mu_radiative": bool(mu_radiative), "gas_state": gas_state, "entry_vds": list(entry_vds),
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
    if kw.get("matched_straws"):
        res["matched_straws_first_pass"] = matched_straws(g1, f1, ent, geom, org)
    res = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(os.path.join(outdir, "fieldcompare.json"), "w") as f:
        json.dump(res, f, indent=1)
    if kw.get("delta_census"):
        res["delta_census"] = delta_census(inputs, part, g4, geom)
        with open(os.path.join(outdir, "fieldcompare.json"), "w") as f:
            json.dump(res, f, indent=1)
    if kw.get("offset_bound") and len(g4):
        res["first_straw_offset_bound"] = offset_bound(g4, fast, ent, inter if inter is not None else
                                                       fs.interior(gdml, geom, env, origin=org), org)
        with open(os.path.join(outdir, "fieldcompare.json"), "w") as f:
            json.dump(res, f, indent=1)
    if inter is not None and len(g4):
        # support material on the straight line from each entry to its first G4 straw entry, per entry type
        key = T.KEY + ["sp"]
        g1 = g4.sort_values("t").groupby(key).first().reset_index()[key + ["x0", "y0", "z0"]]
        m = ent.merge(g1, on=key, suffixes=("", "_s"))
        a = m[["x", "y", "z"]].to_numpy()
        b = geometry.to_mu2e(m[["x0_s", "y0_s", "z0_s"]].to_numpy(), org) if "x0_s" in m else \
            geometry.to_mu2e(m[["x0", "y0", "z0"]].to_numpy(), org)
        dv = b - a
        L = np.linalg.norm(dv, axis=1)
        col, _ = inter._support(a, dv / np.maximum(L, 1e-9)[:, None], L, False)
        from mu2efastsim import eloss as _el, materials as _mt
        xx = sum(c / _el.radiation_length_mm(_mt.get(nm)) for nm, c in col.items()) if col else np.zeros(len(m))
        sc = {}
        for v in np.unique(m.vdid):
            k = (m.vdid == v).to_numpy()
            sc[int(v)] = {"n": int(k.sum()), "x_over_X0_q50_q90_q99": [float(np.quantile(xx[k], q)) for q in (0.5, 0.9, 0.99)],
                          "by_material_mean_mm": {nm: float(c[k].mean()) for nm, c in col.items() if c[k].mean() > 0}}
        res["support_to_first_straw"] = sc
        with open(os.path.join(outdir, "fieldcompare.json"), "w") as f:
            json.dump(res, f, indent=1)
    if kw.get("dump_tracks"):
        # per-track step lists (time order) of the first N high-momentum tracks of each entry type, both arms
        key = T.KEY + ["sp"]
        pe = np.sqrt(ent.px ** 2 + ent.py ** 2 + ent.pz ** 2)
        sel = ent[pe > 2000.0].groupby("vdid").head(int(kw["dump_tracks"]))
        cols = ["sid", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]
        dump = []
        for _, r in sel.iterrows():
            k = {c: r[c] for c in key}
            g = g4.loc[(g4[key] == pd.Series(k)).all(1)].sort_values("t")
            f = fast.loc[(fast[key] == pd.Series(k)).all(1)].sort_values("t")
            dump.append({"key": [int(v) for v in k.values()], "vdid": int(r.vdid),
                         "entry": [float(r[c]) for c in ("x", "y", "z", "px", "py", "pz", "t")],
                         "g4": g[cols].round(3).values.tolist(), "fast": f[cols].round(3).values.tolist()})
        with open(os.path.join(outdir, "tracks.json"), "w") as fo:
            json.dump(dump, fo)
    if kw.get("write_tables"):
        # primary-only step tables for the art injector (StrawGasStepsFromTable): fast, G4 after the entry state, and
        # G4 first passage
        from mu2efastsim import tables as core_tables
        # keys renumbered 1..n per event (resampled-stage keys exceed 2^31, which the stock compression does not take)
        rk = ent[T.KEY + ["sp", "pdg"]].sort_values(T.KEY + ["sp"]).copy()
        rk["key"] = rk.groupby(T.KEY).cumcount() + 1
        parts = rk[T.KEY + ["key", "pdg"]].assign(parent=-1, isprim=1, ismcprim=1)
        rmap = rk[T.KEY + ["sp", "key"]]
        cols = T.KEY + ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len",
                        "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]
        g4w = g4.assign(nsgs=g4.groupby(T.KEY).sid.transform("size"))
        # G4 first passage: steps up to the fast exit time (+ FIRST_PASS_NS) of tracks the fast transport took out
        fx = final[T.KEY + ["sp", "t", "status"]].rename(columns={"t": "t_exit"})
        g4f = g4.merge(fx, on=T.KEY + ["sp"])
        g4f = g4f[(g4f.status != 0) | (g4f.t <= g4f.t_exit + FIRST_PASS_NS)].drop(columns=["t_exit", "status"])
        g4f = g4f.assign(nsgs=g4f.groupby(T.KEY).sid.transform("size"))
        for name, df in (("fasttrk_field.root", fast), ("g4prim_field.root", g4w), ("g4fp_field.root", g4f)):
            df = df.merge(rmap, on=T.KEY + ["sp"]).drop(columns="sp").rename(columns={"key": "sp"})
            core_tables.write(os.path.join(outdir, name), {"sgsteps": df[cols].reset_index(drop=True),
                                                           "particles": parts}, as_ttree=True)
    if kw.get("save_steps"):
        from mu2efastsim import tables as core_tables
        core_tables.write(os.path.join(outdir, "fast_steps.root"), {"sgsteps": fast, "final": final})
    return res
