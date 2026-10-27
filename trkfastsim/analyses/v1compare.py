"""v1compare -- T-V1: primary-muon StrawGasSteps, fast sim against Geant4 on the same generated primaries.

Reference: StrawGasStepTableDumper tables (sgsteps + particles); only steps of the event's MC primary muon
(ismcprim == 1, |pdg| == 13), by truth-particle identity. Fast: the `steps` subcommand's tables (every generated event).

Selection, identical in both arms: events whose primary muon passes the PrimaryFilter tracker term (>= 12 good steps).
The G4 tables hold only filter-accepted events, so a rate is compared per second of live time (both arms ran the same
CRY primaries, so the live times are equal), never by restricting the fast sim to the G4-accepted events: that drops
fast muons whose G4 twin scattered out of the tracker and biases the fast arm low.

Compared distributions among selected events, against the agreed T-V1 thresholds: straws per muon (mean within 1 %),
per-plane counts, path length (mean within 0.5 %, binned ratio +-2 %), eIon (MPV within 2 % after a 2-sigma
allowance, from a Moyal fit to the peak, stats.mpv_fit; mean within 3 %, binned ratio +-5 %); and the world-model observables at the first straw: |p|, direction cosines and entry position (median
bias, binned ratio); the in-tracker scattering: deflection between the first and last straw per |p| bin; the rate per
second of live time per bin of |p| at the first straw (selected muons, Poisson errors). T-V1b (`all_steps`): every StrawGasStep of the selected events, muon and secondaries: steps and
eIon per event, eIon per straw (muon + secondaries summed: what the digitiser sees), non-muon share, in-muon-straw vs
elsewhere. Per-straw matching on the same event is reported only as a geometry diagnostic (exact for stage
A paths; with independent multiple scattering in both arms its position spread measures the scattering, not a bias).
"""
import numpy as np

from mu2efastsim import plots, stats
from .. import geometry, tables as T


def _primary_steps(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="inner")
    return s[(s.ismcprim == 1) & (s.pdg.abs() == 13)].reset_index(drop=True)


def _all_steps(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    return s.merge(p, on=T.KEY + ["sp"], how="left")


def _select(s):
    f = T.trk_filter(s)
    keep = f[f.pass_trk].reset_index()[T.KEY]
    return s.merge(keep, on=T.KEY)


def _first(s):
    f = s.sort_values("t").groupby(T.KEY, sort=False).first().reset_index()
    p = np.sqrt(f.px ** 2 + f.py ** 2 + f.pz ** 2)
    pos = geometry.to_mu2e(f[["x0", "y0", "z0"]].to_numpy())
    return {"p": p.to_numpy(), "ux": (f.px / p).to_numpy(), "uy": (f.py / p).to_numpy(), "uz": (f.pz / p).to_numpy(),
            "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}


DEFL_P_EDGES = np.array([50.0, 300.0, 500.0, 1000.0, 2000.0, 5000.0, 1e9])   # MeV/c, |p| at the first straw


def _deflection(s):
    """Per selected muon: space angle [mrad] between its momentum at the first and at the last straw (time order),
    |p| at the first straw, and the vertical drop between them [mm]."""
    s = s.sort_values("t")
    g = s.groupby(T.KEY, sort=False)
    a, b = g[["px", "py", "pz", "y0"]].first().to_numpy(), g[["px", "py", "pz", "y0"]].last().to_numpy()
    pa, pb = np.linalg.norm(a[:, :3], axis=1), np.linalg.norm(b[:, :3], axis=1)
    c = np.clip(np.einsum("ij,ij->i", a[:, :3], b[:, :3]) / np.maximum(pa * pb, 1e-12), -1, 1)
    return 1e3 * np.arccos(c), pa, a[:, 3] - b[:, 3]


RATE_P_EDGES = np.array([0.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0, 1e9])   # MeV/c


def _rate_by_p(a, b, lt_a, lt_b):
    """Selected muons per second of live time by |p| at the first straw, arm a against arm b, with Poisson errors:
    rows (lo, hi, n_a, n_b, rate_a, rate_b, ratio - 1, error of the ratio)."""
    ha, _ = np.histogram(a, RATE_P_EDGES)
    hb, _ = np.histogram(b, RATE_P_EDGES)
    rows = []
    for k in range(len(RATE_P_EDGES) - 1):
        na, nb = int(ha[k]), int(hb[k])
        if nb == 0:
            continue
        r = (na / lt_a) / (nb / lt_b)
        rows.append({"p": [float(RATE_P_EDGES[k]), float(min(RATE_P_EDGES[k + 1], 1e6))], "n": [na, nb],
                     "rate_per_s": [na / lt_a, nb / lt_b], "ratio_minus_1": float(r - 1),
                     "ratio_err": float(r * np.sqrt(1 / max(na, 1) + 1 / nb))})
    return rows


def _deflection_block(f, r):
    """In-tracker deflection, first to last straw, fast against G4 per |p| bin: median, q90, q99 and the share above
    10x the bin median (the single-scattering tail)."""
    out = {}
    (tf, pf, _), (tr_, pr, _) = _deflection(f), _deflection(r)
    for lo, hi in zip(DEFL_P_EDGES[:-1], DEFL_P_EDGES[1:]):
        mf, mr = (pf >= lo) & (pf < hi), (pr >= lo) & (pr < hi)
        if mf.sum() < 20 or mr.sum() < 20:
            continue
        a, b = tf[mf], tr_[mr]
        med = np.median(b)
        out[f"{int(lo)}-{int(min(hi, 1e6))}"] = {
            "n": [int(mf.sum()), int(mr.sum())],
            "median_mrad": [float(np.median(a)), float(med)],
            "q90_mrad": [float(np.quantile(a, 0.9)), float(np.quantile(b, 0.9))],
            "q99_mrad": [float(np.quantile(a, 0.99)), float(np.quantile(b, 0.99))],
            "share_above_10x_g4_median": [float(np.mean(a > 10 * med)), float(np.mean(b > 10 * med))]}
    return out


def _band(a, b):
    r = stats.ratio_band(a, b)
    return {k: v for k, v in r.items() if k.startswith("max") or k == "nbins_used"}


def _band_ok(band, limit):
    """Binned-ratio threshold: max |ratio - 1| after the 2-sigma allowance <= limit; no usable bins fails."""
    v = band.get("max_abs_dev_minus_2sigma")
    return v is not None and v <= limit


def _rel(a, b):
    return float(a / b - 1)


def _all_steps_block(fa_all, ra_all, f_ev, r_ev):
    """T-V1b: every StrawGasStep (muon + secondaries) in the selected events: per-event step count and eIon sum,
    per-step eIon/length, the non-muon share, and steps sharing a muon straw vs elsewhere."""
    out = {}
    for lab, s, ev in (("fast", fa_all, f_ev), ("g4", ra_all, r_ev)):
        s = s.merge(ev, on=T.KEY)
        nonmu = ~((s.ismcprim == 1) & (s.pdg.abs() == 13))
        mu_st = s[~nonmu][T.KEY + ["sid"]].drop_duplicates().assign(_m=1)
        inmu = s.merge(mu_st, on=T.KEY + ["sid"], how="left")._m.notna().to_numpy()
        per = s.groupby(T.KEY).agg(n=("eion", "size"), e=("eion", "sum"))
        out[lab] = {"steps": int(len(s)), "nonmuon_step_share": float(nonmu.mean()),
                    "nonmuon_eion_share": float(s.eion[nonmu].sum() / s.eion.sum()),
                    "nonmuon_in_muon_straw": int((nonmu & inmu).sum()), "nonmuon_elsewhere": int((nonmu & ~inmu).sum()),
                    "steps_per_event_mean": float(per.n.mean()), "eion_per_event_keV_mean": float(1e3 * per.e.mean()),
                    "_per_n": per.n.to_numpy(float), "_per_e": 1e3 * per.e.to_numpy(), "_eion": 1e3 * s.eion.to_numpy(),
                    "_eion_inmu_nonmu": 1e3 * s.eion[nonmu & inmu].to_numpy(),
                    "_eion_straw": 1e3 * s.groupby(T.KEY + ["sid"]).eion.sum().to_numpy()}
    f, r = out["fast"], out["g4"]
    cmp = {"steps_per_event_ratio_minus_1": _rel(f["steps_per_event_mean"], r["steps_per_event_mean"]),
           "eion_per_event_ratio_minus_1": _rel(f["eion_per_event_keV_mean"], r["eion_per_event_keV_mean"]),
           "eion_per_straw_mean_ratio_minus_1": _rel(f["_eion_straw"].mean(), r["_eion_straw"].mean()),
           "eion_per_straw_mpv": {"fast": stats.mpv_fit(f["_eion_straw"], 0, 4, 200)[0], "g4": stats.mpv_fit(r["_eion_straw"], 0, 4, 200)[0]},
           "eion_per_straw_band": _band(f["_eion_straw"], r["_eion_straw"]),
           "eion_step_band": _band(f["_eion"], r["_eion"]),
           "nonmuon_in_muon_straw_per_event_ratio_minus_1": _rel(f["nonmuon_in_muon_straw"] / len(f["_per_n"]),
                                                                  max(r["nonmuon_in_muon_straw"], 1) / len(r["_per_n"])),
           "nonmuon_instraw_eion_band": _band(f["_eion_inmu_nonmu"], r["_eion_inmu_nonmu"]) if len(f["_eion_inmu_nonmu"]) else {}}
    for lab in ("fast", "g4"):
        for k in [k for k in out[lab] if k.startswith("_")]:
            del out[lab][k]
    out["compare"] = cmp
    return out


def _same_primary_straw(fast, g4, gen):
    """Match step identities through generator index, excluding ambiguous repeated particle/straw crossings."""
    from mu2efastsim.tables import all_primary_muons
    mu = all_primary_muons(gen)
    if "idx" not in mu:
        raise ValueError("same-particle timing requires generator indices")
    mapping = mu[T.KEY + ["mukey", "idx"]].rename(columns={"mukey": "sp"})
    f = fast[T.KEY + ["sp", "sid"]].assign(row=np.arange(len(fast))).merge(
        mapping, on=T.KEY + ["sp"], validate="many_to_one")
    f["sp_g4"] = f.idx + 1
    g = g4[T.KEY + ["sp", "sid"]].rename(columns={"sp": "sp_g4"}).assign(row=np.arange(len(g4)))
    key = T.KEY + ["sp_g4", "sid"]
    f = f[~f.duplicated(key, keep=False)]
    g = g[~g.duplicated(key, keep=False)]
    paired = f[key + ["row"]].merge(g[key + ["row"]], on=key, validate="one_to_one", suffixes=("_f", "_g"))
    return paired.row_f.to_numpy(), paired.row_g.to_numpy()


def run(inputs, outdir, compare=None, **kw):
    if not compare:
        raise SystemExit("v1compare FAST --compare G4REF")
    lt_f, lt_r = T.livetime(T.expand(inputs)), T.livetime(T.expand(compare))
    fa, ra = _primary_steps(inputs), _primary_steps(compare)
    f, r = _select(fa), _select(ra)
    fa_all, ra_all = _all_steps(inputs), _all_steps(compare)
    nf, nr = f.groupby(T.KEY).size(), r.groupby(T.KEY).size()
    out = {"livetime_s": {"fast": lt_f, "g4": lt_r},
           "events_selected": {"fast": int(len(nf)), "g4": int(len(nr))},
           "rate_per_s": {"fast": len(nf) / lt_f, "g4": len(nr) / lt_r, "ratio_minus_1": _rel(len(nf) / lt_f, len(nr) / lt_r),
                          "stat_err_rel": float(np.sqrt(1 / max(len(nf), 1) + 1 / max(len(nr), 1)))},
           "events_any_step": {"fast": int(fa.groupby(T.KEY).ngroups), "g4": int(ra.groupby(T.KEY).ngroups)},
           "straws_per_muon_mean": {"fast": float(nf.mean()), "g4": float(nr.mean()), "ratio_minus_1": _rel(nf.mean(), nr.mean())},
           "straws_per_muon_band": _band(nf.to_numpy(float), nr.to_numpy(float))}
    pf, pr = f.plane.value_counts(normalize=True).sort_index(), r.plane.value_counts(normalize=True).sort_index()
    out["per_plane_share_ratio_minus_1_maxabs"] = float(np.abs(pf.reindex(pr.index, fill_value=0) / pr - 1).max())
    out["len_mean"] = {"fast": float(f.len.mean()), "g4": float(r.len.mean()), "ratio_minus_1": _rel(f.len.mean(), r.len.mean())}
    out["len_band"] = _band(f.len, r.len)
    ef, er = 1e3 * f.eion.to_numpy(), 1e3 * r.eion.to_numpy()   # keV
    (mf, sf), (mr, sr) = stats.mpv_fit(ef, 0, 4, 200), stats.mpv_fit(er, 0, 4, 200)
    out["eion_keV"] = {"mean_fast": float(ef.mean()), "mean_g4": float(er.mean()), "mean_ratio_minus_1": _rel(ef.mean(), er.mean()),
                       "mpv_fast": mf, "mpv_g4": mr, "mpv_err_fast": sf, "mpv_err_g4": sr,
                       "mpv_argmax_fast": stats.mpv(ef, 0, 4, 200), "mpv_argmax_g4": stats.mpv(er, 0, 4, 200)}
    out["eion_keV"]["mpv_ratio_minus_1"] = mf / mr - 1
    out["eion_keV"]["mpv_ratio_err"] = float(mf / mr * np.hypot(sf / mf, sr / mr))
    out["eion_band"] = _band(ef, er)
    a, b = _first(f), _first(r)
    out["first_straw"] = {}
    for k in ("p", "ux", "uy", "uz", "x", "z"):
        out["first_straw"][k] = {"median_fast": float(np.median(a[k])), "median_g4": float(np.median(b[k])),
                                 "q10_q90_fast": np.quantile(a[k], [0.1, 0.9]).tolist(),
                                 "q10_q90_g4": np.quantile(b[k], [0.1, 0.9]).tolist(), "band": _band(a[k], b[k])}
    # Match the same generated primary and straw; time/position proximity never defines identity.
    ia, ib = _same_primary_straw(fa, ra, T.read_many(inputs, "gen"))
    diag = {"n": int(len(ia)), "frac_of_g4": float(len(ia) / max(len(ra), 1)),
            "matching": "event, generator primary identity, straw; repeated crossings excluded"}
    if len(ia):
        dt = fa.t.to_numpy()[ia] - ra.t.to_numpy()[ib]
        dx = 1e3 * (fa[["x0", "y0", "z0"]].to_numpy(float)[ia] - ra[["x0", "y0", "z0"]].to_numpy(float)[ib])
        diag.update({"time_bias_ns": float(np.median(dt)), "time_core_ns": stats.resid_stats(dt)["coresig"],
                     "time_mean_ns": float(np.mean(dt)),
                     "len_ratio_minus_1": _rel(fa.len.to_numpy()[ia].mean(), ra.len.to_numpy()[ib].mean()),
                     # entry position, detector frame, median per axis [um] (spread = scattering, median = bias)
                     "entry_bias_um": np.median(dx, axis=0).tolist()})
        # time bias by the G4 |p| at the step (flight-time convention matters for slow muons)
        pg = np.sqrt(ra.px ** 2 + ra.py ** 2 + ra.pz ** 2).to_numpy()[ib]
        edges = [0.0, 200.0, 300.0, 500.0, 1000.0, 1e9]
        k = np.clip(np.digitize(pg, edges) - 1, 0, len(edges) - 2)
        diag["time_bias_by_p_g4"] = [{"p": edges[i:i + 2], "n": int((k == i).sum()),
                                      "median_ns": float(np.median(dt[k == i])) if (k == i).any() else None,
                                      "mean_ns": float(np.mean(dt[k == i])) if (k == i).any() else None}
                                     for i in range(len(edges) - 1)]
    out["same_straw_diag"] = diag
    th = {"straws_per_muon": abs(out["straws_per_muon_mean"]["ratio_minus_1"]) <= 0.01,
          "len_mean": abs(out["len_mean"]["ratio_minus_1"]) <= 0.005,
          "len_band": _band_ok(out["len_band"], 0.02),
          # |MPV ratio - 1| within 2 % after a 2-sigma statistical allowance (the fit error, not a bin width)
          "eion_mpv": bool(abs(out["eion_keV"]["mpv_ratio_minus_1"]) - 2 * out["eion_keV"]["mpv_ratio_err"] <= 0.02),
          "eion_mean": abs(out["eion_keV"]["mean_ratio_minus_1"]) <= 0.03,
          "eion_band": _band_ok(out["eion_band"], 0.05),
          "time_bias_same_straw": len(ia) > 0 and abs(diag["time_bias_ns"]) <= 0.05,
          "entry_bias_same_straw": len(ia) > 0 and max(abs(x) for x in diag["entry_bias_um"]) <= 10.0}
    out["deflection_first_to_last"] = _deflection_block(f, r)      # [fast, g4] per |p| bin
    out["rate_by_p_first"] = _rate_by_p(a["p"], b["p"], lt_f, lt_r)   # [fast, g4]: muons per live-time second
    out["all_steps"] = _all_steps_block(fa_all, ra_all, f[T.KEY].drop_duplicates(), r[T.KEY].drop_duplicates())
    out["thresholds_pass"] = th
    for k, v in out.items():
        print(f"{k:34s} {v}")
    plots.write_json(outdir, "v1compare.json", out)
