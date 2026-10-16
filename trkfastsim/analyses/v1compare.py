"""v1compare -- T-V1: primary-muon StrawGasSteps, fast sim against Geant4 on the same generated primaries.

Reference: StrawGasStepTableDumper tables (sgsteps + particles); only steps of the event's MC primary muon
(ismcprim == 1, |pdg| == 13), by truth-particle identity. Fast: the `steps` subcommand's tables (every generated event).

Selection, identical in both arms: events whose primary muon passes the PrimaryFilter tracker term (>= 12 good steps).
The G4 tables hold only filter-accepted events, so a rate is compared per second of live time (both arms ran the same
CRY primaries, so the live times are equal), never by restricting the fast sim to the G4-accepted events: that drops
fast muons whose G4 twin scattered out of the tracker and biases the fast arm low.

Compared distributions among selected events, against the agreed T-V1 thresholds: straws per muon (mean within 1 %),
per-plane counts, path length (mean within 0.5 %, binned ratio +-2 %), eIon (MPV within 2 %, mean within 3 %, binned
ratio +-5 %); and the world-model observables at the first straw: |p|, direction cosines and entry position (median
bias, binned ratio). Per-straw matching on the same event is reported only as a geometry diagnostic (exact for stage
A paths; with independent multiple scattering in both arms its position spread measures the scattering, not a bias).
"""
import numpy as np

from mu2efastsim import match, plots, stats
from .. import geometry, tables as T


def _primary_steps(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="inner")
    return s[(s.ismcprim == 1) & (s.pdg.abs() == 13)].reset_index(drop=True)


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


def _band(a, b):
    r = stats.ratio_band(a, b)
    return {k: v for k, v in r.items() if k.startswith("max") or k == "nbins_used"}


def _rel(a, b):
    return float(a / b - 1)


def run(inputs, outdir, compare=None, **kw):
    if not compare:
        raise SystemExit("v1compare FAST --compare G4REF")
    lt_f, lt_r = T.livetime(T.expand(inputs)), T.livetime(T.expand(compare))
    fa, ra = _primary_steps(inputs), _primary_steps(compare)
    f, r = _select(fa), _select(ra)
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
    out["eion_keV"] = {"mean_fast": float(ef.mean()), "mean_g4": float(er.mean()), "mean_ratio_minus_1": _rel(ef.mean(), er.mean()),
                       "mpv_fast": stats.mpv(ef, 0, 4, 200), "mpv_g4": stats.mpv(er, 0, 4, 200)}
    out["eion_keV"]["mpv_ratio_minus_1"] = out["eion_keV"]["mpv_fast"] / out["eion_keV"]["mpv_g4"] - 1
    out["eion_band"] = _band(ef, er)
    a, b = _first(f), _first(r)
    out["first_straw"] = {}
    for k in ("p", "ux", "uy", "uz", "x", "z"):
        out["first_straw"][k] = {"median_fast": float(np.median(a[k])), "median_g4": float(np.median(b[k])),
                                 "q10_q90_fast": np.quantile(a[k], [0.1, 0.9]).tolist(),
                                 "q10_q90_g4": np.quantile(b[k], [0.1, 0.9]).tolist(), "band": _band(a[k], b[k])}
    # geometry diagnostic: same event, same straw (exact for straight stage-A paths)
    kf = np.c_[fa[T.KEY].to_numpy(np.int64), fa.sid.to_numpy(np.int64)]
    kr = np.c_[ra[T.KEY].to_numpy(np.int64), ra.sid.to_numpy(np.int64)]
    ia, ib = match.join(kf, kr)
    diag = {"n": int(len(ia)), "frac_of_g4": float(len(ia) / max(len(ra), 1))}
    if len(ia):
        dt = fa.t.to_numpy()[ia] - ra.t.to_numpy()[ib]
        diag.update({"time_bias_ns": float(np.median(dt)), "time_core_ns": stats.resid_stats(dt)["coresig"],
                     "len_ratio_minus_1": _rel(fa.len.to_numpy()[ia].mean(), ra.len.to_numpy()[ib].mean())})
    out["same_straw_diag"] = diag
    th = {"straws_per_muon": abs(out["straws_per_muon_mean"]["ratio_minus_1"]) <= 0.01,
          "len_mean": abs(out["len_mean"]["ratio_minus_1"]) <= 0.005,
          "len_band": (out["len_band"].get("max_abs_dev_minus_2sigma") or 1) <= 0.02,
          "eion_mpv": abs(out["eion_keV"]["mpv_ratio_minus_1"]) <= 0.02,
          "eion_mean": abs(out["eion_keV"]["mean_ratio_minus_1"]) <= 0.03,
          "eion_band": (out["eion_band"].get("max_abs_dev_minus_2sigma") or 1) <= 0.05,
          "time_bias_same_straw": len(ia) > 0 and abs(diag["time_bias_ns"]) <= 0.05}
    out["thresholds_pass"] = th
    for k, v in out.items():
        print(f"{k:34s} {v}")
    plots.write_json(outdir, "v1compare.json", out)
