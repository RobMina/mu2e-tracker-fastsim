"""v1compare -- T-V1: primary-muon StrawGasSteps, fast sim against Geant4 on the same generated primaries.

Reference: StrawGasStepTableDumper tables (sgsteps + particles); only steps of the event's MC primary muon
(ismcprim == 1, |pdg| == 13) are used, matched by truth-particle identity, never by time or position. Fast: the
`steps` subcommand's tables. Events are matched by (run, subrun, event); straws by StrawId within an event.

Reported against the agreed T-V1 thresholds: straws crossed per primary muon (within 1 %), per-plane counts, path
length (mean within 0.5 %, binned ratio +-2 %), eIon (MPV within 2 %, mean within 3 %, binned ratio +-5 %), and on
straws crossed in both arms the per-step entry position (bias <= 10 um) and time (bias <= 0.05 ns). Also the
PrimaryFilter tracker term computed from primary-muon steps only.
"""
import numpy as np

from mu2efastsim import match, plots, stats
from .. import tables as T


def _primary_steps(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="inner")
    return s[(s.ismcprim == 1) & (s.pdg.abs() == 13)].reset_index(drop=True)


def _per_event(s):
    return s.groupby(T.KEY).size()


def run(inputs, outdir, compare=None, **kw):
    if not compare:
        raise SystemExit("v1compare FAST --compare G4REF")
    f = _primary_steps(inputs)
    r = _primary_steps(compare)
    # restrict both to events present in both inputs' gen trees (same primaries)
    gf = T.read_many(inputs, "gen", T.KEY).drop_duplicates()
    gr = T.read_many(compare, "gen", T.KEY).drop_duplicates()
    common = gf.merge(gr, on=T.KEY)
    f = f.merge(common, on=T.KEY)
    r = r.merge(common, on=T.KEY)
    out = {"events_common": int(len(common)), "steps_fast": int(len(f)), "steps_g4": int(len(r))}
    nf, nr = _per_event(f), _per_event(r)
    out["events_with_steps"] = {"fast": int(len(nf)), "g4": int(len(nr))}
    out["straws_per_muon_mean"] = {"fast": float(nf.mean()), "g4": float(nr.mean()),
                                   "ratio_minus_1": float(nf.mean() / nr.mean() - 1)}
    out["steps_ratio_minus_1"] = float(len(f) / len(r) - 1)
    pf, pr = f.plane.value_counts().sort_index(), r.plane.value_counts().sort_index()
    out["per_plane_ratio_minus_1_maxabs"] = float(np.abs(pf.reindex(pr.index, fill_value=0) / pr - 1).max())
    out["len_mean"] = {"fast": float(f.len.mean()), "g4": float(r.len.mean()),
                       "ratio_minus_1": float(f.len.mean() / r.len.mean() - 1)}
    out["len_band"] = {k: v for k, v in stats.ratio_band(f.len, r.len).items() if k.startswith("max") or k == "nbins_used"}
    ef, er = 1e3 * f.eion.to_numpy(), 1e3 * r.eion.to_numpy()   # keV
    out["eion_keV"] = {"mean_fast": float(ef.mean()), "mean_g4": float(er.mean()),
                       "mean_ratio_minus_1": float(ef.mean() / er.mean() - 1),
                       "mpv_fast": stats.mpv(ef, 0, 4, 200), "mpv_g4": stats.mpv(er, 0, 4, 200)}
    out["eion_keV"]["mpv_ratio_minus_1"] = out["eion_keV"]["mpv_fast"] / out["eion_keV"]["mpv_g4"] - 1
    out["eion_band"] = {k: v for k, v in stats.ratio_band(ef, er).items() if k.startswith("max") or k == "nbins_used"}
    # per-straw matching within an event (primary muon only; one step per straw per particle)
    kf = np.c_[f[T.KEY].to_numpy(np.int64), f.sid.to_numpy(np.int64)]
    kr = np.c_[r[T.KEY].to_numpy(np.int64), r.sid.to_numpy(np.int64)]
    ia, ib = match.join(kf, kr)
    out["straws_matched"] = {"n": int(len(ia)), "frac_of_g4": float(len(ia) / max(len(r), 1)),
                             "frac_of_fast": float(len(ia) / max(len(f), 1))}
    if len(ia):
        dpos = (f.loc[ia, ["x0", "y0", "z0"]].to_numpy() - r.loc[ib, ["x0", "y0", "z0"]].to_numpy())
        dt = f.t.to_numpy()[ia] - r.t.to_numpy()[ib]
        out["entry_bias_um"] = (1e3 * np.median(dpos, 0)).tolist()
        out["entry_core_um"] = [1e3 * stats.resid_stats(dpos[:, k])["coresig"] for k in range(3)]
        out["time_bias_ns"] = float(np.median(dt))
        out["time_core_ns"] = stats.resid_stats(dt)["coresig"]
        out["len_matched_ratio_minus_1"] = float(f.len.to_numpy()[ia].mean() / r.len.to_numpy()[ib].mean() - 1)
    tf, tr = T.trk_filter(f), T.trk_filter(r)
    out["filter_trk_primary_only"] = {"fast": int(tf.pass_trk.sum()), "g4": int(tr.pass_trk.sum())}
    th = {"straws_per_muon": abs(out["straws_per_muon_mean"]["ratio_minus_1"]) <= 0.01,
          "len_mean": abs(out["len_mean"]["ratio_minus_1"]) <= 0.005,
          "len_band": (out["len_band"].get("max_abs_dev_minus_2sigma") or 1) <= 0.02,
          "eion_mpv": abs(out["eion_keV"]["mpv_ratio_minus_1"]) <= 0.02,
          "eion_mean": abs(out["eion_keV"]["mean_ratio_minus_1"]) <= 0.03,
          "eion_band": (out["eion_band"].get("max_abs_dev_minus_2sigma") or 1) <= 0.05,
          "entry_bias": len(ia) > 0 and max(abs(x) for x in out["entry_bias_um"]) <= 10,
          "time_bias": len(ia) > 0 and abs(out["time_bias_ns"]) <= 0.05}
    out["thresholds_pass"] = th
    for k, v in out.items():
        print(f"{k:30s} {v}")
    plots.write_json(outdir, "v1compare.json", out)
