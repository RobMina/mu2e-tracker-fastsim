"""digicmp -- fast digis (trkfastsim.digi) against stock StrawDigisFromStrawGasSteps digis on the same StrawGasSteps.

Inputs: fast digi tables (`digi --events REF`), --ref stock StrawHitTableDumper tables of the same events (path A on
the same step table), optionally --floor stock tables made with another digitisation seed. Only events present in
every set are used; arms are paired event by event, so the floor measures what a different random history alone
does to each statistic.

Statistics (per arm against --ref):
- digis per event: mean and the paired relative difference with its error (std of per-event differences / sqrt N);
- histograms of unnormalised counts: panel and straw occupancy, TDC (both ends: absolute, and relative to the earliest
  cal-end TDC of the event in the same arm), TOT (both ends), PMP, all ADC samples, t_hv - t_cal, and from the fast
  hit reco (trkfastsim.reco.hitreco + combo) wire distance, energy, hits per event, panel hits per event. Three
  bin-by-bin statistics, each sum over the bins with a non-zero variance / their number:
  * deck: sum (N1 - N2)^2 / (N1 + N2), the plain Poisson deck rule;
  * deck_ev: sum (N1 - N2)^2 / (V1 + V2) with the exact event-clustered variance V = sum_events n_eb^2 (one event
    fills many bins of an occupancy or relative-time histogram; V = N when every event puts at most one entry in a bin);
  * paired: sum D^2 / sum_events d_eb^2 with d_eb = n1_eb - n2_eb, D = sum_e d_eb: the arms share their events and
    steps, so the deck statistics sit far below 1 for a correct digitiser; the paired one is ~1 and is the sensitive one.
  Each is flagged if the fast value exceeds max(2, 3 x the floor's value), or 2 without a floor;
- paired single-digi straws (one digi in the straw in both arms): fast - stock and floor - stock differences of TDC,
  TOT and PMP (quantiles), which must agree if the fast digitiser is statistically equivalent;
- with --f2: the fast reco chain (time clusters, line seeds, F2 fit; JAX) on each arm's digis: good fits and the
  paired yield ratio.
Output: <outdir>/digicmp.json and digicmp_hist.npz (the binned counts).
"""
import json
import os

import numpy as np

from ..reco import combo, hitreco

LSB_TDC = 0.01953125


def _keyid(k):
    return (k[:, 0].astype(np.int64) << 40) | (k[:, 1].astype(np.int64) << 20) | k[:, 2].astype(np.int64)


def _restrict(d, keep_ev):
    """Digi arrays restricted to the events keep_ev (indices into d['evkey']), events renumbered in that order."""
    remap = np.full(len(d["evkey"]), -1)
    remap[keep_ev] = np.arange(len(keep_ev))
    m = remap[d["ev"]] >= 0
    aidx = np.repeat(m, d["nadc"])
    out = {k: d[k][m] for k in ("sid", "tdc0", "tdc1", "tot0", "tot1", "nadc")}
    out["adc"] = d["adc"][aidx]
    out["ev"] = remap[d["ev"][m]]
    out["evkey"] = d["evkey"][keep_ev]
    out["pbt"] = d["pbt"][keep_ev]
    return out


def chi2ndf(n1, n2, var=None):
    """sum (n1 - n2)^2 / var over the bins with var > 0, / their number (var default n1 + n2)."""
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    v = n1 + n2 if var is None else np.asarray(var, float)
    nz = v > 0
    return float(((n1[nz] - n2[nz]) ** 2 / v[nz]).sum() / max(nz.sum(), 1)), int(nz.sum())


def ev_bin_counts(x, ev, bins):
    """Histogram of x with its per-(event, bin) counts: (N_b, sum_e n_eb^2, keys ev * nbin + b, n_eb)."""
    nb = len(bins) - 1
    b = np.searchsorted(bins, x, side="right") - 1
    b = np.where(x == bins[-1], nb - 1, b)                 # numpy.histogram: last bin closed
    m = (b >= 0) & (b < nb) & np.isfinite(x)
    k, n = np.unique(np.asarray(ev, np.int64)[m] * nb + b[m], return_counts=True)
    N = np.bincount(k % nb, weights=n, minlength=nb)
    V = np.bincount(k % nb, weights=n.astype(float) ** 2, minlength=nb)
    return N, V, k, n


def paired_var(h1, h2, nb):
    """sum_events (n1_eb - n2_eb)^2 per bin, from two ev_bin_counts results."""
    k = np.concatenate([h1[2], h2[2]])
    w = np.concatenate([h1[3], -h2[3]]).astype(float)
    u, inv = np.unique(k, return_inverse=True)
    d = np.bincount(inv, weights=w)
    return np.bincount(u % nb, weights=d * d, minlength=nb)


def _q(a, qs=(0.05, 0.16, 0.5, 0.84, 0.95)):
    return [float(x) for x in np.quantile(a, qs)] if len(a) else []


def observables(d, cond, settings="mc"):
    """Per-digi and per-event observables of one arm (events renumbered to the common list)."""
    ne = len(d["evkey"])
    ref_t0 = np.full(ne, np.iinfo(np.int64).max)
    np.minimum.at(ref_t0, d["ev"], d["tdc0"].astype(np.int64))
    pmp = hitreco.peak_minus_ped(d["adc"].astype(np.int64), d["nadc"], cond["scalar"]["nADCPreSamples"])
    sh = hitreco.straw_hits(d, cond, settings)
    ph = combo.panel_hits(sh, settings)
    sid = d["sid"].astype(np.int64)
    return {"sid": sid, "upanel": hitreco.unique_panel(sid), "ustraw": hitreco.unique_straw(sid),
            "tdc0": d["tdc0"] - ref_t0[d["ev"]], "tdc1": d["tdc1"] - ref_t0[d["ev"]], "tot0": d["tot0"], "tot1": d["tot1"],
            "tdc0_abs": d["tdc0"], "tdc1_abs": d["tdc1"], "adc": d["adc"], "adc_ev": np.repeat(d["ev"], d["nadc"]),
            "pmp": pmp, "dt": (d["tdc1"] - d["tdc0"]) * LSB_TDC, "wdist": sh["wdist"], "edep": sh["edep"],
            "ndigi": np.bincount(d["ev"], minlength=ne), "nsh": np.bincount(sh["ev"], minlength=ne),
            "nph": np.bincount(ph["ev"], minlength=ne) if len(ph["ev"]) else np.zeros(ne, np.int64), "ev": d["ev"],
            "sh": sh}


HIST = {  # name: (key, bins); ADC samples, wdist, edep use their own entry arrays
    "panel_occupancy": ("upanel", np.arange(217) - 0.5),
    "straw_occupancy": ("ustraw", np.arange(20737) - 0.5),
    "tdc0_rel_ns": ("tdc0", np.linspace(-20, 200, 111)), "tdc1_rel_ns": ("tdc1", np.linspace(-20, 200, 111)),
    "tdc0_abs_ns": ("tdc0_abs", np.linspace(0, 100000, 201)), "tdc1_abs_ns": ("tdc1_abs", np.linspace(0, 100000, 201)),
    "tot0": ("tot0", np.arange(17) - 0.5), "tot1": ("tot1", np.arange(17) - 0.5),
    "pmp_adc": ("pmp", np.arange(0, 801, 10)),
    "adc_sample": ("adc", np.arange(0, 1025, 8)),
    "dt_hv_minus_cal_ns": ("dt", np.linspace(-6, 6, 121)),
    "wdist_mm": ("wdist", np.linspace(-700, 700, 141)),
    "edep_keV": ("edep", np.linspace(0, 12, 121)),
    "digis_per_event": ("ndigi", np.arange(0, 81) - 0.5),
    "strawhits_per_event": ("nsh", np.arange(0, 81) - 0.5),
    "panelhits_per_event": ("nph", np.arange(0, 41) - 0.5),
}
PER_EVENT = {"ndigi", "nsh", "nph"}


def _hist(o, name):
    """ev_bin_counts of one HIST entry for an arm's observables."""
    key, bins = HIST[name]
    x = np.asarray(o[key], float)
    if key in PER_EVENT:
        ev = np.arange(len(x))
    elif key == "adc":
        ev = o["adc_ev"]
    else:
        ev = o["ev"]
    if key.startswith("tdc"):
        x = x * LSB_TDC
    if key == "edep":
        x = x * 1000.0
    return ev_bin_counts(x, ev, bins)


def paired_single(o_arm, o_ref):
    """Straws with exactly one digi in both arms: per-straw differences arm - ref."""
    def single(o):
        k = o["ev"].astype(np.int64) * 65536 + o["sid"]
        u, i, c = np.unique(k, return_index=True, return_counts=True)
        return u[c == 1], i[c == 1]
    ua, ia = single(o_arm)
    ur, ir = single(o_ref)
    both, xa, xr = np.intersect1d(ua, ur, return_indices=True)
    a, r = ia[xa], ir[xr]
    out = {"n": int(len(both))}
    for k, s in (("tdc0", LSB_TDC), ("tdc1", LSB_TDC), ("tot0", 1), ("tot1", 1), ("pmp", 1)):
        out[k] = _q((o_arm[k][a] - o_ref[k][r]) * s)
    return out


def f2_yield(d, cond, drift_ref, settings="mc", niter=4):
    """Fast reco chain on the digis (hit reco, SimpleTimeCluster + LineFinder ports, F2 fit): good fits per event."""
    from ..reco import linefind, linefind_jax, linefit, linefit_jax
    from ..reco.cli import _drift_table
    sh = hitreco.straw_hits(d, cond, settings)
    tc = linefind.time_clusters(sh)
    sd = linefind_jax.line_seeds(sh, tc, cond)
    ns = sd["n"]
    trk = np.repeat(np.arange(len(ns)), ns)
    H = linefit.build_hits(trk, sd["member"], np.zeros(len(trk)), np.ones(len(trk), bool), sh, cond, len(ns))
    x0 = linefit.seed_params(sd["a0"], sd["b0"], sd["a1"], sd["b1"], sd["t0"])
    f = linefit_jax.fit_own_states(x0, H, _drift_table(drift_ref), niter=niter)
    nact = f["active"].sum(1)
    good = (nact >= 5) & np.isfinite(f["chi2"]) & (f["ndof"] >= 1)
    fpe = np.bincount(sd["ev"][good], minlength=len(d["evkey"]))
    return {"seeds": int(len(ns)), "good_fits": int(good.sum()), "events_with_fit": int((fpe > 0).sum()),
            "fits_per_event": fpe,
            "nactive_q": _q(nact[good]), "chi2ndf_q": _q(f["chi2"][good] / np.maximum(f["ndof"][good], 1))}


def run(inputs, outdir, ref=None, floor=None, hit_cond=None, f2=False, **kw):
    os.makedirs(outdir, exist_ok=True)
    if hit_cond is None:
        hit_cond = os.path.join(outdir, "hitcond.json")
        hitreco.extract_conditions(ref[0], hit_cond)
    cond = hitreco.load_conditions(hit_cond)
    arms = {"fast": hitreco.read_digis(inputs), "ref": hitreco.read_digis(ref)}
    if floor:
        arms["floor"] = hitreco.read_digis(floor)
    common = None
    for d in arms.values():
        k = _keyid(d["evkey"])
        common = k if common is None else np.intersect1d(common, k)
    for name, d in list(arms.items()):
        k = _keyid(d["evkey"])
        order = np.argsort(k)
        keep = order[np.searchsorted(k[order], common)]
        arms[name] = _restrict(d, keep)
    ne = len(common)
    obs = {n: observables(d, cond) for n, d in arms.items()}
    res = {"events": int(ne), "digis": {n: int(len(o["sid"])) for n, o in obs.items()}}
    for n in obs:
        if n == "ref":
            continue
        for key, lab in (("ndigi", "digis_per_event"), ("nsh", "strawhits_per_event"), ("nph", "panelhits_per_event")):
            a, b = obs[n][key].astype(float), obs["ref"][key].astype(float)
            dlt = a - b
            res.setdefault(lab, {})[n] = {"mean": float(a.mean()), "ref_mean": float(b.mean()),
                                         "rel_diff": float(dlt.mean() / b.mean()),
                                         "rel_err": float(dlt.std(ddof=1) / np.sqrt(ne) / b.mean())}
    hists, table = {}, {}
    for h in HIST:
        nb = len(HIST[h][1]) - 1
        hr = _hist(obs["ref"], h)
        hists[f"{h}/ref"] = hr[0]
        row = {}
        for n in obs:
            if n == "ref":
                continue
            hn = _hist(obs[n], h)
            hists[f"{h}/{n}"] = hn[0]
            row[n] = {"deck": chi2ndf(hn[0], hr[0]), "deck_ev": chi2ndf(hn[0], hr[0], hn[1] + hr[1]),
                      "paired": chi2ndf(hn[0], hr[0], paired_var(hn, hr, nb))}
        for stat in ("deck", "deck_ev", "paired"):
            fl = row["floor"][stat][0] if "floor" in row else None
            lim = max(2.0, 3.0 * fl) if fl is not None else 2.0
            row.setdefault("limit", {})[stat] = lim
            row.setdefault("flag", {})[stat] = bool(row["fast"][stat][0] > lim)
        table[h] = row
    res["chi2ndf"] = table
    res["paired_single_digi_straws"] = {n: paired_single(obs[n], obs["ref"]) for n in obs if n != "ref"}
    for n, o in obs.items():
        res.setdefault("quantiles", {})[n] = {k: _q(np.asarray(o[k], float)) for k in ("tot0", "tot1", "pmp", "dt", "wdist")}
    if f2:
        yl = {n: f2_yield(arms[n], cond, ref[0]) for n in arms}
        b = yl["ref"]["fits_per_event"].astype(float)
        res["f2"] = {}
        for n, y in yl.items():
            a = y.pop("fits_per_event").astype(float)
            y["paired_rel_diff"] = float((a - b).mean() / max(b.mean(), 1e-12))
            y["paired_rel_err"] = float((a - b).std(ddof=1) / np.sqrt(ne) / max(b.mean(), 1e-12))
            ea, eb = (a > 0).astype(float), (b > 0).astype(float)
            y["events_with_fit_rel_diff"] = float((ea - eb).mean() / max(eb.mean(), 1e-12))
            y["events_with_fit_rel_err"] = float((ea - eb).std(ddof=1) / np.sqrt(ne) / max(eb.mean(), 1e-12))
            res["f2"][n] = y
    with open(os.path.join(outdir, "digicmp.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(outdir, "digicmp_hist.npz"), **{k.replace("/", "__"): v for k, v in hists.items()})
    print(summary(res))
    return res


def summary(res):
    """Compact text table of a digicmp result."""
    arms = [n for n in res["digis"] if n != "ref"]
    out = [f"events {res['events']}, digis " + ", ".join(f"{n} {v}" for n, v in res["digis"].items())]
    for lab in ("digis_per_event", "strawhits_per_event", "panelhits_per_event"):
        out.append(lab + ": " + "; ".join(f"{n} {r['mean']:.4f} vs ref {r['ref_mean']:.4f}, paired rel diff "
                                         f"{100 * r['rel_diff']:+.3f} +- {100 * r['rel_err']:.3f} %"
                                         for n, r in res[lab].items()))
    out.append(f"{'histogram':22s} " + " ".join(f"{n + ' ' + st:>17s}" for st in ("deck", "deck_ev", "paired")
                                                for n in arms) + "  flags(deck, deck_ev, paired)")
    for h, row in res["chi2ndf"].items():
        vals = " ".join(f"{row[n][st][0]:11.3f}/{row[n][st][1]:<5d}" for st in ("deck", "deck_ev", "paired") for n in arms)
        out.append(f"{h:22s} {vals}  {[row['flag'][st] for st in ('deck', 'deck_ev', 'paired')]}")
    for n, r in res.get("f2", {}).items():
        out.append(f"F2 {n}: seeds {r['seeds']}, good fits {r['good_fits']} (paired rel diff vs ref "
                   f"{100 * r['paired_rel_diff']:+.3f} +- {100 * r['paired_rel_err']:.3f} %), events with a fit "
                   f"{r['events_with_fit']} ({100 * r['events_with_fit_rel_diff']:+.3f} +- "
                   f"{100 * r['events_with_fit_rel_err']:.3f} %)")
    return "\n".join(out)
