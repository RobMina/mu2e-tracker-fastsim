"""CE tracker comparison through stock reco: track efficiency and momentum response, fast arm against Geant4, paired by event.

Both arms are the same generated conversion electrons (the dts events); each arm's tracker steps are injected into
those events and run through the same stock digitisation, LoopHelix reconstruction and EventNtuple. Per event:

- the first track with fit status > 0 ("fit"), and whether it passes the CE selection ("ce"): electron fit,
  downstream at the tracker front (in-bounds TT_Front segment, pz > 0), fit consistency > 1e-3, and the fiducial cuts
  of the CE selection from the LoopHelix parameters at the front: tan(dip) in [0.5774, 1], d0 in [-80, 105] mm,
  r_max in [450, 680] mm (CE_CUTS). No t0 window: OffSpill digitisation does not fold the CE times into the
  microbunch, so the on-spill window would cut on the muon decay time, not on the tracker;
- reco momentum at TT_Front; truth = |p| of the arm's own primary at the tracker entrance (VD13: the hand-off state
  in the fast arm, the Geant4 crossing in the reference arm);
- active hits of the track (trkhits with KinKal state > -2, and the persisted trk.nactive) and its chi2/ndof.

Denominator: every dts event (no track counts as a failure). Errors: paired per-event differences (efficiency, tail
fractions, per-bin fractions, mean) and shared-index bootstrap (Gaussian core of the residual, fitted in -1..+1 MeV/c).

  python3 -m trkfastsim.validation.cemom --gen states.root [...] --truth fast=states.root[,...] --truth g4=g4entries.root[,...] \
      --nts fast=DIR --nts g4=DIR --ref g4 --outdir OUT [--plots]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables
from trkfastsim.validation.fieldreco import KEY, TREE, _chi2, _me, _vec

CE_CUTS = dict(tandip=(0.5774, 1.0), d0=(-80.0, 105.0), rmax=(450.0, 680.0), fitcon=1e-3)
FRONT_SID = 0
ACTIVE_STATE_MIN = -2         # KinKal hit state > -2: active (both drift sides and null-ambiguity)
RES_EDGES = np.round(np.arange(-3.0, 2.0 + 1e-9, 0.1), 3)
P_EDGES = np.round(np.arange(100.0, 106.0 + 1e-9, 0.1), 3)
TAILS = (-2.0, -1.0, 1.0, 2.0)


def _first(arr, idx):
    return ak.firsts(arr[idx], axis=1)


def read_ntuple(path):
    """Per ntuple entry: KEY, fit and CE-selection flags of the first fitted track, its front momentum and quantities."""
    with uproot.open(path) as f:
        t = f[TREE]
        a = t.arrays([f"evtinfo/{k}" for k in KEY] + [f"trk/trk.{k}" for k in
                     ("status", "pdg", "nactive", "chisq", "ndof", "fitcon")], library="ak")
        seg = t["trksegs"].array()
        pars = t["trksegpars_lh"].array()
        hits = t["trkhits"].array()
    good = a["trk/trk.status"] > 0
    has = ak.to_numpy(ak.any(good, axis=1))
    i = ak.argmax(good, axis=1, keepdims=True)
    s = ak.fill_none(ak.firsts(seg[i], axis=1), [], axis=0)
    pp = ak.fill_none(ak.firsts(pars[i], axis=1), [], axis=0)
    front = (s.sid == FRONT_SID) & s.inbounds
    j = ak.argmin(ak.where(front, s.time, np.inf), axis=1, keepdims=True)
    s0 = ak.firsts(s[j])
    p0 = ak.firsts(pp[j])
    infront = ak.to_numpy(ak.fill_none(ak.firsts(front[j]), False)) & has
    nan3 = {"fCoordinates": {"fX": np.nan, "fY": np.nan, "fZ": np.nan}}
    mom = _vec(ak.fill_none(s0.mom, nan3))
    df = pd.DataFrame({k: ak.to_numpy(a[f"evtinfo/{k}"]).astype(np.uint64) for k in KEY})
    df["fit"] = has
    df["p_reco"] = np.where(infront, np.linalg.norm(mom, axis=1), np.nan)
    df["pz"] = np.where(infront, mom[:, 2], np.nan)
    df["t0"] = np.where(infront, ak.to_numpy(ak.fill_none(s0.time, np.nan)), np.nan)
    for k, src in (("tandip", "tanDip"), ("d0", "d0"), ("rmax", "maxr")):
        df[k] = np.where(infront, ak.to_numpy(ak.fill_none(p0[src], np.nan)).astype(float), np.nan)
    for k in ("pdg", "nactive", "chisq", "ndof", "fitcon"):
        df[k] = np.where(has, ak.to_numpy(ak.fill_none(_first(a[f"trk/trk.{k}"], i), np.nan)).astype(float), np.nan)
    df["chi2ndf"] = df.chisq / df.ndof
    h = ak.fill_none(ak.firsts(hits[i], axis=1), [], axis=0)
    df["nactive_hits"] = np.where(has, ak.to_numpy(ak.sum(h.state > ACTIVE_STATE_MIN, axis=1)), np.nan)
    if df.duplicated(KEY).any():
        raise ValueError(f"duplicate events in {path}")
    return df


def ce_selected(df, cuts=CE_CUTS):
    c = df.fit & (df.pdg == 11) & (df.pz > 0) & (df.fitcon > cuts["fitcon"])
    for k in ("tandip", "d0", "rmax"):
        lo, hi = cuts[k]
        c &= (df[k] >= lo) & (df[k] <= hi)
    return c.fillna(False).to_numpy(bool)


def read_arm(directory):
    files = sorted(glob.glob(os.path.join(directory, "**", "nts.*.root"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no nts.*.root under {directory}")
    d = pd.concat([read_ntuple(f) for f in files], ignore_index=True)
    d["ce"] = ce_selected(d)
    return d, len(files)


def read_truth(paths, tree):
    t = tables.read_many(paths, tree, KEY + ["px", "py", "pz"])
    for k in KEY:
        t[k] = t[k].astype(np.uint64)
    return t[KEY].assign(p_true=np.sqrt(t.px ** 2 + t.py ** 2 + t.pz ** 2))


def assemble(den, arms, truths):
    d = den[KEY].copy()
    for name, frame in arms.items():
        unknown = len(frame) - len(frame.merge(den[KEY], on=KEY))
        if unknown:
            raise ValueError(f"{unknown} {name} events outside the denominator")
        f = den[KEY].merge(truths[name], on=KEY, how="left", validate="one_to_one")
        f = f.merge(frame, on=KEY, how="left", validate="one_to_one")
        d = d.merge(f.rename(columns={c: f"{name}_{c}" for c in f if c not in KEY}), on=KEY, how="left",
                    validate="one_to_one")
        d[f"{name}_res"] = d[f"{name}_p_reco"] - d[f"{name}_p_true"]
        for c in ("fit", "ce"):
            d[f"{name}_{c}"] = d[f"{name}_{c}"].fillna(False).astype(bool)
    return d


def gauss_core(x, lo=-1.0, hi=1.0, width=0.05):
    """Mean and sigma of a Gaussian fitted (Poisson likelihood) to the histogram of x in [lo, hi]."""
    from scipy.optimize import minimize
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x >= lo) & (x < hi)]
    if len(x) < 20:
        return np.array([np.nan, np.nan])
    e = np.arange(lo, hi + 1e-9, width)
    n, _ = np.histogram(x, e)
    c = 0.5 * (e[1:] + e[:-1])
    from scipy.special import erf

    def nll(q):
        m, s = q
        s = abs(s)
        cdf = 0.5 * (1 + erf((e - m) / (np.sqrt(2) * s)))
        mu = np.maximum(np.diff(cdf) / max(cdf[-1] - cdf[0], 1e-12) * n.sum(), 1e-12)
        return float((mu - n * np.log(mu)).sum())
    q0 = [float(np.median(x)), float(np.std(x))]
    r = minimize(nll, q0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=2000))
    return np.array([r.x[0], abs(r.x[1])]) if r.success else np.array([np.nan, np.nan])


def paired_core(a, b, draws=200, seed=20261105):
    idx = np.random.default_rng(seed).integers(len(a), size=(draws, len(a)))
    ca, cb = gauss_core(a), gauss_core(b)
    diff = np.array([gauss_core(a[k]) - gauss_core(b[k]) for k in idx])
    err = np.nanstd(diff, axis=0, ddof=1)
    return dict(events=int(len(a)), candidate=ca.tolist(), reference=cb.tolist(), difference=(ca - cb).tolist(),
                paired_error=err.tolist(), metrics=["mean_MeV", "sigma_MeV"], fit_range_MeV=[-1.0, 1.0],
                bootstrap_draws=draws)


def binned_pulls(a, b, edges):
    """Per-bin paired difference of the fractions of events in the bin (a, b on the same events) and its pull."""
    ia = np.digitize(a, edges) - 1
    ib = np.digitize(b, edges) - 1
    nb = len(edges) - 1
    oa = (ia[:, None] == np.arange(nb)[None]).astype(float)
    ob = (ib[:, None] == np.arange(nb)[None]).astype(float)
    dd = oa - ob
    m = dd.mean(0)
    se = dd.std(0, ddof=1) / np.sqrt(len(dd))
    pull = np.where(se > 0, m / np.where(se > 0, se, 1), 0.0)
    ok = se > 0
    return dict(edges=edges.tolist(), frac_candidate=oa.mean(0).tolist(), frac_reference=ob.mean(0).tolist(),
                diff=m.tolist(), err=se.tolist(), pull=pull.tolist(),
                chi2_paired=float((pull[ok] ** 2).sum()), ndf=int(ok.sum()))


def compare(d, arm, ref):
    out = {}
    for sel in ("fit", "ce"):
        ia, ir = d[f"{arm}_{sel}"].to_numpy(bool), d[f"{ref}_{sel}"].to_numpy(bool)
        out[f"efficiency_{sel}"] = {arm: _me(ia), ref: _me(ir), "paired_difference": _me(ia.astype(float) - ir),
                                    "ratio": float(ia.sum() / max(ir.sum(), 1)),
                                    "gained": int((ia & ~ir).sum()), "lost": int((~ia & ir).sum())}
    both = d[f"{arm}_ce"].to_numpy(bool) & d[f"{ref}_ce"].to_numpy(bool)
    ra, rr = d.loc[both, f"{arm}_res"].to_numpy(), d.loc[both, f"{ref}_res"].to_numpy()
    out["residual_core"] = paired_core(ra, rr)
    out["residual_mean"] = {arm: _me(ra), ref: _me(rr), "paired_difference": _me(ra - rr)}
    out["residual_median"] = {arm: float(np.median(ra)), ref: float(np.median(rr))}
    out["tails"] = {}
    for c in TAILS:
        ta, tr = (ra < c, rr < c) if c < 0 else (ra > c, rr > c)
        out["tails"][f"{'<' if c < 0 else '>'}{c:+g}"] = {arm: _me(ta), ref: _me(tr),
                                                          "paired_difference": _me(ta.astype(float) - tr)}
    out["residual_shape"] = binned_pulls(ra, rr, RES_EDGES)
    out["residual_chi2_unpaired"] = _chi2(ra, rr, RES_EDGES)
    pa, pr = d.loc[both, f"{arm}_p_reco"].to_numpy(), d.loc[both, f"{ref}_p_reco"].to_numpy()
    out["p_reco_shape"] = binned_pulls(pa, pr, P_EDGES)
    out["p_reco_chi2_unpaired"] = _chi2(pa, pr, P_EDGES)
    for q in ("nactive", "nactive_hits", "chi2ndf"):
        qa, qr = d.loc[both, f"{arm}_{q}"].to_numpy(float), d.loc[both, f"{ref}_{q}"].to_numpy(float)
        edges = np.linspace(0, 5, 41) if q == "chi2ndf" else np.arange(0, 81, 2.0)
        out[q] = {arm: _me(qa), ref: _me(qr), "paired_difference": _me(qa - qr), "shape": _chi2(qa, qr, edges)}
    out["nactive_hits_equal_nactive"] = float(np.mean(d.loc[both, f"{arm}_nactive_hits"] == d.loc[both, f"{arm}_nactive"]))
    out["p_true_paired_difference"] = _me(d.loc[both, f"{arm}_p_true"] - d.loc[both, f"{ref}_p_true"])
    return out


def plots(d, rep, arm, ref, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {ref: "#0072B2", arm: "#D55E00"}
    lab = {ref: "Geant4", arm: "fast"}
    c = rep["compare"][arm]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex="col", gridspec_kw=dict(height_ratios=[3, 1]))
    for k, (key, xl, sh) in enumerate([("p_reco", r"reco $p$ at tracker front [MeV/$c$]", "p_reco_shape"),
                                       ("res", r"$p_{reco}-p_{true}$ at tracker entrance [MeV/$c$]",
                                        "residual_shape")]):
        s = c[sh]
        e = np.array(s["edges"])
        n = c["residual_core"]["events"]
        for name, fr in ((ref, "frac_reference"), (arm, "frac_candidate")):
            ax[0, k].stairs(np.array(s[fr]) * n, e, color=col[name], lw=1.6, label=lab[name])
        ax[0, k].set_ylabel("tracks / 0.1 MeV/c")
        ax[0, k].set_yscale("log")
        ax[0, k].legend(title=f"{n} paired CE tracks", frameon=False)
        cc = 0.5 * (e[1:] + e[:-1])
        ax[1, k].errorbar(cc, s["pull"], yerr=1.0, fmt="o", ms=2.5, color="k", lw=0.6)
        ax[1, k].axhline(0, color="0.5", lw=0.8)
        ax[1, k].set_ylim(-4.5, 4.5)
        ax[1, k].set_ylabel("paired pull")
        ax[1, k].set_xlabel(xl)
        ax[1, k].text(0.02, 0.85, f"paired $\\chi^2$/ndf = {s['chi2_paired']:.1f}/{s['ndf']}",
                      transform=ax[1, k].transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "ce_mom.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    _eff_panels(ax, d, c, arm, ref, col, lab)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "ce_eff.png", dpi=150)
    plt.close(fig)
    deck_figure(d, rep, arm, ref, outdir)


def _eff_panels(ax, d, c, arm, ref, col, lab):
    """CE-selection efficiency per generated event against the Geant4 momentum at the tracker entrance; paired
    fast - G4 difference per bin (pp)."""
    pt = d[f"{ref}_p_true"].to_numpy(float)
    e = np.arange(95.0, 106.01, 1.0)
    ib = np.digitize(np.nan_to_num(pt, nan=-1), e) - 1
    inb = (ib >= 0) & (ib < len(e) - 1)
    cc = 0.5 * (e[1:] + e[:-1])
    nn = np.bincount(ib[inb], minlength=len(cc))
    sel = {}
    for name in (ref, arm):
        sel[name] = d[f"{name}_ce"].to_numpy(bool)
        k = np.bincount(ib[inb], weights=sel[name][inb], minlength=len(cc))
        eff = np.where(nn > 0, k / np.maximum(nn, 1), np.nan)
        ax[0].errorbar(cc, eff, yerr=np.sqrt(eff * (1 - eff) / np.maximum(nn, 1)), fmt="o", color=col[name],
                       label=lab[name], ms=4)
    dd = sel[arm].astype(float) - sel[ref]
    m = np.bincount(ib[inb], weights=dd[inb], minlength=len(cc)) / np.maximum(nn, 1)
    m2 = np.bincount(ib[inb], weights=dd[inb] ** 2, minlength=len(cc)) / np.maximum(nn, 1)
    se = np.sqrt(np.maximum(m2 - m ** 2, 0) / np.maximum(nn - 1, 1))
    ok = nn > 1
    ax[1].errorbar(cc[ok], 100 * m[ok], yerr=100 * se[ok], fmt="o", color="k", ms=3)
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_ylabel("fast $-$ G4 [pp]")
    ax[1].set_xlabel(r"Geant4 $p$ at tracker entrance [MeV/$c$]")
    ef = c["efficiency_ce"]
    ax[0].set_ylabel("CE selection efficiency")
    ax[0].set_ylim(0, 1)
    ax[0].legend(frameon=False, loc="lower left",
                 title=f"all: {100 * ef[arm][0]:.1f} vs {100 * ef[ref][0]:.1f} %, paired "
                       f"{100 * ef['paired_difference'][0]:+.2f} $\\pm$ {100 * ef['paired_difference'][1]:.2f} pp")


def deck_figure(d, rep, arm, ref, outdir, name="ce_deck.png"):
    """Left: p_reco - p_true at the tracker entrance, fast vs Geant4, with the fast/G4 ratio (paired errors); right:
    CE-selection efficiency against the Geant4 entrance momentum with the paired difference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {ref: "#0072B2", arm: "#D55E00"}
    lab = {ref: "Geant4", arm: "fast"}
    c = rep["compare"][arm]
    s = c["residual_shape"]
    e = np.array(s["edges"])
    cc = 0.5 * (e[1:] + e[:-1])
    n = c["residual_core"]["events"]
    fr, fc = np.array(s["frac_reference"]), np.array(s["frac_candidate"])
    fig, ax = plt.subplots(2, 2, figsize=(12, 5.8), gridspec_kw=dict(height_ratios=[3, 1]), sharex="col")
    for arm_name, f in ((ref, fr), (arm, fc)):
        ax[0, 0].stairs(f * n, e, color=col[arm_name], lw=1.6, label=lab[arm_name])
    core = c["residual_core"]
    sig = core["candidate"][1], core["reference"][1]
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel(r"tracks / 0.1 MeV/$c$")
    ax[0, 0].legend(frameon=False, title=f"{n} events, CE-selected in both\n"
                                          f"core $\\sigma$ {sig[0]:.3f} vs {sig[1]:.3f} MeV/$c$")
    ok = fr > 0
    ratio = np.where(ok, fc / np.where(ok, fr, 1), np.nan)
    err = np.where(ok, np.array(s["err"]) / np.where(ok, fr, 1), np.nan)
    ax[1, 0].errorbar(cc, ratio, yerr=err, fmt="o", ms=2.5, color="k", lw=0.6)
    ax[1, 0].axhline(1, color="0.5", lw=0.8)
    ax[1, 0].set_ylim(0.5, 1.5)
    ax[1, 0].set_ylabel("fast / G4")
    ax[1, 0].set_xlabel(r"$p_{reco} - p_{true}$ at tracker entrance [MeV/$c$]")
    ax[1, 0].text(0.02, 0.8, f"paired $\\chi^2$/ndf = {s['chi2_paired']:.0f}/{s['ndf']}", transform=ax[1, 0].transAxes,
                  fontsize=8)
    _eff_panels(ax[:, 1], d, c, arm, ref, col, lab)
    fig.tight_layout()
    fig.savefig(Path(outdir) / name, dpi=150)
    plt.close(fig)


def headline(rep, arm, ref="g4"):
    """Deck numbers: value, reference and paired difference with its error for each headline metric."""
    c = rep["compare"][arm]

    def pde(x, scale=1.0):
        return dict(fast=round(scale * x[arm][0], 5), g4=round(scale * x[ref][0], 5),
                    diff=round(scale * x["paired_difference"][0], 5), err=round(scale * x["paired_difference"][1], 5))
    core = c["residual_core"]
    out = dict(events=rep["events"], paired_tracks=core["events"],
               eff_fit_pct=pde(c["efficiency_fit"], 100), eff_ce_pct=pde(c["efficiency_ce"], 100),
               core_sigma_MeV=dict(fast=round(core["candidate"][1], 5), g4=round(core["reference"][1], 5),
                                   diff=round(core["difference"][1], 5), err=round(core["paired_error"][1], 5)),
               core_mean_MeV=dict(fast=round(core["candidate"][0], 5), g4=round(core["reference"][0], 5),
                                  diff=round(core["difference"][0], 5), err=round(core["paired_error"][0], 5)),
               mean_residual_MeV=pde(c["residual_mean"]),
               tails_pct={k: pde(v, 100) for k, v in c["tails"].items()},
               active_hits=pde(c["nactive_hits"]), chi2ndf=pde(c["chi2ndf"]),
               residual_shape_paired_chi2_ndf=[round(c["residual_shape"]["chi2_paired"], 1), c["residual_shape"]["ndf"]],
               p_reco_shape_paired_chi2_ndf=[round(c["p_reco_shape"]["chi2_paired"], 1), c["p_reco_shape"]["ndf"]])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m trkfastsim.validation.cemom", description=__doc__.split("\n")[0])
    ap.add_argument("--gen", nargs="+", required=True, help="states.root files (dump/gen: one row per dts event)")
    ap.add_argument("--truth", action="append", required=True, help="NAME=ROOT[,ROOT...] with dump/entries (VD13)")
    ap.add_argument("--nts", action="append", required=True, help="NAME=DIR with nts.*.root (recursive)")
    ap.add_argument("--ref", default="g4", help="reference arm; with a single arm only its efficiencies are reported")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args(argv)
    den = tables.read_many(a.gen, "gen", KEY)
    for k in KEY:
        den[k] = den[k].astype(np.uint64)
    if den.duplicated(KEY).any():
        raise ValueError("duplicate events in the denominator")
    truths = {s.split("=", 1)[0]: read_truth(s.split("=", 1)[1].split(","), "entries") for s in a.truth}
    arms, nfiles = {}, {}
    for s in a.nts:
        name, path = s.split("=", 1)
        arms[name], nfiles[name] = read_arm(path)
    if set(arms) != set(truths) or a.ref not in arms:
        ap.error("--truth and --nts need the same arm names, including --ref")
    d = assemble(den, arms, truths)
    rep = dict(events=len(den), ntuple_files=nfiles, ntuple_entries={n: len(f) for n, f in arms.items()},
               ce_cuts={k: list(v) if isinstance(v, tuple) else v for k, v in CE_CUTS.items()},
               truth="|p| of the arm's own primary at VD13 (tracker entrance); reco at the first in-bounds TT_Front",
               arm_efficiency={n: {sel: _me(d[f"{n}_{sel}"].to_numpy(bool)) for sel in ("fit", "ce")} for n in arms},
               compare={n: compare(d, n, a.ref) for n in arms if n != a.ref})
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cemom.json").write_text(json.dumps(rep, indent=1) + "\n")
    d.to_csv(out / "events.csv.gz", index=False)
    (out / "numbers.json").write_text(json.dumps({n: headline(rep, n, a.ref) for n in rep["compare"]}, indent=1) + "\n")
    if a.plots:
        for n in rep["compare"]:
            plots(d, rep, n, a.ref, out)
    c = rep["compare"]
    print(json.dumps(dict(arm_efficiency=rep["arm_efficiency"],
                          **{n: {k: c[n][k] for k in ("efficiency_fit", "efficiency_ce", "residual_core",
                                                       "residual_mean")} for n in c}), indent=1))


if __name__ == "__main__":
    main()
