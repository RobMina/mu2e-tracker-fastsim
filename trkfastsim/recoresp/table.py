"""Reco response tables of the conversion-electron track reconstruction (stock LoopHelix + CE selection), NumPy.

Built from a reference arm whose reconstruction ran (stock digitisation + LoopHelix on Geant4 tracker steps), per
generated event: the electron's true state at the tracker entry, the number of StrawGasStep rows it left (one row per
passage through a straw) and the first fitted track. The response is binned in the entry truth that drives it:

  * the number of step rows `nrow` (how much of the tracker the electron crossed) and
  * the true tan(dip) at the entry;

and holds per bin:

  * eff_fit: P(a fitted track, status > 0);
  * eff_q: P(fit quality and direction pass | fit): electron hypothesis, downstream at the tracker front (pz > 0,
    in-bounds front segment), fit consistency above the CE cut;
  * inverse CDFs (on a fixed probability grid, denser in both tails) of the residuals reco - truth of the momentum at
    the front, tan(dip), d0 and r_max, and of the fit-quality outputs (fit consistency, chi2/ndf, active hits), over
    the quality-passing tracks of the bin;
  * the correlation of those seven quantities as a Gaussian copula: the Cholesky factor of the correlation matrix of
    their normal scores (rank -> standard normal), so a draw keeps e.g. the tan(dip)-d0-r_max correlations that the
    fiducial cuts see.

A fast track is then: fit with eff_fit, quality with eff_q, correlated uniforms from the copula, residuals by inverse
CDF of those uniforms, added to its own entry
truth, and the CE fiducial cuts applied to the smeared tan(dip), d0 and r_max (device.py). Bins with fewer than
`min_eff` reference events (efficiencies) or `min_count` quality-passing tracks (distributions) widen to the +-1,
then +-2 neighbouring tan(dip) bins, then the whole nrow row, then everything.

Truth helix at the entry: uniform field bz along the tracker axis; circle radius R = pt / (K |bz|), centre on the side
the charge bends to; r_centre from the tracker axis; d0 = |r_centre - R|, r_max = r_centre + R (the EventNtuple
LoopHelix d0, which is unsigned, and maxr).
"""
from pathlib import Path

import numpy as np

KB = 0.299792458
NROW_EDGES = (0, 10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 60, 80, 120, 1_000_000)
TANDIP_EDGES = (-1e9, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1e9)
NQ = 129                                     # inverse-CDF nodes
RES = ("dp", "dtandip", "dd0", "drmax")      # residuals reco - truth
QUAL = ("fitcon", "chi2ndf", "nactive")      # fit-quality outputs
CE_CUTS = dict(tandip=(0.5774, 1.0), d0=(-80.0, 105.0), rmax=(450.0, 680.0), fitcon=1e-3)


def qnodes(nq=NQ):
    """Probabilities of the inverse-CDF nodes: 0.5 (1 - cos(pi k / (nq - 1))), denser in both tails."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(nq) / (nq - 1)))


def entry_truth(x, p, axis_xy, bz, q=-1.0):
    """(|p|, tan(dip), d0, r_max) of the helix through the entry states x [n, 3] (Mu2e frame, mm), p [n, 3] (MeV/c)."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    pt = np.hypot(p[:, 0], p[:, 1])
    R = pt / (KB * abs(bz))
    s = np.sign(q * bz)
    ptn = np.maximum(pt, 1e-12)
    cx = x[:, 0] - axis_xy[0] + s * R * p[:, 1] / ptn
    cy = x[:, 1] - axis_xy[1] - s * R * p[:, 0] / ptn
    rc = np.hypot(cx, cy)
    return np.linalg.norm(p, axis=1), p[:, 2] / ptn, np.abs(rc - R), rc + R


def _icdf(v, nq=NQ):
    """Inverse CDF of the sample v at qnodes (linear between the order statistics at (i - 0.5) / m, clamped)."""
    v = np.sort(np.asarray(v, float))
    m = len(v)
    pp = (np.arange(m) + 0.5) / m
    return np.interp(qnodes(nq), pp, v)


def _normal_scores(v):
    from scipy.special import ndtri
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return ndtri((r + 0.5) / len(v))


def _copula(rows):
    """Cholesky factor of the correlation of the normal scores of the columns of rows [m, K] (ridge 1e-6)."""
    K = rows.shape[1]
    if len(rows) < 3:
        return np.eye(K)
    z = np.stack([_normal_scores(rows[:, k]) for k in range(K)], 1)
    c = np.corrcoef(z, rowvar=False)
    c = np.where(np.isfinite(c), c, 0.0)
    np.fill_diagonal(c, 1.0)
    return np.linalg.cholesky(c + 1e-6 * np.eye(K))


def build(df, nrow_edges=NROW_EDGES, tandip_edges=TANDIP_EDGES, min_count=40, min_eff=10, axis_xy=(-3904.0, 0.0),
          bz=1.0):
    """Response tables from the reference events df (one row per generated event): has_entry, nrow, p_true, tandip_t,
    d0_t, rmax_t (entry truth), fit (bool), and for fitted tracks p_reco, pz, tandip, d0, rmax, pdg, fitcon, chi2ndf,
    nactive. Returns a dict of NumPy arrays (save with np.savez)."""
    ne, nt = len(nrow_edges) - 1, len(tandip_edges) - 1
    d = df[df.has_entry.astype(bool)]
    bn = np.clip(np.digitize(d.nrow.to_numpy(float), nrow_edges) - 1, 0, ne - 1)
    bt = np.clip(np.digitize(d.tandip_t.to_numpy(float), tandip_edges) - 1, 0, nt - 1)
    fit = d.fit.to_numpy(bool)
    qual = fit & np.isfinite(d.p_reco.to_numpy(float)) & (d.pdg.to_numpy(float) == 11) & \
        (d.pz.to_numpy(float) > 0) & (d.fitcon.to_numpy(float) > CE_CUTS["fitcon"])
    res = dict(dp=d.p_reco - d.p_true, dtandip=d.tandip - d.tandip_t, dd0=d.d0 - d.d0_t, drmax=d.rmax - d.rmax_t,
               fitcon=d.fitcon, chi2ndf=d.chi2ndf, nactive=d.nactive)
    res = {k: np.asarray(v, float) for k, v in res.items()}
    eff_fit = np.zeros((ne, nt))
    eff_q = np.zeros((ne, nt))
    icdf = {k: np.zeros((ne, nt, NQ)) for k in RES + QUAL}
    chol = np.zeros((ne, nt, len(RES + QUAL), len(RES + QUAL)))
    level = np.zeros((ne, nt), np.int8)        # 0 own bin, 1/2 +-1/+-2 tan(dip) bins, 3 nrow row, 4 all
    count = np.zeros((ne, nt), np.int64)
    def pools(i, j):
        """Event selections of growing width: the bin, then +-1, +-2 neighbouring tan(dip) bins, the nrow row, all."""
        sel = [(bn == i) & (np.abs(bt - j) <= w) for w in (0, 1, 2)]
        return sel + [bn == i, np.ones(len(d), bool)]

    for i in range(ne):
        for j in range(nt):
            sel_e = pools(i, j)
            count[i, j] = int(sel_e[0].sum())
            lv = next(k for k, s_ in enumerate(sel_e) if s_.sum() >= min_eff or k == len(sel_e) - 1)
            s = sel_e[lv]
            eff_fit[i, j] = fit[s].mean() if s.any() else 0.0
            lq0 = next(k for k, s_ in enumerate(sel_e) if (s_ & fit).sum() >= min_eff or k == len(sel_e) - 1)
            eff_q[i, j] = qual[sel_e[lq0]].sum() / max(fit[sel_e[lq0]].sum(), 1)
            sel_q = [sq & qual for sq in sel_e]
            lq = next(k for k, s_ in enumerate(sel_q) if s_.sum() >= min_count or k == len(sel_q) - 1)
            level[i, j] = max(lv, lq)
            for k in RES + QUAL:
                icdf[k][i, j] = _icdf(res[k][sel_q[lq]])
            chol[i, j] = _copula(np.stack([res[k][sel_q[lq]] for k in RES + QUAL], 1))
    out = dict(nrow_edges=np.asarray(nrow_edges, float), tandip_edges=np.asarray(tandip_edges, float),
               qnodes=qnodes(), eff_fit=eff_fit, eff_q=eff_q, level=level, count=count, chol=chol,
               axis_xy=np.asarray(axis_xy, float), bz=np.float64(bz),
               cuts=np.array([*CE_CUTS["tandip"], *CE_CUTS["d0"], *CE_CUTS["rmax"], CE_CUTS["fitcon"]]))
    for k in RES + QUAL:
        out[f"icdf_{k}"] = icdf[k]
    return out


def save(path, tab):
    np.savez_compressed(Path(path), **tab)


def load(path):
    with np.load(Path(path)) as f:
        return {k: f[k] for k in f.files}


def _node(u, nq):
    """Fractional node index of the probability u on the qnodes grid."""
    return (nq - 1) * np.arccos(np.clip(1.0 - 2.0 * u, -1.0, 1.0)) / np.pi


def _sample(icdf, bn, bt, u):
    nq = icdf.shape[-1]
    f = np.clip(_node(u, nq), 0.0, nq - 1.000001)
    i = f.astype(np.int64)
    w = f - i
    return icdf[bn, bt, i] * (1 - w) + icdf[bn, bt, i + 1] * w


def apply_numpy(tab, rng, has_entry, nrow, x, p, q=-1.0):
    """The device response (device.apply) in NumPy, for checks: per event fit, ce and the track observables."""
    ptrue, td, d0, rmax = entry_truth(x, p, tab["axis_xy"], float(tab["bz"]), q)
    ne, nt = tab["eff_fit"].shape
    bn = np.clip(np.digitize(nrow, tab["nrow_edges"]) - 1, 0, ne - 1)
    bt = np.clip(np.digitize(td, tab["tandip_edges"]) - 1, 0, nt - 1)
    from scipy.special import ndtr
    u = rng.random((2, len(nrow)))
    z = np.einsum("nij,nj->in", tab["chol"][bn, bt], rng.standard_normal((len(nrow), len(RES + QUAL))))
    fit = np.asarray(has_entry, bool) & (u[0] < tab["eff_fit"][bn, bt])
    qual = fit & (u[1] < tab["eff_q"][bn, bt])
    draw = {k: _sample(tab[f"icdf_{k}"], bn, bt, ndtr(z[i])) for i, k in enumerate(RES + QUAL)}
    c = tab["cuts"]
    obs = dict(fit=fit, qual=qual, p_true=ptrue, p_reco=ptrue + draw["dp"], tandip=td + draw["dtandip"],
               d0=d0 + draw["dd0"], rmax=rmax + draw["drmax"], fitcon=draw["fitcon"], chi2ndf=draw["chi2ndf"],
               nactive=draw["nactive"])
    obs["ce"] = qual & (obs["tandip"] >= c[0]) & (obs["tandip"] <= c[1]) & (obs["d0"] >= c[2]) & \
        (obs["d0"] <= c[3]) & (obs["rmax"] >= c[4]) & (obs["rmax"] <= c[5])
    for k in obs:
        if k not in ("fit", "qual", "ce", "p_true"):
            obs[k] = np.where(qual, obs[k], np.nan)
    return obs


def reference_events(gen_paths, truth_paths, steps_paths, nts_dir, axis_xy=(-3904.0, 0.0), bz=1.0):
    """Per generated event of the reference arm: entry truth (dumper tree `entries` of truth_paths), step rows
    (`sgsteps` of steps_paths) and the first fitted LoopHelix track of the EventNtuples under nts_dir
    (validation.cemom.read_arm). Denominator: the events of the `gen` trees of gen_paths."""
    import pandas as pd
    from mu2efastsim import tables
    from trkfastsim.validation import cemom
    KEY = tables.KEY

    def keyed(t):
        for k in KEY:
            t[k] = t[k].astype(np.uint64)
        return t
    den = keyed(tables.read_many(gen_paths, "gen", KEY))[KEY].drop_duplicates()
    tr = keyed(tables.read_many(truth_paths, "entries", KEY + ["x", "y", "z", "px", "py", "pz"]))
    tr = tr.drop_duplicates(KEY)
    pt, td, d0, rm = entry_truth(tr[["x", "y", "z"]].to_numpy(), tr[["px", "py", "pz"]].to_numpy(), axis_xy, bz)
    tr = tr.rename(columns={c: f"e_{c}" for c in ("x", "y", "z", "px", "py", "pz")})
    tr = tr.assign(has_entry=True, p_true=pt, tandip_t=td, d0_t=d0, rmax_t=rm)
    st = keyed(tables.read_many(steps_paths, "sgsteps", KEY))
    nrow = st.groupby(KEY).size().rename("nrow").reset_index()
    arm, _ = cemom.read_arm(nts_dir)
    d = den.merge(tr, on=KEY, how="left").merge(nrow, on=KEY, how="left").merge(arm, on=KEY, how="left")
    d["has_entry"] = d.has_entry.fillna(False).astype(bool)
    d["nrow"] = d.nrow.fillna(0).astype(np.int64)
    d["fit"] = d.fit.fillna(False).astype(bool)
    d["ce"] = d.ce.fillna(False).astype(bool)
    return d


def main(argv=None):
    import argparse
    import glob
    import json
    ap = argparse.ArgumentParser(prog="python3 -m trkfastsim.recoresp.table", description="build CE reco response tables")
    ap.add_argument("--steps-dirs", nargs="+", required=True,
                    help="per-file dirs with states.root (gen), g4entries.root (truth) and g4prim_field.root (steps)")
    ap.add_argument("--nts", required=True, help="EventNtuple dir of the reference arm (nts.*.root below it)")
    ap.add_argument("--out", required=True, help="tables .npz")
    ap.add_argument("--events-out", help="optional per-event reference table (.csv.gz)")
    ap.add_argument("--bz", type=float, default=1.0)
    ap.add_argument("--min-count", type=int, default=40)
    a = ap.parse_args(argv)
    dirs = sorted(sum((glob.glob(d) for d in a.steps_dirs), []))
    d = reference_events([f"{x}/states.root" for x in dirs], [f"{x}/g4entries.root" for x in dirs],
                         [f"{x}/g4prim_field.root" for x in dirs], a.nts, bz=a.bz)
    tab = build(d, min_count=a.min_count, bz=a.bz)
    save(a.out, tab)
    if a.events_out:
        d.to_csv(a.events_out, index=False)
    print(json.dumps(dict(events=len(d), with_entry=int(d.has_entry.sum()), fit=int(d.fit.sum()), ce=int(d.ce.sum()),
                          bins=list(tab["eff_fit"].shape), merged_bins=int((tab["level"] > 0).sum())), indent=1))


if __name__ == "__main__":
    main()
