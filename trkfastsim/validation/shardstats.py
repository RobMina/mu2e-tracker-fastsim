"""shardstats -- paired per-shard statistic for tracker comparisons: batch variance, sign-flip null, family-wise limits.

Input: an npz from `shardhist` holding per-shard histograms h|<arm>|<key> (shards x bins, flows included), the bin
edges and each arm's shard list. Two arms are compared on their common shards (subruns sharing the generated
primaries), never event by event.

Method, one rule for every histogram:
- d_sb = n_cand,sb - n_ref,sb per shard s and bin b; D_b = sum_s d_sb; var_b = N/(N-1) sum_s (d_sb - mean_b d)^2. This
  covers event clustering (one event fills many bins of an occupancy or per-straw histogram) and the correlation of the
  arms through shared primaries at once. Bins where no shard differs are left out of ndf.
- chi2/ndf = mean_b D_b^2 / var_b, times (N-3)/(N-1): D/sigma is a Student t with N-1 dof, E[t^2] = (N-1)/(N-3).
- Reference distribution: bins of one histogram are correlated, so chi2(ndf) is the wrong null. The null is a set of
  random sign flips of the shard differences, the same flips for every histogram (correlations within and between
  histograms survive).
- Limit: one per histogram, family-wise rate alpha over all histograms of the comparison. Gamma fits to each
  histogram's flip distribution equalise the histograms; the alpha quantile of the per-flip minimum p sets the common p.
- Size: rms_rel_excess, the n-weighted rms relative difference per bin after the noise is subtracted, and total_rel with
  its paired pull. A flag says a difference is resolved, not that it is large: read the size next to it.
- shape mode: the candidate's shard counts are scaled by the ratio of selected events (key sel/events) before
  differencing, so a rate difference does not flag every histogram; the ratio's own noise is neglected.
- Unpaired split null (`split_null`): half of one arm's shards against the other half, per-arm batch variances; the
  fixed first-vs-second and even-vs-odd splits are the tests, random halves give the reference.

Output: <outdir>/shardstats.json and a text table.
"""
import json
import os

import numpy as np


def batch_var(x):
    """Variance of the sum over shards of x (N, B): N/(N-1) sum_s (x_sb - mean_b)^2."""
    n = x.shape[0]
    return n / (n - 1.0) * ((x - x.mean(0)) ** 2).sum(0)


def batch_var_rows(x):
    """batch_var of x (F, N, B) along the shard axis."""
    n = x.shape[1]
    return n / (n - 1.0) * ((x - x.mean(1, keepdims=True)) ** 2).sum(1)


def bias_factor(n):
    """(N-3)/(N-1): makes the null mean of chi2/ndf 1 for a Student t with N-1 dof."""
    return (n - 3.0) / (n - 1.0)


def t2_mean(D, S2, n):
    """chi2/ndf = mean over bins of D^2 / var, var from the sum S2 of squares and the sum D over n shards."""
    var = n / (n - 1.0) * (S2 - D ** 2 / n)
    return (D ** 2 / np.maximum(var, 1e-12)).mean(-1)


def flips(nflip, n, seed=20261101):
    """Random sign vectors (nflip, n), shared by every histogram of a comparison."""
    return np.random.default_rng(seed).integers(0, 2, size=(nflip, n), dtype=np.int8).astype(np.float64) * 2 - 1


def gamma_fit(t):
    """Shape and scale matching the mean and variance of a null sample; None for a null without spread."""
    m, v = float(t.mean()), float(t.var())
    return None if v <= 0 or m <= 0 else (m * m / v, v / m)


def family_limits(obs, null, alpha):
    """Per-histogram limits with family-wise rate alpha: gamma tails equalise the histograms and the alpha quantile of
    the per-flip minimum p fixes the common p, so the rate holds whatever the true tails are."""
    from scipy.special import gammaincc
    from scipy.stats import gamma
    fits = {k: gamma_fit(t) for k, t in null.items()}
    use = [k for k in null if fits[k] is not None]
    out = {}
    if not use:
        return {k: {"threshold": None, "p": None, "flag": False} for k in null}, {"p_per_hist": None}
    minp = np.stack([gammaincc(fits[k][0], null[k] / fits[k][1]) for k in use]).min(0)
    pstar = float(np.quantile(minp, alpha))
    for k in null:
        if fits[k] is None:
            out[k] = {"threshold": None, "p": None, "flag": False}
            continue
        a, sc = fits[k]
        lim, pv = float(gamma.isf(pstar, a, scale=sc)), float(gammaincc(a, obs[k] / sc))
        out[k] = {"threshold": lim, "p": pv, "flag": bool(obs[k] > lim),
                  "p_flip": float((1 + (null[k] >= obs[k]).sum()) / (len(minp) + 1)),
                  "p_family": float((1 + (minp <= pv).sum()) / (len(minp) + 1)),
                  "z_null": float((obs[k] - null[k].mean()) / null[k].std())}
    tail = float(np.mean([(null[k] > out[k]["threshold"]).mean() for k in use]))
    return out, {"p_per_hist": pstar, "effective_trials": alpha / pstar if pstar > 0 else None,
                 "empirical_tail_at_limit": tail}


class Shards:
    """Per-shard histograms of one or more shardhist npz files (arms must not repeat across files)."""

    def __init__(self, paths):
        paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
        self.d = {}
        for p in paths:
            z = np.load(p, allow_pickle=False)
            for f in z.files:
                if f.startswith(("h|", "shards|")) and f in self.d:
                    raise ValueError(f"{f} appears in more than one input")
                if f.startswith("edges|") and f in self.d and not np.array_equal(self.d[f], z[f]):
                    raise ValueError(f"{f} differs between inputs")
                self.d[f] = z[f]
        self.files = set(self.d)

    def arms(self):
        return sorted(f[7:] for f in self.files if f.startswith("shards|"))

    def shards(self, arm):
        return self.d[f"shards|{arm}"]

    def keys(self, arm):
        n = len(f"h|{arm}|")
        return sorted(f[n:] for f in self.files if f.startswith(f"h|{arm}|"))

    def has(self, arm, key):
        return f"h|{arm}|{key}" in self.files

    def h(self, arm, key, shards=None):
        x = self.d[f"h|{arm}|{key}"].astype(np.float64)
        if shards is None:
            return x
        s = self.shards(arm)
        pos = np.searchsorted(s, shards)
        if np.any(pos >= len(s)) or np.any(s[np.minimum(pos, len(s) - 1)] != shards):
            raise ValueError(f"arm {arm} lacks some of the requested shards")
        return x[pos]

    def edges(self, key):
        return self.d[f"edges|{key}"] if f"edges|{key}" in self.files else None


def common_shards(sh, ref, cand):
    return np.intersect1d(sh.shards(ref), sh.shards(cand))


def _status(r, c, x):
    if r.sum() == 0 and c.sum() == 0:
        return "both_empty"
    if r.sum() == 0 or c.sum() == 0:
        return "one_empty"
    if not (x ** 2).sum(0).any():
        return "identical"
    return "ok"


def compare(sh, ref, cand, keys=None, nflip=20000, alpha=0.01, nboot=500, shape=False, chunk=500, shards=None):
    """One comparison: paired chi2/ndf (bias factor applied), sign-flip null, family-wise limits, sizes."""
    s = common_shards(sh, ref, cand) if shards is None else np.asarray(shards)
    n = len(s)
    if n < 4:
        raise ValueError("need at least four common shards")
    fac = bias_factor(n)
    E = flips(nflip, n)
    boot = np.random.default_rng(5).integers(0, n, size=(nboot, n)) if nboot else None
    keys = keys if keys is not None else [k for k in sh.keys(ref) if sh.has(cand, k)]
    scale = 1.0
    if shape:
        er, ec = sh.h(ref, "sel/events", s).sum(), sh.h(cand, "sel/events", s).sum()
        scale = er / ec if ec > 0 else 1.0
    per, null, obs = {}, {}, {}
    for key in keys:
        if not (sh.has(ref, key) and sh.has(cand, key)):
            continue
        r, c = sh.h(ref, key, s), sh.h(cand, key, s) * scale
        x = c - r
        st = _status(r, c, x) if not (shape and np.allclose(x, 0.0)) else "identical"
        rec = {"status": st, "n_ref": float(r.sum()), "n_cand": float(c.sum()), "bins": int(r.shape[1])}
        per[key] = rec
        if st != "ok":
            continue
        S2 = (x ** 2).sum(0)
        g = S2 > 0
        x, m = x[:, g], (r + c).sum(0)[g]
        S2 = S2[g]
        D = x.sum(0)
        var = batch_var(x)
        obs[key] = rec["chi2ndf"] = float(fac * t2_mean(D, S2, n))
        rec["ndf"] = int(g.sum())
        rec["max_abs_pull"] = float(np.sqrt(np.max(D ** 2 / np.maximum(var, 1e-12))))
        w = (2.0 * (D ** 2 - var) / np.maximum(m, 1.0)).sum() / max(0.5 * m.sum(), 1.0)
        rec["rms_rel_excess"] = float(np.sqrt(max(w, 0.0)))
        rec["total_rel"] = float(D.sum() / max(r.sum(), 1.0))
        vt = batch_var(x.sum(1, keepdims=True))[0]
        rec["total_rel_err"] = float(np.sqrt(vt) / max(r.sum(), 1.0))
        rec["total_pull"] = float(D.sum() / np.sqrt(vt)) if vt > 0 else None
        t = np.empty(nflip)
        for i in range(0, nflip, chunk):
            t[i:i + chunk] = fac * t2_mean(E[i:i + chunk] @ x, S2, n)
        null[key] = t
        rec["null_mean"], rec["null_sd"] = float(t.mean()), float(t.std())
        if nboot:
            xb = x[boot]
            ok = (xb ** 2).sum(1) > 0
            tb = (xb.sum(1) ** 2 / np.maximum(batch_var_rows(xb), 1e-12) * ok).sum(1) / np.maximum(ok.sum(1), 1)
            rec["chi2ndf_err"] = float(fac * tb.std())
    lim, meta = family_limits(obs, null, alpha) if obs else ({}, {})
    for k, v in lim.items():
        per[k].update(v)
    for k, r in per.items():
        r.setdefault("flag", r["status"] == "one_empty")
    meta.update(ref=ref, cand=cand, N=int(n), nflip=nflip, alpha_family=alpha, n_ok=len(null), shape=bool(shape),
                scale=float(scale), nflags=int(sum(r["flag"] for r in per.values())))
    return per, meta


def pulls(sh, ref, cand, keys=None, shards=None):
    """t_b = D_b / sigma_b over all bins of a comparison, one histogram per identical difference vector."""
    s = common_shards(sh, ref, cand) if shards is None else shards
    keys = keys if keys is not None else [k for k in sh.keys(ref) if sh.has(cand, k)]
    out, seen = [], set()
    for key in keys:
        if not (sh.has(ref, key) and sh.has(cand, key)):
            continue
        x = sh.h(cand, key, s) - sh.h(ref, key, s)
        x = x[:, (x ** 2).sum(0) > 0]
        if x.shape[1] == 0:
            continue
        sig = x.sum(0).tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(x.sum(0) / np.sqrt(np.maximum(batch_var(x), 1e-12)))
    return np.concatenate(out) if out else np.zeros(0)


def split_null(sh, arm, keys=None, nsplit=4000, seed=7, alpha=0.01):
    """Half of an arm's shards against the other half (no shared primaries), per-arm batch variances. Random halves
    give the reference distribution; the fixed first-vs-second and even-vs-odd splits are the tests."""
    s = sh.shards(arm)
    n = len(s)
    nh = n // 2
    rng = np.random.default_rng(seed)
    pick = np.stack([np.arange(n) < nh, np.arange(n) % 2 == 0] + [rng.permutation(n) < nh for _ in range(nsplit)])
    pick = pick.astype(np.float64)
    keys = keys if keys is not None else sh.keys(arm)
    out, nullt, o1, o2 = {}, {}, {}, {}
    for key in keys:
        x = sh.h(arm, key)
        g = (x ** 2).sum(0) > 0
        if not g.any() or np.all(batch_var(x[:, g]) == 0):
            continue
        x = x[:, g]
        a1, b1 = pick @ x, (1 - pick) @ x
        a2, b2 = pick @ x ** 2, (1 - pick) @ x ** 2
        na, nb = pick.sum(1, keepdims=True), (1 - pick).sum(1, keepdims=True)
        v = na / (na - 1.0) * (a2 - a1 ** 2 / na) + nb / (nb - 1.0) * (b2 - b1 ** 2 / nb)
        D = a1 - b1
        ok = v > 0
        t2 = np.where(ok, D ** 2 / np.where(ok, v, 1.0), 0.0)
        tb = t2.sum(1) / np.maximum(ok.sum(1), 1) * bias_factor(n - 1)
        nullt[key], o1[key], o2[key] = tb[2:], float(tb[0]), float(tb[1])
        out[key] = {"random_mean": float(tb[2:].mean()), "random_sd": float(tb[2:].std()), "bins": int(g.sum()),
                    "first_vs_second": o1[key], "even_vs_odd": o2[key],
                    "pull_rms_fixed": [float(np.sqrt(t2[i].sum() / max(ok[i].sum(), 1))) for i in (0, 1)]}
    if not out:
        return {"hists": {}, "meta": {"N": int(n)}}
    lim1, _ = family_limits(o1, nullt, alpha)
    lim2, meta = family_limits(o2, nullt, alpha)
    for k in out:
        out[k]["flag_first_vs_second"] = lim1[k]["flag"]
        out[k]["flag_even_vs_odd"] = lim2[k]["flag"]
    meta.update(N=int(n), arm=arm, nsplit=nsplit, alpha_family=alpha,
                nflags=int(sum(out[k]["flag_first_vs_second"] + out[k]["flag_even_vs_odd"] for k in out)))
    return {"hists": out, "meta": meta}


def pull_summary(t):
    if not len(t):
        return {}
    q = np.percentile(t, [15.87, 84.13])
    return {"bins": int(len(t)), "rms": float(np.sqrt(np.mean(t ** 2))), "rms_err": float(np.sqrt(np.mean(t ** 2)) / np.sqrt(2 * len(t))),
            "gauss_width": float(0.5 * (q[1] - q[0])), "mean": float(t.mean())}


def table(per, meta):
    """Text table of one comparison, flagged first."""
    rows = [f"# {meta['cand']} vs {meta['ref']}: N={meta['N']} shards, {meta['nflags']} flags of {meta['n_ok']} "
            f"({'shape' if meta['shape'] else 'counts'}; family alpha {meta['alpha_family']})",
            f"{'histogram':34s} {'chi2/ndf':>15s} {'limit':>7s} {'ndf':>6s} {'rms%':>6s} {'total%':>14s} flag"]
    order = sorted(per, key=lambda k: (not per[k]["flag"], k))
    for k in order:
        r = per[k]
        if r["status"] != "ok":
            rows.append(f"{k:34s} {r['status']:>15s}")
            continue
        lim = f"{r['threshold']:7.2f}" if r.get("threshold") is not None else "      -"
        rows.append(f"{k:34s} {r['chi2ndf']:7.2f} +- {r.get('chi2ndf_err', 0):5.2f} {lim} {r['ndf']:6d} "
                    f"{100 * r['rms_rel_excess']:6.2f} {100 * r['total_rel']:+6.2f} +- {100 * r['total_rel_err']:4.2f} "
                    f"{'F' if r['flag'] else ''}")
    return "\n".join(rows)


def run(inputs, outdir, pairs=None, splits=None, shape=False, nflip=20000, alpha=0.01, keys=None, **kw):
    """pairs: 'ref:cand' strings; splits: arm names for the unpaired split null."""
    os.makedirs(outdir, exist_ok=True)
    sh = Shards(inputs)
    sel = None
    if keys:
        import fnmatch
        sel = lambda ks: [k for k in ks if any(fnmatch.fnmatch(k, p) for p in keys)]  # noqa: E731
    res = {"inputs": [os.path.basename(p) for p in inputs], "comparisons": {}, "splits": {}}
    text = []
    for p in pairs or []:
        ref, cand = p.split(":")
        ks = [k for k in sh.keys(ref) if sh.has(cand, k)]
        ks = sel(ks) if sel else ks
        for mode in ((False, True) if shape else (False,)):
            per, meta = compare(sh, ref, cand, ks, nflip=nflip, alpha=alpha, shape=mode)
            meta["pulls"] = pull_summary(pulls(sh, ref, cand, ks))
            res["comparisons"][f"{cand}_vs_{ref}{'_shape' if mode else ''}"] = {"meta": meta, "hists": per}
            text.append(table(per, meta))
    for arm in splits or []:
        ks = sel(sh.keys(arm)) if sel else None
        res["splits"][arm] = split_null(sh, arm, ks, alpha=alpha)
        m = res["splits"][arm]["meta"]
        text.append(f"# split null {arm}: N={m['N']}, {m.get('nflags', 0)} flags (first-vs-second + even-vs-odd)")
    with open(os.path.join(outdir, "shardstats.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(outdir, "shardstats.txt"), "w") as f:
        f.write("\n\n".join(text) + "\n")
    print("\n\n".join(text))
    return res
