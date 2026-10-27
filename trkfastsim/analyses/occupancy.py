"""Per-panel occupancy shares of two arms with event-clustered errors (T-V2 panel occupancy row).

Hits come in clusters: one muon puts 2-4 digis into each panel it crosses, a delta-ray shower many more. Treating every
hit as an independent Poisson count understates the variance of a panel's share by about the mean cluster size, so a
chi2 over panels with Poisson errors exceeds 1 for two samples of identical physics. The variance used here is the
exact one for a sum over independent events: share s_p = C_p / C with C_p = sum_e n_pe, linearised,
var(s_p) = sum_e (n_pe - s_p n_e)^2 / C^2 = (Q_p - 2 s_p X_p + s_p^2 Q) / C^2, Q_p = sum_e n_pe^2, X_p = sum_e n_pe n_e,
Q = sum_e n_e^2. Cross-panel moments S_pq = sum_e n_pe n_qe give the full share covariance.
The full-covariance quadratic statistic uses its numerical rank; diagonal squared-pull sums are diagnostics only.
All comparisons here assume independent arms; paired arms require an additional cross-arm covariance.
The Poisson form (var = s_p / C) is kept for comparison with the earlier metric.

Panels are uniquePanel = plane * 6 + panel (StrawId: straw bits 0-6, panel 7-9, plane 10-15).
"""
import numpy as np
from scipy.sparse import csr_matrix

NPLANES, NPANELS = 36, 6
NPAN = NPLANES * NPANELS


def unique_panel(sid):
    """StrawId.asUint16() -> uniquePanel (plane * 6 + panel)."""
    sid = np.asarray(sid, np.int64)
    return (sid >> 10) * NPANELS + ((sid >> 7) & 0x7)


def accumulate(ev, panel, npan=NPAN, acc=None):
    """Add hits (event index per hit, panel per hit) to the running moments {C, Q, X, S, N, QN, nev}; events with no hit
    are not seen here, so nev counts events with at least one hit (it does not enter the shares)."""
    ev = np.asarray(ev, np.int64)
    panel = np.asarray(panel, np.int64)
    acc = acc or {"C": np.zeros(npan), "Q": np.zeros(npan), "X": np.zeros(npan),
                  "S": np.zeros((npan, npan)), "N": 0.0, "QN": 0.0, "nev": 0}
    if ev.shape != panel.shape or np.any((panel < 0) | (panel >= npan)):
        raise ValueError("Event and panel arrays must align and panel indices must be in range")
    if not len(ev):
        return acc
    _, e = np.unique(ev, return_inverse=True)
    ne = np.bincount(e).astype(float)
    pe = e * npan + panel
    u, c = np.unique(pe, return_counts=True)
    up, ue = u % npan, u // npan
    counts = csr_matrix((c.astype(float), (ue, up)), shape=(len(ne), npan))
    acc["S"] += (counts.T @ counts).toarray()
    acc["C"] += np.bincount(up, weights=c, minlength=npan)
    acc["Q"] += np.bincount(up, weights=c.astype(float) ** 2, minlength=npan)
    acc["X"] += np.bincount(up, weights=c * ne[ue], minlength=npan)
    acc["N"] += ne.sum()
    acc["QN"] += (ne ** 2).sum()
    acc["nev"] += len(ne)
    return acc


def merge(a, b):
    """Sum of two accumulators."""
    return {k: a[k] + b[k] for k in a}


def shares(acc):
    """(share per panel, clustered variance, Poisson variance)."""
    C, N = acc["C"], max(acc["N"], 1.0)
    s = C / N
    var = (acc["Q"] - 2 * s * acc["X"] + s ** 2 * acc["QN"]) / N ** 2
    return s, np.maximum(var, 0.0), s / N


def covariance(acc):
    """Full linearised covariance of panel shares, including cross-panel event correlations."""
    s, _, _ = shares(acc)
    n = max(acc["N"], 1.0)
    v = (acc["S"] - np.outer(acc["X"], s) - np.outer(s, acc["X"])
         + np.outer(s, s) * acc["QN"]) / n ** 2
    return (v + v.T) / 2


def _quadratic(d, cov):
    """Covariance-rank discrepancy; nonzero residual in the null space is incompatible."""
    if not len(d):
        return float("nan"), 0, False
    eig, vec = np.linalg.eigh(cov)
    keep = eig > max(float(eig.max()) * 1e-10, 1e-30)
    rank = int(keep.sum())
    proj = vec.T @ d
    incompatible = bool(np.linalg.norm(proj[~keep]) > max(np.linalg.norm(d) * 1e-7, 1e-12))
    if not rank:
        return float("nan"), 0, incompatible
    score = float(np.sum(proj[keep] ** 2 / eig[keep]) / rank)
    return (float("inf") if incompatible else score), rank, incompatible


def compare(acc_f, acc_r, min_ref=20):
    """Compare independent-arm shares with full covariance and rank, plus diagonal diagnostic scores.

    Marginal event-clustered pulls supply the individual-panel outliers. Plane shares use
    the covariance of all contributing panels. Fewer than one covariance mode is unavailable.
    """
    sf, vf, pf = shares(acc_f)
    sr, vr, pr = shares(acc_r)
    ok = acc_r["C"] >= min_ref
    ndf = max(int(ok.sum()) - 1, 1)
    d = sf - sr
    chi_p = float((d[ok] ** 2 / (pf[ok] + pr[ok])).sum() / ndf)
    pull = np.where(ok, d / np.sqrt(np.maximum(vf + vr, 1e-30)), 0.0)
    chi_c = float((pull[ok] ** 2).sum() / ndf) if ok.any() else float("nan")
    cov = covariance(acc_f) + covariance(acc_r)
    chi_cov, rank, incompatible = _quadratic(d[ok], cov[np.ix_(ok, ok)])
    worst = np.argsort(-np.abs(pull))[:10]
    plane = np.arange(len(sf)) // NPANELS
    prof = []
    for pl in range(NPLANES):
        m = (plane == pl) & ok
        if not m.any():
            continue
        a, b = sf[m].sum(), sr[m].sum()
        e = np.sqrt(max(float(cov[np.ix_(m, m)].sum()), 0.0))
        prof.append({"plane": pl, "share": [float(a), float(b)], "ratio_minus_1": float(a / b - 1) if b else float("nan"),
                     "pull": float((a - b) / e) if e > 0 else 0.0})
    return {"panels_used": int(ok.sum()), "ndf": ndf, "chi2ndf_poisson": chi_p, "chi2ndf_clustered": chi_c,
            "chi2ndf_covariance": chi_cov, "covariance_rank": rank,
            "covariance_model": "independent arms; full within-event panel covariance",
            "null_space_incompatible": incompatible, "available": bool(rank > 0),
            "diagonal_statistic_note": "sum of marginal squared pulls, not calibrated chi-square",
            "overdispersion_median": float(np.median(((vf + vr) / np.maximum(pf + pr, 1e-30))[ok])) if ok.any() else float("nan"),
            "max_abs_pull": float(np.abs(pull[ok]).max()) if ok.any() else 0.0,
            "panels_beyond_5sigma": int((np.abs(pull[ok]) > 5).sum()),
            "worst": [{"upanel": int(k), "plane": int(k // NPANELS), "panel": int(k % NPANELS), "pull": float(pull[k]),
                       "ratio_minus_1": float(sf[k] / sr[k] - 1) if sr[k] else float("nan")} for k in worst],
            "pulls": pull.tolist(), "by_plane": prof,
            "hits": [float(acc_f["N"]), float(acc_r["N"])], "events_with_hits": [int(acc_f["nev"]), int(acc_r["nev"])]}
