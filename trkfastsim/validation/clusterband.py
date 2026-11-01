"""Event-clustered normalized histogram ratios and entry-weighted means.

An entry may be a per-straw energy, a TDC value, or a track chi2. Supply the
*full* (run, subrun, event) key per entry, or globally unique integer event IDs.
Repeated IDs denote one statistical cluster, not independent observations.

Histogram shares use each arm's random IN-RANGE count, unlike the historical
core ratio_band's all-entry normalization. Reference quantile edges follow that
helper, but uncertainty is conditional on the chosen edges (no edge-selection
uncertainty). There is no finite-cluster/Bessel correction. A paired comparison
joins event IDs, with absent events contributing zero; independent arms never
share clusters. Dependence across different events (e.g. library reuse) is not
modeled. A max marginal deviation minus 2 sigma is NOT a simultaneous confidence
band or a calibrated significance test. Empty and rank-zero comparisons cannot
certify agreement, even if their descriptive ratios coincide exactly.

All public results are JSON-safe, with None rather than NaN for undefined values.
No event-by-bin dense array or Python event loop is constructed.
"""

import numpy as np
from scipy.sparse import csr_matrix


__all__ = ["ratio_band", "mean_compare"]


def _arm(values, events):
    values = np.asarray(values, dtype=float)
    events = np.asarray(events)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not (events.ndim == 1 or (events.ndim == 2 and events.shape[1] == 3)):
        raise ValueError("event IDs must be 1D integers or an Nx3 integer array")
    if len(values) != len(events):
        raise ValueError("values and event IDs must align")
    if events.size and events.dtype.kind not in "iu":
        raise ValueError("event IDs must be integers, without lossy float conversion")
    if events.size and events.dtype.kind == "u" and events.max() > np.iinfo(np.int64).max:
        raise ValueError("event IDs must fit in signed int64")
    return values, events.astype(np.int64)


def _inputs(fast, ref, events_fast, events_ref, paired):
    if not isinstance(paired, (bool, np.bool_)):
        raise ValueError("paired must be a boolean")
    f, ef = _arm(fast, events_fast)
    r, er = _arm(ref, events_ref)
    if paired:
        # An untyped empty list is also acceptable opposite an Nx3 arm.
        if not len(ef):
            ef = np.empty((0,) + er.shape[1:], dtype=np.int64)
        if not len(er):
            er = np.empty((0,) + ef.shape[1:], dtype=np.int64)
        if ef.ndim != er.ndim:
            raise ValueError("paired arms must use the same event-ID representation")
        ids, inv = np.unique(np.concatenate((ef, er)), axis=0, return_inverse=True)
        ie_f, ie_r = inv[:len(f)], inv[len(f):]
        matched = int(np.intersect1d(ie_f, ie_r).size)
        ne = len(ids)
    else:
        uf, ie_f = np.unique(ef, axis=0, return_inverse=True)
        ur, ie_r = np.unique(er, axis=0, return_inverse=True)
        ie_r = ie_r + len(uf)
        ne, matched = len(uf) + len(ur), None
    return f, r, ie_f, ie_r, ne, matched


def _conventions(paired):
    return {
        "covariance_model": "event-clustered delta method; " + (
            "paired by full event ID, absent events zero" if paired else "independent arms"),
        "cluster_correction": "none (sum of event influence outer products)",
        "band_convention": "maximum marginal |ratio-1|, optionally subtracting 2 standard errors; not simultaneous coverage",
        "rank_convention": "numerical covariance rank; zero rank unavailable, including exactly identical paired arms",
        "limitations": "independent event clusters assumed; cross-event library reuse not modeled",
    }


def _hist(values, events, ne, edges):
    finite = np.isfinite(values)
    inside = finite & (values >= edges[0]) & (values <= edges[-1])
    nb = len(edges) - 1
    bins = np.searchsorted(edges, values[inside], side="right") - 1
    bins = np.minimum(bins, nb - 1)  # numpy histogram includes the last right edge
    counts = csr_matrix((np.ones(bins.size), (events[inside], bins)), shape=(ne, nb))
    counts.sum_duplicates()
    flow = {"total": int(len(values)), "finite": int(finite.sum()),
            "in_range": int(inside.sum()),
            "underflow": int((finite & (values < edges[0])).sum()),
            "overflow": int((finite & (values > edges[-1])).sum()),
            "nonfinite": int((~finite).sum()),
            "events_with_entries": int(np.unique(events).size),
            "events_in_range": int(np.unique(events[inside]).size)}
    return counts, flow


def _share(counts):
    h = np.asarray(counts.sum(axis=0)).ravel()
    totals = np.asarray(counts.sum(axis=1)).ravel()
    n = h.sum()
    if not n:
        return h, totals, None, None
    s = h / n
    q = np.asarray(counts.multiply(counts).sum(axis=0)).ravel()
    x = np.asarray(counts.T @ totals).ravel()
    var = (q - 2 * s * x + s * s * (totals @ totals)) / n**2
    return h, totals, s, np.sqrt(np.maximum(var, 0.0))


def _ratio_covariance(cf, cr, nf, nr, sf, sr):
    """Sparse plus rank-one influence Z = D + outer(t, ratio)."""
    ratio = sf / sr
    df = cf.multiply(1.0 / (nf.sum() * sr))
    dr = cr.multiply(ratio / (nr.sum() * sr))
    d = df - dr
    t = nr / nr.sum() - nf / nf.sum()
    dt = np.asarray(d.T @ t).ravel()
    a = (d.T @ d).toarray()
    b = np.outer(dt, ratio)
    c = (t @ t) * np.outer(ratio, ratio)
    cov = a + b + b.T + c
    # Remove cancellation-scale roundoff without introducing variance for identical paired inputs.
    scale = np.abs(a) + np.abs(b) + np.abs(b.T) + np.abs(c)
    cov[np.abs(cov) <= 64 * np.finfo(float).eps * scale] = 0.0
    cov = (cov + cov.T) / 2
    np.fill_diagonal(cov, np.maximum(np.diag(cov), 0.0))
    return ratio, cov


def _rank(cov):
    if not cov.size:
        return 0
    eigenvalues = np.linalg.eigvalsh(cov)
    return int((eigenvalues > max(float(eigenvalues[-1]) * 1e-10, 1e-30)).sum())


def ratio_band(fast, ref, events_fast, events_ref, *, paired=True, edges=None,
               q=(0.05, 0.95), nbins=20, min_ref=400):
    """Compare unit-normalized IN-RANGE histograms, clustered by event.

    Parameters are 1D values and aligned integer event IDs (1D or Nx3) for each
    arm. ``paired=False`` ignores cross-arm ID coincidences. ``edges`` are fixed
    finite increasing bin edges; otherwise use equal-width bins over reference
    finite-value quantiles ``q`` (numpy's +/-0.5 expansion for a constant range).
    ``min_ref`` selects bins for the two maximum-deviation diagnostics.

    Returns counts, explicit finite under/overflow/nonfinite counts, shares and
    share errors, per-bin normalized ratios/errors (None for zero reference
    bins), and the full ratio covariance indexed by ``covariance_bins``. Rank
    refers only to the min_ref-selected submatrix. ``max_abs_dev`` remains a
    descriptive point estimate at rank zero; the 2-sigma band is None unless
    ``available``. Neither field is a goodness-of-fit probability.
    """
    f, r, ef, er, ne, matched = _inputs(fast, ref, events_fast, events_ref, paired)
    if not isinstance(min_ref, (int, float, np.integer, np.floating)) or not np.isfinite(min_ref) or min_ref < 0:
        raise ValueError("min_ref must be finite and nonnegative")
    derived = edges is None
    if derived:
        quantiles = np.asarray(q, dtype=float)
        if quantiles.shape != (2,) or not (0 <= quantiles[0] < quantiles[1] <= 1):
            raise ValueError("q must be two increasing probabilities")
        if not isinstance(nbins, (int, np.integer)) or isinstance(nbins, bool) or nbins < 1:
            raise ValueError("nbins must be a positive integer")
        finite_ref = r[np.isfinite(r)]
        if finite_ref.size:
            lo, hi = np.quantile(finite_ref, quantiles)
            edges = np.histogram_bin_edges(finite_ref, bins=nbins, range=(lo, hi))
        else:
            edges = None
    else:
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be finite and strictly increasing")
    out = {"paired": bool(paired), "available": False, "unavailable_reason": None,
           "normalization": "each arm's random finite in-range entry count",
           "edge_convention": "reference finite quantiles; uncertainty conditional on edges" if derived else "fixed edges",
           "conventions": _conventions(paired), "matched_events_with_entries": matched,
           "min_ref": float(min_ref), "edges": [] if edges is None else edges.tolist(),
           "h_fast": [], "h_ref": [], "share_fast": [], "share_ref": [],
           "share_error_fast": [], "share_error_ref": [], "ratio": [], "ratio_minus_1": [],
           "ratio_error": [], "bins_used": [], "nbins_used": 0,
           "covariance_bins": [], "ratio_covariance": [], "covariance_rank": 0,
           "max_abs_dev": None, "max_abs_dev_minus_2sigma": None}
    if edges is None:
        # With no reference range, do not invent an in-range/flow classification.
        for name, v, ev in (("fast", f, ef), ("ref", r, er)):
            finite = np.isfinite(v)
            out["flow_" + name] = {"total": int(len(v)), "finite": int(finite.sum()),
                "in_range": None, "underflow": None, "overflow": None,
                "nonfinite": int((~finite).sum()), "events_with_entries": int(np.unique(ev).size),
                "events_in_range": None}
        out["unavailable_reason"] = "no finite reference values to derive edges"
        return out
    cf, out["flow_fast"] = _hist(f, ef, ne, edges)
    cr, out["flow_ref"] = _hist(r, er, ne, edges)
    hf, nf, sf, sef = _share(cf)
    hr, nr, sr, ser = _share(cr)
    nb = len(edges) - 1
    for name, h, s, se in (("fast", hf, sf, sef), ("ref", hr, sr, ser)):
        out["h_" + name] = h.astype(np.int64).tolist()
        out["share_" + name] = [None] * nb if s is None else s.tolist()
        out["share_error_" + name] = [None] * nb if se is None else se.tolist()
    for name in ("ratio", "ratio_minus_1", "ratio_error"):
        out[name] = [None] * nb
    if sf is None or sr is None:
        out["unavailable_reason"] = "empty in-range arm"
        return out
    valid = np.flatnonzero(hr > 0)
    ratio, cov = _ratio_covariance(cf[:, valid], cr[:, valid], nf, nr, sf[valid], sr[valid])
    errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    out["covariance_bins"] = valid.tolist()
    out["ratio_covariance"] = cov.tolist()
    for name, values in (("ratio", ratio), ("ratio_minus_1", ratio - 1), ("ratio_error", errors)):
        full = np.full(nb, np.nan)
        full[valid] = values
        out[name] = [float(x) if np.isfinite(x) else None for x in full]
    keep = hr[valid] >= min_ref
    out["bins_used"] = valid[keep].tolist()
    out["nbins_used"] = int(keep.sum())
    out["covariance_rank"] = _rank(cov[np.ix_(keep, keep)])
    if not keep.any():
        out["unavailable_reason"] = "no bins meet min_ref"
        return out
    dev = np.abs(ratio[keep] - 1)
    out["max_abs_dev"] = float(dev.max())
    if out["flow_fast"]["events_in_range"] < 2 or out["flow_ref"]["events_in_range"] < 2:
        out["unavailable_reason"] = "fewer than two in-range event clusters in an arm"
    elif not out["covariance_rank"]:
        out["unavailable_reason"] = "zero covariance rank"
    else:
        out["available"] = True
        out["max_abs_dev_minus_2sigma"] = float(np.maximum(dev - 2 * errors[keep], 0).max())
    return out


def mean_compare(fast, ref, events_fast, events_ref, *, paired=True):
    """Entry-weighted means and paired/independent event-clustered errors.

    For per-straw means pass one energy per straw (already aggregated upstream).
    All finite entries are included, not just the histogram range. Each event's
    mean influence is (sum(values_e) - mean * count_e) / total_count. Returned
    difference and ratio errors include paired cross-arm covariance. Rank-zero
    or fewer-than-two-cluster comparisons are descriptive only (unavailable).
    """
    f, r, ef, er, ne, matched = _inputs(fast, ref, events_fast, events_ref, paired)
    out = {"paired": bool(paired), "available": False, "unavailable_reason": None,
           "conventions": _conventions(paired), "normalization": "each arm's random finite entry count",
           "matched_events_with_entries": matched, "difference": None, "difference_error": None,
           "ratio": None, "ratio_minus_1": None, "ratio_error": None, "cross_covariance": None}
    influences = []
    for name, values, ev in (("fast", f, ef), ("ref", r, er)):
        keep = np.isfinite(values)
        n = int(keep.sum())
        counts = np.bincount(ev[keep], minlength=ne).astype(float)
        mean = float(values[keep].mean()) if n else None
        influence = np.bincount(ev[keep], weights=values[keep] - mean, minlength=ne) / n if n else None
        out[name] = {"mean": mean, "error": float(np.linalg.norm(influence)) if n else None,
                     "count": n, "nonfinite": int((~keep).sum()),
                     "events": int(np.count_nonzero(counts))}
        influences.append(influence)
    uf, ur = influences
    if uf is None or ur is None:
        out["unavailable_reason"] = "empty finite arm"
        return out
    mf, mr = out["fast"]["mean"], out["ref"]["mean"]
    diff_error = float(np.linalg.norm(uf - ur))
    tolerance = 64 * np.finfo(float).eps * (np.linalg.norm(uf) + np.linalg.norm(ur))
    if diff_error <= tolerance:
        diff_error = 0.0
    out.update(difference=float(mf - mr), difference_error=diff_error,
               cross_covariance=float(uf @ ur))
    if mr:
        ratio = mf / mr
        ratio_error = float(np.linalg.norm((uf - ratio * ur) / mr))
        ratio_tolerance = 64 * np.finfo(float).eps * (np.linalg.norm(uf) + abs(ratio) * np.linalg.norm(ur)) / abs(mr)
        out.update(ratio=float(ratio), ratio_minus_1=float(ratio - 1),
                   ratio_error=0.0 if ratio_error <= ratio_tolerance else ratio_error)
    if min(out["fast"]["events"], out["ref"]["events"]) < 2:
        out["unavailable_reason"] = "fewer than two finite event clusters in an arm"
    elif not diff_error:
        out["unavailable_reason"] = "zero difference covariance rank"
    else:
        out["available"] = True
    return out
