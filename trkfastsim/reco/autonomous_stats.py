"""Event-cluster bootstrap for *conditional matched-track* differences, not closure.

Each draw samples N events with replacement from the full supplied population,
including events with no matched/finite tracks. All tracks in a sampled event
receive its integer frequency. Tracks, not event means, enter each quantile.
Matches and per-variable finite-value selection are fixed before resampling.

Quantiles use NumPy's ``method='linear'`` (Hyndman-Fan type 7): in the sorted,
frequency-expanded sample of length M, interpolate at zero-based (M-1)*q.
This is NOT an inverse weighted empirical CDF. The central width is recomputed
as (q84-q16)/2 in each draw. SE is sample SD (ddof=1); CI95 is the linear
2.5/97.5 percentile interval of defined bootstrap estimates. Empty draws are
counted and excluded, so these intervals are conditional on a nonempty draw.
Finite endpoints use overflow-safe interpolation, widths and scaled SDs. Any
remaining arithmetic failure is distinct from an empty draw and suppresses
uncertainties rather than reporting status ok with an infinite SE or interval.

No per-event or per-replicate loops: batched event draws and frequency order
statistics avoid expanded track samples and N x B x P arrays. Temporary arrays
are O(batch * max(events, tracks)); only B x P x 4 estimates persist. Canonical
lexicographic full identities and an explicit PCG64 seed make draws reproducible.
"""
import numbers

import numpy as np

DEFAULT_REPLICATES = 1000
DEFAULT_SEED = 0
_WORK_ELEMENTS = 1_000_000
_PROBABILITIES = np.array([0.16, 0.5, 0.84])
_STATISTICS = ("q16", "q50", "q84", "central_width")


def validate_bootstrap(replicates, seed):
    for name, value, minimum in (("bootstrap-replicates", replicates, 2),
                                  ("bootstrap-seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed > np.iinfo(np.uint64).max:
        raise ValueError("bootstrap-seed must fit in uint64")


def _interpolate(lower, upper, fraction):
    """Type-7 interpolation without overflowing a finite endpoint difference."""
    same_sign = np.signbit(lower) == np.signbit(upper)
    delta = np.subtract(upper, lower, where=same_sign, out=np.zeros_like(lower))
    nearby = np.where(fraction < .5, lower + delta * fraction,
                      upper - delta * (1 - fraction))
    return np.add(lower * (1 - fraction), upper * fraction,
                  where=~same_sign, out=nearby)


def linear_quantiles(values, probabilities):
    """Finite-sample linear quantiles with overflow-safe interpolation."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    probabilities = np.asarray(probabilities)
    if not len(values):
        return np.full(probabilities.shape, np.nan)
    h = (len(values) - 1) * probabilities
    return _interpolate(values[np.floor(h).astype(np.int64)],
                        values[np.ceil(h).astype(np.int64)], h - np.floor(h))


def _central_width(lower, upper):
    # Halving first is necessary only across zero; preserve tiny same-sign widths.
    same_sign = np.signbit(lower) == np.signbit(upper)
    # Undefined draws (or arithmetic failures upstream) remain nonfinite.
    with np.errstate(invalid="ignore"):
        delta = np.subtract(upper, lower, where=same_sign, out=np.zeros_like(lower))
        return np.where(same_sign, delta / 2, upper / 2 - lower / 2)


def _sample_std(values):
    scale = np.max(np.abs(values))
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.std(values / scale, ddof=1) * scale) if scale else 0.0


def _frequency_quantiles(values, owners, event_counts):
    """Type-7 quantiles of frequency-expanded sorted values, without expansion."""
    batch = len(event_counts)
    if not len(values):
        return np.full((batch, 3), np.nan)
    cumulative = np.cumsum(event_counts[:, owners], axis=1)
    total = cumulative[:, -1]
    h = np.maximum(total - 1, 0)[:, None] * _PROBABILITIES
    ranks = np.concatenate((np.floor(h), np.ceil(h)), axis=1).astype(np.int64)
    # Parallel lower-bound searches for the first cumulative count > rank.
    lo = np.zeros_like(ranks)
    hi = np.full_like(ranks, len(values) - 1)
    rows = np.arange(batch)[:, None]
    for _ in range(len(values).bit_length()):
        mid = (lo + hi) // 2
        right = cumulative[rows, mid] <= ranks
        lo = np.where(right, mid + 1, lo)
        hi = np.where(right, hi, mid)
    # Empty rows have no order statistic; index safely and mark them undefined.
    selected = values[np.minimum(lo, len(values) - 1)]
    fraction = h - np.floor(h)
    result = _interpolate(selected[:, :3], selected[:, 3:], fraction)
    result[total == 0] = np.nan
    return result


def event_cluster_bootstrap(event_keys, track_events, differences, *,
                            replicates=DEFAULT_REPLICATES, seed=DEFAULT_SEED):
    """JSON-ready uncertainties for a mapping of per-matched-track differences.

    ``event_keys`` is the full unique (N,3) int64 run/subrun/event population;
    ``track_events`` gives each matched track's row in that population. Multiple
    tracks may share an event. Masked values denote missing fields/crossings;
    unmasked NaN/inf differences are counted separately as nonfinite. Neither
    category contributes to point estimates or resamples. Counts are per variable.

    Fewer than two finite-contributing events, fewer than two defined draws, or
    an empirically constant statistic produce null SE/CI with an explicit status,
    never a spurious zero error. A constant estimate is still reported. Nonempty
    arithmetic failures are counted separately from empty draws and suppress
    SE/CI; nonrepresentable final SE/CI also get status arithmetic_failure.
    """
    validate_bootstrap(replicates, seed)
    keys = np.asanyarray(event_keys)
    if keys.shape == (0,):
        keys = np.empty((0, 3), dtype=np.int64)
    if (keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype.kind not in "iu"
            or np.any(np.ma.getmaskarray(keys)) or np.any(keys > np.iinfo(np.int64).max)):
        raise ValueError("event_keys must contain exact int64 run/subrun/event identities")
    keys = np.asarray(keys, dtype=np.int64)
    unique, canonical = np.unique(keys, axis=0, return_inverse=True)
    canonical = canonical.ravel()
    if len(unique) != len(keys):
        raise ValueError("duplicate run/subrun/event identity")
    owners = np.asanyarray(track_events)
    if (owners.ndim != 1 or np.any(np.ma.getmaskarray(owners)) or
            (owners.size and (owners.dtype.kind not in "iu" or np.any(owners < 0)
                             or np.any(owners >= len(keys))))):
        raise ValueError("track_events must index the full event population")
    owners = canonical[np.asarray(owners, dtype=np.int64)]
    n, m = len(keys), len(owners)
    prepared, variables = {}, {}
    for name, raw in differences.items():
        values = np.ma.asarray(raw, dtype=np.float64)
        if values.shape != (m,):
            raise ValueError(f"{name}: one difference per matched track required")
        missing = np.ma.getmaskarray(values)
        values = values.filled(np.nan)
        finite = ~missing & np.isfinite(values)
        order = np.argsort(values[finite], kind="stable")
        prepared[name] = (values[finite][order], owners[finite][order])
        effective = len(np.unique(owners[finite]))
        variables[name] = {
            "matched_tracks": m, "finite_tracks": int(finite.sum()),
            "missing_tracks": int(missing.sum()),
            "nonfinite_tracks": int((~missing & ~np.isfinite(values)).sum()),
            "effective_events": effective, "events_without_finite_tracks": n - effective,
        }
    draws = {name: np.full((replicates, 4), np.nan) for name in differences}
    empty = {name: replicates for name in differences}
    batch_size = max(1, min(128, _WORK_ELEMENTS // max(n, m, 1)))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    if n and any(len(v) for v, _ in prepared.values()):
        for start in range(0, replicates, batch_size):
            batch = min(batch_size, replicates - start)
            sampled = rng.integers(n, size=(batch, n))
            sampled += n * np.arange(batch)[:, None]
            counts = np.bincount(sampled.ravel(), minlength=batch * n).reshape(batch, n)
            del sampled
            for name, (values, event_owners) in prepared.items():
                empty[name] -= int(np.any(counts[:, event_owners], axis=1).sum())
                q = _frequency_quantiles(values, event_owners, counts)
                draws[name][start:start + batch, :3] = q
                draws[name][start:start + batch, 3] = _central_width(q[:, 0], q[:, 2])
    for name, row in variables.items():
        values, _ = prepared[name]
        q = linear_quantiles(values, _PROBABILITIES)
        estimates = np.r_[q, _central_width(q[0], q[2])]
        valid = draws[name][np.all(np.isfinite(draws[name]), axis=1)]
        row["bootstrap_valid_replicates"] = len(valid)
        row["bootstrap_empty_replicates"] = empty[name]
        failures = replicates - empty[name] - len(valid)
        row["bootstrap_arithmetic_failure_replicates"] = failures
        row["statistics"] = {}
        for j, statistic in enumerate(_STATISTICS):
            status = ("no_finite_tracks" if not len(values) else
                      "arithmetic_failure" if failures or not np.isfinite(estimates[j]) else
                      "insufficient_effective_events" if row["effective_events"] < 2 else
                      "insufficient_valid_replicates" if len(valid) < 2 else
                      "degenerate_bootstrap" if np.all(valid[:, j] == valid[0, j]) else "ok")
            se = _sample_std(valid[:, j]) if status == "ok" else None
            ci = linear_quantiles(valid[:, j], [.025, .975]) if status == "ok" else None
            if status == "ok" and (not np.isfinite(se) or se <= 0 or not np.all(np.isfinite(ci))):
                status, se, ci = "arithmetic_failure", None, None
            row["statistics"][statistic] = {
                "estimate": float(estimates[j]) if np.isfinite(estimates[j]) else None,
                "bootstrap_se": se,
                "ci95": ci.tolist() if ci is not None else None,
                "status": status,
            }
    return {
        "events": n, "matched_tracks": m,
        "events_with_matched_tracks": len(np.unique(owners)),
        "bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed),
        "rng": "numpy.random.PCG64",
        "resampling": "N full events with replacement; canonical run/subrun/event order; all matched tracks clustered",
        "quantile_method": "linear (Hyndman-Fan type 7), rank=(M-1)*q in frequency-expanded track sample",
        "quantile_probabilities": _PROBABILITIES.tolist(),
        "central_width": "(q84-q16)/2, recomputed within each draw",
        "uncertainty": "event bootstrap SD ddof=1 and linear percentile CI95; independent events assumed",
        "undefined_draws": "empty per-variable draws excluded and counted; arithmetic failures counted separately and suppress errors; intervals conditional on nonempty draws",
        "null_errors": "fewer than two effective events/valid draws, degenerate bootstrap statistic, or arithmetic failure",
        "scope": "fixed shared-straw matches, finite matched-track B-minus-A differences only; no native, truth, or unconditional residual closure claim; no model/calibration uncertainty",
        "variables": variables,
    }
