"""NumPy-only, stock-conditioned seed-population diagnostics, never closure certificates.

``validate(payload)`` checks an in-memory array mapping without coercing its types.
``summary(payload, baseline=None)`` returns JSON-safe counts and binomial errors.
Required arrays are keys[N,4] (run/subrun/event/seed), good[N], stockok[N],
kl[N,6], stock_params[N,6], stock_errors[N,6]. KL coordinates, already evaluated
at fitter t0 by the caller, are d0, phi0, z0, theta, t0, mom. Extra numeric
output arrays must have leading dimension N; stat, if present, is integer[N,S].

Acceptance uses every supplied eligible row; eligibility completeness must be
established by the caller. Fidelity uses both-good rows, including nonfinite
outcomes and invalid stock errors as failures. A baseline must contain exactly
the same full-key census and unchanged stock references (matching NaNs allowed).
Paired fidelity uses stockok & candidate.good & baseline.good, never a separate
finite-only population. No native history, autonomous seeding or fit closure is
inferred from these diagnostics.
"""
from collections.abc import Mapping

import numpy as np

KL = ("d0", "phi0", "z0", "theta", "t0", "mom")
REQUIRED = ("keys", "good", "stockok", "kl", "stock_params", "stock_errors")
IDENTITY = {"run": 0, "subrun": 1, "event": 2, "seed": 3, "iseed": 3}
ACCEPTANCE_TARGET = 0.995
FIDELITY_TARGET = 0.99


def validate(payload):
    """Return N, rejecting malformed arrays/identities but retaining NaN/inf outputs."""
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an array mapping")
    missing = set(REQUIRED).difference(payload)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    for name, value in payload.items():
        if not isinstance(name, str):
            raise ValueError("field names must be strings")
        if not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value):
            raise ValueError(f"{name}: expected an unmasked NumPy array")
        if value.dtype.kind not in "biufc":
            raise ValueError(f"{name}: expected numeric or boolean dtype")
    keys = payload["keys"]
    if keys.ndim != 2 or keys.shape[1] != 4 or keys.dtype.kind not in "iu":
        raise ValueError("keys must be integer[N,4] full run/subrun/event/seed keys")
    if not np.isfinite(keys).all() or np.any(keys < 0):
        raise ValueError("keys must be finite and nonnegative")
    n = len(keys)
    if len(np.unique(keys, axis=0)) != n:
        raise ValueError("duplicate full keys")
    for name, value in payload.items():
        if value.ndim == 0 or value.shape[0] != n:
            raise ValueError(f"{name}: leading dimension must equal N={n}")
    for name in ("good", "stockok"):
        if payload[name].shape != (n,) or payload[name].dtype.kind != "b":
            raise ValueError(f"{name}: expected bool[N]")
    for name in ("kl", "stock_params", "stock_errors"):
        if payload[name].shape != (n, 6) or payload[name].dtype.kind != "f":
            raise ValueError(f"{name}: expected float[N,6]")
    if "stat" in payload:
        stat = payload["stat"]
        if stat.ndim != 2 or stat.dtype.kind not in "iu":
            raise ValueError("stat: expected integer[N,S]")
    for name, column in IDENTITY.items():
        if name in payload:
            value = payload[name]
            if value.shape != (n,) or value.dtype.kind not in "iu" or np.any(value < 0):
                raise ValueError(f"{name}: expected nonnegative integer[N] identity")
            if not np.array_equal(value.astype(np.uint64), keys[:, column].astype(np.uint64)):
                raise ValueError(f"{name}: inconsistent with full keys")
    return n


def _fraction(successes, n):
    if not n:
        return None, None
    fraction = successes / n
    return fraction, float(np.sqrt(fraction * (1 - fraction) / n))


def _exchange(good, reference, reference_name="stock"):
    n = len(good)
    gains = int(np.count_nonzero(good & ~reference))
    losses = int(np.count_nonzero(~good & reference))
    agreement = n - gains - losses
    fraction, se = _fraction(agreement, n)
    return {"n": n, f"{reference_name}_good": int(reference.sum()),
            f"{reference_name}_rejected": int((~reference).sum()),
            "f1_good": int(good.sum()), "f1_rejected": int((~good).sum()),
            "both_good": int((good & reference).sum()), "gains": gains, "losses": losses,
            "neither": int((~good & ~reference).sum()), "net": gains - losses,
            "agreement": agreement, "acceptance_agreement": fraction,
            "binomial_se": se, "acceptance_agreement_target": ACCEPTANCE_TARGET}


def _pulls(left, right, errors):
    dtype = np.result_type(left.dtype, right.dtype, errors.dtype, np.float64)
    with np.errstate(all="ignore"):
        delta = left.astype(dtype) - right.astype(dtype)
        delta[:, 1] = (delta[:, 1] + np.pi) % (2 * np.pi) - np.pi
        valid = np.isfinite(errors) & (errors > 0)
        return delta / np.where(valid, errors, np.nan)


def _fidelity(left, right, errors, mask, population):
    n = int(mask.sum())
    pulls = _pulls(left, right, errors)[mask]
    finite = np.isfinite(pulls)
    within = finite & (np.abs(pulls) < 0.1)
    out = {"n_tracks": n, "population": population,
           "target_frac_within_0p1": FIDELITY_TARGET,
           "nonfinite_policy": "fail; retained in the population denominator"}
    for column, name in enumerate(KL):
        count = int(within[:, column].sum())
        nfinite = int(finite[:, column].sum())
        fraction, se = _fraction(count, n)
        out[name] = {"n": n, "n_finite": nfinite, "n_nonfinite": n - nfinite,
                     "within_0p1": count, "frac_within_0p1": fraction,
                     "binomial_se": se, "target_frac_within_0p1": FIDELITY_TARGET}
    return out


def _populations(payload):
    good, stock = payload["good"], payload["stockok"]
    return {"all_eligible": np.ones(len(good), dtype=bool), "stock_good": stock,
            "stock_rejected": ~stock, "f1_good": good, "f1_rejected": ~good,
            "both_good": stock & good, "gains": ~stock & good,
            "losses": stock & ~good, "neither": ~stock & ~good}


def _nonfinite(payload, populations):
    out = {}
    for name, value in payload.items():
        if name == "keys" or name in IDENTITY:
            continue
        bad = ~np.isfinite(value)
        bad_rows = np.any(bad, axis=tuple(range(1, bad.ndim)))
        out[name] = {"n_elements": int(value.size), "n_nonfinite": int(bad.sum()),
                     "n_rows_nonfinite": int(bad_rows.sum()),
                     "by_population": {
                         population: {"n_rows": int(mask.sum()),
                                      "n_nonfinite": int(bad[mask].sum()),
                                      "n_rows_nonfinite": int(bad_rows[mask].sum())}
                         for population, mask in populations.items()}}
    return out


def _status_counts(stat, populations):
    codes, inverse = np.unique(stat, return_inverse=True)
    n, stages = stat.shape
    encoded = inverse.reshape(n, stages) + np.arange(stages) * len(codes)
    out = {"codes": codes.tolist(), "n_stages": stages, "populations": {}}
    for population, mask in populations.items():
        counts = np.bincount(encoded[mask].ravel(), minlength=stages * len(codes))
        counts = counts.reshape(stages, len(codes))
        final = counts[-1] if stages else np.zeros(len(codes), dtype=np.int64)
        out["populations"][population] = {
            "n": int(mask.sum()), "by_stage": counts.tolist(),
            "final": dict(zip(codes.astype(str).tolist(), final.tolist()))}
    return out


def _baseline_rows(payload, baseline):
    if validate(baseline) != len(payload["keys"]):
        raise ValueError("baseline full-key census differs: missing or extra keys")
    keys = payload["keys"].astype(np.uint64)
    bkeys = baseline["keys"].astype(np.uint64)
    order = np.lexsort(keys.T[::-1])
    border = np.lexsort(bkeys.T[::-1])
    if not np.array_equal(keys[order], bkeys[border]):
        raise ValueError("baseline full-key census differs: missing or extra keys")
    rows = np.empty(len(keys), dtype=np.intp)
    rows[order] = border
    for name in ("stockok", "stock_params", "stock_errors"):
        if not np.array_equal(payload[name], baseline[name][rows], equal_nan=True):
            raise ValueError(f"baseline stock reference changed: {name}")
    return rows


def summary(payload, baseline=None):
    """Summarize all eligible outcomes and optional full-key-aligned paired changes.

    ``acceptance`` uses all N rows; ``fidelity`` uses both accepted masks.
    ``summary_all_output_nonfinite`` counts every nonidentity array, including
    rejected rows. ``status_counts`` supplies a code axis and stage-count matrix
    for each population. Baseline ``common_both_good`` reports both arms against
    stock and candidate-minus-baseline in stock-sigma units on one shared mask.
    """
    n = validate(payload)
    rows = _baseline_rows(payload, baseline) if baseline is not None else None
    populations = _populations(payload)
    out = {"n": n, "scope": "supplied eligible stock-conditioned population; not unconditional efficiency",
           "full_fit_closure": False, "full_f1_closure": False, "full_numerical_validation": False,
           "acceptance": _exchange(payload["good"], payload["stockok"]),
           "fidelity": _fidelity(payload["kl"], payload["stock_params"], payload["stock_errors"],
                                  populations["both_good"], "stock and candidate both good"),
           "summary_all_output_nonfinite": _nonfinite(payload, populations)}
    if "stat" in payload:
        out["status_counts"] = _status_counts(payload["stat"], populations)
    if baseline is not None:
        bgood, bkl = baseline["good"][rows], baseline["kl"][rows]
        common = populations["both_good"] & bgood
        stock, errors = payload["stock_params"], payload["stock_errors"]
        population = "stock, candidate and baseline all good"
        out["baseline"] = {
            "stock_reference_exact": True,
            "acceptance": _exchange(payload["good"], bgood, "baseline"),
            "common_both_good": {
                "n_tracks": int(common.sum()),
                "candidate_vs_stock": _fidelity(payload["kl"], stock, errors, common, population),
                "baseline_vs_stock": _fidelity(bkl, stock, errors, common, population),
                "candidate_minus_baseline": _fidelity(payload["kl"], bkl, errors, common, population)},
            "summary_all_output_nonfinite": _nonfinite(baseline, _populations(baseline))}
        if "stat" in baseline:
            out["baseline"]["status_counts"] = _status_counts(baseline["stat"], _populations(baseline))
    return out
