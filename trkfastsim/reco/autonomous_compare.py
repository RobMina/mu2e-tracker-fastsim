"""All-event, identity-paired EventNtuple yields; no reconstruction or fit selection.

Run ``python3 -m trkfastsim.reco.autonomous_compare FAST.root STOCK.root
--out NEW.json``. A is stock/reference and B is fast/candidate throughout.
Counts include every stored track (even failed fits if the writer stored them).
An event "with fit" means an event with at least one stored track, not a
quality-selected event. Zero-track and multi-track events remain in the denominator.

Event identity suffices for paired *counts*, including fast-vs-stock digis.
Shared-straw matching does NOT establish hit identity across different digi
productions. Only --same-digis enables ntcompare's matched-track diagnostics;
it is a caller assertion of common digis, not an inferred or verified fact.
Matched TT_Outer parameter differences also receive event-cluster bootstrap
SEs and percentile CI95 (linear/type-7 quantiles); the full common population,
including unmatched events, is resampled. These conditional diagnostics are not
native, truth, or unconditional residual closure. See autonomous_stats for the
finite-value, insufficient-event and degenerate-bootstrap conventions.
Neither mode certifies that upstream files contain all generated events: only
that the two supplied EventNtuples contain the identical unique event population.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .autonomous_stats import (DEFAULT_REPLICATES, DEFAULT_SEED,
                               event_cluster_bootstrap, linear_quantiles, validate_bootstrap)

TREE = "EventNtuple/ntuple"
PARAMETERS = ("d0", "phi0", "z0", "theta", "t0", "mom")


def _keys(keys, arm):
    if np.ma.isMaskedArray(keys) and np.any(np.ma.getmaskarray(keys)):
        raise ValueError(f"{arm}: missing event identity")
    keys = np.asarray(keys)
    if keys.shape == (0,):
        keys = np.empty((0, 3), dtype=np.int64)
    if keys.ndim != 2 or keys.shape[1] != 3:
        raise ValueError(f"{arm}: keys must have shape (n, 3): run, subrun, event")
    if keys.size and (keys.dtype.kind not in "iu" or
                      np.any(keys > np.iinfo(np.int64).max)):
        raise ValueError(f"{arm}: identities must be exact int64 integers")
    return keys.astype(np.int64)


def _counts(counts, n, arm):
    if np.ma.isMaskedArray(counts) and np.any(np.ma.getmaskarray(counts)):
        raise ValueError(f"{arm}: missing track count")
    counts = np.asarray(counts)
    if counts.ndim != 1 or len(counts) != n:
        raise ValueError(f"{arm}: counts must have one value per event identity")
    if counts.size and (counts.dtype.kind not in "iu" or np.any(counts < 0)
                        or np.any(counts > np.iinfo(np.int64).max)):
        raise ValueError(f"{arm}: counts must be nonnegative int64 integers")
    return counts.astype(np.int64)


def _ordered(keys, counts, arm):
    keys = _keys(keys, arm)
    counts = _counts(counts, len(keys), arm)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    if np.any(np.all(keys[1:] == keys[:-1], axis=1)):
        raise ValueError(f"{arm}: duplicate run/subrun/event identity")
    return keys, counts[order]


def _yield(a, b):
    """Sample (ddof=1) covariance and event-paired delta-method uncertainties."""
    n = len(a)
    x, y = a.astype(np.float64), b.astype(np.float64)
    ma, mb = (float(x.mean()), float(y.mean())) if n else (None, None)
    ratio = mb / ma if n and ma > 0 else None
    va = vb = cov = difference_se = relative_se = None
    if n >= 2:
        dx, dy = x - ma, y - mb
        va = float(dx @ dx / (n - 1))
        vb = float(dy @ dy / (n - 1))
        cov = float(dx @ dy / (n - 1))
        dd = dy - dx
        difference_se = float(np.sqrt(dd @ dd / (n * (n - 1))))
        if ratio is not None:
            # Influence form avoids cancellation for nearly identical arms.
            influence = (dy - ratio * dx) / ma
            relative_se = float(np.sqrt(influence @ influence / (n * (n - 1))))
    return {
        "A_total": int(np.sum(a, dtype=np.uint64)),
        "B_total": int(np.sum(b, dtype=np.uint64)),
        "A_per_event": ma, "B_per_event": mb,
        "sample_variance_A": va, "sample_variance_B": vb,
        "sample_covariance_AB": cov,
        "difference_B_minus_A": mb - ma if n else None,
        "difference_se": difference_se,
        "relative_yield_B_over_A_minus_1": ratio - 1 if ratio is not None else None,
        "relative_yield_se": relative_se,
    }


def _multiplicity(counts):
    bins = np.bincount(np.minimum(counts, 2), minlength=3)
    return {"counts_0_1_2plus": bins.tolist(),
            "fractions_0_1_2plus": (bins / len(counts)).tolist() if len(counts) else [None] * 3}


def summarize_counts(keys_a, counts_a, keys_b, counts_b):
    """Return JSON-ready all-event yields from numpy-compatible arrays.

    Keys are integer arrays of shape (n, 3), columns run/subrun/event; counts
    are nonnegative integer arrays of shape (n,). Rows may be reordered, but
    duplicate identities or unequal populations are errors, never intersected
    away. Empty arrays are supported. A is the reference denominator; all
    differences are B minus A and relative yields are B/A - 1 (not percent).

    The uncertainty sampling unit is an event, including zeros and all track
    multiplicities. Both observables use paired sample covariance (ddof=1).
    For means a,b and r=b/a, relative SE squared is
    (s_BB + r*r*s_AA - 2*r*s_AB)/(n*a*a). Zero reference yield gives a null
    relative yield/SE. Empty means and all SEs for n<2 are null. Events are
    assumed independent; no Poisson/independent-arm error approximation is used.
    """
    ka, a = _ordered(keys_a, counts_a, "A")
    kb, b = _ordered(keys_b, counts_b, "B")
    if not np.array_equal(ka, kb):
        raise ValueError("identical run/subrun/event populations required; missing or extra identities")
    contingency = np.bincount(3 * np.minimum(a, 2) + np.minimum(b, 2), minlength=9).reshape(3, 3)
    return {
        "events": len(a),
        "population": "identical unique run/subrun/event identities; all input events",
        "track_selection": "all stored tracks; no goodfit, status, hit or multiplicity cut",
        "uncertainty": "paired events; sample covariance ddof=1; relative-yield delta method",
        "tracks": _yield(a, b),
        "events_with_fit": _yield(a > 0, b > 0),
        "multiplicity": {"A": _multiplicity(a), "B": _multiplicity(b)},
        "contingency_A_rows_B_columns_0_1_2plus": contingency.tolist(),
    }


def load_counts(path):
    """Read only identity and track multiplicity, not hit or crossing arrays."""
    import awkward as ak
    import uproot

    with uproot.open(path) as root:
        tree = root[TREE]
        ev = tree["evtinfo"].array()
        if not {"run", "subrun", "event"}.issubset(ev.fields):
            raise ValueError(f"{path}: full run/subrun/event identity required")
        columns = [ak.to_numpy(ev[k]) for k in ("run", "subrun", "event")]
        if any(np.ma.isMaskedArray(c) and np.any(np.ma.getmaskarray(c)) for c in columns):
            raise ValueError(f"{path}: missing event identity")
        keys = np.stack(columns, axis=1)
        # Missing trk is a malformed input, not evidence of zero tracks.
        trk = tree["trk"].array()
        fields = [f for f in trk.fields if f != "trk"]  # skip fast writer's split alias
        if not fields:
            raise ValueError(f"{path}: no per-track fields in trk")
        counts = ak.to_numpy(ak.num(trk[fields[0]], axis=1))
        if len(keys) != tree.num_entries or len(counts) != tree.num_entries:
            raise ValueError(f"{path}: identity/track arrays do not cover all tree entries")
    return keys, counts


def _matched_parameters(data, indices, tracks, *, parameters=PARAMETERS):
    """Earliest TT_Outer parameters, preserving absent vs nonfinite values.

    The reference-crossing convention is ntcompare._ref_seg's sid=4, earliest
    time, without an inbounds cut. Missing branches/fields/crossings/options are
    masked, including the writer sentinel -9999; present NaN/inf values remain
    nonfinite, not silently imputed. Missing per-track lists do not shift owners.
    """
    import awkward as ak
    from . import ntcompare

    ntracks = int(ntcompare._ntrk(data)[indices].sum())
    out = {k: np.ma.masked_all(ntracks, dtype=np.float64) for k in parameters}
    if ntracks and data["trksegs"] is not None and data["trksegpars_kl"] is not None:
        segments = data["trksegs"][indices]
        pars = data["trksegpars_kl"][indices]
        expected = ntcompare._ntrk(data)[indices]
        for branch in (segments, pars):
            actual = ak.to_numpy(ak.num(branch, axis=1))
            if np.any(np.ma.getmaskarray(actual)) or not np.array_equal(actual, expected):
                raise ValueError("trksegpars_kl must be parallel to trksegs and trk")
        segments, pars = ak.flatten(segments, axis=1), ak.flatten(pars, axis=1)
        known_s = ~ak.to_numpy(ak.is_none(segments, axis=0))
        known_p = ~ak.to_numpy(ak.is_none(pars, axis=0))
        segments = ak.fill_none(segments, [], axis=0)
        pars = ak.fill_none(pars, [], axis=0)
        counts = ak.to_numpy(ak.num(segments, axis=1))
        pcounts = ak.to_numpy(ak.num(pars, axis=1))
        known = known_s & known_p
        if np.any(known & (counts != pcounts)):
            raise ValueError("trksegpars_kl must be parallel to trksegs and trk")
        # Drop both lists for an unknown pair, retaining original flat-track IDs.
        owner = np.repeat(np.flatnonzero(known), counts[known])
        sf, pf = ak.flatten(segments[known]), ak.flatten(pars[known])
        if not len(owner):
            return {k: v[tracks] for k, v in out.items()}
        selected = np.flatnonzero(ak.to_numpy(ak.fill_none(sf["sid"] == ntcompare.PAR_SID, False)))
        times = ak.to_numpy(ak.fill_none(sf["time"], np.nan))
        order = np.lexsort((times[selected], owner[selected]))
        selected = selected[order]
        if len(selected):
            selected = selected[np.r_[True, owner[selected][1:] != owner[selected][:-1]]]
            for k in parameters:
                if k in pf.fields:
                    values = np.ma.asarray(ak.to_numpy(pf[k]), dtype=np.float64)[selected]
                    out[k][owner[selected]] = np.ma.masked_equal(values, -9999.0)
    return {k: v[tracks] for k, v in out.items()}


def _matched_bootstrap(a, b, *, replicates, seed, comparison=None):
    from . import ntcompare

    # Canonical full identities align tracks and make seeded draws invariant to event ordering.
    summarize_counts(a["keys"], ntcompare._ntrk(a), b["keys"], ntcompare._ntrk(b))
    ia = np.lexsort(a["keys"].T[::-1])
    ib = np.lexsort(b["keys"].T[::-1])
    if len(ia):
        ta, tb, _ = ntcompare.match_tracks(a, b, ia, ib)
    else:
        ta = tb = np.empty(0, dtype=np.int64)
    owners = np.repeat(np.arange(len(ia)), ntcompare._ntrk(a)[ia])[ta]
    fields = PARAMETERS + tuple(k + "err" for k in PARAMETERS)
    pa = _matched_parameters(a, ia, ta, parameters=fields)
    pb = _matched_parameters(b, ib, tb, parameters=fields)
    differences = {}
    for k in PARAMETERS:
        missing = np.ma.getmaskarray(pa[k]) | np.ma.getmaskarray(pb[k])
        with np.errstate(invalid="ignore", over="ignore"):
            delta = pb[k].filled(np.nan) - pa[k].filled(np.nan)
            if k == "phi0":
                delta = (delta + np.pi) % (2 * np.pi) - np.pi
        differences[k] = np.ma.array(delta, mask=missing)
    if comparison is not None:
        descriptive = {}
        for k in PARAMETERS:
            error = pa[k + "err"].filled(np.nan)
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                pull = differences[k] / np.where(error > 0, error, np.nan)
            descriptive[k] = {"A": _parameter_summary(pa[k], ntcompare.QS),
                              "B": _parameter_summary(pb[k], ntcompare.QS),
                              "B_minus_A": _parameter_summary(differences[k], ntcompare.QS),
                              "pull_by_A_err": _parameter_summary(pull, ntcompare.QS)}
        comparison["matched_pars_at_TT_Outer_entry"] = descriptive
    result = event_cluster_bootstrap(a["keys"][ia], owners, differences,
                                     replicates=replicates, seed=seed)
    result["reference"] = "earliest TT_Outer crossing (sid=4), no inbounds cut; phi0 wrapped to [-pi,pi)"
    return result


def _parameter_summary(values, probabilities):
    values = np.ma.asarray(values).compressed()
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    scale = np.max(np.abs(values))
    mean = float(np.mean(values / scale) * scale) if scale else 0.0
    return {"n": len(values), "mean": mean,
            "q": linear_quantiles(values, probabilities).tolist()}


def _descriptive_input(data):
    """Keep legacy hit/segment diagnostics, but never its raw parameter path."""
    import awkward as ak

    segments = data["trksegs"]
    if segments is not None:
        segments = ak.fill_none(segments, [], axis=1)
        if not segments.fields:  # entirely absent lists have no record schema
            segments = None
    return {**data, "trksegs": segments, "trksegpars_kl": None}


def evaluate(fast, stock, *, same_digis=False,
             bootstrap_replicates=DEFAULT_REPLICATES, bootstrap_seed=DEFAULT_SEED):
    """Evaluate files without invoking reconstruction or conditioning on stock fits."""
    validate_bootstrap(bootstrap_replicates, bootstrap_seed)
    import hashlib
    def hashes():
        values = {}
        for path in (stock, fast):
            with open(path, 'rb') as stream:
                values[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return values
    before = hashes()
    ka, ca = load_counts(stock)
    kb, cb = load_counts(fast)
    result = summarize_counts(ka, ca, kb, cb)
    result["inputs"] = {"A_stock": str(stock), "B_fast": str(fast)}
    result["shared_hit_diagnostics"] = {
        "enabled": bool(same_digis),
        "same_digis_asserted_by_caller": bool(same_digis),
        "note": ("Shared-straw matching requires common digi/hit identity; event-paired counts do not. "
                 "No hit-identity or track-match claim is made in counts-only mode."),
    }
    if same_digis:
        from . import ntcompare

        a, b = ntcompare.load(str(stock)), ntcompare.load(str(fast))
        if a["trkhits"] is None or b["trkhits"] is None:
            raise ValueError("--same-digis diagnostics require trkhits in both inputs")
        result["shared_hit_diagnostics"]["quantile_probabilities"] = list(ntcompare.QS)
        parameter_comparison = {}
        bootstrap = _matched_bootstrap(
            a, b, replicates=bootstrap_replicates, seed=bootstrap_seed,
            comparison=parameter_comparison)
        # Share masked parameter extraction between descriptive diagnostics and the bootstrap.
        with np.errstate(invalid="ignore", over="ignore"):
            comparison = ntcompare.compare(_descriptive_input(a), _descriptive_input(b))
        from .ntschema import signature
        for branch in ("trksegs", "trksegpars_kl"):
            x, y = a[branch], b[branch]
            comparison["schema_identical"][branch] = (
                None if x is None or y is None else
                signature(x.layout.form.to_dict()) == signature(y.layout.form.to_dict()))
        comparison.update(parameter_comparison)
        result["shared_hit_diagnostics"]["comparison"] = comparison
        result["shared_hit_diagnostics"]["matched_parameter_bootstrap"] = bootstrap
    if before != hashes():
        raise RuntimeError('comparison input changed during evaluation')
    result['input_sha256'] = before
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fast", help="candidate EventNtuple")
    parser.add_argument("stock", help="reference EventNtuple")
    parser.add_argument("--out", required=True, type=Path, help="new JSON file; never overwritten")
    parser.add_argument("--same-digis", action="store_true",
                        help="assert common digi/hit identity and enable shared-hit matched residual diagnostics")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_REPLICATES,
                        help="event bootstrap draws (>=2; default: %(default)s); used only with --same-digis")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED,
                        help="PCG64 seed (uint64; default: %(default)s); used only with --same-digis")
    args = parser.parse_args(argv)
    try:
        validate_bootstrap(args.bootstrap_replicates, args.bootstrap_seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"refusing to overwrite {args.out}")
    try:
        result = evaluate(args.fast, args.stock, same_digis=args.same_digis,
                          bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        # Exclusive creation also protects against a file appearing during evaluation.
        with args.out.open("x") as out:
            out.write(payload)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.out}: {result['events']} paired events (A=stock, B=fast)")
    return result


if __name__ == "__main__":
    main()
