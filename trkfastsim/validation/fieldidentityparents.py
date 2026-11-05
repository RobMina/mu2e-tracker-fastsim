"""Aggregate saved fieldidentity parents without transport, reconstruction or ID inference.

Singleton restriction is a truth-association diagnostic of the legacy event-first
proxy, not exact compressed-ID mapping, same-location resolution, physics closure,
generated efficiency or a correction. The explored pilot is never pooled with confirmation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trkfastsim.validation import fieldparents
from trkfastsim.validation.cecontract import KEY

SCOPE = ("singleton-restriction truth diagnostic of the original event-first proxy; "
         "not exact compressed-ID mapping, same-location resolution, physics closure, "
         "generated efficiency or a correction")
SUFFIXES = ("relative_median_difference", "relative_width_difference")


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_finite(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_ledger(path):
    """Parse event keys and nullable particle IDs directly as integers, never floats.

    Use the default CSV float parser just as the original fieldparents CLI did;
    no claim of binary equality to pre-serialization native ROOT floats is made.
    """
    columns = pd.read_csv(path, nrows=0).columns
    required = set(KEY + fieldparents.REQUIRED + ["primary_count", "fast_mc_id", "g4_mc_id"])
    if not required.issubset(columns):
        raise ValueError("incomplete fieldidentity ledger")
    dtype = {k: "uint64" for k in KEY}
    particle_ids = {f"{arm}_{name}" for arm in ("fast", "g4")
                    for name in ("mc_id", "mc_raw_rank0_id", "matched_id", "event_first_id")}
    dtype.update({c: "UInt64" for c in columns if c in particle_ids})
    dtype["primary_count"] = "uint64"
    return pd.read_csv(path, dtype=dtype)


def _pairs(d):
    both = (d.fast_ntrk.fillna(0) > 0) & (d.g4_ntrk.fillna(0) > 0)
    ra = (d.fast_p_reco - d.fast_p_first_own) / d.fast_p_first_own
    rr = (d.g4_p_reco - d.g4_p_first_own) / d.g4_p_first_own
    return both & (d.fast_sid == d.g4_sid) & np.isfinite(ra) & np.isfinite(rr)


def counts(d):
    pair = _pairs(d)
    single = d.primary_count == 1
    multi = d.primary_count > 1
    known = (d.fast_mc_id.notna() & d.g4_mc_id.notna()
             & (d.fast_mc_id > 0) & (d.g4_mc_id > 0))
    return {"original_reference_events": len(d), "multiprimary_events": int(multi.sum()),
            "singleton_reference_events": int(single.sum()),
            "original_proxy_pairs": int(pair.sum()),
            "singleton_proxy_pairs": int((pair & single).sum()),
            "proxy_pairs_removed": int((pair & ~single).sum()),
            "two_to_five_GeV_pairs_removed": int((pair & ~single & (d.p_first >= 2000)
                                                   & (d.p_first < 5000)).sum()),
            "same_id_link_known": int((known & (d.fast_mc_id == d.g4_mc_id)).sum())}


def _summary(d, labels):
    """Reuse the frozen pooled formulas, exposing only relative median/width metrics.

    Keep every original parent as a deletion unit even if restriction removes all
    its events. Unavailable cores/errors remain explicit JSON nulls.
    """
    if d.empty:
        return {"parents": labels, "events": 0, "proxy_pairs": 0,
                "estimates": {k: {"pooled": np.nan, "delete_one_parent_error": np.nan,
                                   "delete_one_parent_estimates": [np.nan] * len(labels)}
                              for k in fieldparents.metrics(d) if k.endswith(SUFFIXES)},
                "uncertainty": "delete-one-parent jackknife of event-weighted pooled metrics; parents assumed independent"}
    old = fieldparents.summary(d)
    estimates = {k: v for k, v in old["estimates"].items() if k.endswith(SUFFIXES)}
    if sorted(d.parent.unique()) != labels:
        leave = [fieldparents.metrics(d[d.parent != label]) for label in labels]
        for k, v in estimates.items():
            values = np.asarray([row[k] for row in leave], float)
            error = np.sqrt((len(labels) - 1) / len(labels)
                            * ((values - values.mean()) ** 2).sum())
            v.update(delete_one_parent_error=float(error), delete_one_parent_estimates=values.tolist())
    return {"parents": labels, "events": len(d), "proxy_pairs": int(_pairs(d).sum()),
            "estimates": estimates, "uncertainty": old["uncertainty"]}


def _views(d, labels):
    before = _summary(d, labels)
    after = _summary(d[d.primary_count == 1], labels)
    return {"counts": counts(d), "original_event_proxy": before,
            "single_entry_particle": after,
            "changes": {"reference_events": after["events"] - before["events"],
                        "proxy_pairs": after["proxy_pairs"] - before["proxy_pairs"],
                        "relative_core_pooled": {
                            k: after["estimates"][k]["pooled"] - v["pooled"]
                            for k, v in before["estimates"].items()}},
            "change_uncertainty": "descriptive after-minus-before, not independent-error significance"}


def report(parents, pilot):
    if pilot not in parents:
        raise ValueError("pilot label is absent")
    d = fieldparents.prepare(parents)
    if (not pd.api.types.is_integer_dtype(d.primary_count.dtype)
            or d.primary_count.isna().any() or (d.primary_count < 1).any()):
        raise ValueError("primary_count must be a positive integer")
    confirmation = d[d.parent != pilot]
    labels = sorted(k for k in parents if k != pilot)
    return _finite({"scope": SCOPE, "pilot": pilot, "actual_parent_count": len(parents),
        "actual_confirmation_parent_count": len(labels),
        "denominator": "original reference-clock primary-step event keys, conditional on saved selection; not generated cosmics",
        "truth": "each arm's original first-straw event proxy, not the reconstruction surface",
        "association": "positive persisted MC IDs only; compressed zero IDs are unknown, including for singleton events; no remapping inferred",
        "units": "relative fast-minus-g4 median and IQR/1.349 width differences are fractions, not percent",
        "binning": {"edges": list(fieldparents.fieldquantiles.EDGES),
                    "reference": "original event-first G4 p_first, unchanged by singleton restriction"},
        "float_representation": "saved CSV, default pandas float parser as in original fieldparents CLI",
        "per_parent": {label: counts(d[d.parent == label]) for label in sorted(parents)},
        "pilot_report": _views(d[d.parent == pilot], [pilot]),
        "confirmation_only": _views(confirmation, labels)})


def run(inputs, pilot, outdir):
    """Admit saved baseline receipts and bind all report/ledger bytes before/after."""
    out = Path(outdir)
    if out.exists():
        raise FileExistsError(f"outdir must be fresh: {out}")
    paths = {label: {"report": (Path(directory) / "fieldidentity.json").resolve(strict=True),
                     "ledger": (Path(directory) / "events.csv.gz").resolve(strict=True)}
             for label, directory in inputs.items()}
    before = {label: {kind: _sha256(p) for kind, p in files.items()} for label, files in paths.items()}
    parents = {}
    for label, files in paths.items():
        saved = json.loads(files["report"].read_text())
        if (not isinstance(saved, dict) or not isinstance(saved.get("baseline_check"), dict)
                or saved["baseline_check"].get("unchanged") is not True):
            raise ValueError(f"baseline_check.unchanged must be true before admission: {label}")
        parents[label] = read_ledger(files["ledger"])
        if saved.get("events") != len(parents[label]):
            raise ValueError(f"saved report/ledger event count differs: {label}")
    result = report(parents, pilot)
    after = {label: {kind: _sha256(p) for kind, p in files.items()} for label, files in paths.items()}
    if before != after:
        raise ValueError("closed input changed during parent diagnostic")
    result["inputs"] = {label: {kind: {"path": str(p), "sha256_before": before[label][kind],
                                       "sha256_after": after[label][kind]} for kind, p in files.items()}
                        for label, files in paths.items()}
    result["baseline_admission"] = {label: {"unchanged": True, "events": len(parents[label])}
                                    for label in sorted(parents)}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    out.mkdir(parents=True, exist_ok=False)
    (out / "fieldidentityparents.json").write_text(payload)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="LABEL=fieldidentity-audit-directory")
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--outdir", required=True, help="must not exist")
    a = ap.parse_args(argv)
    inputs = {}
    for spec in a.inputs:
        if "=" not in spec:
            ap.error("input must be LABEL=fieldidentity-audit-directory")
        label, directory = spec.split("=", 1)
        if not label or not directory or label in inputs:
            ap.error(f"empty or repeated parent label/path: {label}")
        inputs[label] = directory
    print(json.dumps(run(inputs, a.pilot, a.outdir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
