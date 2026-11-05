"""Saved-only SGS gap/reco overlap diagnostic; no transport, correction or truth reassociation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from trkfastsim.validation import cecontract, cereco, fieldparents, fieldquantiles

KEY = cecontract.KEY
ARMS = ("g4_all", "g4_clock")
IDENTITY = KEY + ["sp", "sid", "originalrowindex"]


def _strict(value):
    """Convert unavailable cores and open bin edges to strict JSON null."""
    if isinstance(value, dict):
        return {k: _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_strict(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _uint(text):
    if not re.fullmatch(r"[0-9]+", text) or int(text) > np.iinfo(np.uint64).max:
        raise ValueError("invalid exact uint64 identity")
    return np.uint64(int(text))


def _read(path, identities):
    d = pd.read_csv(path, converters={k: _uint for k in identities})
    if not set(identities).issubset(d):
        raise ValueError("missing identity columns")
    for k in identities:
        d[k] = d[k].astype(np.uint64)
    return d


def _receipts(report):
    paths = {}
    for section in ("inputs", "outputs"):
        receipts = report.get(section)
        if not isinstance(receipts, dict) or not receipts:
            raise ValueError(f"missing void-report {section} receipts")
        for label, receipt in receipts.items():
            path = Path(receipt["path"])
            if not path.is_absolute() or str(path.resolve()) != receipt["path"]:
                raise ValueError("receipt path must be canonical")
            expected = receipt["sha256_before"] if section == "inputs" else receipt["sha256"]
            if section == "inputs" and expected != receipt["sha256_after"]:
                raise ValueError("void-report input before/after hashes differ")
            if _sha(path) != expected:
                raise ValueError(f"void-report hash mismatch: {section}/{label}")
            paths[f"void/{section}/{label}"] = path
    return paths


def _check(paths, before):
    after = {name: _sha(path) for name, path in paths.items()}
    if after != before:
        raise ValueError("input changed during gap/reco diagnostic")
    return after


def read_outliers(path, arm_report):
    """Read every retained outlier with exact identities and reject malformed payloads."""
    d = _read(path, IDENTITY)
    required = {"raw_shortfall_mm", "residual_mm"}
    if not required.issubset(d):
        raise ValueError("missing outlier shortfalls")
    for k in d:
        if not pd.api.types.is_numeric_dtype(d[k]) or not np.isfinite(d[k].to_numpy()).all():
            raise ValueError(f"nonfinite or nonnumeric outlier payload: {k}")
    if d.duplicated(IDENTITY).any() or d.originalrowindex.duplicated().any():
        raise ValueError("duplicate saved outlier identity/row index")
    raw, residual = d.raw_shortfall_mm > 1., d.residual_mm > 1.
    if not (raw | residual).all():
        raise ValueError("retained outlier does not exceed strict >1 mm threshold")
    if len(d) != arm_report["all_outlier_rows"]:
        raise ValueError("not ALL retained outlier rows")
    if "saved_rows" in arm_report and (d.originalrowindex >= arm_report["saved_rows"]).any():
        raise ValueError("outlier row index outside saved arm")
    if "raw_short_by1_rows" in arm_report and int(raw.sum()) != arm_report["raw_short_by1_rows"]:
        raise ValueError("raw outlier count disagrees with void report")
    if "raw_short_by1_with_residual_by1_rows" in arm_report and int((raw & residual).sum()) != arm_report["raw_short_by1_with_residual_by1_rows"]:
        raise ValueError("residual outlier count disagrees with void report")
    return d


def event_flags(outliers):
    d = outliers[KEY].copy()
    d["outlier_rows"] = np.ones(len(d), dtype=np.int64)
    for label, column in (("raw", "raw_shortfall_mm"), ("residual", "residual_mm")):
        d[f"{label}_by1_rows"] = (outliers[column] > 1.).astype(np.int64)
        d[f"{label}_positive_shortfall_sum_mm"] = outliers[column].clip(lower=0)
    return d.groupby(KEY, sort=False, observed=True).sum().reset_index()


def _gates(d):
    ia, ir = (d[f"{arm}_ntrk"].fillna(0).to_numpy() > 0 for arm in ("fast", "g4"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = (d.fast_p_reco - d.fast_p_first_own) / d.fast_p_first_own
        rr = (d.g4_p_reco - d.g4_p_first_own) / d.g4_p_first_own
    paired = ia & ir & (d.fast_sid == d.g4_sid) & np.isfinite(ra) & np.isfinite(rr)
    return ia, ir, np.asarray(paired)


def _partition(mask, flag):
    return {"all": int(mask.sum()), "flagged": int((mask & flag).sum()),
            "unflagged": int((mask & ~flag).sum())}


def overlap(d, flag):
    ia, ir, paired = _gates(d)
    in_bin = ((d.p_first >= 2000.) & (d.p_first < 5000.)).to_numpy()
    a, b = in_bin & paired, in_bin & flag
    union = a | b
    inclusion = int(a.sum()) + int(b.sum()) - int((a & b).sum())
    if inclusion != int(union.sum()):
        raise ValueError("inclusion/exclusion mismatch")
    return {"events": _partition(np.ones(len(d), dtype=bool), flag),
            "paired_own_proxy": _partition(paired, flag),
            "gained": _partition(ia & ~ir, flag), "lost": _partition(~ia & ir, flag),
            "pilot_bin2_2_to_5_GeV": {
                "reference_p_first_MeV": {"lo_inclusive": 2000., "hi_exclusive": 5000.},
                "events": int(in_bin.sum()), "paired_own_proxy": int(a.sum()),
                "largegap_events": int(b.sum()), "intersection": int((a & b).sum()),
                "union": int(union.sum()), "inclusion_exclusion": inclusion,
                "paired_without_largegap": int((a & ~b).sum()),
                "gained": _partition(in_bin & ia & ~ir, flag),
                "lost": _partition(in_bin & ~ia & ir, flag)}}


def _metrics(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fieldparents.metrics(d)


def report(baseline, outliers):
    """Attach flags without changing baseline order, denominator or saved first-good-track association."""
    cecontract.unique(baseline, "baseline events")
    if not set(fieldparents.REQUIRED).issubset(baseline):
        raise ValueError("incomplete baseline reco metrics")
    original = baseline.copy(deep=True)
    frozen = _strict(_metrics(baseline))
    bootstrap = fieldquantiles.report(baseline)
    d, outside = baseline.copy(deep=True), {}
    for arm in ARMS:
        grouped = event_flags(outliers[arm])
        columns = [c for c in grouped if c not in KEY]
        renamed = {c: f"{arm}_{c}" for c in columns}
        if set(renamed.values()) & set(d) or f"{arm}_largegap" in d:
            raise ValueError("reserved diagnostic columns in baseline")
        membership = grouped.merge(baseline[KEY], on=KEY, how="left", indicator=True, validate="one_to_one")
        foreign = membership._merge == "left_only"
        outside[arm] = {"outlier_events_total": len(grouped), "outlier_events_outside_baseline": int(foreign.sum()),
                        "outlier_rows_outside_baseline": int(membership.loc[foreign, "outlier_rows"].sum())}
        d = d.merge(grouped.rename(columns=renamed), on=KEY, how="left", sort=False, validate="one_to_one")
        for c in columns:
            name = renamed[c]
            d[name] = d[name].fillna(0)
            if c.endswith("rows"):
                d[name] = d[name].astype(np.int64)
        d[f"{arm}_largegap"] = d[f"{arm}_outlier_rows"] > 0
    clock, all_flag = (d[f"{arm}_largegap"].to_numpy(bool) for arm in ("g4_clock", "g4_all"))
    complement = baseline.loc[~clock].copy()
    complementary_metrics = _metrics(complement)
    complementary_metrics = {k: v for k, v in complementary_metrics.items() if "yield" not in k}
    pd.testing.assert_frame_equal(baseline, original, check_exact=True)
    pd.testing.assert_frame_equal(d[original.columns], original.reset_index(drop=True), check_exact=True)
    if _strict(_metrics(baseline)) != frozen:
        raise ValueError("original baseline metrics changed")
    result = {"scope": "saved-only overlap diagnostic; no physics, reco, correction or new truth association",
              "denominator": "original saved fieldreco event set, unfiltered; not entry particles or generated cosmics",
              "truth": "saved first-good-track, same SID, finite each-arm own first-straw relative residuals",
              "primary_flag": "g4_clock_largegap: any retained raw OR residual >1 mm outlier in event",
              "all_arm_flag": "g4_all_largegap is a separate diagnostic, not a selection alternative",
              "limitations": ["SGS endpoint-secant gas-gap proxy, not a physical G4 trajectory or causal attribution",
                              "Clock is paired fast exit + 1 ns, not a physical first passage",
                              "Complement is a conditional core diagnostic, not efficiency or a corrected metric",
                              "No independently supplied baseline report; baseline metrics are recomputed, not independently verified"],
              "original_baseline": {"events": len(baseline), "metrics": frozen,
                                    "relative_core_event_bootstrap": bootstrap,
                                    "metrics_unchanged_by_flagging": True, "independently_verified": False},
              "clock_complement_diagnostic_not_efficiency": {"events": len(complement),
                  "non_yield_metrics": complementary_metrics,
                  "relative_core_event_bootstrap": fieldquantiles.report(complement)},
              "overlap": {arm: overlap(d, d[f"{arm}_largegap"].to_numpy(bool)) for arm in ARMS},
              "flag_agreement_in_baseline": {"both": int((clock & all_flag).sum()),
                  "clock_only": int((clock & ~all_flag).sum()), "all_only": int((all_flag & ~clock).sum()),
                  "neither": int((~clock & ~all_flag).sum())}, "outside_baseline": outside}
    return d, _strict(result)


def run(void_report, baseline_events, outdir):
    void_report, baseline_events, outdir = Path(void_report).resolve(), Path(baseline_events).resolve(), Path(outdir)
    if outdir.exists():
        raise FileExistsError(outdir)
    paths = {"void_report": void_report, "baseline_events": baseline_events,
             "code": Path(__file__).resolve(), **{f"code/{m.__name__.split('.')[-1]}": Path(m.__file__).resolve()
                 for m in (cecontract, cereco, fieldparents, fieldquantiles)}}
    before = {k: _sha(v) for k, v in paths.items()}
    saved = json.loads(void_report.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")))
    receipts = _receipts(saved)
    paths.update(receipts)
    before.update({k: _sha(v) for k, v in receipts.items()})
    outliers = {arm: read_outliers(saved["outputs"][f"{arm}_outliers"]["path"], saved["arms"][arm]) for arm in ARMS}
    baseline = _read(baseline_events, KEY)
    flagged, result = report(baseline, outliers)
    result["void_denominator_particles_not_reco_events"] = saved.get("denominator_particles")
    _check(paths, before)
    outdir.mkdir(parents=True, exist_ok=False)
    output = outdir / "fieldgapreco_events.csv.gz"
    flagged.to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})
    after = _check(paths, before)
    _receipts(saved)
    result["inputs"] = {k: {"path": str(v.resolve()), "sha256_before": before[k], "sha256_after": after[k]} for k, v in paths.items()}
    result["outputs"] = {"events_with_flags": {"path": str(output.resolve()), "sha256": _sha(output)}}
    with (outdir / "fieldgapreco.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--void-report", required=True)
    parser.add_argument("--baseline-events", required=True)
    parser.add_argument("--outdir", required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
