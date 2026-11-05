"""Audit saved field-on injector step records versus distinct physical straws, without rerunning transport.

The saved G4 clock cut is the paired fast exit plus 1 ns, not an independent physical first-passage boundary.
Injector tables retain all entry particles but cannot recover the exiting-track subset of fieldcompare.first_pass.
Repeat records in one straw need not be distinct physical passages. These are descriptive, selected-input controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim.validation import cecontract
from trkfastsim.validation.fieldreco import _me

KEY = tables.KEY
TRACK = KEY + ["sp"]
STRAW = ["plane", "panel", "straw"]
ARMS = ("g4_all", "g4_clock", "fast")


def track_keys(d, label):
    d = cecontract.integer_keys(d, label)
    if (d.sp.isna().any() or not pd.api.types.is_integer_dtype(d.sp.dtype) or (d.sp < 0).any()):
        raise ValueError(f"invalid particle identities: {label}")
    d["sp"] = d.sp.astype(np.uint64)
    return d


def assemble(particles, arms):
    """Use the complete saved entry-particle table as denominator, with exact per-particle step counts."""
    den = track_keys(particles, "entry particles")[TRACK]
    if den.empty or den.duplicated(TRACK).any():
        raise ValueError("empty or duplicate entry particle identities")
    if set(arms) != set(ARMS):
        raise ValueError("all three saved step arms are required")
    d = den.sort_values(TRACK).reset_index(drop=True)
    for label, steps in arms.items():
        s = track_keys(steps, label)
        membership = s[TRACK].drop_duplicates().merge(den, on=TRACK, how="left", indicator=True, validate="one_to_one")
        if (membership._merge != "both").any():
            raise ValueError(f"step particle outside entry denominator: {label}")
        if (s[STRAW].isna().any().any() or (s[STRAW] < 0).any().any()
                or not all(pd.api.types.is_integer_dtype(s[k].dtype) for k in STRAW)):
            raise ValueError(f"invalid physical straw identity: {label}")
        n = s.groupby(TRACK).size().rename(f"{label}_records").reset_index()
        u = s.drop_duplicates(TRACK + STRAW).groupby(TRACK).size().rename(f"{label}_straws").reset_index()
        d = d.merge(n, on=TRACK, how="left", validate="one_to_one").merge(u, on=TRACK, how="left", validate="one_to_one")
        for metric in ("records", "straws"):
            d[f"{label}_{metric}"] = d[f"{label}_{metric}"].fillna(0).astype(np.int64)
        d[f"{label}_repeat_excess"] = d[f"{label}_records"] - d[f"{label}_straws"]
    return d


def metrics(d):
    out = {"tracks": len(d), "arms": {}, "fast_minus_g4_clock": {}}
    for label in ARMS:
        out["arms"][label] = {
            "tracks_with_records": int((d[f"{label}_records"] > 0).sum()),
            **{metric: {"total": int(d[f"{label}_{metric}"].sum()), "mean_error_n": _me(d[f"{label}_{metric}"])}
               for metric in ("records", "straws", "repeat_excess")}}
    for metric in ("records", "straws", "repeat_excess"):
        delta = d[f"fast_{metric}"] - d[f"g4_clock_{metric}"]
        reference = int(d[f"g4_clock_{metric}"].sum())
        out["fast_minus_g4_clock"][metric] = {
            "total_difference": int(delta.sum()), "paired_mean_error_n": _me(delta),
            "relative_difference": float(delta.sum() / reference) if reference else None}
    change = out["fast_minus_g4_clock"]
    if change["records"]["total_difference"] != change["straws"]["total_difference"] + change["repeat_excess"]["total_difference"]:
        raise ValueError("record deficit partition failed")
    g = d.g4_clock_records.to_numpy() > 0
    f = d.fast_records.to_numpy() > 0
    out["track_presence"] = {"both": int((g & f).sum()), "g4_only": int((g & ~f).sum()),
                              "fast_only": int((~g & f).sum()), "neither": int((~g & ~f).sum())}
    out["count_partition"] = {
        label: {"tracks": int(mask.sum()), **{
            metric: int((d.loc[mask, f"fast_{metric}"] - d.loc[mask, f"g4_clock_{metric}"]).sum())
            for metric in ("records", "straws", "repeat_excess")}}
        for label, mask in (("both", g & f), ("g4_only", g & ~f), ("fast_only", ~g & f), ("neither", ~g & ~f))}
    out["clock_removed_g4_records"] = int((d.g4_all_records - d.g4_clock_records).sum())
    return out


def summarize(parents, pilot):
    """Separate the explored pilot and confirmation parents; no histogram or physics acceptance verdict."""
    if pilot not in parents:
        raise ValueError("pilot label absent")
    all_keys = pd.concat([d[TRACK] for d in parents.values()], ignore_index=True)
    if all_keys.duplicated(TRACK).any():
        raise ValueError("overlapping particle identities across original parents")
    per = {}
    for name, d in parents.items():
        per[name] = {"all_entry_particles": metrics(d),
                     "reference_clock_step_particles": metrics(d[d.g4_clock_records > 0]),
                     "both_clock_and_fast_step_particles": metrics(d[(d.g4_clock_records > 0) & (d.fast_records > 0)])}
    confirmations = [d for name, d in parents.items() if name != pilot]
    return {"pilot": pilot, "per_parent": per,
            "confirmation_only": metrics(pd.concat(confirmations, ignore_index=True)) if confirmations else None,
            "including_pilot": metrics(pd.concat(list(parents.values()), ignore_index=True)),
            "denominator": "all saved tracker-entry particles, including zero-step particles; not fieldcompare exiting subset",
            "reference_clock": "G4 retained until paired fast exit + 1ns; not independently measured G4 first passage",
            "straw_identity": "per particle, distinct (plane,panel,straw); repeated records are not certified passages",
            "errors": "paired particle standard errors; pooled errors are not parent-cluster or family-wise gates",
            "verdict": "descriptive count-definition audit only; no correction or default promotion"}


def load(path):
    """Verify identical entry-particle identities in the three injector tables before counting."""
    path = Path(path)
    files = dict(g4_all="g4prim_field.root", g4_clock="g4fp_field.root", fast="fasttrk_field.root")
    particles, arms = None, {}
    for label, filename in files.items():
        p = track_keys(tables.read(path / filename, "particles").rename(columns={"key": "sp"}), label)
        p = p[TRACK].sort_values(TRACK).reset_index(drop=True)
        if particles is None:
            particles = p
        else:
            pd.testing.assert_frame_equal(particles, p, check_exact=True)
        arms[label] = tables.read(path / filename, "sgsteps")
    return assemble(particles, arms)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="PARENT=saved-fieldcompare-directory")
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    parents = {}
    for spec in a.inputs:
        label, path = spec.split("=", 1)
        if label in parents:
            ap.error("duplicate parent label")
        parents[label] = load(path)
    result = summarize(parents, a.pilot)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fieldmultiplicity.json").write_text(json.dumps(result, indent=2) + "\n")
    for label, d in parents.items():
        d.to_csv(out / f"parent_{list(parents).index(label):02d}.csv.gz", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
