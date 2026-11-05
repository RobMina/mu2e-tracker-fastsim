"""Compare saved CE histories using a closed census or an existing stock-filter replay log.

Exact G4-side tracker-or-calorimeter decisions are conditioning diagnostics, not symmetric fast-filter parity:
fast primaries have no calorimeter/secondary simulation. No Geant4, transport or reconstruction is run here.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from trkfastsim.validation import ceaudit, cerootcause

KEY = ceaudit.KEY
PATTERN = (r"run:\s*(\d+)\s+subRun:\s*(\d+)\s+event:\s*(\d+)\s+"
           r"selecttrk\s*=\s*([01])\s+selectcalo\s*=\s*([01])\s+selectcrv\s*=\s*([01])")


def decisions(text, expected):
    """Require exactly one stock all-event diagnostic row per externally audited completed event."""
    rows = re.findall(PATTERN, text)
    if len(rows) != expected or text.count("selecttrk =") != expected:
        raise ValueError("stock filter diagnostic row count differs from expected complete event count")
    d = pd.DataFrame(rows, columns=KEY + ["stock_trk", "stock_calo", "stock_crv"]).astype(np.uint64)
    d = ceaudit.unique(d, "stock filter decisions")
    for k in ("stock_trk", "stock_calo", "stock_crv"):
        d[k] = d[k].astype(bool)
    d["stock_accept"] = d.stock_trk | d.stock_calo | d.stock_crv
    counts = re.findall(r"TrigReport Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)", text)
    if len(counts) != 1 or [int(k) for k in counts[0]] != [expected, int(d.stock_accept.sum()), int((~d.stock_accept).sum())]:
        raise ValueError("completed stock replay event count mismatch")
    return d


def report(frames, selection):
    first = None
    per = {}
    for label, frame in frames.items():
        d = ceaudit.unique(frame, label)
        pd.testing.assert_frame_equal(d[KEY], selection[KEY], check_exact=True)
        baseline = d[KEY + [k for k in d if k.startswith("g4_")]]
        if first is None:
            first = baseline
        else:
            pd.testing.assert_frame_equal(first, baseline, check_exact=True)
        d = cerootcause.attach(d, selection, "stock filter")
        masks = {"all_events": np.ones(len(d), bool), "stock_OR_pass": d.stock_accept,
                 "stock_OR_fail": ~d.stock_accept, "stock_tracker_pass": d.stock_trk,
                 "stock_calo_only": d.stock_calo & ~d.stock_trk, "stock_tracker_only": d.stock_trk & ~d.stock_calo,
                 "stock_both_tracker_calo": d.stock_trk & d.stock_calo}
        per[label] = {name: cerootcause.migration(d.g4_track[mask], d.chain_track[mask]) for name, mask in masks.items()}
        per[label]["all_particle_vs_primary_tracker_decisions"] = {
            "all_particle_only": int((d.stock_trk & ~d.g4_tracker_term).sum()),
            "primary_only": int((~d.stock_trk & d.g4_tracker_term).sum())}
    return {"events": len(selection), "stock_filter_counts": {k: int(selection[k].sum()) for k in selection if k not in KEY},
            "per_seed": per, "scope": "exact stock G4-side selection on this bounded unfiltered pool, not fast full-filter parity",
            "warning": "shared births/G4 across seeds; no efficiency correction or independent-seed combination"}


def exact_bool(values, label):
    """Accept native booleans or the exact CSV spellings True/False, never truthy strings."""
    def parse(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value in ("True", "False"):
            return value == "True"
        raise ValueError(f"invalid exact boolean: {label}")
    return values.map(parse).astype(bool)


def census_unique(d, label):
    """Check key dtypes before numeric comparisons, including malformed CSV/object keys."""
    if not set(KEY).issubset(d) or not all(pd.api.types.is_integer_dtype(d[k].dtype) for k in KEY):
        raise ValueError(f"invalid event identities: {label}")
    return ceaudit.unique(d, label)


def census_decisions(ledger, expected):
    """Use persisted StockSelectionPath for every generated event; invalid transport is not zero response."""
    required = KEY + ["StockSelectionPath", "PrimaryPath", "status", "overflowSimP", "nStepLimit"]
    missing = sorted(set(required) - set(ledger))
    if missing:
        raise ValueError("missing census columns: " + ", ".join(missing))
    if (isinstance(expected, (bool, np.bool_)) or not isinstance(expected, (int, np.integer))
            or expected <= 0 or len(ledger) != expected):
        raise ValueError("census requires the positive complete expected event count")
    d = census_unique(ledger, "CE census")
    for col in ("status", "overflowSimP", "nStepLimit"):
        if (not pd.api.types.is_integer_dtype(d[col].dtype)
                or pd.api.types.is_bool_dtype(d[col].dtype) or d[col].isna().any() or (d[col] != 0).any()):
            raise ValueError(f"census not physics_eligible: invalid StatusG4 {col}")
    if not exact_bool(d.PrimaryPath, "PrimaryPath").all():
        raise ValueError("census not physics_eligible: inclusive PrimaryPath must pass for every event")
    if "physics_eligible" in d and not exact_bool(d.physics_eligible, "physics_eligible").all():
        raise ValueError("census not physics_eligible")
    accepted = exact_bool(d.StockSelectionPath, "StockSelectionPath")
    if "stock_accept" in d and not exact_bool(d.stock_accept, "stock_accept").equals(accepted):
        raise ValueError("stock_accept contradicts persisted StockSelectionPath")
    selection = d[KEY].copy()
    selection["stock_accept"] = accepted
    return selection


def census_report(frames, selection):
    """Validate full IDs/common G4 before conditioning; OR components were not observed in this ledger."""
    selection = census_unique(selection, "census selection")
    selection["stock_accept"] = exact_bool(selection.stock_accept, "stock_accept")
    first = None
    checked = {}
    for label, frame in frames.items():
        if not {"g4_track", "chain_track"}.issubset(frame):
            raise ValueError("rootcause events require g4_track and chain_track")
        d = census_unique(frame, label)
        pd.testing.assert_frame_equal(d[KEY], selection[KEY], check_exact=True)
        baseline = d[KEY + [k for k in d if k.startswith("g4_")]]
        if first is None:
            first = baseline
        else:
            pd.testing.assert_frame_equal(first, baseline, check_exact=True)
        for col in ("g4_track", "chain_track", "stock_accept", "StockSelectionPath"):
            if col in d:
                d[col] = exact_bool(d[col], col)
        for col in ("stock_accept", "StockSelectionPath"):
            if col in d and not d[col].equals(selection.stock_accept):
                raise ValueError(f"{label}: {col} contradicts persisted StockSelectionPath")
        checked[label] = d
    if not checked:
        raise ValueError("at least one rootcause events frame required")
    masks = {"all_events": np.ones(len(selection), bool), "stock_OR_pass": selection.stock_accept,
             "stock_OR_fail": ~selection.stock_accept}
    return {"events": len(selection),
            "stock_filter_counts": {"stock_accept": int(selection.stock_accept.sum())},
            "per_seed": {label: {name: cerootcause.migration(d.g4_track[mask], d.chain_track[mask])
                                 for name, mask in masks.items()} for label, d in checked.items()},
            "scope": "persisted exact StockSelectionPath G4-history conditioning diagnostic on this complete inclusive pool; "
                     "not an unbiased efficiency estimate, symmetric fast-filter parity, correction model, or physics closure",
            "warning": "shared births/G4 across seeds; no efficiency correction or independent-seed combination; "
                       "tracker/calorimeter decision components were not observed"}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="SEED=rootcause/events.csv.gz")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay-log", help="existing completed stock-filter replay log")
    source.add_argument("--census-events", help="exact closed cecensus/events.csv.gz with persisted StockSelectionPath")
    ap.add_argument("--expected-events", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    if a.census_events:
        # Preserve lexical booleans instead of pandas' permissive bool inference.
        ledger = pd.read_csv(a.census_events, dtype={k: "string" for k in
                             ("StockSelectionPath", "PrimaryPath", "stock_accept", "physics_eligible")})
        sel = census_decisions(ledger, a.expected_events)
    else:
        sel = decisions(Path(a.replay_log).read_text(), a.expected_events)
    frames = {}
    for spec in a.inputs:
        label, path = spec.split("=", 1)
        if label in frames:
            ap.error(f"repeated seed label: {label}")
        kwargs = {}
        if a.census_events:
            kwargs["dtype"] = {k: "string" for k in
                               ("g4_track", "chain_track", "stock_accept", "StockSelectionPath")}
        frames[label] = pd.read_csv(path, **kwargs)
    result = census_report(frames, sel) if a.census_events else report(frames, sel)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "stock_decisions.csv.gz", index=False)
    (out / "cestockfilter.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
