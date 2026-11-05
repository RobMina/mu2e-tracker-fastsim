"""Summarize saved fast histories on identical CE births without treating seeds as new events.

Per-seed paired errors share one Geant4 baseline and are correlated. The spread of seed estimates describes these
few fast histories only; it is neither a production-population error, an independent-confirmation combination nor
an estimate of the stock response's multi-seed uncertainty floor. No preferred seed is selected.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trkfastsim.validation import ceaudit, cerootcause


def report(frames):
    if len(frames) < 2:
        raise ValueError("at least two distinct seed labels are required")
    first = None
    rows, seed_metrics, per, aligned = [], [], {}, {}
    for label, data in frames.items():
        d = ceaudit.unique(data, label)
        cols = ceaudit.KEY + [k for k in d if k.startswith("g4_")]
        if first is None:
            first = d[cols]
        else:
            pd.testing.assert_frame_equal(first, d[cols], check_exact=True)
        r = cerootcause.migrations(d)
        per[label] = r
        aligned[label] = d
        rate = r["stock_good_track"]["paired_fast_minus_g4_error_n"][0]
        cond = r["good_track_by_reference_condition"]["g4_primary_tracker_term"]["paired_fast_minus_g4_error_n"][0]
        seed_metrics.append([rate, cond])
        rows.append(d.chain_track.to_numpy(bool))
    decisions = np.stack(rows)
    counts = decisions.sum(axis=0)
    vals = np.asarray(seed_metrics, float)
    cross = {}
    for ref_label, ref in aligned.items():
        cross[ref_label] = {}
        for label, candidate in aligned.items():
            if label == ref_label:
                continue
            cross[ref_label][label] = {
                "all_events": cerootcause.migration(ref.chain_track, candidate.chain_track),
                "reference_fast_tracker_term": cerootcause.migration(
                    ref.loc[ref.chain_tracker_term, "chain_track"], candidate.loc[ref.chain_tracker_term, "chain_track"]),
                "candidate_fast_tracker_term": cerootcause.migration(
                    ref.loc[candidate.chain_tracker_term, "chain_track"], candidate.loc[candidate.chain_tracker_term, "chain_track"])}
    return {"events": len(first), "seed_labels": list(frames), "fast_fast_history_controls": cross, "seed_metrics": ["all_event_yield_difference", "g4_tracker_term_conditional_yield_difference"],
            "per_seed": per, "descriptive_seed_mean": vals.mean(axis=0).tolist(),
            "descriptive_seed_min": vals.min(axis=0).tolist(), "descriptive_seed_max": vals.max(axis=0).tolist(),
            "descriptive_seed_sd": vals.std(axis=0, ddof=1).tolist(),
            "fast_track_count_across_seeds": {int(n): int((counts == n).sum()) for n in range(len(frames) + 1)},
            "warning": "same births and G4 baseline; no inverse-variance independent-seed combination or new-event denominator"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="SEED=rootcause/events.csv.gz")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    frames = {}
    for spec in a.inputs:
        label, path = spec.split("=", 1)
        if label in frames:
            ap.error(f"repeated seed label: {label}")
        frames[label] = pd.read_csv(path)
    result = report(frames)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ceseeds.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
