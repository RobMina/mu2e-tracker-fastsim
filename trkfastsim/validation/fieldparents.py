"""Summarize saved cosmic parents with delete-one-parent errors and a separate pilot.

The denominator remains reference primary-step events, not generated cosmics. Momentum truth is each arm's own
first-straw proxy, not the reco surface. Parent jackknife errors describe the sampled parents and fixed physics;
this is not the shard-histogram family-wise gate, a multi-seed floor or a production-efficiency certification.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trkfastsim.validation import ceaudit, cereco, fieldquantiles
from trkfastsim.validation.fieldreco import _me


REQUIRED = ["p_first", "fast_ntrk", "g4_ntrk", "fast_sid", "g4_sid", "fast_nactive", "g4_nactive",
            "fast_p_reco", "g4_p_reco", "fast_p_first_own", "g4_p_first_own"]


def prepare(parents):
    """Reject overlapping identities even across parents; retain exact integer keys without float coercion."""
    if len(parents) < 3:
        raise ValueError("at least three distinct parents including the pilot are required")
    parts = []
    for label, frame in parents.items():
        if not label or "parent" in frame:
            raise ValueError("invalid parent label or reserved column")
        d = ceaudit.unique(frame, label)
        if d.empty or not set(REQUIRED).issubset(d):
            raise ValueError(f"empty or incomplete parent: {label}")
        parts.append(d.assign(parent=label))
    d = pd.concat(parts, ignore_index=True)
    ceaudit.unique(d, "all cosmic parents")
    return d


def metrics(d):
    """Fixed metric vector for pooled estimates and delete-one-parent replicates."""
    ia, ir = (d[f"{arm}_ntrk"].fillna(0).to_numpy() > 0 for arm in ("fast", "g4"))
    both = ia & ir
    same = both & (d.fast_sid == d.g4_sid)
    ra = (d.fast_p_reco - d.fast_p_first_own) / d.fast_p_first_own
    rr = (d.g4_p_reco - d.g4_p_first_own) / d.g4_p_first_own
    valid = same & np.isfinite(ra) & np.isfinite(rr)
    hits = (d.fast_nactive - d.g4_nactive)[both]
    out = {"yield_difference": float((ia.astype(float) - ir).mean()),
           "active_hit_difference": _me(hits)[0]}
    for i, (lo, hi) in enumerate(zip(fieldquantiles.EDGES[:-1], fieldquantiles.EDGES[1:])):
        sel = valid & (d.p_first >= lo) & (d.p_first < hi)
        delta = cereco.core(ra[sel]) - cereco.core(rr[sel])
        in_bin = (d.p_first >= lo) & (d.p_first < hi)
        out[f"bin{i}_yield_difference"] = float((ia.astype(float) - ir)[in_bin].mean()) if in_bin.any() else np.nan
        out[f"bin{i}_relative_median_difference"] = float(delta[0])
        out[f"bin{i}_relative_width_difference"] = float(delta[1])
    return out


def summary(d):
    """Event-weighted pooled estimates; independent parent is the deletion unit, not each event."""
    labels = sorted(d.parent.unique())
    pooled = metrics(d)
    leave = pd.DataFrame([metrics(d[d.parent != label]) for label in labels])
    values = leave.to_numpy(float)
    errors = np.sqrt((len(labels) - 1) / len(labels) * ((values - values.mean(axis=0)) ** 2).sum(axis=0))
    ia, ir = (d[f"{arm}_ntrk"].fillna(0).to_numpy() > 0 for arm in ("fast", "g4"))
    return {"parents": labels, "events": len(d), "good_tracks": {"fast": int(ia.sum()), "g4": int(ir.sum())},
            "gained": int((ia & ~ir).sum()), "lost": int((~ia & ir).sum()),
            "event_paired_yield_difference_error_n": _me(ia.astype(float) - ir),
            "estimates": {k: {"pooled": v, "delete_one_parent_error": float(errors[i]),
                               "delete_one_parent_estimates": leave[k].tolist()}
                          for i, (k, v) in enumerate(pooled.items())},
            "uncertainty": "delete-one-parent jackknife of event-weighted pooled metrics; parents assumed independent"}


def step_summary(reports):
    """Pool frozen first-pass step summaries by their reported track counts, never by the reco denominator."""
    labels = sorted(reports)
    if len(labels) < 2:
        raise ValueError("at least two step-report parents are required")
    rows = []
    for label in labels:
        r = reports[label]["first_pass"]
        s = r["straws_per_track"]
        n = r["tracks"]
        if n <= 0 or any(s[k][2] != n for k in ("g4", "fast", "fast_minus_g4")):
            raise ValueError(f"inconsistent first-pass track count: {label}")
        rows.append([n, s["g4"][0] * n, s["fast"][0] * n])
    rows = np.asarray(rows, float)

    def estimate(v):
        n, g, f = v.sum(axis=0)
        return np.array([(f - g) / n, (f - g) / g])

    point = estimate(rows)
    leave = np.stack([estimate(np.delete(rows, i, axis=0)) for i in range(len(rows))])
    error = np.sqrt((len(rows) - 1) / len(rows) * ((leave - leave.mean(axis=0)) ** 2).sum(axis=0))
    return {"parents": labels, "tracks": int(rows[:, 0].sum()), "g4_steps": int(round(rows[:, 1].sum())),
            "fast_steps": int(round(rows[:, 2].sum())), "difference_steps_per_track": float(point[0]),
            "relative_step_difference": float(point[1]), "delete_one_parent_errors": error.tolist(),
            "error_metrics": ["difference_steps_per_track", "relative_step_difference"],
            "denominator": "per-report exiting reference primary tracks with G4 steps; not the reco event denominator",
            "per_parent": {label: {"tracks": reports[label]["first_pass"]["tracks"],
                                     "straws_per_track": reports[label]["first_pass"]["straws_per_track"]}
                           for label in labels}}


def report(parents, pilot, steps=None):
    """Keep the explored pilot separate; do not silently call all parents independent confirmation."""
    if pilot not in parents:
        raise ValueError("pilot label is absent")
    d = prepare(parents)
    per = {}
    for label in sorted(parents):
        frame = d[d.parent == label]
        ia, ir = (frame[f"{arm}_ntrk"].fillna(0).to_numpy() > 0 for arm in ("fast", "g4"))
        per[label] = {"events": len(frame), "metrics": metrics(frame),
                      "yield_difference_error_n": _me(ia.astype(float) - ir),
                      "relative_core_event_bootstrap": fieldquantiles.report(frame.drop(columns="parent"))}
    result = {"pilot": pilot, "denominator": "reference primary first-clock-window step events; conditional on S2 selection",
            "truth": "each arm's own first-straw proxy, not the reconstruction surface",
            "gate": "descriptive metrics only; no histogram family-wise gate or default promotion",
            "units": "yield and relative cores are fractions, multiply by 100 for percentage points; hits are counts",
            "per_parent": per, "confirmation_only": summary(d[d.parent != pilot]), "including_pilot": summary(d)}
    if steps is not None:
        if set(steps) != set(parents):
            raise ValueError("step reports must cover exactly the same parent labels")
        result["steps"] = {"confirmation_only": step_summary({k: v for k, v in steps.items() if k != pilot}),
                           "including_pilot": step_summary(steps)}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="LABEL=events.csv.gz, one distinct original parent per label")
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--steps", nargs="+", help="optional LABEL=fieldcompare.json, exact parent-label coverage required")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    parents = {}
    for spec in a.inputs:
        label, path = spec.split("=", 1)
        if label in parents:
            ap.error(f"repeated parent label: {label}")
        parents[label] = pd.read_csv(path)
    steps = {} if a.steps is not None else None
    for spec in a.steps or []:
        label, path = spec.split("=", 1)
        if label in steps:
            ap.error(f"repeated step-report label: {label}")
        steps[label] = json.loads(Path(path).read_text())
    result = report(parents, a.pilot, steps)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fieldparents.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
