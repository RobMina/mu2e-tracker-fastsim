"""Saved-only descriptive response-draw and domain summaries; no precision closure claims."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest


def summarize(response, domain):
    rows = response["rows"]
    arms = ("merged", "actual_slices", "straightened_slices")
    draws = sorted({row["draw"] for row in rows})
    if len(draws) < 2 or len(rows) != len(arms) * len(draws) or len({(r["arm"], r["draw"]) for r in rows}) != len(rows):
        raise ValueError("complete unique three-arm response draw rectangle required")
    if {r["arm"] for r in rows} != set(arms) or any(sum(r["arm"] == a for r in rows) != len(draws) for a in arms):
        raise ValueError("response arm coverage mismatch")
    moments = {}
    for arm in arms:
        chosen = [row for row in rows if row["arm"] == arm]
        moments[arm] = {}
        for metric in chosen[0]["metrics"]:
            values = np.asarray([row["metrics"][metric] for row in chosen], float)
            moments[arm][metric] = dict(mean=float(values.mean()), response_draw_mean_se=float(values.std(ddof=1) / np.sqrt(len(values))),
                                       draws=len(values), draw_values=values.tolist())
    contrasts = {}
    for candidate, reference in (("actual_slices", "merged"), ("straightened_slices", "actual_slices")):
        contrasts[f"{candidate}_minus_{reference}"] = {}
        for metric in moments[candidate]:
            a, b = moments[candidate][metric], moments[reference][metric]
            contrasts[f"{candidate}_minus_{reference}"][metric] = dict(difference=a["mean"] - b["mean"],
                independent_response_draw_mean_se=float(np.hypot(a["response_draw_mean_se"], b["response_draw_mean_se"])))
    active = domain[domain.interior_recorded].copy()
    domains = {}
    for name in ("loader_aabb", "vd_cylinder"):
        domains[name] = dict(active_lanes=len(active), active_with_any_inside=int((active[f"{name}_inside_mm"] > 0).sum()),
            active_with_outside_path=int((active[f"{name}_outside_mm"] > 0).sum()),
            no_inside_but_gas_lanes=int(((active[f"{name}_inside_mm"] == 0) & (active.gas_entries > 0)).sum()),
            total_inside_mm=float(active[f"{name}_inside_mm"].sum()), total_outside_mm=float(active[f"{name}_outside_mm"].sum()))
    return dict(scope="descriptive fixed-recipient response draw moments and saved-path domain coverage",
        generated_events=response["generated_events"], arm_moments=moments, independent_draw_contrasts=contrasts,
        domain_summary=domains, warnings=["SEs describe only response draws on this fixed 1000-event sample, not recipient or physics uncertainty",
            "four draws and 5..6 fits per trial cannot certify response equivalence", "no physical terminal/world boundary inferred from either diagnostic domain"],
        admitted_for_production=False)


def run(response_path, domain_lanes, out):
    paths = list(map(Path, (response_path, domain_lanes)))
    before = {str(p): digest(p) for p in paths}
    result = summarize(json.loads(paths[0].read_text()), pd.read_csv(paths[1]))
    result.update(input_sha256=before, source_sha256={str(Path(__file__)): digest(__file__)})
    if before != {str(p): digest(p) for p in paths}:
        raise ValueError("saved response/domain input changed")
    with Path(out).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response")
    parser.add_argument("--domain-lanes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.response, args.domain_lanes, args.out)
