"""Audit saved anchor-only census results and placement counters without rerunning transport.

Usage: secondary_placement_audit.py BASE_JSON EDGE_JSON BASE_LOG EDGE_LOG --out NEW_JSON
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


COUNTERS = ("drawn", "reflected", "off_lattice", "in_muon_straw", "placed", "merged_same_particle")


def read_log(path):
    rows = []
    pending = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#   off-axis: "):
            if pending is not None:
                raise ValueError("placement counters without output path")
            pending = ast.literal_eval(line.split(": ", 1)[1])
        elif line.startswith("wrote ") and pending is not None:
            rows.append({"file": Path(line[6:]).name, **pending})
            pending = None
    if pending is not None or not rows or len({r["file"] for r in rows}) != len(rows):
        raise ValueError("missing or duplicate placement groups")
    return rows


def compare(base, edge, base_rows, edge_rows):
    if [r["file"] for r in base_rows] != [r["file"] for r in edge_rows]:
        raise ValueError("placement logs have different file coverage")
    if base["selected_events"] != edge["selected_events"]:
        raise ValueError("anchor arms have different selections")
    for name in ("mu_steps", "mu_eion"):
        if base["totals"]["fast"][name] != edge["totals"]["fast"][name]:
            raise ValueError("anchor arms have different primary denominators")
    sums = [{k: sum(r[k] for r in rows) for k in COUNTERS} for rows in (base_rows, edge_rows)]
    for s in sums:
        s["output_offaxis_rows"] = s["placed"] - s["merged_same_particle"]
        if s["drawn"] != s["placed"] + s["in_muon_straw"] + s["off_lattice"]:
            raise ValueError("nonconserving placement counters")
    changes = {}
    for name in ("location", "plane_distance"):
        a, b = base["breakdowns"][name], edge["breakdowns"][name]
        if [r["bin"] for r in a] != [r["bin"] for r in b]:
            raise ValueError("census bins differ")
        rows = []
        for x, y in zip(a, b):
            if x["steps"][1] != y["steps"][1]:
                raise ValueError("census reference counts differ")
            rows.append({"bin": x["bin"], "fast_steps": [x["steps"][0], y["steps"][0]],
                         "g4_steps": x["steps"][1], "edge_minus_base_steps": y["steps"][0] - x["steps"][0],
                         "relative_to_g4": [x["ratio_minus_1"], y["ratio_minus_1"]],
                         "file_jackknife_error": [x["err"], y["err"]]})
        changes[name] = rows
    draw_fields = ("drawn", "draw_level_counts", "distinct_with_steps", "top10_step_share")
    return {"groups": len(base_rows), "selected_events": base["selected_events"],
            "same_draw_summaries": all(a[k] == b[k] for a, b in zip(base_rows, edge_rows) for k in draw_fields),
            "counters": {"base": sums[0], "edge": sums[1], "edge_minus_base": {k: sums[1][k] - sums[0][k] for k in sums[0]}},
            "migration": changes,
            "interpretation": "descriptive differences, not a donor-to-recipient migration matrix; counters precede analysis selection",
            "uncertainty": "saved per-arm file jackknife errors; covariance of edge-minus-base not recoverable from summaries",
            "control": "same primary denominators, reference counts, file names and draw summaries; use separate ROOT identity audit"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base_json")
    p.add_argument("edge_json")
    p.add_argument("base_log")
    p.add_argument("edge_log")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    paths = [args.base_json, args.edge_json, args.base_log, args.edge_log]
    result = compare(*(json.loads(Path(f).read_text()) for f in paths[:2]), *(read_log(f) for f in paths[2:]))
    result["inputs"] = [{"path": f, "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()} for f in paths]
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with open(args.out, "x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
