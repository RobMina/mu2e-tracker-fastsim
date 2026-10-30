"""Summarise measured ten-file campaign passes without runtime extrapolation."""
import argparse
import json
from pathlib import Path


def passes(path):
    data = json.loads((path / "timing.json").read_text())
    if not data["reproducible"]:
        raise ValueError("nonreproducible campaign")
    groups = {}
    for row in data["runs"]:
        groups.setdefault(row["repeat"], []).append(row)
    summaries = []
    for repeat, rows in sorted(groups.items()):
        if len(rows) != 10 or sorted(r["file_index"] for r in rows) != list(range(10)):
            raise ValueError("each measured pass must contain all ten files exactly once")
        summaries.append({"repeat": repeat, "events": sum(r["events"] for r in rows),
                          "steps": sum(r["steps"] for r in rows), "digis": sum(r["digis"] for r in rows),
                          "wall_s": sum(r["seconds"] for r in rows),
                          "digitize_s": sum(r["stats"]["digitize_s"] for r in rows),
                          "electronics_calls": sum(r["stats"].get("jax_buckets", 0) for r in rows)})
    if len(summaries) != 3:
        raise ValueError("three complete passes required")
    return summaries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--empty-cache-jax", type=Path, help="separate measured cache-fill campaign")
    args = p.parse_args(argv)
    jax = passes(args.root / "jax")
    numpy = passes(args.root / "numpy")
    for a, b in zip(jax, numpy):
        if a["events"] != b["events"] or a["steps"] != b["steps"]:
            raise ValueError("unmatched campaign populations")
    report = {"jax": jax, "numpy": numpy,
              "numpy_over_jax": [b["wall_s"] / a["wall_s"] for a, b in zip(jax, numpy)],
              "all_three_numpy_over_jax": sum(b["wall_s"] for b in numpy) / sum(a["wall_s"] for a in jax),
              "timing": "sum of measured table-to-host wall seconds; first pass includes process first call/cache load; excludes ROOT writing after digitisation"}
    if args.empty_cache_jax:
        empty = passes(args.empty_cache_jax)
        if any(a["events"] != b["events"] or a["steps"] != b["steps"] for a, b in zip(empty, numpy)):
            raise ValueError("unmatched empty-cache populations")
        report["empty_cache_jax_passes"] = empty
        report["empty_cache_first_numpy_over_jax"] = numpy[0]["wall_s"] / empty[0]["wall_s"]
        report["empty_cache_all_three_numpy_over_jax"] = sum(b["wall_s"] for b in numpy) / sum(a["wall_s"] for a in empty)
        report["empty_cache_comparison"] = "separate GPU allocations; not a same-host controlled comparison"
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
