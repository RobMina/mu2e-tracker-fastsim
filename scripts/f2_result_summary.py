"""Summarize persisted paired F2 comparison and measured batch timings; no runtime extrapolation."""
import argparse
import json
from pathlib import Path

def summarize(directory):
    p = Path(directory)
    result = json.loads((p / "digicmp.json").read_text())
    out = {"events": result["events"], "event_population": result.get("event_population"),
           "f2": result["f2"], "histogram_flags": [h for h, r in result["chi2ndf"].items() if any(r["flag"].values())]}
    log = p / "f2_progress.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    out["batch_timings"] = {}
    for arm in sorted({r["arm"] for r in records}):
        rows = [r for r in records if r["arm"] == arm]
        ranges = sorted((r["start"], r["stop"]) for r in rows)
        if not ranges or ranges[0][0] != 0 or ranges[-1][1] != result["events"]:
            raise ValueError("incomplete F2 batch coverage")
        if any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError("overlapping or missing F2 event ranges")
        shapes = {}
        for r in rows:
            for shape, count in r.get("shape_counts", {}).items():
                shapes[shape] = shapes.get(shape, 0) + count
        out["batch_timings"][arm] = {"batches": len(rows), "fit_shapes": shapes,
            **{key: float(sum(r.get(key, 0.) for r in rows)) for key in ("wall_s", "hit_s", "seed_s", "fit_s")}}
    for r in out["f2"].values():
        delta, err = r["paired_rel_diff"], r["paired_rel_err"]
        r["paired_z"] = float(delta / err) if err else None
        r["paired_95pct_normal_interval"] = [float(delta - 1.96 * err), float(delta + 1.96 * err)]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    args = p.parse_args(argv)
    print(json.dumps(summarize(args.directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
