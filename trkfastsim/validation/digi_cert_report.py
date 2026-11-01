"""Summarise persisted closure and timing receipts without accepting partial samples."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def timing(path):
    report = read(path / "timing.json")
    runs = report["runs"]
    if (not report["reproducible"] or len(runs) != 3
            or [r["repeat"] for r in runs] != [0, 1, 2]
            or len({r["events"] for r in runs}) != 1
            or len({r["steps"] for r in runs}) != 1):
        raise ValueError("need cold and two reproducible warm passes of one complete input")
    return {"device": report["device"], "events": runs[0]["events"],
            "steps": runs[0]["steps"], "digis": runs[0]["digis"],
            "cold_s": runs[0]["seconds"],
            "warm_s": [r["seconds"] for r in runs if r["repeat"] > 0],
            "digitize_cold_s": runs[0]["stats"]["digitize_s"],
            "digitize_warm_s": [r["stats"]["digitize_s"] for r in runs if r["repeat"] > 0],
            "peak_process_rss_kib": max(r["maxrss_kib"] for r in runs),
            "jit_cache_last": runs[-1].get("jit_cache"),
            "resource_tuning": report.get("resource_tuning"),
            "device_memory": runs[-1].get("device_memory")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--comparison", default="ten_comparison")
    p.add_argument("--fast-dir", default="ten_jax_cpu")
    p.add_argument("--timing-prefix", default="timing_")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    compare = read(args.run / args.comparison / "digicmp.json")
    populations = {}
    for arm, subdir in (("fast", args.fast_dir), ("ref", "ten_numpy"), ("floor", "ten_floor")):
        receipt = read(args.run / subdir / "timing.json")
        rows = receipt["runs"]
        if len(rows) != 10 or len({r["file_index"] for r in rows}) != 10:
            raise ValueError("ten complete input files required")
        populations[arm] = sum(r["events"] for r in rows)
        if sum(r["digis"] for r in rows) != compare["digis"][arm]:
            raise ValueError("persisted digi count does not match timing receipt")
    if set(populations.values()) != {compare["events"]} or not compare["event_population"]["identical_unique_keys"]:
        raise ValueError("unequal persisted populations")
    report = {"events": compare["events"], "persisted_events": populations, "digis": compare["digis"],
              "rate_changes": {k: compare[k] for k in ("digis_per_event", "panelhits_per_event")},
              "shape_flags": {stat: [h for h, r in compare["chi2ndf"].items() if r["flag"][stat]]
                              for stat in ("deck", "deck_ev", "paired")},
              "paired_chi2ndf": {h: r["fast"]["paired"][0] for h, r in compare["chi2ndf"].items()}}
    if "f2" in compare:
        report["f2"] = compare["f2"]
        for arm, row in report["f2"].items():
            row["paired_rel_diff_95pct_interval"] = [row["paired_rel_diff"] + z * row["paired_rel_err"] for z in (-1.96, 1.96)]
        progress = args.run / args.comparison / "f2_progress.jsonl"
        if progress.is_file():
            records = [json.loads(line) for line in progress.read_text().splitlines()]
            report["f2_batches"] = {}
            for arm in populations:
                rows = [r for r in records if r["arm"] == arm]
                if not rows or rows[-1]["stop"] != populations[arm]:
                    raise ValueError("incomplete F2 batching receipt")
                shapes = {}
                for row in rows:
                    for shape, count in row["shape_counts"].items():
                        shapes[shape] = shapes.get(shape, 0) + count
                report["f2_batches"][arm] = {"batches": len(rows), "shape_counts": shapes,
                    "measured_batch_wall_s": sum(r["wall_s"] for r in rows)}
    report["timing"] = {}
    for device in ("cpu", "gpu"):
        root = args.run / (args.timing_prefix + device)
        if (root / "numpy/timing.json").is_file():
            report["timing"][device] = {backend: timing(root / backend) for backend in ("jax", "numpy")}
            pair = report["timing"][device]
            report["timing"][device]["numpy_over_jax_warm"] = float(np.mean(pair["numpy"]["warm_s"]) / np.mean(pair["jax"]["warm_s"]))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
