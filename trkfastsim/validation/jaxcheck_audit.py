"""Audit saved jaxcheck runs; report timing and hard correctness gates without inventing a physics verdict.

Usage: python3 scripts/jaxcheck_audit.py --out NEW.json [--require-cross] RUN/jaxcheck.json [...]
Compilation, library-off scope and legacy step-statistics limitations remain explicit in the original reports.
"""
import argparse
import json
import math
from pathlib import Path


def audit(paths, require_cross=False):
    errors, rows = [], []
    if not paths:
        errors.append("no report paths")
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get("incomplete"):
            errors.append(f"{path}: incomplete checkpoint, not a finished report")
        if not report.get("files"):
            errors.append(f"{path}: no file records")
        for rec in report["files"]:
            label = f"{path}:{rec['file']}"
            if not rec.get("jax"):
                errors.append(label + ": no JAX passes")
                continue
            if rec.get("reproducible", {}).get("numpy") is False:
                errors.append(label + ": failed numpy reproducibility")
            if len(rec["jax"]) < 2 or rec.get("reproducible", {}).get("jax") is not True:
                errors.append(label + ": missing/failed JAX reproducibility")
            for backend in ("numpy", "jax"):
                for iteration, record in enumerate(rec[backend]):
                    if not record.get("stats"):
                        errors.append(f"{label}: {backend} pass {iteration} missing transport counters")
                    if backend == "jax" and "rec_overflow" not in record.get("stats", {}):
                        errors.append(f"{label}: JAX pass {iteration} missing overflow counters")
                    for name, count in record.get("stats", {}).items():
                        if ("overflow" in name or name.endswith("capped")) and count:
                            errors.append(f"{label}: {backend} pass {iteration} {name}={count}")
            cross = rec.get("cross")
            if require_cross and cross is None:
                errors.append(label + ": missing cross-device comparison")
            if cross is not None:
                for col in ("x0", "y0", "z0", "t", "eion", "len"):
                    value = cross.get("max_abs_diff_" + col)
                    if value is None or not math.isfinite(value):
                        errors.append(f"{label}: missing/nonfinite cross-device difference {col}")
                if cross["only_this"] or cross["only_cross"]:
                    errors.append(label + ": cross-device step identities differ")
                pf = cross["primaryfilter_trk"]
                if pf["this"] != pf["both"] or pf["cross"] != pf["both"]:
                    errors.append(label + ": cross-device filter decisions differ")
            row = {"file": rec["file"], "report": str(path), "backend": report["backend"],
                   "device": report["device"], "generated_events": rec.get("generated"),
                   "jax_warm_seconds": [q["timing"]["total"] for q in rec["jax"][1:]],
                   "jax_compile_inclusive_seconds": rec["jax"][0]["timing"]["total"],
                   "numpy_seconds": [q["timing"]["total"] for q in rec["numpy"]],
                   "world_warm_seconds": [q["timing"]["to_plane"] for q in rec["jax"][1:]],
                   "cross": {k: v for k, v in (cross or {}).items() if k != "stats"}}
            if rec.get("compare"):
                row["primaryfilter_trk"] = rec["compare"]["primaryfilter_trk"]
                row["event_clustered"] = rec["compare"].get("event_clustered")
            rows.append(row)
    return {"hard_gates_pass": not errors, "errors": errors, "runs": rows,
            "scope": "primary-only steps; identity/reproducibility/cap checks only; numerical magnitudes need separate review, not G4 or library-on closure"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--require-cross", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.reports, args.require_cross)
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["hard_gates_pass"] else 1)


if __name__ == "__main__":
    main()
