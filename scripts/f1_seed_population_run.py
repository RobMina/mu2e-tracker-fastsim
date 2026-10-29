"""Run new component populations from a completed, immutable bounded-control receipt."""
import argparse
import json
from pathlib import Path
import re

import jax
import numpy as np

import f1_seed_population as R
from f1_seed_population import A, D, Q

EXPECTED = ("parameters", "momentum", "geometry_time", "covariance", "range")


def inherited(args, watched, report):
    A.require(re.fullmatch("[0-9a-f]{64}", args.controls_sha) is not None, "invalid controls hash")
    control = D.receipt(args.controls, args.controls_sha, watched, ("completed", "controls_passed", "hashes_unchanged", "controls_only"))
    A.require(control["mode"] == args.mode and control["arms"] == {} and control["controls"]["passed"] is True,
              "control mode or scope differs")
    A.require(control["full_f1_closure"] is False, "control closure scope differs")
    for artifact in control["artifacts"].values():
        D.watch(artifact["path"], watched, artifact["sha256"])
    report["execution"] = R.helpers(watched)
    A.require(report["execution"] == control["execution"], "control runtime differs")
    data = R.prepare(args, watched, report)
    A.require(report["keys"] == control["keys"] and report["schedule"] == control["schedule"]
              and report["baseline_metadata"] == control["baseline_metadata"]
              and report["native_time"] == control["native_time"] and report["native_seed"] == control["native_seed"],
              "prepared inputs differ from controls")
    for path, sha in watched.items():
        if path not in control["before_sha256"]:
            continue
        A.require(control["before_sha256"][path] == sha, "control source/input differs")
    report.update(controls=control["controls"], controls_receipt=str(args.controls), controls_sha256=args.controls_sha)
    return data


def run(args):
    out = Path(args.out).resolve()
    A.require(not out.is_relative_to(Path(__file__).resolve().parents[1]), "protected checkout output")
    out.mkdir(parents=True, exist_ok=False)
    watched, report = {}, dict(completed=False, controls_passed=False, full_fit_closure=False, full_f1_closure=False,
                               full_numerical_validation=False, limits=R.LIMITS+[
                                   "The completed bounded controls are inherited and reauthenticated, not refitted.",
                                   "Only five new component populations run; ordinary and combined-full population baselines are not regenerated."], arms={})
    try:
        D.watch(__file__, watched)
        path = Path(__file__).resolve()
        for function in (inherited, run, main):
            D.code_origin(function, path)
        A.require(R.ARMS == EXPECTED, "new component inventory differs")
        data = inherited(args, watched, report)
        report["controls_passed"] = True
        for arm in EXPECTED:
            R.population(data, arm, out, report)
            jax.clear_caches()
        A.require(tuple(report["arms"]) == EXPECTED, "incomplete population inventory")
        A.require(tuple(report.get("artifacts", {})) == EXPECTED, "incomplete population artifacts")
        report["completed"] = True
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", controls_passed=False)
    finally:
        end, errors = {}, []
        for path in watched:
            try:
                end[path] = A.digest(path)
            except Exception as error:
                errors.append(f"{path}: {type(error).__name__}: {error}")
        for artifact in report.get("artifacts", {}).values():
            try:
                A.require(A.digest(artifact["path"]) == artifact["sha256"], "artifact changed")
            except Exception as error:
                errors.append(f"{artifact['path']}: {type(error).__name__}: {error}")
        report.update(before_sha256=watched, end_sha256=end, hash_errors=errors,
                      hashes_unchanged=bool(watched) and watched == end and not errors)
        if not report["hashes_unchanged"]:
            report["controls_passed"] = False
        report, text = D.serialize(report)
        if "controls_passed" not in report:
            report.update(controls_passed=False, full_numerical_validation=False)
            text = json.dumps(report, indent=2, allow_nan=False)+"\n"
        with (out/"manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("out", "controls", "seed-contract", "covariance-contract", "check", "conditions", "straw-axis", "baseline"):
        parser.add_argument("--"+field, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--controls-sha", required=True)
    parser.add_argument("--mode", choices=("own", "fixed"), required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "controls_passed", "hashes_unchanged", "error")}, indent=2))
    raise SystemExit(0 if report["completed"] and report["controls_passed"] else 1)


if __name__ == "__main__":
    main()
