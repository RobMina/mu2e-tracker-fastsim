#!/usr/bin/env python3
r"""Replay only library secondaries from saved fast tables; never run transport.

Example (a contiguous subset must retain its ORIGINAL file indices)::

    python scripts/secondary_replay.py fast.007.root fast.008.root \
        --start-index 7 --seed 1 --library original.root --annotated topology.root \
        --gdml detector.gdml --outdir NEW_DIRECTORY

Inputs are explicit paths, in original order: no glob expansion or sorting is
performed. For noncontiguous subsets, invoke separately with the correct index.
All defaults are checked before ANY candidate ROOT file is written. Equality
means identical ordered column names and exact values (including row order),
not identical ROOT bytes/dtypes; dtype differences are recorded, never rounded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

# Support direct execution from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import uproot
from mu2efastsim import tables as core_tables
from trkfastsim import geometry, library, placement, steps
from trkfastsim.analyses import steps as steps_analysis

TABLES = ("sgsteps", "particles", "gen", "subruns", "surfsteps")


class ReplayIdentityError(ValueError):
    """An exact identity gate failed; no candidate is safe to write."""

    def __init__(self, message, checks):
        super().__init__(message)
        self.checks = checks


def exact_check(expected, actual):
    """JSON-ready exact values/ordered-columns check, ignoring index and dtype."""
    if expected is None or actual is None:
        return {"passed": expected is actual, "absent": expected is None}
    result = {"passed": False, "expected_rows": len(expected), "actual_rows": len(actual),
              "expected_columns": list(expected.columns), "actual_columns": list(actual.columns),
              "dtype_differences": {c: [str(expected[c].dtype), str(actual[c].dtype)]
                                    for c in expected.columns if c in actual.columns
                                    and expected[c].dtype != actual[c].dtype}}
    try:
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True),
                                      check_dtype=False, check_exact=True, check_like=False,
                                      check_names=False, check_categorical=False)
    except AssertionError as exc:
        result["error"] = str(exc)
    else:
        result["passed"] = True
    return result


def require_checks(checks, message):
    if not all(check["passed"] for check in checks.values()):
        raise ReplayIdentityError(message, checks)
    return checks


def verify_legacy_payload(original, annotated):
    """Annotations may add columns, but cannot change any legacy table payload."""
    checks = {}
    for name, frame in original.items():
        other = annotated.get(name)
        if other is not None and set(frame.columns).issubset(other.columns):
            other = other.loc[:, frame.columns]
        checks[name] = exact_check(frame, other)
    return require_checks(checks, "annotated library changed legacy payload")


def reconstruct_primary(baseline):
    """Recover the make_steps state immediately before off-axis/in-straw paste.

    Select by key, not isprim/pdg: the sampler includes every primary muon key,
    including multi-muon events. nsgs is an event count, not a particle count.
    """
    sgs = baseline["sgsteps"]
    parts = baseline["particles"]
    sgs = sgs.loc[sgs.sp < steps.SECONDARY_KEY0].copy().reset_index(drop=True)
    parts = parts.loc[parts.key < steps.SECONDARY_KEY0].copy().reset_index(drop=True)
    sgs["nsgs"] = sgs.groupby(core_tables.KEY)["sid"].transform("size")
    return sgs, parts


def repaste(baseline, lib, geom, seed, file_index, *, topology=False):
    """Pure table reconstruction; independent original RNG streams and order."""
    if seed < 0 or file_index < 0:
        raise ValueError("seed and original file index must be nonnegative")
    sgs, parts = reconstruct_primary(baseline)
    off = library.OffAxisSampler(lib, np.random.default_rng([int(seed), int(file_index), 2]),
                                 topology_anchor=topology)
    ins = library.InStrawSampler(lib, np.random.default_rng([int(seed), int(file_index), 1]))
    sgs, parts, stats = steps.add_offaxis_secondaries(sgs, parts, off, geom)
    sgs, parts = steps.add_instraw_secondaries(sgs, parts, ins, geom)
    return {**baseline, "sgsteps": sgs, "particles": parts}, stats


def validate_default(baseline, original, annotated, geom, seed, file_index):
    """Both legacy and annotated-default pastes must exactly close on saved data."""
    checks, stats = {}, {}
    for name, lib in (("original", original), ("annotated", annotated)):
        replay, stats[name] = repaste(baseline, lib, geom, seed, file_index)
        for table in ("sgsteps", "particles"):
            checks[f"{name}.{table}"] = exact_check(baseline[table], replay[table])
    require_checks(checks, f"default identity failed at original file index {file_index}")
    return {"checks": checks, "placement": stats}


def verify_preserved(baseline, candidate):
    checks = {}
    for table, key in (("sgsteps", "sp"), ("particles", "key")):
        a, b = baseline[table], candidate[table]
        for name, lo, hi in (("primary", -np.inf, steps.SECONDARY_KEY0),
                             ("instraw", steps.SECONDARY_KEY0, steps.OFFAXIS_KEY0)):
            # nsgs necessarily changes when the off-axis population changes.
            x = a.loc[(a[key] >= lo) & (a[key] < hi)].drop(columns="nsgs", errors="ignore")
            y = b.loc[(b[key] >= lo) & (b[key] < hi)].drop(columns="nsgs", errors="ignore")
            checks[f"{name}.{table}"] = exact_check(x, y)
    for table in ("gen", "subruns", "surfsteps"):
        checks[table] = exact_check(baseline.get(table), candidate.get(table))
    sgs = candidate["sgsteps"]
    checks["nsgs"] = exact_check(sgs[["nsgs"]],
        sgs.groupby(core_tables.KEY)["sid"].transform("size").to_frame("nsgs"))
    return require_checks(checks, "candidate changed protected saved tables")


def replay_file(baseline, original, annotated, geom, seed, file_index):
    """Testable, fail-closed in-memory replay of a single file."""
    verify_legacy_payload(original, annotated)
    default = validate_default(baseline, original, annotated, geom, seed, file_index)
    candidate, stats = repaste(baseline, annotated, geom, seed, file_index, topology=True)
    checks = verify_preserved(baseline, candidate)
    return candidate, {"default": default, "preserved": checks, "placement": stats}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def read_baseline(path):
    with uproot.open(path) as root:
        if "dump" not in root:
            raise ValueError(f"{path}: missing dump directory")
        names = set(root["dump"].keys(cycle=False))
    missing = set(TABLES[:-1]) - names
    if missing:
        raise ValueError(f"{path}: missing required saved tables {sorted(missing)}")
    return {name: core_tables.read(path, name) if name in names else None for name in TABLES}


def counts(tables):
    sgs, parts = tables["sgsteps"], tables["particles"]
    return {"rows": {name: len(frame) if frame is not None else None for name, frame in tables.items()},
            "events_with_steps": len(sgs[core_tables.KEY].drop_duplicates()),
            "primary_steps": int((sgs.sp < steps.SECONDARY_KEY0).sum()),
            "instraw_steps": int(sgs.sp.between(steps.SECONDARY_KEY0, steps.OFFAXIS_KEY0 - 1).sum()),
            "offaxis_steps": int((sgs.sp >= steps.OFFAXIS_KEY0).sum()),
            "primary_particles": int((parts.key < steps.SECONDARY_KEY0).sum()),
            "instraw_particles": int(parts.key.between(steps.SECONDARY_KEY0, steps.OFFAXIS_KEY0 - 1).sum()),
            "offaxis_particles": int((parts.key >= steps.OFFAXIS_KEY0).sum())}


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def run(baselines, *, start_index=0, library_path, annotated_path, gdml, outdir, seed=1):
    """Two file passes: globally validate defaults, then write topology candidates.

    The exclusively-created directory also holds counts.json, including failure
    diagnostics. Required inputs are never expanded/reordered. No event loops.
    """
    if start_index < 0 or seed < 0:
        raise ValueError("seed and --start-index must be nonnegative")
    paths = [Path(path).resolve() for path in baselines]
    if not paths or len({p.name for p in paths}) != len(paths):
        raise ValueError("provide explicit baseline files with unique filenames")
    if any(not p.is_file() for p in paths):
        raise ValueError("baseline files must exist; glob patterns are not expanded")
    if any(p.name == "counts.json" for p in paths):
        raise ValueError("baseline filename conflicts with counts.json")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    report = {"status": "validating", "seed": int(seed), "start_index": int(start_index),
              "equality": "exact values and ordered column names; row order preserved; dtype ignored and reported",
              "all_defaults_passed": False, "files": []}
    try:
        report["inputs"] = {"library": file_record(library_path), "annotated": file_record(annotated_path),
                            "gdml": file_record(gdml)}
        report["sources"] = [file_record(p) for p in
            (__file__, steps.__file__, library.__file__, placement.__file__, geometry.__file__,
             steps_analysis.__file__, core_tables.__file__)]
        report["versions"] = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                              "uproot": uproot.__version__}
        original, annotated = library.read(library_path), library.read(annotated_path)
        report["legacy_payload"] = verify_legacy_payload(original, annotated)
        placement.validate(annotated["offaxis"])
        geom = geometry.from_gdml(gdml)
        # Complete every default gate before writing even the first candidate.
        for index, path in enumerate(paths, start=int(start_index)):
            begin = time.perf_counter()
            entry = {"input": file_record(path), "file_index": index, "output_filename": path.name}
            report["files"].append(entry)
            baseline = read_baseline(path)
            entry["baseline_counts"] = counts(baseline)
            try:
                entry["default"] = validate_default(baseline, original, annotated, geom, seed, index)
            except ReplayIdentityError as exc:
                entry["default"] = {"checks": exc.checks}
                raise
            finally:
                entry["default_elapsed_s"] = time.perf_counter() - begin
        report["all_defaults_passed"] = True
        report["status"] = "writing"
        for entry in report["files"]:
            begin = time.perf_counter()
            path = Path(entry["input"]["path"])
            if sha256(path) != entry["input"]["sha256"]:
                raise ValueError(f"baseline changed during replay: {path}")
            baseline = read_baseline(path)
            candidate, entry["placement"] = repaste(baseline, annotated, geom, seed,
                                                    entry["file_index"], topology=True)
            entry["preserved"] = verify_preserved(baseline, candidate)
            entry["candidate_counts"] = counts(candidate)
            output = outdir / path.name
            if output.exists():
                raise FileExistsError(output)
            core_tables.write(str(output), candidate, as_ttree=True)
            entry["output"] = file_record(output)
            entry["candidate_elapsed_s"] = time.perf_counter() - begin
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ReplayIdentityError):
            report["failed_checks"] = exc.checks
        raise
    finally:
        report["elapsed_s"] = time.perf_counter() - t0
        with (outdir / "counts.json").open("x") as stream:
            json.dump(report, stream, indent=2, default=_json_scalar, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baselines", nargs="+", help="explicit saved fast files in original order (no internal globbing)")
    parser.add_argument("--start-index", type=int, default=0, help="original index of first file; inputs must be contiguous")
    parser.add_argument("--library", required=True, dest="library_path")
    parser.add_argument("--annotated", required=True, dest="annotated_path")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True, help="must not already exist")
    parser.add_argument("--seed", type=int, default=1, help="original make_steps seed")
    args = vars(parser.parse_args(argv))
    try:
        run(**args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"secondary replay refused: {exc}\n")


if __name__ == "__main__":
    main()
