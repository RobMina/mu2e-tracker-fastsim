"""Fixed-3000 passive crossing capture gate; NOT independent physics or full F1 closure.

Only this driver is new. Receipt-bound helpers are imported unchanged and checked
at their executing paths (including transitive function bindings). ROOT tables are
read without projection except trace membership; geometry is validated separately
from old/payload tables. No native process, refit, build or source edit is performed.
"""
import argparse
import inspect
import json
from pathlib import Path
import math
import re
import types

import awkward as ak
import numpy as np
import uproot

import f1_crossing_check as C
import f1_crossing_geometry_check as G
import f1_crossing_record_check as R
import f1_hit_operand_check as H
import f1_native_decision_audit as A

SMOKE_SHA = "9b9983124f8df4f48b253158f414f7aa9712d03d1eb0b22e250ea47810f12eb1"
REFERENCE_SHA = "101623117e0f245ad02982cd01ff88a15b7411469319199fee906ae22232c1da"
HIT_SHA = "b97804739a88d739d6fcda45954edd4a88b322e67d2a58929b333971f4c7beb6"
PRED_MANIFEST_SHA = "413b06ff800be415a207a426d6b48bf068d69d09608aefec79cf7c081aee7fca"
EXPANDED_SHA = {"control": "ae36805ddb56e68d9abe05358a9ea59857d45887e0baf7a5b1c0154e432db153",
                "capture": "88106635d1ecaa6d802ff4d27db1e5740803eb27baaa90d8f936e2373c5e150c"}
MODULE_SHA = {
    "f1_crossing_check": "cc4eba8987c79cc1b7135b228a56d88f31104df3b89e69f625ef1a9993298f7a",
    "f1_crossing_geometry_check": "0b390e564e30eaecf92a1927f23bcb34a79ceb791d056d8b9cae3e90b8a754b6",
    "f1_crossing_record_check": "60f5e9ca99e0301c5885956d3faa9c7d5dbbfcf145d837ec9a89c8ce7921d414",
    "f1_hit_operand_check": "cb4d48043feb92c8a09959e0d4240f15e0de73c10c7c8a766474b3e069a3832f",
    "f1_native_decision_audit": "9c32866050627145ed1cff5d245d7a4e79a53a509631c29b87d08b0790a28102",
}
COUNTS = {"n_events": 3000, "baseline_events": 3000, "n_seeds": 2319,
          "n_targets": 147, "n_targets_found": 147, "n_iterations": 3449}
TRUE_FLAGS = ("passed", "final_capture_gate", "complete_event_coverage",
              "full_target_coverage", "recorded_capture_equivalent")
LIMITS = ["Fixed 3000 events, 2319 seeds, and all 147 selected targets; not every seed is captured.",
          "Full selected-target capture validity is not full F1, physics, or full-fit closure.",
          "Exact captured payload/quotient and row-algebra checks are not independent CA, tube-line, retained-history or discovery generation.",
          "No native build, allocation-failure, threaded/nested-scope or arbitrary-configuration certification.",
          "No fitter, thresholds, defaults, tolerances or acceptance changes.",
          "Memory-bounded table lifetimes, not a measured peak-RSS or runtime certificate.",
          "Raw launch-input ROOT is bound by the frozen input-list/path and output event equality, not a historical content hash."]


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number: " + value)
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=constant, parse_float=floating)


def watch(path, watched, expected=None):
    path = str(Path(path).resolve())
    actual = A.digest(path)
    A.require(path not in watched or watched[path] == actual, "conflicting/changed watched hash: " + path)
    watched[path] = actual  # retain even an unexpected digest for the end check
    A.require(expected is None or actual == expected, "hash mismatch: " + path)
    return actual


def count(value, label, expected=None):
    A.require(type(value) is int and 0 <= value <= np.iinfo(np.int64).max, label + ": integer count required")
    A.require(expected is None or value == expected, label + ": fixed population differs")
    return value


def verify_helpers(watched):
    here = Path(__file__).resolve()
    watch(here, watched)
    modules = {m.__name__: m for m in (C, G, R, H, A)}
    paths = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        A.require(path == here.with_name(name + ".py"), "executing module from other checkout: " + name)
        watch(path, watched, MODULE_SHA[name])
        paths[name] = path
    def code_paths(code, path):
        A.require(Path(code.co_filename).resolve() == path, "executing function from other checkout")
        for child in code.co_consts:
            if isinstance(child, types.CodeType):
                code_paths(child, path)
    for module in modules.values():
        for value in vars(module).values():
            if inspect.isfunction(value):
                owner = value.__module__
                A.require(owner in modules, "unexpected transitive helper function")
                A.require(value is getattr(modules[owner], value.__name__, None), "transitive function binding differs")
                A.require(value.__globals__ is vars(modules[owner]), "transitive function globals differ")
                code_paths(value.__code__, paths[owner])
            elif isinstance(value, types.ModuleType) and value.__name__ in modules:
                A.require(value is modules[value.__name__], "transitive module binding differs")
    return {name: MODULE_SHA[name] for name in modules}


def full_receipt(path, watched, expected=None):
    watch(path, watched, expected)
    gate = strict_json(path)
    for field in TRUE_FLAGS:
        A.require(gate.get(field) is True, "requires literal full gate: " + field)
    for field in ("smoke_only", "full_numerical_validation"):
        A.require(gate.get(field) is False, "requires literal false: " + field)
    for field, value in COUNTS.items():
        count(gate.get(field), field, value)
    A.require(gate.get("missing_target_keys") == [], "missing full targets")
    A.require(set(gate["inputs"]) == {"control", "capture", "baseline", "targets"}, "four evidence inputs required")
    for item in gate["inputs"].values():
        watch(item["path"], watched, item["sha256"])
    return gate


def historical(path, expected, watched):
    watch(path, watched, expected)
    report = strict_json(path)
    for field in ("passed", "hashes_unchanged"):
        A.require(report.get(field) is True, "historical receipt failed: " + field)
    end = report["end_sha256"]
    A.require(isinstance(end, dict) and bool(end), "missing historical end hashes")
    if "before_sha256" in report:
        A.require(report["before_sha256"] == end, "historical before/end mismatch")
    for source, digest in end.items():
        watch(source, watched, digest)
    return report


def bind_history(previous, hit, smoke):
    for field in ("all_four_evidence_hashes_verified", "capture_off_hit_operands_empty"):
        A.require(hit.get(field) is True, "predecessor hit gate: " + field)
    A.require(hit.get("check_sha256") == REFERENCE_SHA, "hit receipt belongs to different predecessor")
    A.require(hit["inputs"] == previous["inputs"], "hit receipt input binding differs")
    count(hit.get("n_hits"), "n_hits", 61622)
    count(hit.get("n_expected_hits"), "n_expected_hits", 61622)
    count(hit.get("n_excluded_hits"), "n_excluded_hits", 0)
    count(hit.get("n_iterations"), "hit iterations", COUNTS["n_iterations"])
    A.require(hit.get("full_closure") is False and hit.get("full_numerical_validation") is False,
              "historical hit closure flags")
    A.require(hit.get("rtol") == H.RTOL and hit.get("atol") == H.ATOL, "historical hit tolerances differ")
    A.require(set(hit["original_tree_passivity"]) == set(H.ORIGINAL_KEYS), "missing original passivity gates")
    for item in hit["original_tree_passivity"].values():
        A.require(item.get("passed") is True, "historical original gate failed")
    A.require(smoke.get("smoke_only") is True and smoke.get("full_fit_closure") is False, "historical smoke scope")
    count(smoke.get("n_events"), "smoke events", 100)
    count(smoke.get("n_targets"), "smoke targets", 2)
    for name, digest in MODULE_SHA.items():
        matches = [v for p, v in smoke["before_sha256"].items() if Path(p).name == name + ".py"]
        A.require(matches == [digest], "smoke helper pin missing/different: " + name)


def source_pins(area, preparation, watched):
    """Read recorded repo pins only; never execute git or a native command."""
    receipt = strict_json(preparation)
    expected = receipt["pins"]  # preparation itself is hash-pinned by C.source_manifest
    path = Path(area).resolve() / "source-pins.txt"
    watch(path, watched)
    rows = [line.split() for line in path.read_text().splitlines()]
    A.require(len(rows) == 5 and all(len(row) == 2 for row in rows), "source repo pin population")
    A.require(len({row[0] for row in rows}) == 5 and dict(rows) == expected, "source repo pins differ")
    return expected


def manifest_fields(path):
    text = Path(path).read_text().splitlines()
    result = {}
    for field in ("area", "input", "events", "config"):
        values = [line[len(field)+1:] for line in text if line.startswith(field + "=")]
        A.require(len(values) == 1 and values[0] != "", "ambiguous launch field: " + field)
        result[field] = values[0]
    return result


def configuration(gate, previous, smoke, area, watched):
    directory = Path(gate["inputs"]["capture"]["path"]).resolve().parent
    old = Path(previous["inputs"]["capture"]["path"]).resolve().parent
    A.require(Path(gate["inputs"]["control"]["path"]).resolve().parent == directory, "pair directories differ")
    a, b = directory / "manifest.txt", old / "manifest.txt"
    watch(a, watched)
    watch(b, watched, PRED_MANIFEST_SHA)
    now, before = manifest_fields(a), manifest_fields(b)
    A.require(now["events"] == before["events"] == "3000", "requires fixed 3000 launch")
    A.require(Path(now["area"]).resolve() == Path(area).resolve(), "launch/native area differs")
    for field in ("input", "config"):
        A.require(now[field] == before[field], "frozen launch differs: " + field)
    smoke_launch = [manifest_fields(p) for p in smoke["before_sha256"] if Path(p).name == "manifest.txt"]
    smoke_launch = [m for m in smoke_launch if m["area"] == now["area"]]
    A.require(len(smoke_launch) == 1, "missing/ambiguous crossing smoke launch")
    A.require(smoke_launch[0]["events"] == "100", "smoke launch event count")
    for field in ("input", "config"):
        A.require(now[field] == smoke_launch[0][field], "launch differs from 100 smoke: " + field)
    config_pins = {"control.fcl": "b228c36d268e8d49fc48e879cd7cbbc76285ba857ec3325f4992f73f81c77335",
                   "capture.fcl": "4154132908087c1379314309fb72996bdabc5214b27c734e079c27f77dd4808a",
                   "targets.json": "f32c3e49bec3746fc15a57bc588a3858a15195820fd1705a4d756cb7504fce90"}
    for name, digest in config_pins.items():
        watch(Path(now["config"]) / name, watched, digest)
    A.require(gate["inputs"]["targets"]["sha256"] == config_pins["targets.json"], "launch target set differs")
    # Bind the frozen list, not a new historical hash or metadata query on the PNFS input.
    input_list = Path(now["config"]).parent / "au_dig_3f.txt"
    watch(input_list, watched, "91c49c9a256a0fd6f7af34620db96ad40cab4336f616316ef31855e130090802")
    A.require(input_list.read_text().splitlines()[0] == now["input"], "launch input differs from frozen list")
    for arm, digest in EXPANDED_SHA.items():
        a, b = directory / (arm + ".expanded.fcl"), old / (arm + ".expanded.fcl")
        watch(a, watched)
        watch(b, watched, digest)
        C.expanded_equal(a, b)
        exit_path = directory / (arm + ".exit")
        watch(exit_path, watched)
        A.require(exit_path.read_text().strip() == "0", "native arm exit failure")
    return {"passed": True, "events": 3000, "same_input_config_as_smoke": True,
            "allowed_expanded_difference": "unique debug-output filename only"}


def manifest_hashes(path, area):
    hashes = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match:
            digest, name = match.groups()
            name = Path(name)
            A.require(not str(name).startswith("/pnfs/"), "unexpected PNFS manifest content hash")
            name = str((name if name.is_absolute() else Path(area) / name).resolve())
            A.require(name not in hashes or hashes[name] == digest, "conflicting manifest hash: " + name)
            hashes[name] = digest
    return hashes


def manifest_integrity(args, gate, pins, watched):
    area = Path(args.native_source).resolve()
    directory = Path(gate["inputs"]["capture"]["path"]).resolve().parent
    manifest = directory / "manifest.txt"
    recorded = manifest_hashes(manifest, area)
    expected_sources = manifest_hashes(area / "capture-source-manifest.sha256", area)
    rows = [line.split() for line in manifest.read_text().splitlines()]
    repo_rows = [row for row in rows if len(row) == 2 and row[0] in pins]
    A.require(len(repo_rows) == len(pins) and dict(repo_rows) == pins, "launch repository pins differ")
    root = Path(args.smoke_check).resolve().parent.parent
    preflight = root / "crossing_build_preflight_20261028_v1/verification.txt"
    watch(preflight, watched, "c79577c3aeea7e18216091e431f60bd4f19ac11522ec3a06e3f8b871cecc6da0")
    built = manifest_hashes(preflight, area)
    libdir = area / "build/al9-prof-e29-p109/Offline/lib"
    for component in ("Detector", "Examples", "Fit", "General", "Geometry", "MatEnv", "Trajectory"):
        path = str(libdir / ("libKinKal_" + component + ".so.0.0.0"))
        expected_sources[path] = built[path]
    plugin = area / "build/al9-prof-e29-p109/TrkFastSim/lib/libtrkfastsim_KinematicLineFitTrace_module.so"
    expected_sources[str(plugin)] = "7572aa4fc5265ce90bbc215670232e37503f389f7e8c7af55ddfc9dd3f9f6a63"
    here = Path(__file__).resolve().parent
    required = [here / name for name in ("run_f1_crossing_full_pair.sh", "f1_crossing_full_check.py",
                "f1_crossing_check.py", "f1_crossing_geometry_check.py", "f1_crossing_record_check.py")]
    required += [Path(getattr(args, name)).resolve() for name in
                 ("preparation", "smoke_check", "reference_check", "reference_hit_check")]
    config = Path(manifest_fields(manifest)["config"])
    required += [config / name for name in ("control.fcl", "capture.fcl", "targets.json")]
    required += [config.parent / "au_dig_3f.txt", preflight, directory / "space_plan.txt"]
    for path in required:
        name = str(path.resolve())
        expected_sources[name] = watched[name] if name in watched else watch(path, watched)
    A.require(set(expected_sources) <= set(recorded), "missing required launch provenance entries")
    for path, digest in expected_sources.items():
        A.require(recorded[path] == digest, "launch provenance differs: " + path)
    for path, digest in recorded.items():
        watch(path, watched, digest)
    return {"passed": True, "n_recorded_hashes": len(recorded), "n_required_hashes": len(expected_sources),
            "repository_pins_match": True, "installed_artifacts_match_verified_smoke": True}


def integer_keys(table, names=A.KEYS, unique=True):
    for name in names:
        value = A.column(table, name, "iu")
        A.require(value >= 0, "negative identity")
        A.require(value.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), "identity overflow")
    return A.keys(table, names, unique=unique)


def targets(document):
    raw = document["keys"]
    A.require(not np.ma.isMaskedArray(raw), "masked targets")
    # JSON bool is an int subclass; reject before NumPy can coerce it to int.
    if isinstance(raw, list):
        A.require(all(isinstance(row, list) and all(type(x) is int for x in row) for row in raw), "integer target keys required")
    value = np.asarray(raw)
    A.require(value.ndim == 2 and value.shape == (147, 4) and value.dtype.kind in "iu", "fixed target matrix required")
    A.require(value >= 0, "negative target")
    A.require(value.astype(np.uint64) <= np.uint64(np.iinfo(np.int32).max), "native target overflow")
    A.require(len(np.unique(value, axis=0)) == 147, "duplicate targets")
    return value.astype(np.int64)


def selected(seed, iterations, trace, event_keys, expected):
    sk, ik, tk = integer_keys(seed), integer_keys(iterations, A.ITER_KEYS), integer_keys(trace)
    A.require(len(sk) == len(tk) == 2319 and len(ik) == 3449, "highest-cycle seed/iteration census")
    A.require(np.array_equal(sk, tk), "trace/seed order differs")
    A.match_rows(sk[:, :3], event_keys)
    wanted = np.zeros(len(seed), dtype=bool)
    wanted[A.match_rows(expected, sk)] = True
    A.require(np.array_equal(H.boolean(seed, "capture_requested"), wanted), "seed target requests differ")
    actual = np.unique(ik[:, :4], axis=0)
    A.require(len(actual) == 147, "iteration target population")
    A.match_rows(expected, actual)
    parent = A.match_rows(ik[:, :4], sk)
    counts = np.bincount(parent, minlength=len(seed))
    for field in ("iteration_capture_expected", "iteration_capture_written"):
        v = A.column(trace, field, "iu")
        A.require(v >= 0, "negative iteration count")
        A.require(v.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), "iteration count overflow")
        A.require(v == counts, field + " mismatch")
    for field in ("iteration_capture_requested", "iteration_capture_valid"):
        A.require(A.column(trace, field, "b") == wanted, field + " differs")
    A.require(A.column(trace, "iteration_capture_errors") == "", "trace capture error")
    order = np.lexsort((ik[:, 4], parent))
    A.require(ik[order, 4] == np.arange(len(ik)) - np.repeat(np.cumsum(counts)-counts, counts), "iteration ordinal gaps")
    return sk


def read(path, directory, name, cycles, fields=None):
    with uproot.open(path, object_cache=None, array_cache=None) as root:
        return A.read_tree(root, directory, name, cycles, fields)


def empty_schema(path, name, enabled, cycles):
    with uproot.open(path, object_cache=None, array_cache=None) as root:
        dkey = A.highest(root, "kktrace")
        tkey = A.highest(root[dkey], name)
        tree = root[dkey][tkey]
        cycles["kktrace/" + name] = dkey + "/" + tkey
        A.require(tree.num_entries == 0, "capture-off tree not empty: " + name)
        disabled = tree.arrays(entry_stop=0)
    A.require(set(ak.fields(disabled)) == set(ak.fields(enabled)), "capture-off partial schema: " + name)
    for field in ak.fields(enabled):
        A.require(ak.type(disabled[field]).content == ak.type(enabled[field]).content,
                  "capture-off type differs: " + name + "/" + field)


def compact_hit(report):
    """Do not serialize H's O(hits) diagnostic lists; keep every summary count."""
    report.pop("hit_keys", None)
    for name in ("weight_matrix", "weight_vector"):
        metric = report[name]
        metric.pop("per_hit_failed_components", None)
        metric.pop("failed_hit_keys", None)
    report["per_hit_lists_omitted"] = True
    return report


def geometry_census(tables, expected, smoke):
    """Fresh complete detector geometry per target, not merely self-consistent emptiness."""
    summary = smoke["geometry"]
    A.require(summary.get("passed") is True, "historical geometry gate failed")
    n = count(summary.get("n_targets"), "smoke geometry targets", 2)
    bindings = tables[G.PREFIX + "bindings"]
    for field, total in (("nplanes", "n_planes"), ("npanels", "n_panels"), ("nstraws", "n_straws")):
        size = count(summary.get(total), "smoke geometry " + total)
        A.require(size > 0 and size % n == 0, "smoke geometry per-target census")
        A.require(G._integer(bindings, field) == size // n, "full per-target geometry census differs")
    A.require(len(bindings) == 2 * len(expected), "full geometry binding count")


def population(gate, previous, hit, expected, report, smoke):
    paths = {name: gate["inputs"][name]["path"] for name in ("capture", "control", "baseline")}
    paths["predecessor"] = previous["inputs"]["capture"]["path"]
    cycles = {name: {} for name in paths}
    report["highest_cycles"] = cycles
    events = read(paths["capture"], "trkdump", "events", cycles["capture"])
    ek = integer_keys(events, A.KEYS[:3])
    A.require(len(ek) == 3000, "highest-cycle event census")
    report["event_passivity"] = {}
    for name in ("control", "predecessor", "baseline"):
        other = read(paths[name], "trkdump", "events", cycles[name])
        integer_keys(other, A.KEYS[:3])
        report["event_passivity"][name] = C.exact_table(events, other, A.KEYS[:3], name + " full events")
        del other
    del events
    trace_fields = list(A.KEYS) + ["iteration_capture_" + x for x in ("expected", "written", "requested", "valid", "errors")]
    trace = read(paths["capture"], "kktrace", "trace", cycles["capture"], trace_fields)
    sk = integer_keys(trace)
    for name in ("control", "predecessor", "baseline"):
        other = read(paths[name], "kktrace", "trace", cycles[name], list(A.KEYS))
        other_keys = integer_keys(other)
        A.require(np.array_equal(sk, other_keys), name + " trace seed membership/order differs")
        A.match_rows(other_keys[:, :3], ek)
        del other
    data = {}
    report["original_tree_passivity"] = {}
    for name, identity in {**H.ORIGINAL_KEYS, "hit_operands": H.HIT_KEYS}.items():
        current = read(paths["capture"], "kktrace", name, cycles["capture"])
        reference = read(paths["predecessor"], "kktrace", name, cycles["predecessor"])
        integer_keys(current, identity)
        integer_keys(reference, identity)
        result = C.exact_table(current, reference, identity, name)
        expected_rows = hit["n_hits"] if name == "hit_operands" else hit["original_tree_passivity"][name]["n_rows"]
        count(expected_rows, name + " receipt rows")
        A.require(len(current) == expected_rows, name + " receipt census differs")
        del reference
        if name == "hit_operands":
            report["hit_operand_passivity"] = result
        else:
            report["original_tree_passivity"][name] = result
        if name == "seed_contract":
            other = read(paths["control"], "kktrace", name, cycles["control"])
            A.require(~H.boolean(other, "capture_requested"), "off seed capture requested")
            C.exact_table(ak.without_field(current, "capture_requested"),
                          ak.without_field(other, "capture_requested"), A.KEYS, "off/on seed")
            del other
        else:
            empty_schema(paths["control"], name, current, cycles["control"])
        if name != "pieces":  # compared all fields; not needed by H/R/G
            data[name] = current
        del current
    selected(data["seed_contract"], data["iterations"], trace, ek, expected)
    del trace
    report["hit_operand_consistency"] = compact_hit(H.validate(data["hit_operands"], data["effects"], data["iterations"]))
    A.require(report["hit_operand_consistency"]["passed"] is True, "hit operand consistency failed")
    del data["hit_operands"], data["seed_contract"], data["iterations"]
    for name in R.SCHEMAS:
        data[name] = read(paths["capture"], "kktrace", name, cycles["capture"])
        empty_schema(paths["control"], name, data[name], cycles["control"])
    report["crossings"] = R.validate(data, data["effects"], expected)
    A.require(report["crossings"].get("passed") is True, "crossing gate failed")
    data.clear()  # free old/record payloads BEFORE ~3 million geometry rows
    for suffix in G.FIELDS:
        name = G.PREFIX + suffix
        data[name] = read(paths["capture"], "kktrace", name, cycles["capture"])
        empty_schema(paths["control"], name, data[name], cycles["control"])
    geometry_census(data, expected, smoke)
    report["geometry"] = G.validate(data, expected)  # one vectorized full-population call
    A.require(report["geometry"].get("passed") is True, "geometry gate failed")
    data.clear()
    report.update(n_events=3000, n_seeds=2319, n_targets=147, n_iterations=3449,
                  missing_target_keys=[], capture_off_sidecars_empty=True,
                  complete_event_coverage=True, complete_seed_membership=True,
                  full_target_coverage=True, recorded_capture_equivalent=True)


def run(args):
    out = Path(args.out).resolve()
    protected = [Path(getattr(args, name)).resolve() for name in
                 ("check", "reference_check", "reference_hit_check", "smoke_check", "preparation")]
    A.require(out not in protected and not out.is_relative_to(Path(args.native_source).resolve())
              and not out.is_relative_to(Path(__file__).resolve().parents[1]), "protected report path")
    with out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = {"passed": False, "smoke_only": False, "full_target_capture_valid": False,
                  "full_fit_closure": False, "full_f1_closure": False, "full_numerical_validation": False,
                  "limits": LIMITS}
        try:
            report["helper_sha256"] = verify_helpers(watched)
            C.source_manifest(args.native_source, args.preparation, watched)
            report["native_repo_pins"] = source_pins(args.native_source, args.preparation, watched)
            gate = full_receipt(args.check, watched)
            previous = full_receipt(args.reference_check, watched, REFERENCE_SHA)
            hit = historical(args.reference_hit_check, HIT_SHA, watched)
            smoke = historical(args.smoke_check, SMOKE_SHA, watched)
            bind_history(previous, hit, smoke)
            for name in ("baseline", "targets"):
                A.require(gate["inputs"][name] == previous["inputs"][name], "predecessor input mismatch: " + name)
            current_paths = [str(Path(gate["inputs"][name]["path"]).resolve()) for name in ("control", "capture")]
            old_paths = {str(Path(previous["inputs"][name]["path"]).resolve()) for name in ("control", "capture", "baseline")}
            A.require(len(set(current_paths)) == 2 and not set(current_paths) & old_paths, "independent new off/on pair required")
            report["configuration"] = configuration(gate, previous, smoke, args.native_source, watched)
            report["launch_provenance"] = manifest_integrity(args, gate, report["native_repo_pins"], watched)
            expected = targets(strict_json(gate["inputs"]["targets"]["path"]))
            population(gate, previous, hit, expected, report, smoke)
            report.update(passed=True, full_target_capture_valid=True)
        except Exception as error:
            # Includes malformed ROOT/JSON/library errors; never a successful partial report.
            report.update(passed=False, full_target_capture_valid=False, error=f"{type(error).__name__}: {error}")
        finally:
            report["before_sha256"] = watched
            end, errors = {}, []
            for path in watched:
                try:
                    end[path] = A.digest(path)
                except Exception as error:
                    errors.append(f"{path}: {type(error).__name__}: {error}")
            report.update(end_sha256=end, hashes_unchanged=end == watched and not errors)
            if not report["hashes_unchanged"]:
                report.update(passed=False, full_target_capture_valid=False,
                              integrity_error="Evidence/source changed during checker", hash_errors=errors)
            try:
                text = json.dumps(report, indent=2, allow_nan=False)
            except (TypeError, ValueError) as error:
                report = {"passed": False, "full_target_capture_valid": False, "full_fit_closure": False,
                          "error": "non-JSON report: " + str(error), "before_sha256": watched,
                          "end_sha256": end, "hashes_unchanged": report["hashes_unchanged"]}
                text = json.dumps(report, indent=2, allow_nan=False)
            stream.write(text + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("check", "reference-check", "reference-hit-check", "smoke-check", "native-source", "preparation", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
