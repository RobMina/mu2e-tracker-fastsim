"""Pinned full-capture audit of all 81 associated-extension crossing constructors.

This is conditional construction on supplied native trajectories/geometry and the
prior independently reconstructed upstream hit-constructor hints. The recorded
crossing line, hints, CA, reference selection, speed and offset are comparison-only.
No native job, baseline regeneration, fitter edit or full-history claim is made.
"""
import argparse
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace

import awkward as ak
import numpy as np
import uproot

import f1_crossing_constructor as B
import f1_later_ca as L
import f1_native_decision_audit as A
from trkfastsim.reco import nativearith as N, nativeca as C, nativemeasurement as M

FULL_SHA = "c8e1225b1cc71ad82ff7055c77cd382c1b564956a1470522136d48575d53f8ec"
OBS_SHA = "20217e4e7a32791d14291d64418ac2bf5873140b8a5b2930090fb9613e8b6626"
CAPTURE_SHA = "d8aad2725e45ac0839456605b375aff229e98af33b772a16cb007e23bc72a667"
PINS = {
    "scripts/f1_later_ca.py": "c6f9009592a381c3a9590821e20c4364f00eba55b98f645ec13718ecfa777c53",
    "scripts/f1_native_decision_audit.py": "9c32866050627145ed1cff5d245d7a4e79a53a509631c29b87d08b0790a28102",
    "trkfastsim/reco/nativearith.py": "9efa38186cbf4d08df478d9fc85349b92d8af5317daf472721143fcec86036ae",
    "trkfastsim/reco/nativeca.py": "1c2392271df009c0352ab4290b4c1955b9cace817b99b13c0d68f43c4eb34531",
    "trkfastsim/reco/nativemeasurement.py": "b8399c3d79cbc295aae82c1bdab1899da5aa9efe7e67b68f2d907e678dc35f47",
}
SOURCE_PINS = {
    "KinKal/Geometry/LineSegment.hh": "0b37bfcadc5686bb6211fd09beefbf214642f986c989c8385c0989124343db53",
    "KinKal/Geometry/Ray.hh": "266029ed823620801e28f82d621c31220e6241ec038159513469087e249d06a8",
    "KinKal/Trajectory/KinematicLine.cc": "002ba5f14971945132540e43868ab4f10f584dfaf4ca419311f1173305fa66bb",
    "KinKal/Trajectory/KinematicLine.hh": "443887d08856638d372a3a00ed232e17a98d5aeadb540191c81af634577b0995",
    "KinKal/Trajectory/ConstantDistanceToTime.hh": "7520e8a71da99637de7b96b138e27a3ab94fb981b1d4b274a3d9bf713f1bbbe7",
}
LIMITS = [
    "All81 associated-extension constructors/50targets; initial2538 and material-only2471 constructors are outside this declared scope, not failed comparisons removed.",
    "Native construction trajectories, mass and target-specific raw geometry are supplied inputs; not independently generated F1 trajectories/geometry conditions.",
    "Upstream hit-constructor hints are independently reconstructed by the pinned prior audit on native trajectories and accepted membership; no autonomous hit discovery claim.",
    "No recorded crossing CA/hints/line/offset is an input to generate; no material lengths or effect times are inverted to recover an offset.",
    "Wire radius divided by FINAL reference-piece speed, not necessarily the retained solution-piece speed; offset is construction-fixed.",
    "Retained CA geometry/status/piece visits are compared; cached derivatives/variances, later wrapper/reference history, full partner lifecycle and physical discovery are separate gates.",
    "Fixed source grouping and NumPy iterative search are not universal ROOT/compiler parity; no correction fitting, regularization or tolerance change.",
    "No full numerical, physics, decision, acceptance or full-fit/F1 closure follows from this conditional constructor comparison.",
]


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            A.require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def floating(value):
        x = float(value)
        A.require(np.isfinite(x), "nonfinite JSON number")
        return x
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=constant, parse_float=floating)


def watch(path, watched, expected=None):
    p = str(Path(path).resolve())
    if expected is not None:
        A.require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), "invalid expected digest")
    sha = A.digest(p)
    A.require(p not in watched or watched[p] == sha, "changed watched input: " + p)
    watched[p] = sha
    A.require(expected is None or sha == expected, "hash mismatch: " + p)
    return sha


def receipt(path, expected, watched, flags):
    path = Path(path).resolve()
    watch(path, watched, expected)
    document = strict_json(path)
    for flag in flags:
        A.require(document.get(flag) is True, "missing literal receipt flag: " + flag)
    manifest = document.get("before_sha256")
    A.require(isinstance(manifest, dict) and bool(manifest) and manifest == document.get("end_sha256"), "incomplete receipt manifest")
    for p, sha in manifest.items():
        A.require(isinstance(p, str) and Path(p).is_absolute() and not p.startswith("/pnfs/"), "invalid manifest path")
        watch(p, watched, sha)
    return document


def code_origin(function, path):
    seen = set()
    while True:
        A.require(inspect.isfunction(function) and id(function) not in seen, "missing/cyclic required function")
        seen.add(id(function))
        A.require(Path(function.__code__.co_filename).resolve() == path, "function code from different checkout")
        if not hasattr(function, "__wrapped__"):
            break
        function = function.__wrapped__


def helpers(watched):
    root = Path(__file__).resolve().parents[1]
    modules = {"scripts/f1_crossing_constructor_audit.py": __import__(__name__),
               "scripts/f1_crossing_constructor.py": B,
               "scripts/f1_later_ca.py": L, "scripts/f1_native_decision_audit.py": A,
               "trkfastsim/reco/nativearith.py": N, "trkfastsim/reco/nativeca.py": C,
               "trkfastsim/reco/nativemeasurement.py": M}
    required = {
        "scripts/f1_crossing_constructor_audit.py": ("strict_json", "watch", "receipt", "code_origin", "helpers", "census", "population", "serialize", "run"),
        "scripts/f1_crossing_constructor.py": ("integer_keys", "integer_matrix", "associated_inputs", "pack_trajectories", "geometry_inputs", "generate", "boolean", "comparison", "compare"),
        "scripts/f1_later_ca.py": ("_array", "_real", "_integer", "_line", "_pieces", "_nearest", "_solve", "piecewise"),
        "scripts/f1_native_decision_audit.py": ("require", "array", "column", "dense", "keys", "match_rows", "digest", "highest", "read_tree"),
        "trkfastsim/reco/nativearith.py": ("_dot3", "sensor"),
        "trkfastsim/reco/nativeca.py": ("_vector", "particle", "sensor", "sensor_position", "solve"),
        "trkfastsim/reco/nativemeasurement.py": ("_finite",),
    }
    for name, module in modules.items():
        path = root/name
        A.require(Path(module.__file__).resolve() == path, "executing module path differs: " + name)
        watch(path, watched, PINS.get(name))
        for field in required[name]:
            code_origin(getattr(module, field, None), path)
        for value in vars(module).values():
            if inspect.isfunction(value) and value.__module__ == module.__name__:
                code_origin(value, path)
    A.require(B.L is L and B.N is N and B.C is C and L.C is C and C.N is M and N.N is M, "transitive helper binding differs")
    for field in ("array", "column", "dense", "keys", "match_rows", "require"):
        A.require(getattr(B, field) is getattr(A, field), "imported helper binding differs: " + field)
    return {name: str(Path(module.__file__).resolve()) for name, module in modules.items()}


def census(records, discovery):
    stage, category = A.column(records, "stage", "iu"), A.column(records, "category", "iu")
    rows = records[stage == 0]
    B.integer_keys(rows, B.OBJECT)
    counts = {name: int(np.sum(A.column(rows, "category") == code))
              for code, name in enumerate(("initial", "associated", "material"))}
    A.require(counts == {"initial": 2538, "associated": 81, "material": 2471}, "frozen constructor census differs")
    exhausted = B.boolean(records, "has_search") & B.boolean(records, "search_exhausted")
    groups = {f"stage{s}_category{c}": int(np.sum(exhausted & (stage == s) & (category == c)))
              for s in range(5) for c in range(3)}
    dcap = B.boolean(discovery, "has_search") & B.boolean(discovery, "search_exhausted")
    A.require(int(exhausted.sum()+dcap.sum()) == 7, "full captured exhausted-search census differs")
    return dict(constructors=counts, exhausted_record_searches=groups,
                exhausted_discovery_searches=int(dcap.sum()),
                exhausted_record_keys=B.integer_keys(records[exhausted], B.KEYS+("sequence",)).tolist(),
                exhausted_discovery_keys=B.integer_keys(discovery[dcap], B.KEYS+("sequence",)).tolist())


def population(capture, hints, report):
    cycles = {}
    report["highest_cycles"] = cycles
    with uproot.open(capture, object_cache=None, array_cache=None) as root:
        read = lambda name, fields=None: A.read_tree(root, "kktrace", name, cycles, fields)
        records = read("crossing_records")
        discovery = read("crossing_discovery", list(B.KEYS)+["sequence", "has_search", "search_exhausted"])
        report["population_census"] = census(records, discovery)
        del discovery
        hits = read("hit_operands", list(B.KEYS)+["iteration_ordinal", "effect_index", "shindex", "strawid"])
        constructors, source_keys, upstream = B.associated_inputs(records, hits, hints)
        del hits, records
        A.require(len(constructors) == 81 and len(np.unique(source_keys[:, :4], axis=0)) == 50, "associated target census differs")
        trajectories = read("crossing_trajectories")
        q, ranges, counts, parent, mass = B.pack_trajectories(trajectories, constructors)
        report["construction_trajectories"] = dict(n_total=len(counts), n_with_associated_constructor=len(np.unique(parent)),
            n_without_associated_constructor=len(counts)-len(np.unique(parent)),
            scope="extension entry captures a trajectory before selection, including targets with no accepted associated crossing")
        del trajectories
        bindings = read("crossing_geometry_bindings")
        tracker = read("crossing_geometry_tracker")
        straws = read("crossing_geometry_straws", ["geometry_index", "fullsid", "tube_cal", "tube_hv"])
        cal, hv, radius = B.geometry_inputs(constructors, bindings, tracker, straws)
        del bindings, tracker, straws
    generated = B.generate(q, ranges, counts, parent, mass, cal, hv,
                           upstream["particle_hint"], upstream["sensor_hint"], radius)
    ids = B.integer_keys(constructors, B.OBJECT)
    report["comparison"] = B.compare(generated, constructors, ids)
    report["identities"] = dict(object_keys=ids.tolist(), source_keys=source_keys.tolist(),
                                partner_shindex=A.column(constructors, "partner").tolist(),
                                strawid=A.column(constructors, "strawid").tolist())
    report["generated"] = {name: generated[name].tolist() for name in
                            ("offset", "effect_time", "reference_speed", "particle_hint", "sensor_hint")}
    report["generated"]["state"] = {name: generated["state"][name].tolist() for name in
                                     ("particle_time", "sensor_time", "particle_poca", "sensor_poca", "direction",
                                      "sensor_direction", "index", "solution_index", "moves", "exhausted", "status")}
    report["n_targets"] = 50


def serialize(report):
    try:
        text = json.dumps(report, indent=2, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        report = dict(completed=False, passed=False, full_fit_closure=False, full_f1_closure=False,
                      error=f"strict JSON serialization failed: {type(error).__name__}: {error}",
                      before_sha256=report["before_sha256"], end_sha256=report["end_sha256"],
                      hashes_unchanged=report["hashes_unchanged"], hash_errors=report["hash_errors"])
        text = json.dumps(report, indent=2, allow_nan=False)
    return report, text+"\n"


def run(args):
    args = SimpleNamespace(**{name: Path(getattr(args, name)).resolve() for name in
                              ("out", "full_check", "observability", "capture", "native_source")})
    out = args.out
    protected = [Path(getattr(args, name)).resolve() for name in ("full_check", "observability", "capture")]
    A.require(out not in protected and not out.is_relative_to(Path(__file__).resolve().parents[1])
              and not out.is_relative_to(Path(args.native_source).resolve()), "protected report path")
    with out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = dict(completed=False, passed=False, full_fit_closure=False, full_f1_closure=False,
                      full_history_closure=False, full_numerical_validation=False,
                      independent_discovery_replayed=False, rtol=B.RTOL, atol=B.ATOL, limits=LIMITS)
        try:
            report["executing_modules"] = helpers(watched)
            full = receipt(args.full_check, FULL_SHA, watched,
                           ("passed", "full_target_capture_valid", "hashes_unchanged", "full_target_coverage"))
            for name, count in (("n_events", 3000), ("n_seeds", 2319), ("n_targets", 147), ("n_iterations", 3449)):
                A.require(type(full.get(name)) is int and full[name] == count, "full receipt census differs")
            A.require(full["before_sha256"].get(str(Path(args.capture).resolve())) == CAPTURE_SHA, "capture not bound by full gate")
            watch(args.capture, watched, CAPTURE_SHA)
            obs = receipt(args.observability, OBS_SHA, watched, ("completed", "consistency_passed", "hashes_unchanged"))
            A.require(obs.get("rtol") == B.RTOL and obs.get("atol") == B.ATOL, "upstream tolerance differs")
            area = Path(args.native_source).resolve()
            for name in ("Offline/Mu2eKinKal/inc/KKFit.hh", "Offline/Mu2eKinKal/inc/KKStrawXing.hh",
                         "Offline/Mu2eKinKal/src/KKFitUtilities.cc", "KinKal/Trajectory/SensorLine.cc",
                         "KinKal/Trajectory/PiecewiseClosestApproach.hh"):
                A.require(str(area/name) in full["before_sha256"], "native source not bound by full capture")
            for name, sha in SOURCE_PINS.items():
                watch(area/name, watched, sha)
            report["inputs"] = dict(full_check=str(args.full_check), observability=str(args.observability), capture=str(args.capture))
            population(args.capture, obs["associated_constructor_inputs"], report)
            report.update(completed=True, passed=report["comparison"]["passed"])
        except Exception as error:
            report.update(error=f"{type(error).__name__}: {error}", passed=False)
        finally:
            end, errors = {}, []
            for path in watched:
                try:
                    end[path] = A.digest(path)
                except Exception as error:
                    errors.append(f"{path}: {type(error).__name__}: {error}")
            report.update(before_sha256=watched, end_sha256=end, hashes_unchanged=end == watched and not errors,
                          hash_errors=errors)
            if not report["hashes_unchanged"]:
                report.update(passed=False, integrity_error="Evidence/source changed during audit")
            report, text = serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-check", "observability", "capture", "native-source", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({name: report.get(name) for name in ("completed", "passed", "error", "hashes_unchanged", "n_targets")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
