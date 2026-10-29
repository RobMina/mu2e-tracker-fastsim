"""Pinned conditional replay of all 2538 initial hit/tube-crossing constructors."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uproot

import f1_crossing_constructor_audit as D
import f1_initial_crossing as I
from f1_crossing_constructor_audit import A, B, C, L, N

ASSOCIATED_SHA = "8f13afc822ef99590b36c47764251f1fdefb994117cb0bdeb6b48a98818bfe2e"
PINS = {
    "scripts/f1_crossing_constructor.py": "7a1dd18bd18f81a3f7ee69901112c78068cec988a7760752cbc5cc5963f213b5",
    "scripts/f1_crossing_constructor_audit.py": "f3268c65ef0083dbaacd615d0dbb8bcca060d3c53617d84077c7c99515000bd7",
    "scripts/f1_captured_ca_audit.py": "fb175b3ffc0e5ebc45dbaa95dbe06ce2463dda677ac9c75d38e520aca2feedb8",
}
LIMITS = [
    "All2538 initial constructors/147targets, zero exclusions; associated81 and material2471 are outside this numerical comparison.",
    "Initial membership, raw target-specific geometry and native single-piece seed trajectories/mass are conditioned inputs, not autonomous seed generation or discovery.",
    "Hit wire line and constructor hints are generated from measured ComboHit endpoint/corrected times and energy plus source-FCL signal speed.",
    "Tube line uses separate raw tube endpoints and generated hit-CA times; recorded crossing outputs and seed hit times enter comparison only.",
    "Fixed constructor offset uses geometry wire radius divided by final reference speed; never recovered by subtraction or material inversion.",
    "No cached derivative/variance, per-local-solve scalar trace, later reference/wrapper/partner, seven-cap history or material-discovery closure.",
    "Unchanged numerical thresholds and source grouping; not universal ROOT/compiler parity, fitter integration or full numerical/fit/F1 closure.",
]


def helpers(watched):
    result = D.helpers(watched)
    root = Path(__file__).resolve().parents[1]
    for name, sha in PINS.items():
        D.watch(root/name, watched, sha)
    modules = {"scripts/f1_initial_crossing.py": I,
               "scripts/f1_initial_crossing_audit.py": __import__(__name__)}
    required = {"scripts/f1_initial_crossing.py": ("initial_inputs", "construction_inputs", "geometry_inputs", "measurements", "generate", "speed_configuration", "signal_speed"),
                "scripts/f1_initial_crossing_audit.py": ("helpers", "bound_input", "population", "run")}
    for name, module in modules.items():
        path = root/name
        A.require(Path(module.__file__).resolve() == path, "executing module path differs: "+name)
        D.watch(path, watched, PINS.get(name))
        for field in required[name]:
            D.code_origin(getattr(module, field, None), path)
        result[name] = str(path)
    A.require(I.B is B and I.L is L and I.C is C and I.N is N, "transitive initial binding differs")
    for name in ("array", "column", "dense", "match_rows", "require"):
        A.require(getattr(I, name) is getattr(A, name), "initial imported alias differs: "+name)
    for name in ("watch", "receipt", "strict_json", "code_origin", "helpers", "serialize", "census"):
        D.code_origin(getattr(D, name), root/"scripts/f1_crossing_constructor_audit.py")
    D.code_origin(C.constructor_hint, root/"trkfastsim/reco/nativeca.py")
    for module in (L, N):
        for value in vars(module).values():
            if callable(value) and hasattr(value, "__wrapped__"):
                cells = dict(zip(value.__code__.co_freevars, value.__closure__ or ()))
                A.require("function" in cells and cells["function"].cell_contents is value.__wrapped__, "wrapper execution/metadata differs")
    return result


def bound_input(path, manifest, watched):
    path = Path(path).resolve()
    A.require(str(path) in manifest, "input not bound by full capture: "+str(path))
    D.watch(path, watched, manifest[str(path)])
    return path


def population(capture, configuration, report):
    cycles = report["highest_cycles"] = {}
    with uproot.open(capture, object_cache=None, array_cache=None) as root:
        read = lambda name, fields=None: A.read_tree(root, "kktrace", name, cycles, fields)
        records = read("crossing_records")
        discovery = read("crossing_discovery", list(B.KEYS)+["sequence", "has_search", "search_exhausted"])
        report["population_census"] = D.census(records, discovery)
        del discovery
        seeds = read("seed_contract")
        constructors, hit_keys, selected_seeds, captured_hit_time = I.initial_inputs(records, seeds)
        del seeds, records
        targets = np.unique(hit_keys[:, :4], axis=0)
        iterations = read("iterations", list(B.KEYS)+["iteration_ordinal"])
        expected = B.integer_keys(iterations[A.column(iterations, "iteration_ordinal") == 0], B.KEYS)
        A.require(len(constructors) == 2538 and len(targets) == 147, "initial population census differs")
        A.require(len(expected) == len(targets), "initial target census differs")
        A.match_rows(expected, targets)
        trajectories = read("crossing_trajectories")
        q, ranges, counts, parent, mass = I.construction_inputs(trajectories, constructors, selected_seeds)
        seed_range = np.column_stack((A.column(selected_seeds, "range_begin"), A.column(selected_seeds, "range_end")))
        report["construction_trajectories"] = dict(n_contexts=len(counts), n_pieces=int(counts.sum()), single_piece=True,
            n_hits_with_different_postconstruction_seed_range=int(np.sum(np.any(ranges[parent, 0] != seed_range, axis=1))),
            construction_range_per_hit=ranges[parent, 0].tolist(), postconstruction_seed_range_per_hit=seed_range.tolist(),
            scope="makeStrawHits construction range precedes KKFit::range and the seed_contract snapshot; q/mass remain identical")
        del trajectories, selected_seeds, iterations
        events = A.read_tree(root, "trkdump", "events", cycles, list(B.KEYS[:3])+["s_"+k for k in I.EVENT_FIELDS])
        measured = I.measurements(events, hit_keys, A.column(constructors, "strawid", "iu"))
        del events
        bindings, tracker = read("crossing_geometry_bindings"), read("crossing_geometry_tracker")
        straws = read("crossing_geometry_straws", ["geometry_index", "fullsid", "tube_cal", "tube_hv", "wire_cal", "wire_hv"])
        geometry = I.geometry_inputs(constructors, bindings, tracker, straws)
        del bindings, tracker, straws
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        generated = I.generate(q, ranges, counts, parent, mass, geometry, measured, configuration)
    ids = B.integer_keys(constructors, B.OBJECT)
    report["comparison"] = B.compare(generated["crossing"], constructors, ids)
    report["hit_constructor_time"] = B.comparison(generated["hit"]["particle_time"], captured_hit_time, hit_keys)
    report["identities"] = dict(object_keys=ids.tolist(), hit_keys=hit_keys.tolist(), strawid=A.column(constructors, "strawid").tolist())
    report["generated"] = dict(hit_particle_hint=generated["hit_particle_hint"].tolist(), hit_sensor_hint=generated["hit_sensor_hint"].tolist())
    fields = ("particle_time", "sensor_time", "particle_poca", "sensor_poca", "direction", "sensor_direction", "index", "solution_index", "moves", "exhausted", "status", "solution_history")
    for label, state in (("hit", generated["hit"]), ("crossing", generated["crossing"]["state"])):
        report["generated"][label] = {k: state[k].tolist() for k in fields}
    report["generated"].update({k: generated["crossing"][k].tolist() for k in ("offset", "effect_time", "reference_speed", "particle_hint", "sensor_hint")})
    report.update(n_targets=len(targets), n_hit_search_moves=int(np.sum(generated["hit"]["moves"] > 0)),
                  n_hit_search_caps=int(generated["hit"]["exhausted"].sum()))


def run(args):
    args = SimpleNamespace(**{k: Path(getattr(args, k)).resolve() for k in
                              ("out", "full_check", "associated", "capture", "expanded", "native_source")})
    protected = [args.full_check, args.associated, args.capture, args.expanded]
    A.require(args.out not in protected and not args.out.is_relative_to(Path(__file__).resolve().parents[1])
              and not args.out.is_relative_to(args.native_source), "protected report path")
    with args.out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = dict(completed=False, passed=False, full_fit_closure=False, full_f1_closure=False,
                      full_history_closure=False, full_numerical_validation=False, independent_discovery_replayed=False,
                      rtol=B.RTOL, atol=B.ATOL, limits=LIMITS)
        try:
            report["executing_modules"] = helpers(watched)
            full = D.receipt(args.full_check, D.FULL_SHA, watched,
                             ("passed", "full_target_capture_valid", "hashes_unchanged", "full_target_coverage"))
            for key, n in (("n_events", 3000), ("n_seeds", 2319), ("n_targets", 147), ("n_iterations", 3449)):
                A.require(type(full.get(key)) is int and full[key] == n, "full receipt census differs")
            associated = D.receipt(args.associated, ASSOCIATED_SHA, watched, ("completed", "passed", "hashes_unchanged"))
            A.require(associated["comparison"]["exact"] is True, "associated certificate not exact")
            manifest = full["before_sha256"]
            bound_input(args.capture, manifest, watched)
            A.require(manifest[str(args.capture)] == D.CAPTURE_SHA, "wrong full capture")
            bound_input(args.expanded, manifest, watched)
            for relative in ("Offline/Mu2eKinKal/inc/KKFit.hh", "Offline/Mu2eKinKal/inc/KKStrawXing.hh",
                             "Offline/Mu2eKinKal/src/KKFitUtilities.cc", "TrkFastSim/inc/IterationCaptureWriter.hh",
                             "TrkFastSim/src/KinematicLineFitTrace_module.cc"):
                bound_input(args.native_source/relative, manifest, watched)
            for relative, sha in D.SOURCE_PINS.items():
                D.watch(args.native_source/relative, watched, sha)
            config = I.speed_configuration(args.expanded)
            report["speed_configuration"] = config
            report["inputs"] = {k: str(getattr(args, k)) for k in ("capture", "full_check", "associated", "expanded", "native_source")}
            population(args.capture, config, report)
            report.update(completed=True, passed=report["comparison"]["passed"] and report["hit_constructor_time"]["passed"],
                          exact=report["comparison"]["exact"] and report["hit_constructor_time"]["exact"])
        except Exception as error:
            report.update(error=f"{type(error).__name__}: {error}", passed=False)
        finally:
            end, errors = {}, []
            for path in watched:
                try:
                    end[path] = A.digest(path)
                except Exception as error:
                    errors.append(f"{path}: {type(error).__name__}: {error}")
            report.update(before_sha256=watched, end_sha256=end, hashes_unchanged=bool(watched) and end == watched and not errors, hash_errors=errors)
            if not report["hashes_unchanged"]:
                report.update(passed=False, integrity_error="Evidence/source changed during audit")
            report, text = D.serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-check", "associated", "capture", "expanded", "native-source", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "passed", "exact", "hashes_unchanged", "error", "n_targets")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
