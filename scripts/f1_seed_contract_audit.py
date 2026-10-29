"""Pinned native seed decomposition; no refit, captured seed q/C input, or defaults changed."""
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import uproot

import f1_seed_contract as S
import f1_initial_crossing_audit as E
from f1_initial_crossing_audit import I, D, A, B

INITIAL_SHA = "100542dfd9b0f6287c06b79c4a3a775883c1d602580418f703112fbc5a2acbf7"
NEW_SOURCE_PINS = {
    "Offline/CosmicReco/src/LineFinder_module.cc": "9b2d470a099d4e2a2784bb8093ac1965990fb3cc7187065289baa468362552d6",
    "Offline/RecoDataProducts/inc/CosmicTrack.hh": "ea2bf214b356f4caf06ab256874f9184394f7cdde38a08dd30b8106a2e1d2b7a",
    "KinKal/Trajectory/POCAUtil.cc": "42043a5075353ebfc7656a762f15098639300afcb05cd0e28edb0ced1eecd8fd",
    "KinKal/Trajectory/POCAUtil.hh": "9cab5eac2f2afad14581d21980150cf146dbb64e0062d03b477195f4fd4a1f0c",
    "KinKal/General/Vectors.hh": "01eeedc34290223cf565667a4e3f16c99bbaccdd83bae3a45375fa73c0fae4a5",
}
ROOT_HEADERS = Path("/cvmfs/mu2e.opensciencegrid.org/spackages/241207/spack/var/spack/environments/muse-al9-prof-e29-p109/.spack-env/view/include/root/Math/GenVector")
LIMITS = [
    "All2319 captured seed rows, no fit-status or eligible-track filtering;147 targets have complete raw geometry for regenerated hit/crossing ranges.",
    "Inputs are saved LineFinder FitParams/source-hit membership, measured ComboHits, supplied native mass, raw target geometry and source-FCL constants; no autonomous LineFinder or association discovery claim.",
    "Frozen LineFinder assigns identical FitParams/MinuitParams/_t0 values; this source equivalence is producer-specific, not valid for arbitrary CosmicTrackSeed producers.",
    "New upstream source/header hashes describe the inspected implementation, not an additional historical binary provenance certificate.",
    "All-seed range/constraint-time checks use captured initial hit/crossing times; the separate147-target range uses independently generated seed q and construction range plus generated CAs/offsets.",
    "Native seed covariance is configured diagonal KL covariance, not propagated LineFinder covariance; no regularization or tolerance changes.",
    "Numeric pass uses unchanged RTOL1e-5 ATOL1e-7; exact is reported independently, not universal ROOT/compiler parity.",
    "No fitter q/C/range ablation, prior/filter/material/decision/acceptance or full F1 closure; combined historical seed fit arms remain unchanged.",
]


def helpers(watched):
    result = E.helpers(watched)
    root = Path(__file__).resolve().parents[1]
    for name, sha in (("scripts/f1_initial_crossing.py", "3523bbeecbd4674fcb45f7e82aa75021cbbeba609062c200b647b61d75210c2b"),
                      ("scripts/f1_initial_crossing_audit.py", "b7ffbe665b73aa5155e08fb9ac09152ec9df9cd20ab20d6218efba2aa7a0df26")):
        D.watch(root/name, watched, sha)
    for name, module in (("scripts/f1_seed_contract.py", S), ("scripts/f1_seed_contract_audit.py", __import__(__name__))):
        path = root/name
        A.require(Path(module.__file__).resolve() == path, "executing seed module path differs")
        D.watch(path, watched)
        required = ("configuration", "inputs", "bounds", "dot", "unit", "generate", "constraint") if module is S else ("helpers", "flat", "compare_all", "compare_targets", "population", "run", "main")
        for field in required:
            D.code_origin(getattr(module, field, None), path)
        for value in vars(module).values():
            if inspect.isfunction(value) and value.__module__ == module.__name__:
                D.code_origin(value, path)
        result[name] = str(path)
    A.require(S.B is B and E.I is I and E.D is D, "seed helper binding differs")
    for name in ("array", "column", "match_rows", "require"):
        A.require(getattr(S, name) is getattr(A, name), "seed alias differs: "+name)
    return result


def flat(seeds, name, counts):
    A.require(A.array(ak.num(seeds[name], axis=1)) == counts, "seed collection lengths differ: "+name)
    return A.array(ak.flatten(seeds[name], axis=1))


def compare_all(events, seeds, config):
    """Independent q/C, ordered membership, then explicitly conditional all-seed range."""
    ids = B.integer_keys(seeds, B.KEYS)
    A.require(A.column(seeds, "schema_version", "iu") == 1, "unsupported seed schema")
    A.require(B.boolean(seeds, "valid"), "invalid seed capture")
    source = S.inputs(events, ids)
    mass = A.column(seeds, "mass", "f")
    A.require(mass == 105.658367, "unsupported supplied native mass")
    generated = S.generate(source["raw"], mass, config)
    n = len(ids)
    comps = {}
    def check(name, actual, expected, keys=ids):
        comps[name] = B.comparison(actual, expected, keys, exact=np.asarray(actual).dtype.kind in "biu")
    check("parameters", generated["parameters"], A.dense(seeds, "parameters", 6))
    check("covariance", generated["covariance"].reshape(n, 36), A.dense(seeds, "covariance", 36))
    check("bnom", np.broadcast_to([0., 0., .001], (n, 3)), A.dense(seeds, "bnom", 3))
    check("charge", np.full(n, -1, dtype=np.int64), A.column(seeds, "charge", "iu"))
    counts = np.bincount(source["parent"], minlength=n)
    ordered = source["hit_keys"]
    for name in ("source_shindices", "sh_shindex", "sx_partner_shindex"):
        check(name, ordered[:, -1], flat(seeds, name, counts), ordered)
    for name in ("sh_strawid", "sx_strawid"):
        check(name, source["strawid"], flat(seeds, name, counts), ordered)
    local = np.arange(len(ordered))-np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    for name in ("sh_index", "sx_index", "sx_partner_initial_index"):
        check(name, local, flat(seeds, name, counts), ordered)
    times = np.concatenate((flat(seeds, "sh_time", counts), flat(seeds, "sx_time", counts)))
    ranges = S.bounds(np.tile(source["parent"], 2), times, n)
    check("range_from_captured_times", ranges, np.column_stack((A.column(seeds, "range_begin"), A.column(seeds, "range_end"))))
    ph = S.constraint(generated["parameters"], ranges, config)
    check("constraint_parameters", ph["parameters"], A.dense(seeds, "ph_parameters", 6))
    check("constraint_covariance", ph["covariance"].reshape(n, 36), A.dense(seeds, "ph_covariance", 36))
    check("constraint_mask", ph["mask"], A.dense(seeds, "ph_mask", 6))
    check("constraint_time_from_captured_times", ph["time"], A.dense(seeds, "ph_time", 1)[:, 0])
    check("constraint_index", np.zeros((n, 1), dtype=np.int64), A.dense(seeds, "ph_index", 1))
    momentum = generated["parameters"][:, 5]
    diagnostic = dict(n_seeds=n, n_initial_hits=len(ordered), n_parallel=int(generated["parallel"].sum()),
                      n_non_nominal_momentum=int(np.sum(momentum != 2000.)),
                      momentum_minus_2000_quantiles=np.quantile(momentum-2000., [0, .1, .5, .9, 1]).tolist(),
                      generated_momentum_exact=bool(np.array_equal(momentum, A.dense(seeds, "parameters", 6)[:, 5])))
    return source, generated, comps, diagnostic


def compare_targets(root, seeds, source, generated, config, speed_config, cycles):
    """Regenerate target hit/crossing CAs on generated q/range, not native seed states."""
    read = lambda name, fields=None: A.read_tree(root, "kktrace", name, cycles, fields)
    records = read("crossing_records")
    constructors, hit_keys, selected, recorded_time = I.initial_inputs(records, seeds)
    del records
    ids = B.integer_keys(seeds, B.KEYS)
    target_keys = np.unique(hit_keys[:, :4], axis=0)
    seedrows = A.match_rows(target_keys, ids)
    parent = A.match_rows(hit_keys[:, :4], target_keys)
    q = generated["parameters"][seedrows, None, :]
    ranges = source["construction_range"][seedrows, None, :]
    counts = np.ones(len(target_keys), dtype=np.int64)
    mass = A.column(seeds, "mass")[seedrows[parent]]
    trajectories = read("crossing_trajectories")
    t = trajectories[A.column(trajectories, "category", "iu") == 0]
    tk = B.integer_keys(t, B.TRAJECTORY+("piece_index",))
    A.require(len(t) == len(target_keys) and np.all(tk[:, -1] == 0), "initial context population differs")
    linked = A.match_rows(B.integer_keys(constructors, B.TRAJECTORY, unique=False), B.integer_keys(t, B.TRAJECTORY))
    A.require(len(np.unique(linked)) == len(t), "unreferenced initial construction context")
    t = t[A.match_rows(target_keys, B.integer_keys(t, B.KEYS))]
    comps = dict(construction_parameters=B.comparison(q[:, 0], A.dense(t, "parameters", 6), target_keys),
                 construction_covariance=B.comparison(generated["covariance"][seedrows].reshape(len(t), 36), A.dense(t, "covariance", 36), target_keys),
                 construction_range=B.comparison(ranges[:, 0], A.dense(t, "range", 2), target_keys))
    del trajectories, t
    events = A.read_tree(root, "trkdump", "events", cycles, list(B.KEYS[:3])+["s_"+k for k in I.EVENT_FIELDS])
    measured = I.measurements(events, hit_keys, A.column(constructors, "strawid", "iu"))
    bindings, tracker = read("crossing_geometry_bindings"), read("crossing_geometry_tracker")
    straws = read("crossing_geometry_straws", ["geometry_index", "fullsid", "tube_cal", "tube_hv", "wire_cal", "wire_hv"])
    geometry = I.geometry_inputs(constructors, bindings, tracker, straws)
    rebuilt = I.generate(q, ranges, counts, parent, mass, geometry, measured, speed_config)
    ca = B.compare(rebuilt["crossing"], constructors, B.integer_keys(constructors, B.OBJECT))
    comps["hit_time"] = B.comparison(rebuilt["hit"]["particle_time"], recorded_time, hit_keys)
    fitrange = S.bounds(np.tile(parent, 2), np.concatenate((rebuilt["hit"]["particle_time"], rebuilt["crossing"]["effect_time"])), len(target_keys))
    chosen = seeds[seedrows]
    comps["generated_fit_range"] = B.comparison(fitrange, np.column_stack((A.column(chosen, "range_begin"), A.column(chosen, "range_end"))), target_keys)
    ph = S.constraint(q[:, 0], fitrange, config)
    comps["generated_constraint_time"] = B.comparison(ph["time"], A.dense(chosen, "ph_time", 1)[:, 0], target_keys)
    A.require(len(target_keys) == 147 and len(constructors) == 2538, "target census differs")
    return dict(n_targets=len(target_keys), n_constructors=len(constructors), comparison=comps, crossing_comparison=ca,
                generated=dict(keys=target_keys.tolist(), parameters=q[:, 0].tolist(), construction_range=ranges[:, 0].tolist(),
                               fit_range=fitrange.tolist(), constraint_time=ph["time"].tolist()))


def population(args, report):
    cycles = report["highest_cycles"] = {}
    config = report["configuration"] = S.configuration(args.expanded)
    with uproot.open(args.capture, object_cache=None, array_cache=None) as root:
        seeds = A.read_tree(root, "kktrace", "seed_contract", cycles)
        fields = list(B.KEYS[:3])+["sd_"+k for k in S.SEED_FIELDS]+["sd_n", "sd_ind", "s_time", "s_sid"]
        events = A.read_tree(root, "trkdump", "events", cycles, fields)
        A.require(len(events) == 3000 and len(seeds) == 2319, "seed/event census differs")
        source, generated, comparisons, diagnostic = compare_all(events, seeds, config)
        report.update(all_seeds=comparisons, diagnostics=diagnostic)
        report["targets"] = compare_targets(root, seeds, source, generated, config, I.speed_configuration(args.expanded), cycles)
    results = list(comparisons.values())+list(report["targets"]["comparison"].values())+[report["targets"]["crossing_comparison"]]
    report.update(completed=True, passed=all(x["passed"] for x in results), exact=all(x["exact"] for x in results))


def run(args):
    args = SimpleNamespace(**{k: Path(getattr(args, k)).resolve() for k in ("out", "capture", "expanded", "native_source", "full_check", "initial")})
    root = Path(__file__).resolve().parents[1]
    protected = (args.capture, args.expanded, args.full_check, args.initial)
    A.require(args.out not in protected and not args.out.is_relative_to(root) and not args.out.is_relative_to(args.native_source), "protected report path")
    with args.out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = dict(completed=False, passed=False, full_fit_closure=False, full_f1_closure=False,
                      full_numerical_validation=False, independent_seed_finder=False, rtol=B.RTOL, atol=B.ATOL, limits=LIMITS)
        try:
            report["executing_modules"] = helpers(watched)
            full = D.receipt(args.full_check, D.FULL_SHA, watched, ("passed", "full_target_capture_valid", "hashes_unchanged", "full_target_coverage"))
            initial = D.receipt(args.initial, INITIAL_SHA, watched, ("completed", "passed", "exact", "hashes_unchanged"))
            A.require(initial["n_targets"] == 147 and full["n_seeds"] == 2319, "receipt census differs")
            manifest = full["before_sha256"]
            E.bound_input(args.capture, manifest, watched)
            A.require(manifest[str(args.capture)] == D.CAPTURE_SHA, "wrong capture")
            E.bound_input(args.expanded, manifest, watched)
            for name in ("TrkFastSim/src/KinematicLineFitTrace_module.cc", "TrkFastSim/src/StrawHitTableDumper_module.cc", "TrkFastSim/inc/IterationCaptureWriter.hh", "Offline/Mu2eKinKal/inc/KKFit.hh", "Offline/Mu2eKinKal/src/KKFitUtilities.cc"):
                E.bound_input(args.native_source/name, manifest, watched)
            for name, sha in {**D.SOURCE_PINS, **NEW_SOURCE_PINS}.items():
                D.watch(args.native_source/name, watched, sha)
            for name, sha in {
                "DisplacementVector3D.h": "c9edba47c39646831c385515536f91cd7bf258ddeef2125a51102f8148e914fb",
                "Cartesian3D.h": "3a05f149b0290aacdca2a8b4988cda5cae6e551c586961025217e41288d7f310",
                "PxPyPzM4D.h": "2c83c30caa513a6f9864eceb38cd7e0cf93b41b41b66d34f6dfc859d9ae05c5d",
            }.items():
                D.watch(ROOT_HEADERS/name, watched, sha)
            report["inputs"] = {k: str(getattr(args, k)) for k in ("capture", "expanded", "native_source", "full_check", "initial")}
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                population(args, report)
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
                report.update(passed=False, integrity_error="Evidence/source changed during seed audit")
            report, text = D.serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "capture", "expanded", "native-source", "full-check", "initial"):
        parser.add_argument("--"+name, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "passed", "exact", "hashes_unchanged", "error", "diagnostics")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
