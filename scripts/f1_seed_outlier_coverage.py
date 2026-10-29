"""Receipt-bound momentum outlier/acceptance coverage in preserved native history.

Only momentum NPZ artifacts are read. This is a full-key census/intersection,
not a fitter, trajectory replay, cause attribution or closure certificate.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import uproot

import f1_crossing_constructor_audit as D
import f1_seed_population_compare as Q

A, B = D.A, D.B
PINS = {"fixed": "72de580e575bc7d653d5697ffb7af0bfcd4fbffd6fe7b54ba3bf3029819e1ea0",
        "own": "3131e7f6b8781604e8823b11bc0685ef676b0182f429a3dc0be307c4908d766b"}
FULL_SHA = "c8e1225b1cc71ad82ff7055c77cd382c1b564956a1470522136d48575d53f8ec"
CAPTURE_SHA = "d8aad2725e45ac0839456605b375aff229e98af33b772a16cb007e23bc72a667"
SOURCE_PINS = {
    "scripts/f1_seed_population_compare.py": "cea73776b07b4494615fa2683a9bd113fbbc1a098afface51d05f4cbac450188",
    "scripts/f1_crossing_constructor_audit.py": "f3268c65ef0083dbaacd615d0dbb8bcca060d3c53617d84077c7c99515000bd7",
    "scripts/f1_crossing_constructor.py": "7a1dd18bd18f81a3f7ee69901112c78068cec988a7760752cbc5cc5963f213b5",
}
ARMS = ("parameters", "momentum", "geometry_time", "covariance", "range")
LIMITS = [
    "Preserved native histories cover only147 selected targets/3449 iterations, not the full eligible population.",
    "Full (run,subrun,event,iseed) intersection only; covered does not mean a failing operand was replayed or its cause identified.",
    "Fixed fidelity uses all both-good rows; nonfinite pulls and invalid stock errors fail, never shrink the denominator.",
    "Fixed fit rejections are separate from both-good momentum failures; own-state gains/losses use all eligible rows.",
    "Momentum-only includes prior/constraint-target consequences. No fits, recapture, tolerance changes or default promotion.",
    "Inherited before_sha256 inputs are rehashed; only momentum artifact entries are authenticated/loaded, not other arms.",
]
CLOSURE = dict(full_fit_closure=False, full_f1_closure=False, full_history_closure=False,
               full_numerical_validation=False, independent_discovery_replayed=False)


def helpers(watched):
    """Authenticate current callable origins as well as their on-disk sources."""
    root = Path(__file__).resolve().parents[1]
    modules = D.helpers(watched)
    current = sys.modules[__name__]
    for relative, module, required in (
        ("scripts/f1_seed_population_compare.py", Q, ("validate", "_pulls")),
        ("scripts/f1_seed_outlier_coverage.py", current,
         ("helpers", "history", "read_history", "membership", "safe_numeric", "example", "group",
          "coverage", "load_mode", "evidence", "serialize", "run", "main")),
    ):
        path = root/relative
        A.require(Path(module.__file__).resolve() == path, "executing module path differs: " + relative)
        D.watch(path, watched, SOURCE_PINS.get(relative))
        for name in required:
            D.code_origin(getattr(module, name, None), path)
        for function in vars(module).values():
            if inspect.isfunction(function) and function.__module__ == module.__name__:
                D.code_origin(function, path)
        modules[relative] = str(path)
    for relative, sha in SOURCE_PINS.items():
        D.watch(root/relative, watched, sha)
    A.require(A is D.A and B is D.B, "helper aliases differ")
    return modules


def history(columns, n_targets=None, n_iterations=None):
    """Validate scalar capture fields and zero-based contiguous per-key ordinals."""
    names = B.KEYS + ("iteration_ordinal",)
    values = [np.asarray(columns[name]) for name in names]
    A.require(all(x.ndim == 1 and x.dtype.kind in "iu" for x in values), "history integer columns required")
    A.require(len({len(x) for x in values}) == 1, "history column lengths differ")
    # Check each dtype before stacking: mixed signed/unsigned must not silently float.
    for x in values:
        A.require(np.all(x >= 0) and np.all(x.astype(np.uint64) <= np.iinfo(np.int64).max), "invalid history identity")
    keys = B.integer_matrix(np.column_stack([x.astype(np.int64) for x in values]), 5, "iteration keys")
    valid, error = np.asarray(columns["capture_valid"]), np.asarray(columns["capture_error"])
    A.require(valid.shape == (len(keys),) and valid.dtype.kind == "b" and valid.all(), "invalid iteration capture")
    # Uproot TTree strings use object dtype; RNTuple/analytic arrays may use U/S.
    A.require(error.shape == (len(keys),) and error.dtype.kind in "USO", "invalid capture error representation")
    A.require(np.all(error == (b"" if error.dtype.kind == "S" else "")), "iteration capture error")
    A.require(len(keys) > 0, "empty history")
    targets, parent, counts = np.unique(keys[:, :4], axis=0, return_inverse=True, return_counts=True)
    order = np.lexsort((keys[:, 4], parent))
    expected = np.arange(len(keys)) - np.repeat(np.cumsum(counts)-counts, counts)
    A.require(np.array_equal(keys[order, 4], expected), "iteration ordinal gaps")
    A.require(n_targets is None or len(targets) == n_targets, "history target census differs")
    A.require(n_iterations is None or len(keys) == n_iterations, "history iteration census differs")
    return targets, counts


def read_history(path, cycles):
    fields = list(B.KEYS) + ["iteration_ordinal", "capture_valid", "capture_error"]
    with uproot.open(path, object_cache=None, array_cache=None) as root:
        directory = A.highest(root, "kktrace")
        tree = A.highest(root[directory], "iterations")
        cycles["kktrace/iterations"] = directory + "/" + tree
        columns = root[directory][tree].arrays(fields, library="np")
    return history(columns, 147, 3449)


def membership(keys, captured):
    """Vectorized full-key membership, never event-only identity or Python row loops."""
    keys = B.integer_matrix(keys, 4, "population keys")
    captured = B.integer_matrix(captured, 4, "captured keys")
    _, inverse = np.unique(np.concatenate((keys, captured)), axis=0, return_inverse=True)
    return np.isin(inverse[:len(keys)], inverse[len(keys):])


def safe_numeric(value):
    """Preserve signed finite values; nonfinite entries are explicit null plus flags."""
    value = np.asarray(value)
    finite = np.isfinite(value)
    clean = value.astype(object)
    clean[~finite] = None
    return dict(values=clean.tolist(), nonfinite=(~finite).tolist())


def example(payload, pulls, row):
    result = dict(key=payload["keys"][row].tolist(), good=bool(payload["good"][row]),
                  stockok=bool(payload["stockok"][row]), signed_pulls=safe_numeric(pulls[row]),
                  kl=safe_numeric(payload["kl"][row]), stock_params=safe_numeric(payload["stock_params"][row]),
                  stock_errors=safe_numeric(payload["stock_errors"][row]))
    for field in ("stat", "niter", "kl_status"):
        if field in payload:
            result[field] = safe_numeric(payload[field][row])
    return result


def group(payload, pulls, mask, covered):
    keys = payload["keys"]
    order = np.lexsort(keys.T[::-1])
    selected = order[mask[order]]
    yes, no = selected[covered[selected]], selected[~covered[selected]]
    return dict(n=len(selected), n_covered=len(yes), n_uncovered=len(no),
                keys=keys[selected].tolist(), covered_keys=keys[yes].tolist(), uncovered_keys=keys[no].tolist(),
                first_covered_example=example(payload, pulls, yes[0]) if len(yes) else None)


def coverage(fixed, own, captured):
    """Pure in-memory coverage kernel; toy populations need not have native census sizes."""
    captured = B.integer_matrix(captured, 4, "captured keys")
    result = {}
    for name, payload in (("fixed", fixed), ("own", own)):
        n = Q.validate(payload)
        keys = B.integer_matrix(payload["keys"], 4, name + " keys")
        covered = membership(keys, captured)
        pulls = Q._pulls(payload["kl"], payload["stock_params"], payload["stock_errors"])
        within = np.isfinite(pulls) & (np.abs(pulls) < .1)
        both = payload["good"] & payload["stockok"]
        masks = dict(all_eligible=np.ones(n, dtype=bool), both_good=both)
        if name == "fixed":
            A.require(payload["stockok"].all(), "fixed population includes native rejection")
            masks.update(momentum_failures=both & ~within[:, 5], fit_rejected=~payload["good"])
            masks.update({coordinate + "_failures": both & ~within[:, i] for i, coordinate in enumerate(Q.KL[:5])})
        else:
            masks.update(gains=payload["good"] & ~payload["stockok"],
                         losses=~payload["good"] & payload["stockok"],
                         neither=~payload["good"] & ~payload["stockok"])
        result[name] = dict(n=n, stock_good=int(payload["stockok"].sum()), f1_good=int(payload["good"].sum()),
            both_good=int(both.sum()), groups={key: group(payload, pulls, mask, covered) for key, mask in masks.items()},
            fidelity={coordinate: dict(n=int(both.sum()), n_pass=int((both & within[:, i]).sum()),
                n_fail=int((both & ~within[:, i]).sum()),
                n_nonfinite=int((both & ~np.isfinite(pulls[:, i])).sum())) for i, coordinate in enumerate(Q.KL)})
    return result


def load_mode(data, mode, watched, inputs):
    path = (data/f"seed_population_{mode}_v1"/"manifest.json").resolve()
    document = D.receipt(path, PINS[mode], watched, ("completed", "controls_passed", "hashes_unchanged"))
    A.require(document.get("mode") == mode and tuple(document.get("arms", {})) == ARMS, "population scope differs")
    artifact = document["artifacts"]["momentum"]
    raw_path = artifact["path"]
    A.require(isinstance(raw_path, str) and Path(raw_path).is_absolute() and not raw_path.startswith("/pnfs/"), "invalid momentum artifact path")
    artifact_path = Path(raw_path).resolve()
    A.require(artifact_path.suffix == ".npz", "momentum artifact is not NPZ")
    D.watch(artifact_path, watched, artifact["sha256"])
    inputs[mode] = dict(manifest=str(path), momentum=str(artifact_path), sha256=artifact["sha256"])
    with np.load(artifact_path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    n = Q.validate(payload)
    A.require(n == (2151 if mode == "fixed" else 2315), "population census differs")
    if mode == "fixed":
        A.require(payload["stockok"].all() and int(payload["good"].sum()) == 2139, "fixed acceptance census differs")
    return payload


def evidence(data, watched, report):
    directory = data/"crossing_3k_20261028_v2"
    full_path, capture = (directory/"crossing_full_check_v1.json").resolve(), (directory/"capture.root").resolve()
    full = D.receipt(full_path, FULL_SHA, watched,
                     ("passed", "full_target_capture_valid", "hashes_unchanged", "full_target_coverage"))
    for name, count in (("n_events", 3000), ("n_seeds", 2319), ("n_targets", 147), ("n_iterations", 3449)):
        A.require(type(full.get(name)) is int and full[name] == count, "full receipt census differs")
    A.require(full["before_sha256"].get(str(capture)) == CAPTURE_SHA, "capture not bound by full gate")
    D.watch(capture, watched, CAPTURE_SHA)
    report["inputs"] = dict(full_check=str(full_path), capture=str(capture))
    report["highest_cycles"] = {}
    captured, counts = read_history(capture, report["highest_cycles"])
    report["history"] = dict(n_targets=len(captured), n_iterations=int(counts.sum()), keys=captured.tolist(),
                              iteration_counts=counts.tolist(), unique_contiguous_ordinals=True, capture_valid=True)
    fixed = load_mode(data, "fixed", watched, report["inputs"])
    own = load_mode(data, "own", watched, report["inputs"])
    # Independently confirm the fixed census is exactly the native-good own census.
    reference = np.flatnonzero(own["stockok"])
    A.require(len(reference) == len(fixed["keys"]), "fixed/own native-good census differs")
    rows = reference[A.match_rows(B.integer_matrix(fixed["keys"], 4, "fixed keys"),
                                  B.integer_matrix(own["keys"][reference], 4, "own native-good keys"))]
    for field in ("stock_params", "stock_errors"):
        A.require(np.array_equal(fixed[field], own[field][rows], equal_nan=True), "fixed/own stock references differ")
    report["coverage"] = coverage(fixed, own, captured)
    A.require(report["coverage"]["fixed"]["groups"]["momentum_failures"]["n"] == 218, "fixed momentum failure census differs")


def serialize(report):
    try:
        return report, json.dumps(report, indent=2, allow_nan=False) + "\n"
    except (ValueError, TypeError, OverflowError) as error:
        failure = dict(completed=False, passed=False, **CLOSURE, limits=LIMITS,
                       error=f"strict JSON serialization failed: {type(error).__name__}: {error}")
        for field in ("before_sha256", "end_sha256", "hashes_unchanged", "hash_errors"):
            failure[field] = report[field]
        return failure, json.dumps(failure, indent=2, allow_nan=False) + "\n"


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    A.require(not out.is_relative_to(Path(__file__).resolve().parents[1]), "protected checkout output")
    with out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = dict(completed=False, passed=False, **CLOSURE, limits=LIMITS,
                      key_order=list(B.KEYS), coordinate_order=list(Q.KL),
                      criterion="finite(Q._pulls) & abs(Q._pulls) < 0.1; both-good denominator",
                      example_selection="lexicographically first covered full key in each group")
        try:
            report["executing_modules"] = helpers(watched)
            evidence(data, watched, report)
            report.update(completed=True, passed=True)
        except Exception as error:
            report.update(passed=False, error=f"{type(error).__name__}: {error}")
        finally:
            end, errors = {}, []
            for path in watched:
                try:
                    end[path] = A.digest(path)
                except Exception as error:
                    errors.append(f"{path}: {type(error).__name__}: {error}")
            report.update(before_sha256=watched, end_sha256=end, hash_errors=errors,
                          hashes_unchanged=bool(watched) and watched == end and not errors)
            if not report["hashes_unchanged"]:
                report.update(passed=False, integrity_error="Evidence/source changed during coverage audit")
            report, text = serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data, args.out)
    print(json.dumps({key: result.get(key) for key in ("completed", "passed", "hashes_unchanged", "error")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
