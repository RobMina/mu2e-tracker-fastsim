"""Supplement the sealed seed audit with source-typed float32 seed-error squares."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import uproot

import f1_seed_contract as S
import f1_crossing_constructor_audit as D
from f1_crossing_constructor_audit import A, B

SEED_SHA = "c1c7c19c7fc5cabdd85cd195ffa6973998a478fd9444cabd66882f3e677fa633"
SEED_HELPER_SHA = "e5ddfe5819df681bb8dd357d4f59fa8090fedbe541970daf658794c2a4a5fd64"
SETTINGS_SHA = "9e1f006c80cbd080a64ec5bcb2c4d3a1ef5de0d5ab8c6d312cc7395f43abd34d"
LIMITS = [
    "Supplement only: the original seed_contract_v1 receipt and its double-square covariance generator remain unchanged.",
    "Native KKModuleConfig SeedErrors is fhicl::Sequence<float>; the captured constructor squares float operands before promotion to double DMAT.",
    "All2319 seed covariances and147 initial construction covariances, no status filtering; captured matrices are comparison-only.",
    "All other seed parameters/membership/ranges/constraints retain the scope and exactness of the pinned original seed receipt.",
    "The inspected KKFitSettings.hh hash matches the earlier capture-area header already in that receipt; this adds no retrospective binary provenance.",
    "No fitter/default/tolerance change, regularization, population acceptance/fidelity or full F1/numerical closure.",
]


def covariance(config):
    """Square configured float operands, then promote their products to double."""
    A.require(config == dict(momentum=2000., errors=S.ERRORS.tolist(), constraints=S.CONSTRAINTS.tolist()),
              "unsupported seed configuration")
    errors = np.asarray(config["errors"], dtype=np.float32)
    return np.diag((errors*errors).astype(np.float64))


def helpers(watched):
    root = Path(__file__).resolve().parents[1]
    D.helpers(watched)
    seed_path = root/"scripts/f1_seed_contract.py"
    A.require(Path(S.__file__).resolve() == seed_path, "seed helper checkout differs")
    D.watch(seed_path, watched, SEED_HELPER_SHA)
    D.code_origin(S.configuration, seed_path)
    path = root/"scripts/f1_seed_covariance_audit.py"
    D.watch(path, watched)
    module = __import__(__name__)
    for name in ("covariance", "helpers", "compare", "population", "run", "main"):
        D.code_origin(getattr(module, name, None), path)
    for value in vars(module).values():
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            D.code_origin(value, path)


def compare(seeds, trajectories, target_keys, config):
    keys = B.integer_keys(seeds, B.KEYS)
    A.require(len(keys) == 2319, "seed covariance census differs")
    chosen = trajectories[A.column(trajectories, "category", "iu") == 0]
    tk = B.integer_keys(chosen, B.TRAJECTORY+("piece_index",))
    A.require(len(chosen) == 147 and np.all(tk[:, -1] == 0), "construction covariance census differs")
    target_keys = B.integer_matrix(target_keys, 4, "seed targets")
    A.require(len(target_keys) == 147, "target key census differs")
    A.match_rows(target_keys, B.integer_keys(chosen, B.KEYS))
    A.match_rows(B.integer_keys(chosen, B.KEYS), keys)
    matrix = covariance(config)
    seed_c = A.dense(seeds, "covariance", 36)
    initial_c = A.dense(chosen, "covariance", 36)
    metrics = dict(
        all_seeds=B.comparison(np.broadcast_to(matrix.reshape(36), seed_c.shape), seed_c, keys),
        initial_construction=B.comparison(np.broadcast_to(matrix.reshape(36), initial_c.shape), initial_c,
                                          B.integer_keys(chosen, B.KEYS)))
    ideal = np.diag(np.asarray(config["errors"], dtype=np.float64)**2)
    return dict(comparison=metrics, generated_diagonal=np.diag(matrix).tolist(),
                double_square_diagonal=np.diag(ideal).tolist(),
                generated_minus_double_diagonal=np.diag(matrix-ideal).tolist(),
                n_seeds=len(keys), n_targets=len(chosen),
                completed=True, passed=all(x["passed"] for x in metrics.values()),
                exact=all(x["exact"] for x in metrics.values()))


def population(seed, watched, report):
    inputs = seed["inputs"]
    source = Path(inputs["native_source"])
    D.watch(source/"Offline/Mu2eKinKal/inc/KKFitSettings.hh", watched, SETTINGS_SHA)
    capture = Path(inputs["capture"])
    expanded = Path(inputs["expanded"])
    D.watch(capture, watched, seed["before_sha256"][str(capture)])
    D.watch(expanded, watched, seed["before_sha256"][str(expanded)])
    config = S.configuration(expanded)
    cycles = report["highest_cycles"] = {}
    with uproot.open(capture, object_cache=None, array_cache=None) as root:
        seeds = A.read_tree(root, "kktrace", "seed_contract", cycles, list(B.KEYS)+["covariance"])
        fields = list(dict.fromkeys(B.TRAJECTORY+("piece_index", "category", "covariance")))
        trajectories = A.read_tree(root, "kktrace", "crossing_trajectories", cycles, fields)
        report.update(compare(seeds, trajectories, seed["targets"]["generated"]["keys"], config))


def run(seed_contract, out):
    seed_contract, out = Path(seed_contract).resolve(), Path(out).resolve()
    root = Path(__file__).resolve().parents[1]
    A.require(out != seed_contract and not out.is_relative_to(root), "protected output path")
    seed = D.strict_json(seed_contract)
    A.require(not out.is_relative_to(Path(seed["inputs"]["native_source"]))
              and str(out) not in seed["before_sha256"], "protected evidence path")
    with out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = dict(completed=False, passed=False, exact=False, full_fit_closure=False,
                      full_f1_closure=False, full_numerical_validation=False, limits=LIMITS,
                      rtol=B.RTOL, atol=B.ATOL, seed_contract=str(seed_contract), seed_contract_sha256=SEED_SHA)
        try:
            helpers(watched)
            seed = D.receipt(seed_contract, SEED_SHA, watched, ("completed", "passed", "hashes_unchanged"))
            A.require(seed["all_seeds"]["parameters"]["exact"] is True and seed["targets"]["n_targets"] == 147,
                      "seed receipt scope differs")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                population(seed, watched, report)
        except Exception as error:
            report.update(error=f"{type(error).__name__}: {error}", passed=False, exact=False)
        finally:
            end, errors = {}, []
            for path in watched:
                try:
                    end[path] = A.digest(path)
                except Exception as error:
                    errors.append(f"{path}: {type(error).__name__}: {error}")
            report.update(before_sha256=watched, end_sha256=end,
                          hashes_unchanged=bool(watched) and watched == end and not errors, hash_errors=errors)
            if not report["hashes_unchanged"]:
                report.update(passed=False, exact=False)
            report, text = D.serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed_contract, args.out)
    print(json.dumps({k: report.get(k) for k in ("completed", "passed", "exact", "hashes_unchanged", "error",
                                              "generated_diagonal", "generated_minus_double_diagonal")}, indent=2))
    raise SystemExit(0 if report["passed"] and report["exact"] else 1)


if __name__ == "__main__":
    main()
