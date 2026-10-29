"""Hash-bound paired seed-population outcomes and preserved combined-full result."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import f1_seed_population as R
from f1_seed_population import A, B, D, K, Q

PINS = {"own": "3131e7f6b8781604e8823b11bc0685ef676b0182f429a3dc0be307c4908d766b",
        "fixed": "72de580e575bc7d653d5697ffb7af0bfcd4fbffd6fe7b54ba3bf3029819e1ea0"}
FULL_PINS = {
    "own": {"ids.npz": "8e438c64fc6e9d171ef553d5bbb2979be631ae55d4345e5fa8ef6ecac0ed6c7a",
            "vr2f1.json": "fc0e516f3657f4d994b12bf74cf9ad28d65c72938c87f9194bf6f20188ecef8c"},
    "fixed": {"ids.npz": "796dc49f2acaccc29731dbf3ed5a3f062eb9b4a2480a0311fe90cdbf2340050c",
              "vr2f1.json": "f0522aac748cda70fd28356c08ad914b0fda6f9e771a54bd892091267e7def59"}}


def compare(left, right):
    Q.validate(left)
    Q.validate(right)
    rows = A.match_rows(left["keys"], right["keys"])
    A.require(len(rows) == len(right["keys"]), "comparison census differs")
    for field in ("stockok", "stock_params", "stock_errors"):
        A.require(np.array_equal(left[field], right[field][rows], equal_nan=True), "stock comparison differs")
    good = left["good"] & right["good"][rows] & left["stockok"]
    lp, rp = Q._pulls(left["kl"], left["stock_params"], left["stock_errors"]), Q._pulls(right["kl"][rows], left["stock_params"], left["stock_errors"])
    lpass, rpass = np.isfinite(lp) & (np.abs(lp) < .1), np.isfinite(rp) & (np.abs(rp) < .1)
    result = dict(n=len(rows), common_both_good=int(good.sum()),
                  acceptance_identical=bool(np.array_equal(left["good"], right["good"][rows])), parameters={})
    for i, name in enumerate(Q.KL):
        changed = good & (lpass[:, i] != rpass[:, i])
        gain = good & lpass[:, i] & ~rpass[:, i]
        loss = good & ~lpass[:, i] & rpass[:, i]
        result["parameters"][name] = dict(n_changed=int(changed.sum()), gains=int(gain.sum()), losses=int(loss.sum()),
            changed_keys=left["keys"][changed].tolist(), gain_keys=left["keys"][gain].tolist(), loss_keys=left["keys"][loss].tolist(),
            numeric_exact=bool(np.array_equal(left["kl"][good, i], right["kl"][rows[good], i], equal_nan=True)))
    return result


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def mode(directory, full_dir, name, watched):
    manifest = D.receipt(directory/"manifest.json", PINS[name], watched, ("completed", "controls_passed", "hashes_unchanged"))
    A.require(manifest["mode"] == name and tuple(manifest["arms"]) == R.ARMS, "population scope differs")
    arrays = {}
    for arm, artifact in manifest["artifacts"].items():
        D.watch(artifact["path"], watched, artifact["sha256"])
        arrays[arm] = load(artifact["path"])
    ref = arrays["momentum"]
    base_path = Path(next(p for p in manifest["before_sha256"] if p.endswith(f"/{name}_stock_native/ids.npz")))
    D.watch(base_path, watched, R.BASE_PINS[name]["ids.npz"])
    for file, sha in FULL_PINS[name].items():
        D.watch(full_dir/file, watched, sha)
    fullmeta = D.strict_json(full_dir/"vr2f1.json")
    A.require(fullmeta["native_time"] == manifest["native_time"]
              and fullmeta["reference_model"] == "previous" and fullmeta["crossing_time"] == "current",
              "combined-full configuration differs")
    for label, path in (("baseline", base_path), ("full", full_dir/"ids.npz")):
        values = load(path)
        rows = A.match_rows(B.integer_matrix(values["event_index"][:, None], 1, label+" events"),
                            B.integer_matrix(ref["event_index"][:, None], 1, "current events"))
        A.require(len(rows) == len(ref["keys"]), "preserved census differs")
        A.require(np.array_equal(values["subrun"], ref["keys"][rows, 1])
                  and np.array_equal(values["event"], ref["keys"][rows, 2]), "preserved identity differs")
        values["keys"] = ref["keys"][rows]
        values["kl"] = np.asarray(jax.jit(jax.vmap(K.kl_params))(jnp.asarray(values["xt0"])))
        arrays[label] = values
    return dict(summaries={arm: Q.summary(value)["fidelity"] for arm, value in arrays.items()},
                momentum_vs_baseline=compare(ref, arrays["baseline"]),
                momentum_vs_parameters=compare(ref, arrays["parameters"]),
                momentum_vs_full=compare(ref, arrays["full"]),
                range_vs_baseline=compare(arrays["range"], arrays["baseline"]))


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    A.require(not out.is_relative_to(Path(__file__).resolve().parents[1]), "protected checkout output")
    with out.open("x", encoding="utf-8") as stream:
        watched, report = {}, dict(completed=False, passed=False, full_fit_closure=False, full_f1_closure=False,
            full_numerical_validation=False, limits=[
                "Paired acceptance/fidelity predicates, not bitwise fit equality or independent reconstruction.",
                "Momentum-only includes initial prior/constraint-target consequences; baseline geometry/time/C/range are held fixed.",
                "Ordinary and combined-full outputs reused; no native or F1 fits run by this report.",
                "All old conditioning and incomplete native-history coverage remain; numerical closure flags false."])
        try:
            R.helpers(watched)
            D.watch(__file__, watched)
            for function in (compare, load, mode, run, main):
                D.code_origin(function, Path(__file__).resolve())
            for name in ("own", "fixed"):
                report[name] = mode(data/f"seed_population_{name}_v1", data/"seed_arms_v1"/f"{name}_stock_native_seednative", name, watched)
            report.update(completed=True, passed=True)
        except Exception as error:
            report.update(error=f"{type(error).__name__}: {error}")
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
                report["passed"] = False
            report, text = D.serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data, args.out)
    print(json.dumps({k: result.get(k) for k in ("completed", "passed", "hashes_unchanged", "error")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
