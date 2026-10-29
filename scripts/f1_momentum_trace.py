"""Bounded actual-F1 momentum-only traces with preserved-population compatibility."""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import uproot

import f1_seed_population as R
import f1_seed_outlier_coverage as C
import f1_trace_compare as T
from f1_seed_population import A, B, D, F, K, P, Q
from trkfastsim.reco import fittrace

COVERAGE_SHA = "e3a68e4fcd67c86c385900b97950b92f30506906991fe09c827740ca1bde7984"
LIMITS = [
    "One explicit captured target, not a population rerun or attribution for198 uncovered momentum failures/8 uncovered fixed rejections.",
    "Momentum-only includes prior/constraint-target consequences; ordinary first5 fitter coordinates/C/range stay fixed inside the same JIT.",
    "Fixed states are saved FINAL stock hit states, not native iteration-local states; initial activity differences can be intentional conditioning.",
    "Unbatched compact execution differs from production vmap/padding; saved projection compatibility is separate from trace passivity, not population bitwise certification.",
    "Cap3/10 is selected from saved cumulative F1 iterations, not native chronology; only trace.valid records represent executed iterations.",
    "Existing first-iteration comparator phase order is report order, not proof of causal precedence. Later histories/candidates/decisions remain uncompared.",
    "Native material conversion is first-order. No native recapture, fitter/default/tolerance/regularization change or full closure.",
]


def helpers(watched):
    runtime = R.helpers(watched)
    root = Path(__file__).resolve().parents[1]
    for module, names in ((sys.modules[__name__], ("helpers", "cap", "kernel", "executed", "save_tree", "prepare", "native_first", "execute", "serialize", "run", "main")),
                          (T, ("compare_first",)), (fittrace, ("shapes", "allocation_bytes", "empty"))):
        path = Path(module.__file__).resolve()
        A.require(path.is_relative_to(root), "trace module from different checkout")
        D.watch(path, watched)
        for name in names:
            D.code_origin(getattr(module, name, None), path)
        for function in vars(module).values():
            if inspect.isfunction(function) and function.__module__ == module.__name__:
                D.code_origin(function, path)
    D.watch(Path(C.__file__), watched)
    return runtime


def cap(niter):
    values = np.asarray(niter)
    A.require(values.ndim == 1 and len(values) > 0 and values.dtype.kind in "iu", "invalid cumulative iterations")
    A.require(np.all(values <= np.iinfo(np.int64).max), "iteration overflow")
    delta = np.diff(np.r_[np.int64(0), values.astype(np.int64)])
    A.require(np.all(delta >= 0) and delta.sum() > 0, "nonmonotonic/empty executed history")
    return 10 if np.any(delta >= 3) else 3


def kernel(arm, x, h, c, native, time, drift, stages, ann, *, maxniter, tracing=False):
    A.require(arm in ("ordinary", "control", "momentum"), "unsupported trace arm")
    A.require(type(maxniter) is int and maxniter in (3, 10), "unsupported iteration cap")
    A.require(type(tracing) is bool and (not tracing or arm == "momentum"), "only momentum tracing supported")
    fixed = dict(active=h["active"], lr=h["lr"], nulldv=jnp.ones_like(h["active"]))
    if not tracing:
        return F.fit(arm, x, h, c, drift, stages, ann, native, time, fixed=fixed, maxniter=maxniter)
    base = F.S.baseline(x, h, c)
    seed = F.select("momentum", base, native)
    return K.fit_track(x, h, c, drift, stages, ann, fixed=fixed, maxniter=maxniter,
        time_model="stock", reference_model="previous", crossing_time="current", xing_activity="hit_or_doca",
        constraint_time=time, seed_input=seed, trace=True)


def executed(result):
    cf, hist, good = result
    A.require("trace" in cf, "missing trace")
    ordinary = (dict(cf), hist, good)
    records = {name: np.asarray(value) for name, value in ordinary[0].pop("trace").items()}
    valid = records["valid"]
    A.require(valid.ndim == 2 and valid.dtype.kind == "b" and valid.any(), "invalid/empty trace validity")
    A.require(int(valid.sum()) == int(cf["niter"]), "trace count differs from executed iterations")
    stage, iteration = np.nonzero(valid)
    A.require(stage[0] == iteration[0] == 0, "missing initial executed iteration")
    A.require(np.array_equal(records["iteration"][valid], iteration), "trace local iteration mismatch")
    A.require(np.all(valid == (np.arange(valid.shape[1])[None] < valid.sum(1)[:, None])), "nonprefix executed stage records")
    kept = {name: value[valid] for name, value in records.items()}
    return ordinary, dict(kept, stage_index=stage)


def save_tree(out, name, result, report):
    leaves, _ = jax.tree_util.tree_flatten_with_path(result)
    values = {f"leaf_{i:03d}": np.asarray(value) for i, (_, value) in enumerate(leaves)}
    R.save(out, name, values, report)
    report["artifacts"][name]["leaf_paths"] = {f"leaf_{i:03d}": jax.tree_util.keystr(path) for i, (path, _) in enumerate(leaves)}


def prepare(args, watched, report):
    data_dir = args.data.resolve()
    coverage = D.receipt(data_dir/"seed_outlier_coverage_v1.json", COVERAGE_SHA, watched,
                         ("completed", "passed", "hashes_unchanged"))
    captured = B.integer_matrix(coverage["history"]["keys"], 4, "captured targets")
    A.match_rows(args.key[None], captured)
    manifest_path = data_dir/"seed_population_fixed_v1/manifest.json"
    manifest = D.receipt(manifest_path, C.PINS["fixed"], watched, ("completed", "controls_passed", "hashes_unchanged"))
    A.require(manifest["mode"] == "fixed" and tuple(manifest["arms"]) == R.ARMS, "population scope differs")
    artifact = manifest["artifacts"]["momentum"]
    D.watch(artifact["path"], watched, artifact["sha256"])
    with np.load(artifact["path"], allow_pickle=False) as source:
        population = {k: source[k] for k in source.files}
    A.require(Q.validate(population) == 2151 and population["stockok"].all(), "fixed population differs")
    selected = A.match_rows(args.key[None], B.integer_matrix(population["keys"], 4, "population keys"))
    expected = {k: v[selected] for k, v in population.items()}
    report["maxniter"] = cap(expected["niter"][0])
    baseline_paths = [p for p in manifest["before_sha256"] if p.endswith("/fixed_stock_native/ids.npz")]
    A.require(len(baseline_paths) == 1, "ambiguous baseline path")
    prep = SimpleNamespace(mode="fixed", seed_contract=data_dir/"seed_contract_v1.json",
        covariance_contract=data_dir/"seed_covariance_v1.json", check=Path(manifest["native_time"]["check"]),
        conditions=args.conditions.resolve(), straw_axis=args.straw_axis.resolve(), baseline=Path(baseline_paths[0]).parent)
    data = R.prepare(prep, watched, report)
    row = A.match_rows(args.key[None], data["keys"])[0]
    for name in ("event_index", "stockok", "stock_params", "stock_errors", "kl_status", "valid", "xvalid"):
        A.require(np.array_equal(expected[name], data["baseline"][name][row:row+1], equal_nan=True), "prepared reference differs: "+name)
    nh, nx = int(data["h"]["valid"][row].sum()), int(data["c"]["valid"][row].sum())
    inputs = (jnp.asarray(data["x"][row]), {k: jnp.asarray(v[row, :nh]) for k, v in data["h"].items()},
              {k: jnp.asarray(v[row, :nx]) for k, v in data["c"].items()},
              {k: jnp.asarray(v[row]) for k, v in data["native"].items()}, jnp.asarray(data["times"][row]))
    required = fittrace.allocation_bytes(nh, nx, len(data["stages"]["T"]), report["maxniter"])
    report.update(key=args.key.tolist(), n_hits=nh, n_crossings=nx, trace_bytes_bound=required,
                  population_manifest=str(manifest_path), population_artifact=artifact,
                  comparison_scope="fixed final stock states; compact single-target compiler context")
    A.require(required <= fittrace.MAX_BYTES, "trace allocation exceeds bound")
    source = Path(D.strict_json(prep.check)["inputs"]["baseline"]["path"])
    with uproot.open(source, object_cache=None, array_cache=None) as root:
        directory = A.highest(root, "trkdump")
        tree = root[directory][A.highest(root[directory], "events")]
        event = int(expected["event_index"][0])
        raw = tree.arrays(["run", "subrun", "event", "kh_ind", "kst_sid"], entry_start=event, entry_stop=event+1)
    A.require(np.array_equal(A.keys(raw, B.KEYS[:3]), args.key[None, :3]), "raw event identity differs")
    hids = A.dense(raw, "kh_ind", nh)[0].astype(np.int64)
    xids = A.dense(raw, "kst_sid", nx)[0].astype(np.int64)
    partner = np.asarray(inputs[2]["hit"])
    partner = np.where(partner >= 0, hids[np.maximum(partner, 0)], -1)
    identities = dict(key=args.key, hit_ids=hids, crossing_ids=xids, crossing_partner=partner,
                      seed_hit_mask=np.asarray(inputs[1]["inseed"]), hit_mid=np.asarray(inputs[1]["mid"]),
                      crossing_mid=np.asarray(inputs[2]["mid"]))
    return data, inputs, expected, row, identities, coverage


def native_first(capture, key, cycles):
    tables = {}
    with uproot.open(capture, object_cache=None, array_cache=None) as root:
        for name in ("iterations", "pieces", "effects"):
            table = A.read_tree(root, "kktrace", name, cycles)
            mask = np.all(A.keys(table, unique=False) == key, axis=1) & (A.column(table, "iteration_ordinal") == 0)
            if name == "pieces":
                mask &= A.column(table, "owner") == 0
            tables[name] = table[mask]
    A.require(len(tables["iterations"]) == 1, "missing native first iteration")
    return tables


def execute(data, inputs, expected, row, identities, capture, out, report):
    outputs = {}
    for arm, tracing in (("ordinary", False), ("control", False), ("momentum", False), ("momentum", True)):
        name = arm+"_traced" if tracing else arm
        call = jax.jit(lambda x, h, c, n, t: kernel(arm, x, h, c, n, t, data["drift"], data["stages"], data["ann"],
                                                   maxniter=report["maxniter"], tracing=tracing))
        result = call(*inputs)
        jax.block_until_ready(result)
        if tracing:
            save_tree(out, "momentum_traced_raw", result, report)
            result, records = executed(result)
            R.save(out, "trace", dict(records, **identities), report)
            report.update(n_iterations=len(records["valid"]), trace_sha256=report["artifacts"]["trace"]["sha256"])
        save_tree(out, name, result, report)
        outputs[name] = result
        if name == "control":
            report["baseline_injection_control"] = P.compare(result, outputs["ordinary"])
            A.require(report["baseline_injection_control"]["exact"], "ordinary/control differs")
        elif tracing:
            report["trace_passivity"] = P.compare(result, outputs["momentum"])
            A.require(report["trace_passivity"]["exact"], "trace changes momentum result")
        print(name+" saved", flush=True)
    projected = F.project(outputs["momentum"])
    projected = R.expand({k: np.asarray(v)[None] for k, v in projected.items()}, data["h"]["valid"].shape[1], data["c"]["valid"].shape[1])
    actual = R.payload(data, projected, np.array([row]))
    R.save(out, "projected", actual, report)
    report["population_compatibility"] = R.compatibility(actual, expected)
    report["single_target_fidelity"] = Q.summary(actual)
    A.require(report["population_compatibility"]["passed"], "compact execution differs from saved population")
    report["controls_passed"] = True
    report["highest_cycles"] = {}
    tables = native_first(capture, identities["key"], report["highest_cycles"])
    report["first_iteration_comparison"] = T.compare_first({k: v[0] for k, v in records.items() if k != "stage_index"},
        tables["iterations"], tables["pieces"], tables["effects"], identities["hit_ids"], identities["crossing_ids"],
        identities["seed_hit_mask"], identities["crossing_partner"])
    report.update(completed=True, passed=True)


def serialize(report):
    try:
        return report, json.dumps(report, indent=2, allow_nan=False)+"\n"
    except (TypeError, ValueError, OverflowError) as error:
        fields = ("artifacts", "key", "maxniter", "n_hits", "n_crossings", "n_iterations", "trace_sha256",
                  "trace_bytes_bound", "population_manifest", "population_artifact", "comparison_scope",
                  "before_sha256", "end_sha256", "hashes_unchanged", "hash_errors", "inputs", "execution")
        failure = {k: report[k] for k in fields if k in report}
        failure.update(completed=False, passed=False, controls_passed=False, full_fit_closure=False,
                       full_f1_closure=False, full_numerical_validation=False, limits=LIMITS,
                       error=f"strict JSON serialization failed: {type(error).__name__}: {error}")
        return failure, json.dumps(failure, indent=2, allow_nan=False)+"\n"


def run(args):
    args = SimpleNamespace(**vars(args))
    args.out = args.out.resolve()
    A.require(not args.out.is_relative_to(Path(__file__).resolve().parents[1]), "protected checkout output")
    args.out.mkdir(parents=True, exist_ok=False)
    watched, report = {}, dict(completed=False, passed=False, controls_passed=False, full_fit_closure=False,
        full_f1_closure=False, full_numerical_validation=False, limits=LIMITS)
    try:
        report["execution"] = helpers(watched)
        data, inputs, expected, row, identities, coverage = prepare(args, watched, report)
        capture = Path(coverage["inputs"]["capture"])
        D.watch(capture, watched, C.CAPTURE_SHA)
        execute(data, inputs, expected, row, identities, capture, args.out, report)
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", passed=False)
    finally:
        end, errors = {}, []
        for path in watched:
            try:
                end[path] = A.digest(path)
            except Exception as error:
                errors.append(f"{path}: {type(error).__name__}: {error}")
        report.update(before_sha256=watched, end_sha256=end, hash_errors=errors,
                      hashes_unchanged=bool(watched) and watched == end and not errors)
        for artifact in report.get("artifacts", {}).values():
            try:
                A.require(A.digest(artifact["path"]) == artifact["sha256"], "output hash changed")
            except Exception as error:
                errors.append(f"artifact: {type(error).__name__}: {error}")
        if not report["hashes_unchanged"] or errors:
            report.update(passed=False, controls_passed=False)
        report, text = serialize(report)
        with (args.out/"manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "out", "conditions", "straw-axis"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--key", type=P.identity, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "passed", "controls_passed", "hashes_unchanged", "error")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
