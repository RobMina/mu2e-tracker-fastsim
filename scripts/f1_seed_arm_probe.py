"""Receipt-gated single-target F1 seed arms, never a full-population closure certificate."""
import argparse
import inspect
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import f1_seed_arms as S
import f1_crossing_constructor_audit as D
from f1_crossing_constructor_audit import A, B
from trkfastsim.reco import driftann, hitreco, kalman_jax as K, nativeseed, nativetime, vr_f1
from trkfastsim.reco.linefit_jax import HIT_KEYS, DRIFT_KEYS

FITTER_SHA = "988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a"
LIMITS = [
    "One explicit eligible single-seed target, not an acceptance/fidelity population or causal coverage of missing captured targets.",
    "Captured native seed q/C/range are diagnostic substitutions after the independent seed audit, not an autonomous seed finder.",
    "Mixed tuples change only selected input fields, not all downstream operands; q couples prior means and momentum constraint target; C couples prior uncertainty/material operands.",
    "All arms use the same explicit native constraint time, full stock schedule/maxniter10 and previous/current policies; no tolerance/regularization/default changes.",
    "Ordinary is a bounded execution control, not a regenerated full baseline; preserved full-population baseline outputs remain authoritative.",
    "Complete ordinary/control returned trees must match exactly, including nonfinite positions; numerical finiteness is reported separately, never repaired.",
    "Range-only may retain different owner-range bookkeeping if no successful update; initial range equality is not required between these distinct arms.",
    "No full fit/F1/numerical/status/acceptance closure follows from this diagnostic.",
]


def identity(text):
    fields = text.split(":")
    A.require(len(fields) == 4 and all(x.isdecimal() for x in fields), "identity requires run:subrun:event:iseed")
    key = np.array([int(x) for x in fields], dtype=np.int64)
    A.require(key[0] > 0 and key[2] > 0 and key[3] == 0, "positive run/event and single seed zero required")
    return key


def helpers(watched):
    root = Path(__file__).resolve().parents[1]
    D.helpers(watched)
    D.watch(root/"trkfastsim/reco/kalman_jax.py", watched, FITTER_SHA)
    for name, module, functions in (
        ("scripts/f1_seed_arms.py", S, ("validate", "baseline", "select", "fit")),
        ("scripts/f1_seed_arm_probe.py", __import__(__name__), ("identity", "helpers", "compare", "prepare", "execute", "run")),
    ):
        path = root/name
        A.require(Path(module.__file__).resolve() == path, "executing arm module path differs")
        D.watch(path, watched)
        for function in functions:
            D.code_origin(getattr(module, function, None), path)
    A.require(S.K is K and nativeseed.K is K, "fitter binding differs")
    for module in (K, nativeseed, nativetime, vr_f1, hitreco, driftann):
        path = Path(module.__file__).resolve()
        A.require(path.parent == root/"trkfastsim/reco", "reco module checkout differs")
        required = {K: ("fit_track", "seed_covariance", "straw_poca", "speed", "stock_stages", "load_anns"),
                    nativeseed: ("load_seed", "transform_seed", "to_fitter6"), nativetime: ("load_times", "join_keys", "digest"),
                    vr_f1: ("read_inputs",), hitreco: ("load_conditions",), driftann: ("find", "load", "mlp")}
        for name in required[module]:
            D.code_origin(getattr(module, name, None), path)
        for function in vars(module).values():
            if inspect.isfunction(function) and function.__module__ == module.__name__:
                D.code_origin(function, path)
    defaults = dict(INV="lu", FILTER="cov", PARALLEL=False, SCATSCALE=1., FIXED_NULLDV_ALL=False,
                    NOHIT_IN_SEED=False, REF_AT_TIME=True, KL_TEST=True, UNROLL=8, ELOSS=1.,
                    MOM=2000., MOM_CONSTRAINT=1.e-4, MMU=105.6583755)
    A.require(all(getattr(K, name) == value for name, value in defaults.items()), "fitter constants differ")
    material = K.kkmaterial
    A.require(jax.config.jax_enable_x64 and jax.default_backend() == "cpu", "probe requires x64 CPU execution")
    A.require(material.GAS_SCATTERFRAC == material.STATIC_SCATTERFRAC == .999999
              and material.SCATTERFRAC == {"solid": .999999, "gas": .9999999}
              and material.STRAW == {"orad": 2.5235, "irad": 2.5235-.015, "wall": .015, "wrad": .0125},
              "material settings differ")
    for table, compute in ((material._MC, material.material_constants), (material._IC, material.ionisation_constants)):
        for name, values in table.items():
            expected = compute(name)
            A.require(values.keys() == expected.keys() and all(np.array_equal(values[k], expected[k]) for k in values),
                      "material cached constants differ")
    for path in sorted((root/"trkfastsim/reco").glob("*.py")):
        D.watch(path, watched)
    return dict(fitter_sha256=FITTER_SHA, backend=jax.default_backend(), jax_version=jax.__version__, numpy_version=np.__version__,
                effective_fitter=defaults, x64=True, gas_scatter_fraction=material.GAS_SCATTERFRAC,
                straw_geometry=dict(material.STRAW))


def compare(actual, expected):
    """Compare whole returned trees, preserving nonfinite outcomes in the diagnostic."""
    left, tree = jax.tree_util.tree_flatten_with_path(actual)
    right, other = jax.tree_util.tree_flatten_with_path(expected)
    A.require(tree == other, "arm output structures differ")
    unequal, nonfinite = [], []
    for (path, value), (other_path, reference) in zip(left, right):
        A.require(path == other_path, "arm output paths differ")
        a, b = np.asarray(value), np.asarray(reference)
        label = jax.tree_util.keystr(path)
        if a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
            unequal.append(label)
        if a.dtype.kind in "fc" and not np.all(np.isfinite(a)):
            nonfinite.append(label)
    return dict(exact=not unequal, unequal_fields=unequal, actual_nonfinite_fields=nonfinite, n_leaves=len(left))


def prepare(args, watched, report):
    A.require(not {k: v for k, v in os.environ.items() if k.startswith("F1_")}, "probe requires default F1 environment")
    A.require(re.fullmatch("[0-9a-f]{64}", args.seed_contract_sha) is not None, "invalid explicit seed receipt hash")
    seed = D.receipt(args.seed_contract, args.seed_contract_sha, watched, ("completed", "passed", "hashes_unchanged"))
    A.require(seed.get("diagnostics", {}).get("n_seeds") == 2319 and seed.get("targets", {}).get("n_targets") == 147,
              "seed receipt population differs")
    A.require(seed.get("full_f1_closure") is False, "seed receipt scope differs")
    target_keys = B.integer_matrix(seed["targets"]["generated"]["keys"], 4, "seed targets")
    A.match_rows(args.key[None], target_keys)
    for field in ("check", "conditions", "straw_axis"):
        D.watch(getattr(args, field), watched)
    gate = D.strict_json(args.check)
    baseline_path = Path(gate["inputs"]["baseline"]["path"]).resolve()
    A.require(str(baseline_path) in seed["before_sha256"], "baseline absent from seed receipt")
    D.watch(baseline_path, watched, seed["before_sha256"][str(baseline_path)])
    cond = hitreco.load_conditions(args.conditions)
    x, h, c, ref, drift, n = vr_f1.read_inputs(str(baseline_path), cond, "mc", straw_axis=str(args.straw_axis), fitok_only=False)
    keys = np.column_stack([ref[k] for k in B.KEYS[:3]]+[np.zeros(n, dtype=np.int64)])
    B.integer_matrix(keys, 4, "eligible F1 keys")
    row = A.match_rows(args.key[None], keys)[0]
    nh, nx = int(h["valid"][row].sum()), int(c["valid"][row].sum())
    A.require(nh > 0 and nx > 0, "empty target collections")
    A.require(np.array_equal(h["valid"][row], np.arange(h["valid"].shape[1]) < nh), "nonprefix hit padding")
    A.require(np.array_equal(c["valid"][row], np.arange(c["valid"].shape[1]) < nx), "nonprefix crossing padding")
    hits = {k: h[k][row, :nh] for k in HIT_KEYS+("valid", "inseed")}
    crossings = {k: c[k][row, :nx] for k in ("mid", "wdir", "valid", "hit")}
    time, meta = nativetime.load_times(args.check, baseline_path, args.key[None])
    q, covariance, ranges, seed_meta = nativeseed.load_seed(args.check, baseline_path, args.key[None])
    for item in meta["inputs"].values():
        path = str(Path(item["path"]).resolve())
        A.require(path in seed["before_sha256"] and seed["before_sha256"][path] == item["sha256"],
                  "native loader evidence not bound by seed receipt")
        D.watch(path, watched, item["sha256"])
    native = dict(parameters=q[0], covariance=covariance[0], range=ranges[0])
    S.validate(x[row], hits, crossings, native, time[0])
    fixed = dict(active=hits["active"], lr=hits["lr"], nulldv=np.ones_like(hits["active"])) if args.fixed else None
    for name in ("TrainBkgSeed.dat", "TrainBkgFinal.dat", "TrainSign_Stage0.dat", "TrainCluster_Stage0.dat", "TrainSign_Stage1.dat", "TrainCluster_Stage1.dat"):
        D.watch(driftann.find(name), watched)
    stages, ann = K.stock_stages(), K.load_anns()
    report.update(key=args.key.tolist(), fixed_states=args.fixed, native_time=meta, native_seed=seed_meta,
                  n_eligible=n, n_hits=nh, n_crossings=nx, stock_fitok=bool(int(ref["kl_status"][row]) & vr_f1.FITOK),
                  schedule={k: np.asarray(v).tolist() for k, v in stages.items()}, couplings=S.COUPLINGS,
                  inputs={k: str(getattr(args, k)) for k in ("check", "conditions", "straw_axis", "seed_contract")},
                  seed_contract_sha256=args.seed_contract_sha)
    return dict(x=jnp.asarray(x[row]), hits=jax.tree_util.tree_map(jnp.asarray, hits), crossings=jax.tree_util.tree_map(jnp.asarray, crossings),
                drift={k: jnp.asarray(drift[k], dtype=jnp.float64) for k in DRIFT_KEYS}, stages=stages, ann=ann,
                native=jax.tree_util.tree_map(jnp.asarray, native), constraint_time=jnp.asarray(time[0]),
                fixed=None if fixed is None else jax.tree_util.tree_map(jnp.asarray, fixed))


def execute(inputs, out, report):
    expected_arms = ("ordinary", "control", "parameters", "covariance", "range", "full")
    A.require(S.ARMS == expected_arms, "seed arm inventory differs")
    reference = None
    report["arms"] = {}
    for arm in expected_arms:
        call = jax.jit(lambda x, h, c, n, t: S.fit(arm, x, h, c, inputs["drift"], inputs["stages"], inputs["ann"], n, t, fixed=inputs["fixed"]))
        result = call(inputs["x"], inputs["hits"], inputs["crossings"], inputs["native"], inputs["constraint_time"])
        jax.block_until_ready(result)
        if reference is None:
            reference = result
        cf, hist, good = result
        leaves, _ = jax.tree_util.tree_flatten_with_path(result)
        payload = {f"leaf_{i:03d}": np.asarray(value) for i, (_, value) in enumerate(leaves)}
        artifact = out/(arm+".npz")
        with artifact.open("xb") as stream:
            np.savez(stream, **payload)
        values = report["arms"][arm] = dict(good=bool(good), n_iterations=int(cf["niter"]), status=int(cf["status"]),
            comparison_to_ordinary=compare(result, reference), artifact=str(artifact), sha256=A.digest(artifact),
            leaf_paths={f"leaf_{i:03d}": jax.tree_util.keystr(path) for i, (path, _) in enumerate(leaves)})
        if arm == "control":
            A.require(values["comparison_to_ordinary"]["exact"], "baseline injection control differs; no component arms executed")
    A.require(tuple(report["arms"]) == expected_arms and report["arms"]["control"]["comparison_to_ordinary"]["exact"],
              "missing ordinary/control or incomplete arm inventory")
    for values in report["arms"].values():
        A.require(A.digest(values["artifact"]) == values["sha256"], "arm artifact changed during execution")
    report.update(completed=True, controls_passed=True)


def run(args):
    args = SimpleNamespace(**vars(args))
    for name in ("out", "check", "conditions", "straw_axis", "seed_contract"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    root = Path(__file__).resolve().parents[1]
    A.require(not args.out.is_relative_to(root), "protected checkout output")
    args.out.mkdir(parents=True, exist_ok=False)
    watched, report = {}, dict(completed=False, controls_passed=False, full_fit_closure=False, full_f1_closure=False,
                              full_numerical_validation=False, limits=LIMITS)
    try:
        report["execution"] = helpers(watched)
        inputs = prepare(args, watched, report)
        execute(inputs, args.out, report)
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", controls_passed=False)
    finally:
        end, errors = {}, []
        for path in watched:
            try:
                end[path] = A.digest(path)
            except Exception as error:
                errors.append(f"{path}: {type(error).__name__}: {error}")
        report.update(before_sha256=watched, end_sha256=end, hashes_unchanged=bool(watched) and watched == end and not errors, hash_errors=errors)
        if not report["hashes_unchanged"]:
            report["controls_passed"] = False
        report, text = D.serialize(report)
        if "controls_passed" not in report:
            report.update(controls_passed=False, full_numerical_validation=False)
            text = json.dumps(report, indent=2, allow_nan=False)+"\n"
        with (args.out/"manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "check", "conditions", "straw-axis", "seed-contract"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--seed-contract-sha", required=True)
    parser.add_argument("--key", type=identity, required=True)
    parser.add_argument("--fixed", action="store_true")
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "controls_passed", "hashes_unchanged", "error")}, indent=2))
    raise SystemExit(0 if report["completed"] and report["controls_passed"] else 1)


if __name__ == "__main__":
    main()
