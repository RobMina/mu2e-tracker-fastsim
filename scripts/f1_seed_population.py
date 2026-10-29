"""Receipt-gated seed component populations with bounded preserved-baseline controls."""
import argparse
import inspect
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import f1_seed_arm_probe as P
import f1_seed_population_kernel as F
import f1_seed_population_compare as Q
from f1_seed_arm_probe import A, B, D, K, S, driftann, hitreco, nativeseed, nativetime, vr_f1
from trkfastsim.reco.linefit_jax import HIT_KEYS, DRIFT_KEYS

SEED_SHA = "c1c7c19c7fc5cabdd85cd195ffa6973998a478fd9444cabd66882f3e677fa633"
COV_SHA = "e3b649e3b135bd68131b25c4faaaa2b14641a2c07c664ce1defa04084af4d740"
BASE_PINS = {
    "own": {"ids.npz": "b6167bb3e9b5033958669da4660288f6a10a3d488993c4bc80c15248b6334413",
            "vr2f1.json": "ee1caccaded1fa27de75f63c7a44fdd3f7ace5bde5d842e72c4a96c6c90869e6"},
    "fixed": {"ids.npz": "60f963bfb71a58bd9409e0c1e4a130d40add8985b984d11f4d12088cce81f3e6",
              "vr2f1.json": "2df1697bf3eb77f01e9ff96ead7e31ddc71dac4af330393b4f233be7f982a48b"},
}
SOURCE_LIST_SHA = "c0f905897299b82df9c6e838c3436754fb762498f60b4bf82b90b23afa2e44f1"
ARMS = ("parameters", "momentum", "geometry_time", "covariance", "range")
MATCH_FIELDS = ("xt0", "Ct0", "good", "stat", "niter", "chi2", "ndof", "fact", "flr", "xact_fit")
LIMITS = [
    "Own mode covers all2315 eligible single-seed/single-KL events; fixed mode covers all2151 native FitOK events including F1 failures, matching preserved baseline populations.",
    "The164 native-rejected events are outside fixed-state fidelity by definition, not silently lost; own mode retains them. Native audited seed census2319 is a different population.",
    "Preserved native-time-only baseline is reused, not rerun as a full ordinary/control population. Bounded stratified controls do not certify unrerun population bitwise equivalence.",
    "Captured native q/C/range substitutions are conditioned diagnostics; momentum-only changes fitter p and its prior/constraint-target consequences, geometry_time changes first5 fitter coordinates.",
    "Mixed tuples keep baseline covariance/range for q splits, even when those tuples are not individually native-exact seeds. Geometry/time coordinates include the native-to-fitter transform.",
    "All arms use same explicit native constraint time, previous/current reference/crossing policies and unchanged full stock schedule with production3/10 two-pass cap policy.",
    "Compatibility: exact discrete fields and ordinary/control complete trees; floating preserved outputs use unchanged1e-5/1e-7 plus identical0.1-stock-sigma predicates on controls. Exactness is reported separately.",
    "Nonfinite results remain in NPZ and denominators. No regularization, physics threshold/default changes, full numerical/F1 closure or uncaptured-history causal coverage.",
]


def helpers(watched):
    execution = P.helpers(watched)
    root = Path(__file__).resolve().parents[1]
    for name, sha in (("f1_seed_arms.py", "945f22d6ff6059b67d85b8e81bc567cc27dd43fee933f3d7475c73d831188692"),
                      ("f1_seed_arm_probe.py", "a7f9b2f97e01b6cf21ecc443238f7d2670041f379246462467b6ab0a4e9752fa")):
        D.watch(root/"scripts"/name, watched, sha)
    A.require(ARMS == ("parameters", "momentum", "geometry_time", "covariance", "range"), "population inventory differs")
    for module in (F, Q, __import__(__name__)):
        path = root/"scripts"/(module.__name__.split(".")[-1]+".py") if module.__name__ != "__main__" else Path(__file__).resolve()
        A.require(Path(module.__file__).resolve() == path, "population module origin differs")
        D.watch(path, watched)
        required = ("select", "fit", "project") if module is F else ("validate", "summary") if module is Q else (
            "helpers", "validate_batch", "layout", "baseline", "prepare", "control_indices", "arguments",
            "compiler", "expand", "payload", "compatibility", "save", "controls", "population", "run", "main")
        for name in required:
            D.code_origin(getattr(module, name, None), path)
        for function in vars(module).values():
            if inspect.isfunction(function) and function.__module__ == module.__name__:
                D.code_origin(function, path)
    A.require(F.ARMS == ("ordinary", "control", "parameters", "momentum", "geometry_time", "covariance", "range", "full"),
              "population arm inventory differs")
    return execution


def validate_batch(x, h, c, native, times):
    n = len(x)
    A.require(n > 0 and x.shape in ((n, 5), (n, 6)), "invalid population seed shape")
    for name, value in {"x": x, "times": times, **native}.items():
        A.require(not np.ma.isMaskedArray(value) and np.asarray(value).dtype.kind == "f"
                  and np.all(np.isfinite(value)), "invalid population operand: "+name)
    A.require(times.shape == (n,) and native["parameters"].shape == (n, 6)
              and native["covariance"].shape == (n, 6, 6) and native["range"].shape == (n, 2), "native operand shape")
    A.require(native["parameters"][:, 5] > 0, "nonpositive native momentum")
    A.require(native["range"][:, 0] <= native["range"][:, 1], "reversed native range")
    cov = native["covariance"]
    A.require(np.isclose(cov, cov.swapaxes(1, 2), rtol=B.RTOL, atol=B.ATOL), "asymmetric covariance")
    A.require(np.linalg.eigvalsh(cov) > 0, "nonpositive covariance; no regularization")
    for label, collection in (("hit", h), ("crossing", c)):
        valid = collection["valid"]
        A.require(not np.ma.isMaskedArray(valid) and valid.ndim == 2 and valid.shape[0] == n
                  and valid.dtype.kind == "b" and valid.shape[1] > 0, "invalid "+label+" mask")
        A.require(valid == (np.arange(valid.shape[1])[None] < valid.sum(1)[:, None]), "nonprefix "+label+" padding")
        A.require(valid.sum(1) > 0, "empty "+label+" collection")
        for field in ("mid", "wdir"):
            values = collection[field]
            A.require(not np.ma.isMaskedArray(values) and values.shape == valid.shape+(3,)
                      and values.dtype.kind == "f" and np.isfinite(values).all(), "invalid "+label+" geometry")
        A.require(np.any(collection["wdir"] != 0, axis=2), "zero wire direction")
    A.require(h["inseed"].shape == h["valid"].shape and h["inseed"].dtype.kind == "b"
              and not np.ma.isMaskedArray(h["inseed"]), "invalid seed membership")
    A.require((h["inseed"] & h["valid"]).any(1), "empty seed membership")
    partner = c["hit"]
    A.require(not np.ma.isMaskedArray(partner) and partner.shape == c["valid"].shape
              and partner.dtype.kind in "iu", "invalid partner array")
    A.require((partner >= -1) & (partner < h["valid"].shape[1]), "partner out of range")
    A.require(~(c["valid"] & (partner >= 0)) | np.take_along_axis(h["valid"], np.maximum(partner, 0), axis=1),
              "partner points to padding")


def layout(h, c):
    nh, nx = h["valid"].sum(1), c["valid"].sum(1)
    nf, mf = h["valid"].shape[1], c["valid"].shape[1]
    classes = [(min(nf, 32), min(mf, 64)), (min(nf, 64), min(mf, 128)), (nf, mf)]
    group = np.where((nh <= 32) & (nx <= 64), 0, np.where((nh <= 64) & (nx <= 128), 1, 2))
    return classes, group, np.lexsort((nx, nh, group))


def baseline(directory, mode, ref, keys, watched, input_path, gate_path):
    directory = Path(directory).resolve()
    for name, sha in {**BASE_PINS[mode], "source.sha256": SOURCE_LIST_SHA}.items():
        D.watch(directory/name, watched, sha)
    metadata = D.strict_json(directory/"vr2f1.json")
    A.require(metadata["input"] == str(input_path) and metadata["own_states"] is (mode == "own")
              and metadata["all_fits"] is False and metadata["time_model"] == "stock"
              and metadata["time_legacy"] == [] and metadata["xing_activity"] == "hit_or_doca"
              and metadata["seed_time_provenance"] == "native_ParameterHit_time_only"
              and metadata["native_time"]["check"] == str(gate_path)
              and metadata["timing"]["fast_niter"] == 3, "preserved baseline configuration differs")
    A.require(set(metadata["env_f1"]) <= {"F1_JAX_CACHE"}, "preserved baseline physics overrides")
    with np.load(directory/"ids.npz", allow_pickle=False) as saved:
        result = {k: saved[k] for k in saved.files}
    event_index = B.integer_matrix(result["event_index"][:, None], 1, "baseline event indices")
    rows = A.match_rows(event_index, B.integer_matrix(ref["event_index"][:, None], 1, "input event indices"))
    A.require(len(rows) == len(keys), "baseline population differs")
    for name in ("subrun", "event", "kl_status"):
        A.require(result[name] == ref[name][rows], "baseline identity/reference differs: "+name)
    result["keys"] = keys[rows]
    order = A.match_rows(keys, result["keys"])
    result = {k: v[order] for k, v in result.items()}
    result["kl"] = np.asarray(jax.jit(jax.vmap(K.kl_params))(jnp.asarray(result["xt0"])))
    Q.validate(result)
    A.require(result["stockok"] == ((ref["kl_status"].astype(np.int64) & vr_f1.FITOK) != 0), "baseline stock flags differ")
    names = vr_f1.KL+("mom",)
    for field, suffix in (("stock_params", ""), ("stock_errors", "err")):
        expected = np.stack([ref["kl_"+k+suffix] for k in names], axis=1)
        A.require(np.array_equal(result[field], expected, equal_nan=True), "baseline stock operands differ")
    return result, metadata


def prepare(args, watched, report):
    A.require(not any(k.startswith("F1_") for k in os.environ), "population requires default F1 environment")
    seed = D.receipt(args.seed_contract, SEED_SHA, watched, ("completed", "passed", "hashes_unchanged"))
    cov = D.receipt(args.covariance_contract, COV_SHA, watched, ("completed", "passed", "exact", "hashes_unchanged"))
    A.require(seed["diagnostics"]["n_seeds"] == 2319 and cov["n_seeds"] == 2319, "seed census differs")
    gate = D.strict_json(args.check)
    path = Path(gate["inputs"]["baseline"]["path"]).resolve()
    D.watch(path, watched, seed["before_sha256"][str(path)])
    for name in ("conditions", "straw_axis", "check"):
        operand = str(Path(getattr(args, name)).resolve())
        A.require(operand in seed["before_sha256"], "input absent from seed receipt: "+name)
        D.watch(operand, watched, seed["before_sha256"][operand])
    cond = hitreco.load_conditions(args.conditions)
    x, h, c, ref, drift, n = vr_f1.read_inputs(str(path), cond, "mc", straw_axis=str(args.straw_axis), fitok_only=args.mode == "fixed")
    A.require(n == (2315 if args.mode == "own" else 2151), "eligible population differs")
    keys = B.integer_matrix(np.column_stack([ref[k] for k in B.KEYS[:3]]+[np.zeros(n, dtype=np.int64)]), 4, "eligible keys")
    times, meta = nativetime.load_times(args.check, path, keys)
    q, covariance, ranges, seed_meta = nativeseed.load_seed(args.check, path, keys)
    for item in meta["inputs"].values():
        p = str(Path(item["path"]).resolve())
        A.require(seed["before_sha256"].get(p) == item["sha256"], "loader input not bound by seed receipt")
        D.watch(p, watched, item["sha256"])
    native = dict(parameters=q, covariance=covariance, range=ranges)
    validate_batch(x, h, c, native, times)
    base, metadata = baseline(args.baseline, args.mode, ref, keys, watched, path, args.check)
    A.require(metadata["straw_axis"] == str(args.straw_axis), "baseline alignment path differs")
    for field, values in (("valid", h["valid"]), ("stock_active", h["active"]), ("stock_lr", h["lr"]),
                          ("xvalid", c["valid"]), ("xhit", c["hit"])):
        A.require(np.array_equal(base[field], values), "baseline candidate operands differ: "+field)
    ann_manifest = Path(args.baseline)/"ann.sha256"
    D.watch(ann_manifest, watched, "cb77f9bc137d4d0ca2b4695d666c765d9bc4c754123c70e14d1dd0eae46d4d72")
    ann_pins = {Path(path).name: (Path(path).resolve(), sha) for sha, path in
                (line.split(maxsplit=1) for line in ann_manifest.read_text().splitlines())}
    for name in ("TrainBkgSeed.dat", "TrainBkgFinal.dat", "TrainSign_Stage0.dat", "TrainCluster_Stage0.dat", "TrainSign_Stage1.dat", "TrainCluster_Stage1.dat"):
        path = Path(driftann.find(name)).resolve()
        A.require(path == ann_pins[name][0], "ANN baseline path differs")
        D.watch(path, watched, ann_pins[name][1])
    stages, ann = K.stock_stages(), K.load_anns()
    report.update(mode=args.mode, n_eligible=n, fixed_excluded_native_rejected=164 if args.mode == "fixed" else 0,
                  native_time=meta, native_seed=seed_meta, baseline_metadata=metadata,
                  baseline_summary=Q.summary(base), keys=keys.tolist(),
                  schedule={k: np.asarray(v).tolist() for k, v in stages.items()})
    return dict(x=x, h={k: h[k] for k in HIT_KEYS+("valid", "inseed")},
                c={k: c[k] for k in ("mid", "wdir", "valid", "hit")}, native=native, times=times,
                drift={k: jnp.asarray(drift[k], dtype=jnp.float64) for k in DRIFT_KEYS}, stages=stages, ann=ann,
                keys=keys, ref=ref, baseline=base, mode=args.mode)


def control_indices(data):
    _, group, _ = layout(data["h"], data["c"])
    base = data["baseline"]
    retry = (np.diff(np.c_[np.zeros(len(group)), base["niter"]], axis=1) >= 3).any(1)
    selected, strata = [], []
    for cls in range(3):
        for cap in (False, True):
            for good in (False, True):
                rows = np.flatnonzero((group == cls) & (retry == cap) & (base["good"] == good))
                strata.append(dict(size_class=cls, retry=cap, good=good, population=len(rows)))
                if len(rows):
                    selected.append(int(rows[0]))
    for event in (3700, 7033):
        rows = np.flatnonzero(np.all(data["keys"] == [1400, 0, event, 0], axis=1))
        selected.extend(rows.tolist())
    return np.unique(selected), strata


def arguments(data, indices, nb, mb, lanes):
    A.require(len(indices) > 0 and lanes >= len(indices), "invalid batch lane count")
    sel = np.r_[indices, np.repeat(indices[-1], lanes-len(indices))]
    h = {k: jnp.asarray(v[sel, :nb]) for k, v in data["h"].items()}
    c = {k: jnp.asarray(v[sel, :mb]) for k, v in data["c"].items()}
    native = {k: jnp.asarray(v[sel]) for k, v in data["native"].items()}
    return jnp.asarray(data["x"][sel]), h, c, native, jnp.asarray(data["times"][sel])


def compiler(data, arm, maxniter, full_tree=False):
    def one(x, h, c, native, t):
        fixed = dict(active=h["active"], lr=h["lr"], nulldv=jnp.ones_like(h["active"])) if data["mode"] == "fixed" else None
        result = F.fit(arm, x, h, c, data["drift"], data["stages"], data["ann"], native, t, fixed=fixed, maxniter=maxniter)
        return result if full_tree else F.project(result)
    return jax.jit(jax.vmap(one))


def expand(output, nf, mf):
    result = {k: np.asarray(v).copy() for k, v in output.items()}
    for name, width in (("fact", nf), ("flr", nf), ("xact_fit", mf)):
        old = result[name]
        full = np.zeros((len(old), width), dtype=old.dtype)
        full[:, :old.shape[1]] = old
        result[name] = full
    return result


def payload(data, output, indices):
    base = data["baseline"]
    return dict(output, **{k: base[k][indices] for k in ("keys", "event_index", "stockok", "stock_params", "stock_errors", "kl_status",
                                                       "valid", "stock_active", "stock_lr", "xvalid")})


def compatibility(actual, expected):
    metrics = {}
    for name in MATCH_FIELDS:
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        A.require(a.shape == b.shape, "baseline comparison shape: "+name)
        mask = actual["valid"] if name in ("fact", "flr") else actual["xvalid"] if name == "xact_fit" else np.ones(a.shape, dtype=bool)
        discrete = name in ("good", "stat", "niter", "ndof", "fact", "flr", "xact_fit")
        same_nonfinite = (np.isnan(a) & np.isnan(b)) | ((a == b) & ~np.isfinite(a))
        equal = (a == b) | same_nonfinite
        close = equal if discrete else np.isclose(a, b, rtol=B.RTOL, atol=B.ATOL) | same_nonfinite
        finite = mask & np.isfinite(a) & np.isfinite(b)
        metrics[name] = dict(exact=bool(np.all(equal | ~mask)), passed=bool(np.all(close | ~mask)),
                             n_differing=int(np.sum(mask & ~equal)), n_failed=int(np.sum(mask & ~close)),
                             max_absolute_residual=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() and a.dtype.kind != "b" else None)
    def predicates(p):
        delta = p["kl"]-p["stock_params"]
        delta[:, 1] = (delta[:, 1]+np.pi)%(2*np.pi)-np.pi
        err = p["stock_errors"]
        with np.errstate(invalid="ignore", divide="ignore"):
            pull = delta/np.where(np.isfinite(err) & (err > 0), err, np.nan)
        return np.isfinite(pull) & (np.abs(pull) < .1)
    predicate_equal = bool(np.array_equal(predicates(actual), predicates(expected)))
    return dict(metrics=metrics, fidelity_predicates_exact=predicate_equal,
                passed=all(v["passed"] for v in metrics.values()) and predicate_equal,
                exact=all(v["exact"] for v in metrics.values()))


def save(out, name, values, report):
    path = out/(name+".npz")
    with path.open("xb") as stream:
        np.savez(stream, **values)
    report.setdefault("artifacts", {})[name] = dict(path=str(path), sha256=A.digest(path))


def controls(data, out, report):
    selected, strata = control_indices(data)
    classes, group, _ = layout(data["h"], data["c"])
    report["controls"] = dict(keys=data["keys"][selected].tolist(), strata=strata, batches=[])
    nf, mf = data["h"]["valid"].shape[1], data["c"]["valid"].shape[1]
    functions = {(arm, cap): compiler(data, arm, cap, full_tree=True) for arm in ("ordinary", "control") for cap in (3, 10)}
    passed = True
    for cls in range(3):
        indices = selected[group[selected] == cls]
        if not len(indices):
            continue
        nb, mb = classes[cls]
        args = arguments(data, indices, nb, mb, 256)
        batch = dict(size_class=cls, keys=data["keys"][indices].tolist(), lanes=256, hit_width=nb, crossing_width=mb)
        report["controls"]["batches"].append(batch)
        outputs = {}
        for cap in (3, 10):
            first = None
            for arm in ("ordinary", "control"):
                result = functions[arm, cap](*args)
                jax.block_until_ready(result)
                leaves, _ = jax.tree_util.tree_flatten_with_path(result)
                save(out, f"control_class{cls}_{arm}_cap{cap}", {f"leaf_{i:03d}": np.asarray(v)[:len(indices)] for i, (_, v) in enumerate(leaves)}, report)
                batch["leaf_paths"] = {f"leaf_{i:03d}": jax.tree_util.keystr(p) for i, (p, _) in enumerate(leaves)}
                if first is None:
                    first = result
                else:
                    comparison = P.compare(result, first)
                    batch[f"cap{cap}_injection"] = comparison
                    passed &= comparison["exact"]
                if arm == "ordinary":
                    outputs[cap] = {k: np.asarray(v)[:len(indices)] for k, v in jax.vmap(F.project)(result).items()}
        retry = (np.diff(np.c_[np.zeros(len(indices)), outputs[3]["niter"]], axis=1) >= 3).any(1)
        result = {k: np.where(retry.reshape((len(indices),)+(1,)*(v.ndim-1)), outputs[10][k], v) for k, v in outputs[3].items()}
        result = payload(data, expand(result, nf, mf), indices)
        expected = {k: v[indices] for k, v in data["baseline"].items()}
        batch["preserved_baseline"] = compatibility(result, expected)
        batch["retry"] = retry.tolist()
        save(out, f"control_class{cls}_projected", result, report)
        passed &= batch["preserved_baseline"]["passed"]
    report["controls"]["passed"] = bool(passed)
    A.require(passed, "bounded baseline compatibility failed; no population arms executed")


def population(data, arm, out, report):
    classes, group, order = layout(data["h"], data["c"])
    n = len(order)
    nf, mf = data["h"]["valid"].shape[1], data["c"]["valid"].shape[1]
    fast, full = compiler(data, arm, 3), compiler(data, arm, 10)
    outputs, indices, pending, logs = [], [], {}, []
    for cls in range(3):
        members = order[group[order] == cls]
        nb, mb = classes[cls]
        for start in range(0, len(members), 512):
            idx = members[start:start+512]
            lanes = int(256*np.ceil(len(idx)/256))
            t = time.perf_counter()
            result = fast(*arguments(data, idx, nb, mb, lanes))
            jax.block_until_ready(result)
            result = {k: np.asarray(v)[:len(idx)] for k, v in result.items()}
            retry = (np.diff(np.c_[np.zeros(len(idx)), result["niter"]], axis=1) >= 3).any(1)
            if retry.any():
                pending.setdefault(cls, []).append((len(outputs), np.flatnonzero(retry), idx[retry]))
            outputs.append(expand(result, nf, mf))
            indices.append(idx)
            logs.append(dict(pass_index=1, size_class=cls, n=len(idx), lanes=lanes, seconds=time.perf_counter()-t))
            print(f"{arm}: pass1 {sum(len(i) for i in indices)}/{n}, class{cls}", flush=True)
    for cls, chunks in sorted(pending.items()):
        idx = np.concatenate([c[2] for c in chunks])
        nb, mb = classes[cls]
        t = time.perf_counter()
        result = full(*arguments(data, idx, nb, mb, int(256*np.ceil(len(idx)/256))))
        jax.block_until_ready(result)
        result = expand({k: np.asarray(v)[:len(idx)] for k, v in result.items()}, nf, mf)
        offset = 0
        for which, rows, _ in chunks:
            for name, value in result.items():
                outputs[which][name][rows] = value[offset:offset+len(rows)]
            offset += len(rows)
        logs.append(dict(pass_index=2, size_class=cls, n=len(idx), seconds=time.perf_counter()-t))
    inverse = np.argsort(np.concatenate(indices))
    combined = {k: np.concatenate([o[k] for o in outputs])[inverse] for k in outputs[0]}
    combined = payload(data, combined, np.arange(n))
    save(out, arm, combined, report)
    report.setdefault("arms", {})[arm] = dict(summary=Q.summary(combined, data["baseline"]), timing=logs)


def run(args):
    out = Path(args.out).resolve()
    A.require(not out.is_relative_to(Path(__file__).resolve().parents[1]), "protected checkout output")
    out.mkdir(parents=True, exist_ok=False)
    watched, report = {}, dict(completed=False, controls_passed=False, hashes_unchanged=False, full_fit_closure=False,
                               full_f1_closure=False, full_numerical_validation=False, limits=LIMITS, arms={})
    try:
        report["execution"] = helpers(watched)
        data = prepare(args, watched, report)
        controls(data, out, report)
        jax.clear_caches()
        report["controls_passed"] = True
        if not args.controls_only:
            for arm in ARMS:
                population(data, arm, out, report)
                jax.clear_caches()
        A.require(tuple(report["arms"]) == (() if args.controls_only else ARMS), "incomplete population inventory")
        report.update(completed=True, controls_only=args.controls_only)
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", controls_passed=False)
    finally:
        end, errors = {}, []
        for path in watched:
            try:
                end[path] = A.digest(path)
            except Exception as error:
                errors.append(f"{path}: {type(error).__name__}: {error}")
        for artifact in report.get("artifacts", {}).values():
            try:
                A.require(A.digest(artifact["path"]) == artifact["sha256"], "artifact changed")
            except Exception as error:
                errors.append(f"{artifact['path']}: {type(error).__name__}: {error}")
        report.update(before_sha256=watched, end_sha256=end, hash_errors=errors,
                      hashes_unchanged=bool(watched) and watched == end and not errors)
        if not report["hashes_unchanged"]:
            report["controls_passed"] = False
        report, text = D.serialize(report)
        if "controls_passed" not in report:
            report.update(controls_passed=False, full_numerical_validation=False)
            text = json.dumps(report, indent=2, allow_nan=False)+"\n"
        with (out/"manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("out", "seed-contract", "covariance-contract", "check", "conditions", "straw-axis", "baseline"):
        parser.add_argument("--"+field, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--mode", choices=("own", "fixed"), required=True)
    parser.add_argument("--controls-only", action="store_true")
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "controls_passed", "hashes_unchanged", "error")}, indent=2))
    raise SystemExit(0 if report["completed"] and report["controls_passed"] else 1)


if __name__ == "__main__":
    main()
