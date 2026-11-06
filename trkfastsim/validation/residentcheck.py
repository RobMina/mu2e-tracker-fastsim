"""Reproduce the opt-in resident measured-hit path with explicit timing boundaries."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from trkfastsim.reco.residentio import prepare


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def validate_output(keys, hits, out):
    """Check complete input identity and census, without claiming physical validity."""
    if not np.array_equal(out["event_keys"], keys):
        raise RuntimeError("event census changed")
    for k, a in hits.items():
        b = out["input_" + k]
        equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
        if a.dtype != b.dtype or not equal:
            raise RuntimeError(f"input payload changed: {k}")
    if not np.array_equal(out["event_n_hits"], np.bincount(hits["ev"], minlength=len(keys))):
        raise RuntimeError("input hit census changed")
    ev, sid, flag = (hits[k] for k in ("ev", "sid", "flag"))
    selected = hits["valid"] & ((flag & ((1 << 6) | (1 << 23) | (1 << 24))) == 0) & ((flag & (1 << 10)) != 0)
    member = np.flatnonzero(selected & np.isfinite(hits["time"]))
    member = member[np.lexsort((member, sid[member], hits["time"][member], ev[member]))]
    begin = np.r_[True, (np.diff(ev[member]) != 0) | (np.diff(hits["time"][member]) > 20.)][:len(member)]
    expected_hg = np.full(len(ev), -1, np.int64)
    expected_hg[member] = np.cumsum(begin) - 1
    if not np.array_equal(out["hit_group"], expected_hg):
        raise RuntimeError("independent time-group membership differs")
    group_ev = ev[member[begin]]
    ng = int(out["n_groups"])
    if ng != len(group_ev) or not np.array_equal(out["group_ev"][:ng], group_ev):
        raise RuntimeError("group event identity differs")
    for name, expected_count in (("event_n_groups", np.bincount(group_ev, minlength=len(keys))),
                                 ("event_n_converged", np.bincount(group_ev[out["group_success"][:ng]], minlength=len(keys)))):
        if not np.array_equal(out[name], expected_count):
            raise RuntimeError(f"event census differs: {name}")
    expected = np.arange(len(hits["ev"])) < ng
    if not np.array_equal(out["group_valid"], expected):
        raise RuntimeError("group census is not a complete prefix")
    hg = out["hit_group"]
    if np.any(hg < -1) or np.any(hg >= ng):
        raise RuntimeError("invalid original-hit membership")
    counts = np.bincount(hg[hg >= 0], minlength=len(hg))
    if not np.array_equal(counts, out["group_n_hits"]):
        raise RuntimeError("group counts do not account for all memberships")
    if not np.array_equal(out["group_success"], out["group_status"] == 1):
        raise RuntimeError("success/status contradiction")
    ok = out["group_objective_valid"]
    if np.any(~np.isfinite(out["group_initial_objective"][ok])) or np.any(~np.isfinite(out["group_final_objective"][ok])):
        raise RuntimeError("nonfinite valid objective")
    if np.any(out["group_final_objective"][ok] > out["group_initial_objective"][ok]):
        raise RuntimeError("wire objective increased")
    if np.any(out["group_success"] & ~ok):
        raise RuntimeError("success without a valid objective")


def compare_outputs(reference, candidate):
    """Require structural conservation; report numerical differences without relabeling."""
    structural = ("event_keys", "input_valid", "input_ev", "input_sid", "input_flag", "n_groups",
                  "event_n_hits", "event_n_groups", "event_invalid", "hit_group", "hit_status",
                  "group_valid", "group_ev", "group_n_hits", "group_time_span", "batch_valid")
    for key in tuple(k for k in candidate if k.startswith("input_")):
        a, b = reference[key], candidate[key]
        equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
        if a.dtype != b.dtype or not equal:
            raise RuntimeError(f"comparison input differs: {key}")
    for key in structural:
        if reference[key].dtype != candidate[key].dtype or not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"structural comparison differs: {key}")
    live = candidate["group_valid"]
    old, new = reference["group_status"][live], candidate["group_status"][live]
    transitions, counts = np.unique(np.column_stack((old, new)), axis=0, return_counts=True)
    changed = np.flatnonzero(live & (reference["group_status"] != candidate["group_status"]))
    both = reference["group_objective_valid"] & candidate["group_objective_valid"]
    delta = candidate["group_final_objective"][both] - reference["group_final_objective"][both]
    return {"structure_exact": True, "status_changed_group_rows": changed.tolist(),
            "status_transitions": [{"reference": int(a), "candidate": int(b), "groups": int(n)}
                                   for (a, b), n in zip(transitions, counts)],
            "both_objective_valid": int(both.sum()),
            "objective_abs_difference_max": float(np.max(np.abs(delta))) if len(delta) else None,
            "scope": "numerical comparison only; no cross-device bitwise or physics equivalence claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bz", type=float, required=True)
    parser.add_argument("--max-hits", type=int, default=128)
    parser.add_argument("--group-batch", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--compare", type=Path, help="saved resident.npz; exact structure, numerical differences reported")
    args = parser.parse_args(argv)
    if args.outdir.exists():
        parser.error("outdir already exists; use a new directory")
    if args.repeats < 2 or not np.isfinite(args.bz) or args.bz == 0:
        parser.error("require >=2 repeats and finite nonzero bz")
    import jax
    from trkfastsim.reco import resident
    from trkfastsim.reco import helixfit_jax, helixseed_jax
    device = jax.devices()[0]
    if args.require_gpu and device.platform != "gpu":
        raise RuntimeError("GPU required; CPU fallback is forbidden")
    timings = {}
    start = time.perf_counter()
    with np.load(args.hits, allow_pickle=False) as saved:
        keys = saved["evkey"].copy()
        sh = {k[2:]: saved[k].copy() for k in saved.files if k.startswith("s_")}
    conditions = json.loads(args.conditions.read_text())
    keys, hits = prepare(keys, sh, conditions)
    timings["read_prepare_s"] = time.perf_counter() - start
    start = time.perf_counter()
    dk, dh, bz = jax.device_put((keys, hits, np.float64(args.bz)), device)
    jax.block_until_ready((dk, dh, bz))
    timings["upload_s"] = time.perf_counter() - start
    options = dict(max_hits=args.max_hits, group_batch=args.group_batch, max_iterations=args.max_iterations)
    start = time.perf_counter()
    executable = resident.reconstruct.lower(dk, dh, bz=bz, **options).compile()
    timings["lower_compile_s"] = time.perf_counter() - start
    runs, previous, differences = [], None, []
    for _ in range(args.repeats):
        start = time.perf_counter()
        with jax.transfer_guard("disallow_explicit"):
            result = executable(dk, dh, bz=bz)
            jax.block_until_ready(result)
        runs.append(time.perf_counter() - start)
        if any(a.devices() != {device} for a in jax.tree.leaves(result)):
            raise RuntimeError("output is not resident on the selected device")
        start = time.perf_counter()
        out = jax.device_get(result)
        timings.setdefault("download_s", []).append(time.perf_counter() - start)
        validate_output(keys, hits, out)
        if previous is not None:
            differences = [k for k in out if not np.array_equal(out[k], previous[k], equal_nan=True)]
            if differences:
                raise RuntimeError(f"repeat changed arrays: {differences}")
        previous = out
    timings["resident_runs_s"] = runs
    group_live = out["group_valid"]
    codes, counts = np.unique(out["group_status"][group_live], return_counts=True)
    names = {v: k for k, v in resident.GROUP_STATUS.items()}
    modules = (resident, helixseed_jax, helixfit_jax)
    source_files = [Path(m.__file__) for m in modules] + [Path(__file__), Path(__file__).parents[1] / "reco/residentio.py"]
    report = {"schema_version": 1, "scope": "measured_hits_to_wire_helix; not drift/CH/physics validation",
              "inputs_sha256": {"hits": digest(args.hits), "conditions": digest(args.conditions)},
              "source_sha256": {p.name: digest(p) for p in source_files},
              "jax": jax.__version__, "jaxlib": jax.lib.__version__, "platform": device.platform,
              "device": str(device), "device_kind": device.device_kind, "x64": jax.config.x64_enabled,
              "config": options | {"bz": args.bz}, "events": len(keys), "hits": len(hits["ev"]),
              "empty_events": int(np.sum(out["event_n_hits"] == 0)), "groups": int(out["n_groups"]),
              "group_status": {names[int(c)]: int(n) for c, n in zip(codes, counts)},
              "numerically_converged_groups": int(out["group_success"].sum()),
              "batch_valid": bool(out["batch_valid"]), "repeat_arrays_exact": True,
              "timings": timings, "timing_scope": "compile/upload/download/I/O separate; no stock speedup",
              "unsupported": ["drift-time fit", "multitrack/outlier separation", "charge/flight direction",
                              "covariance", "field map", "material", "calibrated chi-square", "physics efficiency"]}
    if args.compare:
        with np.load(args.compare, allow_pickle=False) as saved:
            reference = {k: saved[k] for k in saved.files}
        report["comparison"] = compare_outputs(reference, out)
        report["comparison_sha256"] = digest(args.compare)
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir / "resident.npz", **out)
    report["output_sha256"] = digest(args.outdir / "resident.npz")
    (args.outdir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
