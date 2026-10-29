"""Capture one explicit F1 target at full iteration cap; preserve native evidence and verify trace passivity."""
import argparse
import json
import os
from pathlib import Path

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import uproot

from f1_native_decision_audit import load_evidence, read_tree, keys, dense, digest, require
from trkfastsim.reco import driftann, fittrace, hitreco, kalman_jax as K, nativeref, nativeseed, nativetime, vr_f1
from trkfastsim.reco.linefit_jax import HIT_KEYS, DRIFT_KEYS


def identity(text):
    parts = text.split(":")
    if len(parts) != 4 or any(not p.isdecimal() for p in parts):
        raise argparse.ArgumentTypeError("identity needs run:subrun:event:iseed nonnegative integers")
    value = np.asarray([int(p) for p in parts], dtype=np.int64)
    if value[0] == 0 or value[2] == 0 or value[3] != 0:
        raise argparse.ArgumentTypeError("trace requires positive run/event and single-seed iseed=0")
    return value


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "key": args.key.tolist(), "full_numerical_validation": False,
                "all_captured_iterations_included": False, "scope": "one explicit target, actual F1 operands",
                "native_seed": args.native_seed, "native_time": args.native_time or args.native_seed,
                "time_model": "stock", "reference_model": "previous",
                "straw_geometry": "approximate", "crossing_time": "current", "maxniter": 10,
                "source": {}, "environment": {k: v for k, v in os.environ.items() if k.startswith("F1_")}}
    try:
        require(not metadata["environment"], "trace driver requires default F1 environment; use explicit options")
        iterations, pieces = load_evidence(args.check, metadata)
        metadata["all_captured_iterations_included"] = False
        k = args.key
        imask = np.all(keys(iterations, unique=False) == k, axis=1)
        require(imask.any(), "requested identity has no native iterations")
        ni = iterations[imask]
        require(np.min(ak.to_numpy(ni.iteration_ordinal)) == 0, "missing initial native iteration")
        source = metadata["inputs"]["baseline"]["path"]
        cond = hitreco.load_conditions(args.conditions)
        x, h, c, ref, drift, n = vr_f1.read_inputs(source, cond, "mc", straw_axis=str(args.straw_axis), fitok_only=False)
        allkeys = np.column_stack([ref[f] for f in ("run", "subrun", "event")] + [np.zeros(n, np.int64)])
        row = nativeref.match_rows(k[None], allkeys)[0]
        require(row >= 0, "requested identity absent from eligible F1 population")
        nh, nx = int(h["valid"][row].sum()), int(c["valid"][row].sum())
        require(nh > 0 and nx > 0, "empty target hits/crossings")
        hj = {name: jnp.asarray(h[name][row, :nh]) for name in HIT_KEYS + ("valid", "inseed")}
        cj = {name: jnp.asarray(c[name][row, :nx]) for name in ("mid", "wdir", "valid", "hit")}
        dj = {name: jnp.asarray(drift[name], dtype=jnp.float64) for name in DRIFT_KEYS}
        seed, native_time = None, None
        if args.native_time or args.native_seed:
            native_time = nativetime.load_times(args.check, source, k[None])[0][0]
        if args.native_seed:
            sx, sc, sr, sm = nativeseed.load_seed(args.check, source, k[None])
            seed = {"parameters": jnp.asarray(sx[0]), "covariance": jnp.asarray(sc[0]), "range": jnp.asarray(sr[0])}
            metadata["seed_loader"] = sm
        stages, ann = K.stock_stages(), K.load_anns()
        metadata["trace_bytes_bound"] = fittrace.allocation_bytes(nh, nx, len(stages["T"]), 10)
        require(metadata["trace_bytes_bound"] <= fittrace.MAX_BYTES, "trace allocation exceeds limit")
        metadata["schedule"] = {name: np.asarray(value).tolist() for name, value in stages.items()}
        root = Path(__file__).resolve().parents[1]
        source_files = [Path(__file__), root / "scripts/f1_native_decision_audit.py", root / "scripts/f1_trace_compare.py",
                        root / "scripts/run_f1_trace.sh"]
        source_files += sorted((root / "trkfastsim/reco").glob("*.py"))
        metadata["source"] = {str(p.relative_to(root)): digest(p) for p in source_files}
        metadata["conditions"] = {"path": str(args.conditions), "sha256": digest(args.conditions)}
        metadata["alignment"] = {"path": str(args.straw_axis), "sha256": digest(args.straw_axis)}
        names = ["TrainBkgSeed.dat", "TrainBkgFinal.dat", "TrainSign_Stage0.dat", "TrainCluster_Stage0.dat",
                 "TrainSign_Stage1.dat", "TrainCluster_Stage1.dat"]
        metadata["ann"] = {name: {"path": str(driftann.find(name)), "sha256": digest(driftann.find(name))} for name in names}
        def fit(tracing):
            return jax.jit(lambda x0: K.fit_track(x0, hj, cj, dj, stages, ann, time_model="stock", maxniter=10,
                constraint_time=native_time, seed_input=seed, trace=tracing))(jnp.asarray(x[row]))
        ordinary = fit(False)
        jax.block_until_ready(ordinary)
        captured = fit(True)
        jax.block_until_ready(captured)
        cf, hist, good = captured
        records = cf.pop("trace")
        a, atree = jax.tree_util.tree_flatten(ordinary)
        b, btree = jax.tree_util.tree_flatten((cf, hist, good))
        require(atree == btree, "trace changes output structure")
        require(all(np.array_equal(np.asarray(u), np.asarray(v), equal_nan=True) for u, v in zip(a, b)),
                "trace changes ordinary fit outputs")
        records = {name: np.asarray(value) for name, value in records.items()}
        valid = records["valid"]
        require(int(valid.sum()) == int(cf["niter"]), "trace count differs from executed iteration count")
        stage, iteration = np.nonzero(valid)
        kept = {name: value[valid] for name, value in records.items()}
        with uproot.open(source) as f:
            candidates = f["trkdump/events"].arrays(["kst_sid"])
        xsid = ak.to_numpy(candidates.kst_sid[int(ref["event_index"][row])]).astype(np.int64)
        hids = h["shloc"][row, :nh]
        partner = np.where(c["hit"][row, :nx] >= 0, hids[np.maximum(c["hit"][row, :nx], 0)], -1)
        np.savez(args.out / "trace.npz", **kept, stage_index=stage, key=k, hit_ids=hids, crossing_ids=xsid,
                 seed_hit_mask=h["inseed"][row, :nh], crossing_partner=partner,
                 hit_mid=h["mid"][row, :nh], crossing_mid=c["mid"][row, :nx])
        metadata.update(trace_passivity_exact=True, n_iterations=int(valid.sum()), f1_good=bool(good),
                        stock_fitok=bool(int(ref["kl_status"][row]) & vr_f1.FITOK),
                        native_n_iterations=len(ni), trace_sha256=digest(args.out / "trace.npz"))
        first = {name: value[0] for name, value in kept.items()}
        ni = ni[ni.iteration_ordinal == 0]
        pmask = np.all(keys(pieces, unique=False) == k, axis=1) & (ak.to_numpy(pieces.iteration_ordinal) == 0)
        old = pieces[pmask & (ak.to_numpy(pieces.owner) == 0)]
        with uproot.open(metadata["inputs"]["capture"]["path"]) as f:
            effects = read_tree(f, "kktrace", "effects", metadata["highest_cycles"]["capture"])
        emask = np.all(keys(effects, unique=False) == k, axis=1) & (ak.to_numpy(effects.iteration_ordinal) == 0)
        from f1_trace_compare import compare_first
        metadata["first_iteration_comparison"] = compare_first(first, ni, old, effects[emask], hids, xsid,
                                                               h["inseed"][row, :nh], partner)
        metadata["completed"] = True
    except Exception as error:
        metadata.update(completed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (args.out / "manifest.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k: v for k, v in metadata.items() if k in ("key", "completed", "n_iterations", "f1_good",
        "stock_fitok", "trace_passivity_exact", "first_iteration_comparison")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--straw-axis", type=Path, required=True)
    parser.add_argument("--key", type=identity, required=True)
    parser.add_argument("--native-time", action="store_true")
    parser.add_argument("--native-seed", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
