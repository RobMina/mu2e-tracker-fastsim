"""Cold/warm real-table digitiser benchmark with persisted populations and repeat checks.

The input step tables and event populations must describe the same simulation.
Use a separate output directory per backend/device; no stock inputs are rerun.
"""
import argparse
import cProfile
import hashlib
import pstats
import json
from pathlib import Path
import resource
import time

import numpy as np

from trkfastsim import digicond
from trkfastsim.analyses import fastdigi


def fingerprint(d):
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in d.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", nargs="+", required=True)
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--cond", required=True)
    p.add_argument("--backend", choices=("numpy", "jax"), required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--profile-warm", action="store_true", help="cProfile the first warm pass, not compilation")
    p.add_argument("--profile-stages", action="store_true", help="synchronised per-kernel diagnostic on first warm pass")
    p.add_argument("--waveform-lanes", type=int, help="explicit JAX resource tuning probe")
    p.add_argument("--straw-batch", type=int, help="explicit JAX resource tuning probe")
    p.add_argument("--cluster-batch", type=int, help="explicit JAX resource tuning probe")
    p.add_argument("--saturation-tile", type=int, help="explicit JAX resource tuning probe")
    args = p.parse_args(argv)
    if len(args.steps) != len(args.events) or args.repeats < 1 or args.limit < 0:
        p.error("need matched step/event files, positive repeats and nonnegative limit")
    args.outdir.mkdir(parents=True, exist_ok=False)
    c = digicond.Conditions.load(args.cond)
    if args.backend == "jax":
        import jax
        from trkfastsim import digi_jax, digi_clusters_jax
        for value in (args.waveform_lanes, args.straw_batch, args.cluster_batch, args.saturation_tile):
            if value is not None and value < 1:
                p.error("resource tuning limits must be positive")
        if args.waveform_lanes is not None:
            digi_jax.WAVEFORM_CLUSTER_LANES = args.waveform_lanes
        if args.straw_batch is not None:
            digi_jax.BATCH = args.straw_batch
        if args.cluster_batch is not None:
            digi_clusters_jax.BATCH_SIZE = args.cluster_batch
        if args.saturation_tile is not None:
            digi_jax.SATURATION_TILE = args.saturation_tile
        device = str(jax.devices())
        rng_for = lambda i: jax.random.fold_in(jax.random.key(args.seed), i)
    else:
        device = "numpy CPU"
        rng_for = lambda i: np.random.default_rng([args.seed, i])
    rows, first = [], {}
    for repeat in range(args.repeats):
        for i, (steps, events) in enumerate(zip(args.steps, args.events)):
            keys = fastdigi.event_keys([events])
            if len(np.unique(keys, axis=0)) != len(keys):
                raise ValueError("duplicate event identities")
            if args.limit:
                keys = keys[:args.limit]
            stats = {}
            start = time.perf_counter()
            profiler = cProfile.Profile() if args.profile_warm and repeat == 1 else None
            if profiler is not None:
                profiler.enable()
            from digi_stage_profile import stage_profile
            with stage_profile(args.profile_stages and repeat == 1 and args.backend == "jax") as stages:
                written, d, nstep = fastdigi.digitize_table(steps, c, rng_for(i), keys, stats=stats, backend=args.backend)
            if stages:
                (args.outdir / f"stages_warm_{i}.json").write_text(json.dumps(stages, indent=2) + "\n")
            if profiler is not None:
                profiler.disable()
                profiler.dump_stats(str(args.outdir / f"profile_warm_{i}.pstats"))
                with (args.outdir / f"profile_warm_{i}.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(60)
            seconds = time.perf_counter() - start
            if d is None or not np.array_equal(written, keys):
                raise ValueError("lost event population")
            digest = fingerprint(d)
            if repeat == 0:
                first[i] = digest
                fastdigi.write_digis(args.outdir / ("fastdigi." + Path(steps).name), written, d, c.nADCSamples)
            phase = "cold" if repeat == 0 and i == 0 else ("first-pass" if repeat == 0 else "warm")
            row = dict(repeat=repeat, file_index=i, phase=phase,
                       events=len(keys), steps=nstep, digis=len(d["sid"]), seconds=seconds,
                       stats=stats, reproducible=digest == first[i],
                       maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if args.backend == "jax":
                from trkfastsim import digi_jax as dj, digi_clusters_jax as dc
                row["device_memory"] = jax.devices()[0].memory_stats()
                row["jit_cache"] = {"jitter": dj._jitter_block._cache_size(),
                                    "charge": dj._charge_kernel(c)._cache_size(),
                                    "charge_group": dj._charge_blocks_kernel(c)._cache_size(),
                                    "electronics_factory": dj._electronics_kernel.cache_info()._asdict(),
                                    "counts": dc._counts._cache_size(), "sample": dc._sample._cache_size()}
            rows.append(row)
            print(json.dumps(row), flush=True)
            report = dict(backend=args.backend, device=device, seed=args.seed, steps=args.steps,
                          events=args.events, conditions=args.cond, chunk=1000, runs=rows,
                          resource_tuning=({"waveform_lanes": digi_jax.WAVEFORM_CLUSTER_LANES,
                                            "straw_batch": digi_jax.BATCH, "cluster_batch": digi_clusters_jax.BATCH_SIZE,
                                            "sample_lanes": digi_clusters_jax.MAX_SAMPLE_LANES,
                                            "charge_blocks": digi_jax.CHARGE_BLOCKS, "saturation_tile": digi_jax.SATURATION_TILE}
                                           if args.backend == "jax" else {}),
                          reproducible=all(r["reproducible"] for r in rows))
            (args.outdir / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
            if not row["reproducible"]:
                raise AssertionError("same seed/input did not reproduce digitisation")


if __name__ == "__main__":
    main()
