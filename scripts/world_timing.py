"""Warm whole world transport timing plus named kernel timing on real primaries.

python3 scripts/world_timing.py --gen GEN.root --gdml WORLD.gdml --out NEW.json [--lanes 16384]
"""
import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np
from trkfastsim import geometry, steps, steps_jax, tables, world_jax
from trkfastsim.analyses.jaxcheck import _kernel_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gen", "gdml", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--lanes", type=int, default=16384)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    gen = tables.read(args.gen, "gen")
    geom = geometry.from_gdml(args.gdml)
    world = steps.load_world(args.gdml, geom)
    steps_jax.LANES = args.lanes
    run = lambda: world_jax.to_plane(gen, geom, steps_jax.key_from(np.random.default_rng([1, 0])), world, sliced_msc=True)
    results, times = [], []
    for _ in range(2):
        start = time.perf_counter()
        results.append(run())
        times.append(time.perf_counter() - start)
    for a, b in zip(results[0][1:], results[1][1:]):
        np.testing.assert_array_equal(a, b)
    kt = _kernel_times(steps_jax, run)
    report = {"device": str(jax.devices()[0]), "times": times, "lanes": args.lanes,
              "generated_events": len(gen[tables.KEY].drop_duplicates()), "kernel_times": kt,
              "survived": int(results[0][-1].sum()), "reproducible": True}
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
