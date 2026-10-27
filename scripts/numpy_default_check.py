"""Compare the current numpy default to a saved base steps.py on the same real generator table.

Usage: python3 scripts/numpy_default_check.py --base-source BASE/steps.py --gen GEN.root --gdml WORLD.gdml --out NEW.json
Run with the same pinned core for both modules; no JAX imports or output ROOT files are needed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from trkfastsim import geometry, steps, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-source", "gen", "gdml", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    spec = importlib.util.spec_from_file_location("trkfastsim._numpy_base", args.base_source)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    geom = geometry.from_gdml(args.gdml)
    gen = tables.read(args.gen, "gen")
    results, timings, states = [], [], []
    for mod in (base, steps):
        rng = np.random.default_rng([1, 0])
        world = mod.load_world(args.gdml, geom)
        interior = mod.TrackerInterior(args.gdml, geom)
        t0 = time.perf_counter()
        sgs, parts = mod.make_steps(gen, geom, rng, stage="C", world=world, interior=interior, sliced_msc=True)
        timings.append(time.perf_counter() - t0)
        results.append((sgs, parts, mod.make_steps.path_vertices))
        states.append(rng.bit_generator.state)
    for a, b in zip(*results):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert states[0] == states[1]
    report = {"identical": True, "generated_events": len(gen[tables.KEY].drop_duplicates()),
              "generator_rows": len(gen), "steps": len(results[0][0]), "muons": len(results[0][1]),
              "path_vertices": len(results[0][2]), "seconds": timings,
              "sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in (args.base_source, steps.__file__, args.gen, args.gdml)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
