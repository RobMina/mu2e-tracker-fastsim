"""Bounded primary-only producer-capture/default identity check, no physics-closure claim.

python3 scripts/transport_capture_check.py --gen INPUT --gdml GEOMETRY --base-source BASE_STEPS \
    --max-events 50000 --outdir NEW_DIR [--arm legacy|AMT]
The baseline source is loaded read-only beside the candidate, using the same pinned core.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from trkfastsim import geometry, steps, surfaces, tables, transportcapture


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inventory(capture, outdir):
    """Summarize model-state observations without assigning physical terminal positions."""
    result = {}
    for name in ("interior", "exit"):
        phase = capture.get_phase(name)
        summary = phase.summary()
        summary.to_csv(outdir / f"{name}_summary.csv", index=False)
        restarts = phase.records("legacy_restart")[0][1]
        path_restart = restarts["path"] + summary.phase_path_offset_mm.to_numpy()
        delta = summary.last_path_mm.to_numpy() - path_restart
        result[name] = dict(lanes=len(summary), dispositions=summary.disposition.value_counts().to_dict(),
                            last_kind=summary.last_kind.value_counts().to_dict(),
                            observations=sum(len(v["lane"]) for _, v in phase.records()),
                            last_observed_ahead_of_legacy_restart=int((delta > 1e-9).sum()),
                            max_last_observed_minus_restart_mm=float(delta.max()) if len(delta) else None)
        if name == "interior":
            stopped = summary.disposition == "unresolved_stop"
            result[name]["stopped_observed_lanes"] = int(stopped.sum())
            result[name]["legacy_stopped_flags"] = int(restarts["stopped"].sum())
            result[name]["stopped_flag_identity_equal"] = bool(np.array_equal(stopped, restarts["stopped"]))
    return result


def run(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    paths = (args.gen, args.gdml, args.base_source, __file__, *transportcapture.source_paths().values())
    before = {str(p): digest(p) for p in paths}
    spec = importlib.util.spec_from_file_location("trkfastsim._capture_base", args.base_source)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    geom = geometry.from_gdml(args.gdml)
    gen = tables.read(args.gen, "gen")
    all_events = gen[tables.KEY].drop_duplicates()
    selected = all_events.iloc[:args.max_events]
    gen = gen.merge(selected, on=tables.KEY, how="inner", sort=False, validate="many_to_one")
    results, runtimes, observed = [], [], None
    for module, capture in ((base, False), (steps, False), (steps, True)):
        rng = np.random.default_rng([args.seed, 0])
        world, interior = module.load_world(args.gdml, geom), module.TrackerInterior(args.gdml, geom)
        started = time.perf_counter()
        sgs, parts = module.make_steps(gen, geom, rng, stage="C", world=world, interior=interior,
            sliced_msc=True, arms=args.arm, history=True, **({"capture_transport": True} if capture else {}))
        vertices = module.make_steps.path_vertices.copy()
        surf = surfaces.surface_steps(sgs, vertices)
        runtimes.append(time.perf_counter() - started)
        results.append(dict(steps=sgs, particles=parts, vertices=vertices, surfaces=surf,
            segments=module.make_steps.segment_ledger.copy(), muons=module.make_steps.muon_ledger.copy(),
            rng=copy.deepcopy(rng.bit_generator.state), seed_sequence=copy.deepcopy(rng.bit_generator.seed_seq.state),
            subsequent=rng.random(16), stats=copy.deepcopy(module.make_steps.interior_stats)))
        if capture:
            observed = module.make_steps.transport_capture
    for candidate in results[1:]:
        for key in ("steps", "particles", "vertices", "surfaces", "segments", "muons"):
            pd.testing.assert_frame_equal(results[0][key], candidate[key], check_exact=True)
        for key in ("rng", "seed_sequence", "stats"):
            assert results[0][key] == candidate[key], key
        np.testing.assert_array_equal(results[0]["subsequent"], candidate["subsequent"])
    assert observed.complete
    observed.save(out / "capture.npz")
    summary = inventory(observed, out)
    after = {str(p): digest(p) for p in paths}
    if before != after:
        raise RuntimeError("input/source content changed during verification")
    report = dict(scope="bounded primary-only algorithm capture; not physical history or rate/reco validation",
        identical=True, arm=args.arm, seed=args.seed, generated_file_events=len(all_events),
        processed_events=len(selected), generated_rows=len(gen), primary_muons=len(results[0]["particles"]),
        steps=len(results[0]["steps"]), vertices=len(results[0]["vertices"]), surfaces=len(results[0]["surfaces"]),
        seconds_base_off_on=runtimes, sha256=before, captured_source_sha256=observed.sources,
        environment=observed.environment,
        phases=summary, capture_sha256=digest(out / "capture.npz"))
    with (out / "complete.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gen", "gdml", "base-source", "outdir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arm", choices=("legacy", "AMT"), default="legacy")
    args = parser.parse_args()
    if args.max_events < 1:
        parser.error("max-events must be positive")
    run(args)


if __name__ == "__main__":
    main()
