"""Passive world-column capture on the exact saved producer prefix; no core edits or interior rerun."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import world as world_module
from scripts.transport_capture_audit import digest
from trkfastsim import convexpath, geometry, steps, tables

KEY = tables.KEY + ["sp"]


def expected_queries(gen, geom, world):
    mu = steps._muons(gen, True)
    p = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    o = mu[["x", "y", "z"]].to_numpy(float)
    d = mu[["px", "py", "pz"]].to_numpy(float) / p[:, None]
    plane = steps.plane_y(geom)
    near = steps.near_tracker(o, d, geom, plane, margin=steps.FOOTPRINT_MARGIN)
    labels = mu[tables.KEY].copy().assign(sp=mu.mukey)
    o, d, labels = o[near], d[near], labels.loc[near].reset_index(drop=True)
    length = (o[:, 1] - plane) / -d[:, 1]
    queries = []
    for band in range(len(world.edges) - 1):
        top, bot = world.edges[band:band + 2]
        a, b = np.clip((o[:, 1] - top) / -d[:, 1], 0, length), np.clip((o[:, 1] - bot) / -d[:, 1], 0, length)
        use = (b - a) > 0
        if use.any() and len(world.band(band)):
            queries.append(dict(band=band, origin=(o + d * a[:, None])[use], direction=d[use],
                length=(b - a)[use], labels=labels.loc[use].reset_index(drop=True)))
    return mu, queries


def run(directory, gdml, outdir):
    directory, gdml, outdir = map(Path, (directory, gdml, outdir))
    paths = [directory / "ordered_steps.root", directory / "muons.csv", directory / "complete.json", gdml]
    before = {str(p): digest(p) for p in paths}
    receipt = json.loads(paths[2].read_text())
    if receipt["status"] != "bounded_full_chain_complete":
        raise ValueError("completed producer prefix required")
    gen = tables.read(paths[0], "gen")
    original_gen_path = next((p for p in receipt["input_sha256"] if "gen." in Path(p).name), None)
    if original_gen_path is None or digest(original_gen_path) != receipt["input_sha256"][original_gen_path]:
        raise ValueError("original generator input identity unavailable")
    original_gen = tables.read(original_gen_path, "gen")
    original_gen = original_gen.merge(gen[tables.KEY].drop_duplicates(), on=tables.KEY, validate="many_to_one")
    pd.testing.assert_frame_equal(gen, original_gen, check_exact=True, check_dtype=True)
    paths.append(Path(original_gen_path))
    before[str(original_gen_path)] = digest(original_gen_path)
    geom = geometry.from_gdml(gdml)
    world = steps.load_world(gdml, geom)
    mu, expected = expected_queries(gen, geom, world)
    args = dict(sliced_msc=True, msc_tail=False, history=True)
    rng = np.random.default_rng(19)
    baseline = steps.to_plane(gen, geom, rng, world, **args)
    baseline_rng, baseline_seq = copy.deepcopy(rng.bit_generator.state), copy.deepcopy(rng.bit_generator.seed_seq.state)
    baseline_time = steps.to_plane.last_history["t_world"].copy()
    captured = []
    original = world_module.column

    def observe(pieces, origin, direction, length, *a, **kw):
        index = len(captured)
        if index >= len(expected):
            raise ValueError("unexpected world-column call")
        query = expected[index]
        for left, right in ((origin, query["origin"]), (direction, query["direction"]), (length, query["length"])):
            np.testing.assert_array_equal(left, right)
        values = original(pieces, origin, direction, length, *a, **kw)
        captured.append(dict(query, columns={name: value.copy() for name, value in values.items()}))
        return values

    world_module.column = observe
    rng = np.random.default_rng(19)
    try:
        replay = steps.to_plane(gen, geom, rng, world, **args)
    finally:
        world_module.column = original
    if len(captured) != len(expected):
        raise ValueError("missing world-column calls")
    pd.testing.assert_frame_equal(baseline[0], replay[0], check_exact=True)
    for a, b in zip(baseline[1:], replay[1:]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(baseline_time, steps.to_plane.last_history["t_world"])
    if baseline_rng != rng.bit_generator.state or baseline_seq != rng.bit_generator.seed_seq.state:
        raise ValueError("passive capture changed RNG state/seed addressing")
    ledger = pd.read_csv(paths[1])
    plane = mu[tables.KEY].copy().assign(sp=mu.mukey, p_plane_MeV_c=replay[3], alive_plane=replay[5])
    compared = plane.merge(ledger[KEY + ["p_plane_MeV_c", "alive_plane"]], on=KEY, validate="one_to_one", suffixes=("_replay", "_saved"))
    if len(compared) != len(plane) or not np.array_equal(compared.p_plane_MeV_c_replay, compared.p_plane_MeV_c_saved):
        if not np.allclose(compared.p_plane_MeV_c_replay, compared.p_plane_MeV_c_saved, rtol=3e-16, atol=0):
            raise ValueError("world replay differs from saved producer plane momentum")
    if not np.array_equal(compared.alive_plane_replay, compared.alive_plane_saved):
        raise ValueError("world replay differs from saved delivered identities")
    outdir.mkdir(parents=True, exist_ok=False)
    arrays, summaries, target_rows = {}, [], []
    for index, query in enumerate(captured):
        prefix = f"query{index}_"
        col = query["columns"]
        solid = sum(col.values(), np.zeros(len(query["length"])))
        residual = np.maximum(query["length"] - solid, 0.)
        arrays.update({prefix + k: v for k, v in (("keys", query["labels"][KEY].to_numpy(np.int64)),
            ("origin", query["origin"]), ("direction", query["direction"]), ("length", query["length"]), ("residual_air", residual))})
        for material, value in col.items():
            arrays[prefix + "column_" + material] = value
        summaries.append(dict(index=index, band=query["band"], rows=len(solid), materials=list(col),
            summed_solid_mm=float(solid.sum()), residual_air_mm=float(residual.sum()),
            overlap_excess_mm=float(np.maximum(solid - query["length"], 0).sum())))
        target = query["labels"].event.to_numpy() == 6499
        if target.any():
            chosen = query["labels"].loc[target].copy()
            chosen["band"], chosen["length_mm"], chosen["residual_air_mm"] = query["band"], query["length"][target], residual[target]
            for name, value in col.items():
                chosen[f"column_{name}_mm"] = value[target]
            target_rows.append(chosen)
    np.savez_compressed(outdir / "world_queries.npz", **arrays)
    pd.concat(target_rows, ignore_index=True).to_csv(outdir / "selected6499_columns.csv", index=False)
    result = dict(scope="passive actual upstream world-column capture, exact off/on state+RNG check; no material correction",
        generated_events=receipt["generated_events"], primary_muons=len(mu), captured_calls=len(captured), queries=summaries,
        exact_off_on_states_rng=True, saved_plane_max_abs_MeV_c=float(np.max(np.abs(compared.p_plane_MeV_c_replay - compared.p_plane_MeV_c_saved))),
        warnings=["columns sum stored source intersections, not physical leaf ownership", "residual AIR reproduces core gap policy, not proof of actual air",
                  "interior ray differs after world scattering; matching material totals cannot attribute a torus debit"],
        input_sha256=before, source_sha256={str(p): digest(p) for p in (Path(__file__), Path(steps.__file__), Path(world_module.__file__), Path(steps.tr.__file__))},
        admitted_for_production=False)
    if before != {str(p): digest(p) for p in paths}:
        raise ValueError("world capture inputs changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.gdml, args.outdir)
