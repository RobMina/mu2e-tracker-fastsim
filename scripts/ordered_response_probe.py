"""Bounded actual ordered-step response probe; slice arms are diagnostics, not filter inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables
from scripts.secondary_bank_response import digi_view, measure
from scripts.transport_capture_audit import digest
from trkfastsim import crossings, digicond, geometry, ledger, steps, tables
from trkfastsim.analyses import digicmp, fastdigi
from trkfastsim.reco import hitreco

KEY = tables.KEY + ["sp"]


def slice_rows(merged, saved, geom):
    """Reconstruct actual gas slices and compare them to the unchanged merged writer."""
    gas = saved["slice_material"] == steps.STRAW_GAS.name
    identities = np.asarray(saved["muon_keys"])
    if (identities.ndim != 2 or identities.shape[1] != len(KEY) or identities.dtype.kind not in "iu"
            or (identities < 0).any() or pd.DataFrame(identities).duplicated().any()):
        raise ValueError("unique nonnegative integer full primary identities required")
    lane = saved["slice_it"][gas]
    if lane.dtype.kind not in "iu" or (lane < 0).any() or (lane >= len(identities)).any():
        raise ValueError("gas slice lane outside primary identity universe")
    origin, end = saved["slice_pos0"][gas], saved["slice_pos1"][gas]
    direction = saved["slice_dir"][gas]
    length = saved["slice_end"][gas] - saved["slice_start"][gas]
    deposit = saved["slice_deposited"][gas]
    if (not all(np.isfinite(value).all() for value in (origin, end, direction, length, deposit,
                 saved["slice_ke0"][gas], saved["slice_t0"][gas], saved["t_start_ns"]))
            or (length <= 0).any() or (deposit < 0).any() or (saved["slice_ke0"][gas] < 0).any()
            or not np.allclose(np.linalg.norm(direction, axis=1), 1., rtol=0, atol=1e-10)
            or not np.allclose(end, origin + direction * length[:, None], rtol=0, atol=1e-7)):
        raise ValueError("nonphysical saved gas slice state")
    if np.asarray(saved["t_start_ns"]).shape != (len(identities),):
        raise ValueError("absolute clock lane coverage mismatch")
    js = crossings.point_straw((origin + end) * .5, geom)
    if (js < 0).any():
        raise ValueError("saved gas slice lacks a straw owner")
    labels = pd.DataFrame(saved["muon_keys"][lane], columns=KEY)
    labels["sid"] = geom.sid[js]
    labels["_slice"] = np.arange(len(lane))
    index = labels.merge(merged[KEY + ["sid"]].assign(_row=np.arange(len(merged))),
                         on=KEY + ["sid"], validate="many_to_one", how="left")
    if index._row.isna().any() or merged.duplicated(KEY + ["sid"]).any():
        raise ValueError("saved gas slices and merged rows differ in primary/straw coverage")
    index = index.sort_values("_slice", kind="stable")
    row = index._row.to_numpy(np.int64)
    resolved = merged.iloc[row].reset_index(drop=True).copy()
    resolved["len"] = length
    resolved["eion"] = saved["slice_deposited"][gas]
    resolved["t"] = saved["t_start_ns"][lane] + saved["slice_t0"][gas]
    ke = saved["slice_ke0"][gas]
    momentum = ledger.momentum(ke)
    resolved["ion"] = np.where((momentum / steps.eloss.MMU > steps.MINION_BG) & (ke > steps.MINION_KE), 0, 1)
    p0, p1 = geometry.to_detector(origin), geometry.to_detector(end)
    for j, axis in enumerate("xyz"):
        resolved[f"{axis}0"], resolved[f"{axis}1"] = p0[:, j], p1[:, j]
        resolved[f"p{axis}"] = saved["slice_dir"][gas, j] * momentum
    resolved["nsgs"] = resolved.groupby(tables.KEY)["sid"].transform("size")
    sums = pd.DataFrame(dict(row=row, length=length, eion=resolved.eion)).groupby("row").sum()
    if not np.array_equal(sums.index.to_numpy(), np.arange(len(merged))):
        raise ValueError("merged writer has rows without actual gas slices")
    for column, value in (("len", sums.length), ("eion", sums.eion)):
        if not np.allclose(merged[column], value, rtol=0, atol=1e-10):
            raise ValueError(f"saved slice {column} differs from merged writer")
    grouped = pd.DataFrame(dict(row=row, index=np.arange(len(row)))).groupby("row").index
    first, last = grouped.first().to_numpy(), grouped.last().to_numpy()
    for endpoint, positions in ((0, p0[first]), (1, p1[last])):
        if not np.allclose(merged[[f"{a}{endpoint}" for a in "xyz"]], positions, rtol=0, atol=1e-8):
            raise ValueError("saved slice endpoints differ from merged writer")
    if not np.allclose(merged.t, resolved.t.to_numpy()[first], rtol=0, atol=1e-8):
        raise ValueError("saved slice clock differs from merged entry")
    a = merged[[f"{v}0" for v in "xyz"]].to_numpy(float)
    b = merged[[f"{v}1" for v in "xyz"]].to_numpy(float)
    chord = np.linalg.norm(b - a, axis=1)
    if (chord > merged.len.to_numpy() + 1e-8).any():
        raise ValueError("merged chord exceeds actual gas arc")
    tangent = b - a
    fraction = np.divide(np.sum((p0 - a[row]) * tangent[row], axis=1), chord[row] ** 2,
                         out=np.zeros(len(row)), where=chord[row] > 0)
    displacement = np.linalg.norm(p0 - (a[row] + np.clip(fraction, 0, 1)[:, None] * tangent[row]), axis=1)
    multiplicity = np.bincount(row, minlength=len(merged))
    spans = saved["slice_end"][gas][last] - saved["slice_start"][gas][first]
    straight = resolved.copy()
    accumulated = pd.Series(length).groupby(row, sort=False).cumsum().to_numpy()
    f1 = accumulated / merged.len.to_numpy()[row]
    f0 = (accumulated - length) / merged.len.to_numpy()[row]
    for j, axis in enumerate("xyz"):
        straight[f"{axis}0"] = a[row, j] + f0 * tangent[row, j]
        straight[f"{axis}1"] = a[row, j] + f1 * tangent[row, j]
    audit = dict(merged_rows=len(merged), actual_gas_slices=len(row), multiple_slice_rows=int((multiplicity > 1).sum()),
        max_slices_per_row=int(multiplicity.max(initial=0)), total_gas_arc_mm=float(length.sum()),
        total_merged_chord_mm=float(chord.sum()), max_arc_minus_chord_mm=float(np.max(merged.len.to_numpy() - chord, initial=0)),
        max_slice_start_distance_from_merged_chord_mm=float(displacement.max(initial=0)),
        non_gas_path_inside_merged_span_mm=float(np.maximum(spans - merged.len.to_numpy(), 0).sum()),
        max_slice_entry_delay_from_merged_ns=float(np.max(resolved.t.to_numpy() - merged.t.to_numpy()[row], initial=0)),
        energy_sum_MeV=float(resolved.eion.sum()))
    return resolved, straight, audit


def complete_keys(gen):
    keys = gen[tables.KEY].drop_duplicates().sort_values(tables.KEY).to_numpy(np.int64)
    if not 1 <= len(keys) <= 1000:
        raise ValueError("probe requires 1..1000 complete generated events")
    return keys


def run(directory, gdml, conditions, hit_conditions, drift_source, outdir, *, draws=4, expected_baseline=None):
    if not 2 <= draws <= 8:
        raise ValueError("response probe needs2..8 independent response draws per arm")
    directory, outdir = Path(directory), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    paths = [directory / "complete.json", directory / "ordered_steps.root", directory / "traversed.npz",
             *map(Path, (gdml, conditions, hit_conditions, drift_source))]
    if expected_baseline is not None:
        paths.append(Path(expected_baseline))
    hashes = {str(p): digest(p) for p in paths}
    receipt = json.loads(paths[0].read_text())
    if receipt.get("status") != "bounded_full_chain_complete" or receipt["input_sha256"].get(str(gdml)) != hashes[str(gdml)]:
        raise ValueError("completed ordered chain with the same geometry required")
    merged = tables.read(paths[1], "sgsteps")
    if expected_baseline is not None:
        for tree in ("sgsteps", "particles", "gen", "surfsteps"):
            pd.testing.assert_frame_equal(tables.read(paths[1], tree), tables.read(expected_baseline, tree),
                                          check_exact=True, check_dtype=True)
    gen = tables.read(paths[1], "gen")
    keys = complete_keys(gen)
    if len(keys) != receipt["generated_events"]:
        raise ValueError("generated denominator differs from completed producer")
    with np.load(paths[2], allow_pickle=False) as saved:
        resolved, straight, audit = slice_rows(merged, saved, geometry.from_gdml(gdml))
    np.save(outdir / "event_keys.npy", keys)
    c, hc = digicond.Conditions.load(conditions), hitreco.load_conditions(hit_conditions)
    selected = tables.trk_filter(merged).reset_index()
    selected.to_csv(outdir / "unchanged_merged_filter.csv", index=False)
    rows, started = [], time.perf_counter()
    for arm, frame in (("merged", merged), ("actual_slices", resolved), ("straightened_slices", straight)):
        step_path = outdir / f"{arm}.root"
        core_tables.write(str(step_path), dict(sgsteps=frame, gen=gen), as_ttree=True)
        for draw in range(draws):
            seed = int(np.random.SeedSequence([314159, draw, {"merged": 0, "actual_slices": 1, "straightened_slices": 2}[arm]]).generate_state(1)[0])
            stats = {}
            returned, digis, count = fastdigi.digitize_table(str(step_path), c, np.random.default_rng(seed), keys=keys, chunk=64, stats=stats)
            if not np.array_equal(returned, keys) or count != len(frame):
                raise ValueError("digitizer changed generated-event or step coverage")
            fits = digicmp.f2_yield(digi_view(keys, digis), hc, drift_source, chunk=64, fixed_shapes=True)
            metrics = measure(digis, fits["fits_per_event"], len(keys))
            np.savez_compressed(outdir / f"response.{arm}.{draw}.npz", evkey=keys, fits_per_event=fits["fits_per_event"], **digis)
            row = dict(arm=arm, draw=draw, seed=seed, metrics=metrics, stats=stats, good_fits=fits["good_fits"],
                       events_with_fit=fits["events_with_fit"], seeds=fits["seeds"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    sources = [Path(__file__), Path(fastdigi.__file__), Path(digicmp.__file__), Path(steps.__file__)]
    result = dict(scope="bounded ordered primary-only NumPy digi/fast F2 representation sensitivity; not stock reco or rate closure",
        generated_events=len(keys), unchanged_merged_filter_passes=int(selected.pass_trk.sum()), geometry=audit, rows=rows,
        exact_previous_product_match=expected_baseline is not None,
        arms="actual_slices uses physical gas slice endpoints/loss/time; straightened_slices uses same per-slice loss/time and cumulative gas fraction on merged chord",
        warnings=["all generated keys retained including empty events", "slice tables must never feed PrimaryFilter; multiplicity is diagnostic only",
                  "independent response draws, not same-random-history differences", "straightened geometry is an explicit counterfactual, not production correction",
                  "finite path/omitted solids and gas energy partition gates remain open"],
        elapsed_s=time.perf_counter() - started, input_sha256=hashes, source_sha256={str(p): digest(p) for p in sources}, admitted_for_production=False)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("ordered response input changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("gdml", "conditions", "hit-conditions", "drift-source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--expected-baseline", default=None)
    args = parser.parse_args()
    run(args.directory, args.gdml, args.conditions, args.hit_conditions, args.drift_source, args.outdir,
        draws=args.draws, expected_baseline=args.expected_baseline)
