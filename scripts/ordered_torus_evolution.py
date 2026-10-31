"""Conditional saved-handoff torus evolution with re-navigation; not inclusive physics closure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_torus_pair_probe import KEY, load_pair_sources, stored_hits
from scripts.transport_capture_audit import digest
from trkfastsim import crossings, geometry, guardednavigation, ledger, orderedinterior, steps


def safe_payload(values):
    """Store material/provenance labels as Unicode; never require NumPy object-pickle loading."""
    result = {}
    for name, value in values.items():
        value = np.asarray(value)
        if value.dtype.kind == "O":
            if value.size and pd.api.types.infer_dtype(value.ravel(), skipna=False) != "string":
                raise ValueError("only string labels may use object dtype before serialization")
            value = value.astype(str)
        result[name] = value
    return result


def initial_state(directory, identities):
    with np.load(Path(directory) / "traversed.npz", allow_pickle=False) as saved:
        keys = pd.DataFrame(saved["muon_keys"], columns=KEY).assign(lane=np.arange(len(saved["muon_keys"])))
        chosen = keys.merge(identities, on=KEY, validate="one_to_one")
        if len(chosen) != len(identities) or chosen.empty:
            raise ValueError("all requested saved primary identities required")
        starts = pd.DataFrame(dict(lane=saved["slice_it"], row=np.arange(len(saved["slice_it"])), path=saved["slice_start"]))
        rows = chosen.merge(starts[starts.path == 0], on="lane", validate="one_to_one").row.to_numpy(np.int64)
        gas = pd.read_csv(Path(directory) / "segments.csv").groupby(KEY).path_mm.min().reset_index()
        limits = chosen.merge(gas, on=KEY, validate="one_to_one").path_mm.to_numpy() + 10.
        if len(rows) != len(chosen) or len(limits) != len(chosen) or not np.isfinite(limits).all() or (limits <= 10.).any():
            raise ValueError("every requested primary requires initial slice and later saved gas entry")
        return chosen[KEY], saved["slice_pos0"][rows], saved["slice_dir"][rows], ledger.momentum(saved["slice_ke0"][rows]), limits


def bind_pair_inputs(pair_receipt, directory, gdml, source):
    required = [Path(directory) / n for n in ("traversed.npz", "segments.csv", "complete.json")]
    required += [Path(gdml), *[Path(source) / f"{n}.py" for n in (
        "navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope")]]
    recorded = pair_receipt["input_sha256"]
    for path in required:
        if str(path) not in recorded or digest(path) != recorded[str(path)]:
            raise ValueError("evolution input is not bound to paired witness: " + str(path))


def save_navigation(provider, out):
    for i, call in enumerate(provider.calls):
        target = out / f"navigation_{i:04d}"
        target.mkdir()
        ray_arrays = {key: call[key] for key in ("origin", "direction", "lengths", "interior_lanes", "primary_keys", "source_selection_bbox_mm") if key in call}
        np.savez_compressed(target / "rays.npz", **ray_arrays)
        def save_products(products, prefix=""):
            for name, value in products.items():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(target / f"{prefix}{name}.csv", index=False)
                elif isinstance(value, dict):
                    save_products(value, prefix + name + ".")
        save_products(call)
        if "reports" in call:
            (target / "placements.json").write_text(json.dumps(call["reports"], indent=2) + "\n")
        if "expanded_census" in call:
            (target / "expanded_census.json").write_text(json.dumps(call["expanded_census"], indent=2) + "\n")
        receipt = {k: call[k] for k in ("stage", "status", "reason", "details", "iteration", "replacements_strictly_below_plane", "replacements_within_forward_domain", "allow_start_contact") if k in call}
        receipt.update(policy=provider.policy, world_plane_y_mm=provider.plane, endpoint_budget_mm=provider.endpoint_budget,
            composition_additional_endpoint_shift_mm=0., min_cell_mm=provider.min_cell,
            max_component_mm=provider.component_budget, max_ray_unknown_mm=provider.ray_budget,
            max_cells=provider.max_cells, max_pairs=provider.max_pairs, forward_only=getattr(provider, "forward_only", False),
            endpoint_summary=call.get("endpoints", {}).get("summary"),
            envelope_summary=call.get("result", {}).get("summary"))
        (target / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def run(directory, pair, gdml, source, outdir):
    directory, pair, gdml, source, outdir = map(Path, (directory, pair, gdml, source, outdir))
    paths = [directory / "traversed.npz", directory / "segments.csv", directory / "complete.json",
             pair / "complete.json", gdml, *[source / f"{n}.py" for n in (
                 "navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope")]]
    hashes = {str(p): digest(p) for p in paths}
    if json.loads(paths[2].read_text())["status"] != "bounded_full_chain_complete":
        raise ValueError("completed saved ordered producer required")
    pair_receipt = json.loads((pair / "complete.json").read_text())
    bind_pair_inputs(pair_receipt, directory, gdml, source)
    for path, value in pair_receipt["input_sha256"].items():
        if digest(path) != value:
            raise ValueError("paired geometry source identity changed")
    keys, origin, direction, momentum, limits = initial_state(directory, pd.DataFrame(pair_receipt["identities"]))
    if len(keys) != 1:
        raise ValueError("this bounded incoming-witness runner requires exactly one primary")
    modules = load_pair_sources(source)
    geom = geometry.from_gdml(gdml)
    lo, hi = crossings.panel_boxes(geom)
    bbox = (lo.min(0) - 300., hi.max(0) + 300.)
    census = modules["navigationcensus"].census(gdml, bbox)
    outdir.mkdir(parents=True, exist_ok=False)
    keys.to_csv(outdir / "identities.csv", index=False)
    np.savez_compressed(outdir / "initial.npz", origin=origin, direction=direction, momentum=momentum, limits=limits)
    summaries = {}
    for mode in orderedinterior.MODES:
        for arm in ("stored_control", "inner", "outer"):
            tag = ("sampled" if mode == orderedinterior.SAMPLED_MODE else "mean") + "_" + arm
            output = outdir / tag
            output.mkdir()
            interior = steps.TrackerInterior(gdml, geom)
            provider = None if arm == "stored_control" else guardednavigation.GuardedNavigator(arm=arm, gdml=gdml,
                census=census, modules=modules, raw_hits=lambda o, d, l: stored_hits(gdml, bbox, o, d, l),
                world_plane_y_mm=steps.plane_y(geom), endpoint_budget_mm=1e-10, min_cell_mm=1e-7,
                max_component_mm=1e-6, max_ray_unknown_mm=1e-5)
            try:
                rows = orderedinterior.transport(interior, np.random.default_rng(19), origin, direction, momentum, geom,
                    tail=False, tail_k=1., mode=mode, stopping_evaluation="midpoint", signed_endpoint_budget_mm=1e-10,
                    max_path_mm=float(limits[0]), navigation_provider=provider,
                    progress=lambda v: print(json.dumps(dict(arm=tag, progress=v)), flush=True))
            except (ValueError, RuntimeError) as error:
                summary = dict(status="refused", reason=str(error), details=getattr(error, "details", None))
            else:
                payload = dict(**{f"gas_{k}": v for k, v in rows.items()},
                    **{f"slice_{k}": v for k, v in interior.last_slices.items()},
                    **{f"final_{k}": v for k, v in interior.last_final.items()})
                np.savez_compressed(output / "evolved.npz", **safe_payload(payload))
                slices = interior.last_slices
                gas = slices["material"] == steps.STRAW_GAS.name if slices else np.zeros(0, bool)
                support = slices["kind"] == "support" if slices else np.zeros(0, bool)
                summary = dict(status="conditional_evolution_complete", stats=interior.last_stats,
                    gas_rows=len(rows["it"]), gas_slices=int(gas.sum()),
                    first_gas_path_mm=float(slices["start"][gas][0]) if gas.any() else None,
                    first_gas_ke_MeV=float(slices["ke0"][gas][0]) if gas.any() else None,
                    first_gas_dt_ns=float(slices["t0"][gas][0]) if gas.any() else None,
                    support_debit_MeV=float((slices["ke0"][support] - slices["ke1"][support]).sum()) if slices else 0.,
                    final_ke_MeV=interior.last_final["ke"].tolist(), final_dt_ns=interior.last_final["dt"].tolist(),
                    balance_max_MeV=float(np.max(np.abs(ledger.kinetic(momentum) - interior.last_final["ke"] -
                        interior.last_final["lost_support"] - interior.last_final["lost_thin"]))))
            if provider is not None:
                save_navigation(provider, output)
                summary["navigation_calls"] = len(provider.calls)
            summaries[tag] = summary
            (output / "receipt.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            print(json.dumps(dict(arm=tag, result=summary), allow_nan=False), flush=True)
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("conditional evolution inputs changed")
    result = dict(policy=guardednavigation.POLICY, scope="fixed saved post-world/backprojected handoff; first baseline gas path plus 10mm",
        effective_settings=dict(max_path_mm=float(limits[0]), max_step_mm=10., fractional_loss=.05, stop_ke_MeV=.5,
            max_iter=20000, boundary_resolution_mm=1e-9, max_boundary_relabel_mm=1e-7, signed_endpoint_budget_mm=1e-10,
            stopping_evaluation="midpoint", tail=False, rng_seed=19, min_cell_mm=1e-7,
            max_component_mm=1e-6, max_ray_unknown_mm=1e-5, max_cells=1000000, max_pairs=2000000,
            world_plane_y_mm=float(steps.plane_y(geom))),
        handoff="conditional immutable state; aggregate upstream source ownership and backprojection not certified",
        warnings=["independently evolved geometric arms are sensitivity comparisons, not energy/time enclosures",
                  "finite requested arc may not reach gas after re-navigation; no extra tail added",
                  "same seed per arm is not a common-variates physical uncertainty model",
                  "no full GDML admission, rate closure or default promotion"], summaries=summaries,
        input_sha256=hashes, source_sha256={str(p): digest(p) for p in [Path(__file__), Path(guardednavigation.__file__),
            Path(orderedinterior.__file__)]}, admitted_for_production=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("pair", "gdml", "source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    run(args.directory, args.pair, args.gdml, args.source, args.outdir)
