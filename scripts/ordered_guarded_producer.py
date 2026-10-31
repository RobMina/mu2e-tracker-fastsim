"""Bounded all-lane generator/world/guarded-interior/writer run with explicit forward handoff."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables
import mu2efastsim
import trkfastsim
from scripts.ordered_torus_pair_probe import StoredHitQuery, load_pair_sources
from scripts.ordered_torus_evolution import safe_payload, save_navigation
from scripts.transport_capture_audit import digest
from trkfastsim import crossings, geometry, guardednavigation, orderedhandoff, orderedinterior, placementownership, steps, surfaces, tables


def run(gen_path, gdml, source, outdir, *, max_events, arm, clearance_mm=20., max_path_mm=4000., physical_ownership=False, probe_cell_mm=1e-7, wide_torus=False, mixed_extent=False, native_planar=False, extra_materials=(), bounded_forward_lookahead=False):
    if isinstance(max_events, bool) or not isinstance(max_events, int) or not 0 < max_events <= 10000:
        raise ValueError("whole-event bounded producer requires 1..10000 events")
    if arm not in ("stored_control", "inner", "outer"):
        raise ValueError("explicit stored_control/inner/outer required")
    if not np.isfinite(probe_cell_mm) or not 0 < probe_cell_mm <= 1e-7:
        raise ValueError("probe refinement may only reduce the original 1e-7mm cell width")
    gen_path, gdml, source, outdir = map(Path, (gen_path, gdml, source, outdir))
    names = ("navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope")
    paths = [gen_path, gdml, *[source / f"{name}.py" for name in names]]
    hashes = {str(p): digest(p) for p in paths}
    dependencies = [*Path(mu2efastsim.__file__).parent.glob("*.py"), *Path(trkfastsim.__file__).parent.glob("*.py"),
        Path(__file__), *[Path(__file__).parent / (name + ".py") for name in (
            "ordered_torus_pair_probe", "ordered_torus_evolution", "ordered_omitted_solid_audit", "transport_capture_audit")]]
    source_hashes = {str(p): digest(p) for p in dependencies}
    gen = tables.read(gen_path, "gen")
    selected = gen[tables.KEY].drop_duplicates().iloc[:max_events]
    if len(selected) != max_events:
        raise ValueError("input does not cover all independently requested generated events")
    gen = gen.merge(selected, on=tables.KEY, validate="many_to_one")
    from trkfastsim.orderedmaterials import MaterialResolver
    resolver = MaterialResolver(gdml, extra_materials) if extra_materials else None
    modules = load_pair_sources(source)
    if native_planar and not mixed_extent:
        raise ValueError("native planar geometry requires explicit mixed full-ray extent")
    if mixed_extent and (not wide_torus or not physical_ownership):
        raise ValueError("mixed ray-extent policy requires explicit wide arithmetic and physical ownership")
    if wide_torus:
        from trkfastsim.wideomission import WideOmission
        modules = dict(modules, omittedpath=WideOmission(modules["omittedpath"], authenticate_xtru=mixed_extent))
    if native_planar:
        from trkfastsim.inclusiveenvelope import CertifiedOmission
        from trkfastsim.nativeenvelope import NativeOmission, NativeEnvelope
        reviewed_omission = modules["omittedpath"].reviewed
        probe = NativeOmission(CertifiedOmission(modules["omittedpath"], reviewed_omission), reviewed_omission)
        modules = dict(modules, omittedpath=probe,
            supportenvelope=NativeEnvelope(modules["supportenvelope"], modules["supportnavigation"], probe))
    elif mixed_extent:
        from trkfastsim.mixedenvelope import MixedEnvelope
        modules = dict(modules, supportenvelope=MixedEnvelope(modules["supportenvelope"], modules["supportnavigation"]))
    geom = geometry.from_gdml(gdml)
    interior = steps.TrackerInterior(gdml, geom)
    lo, hi = crossings.panel_boxes(geom)
    bbox = (lo.min(0) - 300., hi.max(0) + 300.)
    census = modules["navigationcensus"].census(gdml, bbox)
    handoff = orderedhandoff.plan(geom, interior, census, clearance_mm=clearance_mm)
    world = steps.load_world(gdml, geom, endpoint_y_mm=handoff["plane_y_mm"])
    raw_query = (placementownership.RaySelectedSources(gdml, bbox, local_planes=True) if mixed_extent else
                 placementownership.PlacementSources(gdml, bbox) if physical_ownership else StoredHitQuery(gdml, bbox))
    census_cache = {}
    def expanded_census(box):
        key = box.tobytes()
        if key not in census_cache:
            census_cache[key] = modules["navigationcensus"].census(gdml, box)
        return census_cache[key]
    provider = None if arm == "stored_control" else guardednavigation.GuardedNavigator(arm=arm, gdml=gdml, census=census,
        modules=modules, raw_hits=raw_query, world_plane_y_mm=handoff["plane_y_mm"], endpoint_budget_mm=1e-10,
        min_cell_mm=probe_cell_mm, max_component_mm=1e-6, max_ray_unknown_mm=1e-5, forward_only=True,
        ownership_normalizer=raw_query.normalize if physical_ownership else None,
        census_provider=expanded_census if mixed_extent else None, allow_start_contact=native_planar)
    if native_planar and provider is not None:
        from trkfastsim.nativeenvelope import POLICY as native_policy
        provider.policy += ":" + native_policy
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "census.json").write_text(json.dumps(census, indent=2) + "\n")
    (outdir / "handoff_plan.json").write_text(json.dumps(handoff, indent=2) + "\n")
    selected.to_csv(outdir / "generated_events.csv", index=False)
    primary_inventory = steps._muons(gen, True)[tables.KEY + ["mukey"]].rename(columns={"mukey": "sp"})
    primary_inventory.to_csv(outdir / "input_primaries.csv", index=False)
    (outdir / "execution_sources.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    if physical_ownership:
        raw_query.relationships.to_csv(outdir / "calo_physical_source_terms.csv", index=False)
    result = dict(scope="all requested generated events; explicit computational forward split, not physical world/GDML certification",
        generated_events=max_events, particles_generated=len(gen), navigation=orderedinterior.SAMPLED_MODE,
        geometry_policy=provider.policy if provider else "stored-same-medium-union-v1", handoff=handoff,
        arm=arm, input_sha256=hashes, source_sha256=source_hashes, admitted_for_production=False,
        ownership_policy=placementownership.POLICY if physical_ownership else "raw-unresolved-cancellation-refuses",
        torus_arithmetic="stored-placement-wide-torus-v1" if wide_torus else "reviewed-binary64-control",
        source_extent_policy="actual-ray-enclosing-selection-and-refreshed-census-v1" if mixed_extent else "original-loader-selection",
        convex_arithmetic="unflattened-local-binary64-planes-wide-rays-v1" if mixed_extent else "stored-global-binary64-planes",
        envelope_policy="native-planar-joint-root-projection-sensitivity-v1" if native_planar else "mixed-torus-xtru-plane-offset-v1" if mixed_extent else "reviewed-torus-only",
        planar_geometry_model="literal-gdml-native-planar-quarterturn-v1" if native_planar else "stored-binary64-planes-and-transforms", 
        input_primary_muons=len(primary_inventory), handoff_evidence_status="not-yet-available",
        extra_materials=resolver.receipts if resolver else {},
        lookahead_policy="bounded-in-domain-query-not-path-v1" if bounded_forward_lookahead else "full-remaining-arc-query", 
        energy_accounting="restricted gas proposal plus nonnegative unrestricted-minus-restricted mean transfer debits primary; complement unassigned, not demonstrated escape",
        effective_settings=dict(max_path_mm=max_path_mm, max_step_mm=10., fractional_loss=.05, stop_ke_MeV=.5,
            boundary_resolution_mm=1e-9, max_boundary_relabel_mm=1e-7, max_iter=20000, rng_seed=19,
            stopping_evaluation="midpoint", signed_endpoint_budget_mm=1e-10, min_cell_mm=probe_cell_mm,
            max_component_mm=1e-6, max_ray_unknown_mm=1e-5, max_cells=1000000, max_pairs=2000000, msc_tail=False))
    started = time.perf_counter()
    try:
        rows, particles = steps.make_steps(gen, geom, np.random.default_rng(19), stage="C", world=world,
            interior=interior, history=True, sliced_msc=True, msc_tail=False, navigation=orderedinterior.SAMPLED_MODE,
            handoff=orderedhandoff.POLICY, handoff_plane_y_mm=handoff["plane_y_mm"],
            ordered_options=dict(signed_endpoint_budget_mm=1e-10, stopping_evaluation="midpoint", max_path_mm=max_path_mm,
                navigation_provider=provider, material_resolver=resolver, bounded_forward_lookahead=bounded_forward_lookahead, progress=lambda v: print(json.dumps(dict(progress=v)), flush=True)))
        surf = surfaces.surface_steps(rows, steps.make_steps.path_vertices)
    except (ValueError, RuntimeError, AttributeError, TypeError, KeyError) as error:
        result.update(status="implementation-error" if isinstance(error, (AttributeError, TypeError)) else "refused",
            reason=str(error), refusal_details=getattr(error, "details", None),
            warning="whole producer refused; no rate/selection denominator or partial physics sample emitted")
    else:
        active = steps.make_steps.muon_ledger.loc[steps.make_steps.muon_ledger.path_interior_mm.notna()]
        if len(active) != len(interior.last_final["path"]):
            raise ValueError("guarded producer slice/primary coverage mismatch")
        payload = dict(muon_keys=active[tables.KEY + ["sp"]].to_numpy(np.int64),
            t_start_ns=active.t_interior_start_ns.to_numpy(float),
            **{f"final_{k}": v for k, v in interior.last_final.items()},
            **{f"slice_{k}": v for k, v in interior.last_slices.items()})
        payload = safe_payload(payload)
        result.update(status="bounded_full_chain_complete", straw_steps=len(rows), surface_steps=len(surf),
            primary_muons=len(particles), primary_pass_events=int(tables.trk_filter(rows).pass_trk.sum()),
            stats=interior.last_stats, timing=steps.make_steps.timing)
    handoff_ledger = getattr(steps.make_steps, "handoff_ledger", None)
    if handoff_ledger is not None:
        handoff_ledger.to_csv(outdir / "handoff_primaries.csv", index=False)
        result["handoff_dispositions"] = {str(k): int(v) for k, v in handoff_ledger.disposition.value_counts().items()}
        result["world_delivered"] = int(handoff_ledger.alive_world.sum())
        if len(handoff_ledger) != len(primary_inventory) or not np.array_equal(
                handoff_ledger[tables.KEY + ["sp"]].to_numpy(), primary_inventory.to_numpy()):
            raise ValueError("handoff ledger does not cover the complete input primary inventory")
        result["handoff_evidence_status"] = "complete-primary-coverage"
    forward_refusal = getattr(interior, "last_forward_refusal", None)
    if forward_refusal is not None:
        mapping = handoff_ledger.loc[handoff_ledger.accepted_forward, tables.KEY + ["sp"]].to_numpy(np.int64)
        mapped = mapping[forward_refusal["interior_lanes"]]
        np.savez_compressed(outdir / "forward_refusal.npz", **forward_refusal, primary_keys=mapped)
        result["forward_refusal_iteration"] = int(forward_refusal["iteration"])
        result["forward_refusal_primary_keys"] = mapped.tolist()
    if provider is not None:
        save_navigation(provider, outdir)
        result["navigation_queries"] = len(provider.calls)
        result["last_query_stage"] = provider.calls[-1]["stage"] if provider.calls else None
    result["elapsed_s"] = time.perf_counter() - started
    if hashes != {str(p): digest(p) for p in paths} or source_hashes != {str(p): digest(p) for p in dependencies}:
        raise ValueError("guarded producer immutable inputs or executed sources changed")
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if result["status"] == "bounded_full_chain_complete":
        staging = outdir / "unpublished"
        staging.mkdir()
        core_tables.write(str(staging / "ordered_steps.root"), dict(sgsteps=rows, particles=particles, gen=gen, surfsteps=surf), as_ttree=True)
        for name, frame in (("segments", steps.make_steps.segment_ledger), ("muons", steps.make_steps.muon_ledger),
                            ("vertices", steps.make_steps.path_vertices)):
            frame.to_csv(staging / f"{name}.csv", index=False)
        np.savez_compressed(staging / "traversed.npz", **payload)
        interior.last_lookahead_evidence.to_csv(staging / "lookahead_queries.csv", index=False)
        interior.last_endpoint_evidence.to_csv(staging / "endpoint_shifts.csv", index=False)
        interior.last_leading_source_evidence.to_csv(staging / "leading_signed_sources.csv", index=False)
        if hashes != {str(p): digest(p) for p in paths} or source_hashes != {str(p): digest(p) for p in dependencies}:
            raise ValueError("producer sources or inputs changed during output serialization; products unpublished")
        for path in staging.iterdir():
            path.rename(outdir / path.name)
        staging.rmdir()
    temporary = outdir / "completion.pending"
    temporary.write_text(serialized)
    temporary.rename(outdir / "complete.json")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gen")
    for name in ("gdml", "source", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--arm", required=True, choices=("stored_control", "inner", "outer"))
    parser.add_argument("--max-events", required=True, type=int)
    parser.add_argument("--clearance-mm", type=float, default=20.)
    parser.add_argument("--max-path-mm", type=float, default=4000.)
    parser.add_argument("--physical-ownership", action="store_true")
    parser.add_argument("--probe-cell-mm", type=float, default=1e-7)
    parser.add_argument("--wide-torus", action="store_true")
    parser.add_argument("--mixed-extent", action="store_true")
    parser.add_argument("--native-planar", action="store_true")
    parser.add_argument("--extra-material", action="append", default=[])
    parser.add_argument("--bounded-forward-lookahead", action="store_true")
    args = parser.parse_args()
    run(args.gen, args.gdml, args.source, args.outdir, max_events=args.max_events, arm=args.arm,
        clearance_mm=args.clearance_mm, max_path_mm=args.max_path_mm, physical_ownership=args.physical_ownership,
        probe_cell_mm=args.probe_cell_mm, wide_torus=args.wide_torus, mixed_extent=args.mixed_extent, native_planar=args.native_planar, extra_materials=args.extra_material, bounded_forward_lookahead=args.bounded_forward_lookahead)
