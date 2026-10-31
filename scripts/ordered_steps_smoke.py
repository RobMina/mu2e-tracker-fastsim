"""Bounded full generator/world/interior/writer/surface smoke; no G4 production."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from mu2efastsim import tables as core_tables
from scripts.transport_capture_audit import digest
from trkfastsim import geometry, orderedinterior, steps, surfaces, tables


def run(gen_path, gdml, outdir, *, max_events=100, signed_endpoint_budget_mm=None, navigation=orderedinterior.MODE,
        stopping_evaluation="entry", max_step_mm=10.):
    if not 0 < max_events <= 10000:
        raise ValueError("bounded smoke requires 1..10000 generated events")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    source = [Path(gen_path), Path(gdml)]
    before = {str(p): digest(p) for p in source}
    gen = tables.read(gen_path, "gen")
    selected = gen[tables.KEY].drop_duplicates().iloc[:max_events]
    gen = gen.merge(selected, on=tables.KEY, how="inner", validate="many_to_one")
    geom = geometry.from_gdml(gdml)
    interior = steps.TrackerInterior(gdml, geom)
    world = steps.load_world(gdml, geom)
    result = dict(generated_events=len(selected), particles_generated=len(gen), navigation=navigation,
        scope="bounded full make_steps smoke; not a rate/physics certification", input_sha256=before,
        stopping_evaluation=stopping_evaluation, max_step_mm=max_step_mm,
        admitted_for_production=False)
    started = time.perf_counter()
    try:
        rows, particles = steps.make_steps(gen, geom, np.random.default_rng(19), stage="C", world=world,
            interior=interior, sliced_msc=True, navigation=navigation, history=True, msc_tail=False,
            ordered_options={"progress": lambda v: print(json.dumps(dict(progress=v)), flush=True),
                             "signed_endpoint_budget_mm": signed_endpoint_budget_mm,
                             "stopping_evaluation": stopping_evaluation, "max_step_mm": max_step_mm})
        surf = surfaces.surface_steps(rows, steps.make_steps.path_vertices)
    except (ValueError, RuntimeError) as error:
        details = getattr(error, "details", None)
        if details is not None and "lane" in details and getattr(steps.make_steps, "last_plane", None) is not None:
            plane = steps.make_steps.last_plane
            active = np.flatnonzero(plane["alive"])
            local = details.get("interior_lane", details["lane"])
            if local < len(active):
                keys = steps.make_steps.last_plane["key"][active[local]]
                details = dict(details, event_identity=[int(v) for v in keys],
                               mapping_scope="stable interior lane mapped through initial alive selection")
        result.update(status="refused", reason=str(error), refusal_details=details)
    else:
        core_tables.write(str(outdir / "ordered_steps.root"),
            dict(sgsteps=rows, particles=particles, gen=gen, surfsteps=surf), as_ttree=True)
        steps.make_steps.segment_ledger.to_csv(outdir / "segments.csv", index=False)
        steps.make_steps.muon_ledger.to_csv(outdir / "muons.csv", index=False)
        steps.make_steps.path_vertices.to_csv(outdir / "vertices.csv", index=False)
        active = steps.make_steps.muon_ledger.loc[steps.make_steps.muon_ledger.path_interior_mm.notna()]
        if len(active) != len(interior.last_final["path"]):
            raise ValueError("ordered slice identity/terminal coverage mismatch")
        np.savez_compressed(outdir / "traversed.npz", muon_keys=active[tables.KEY + ["sp"]].to_numpy(np.int64),
            t_start_ns=active.t_interior_start_ns.to_numpy(float),
            **{f"final_{k}": v for k, v in interior.last_final.items()},
            **{f"slice_{k}": v for k, v in interior.last_slices.items()})
        interior.last_endpoint_evidence.to_csv(outdir / "endpoint_shifts.csv", index=False)
        interior.last_leading_source_evidence.to_csv(outdir / "leading_signed_sources.csv", index=False)
        result.update(status="bounded_full_chain_complete", straw_steps=len(rows), surface_steps=len(surf),
            primary_muons=len(particles), primary_pass_events=int(tables.trk_filter(rows).pass_trk.sum()),
            stats=steps.make_steps.interior_stats, timing=steps.make_steps.timing)
    result["elapsed_s"] = time.perf_counter() - started
    result["source_sha256"] = {str(p): digest(p) for p in [Path(__file__), *sorted(Path(steps.__file__).parent.glob("*.py"))]}
    if before != {str(p): digest(p) for p in source}:
        raise ValueError("input changed during full-chain smoke")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gen")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--signed-endpoint-budget-mm", type=float, default=None)
    parser.add_argument("--navigation", choices=orderedinterior.MODES, default=orderedinterior.MODE)
    parser.add_argument("--stopping-evaluation", choices=("entry", "midpoint"), default="entry")
    parser.add_argument("--max-step-mm", type=float, default=10.)
    args = parser.parse_args()
    run(args.gen, args.gdml, args.outdir, max_events=args.max_events,
        signed_endpoint_budget_mm=args.signed_endpoint_budget_mm, navigation=args.navigation,
        stopping_evaluation=args.stopping_evaluation, max_step_mm=args.max_step_mm)
