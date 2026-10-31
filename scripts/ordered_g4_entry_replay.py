"""Truth-conditioned interior replay from saved G4 first-straw states, not inclusive validation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest
from scripts.ordered_g4_state_compare import summary
from trkfastsim import geometry, ledger, orderedinterior, steps, tables

KEY = tables.KEY + ["sp"]


def entrances(sgs, particles, lanes, selection="identity"):
    primary = particles[(particles.ismcprim == 1) & (particles.pdg.abs() == 13)][tables.KEY + ["key", "pdg"]]
    primary = primary.rename(columns={"key": "sp"})
    if primary.duplicated(KEY).any():
        raise ValueError("duplicate G4 primary identity")
    selected = sgs.merge(primary, on=KEY, validate="many_to_one").sort_values(KEY + ["t"], kind="stable")
    first = selected.drop_duplicates(KEY)
    if selection == "lowest-momentum":
        first = first.assign(_p=np.linalg.norm(first[["px", "py", "pz"]].to_numpy(float), axis=1))
        first = first.sort_values(["_p"] + KEY, kind="stable").drop(columns="_p")
    elif selection != "identity":
        raise ValueError("unknown truth diagnostic selection")
    first = first.iloc[:lanes].copy()
    return first, selected.merge(first[KEY], on=KEY, validate="many_to_one")


def observables(entries, rows, first, reference):
    f = pd.DataFrame({name: first[name].to_numpy()[entries["it"]] for name in KEY})
    f["sid"] = rows.sid.to_numpy()
    f["t"] = first.t.to_numpy()[entries["it"]] + entries["dt"]
    f["path_mm"] = entries["path"]
    initial_p = np.linalg.norm(first[["px", "py", "pz"]].to_numpy(float), axis=1)
    f["ke_MeV"] = ledger.kinetic(initial_p)[entries["it"]] - entries["lost"]
    ambiguous = reference.duplicated(KEY + ["sid"], keep=False)
    pairs = f.merge(reference.loc[~ambiguous], on=KEY + ["sid"], suffixes=("_fast", "_g4"), validate="one_to_one")
    pairs["delta_time_ns"] = pairs.t_fast - pairs.t_g4
    pg = np.linalg.norm(pairs[["px", "py", "pz"]].to_numpy(float), axis=1)
    pairs["delta_ke_MeV"] = pairs.ke_MeV - ledger.kinetic(pg)
    later = pairs.path_mm > 1e-3
    report = dict(shared_straws=len(pairs), ambiguous_g4_rows_excluded=int(ambiguous.sum()),
        shared_time_ns=summary(pairs.delta_time_ns, pairs),
        shared_ke_MeV=summary(pairs.delta_ke_MeV, pairs),
        later_shared_straws=int(later.sum()), later_time_ns=summary(pairs.loc[later, "delta_time_ns"], pairs.loc[later]),
        later_ke_MeV=summary(pairs.loc[later, "delta_ke_MeV"], pairs.loc[later]),
        matching="full G4 primary-particle identity plus straw; repeated G4 crossings excluded")
    return report, pairs


def run(g4, gdml, outdir, *, lanes=8, mode=orderedinterior.SAMPLED_MODE, length_mm=1500., seed=271,
        max_step_mm=10., scatter=True, endpoint_budget=1e-10, selection="identity", stopping_evaluation="entry"):
    if not 1 <= lanes <= 64:
        raise ValueError("bounded entry replay requires 1..64 primaries")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    paths = [Path(g4), Path(gdml)]
    hashes = {str(p): digest(p) for p in paths}
    first, reference = entrances(tables.read(g4, "sgsteps"), tables.read(g4, "particles"), lanes, selection)
    if first.empty:
        raise ValueError("reference has no primary-muon straw entrances")
    origin = geometry.to_mu2e(first[["x0", "y0", "z0"]].to_numpy(float))
    pvec = first[["px", "py", "pz"]].to_numpy(float)
    momentum = np.linalg.norm(pvec, axis=1)
    direction = pvec / momentum[:, None]
    geom = geometry.from_gdml(gdml)
    interior = steps.TrackerInterior(gdml, geom)
    result = dict(scope="truth-conditioned G4 first-straw initial state; not inclusive rate or same-random-history validation",
        mode=mode, lanes=len(first), requested_length_mm=length_mm, seed=seed, scatter=scatter,
        max_step_mm=max_step_mm, signed_endpoint_budget_mm=endpoint_budget, input_sha256=hashes,
        selection=selection, stopping_evaluation=stopping_evaluation,
        identities=first[KEY].to_numpy().tolist(), admitted_for_production=False)
    first.to_csv(outdir / "initial_g4.csv", index=False)
    reference.to_csv(outdir / "reference_steps.csv", index=False)
    original = steps._kink_core
    if not scatter:
        steps._kink_core = lambda rng, v, a, b: (np.zeros((len(v), 2)), np.zeros((len(v), 2)), np.zeros(len(v)))
    started = time.perf_counter()
    try:
        entries = interior.transport(np.random.default_rng(seed), origin, direction, momentum, geom,
            navigation=mode, tail=False, ordered_options=dict(max_path_mm=length_mm, max_step_mm=max_step_mm,
                signed_endpoint_budget_mm=endpoint_budget, stopping_evaluation=stopping_evaluation,
                progress=lambda v: print(json.dumps(dict(progress=v)), flush=True)))
    except (ValueError, RuntimeError) as error:
        result.update(status="refused", reason=str(error), details=getattr(error, "details", None))
    else:
        labels = pd.DataFrame(dict(sid=geom.sid[entries["js"]]))
        # The entry arrays preserve primary identity and accumulated state, independent of output row order.
        metrics, pairs = observables(entries, labels, first, reference)
        pairs.to_csv(outdir / "paired_straws.csv", index=False)
        np.savez_compressed(outdir / "traversed.npz", **{f"final_{k}": v for k, v in interior.last_final.items()},
            **{f"slice_{k}": v for k, v in interior.last_slices.items()}, **{f"entry_{k}": v for k, v in entries.items()})
        interior.last_endpoint_evidence.to_csv(outdir / "endpoint_shifts.csv", index=False)
        interior.last_leading_source_evidence.to_csv(outdir / "leading_sources.csv", index=False)
        result.update(status="bounded_g4_entry_complete", stats=interior.last_stats, metrics=metrics,
            final_ke_MeV=interior.last_final["ke"].tolist(), stopped=interior.last_final["stopped"].tolist(),
            path_mm=interior.last_final["path"].tolist())
    finally:
        steps._kink_core = original
    result["elapsed_s"] = time.perf_counter() - started
    result["source_sha256"] = {str(p): digest(p) for p in [Path(__file__), Path(orderedinterior.__file__)]}
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("reference input changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("g4")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--mode", choices=orderedinterior.MODES, default=orderedinterior.SAMPLED_MODE)
    parser.add_argument("--length-mm", type=float, default=1500.)
    parser.add_argument("--max-step-mm", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--no-scatter", action="store_true")
    parser.add_argument("--selection", choices=("identity", "lowest-momentum"), default="identity")
    parser.add_argument("--stopping-evaluation", choices=("entry", "midpoint"), default="entry")
    args = parser.parse_args()
    run(args.g4, args.gdml, args.outdir, lanes=args.lanes, mode=args.mode, length_mm=args.length_mm,
        max_step_mm=args.max_step_mm, seed=args.seed, scatter=not args.no_scatter,
        selection=args.selection, stopping_evaluation=args.stopping_evaluation)
