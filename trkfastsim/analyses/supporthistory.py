"""Bounded seed-table runner for experimental ordered support transport, not tracker production."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials
from .. import geometry, orderedtransport, steps, transportledger
from ..interiorpath import interior_intervals
from .ledgercheck import sha256, source_manifest

SEED_COLUMNS = ["run", "subrun", "event", "sp", "x", "y", "z", "px", "py", "pz", "length_mm"]
SCOPE = ("Experimental support-only straight transport from supplied seed states; no straws, scattering, "
         "upstream world or detector response. Not G4 closure or production acceptance.")


def run(seeds, *, gdml, outdir, tracker_ref, core_ref, mode, seed=1, timing="unknown",
        max_rows=1000, max_step_mm=1., fractional_loss=.05, stop_ke_MeV=1., max_steps=10000):
    """Read explicit Mu2e-frame seeds, reject unsupported/overlapping paths, persist immutable new output."""
    if (not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0 or seed < 0):
        raise ValueError("positive row bound and nonnegative seed required")
    seeds, gdml, output = Path(seeds), Path(gdml), Path(outdir)
    if output.exists():
        raise FileExistsError(output)
    source = pd.read_csv(seeds, nrows=max_rows + 1)
    if len(source) > max_rows:
        raise ValueError("seed table exceeds explicit row bound; no silent truncation")
    if set(source.columns) != set(SEED_COLUMNS):
        raise ValueError(f"seed table must contain exactly {SEED_COLUMNS}")
    key = source[["run", "subrun", "event", "sp"]].to_numpy()
    o, pv = source[["x", "y", "z"]].to_numpy(float), source[["px", "py", "pz"]].to_numpy(float)
    p = np.linalg.norm(pv, axis=1)
    if not np.isfinite(pv).all() or (p <= 0).any():
        raise ValueError("seed momentum vectors must be finite and nonzero")
    direction = pv / p[:, None]
    kinetic = p * (p / (np.hypot(p, eloss.MMU) + eloss.MMU))
    lengths = source.length_mm.to_numpy(float)
    geom = geometry.from_gdml(str(gdml))
    interior = steps.TrackerInterior(str(gdml), geom)
    if interior.calo_skipped:
        raise ValueError(f"calorimeter loader skipped solids: {interior.calo_skipped}")
    spans = interior_intervals(interior.shells, o, direction, lengths)
    names = sorted(set(spans.material) - {"vacuum"})
    media = [transportledger.medium()] + [transportledger.medium(materials.get(name), cut_MeV=1e9,
                                                                  loss_basis="unrestricted") for name in names]
    metadata = transportledger.provenance(geometry_sha256=sha256(gdml), tracker_commit=tracker_ref,
        core_commit=core_ref, inputs=[{"name": str(seeds.resolve()), "sha256": sha256(seeds)}], media=media,
        frame="Mu2e world Cartesian", settings=dict(scope=SCOPE, seed=seed, seed_rows=len(source),
        source_events=len(source[["run", "subrun", "event"]].drop_duplicates()),
        seed_domain="supplied seeds only; no generated-event acceptance denominator",
        sources_sha256=source_manifest(), geometry_representation="stored tube and signed-convex support model",
        calorimeter_skipped=interior.calo_skipped))
    result = orderedtransport.transport_intervals(np.random.default_rng(seed), key, o, direction, kinetic,
        spans, metadata, mode=mode, timing=timing, max_step_mm=max_step_mm, fractional_loss=fractional_loss,
        stop_ke_MeV=stop_ke_MeV, max_steps=max_steps)
    history = result["history"]
    summary = dict(scope=SCOPE, seed_rows=len(source), seed_events=metadata["settings"]["source_events"],
        physical_rows=len(history), known_partition_rows=int(history.accounting()["partition_complete"].sum()),
        states={str(k): int(v) for k, v in zip(*np.unique(result["status"], return_counts=True))},
        mode=mode, timing=timing, complete=True)
    final = source[["run", "subrun", "event", "sp"]].copy()
    final["kinetic_MeV"], final["path_mm"], final["status"] = result["final_ke_MeV"], result["traveled_mm"], result["status"]
    final[["x", "y", "z"]] = result["position_mm"]
    output.mkdir(parents=True, exist_ok=False)
    history.save(output / "history.npz")
    spans.to_csv(output / "requested_intervals.csv", index=False)
    final.to_csv(output / "final.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", help="bounded CSV in Mu2e frame, not generated ROOT or an acceptance sample")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tracker-ref", required=True)
    parser.add_argument("--core-ref", required=True)
    parser.add_argument("--mode", choices=list(orderedtransport.MODES), required=True)
    parser.add_argument("--timing", choices=["unknown", "linear-in-path"], default="unknown")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--max-step-mm", type=float, default=1.)
    parser.add_argument("--fractional-loss", type=float, default=.05)
    parser.add_argument("--stop-ke-MeV", dest="stop_ke_MeV", type=float, default=1.)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = vars(parser.parse_args(argv))
    print(json.dumps(run(**args), indent=2))


if __name__ == "__main__":
    main()
