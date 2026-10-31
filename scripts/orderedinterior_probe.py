"""Replay saved entrance states through the actual opt-in interior transport API."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.transport_capture_audit import digest, flatten, load_capture
from trkfastsim import geometry, ledger, orderedinterior, steps


def run(capture_dir, gdml, outdir, *, lanes=16, length_mm=2500., scatter=True, max_step_mm=10.,
        boundary_resolution_mm=1e-9, identity=None, navigation=orderedinterior.MODE):
    if not 0 < lanes <= 64:
        raise ValueError("bounded replay requires 1..64 retained lanes")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    sources = [Path(capture_dir) / "capture.npz", Path(capture_dir) / "complete.json", Path(gdml)]
    before = {str(p): digest(p) for p in sources}
    _, phases = load_capture(capture_dir)
    initial = flatten(phases["interior"][1], "initial")
    if identity is not None:
        wanted = np.asarray(identity, dtype=np.int64)
        if wanted.shape != (4,):
            raise ValueError("identity needs run/subrun/event/sp")
        matches = np.flatnonzero(np.all(initial["identity"] == wanted, axis=1))
        if len(matches) != 1:
            raise ValueError("requested identity must occur exactly once in saved capture")
        initial = {k: v[matches] for k, v in initial.items()}
    n = min(lanes, len(initial["identity"]))
    geom = geometry.from_gdml(gdml)
    interior = steps.TrackerInterior(gdml, geom)
    result = dict(mode=navigation, identities=initial["identity"][:n].tolist(),
        lanes=n, input_sha256=before, requested_length_mm=length_mm, scatter=scatter,
        max_step_mm=max_step_mm, boundary_resolution_mm=boundary_resolution_mm,
        scope="saved entrance replay; no generation or primary rate denominator; incomplete stored geometry",
        skipped_calorimeter_solids=[dict(solid=str(k), count=int(v)) for k, v in interior.calo_skipped.items()],
        admitted_for_production=False)
    started = time.perf_counter()
    if not scatter:
        original = steps._kink_core
        steps._kink_core = lambda rng, v, a, b: (np.zeros((len(v), 2)), np.zeros((len(v), 2)), np.zeros(len(v)))
    def progress(values):
        print(json.dumps(dict(progress=values)), flush=True)
    try:
        rows = interior.transport(np.random.default_rng(19), initial["position_mm"][:n],
            initial["direction"][:n], ledger.momentum(initial["model_ke_MeV"][:n]), geom,
            navigation=navigation, tail=False, ordered_options=dict(max_path_mm=length_mm,
                max_step_mm=max_step_mm, boundary_resolution_mm=boundary_resolution_mm, progress=progress))
    except (ValueError, RuntimeError) as error:
        result.update(status="refused", reason=str(error), refusal_details=getattr(error, "details", None))
    else:
        final = interior.last_final
        residual = initial["model_ke_MeV"][:n] - final["ke"] - final["lost_support"] - final["lost_thin"]
        np.savez_compressed(outdir / "traversed.npz", **{f"slice_{k}": v for k, v in interior.last_slices.items()},
                            **{f"entry_{k}": v for k, v in rows.items()}, **{f"final_{k}": v for k, v in final.items()})
        result.update(status="bounded_transport_complete", entries=len(rows["it"]), stats=interior.last_stats,
            final_ke_MeV=final["ke"].tolist(), path_mm=final["path"].tolist(),
            stopped=final["stopped"].tolist(), boundary_relabel_mm=final["boundary_relabel_mm"].tolist(),
            primary_balance_max_abs_MeV=float(np.max(np.abs(residual))))
    finally:
        if not scatter:
            steps._kink_core = original
    result["elapsed_s"] = time.perf_counter() - started
    result["source_sha256"] = {str(p): digest(p) for p in [Path(__file__), *sorted(Path(steps.__file__).parent.glob("*.py"))]}
    if before != {str(p): digest(p) for p in sources}:
        raise ValueError("input changed during replay")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--navigation", choices=orderedinterior.MODES, default=orderedinterior.MODE)
    parser.add_argument("--identity", nargs=4, type=int, metavar=("RUN", "SUBRUN", "EVENT", "SP"))
    parser.add_argument("--length-mm", type=float, default=2500.)
    parser.add_argument("--no-scatter", action="store_true")
    parser.add_argument("--max-step-mm", type=float, default=10.)
    parser.add_argument("--boundary-resolution-mm", type=float, default=1e-9)
    args = parser.parse_args()
    run(args.capture_dir, args.gdml, args.outdir, lanes=args.lanes, length_mm=args.length_mm,
        scatter=not args.no_scatter, max_step_mm=args.max_step_mm, boundary_resolution_mm=args.boundary_resolution_mm,
        identity=args.identity, navigation=args.navigation)
