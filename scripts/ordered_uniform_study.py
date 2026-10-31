"""Actual ordered-interior homogeneous-medium convergence, conditional on the existing loss model."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad

from mu2efastsim import eloss, materials
from mu2efastsim.world import CoaxialShells
from scripts.transport_capture_audit import digest
from trkfastsim import geometry, ledger, orderedinterior, steps

C_MM_NS = 299.792458


def dummy_geometry():
    zeros = np.zeros(1, np.int64)
    return geometry.StrawGeometry(plane=zeros, panel=zeros, straw=zeros,
        mid=np.array([[1e8, 0., 0.]]), dir=np.array([[1., 0., 0.]]), halflen=np.array([1.]), radius=2.5,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None],
        plane_t=np.zeros((1, 3)), wall_rmax=2.515, metal_t=.0002, wire_r=0.)


def reference(material, kinetic, cutoff):
    def inverse_loss(ke):
        return 1. / float(eloss.dedx_restricted(material, ledger.momentum(np.array([ke])), 1e9)[0])
    length, error = quad(inverse_loss, cutoff, kinetic, epsabs=1e-9, epsrel=1e-9, limit=200)
    clock, clock_error = quad(lambda ke: inverse_loss(ke) / (C_MM_NS * ledger.beta(ledger.momentum(ke))),
                              cutoff, kinetic, epsabs=1e-10, epsrel=1e-9, limit=200)
    return {"range_mm": length, "time_ns": clock, "quadrature_error_mm": error,
            "quadrature_error_ns": clock_error}


def moments(value):
    value = np.asarray(value, float)
    return {"n": len(value), "mean": float(value.mean()),
            "mean_se": float(value.std(ddof=1) / np.sqrt(len(value))) if len(value) > 1 else None,
            "std": float(value.std(ddof=1)) if len(value) > 1 else None,
            "quantiles": np.quantile(value, [0, .1, .5, .9, 1]).tolist()}


def run(material_name, outdir, *, mode, max_step_mm, fractional_loss=.05, cutoff=.5,
        samples=512, seed=31, scatter=False, stopping_evaluation="entry"):
    if not 2 <= samples <= 4096:
        raise ValueError("study requires 2..4096 independent histories per energy")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    material = materials.get(material_name)
    energies = np.array([5., 20., 100.])
    kinetic = np.repeat(energies, samples)
    geom = dummy_geometry()
    interior = steps.TrackerInterior.__new__(steps.TrackerInterior)
    interior.shells = CoaxialShells([(material.name, np.eye(3), np.zeros(3), 0., 1e6, 1000., 0., 2 * np.pi)],
                                   geometry.DETECTOR_ORIGIN_EXTRACTED)
    n = len(kinetic)
    origin = np.zeros((n, 3))
    direction = np.tile([0., 0., 1.], (n, 1))
    original = steps._kink_core
    if not scatter:
        steps._kink_core = lambda rng, v, a, b: (np.zeros((len(v), 2)), np.zeros((len(v), 2)), np.zeros(len(v)))
    started = time.perf_counter()
    try:
        interior.transport(np.random.default_rng(seed), origin, direction, ledger.momentum(kinetic), geom,
            tail=False, navigation=mode, ordered_options=dict(max_path_mm=500., max_step_mm=max_step_mm,
                fractional_loss=fractional_loss, stop_ke_MeV=cutoff, stopping_evaluation=stopping_evaluation,
                progress=lambda v: print(json.dumps(dict(progress=v)), flush=True)))
    finally:
        steps._kink_core = original
    final = interior.last_final
    balance = kinetic - final["ke"] - final["lost_support"] - final["lost_thin"]
    if not np.allclose(balance, 0., rtol=0, atol=1e-9):
        raise ValueError("homogeneous primary balance failed")
    np.savez_compressed(outdir / "final.npz", initial_ke_MeV=kinetic, **final)
    rows = []
    for energy in energies:
        use = kinetic == energy
        stopped = final["stopped"][use]
        ref = reference(material, energy, cutoff)
        ranges = moments(final["path"][use])
        clock = moments(final["dt"][use])
        rows.append(dict(initial_ke_MeV=energy, stopped=int(stopped.sum()), not_stopped=int((~stopped).sum()),
            range_mm=ranges, time_ns=clock, continuous_mean_model_reference=ref,
            range_mean_minus_continuous_mm=ranges["mean"] - ref["range_mm"],
            time_mean_minus_continuous_ns=clock["mean"] - ref["time_ns"],
            final_ke_MeV=moments(final["ke"][use]),
            terminal_angle_rad=moments(np.arccos(np.clip(final["d"][use, 2], -1., 1.))),
            transverse_radius_mm=moments(np.linalg.norm(final["o"][use, :2], axis=1)),
            longitudinal_mm=moments(final["o"][use, 2])))
    result = dict(scope="homogeneous stored-cylinder actual interior; existing loss-model convergence, not G4 validation",
        material=material_name, mode=mode, max_step_mm=max_step_mm, fractional_loss=fractional_loss,
        stop_ke_MeV=cutoff, samples_per_energy=samples, seed=seed, scatter=scatter,
        stopping_evaluation=stopping_evaluation,
        independent_history_warning="setting comparisons use independent seeds; weak distribution tests, not same-history strong errors",
        elapsed_s=time.perf_counter() - started, stats=interior.last_stats, energies=rows,
        primary_balance_max_abs_MeV=float(np.max(np.abs(balance))), admitted_for_production=False,
        source_sha256={str(path): digest(path) for path in [Path(__file__), Path(orderedinterior.__file__), Path(eloss.__file__)]})
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="Al6061")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mode", choices=orderedinterior.MODES, default=orderedinterior.SAMPLED_MODE)
    parser.add_argument("--max-step-mm", type=float, default=1.)
    parser.add_argument("--fractional-loss", type=float, default=.05)
    parser.add_argument("--cutoff", type=float, default=.5)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--stopping-evaluation", choices=("entry", "midpoint"), default="entry")
    args = parser.parse_args()
    run(args.material, args.outdir, mode=args.mode, max_step_mm=args.max_step_mm,
        fractional_loss=args.fractional_loss, cutoff=args.cutoff, samples=args.samples, seed=args.seed, scatter=args.scatter,
        stopping_evaluation=args.stopping_evaluation)
