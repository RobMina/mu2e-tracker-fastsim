"""Compare completed independent homogeneous-medium ensembles without treating seeds as strong coupling."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from scripts.transport_capture_audit import digest


def compare(first, second, first_values, second_values):
    for key in ("mode", "material", "stop_ke_MeV", "samples_per_energy", "scatter", "stopping_evaluation"):
        if first.get(key, "entry") != second.get(key, "entry"):
            raise ValueError(f"incompatible ensemble {key}")
    if first["seed"] == second["seed"]:
        raise ValueError("weak distribution comparison requires distinct seeds")
    rows = []
    for a, b in zip(first["energies"], second["energies"]):
        if a["initial_ke_MeV"] != b["initial_ke_MeV"]:
            raise ValueError("energy grids differ")
        row = dict(initial_ke_MeV=a["initial_ke_MeV"], stopped=[a["stopped"], b["stopped"]], observables={})
        use_a = first_values["initial_ke_MeV"] == a["initial_ke_MeV"]
        use_b = second_values["initial_ke_MeV"] == b["initial_ke_MeV"]
        metrics = [("range_mm", "path"), ("time_ns", "dt")]
        if "terminal_angle_rad" in a and "terminal_angle_rad" in b:
            metrics += [("terminal_angle_rad", None), ("transverse_radius_mm", None), ("longitudinal_mm", None)]
        for name, raw in metrics:
            ma, mb = a[name], b[name]
            error = np.hypot(ma["mean_se"], mb["mean_se"])
            delta = mb["mean"] - ma["mean"]
            def values(arrays, mask):
                if raw is not None:
                    return arrays[raw][mask]
                if name == "terminal_angle_rad":
                    return np.arccos(np.clip(arrays["d"][mask, 2], -1., 1.))
                if name == "transverse_radius_mm":
                    return np.linalg.norm(arrays["o"][mask, :2], axis=1)
                return arrays["o"][mask, 2]
            ks = ks_2samp(values(first_values, use_a), values(second_values, use_b))
            row["observables"][name] = dict(means=[ma["mean"], mb["mean"]], delta=float(delta),
                independent_mean_error=float(error), normal_z=float(delta / error) if error > 0 else None,
                relative_delta_percent=100 * delta / ma["mean"] if ma["mean"] != 0 else None, ks_statistic=float(ks.statistic),
                ks_pvalue=float(ks.pvalue))
        rows.append(row)
    return rows


def run(paths, out):
    reports = [json.loads((Path(path) / "complete.json").read_text()) for path in paths]
    hashes = {str(Path(path) / name): digest(Path(path) / name) for path in paths for name in ("complete.json", "final.npz")}
    comparisons = []
    for i, report in enumerate(reports):
        if report["mode"] != "ordered-stored-sampled-v1" or report["max_step_mm"] != 10.:
            continue
        for j, other in enumerate(reports):
            if (i == j or other["mode"] != report["mode"] or other["material"] != report["material"]
                    or other["stop_ke_MeV"] != report["stop_ke_MeV"] or other["scatter"] != report["scatter"]
                    or other.get("stopping_evaluation", "entry") != report.get("stopping_evaluation", "entry")):
                continue
            with np.load(Path(paths[i]) / "final.npz", allow_pickle=False) as first:
                with np.load(Path(paths[j]) / "final.npz", allow_pickle=False) as second:
                    values = compare(report, other, first, second)
            comparisons.append(dict(paths=[paths[i], paths[j]],
                settings=[{k: r[k] for k in ("max_step_mm", "fractional_loss", "seed")} for r in (report, other)],
                energies=values))
    result = dict(scope="independent-history homogeneous weak convergence; no G4 or strong-path equivalence",
        multiple_comparison_warning="descriptive unadjusted z and KS; no post-hoc tolerance or certification",
        reports=[dict(path=path, material=r["material"], mode=r["mode"], max_step_mm=r["max_step_mm"],
            fractional_loss=r["fractional_loss"], stop_ke_MeV=r["stop_ke_MeV"], scatter=r["scatter"],
            energies=r["energies"], stats=r["stats"]) for path, r in zip(paths, reports)],
        comparisons=comparisons, input_sha256=hashes)
    if hashes != {str(Path(path) / name): digest(Path(path) / name) for path in paths for name in ("complete.json", "final.npz")}:
        raise ValueError("uniform study input changed")
    with Path(out).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for comparison in comparisons:
        print(json.dumps(comparison), flush=True)
    for path, report in zip(paths, reports):
        if report["mode"] == "ordered-stored-mean-v1":
            print(json.dumps(dict(path=path, mean_model_errors=[dict(ke=r["initial_ke_MeV"],
                range_relative_percent=100 * r["range_mean_minus_continuous_mm"] / r["continuous_mean_model_reference"]["range_mm"],
                time_relative_percent=100 * r["time_mean_minus_continuous_ns"] / r["continuous_mean_model_reference"]["time_ns"])
                for r in report["energies"]])), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.paths, args.out)
