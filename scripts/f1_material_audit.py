"""Isolate material scattering policy on immutable first-iteration traces; no refit."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import uproot

from f1_native_decision_audit import load_evidence, read_tree, keys, digest, require
from f1_trace_compare import compare_first, _col, _dense, _metric, _match
from trkfastsim.reco import fittrace, nativecompare, nativematerial as N

RTOL, ATOL = 1e-5, 1e-7
RECONSTRUCTION_RTOL = 1e-10
D_RECONSTRUCTION_ATOL = 1e-12
NATIVE_SOURCES = ("KinKal/MatEnv/DetMaterial.cc", "KinKal/Detector/ElementXing.hh",
                  "KinKal/Trajectory/KinematicLine.cc", "Offline/Mu2eKinKal/inc/KKStrawXing.hh")
# Record unused source differences; no input-loader execution is needed for saved material operands.
REQUIRED_SOURCES = ("trkfastsim/reco/kalman_jax.py", "trkfastsim/reco/linefit.py",
                    "trkfastsim/reco/kkmaterial.py", "trkfastsim/reco/nativecompare.py",
                    "trkfastsim/reco/fittrace.py")
LIMITS = [
    "Only saved actual iteration zero, after the full native gate and first-iteration identity checks.",
    "Only the material participation intersection is compared; excluded cached native changes remain counted.",
    "V1 saved and explicit_p109_gas variants are unchanged: only gas fraction changes; saved D stays fixed.",
    "The third variant adds the exact linear-in-distance offset*speed increment to saved D, avoiding absolute-time subtraction.",
    "Absolute-time D reconstruction must pass the established 1e-5/1e-7 gate; the extra 1e-10/1e-12 diagnostic can fail and is retained.",
    "All variants retain saved references, paths, momentum, dm, dmv and participation; no native-reference/time substitution.",
    "Mean change is recomputed with shifted D: its momentum-column t0 term is time dependent and reported explicitly.",
    "Native KKStrawXing time is CA particle TOCA plus constructor-frozen offset; first-iteration captured offset matches seed speed only, not a later-iteration claim.",
    "Temperature scale (1+T[0])^2 comes from the saved schedule; no fitted scale or variance floor.",
    "Native Q[theta,theta] and Q[mom,mom] isolate projected angular and longitudinal momentum variances.",
    "Covariance conversion remains first-order with unchanged strict rtol=1e-5, atol=1e-7 gate; physical-time D is diagnostic, not native arithmetic closure.",
    "No refit, decision/acceptance causality, later iterations, full closure or fitter/default change.",
]


def metric(actual, native):
    """Existing gate plus ALL failures and true, unfloored relative residuals.

    Zero reference/nonzero residual is counted explicitly, not hidden behind
    ATOL. max_relative_residual from the legacy metric is separately labeled as
    floored; the unfloored diagnostic never changes the native comparison gate.
    """
    a, b = np.asarray(actual, float), np.asarray(native, float)
    require(a.shape == b.shape and a.ndim >= 1, "metric shape mismatch")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite metric operand")
    result = _metric(a, b)
    delta = np.abs(a-b)
    require(np.isfinite(delta).all(), "nonfinite residual")
    bad = ~np.isclose(a, b, rtol=RTOL, atol=ATOL)
    flat = np.flatnonzero(bad)
    axes = tuple(range(1, a.ndim))
    effect_bad = np.any(bad, axis=axes) if axes else bad
    per_effect = np.max(delta, axis=axes) if axes else delta
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rel = np.divide(delta, np.abs(b), out=np.zeros_like(delta), where=b != 0)
    valid_relative = (b != 0) & np.isfinite(rel)
    relative_only_bad = ~np.isclose(a, b, rtol=RTOL, atol=0.)
    result.update(
        relative_only_diagnostic={"rtol": RTOL, "atol": 0., "passed": bool(not relative_only_bad.any()),
                                  "mismatches": int(relative_only_bad.sum()),
                                  "mismatch_indices": np.argwhere(relative_only_bad).tolist()},
        max_relative_residual_atol_floored=result.pop("max_relative_residual"),
        max_relative_residual_unfloored=float(rel[valid_relative].max()) if valid_relative.any() else None,
        relative_zero_reference_nonzero_residual=int(((b == 0) & (delta != 0)).sum()),
        relative_overflow_components=int(((b != 0) & ~np.isfinite(rel)).sum()),
        n_effects=int(a.shape[0]), n_failing_effects=int(effect_bad.sum()),
        failing_effect_positions=np.flatnonzero(effect_bad).tolist(),
        mismatch_indices=np.argwhere(bad).tolist(), actual_mismatch_values=a.ravel()[flat].tolist(),
        native_mismatch_values=b.ravel()[flat].tolist(), per_effect_max_absolute_residual=per_effect.tolist())
    return result


def initial_scale(manifest):
    temperature = np.asarray(manifest["schedule"]["T"], float)
    require(temperature.ndim == 1 and len(temperature) > 0 and np.isfinite(temperature).all()
            and np.all(temperature >= 0), "invalid saved temperature schedule")
    scale = float((1+temperature[0])**2)
    require(np.isfinite(scale), "invalid saved variance scale")
    return scale


def substitutions(x, D, saved_sv, explicit_sv, dm, dmv, native_mean, native_covariance, *, shifted_D=None):
    """Preserve v1 variants; optionally change only D for a third explicit-gas variant."""
    variants = {}
    operands = [("saved", saved_sv, D), ("explicit_p109_gas", explicit_sv, D)]
    if shifted_D is not None:
        operands.append(("explicit_p109_gas_physical_time_D", explicit_sv, shifted_D))
    for name, sv, local_D in operands:
        mean, covariance = nativecompare.transform_material(x, local_D, sv, dm, dmv)
        variants[name] = {"change_parameters": metric(mean, native_mean),
                          "change_covariance": metric(covariance, native_covariance),
                          "projected_angular_scalar": metric(sv, native_covariance[:, 3, 3]),
                          "momentum_variance_scalar": metric(dmv, native_covariance[:, 5, 5]),
                          "transformed_angular_diagonal_vs_input": metric(covariance[:, 3, 3], sv),
                          "transformed_momentum_diagonal_vs_input": metric(covariance[:, 5, 5], dmv)}
    if shifted_D is not None:
        baseline_mean, _ = nativecompare.transform_material(x, D, explicit_sv, dm, dmv)
        shifted_mean, _ = nativecompare.transform_material(x, shifted_D, explicit_sv, dm, dmv)
        variants["explicit_p109_gas_physical_time_D"]["mean_shift_from_saved_D"] = {
            "native_parameter_delta": (shifted_mean-baseline_mean).tolist(),
            "local_t0_delta": ((shifted_D[:, 4, 2]-D[:, 4, 2])*dm).tolist(),
            "native_t0_delta": (shifted_mean[:, 4]-baseline_mean[:, 4]).tolist()}
    return variants


def kink_time_diagnostic(x, D, physical_time, crossing_offset):
    """Check reconstructed time at the established gate; add the affine offset to saved D."""
    time, offset = np.asarray(physical_time, float), np.asarray(crossing_offset, float)
    require(time.shape == offset.shape and np.isfinite(offset).all(), "invalid crossing offset")
    ca_time = time-offset
    ca_s, ca_D = N.kink_at_time(x, ca_time)
    shifted_s, direct_D = N.kink_at_time(x, time)
    shifted_D, delta_s = N.offset_kink_D(x, D, offset)
    reconstruction = metric(ca_D, D)
    tight = np.isclose(ca_D, D, rtol=RECONSTRUCTION_RTOL, atol=D_RECONSTRUCTION_ATOL)
    reconstruction["strict_D_diagnostic"] = {
        "rtol": RECONSTRUCTION_RTOL, "atol": D_RECONSTRUCTION_ATOL,
        "passed": bool(tight.all()), "mismatches": int((~tight).sum()),
        "mismatch_indices": np.argwhere(~tight).tolist()}
    require(reconstruction["passed"], "CA-time D does not reconstruct saved D: " + json.dumps(reconstruction))
    direct_metric = metric(direct_D, shifted_D)
    require(direct_metric["passed"], "direct-time and affine-offset D disagree")
    return shifted_D, {"saved_D_reconstruction": reconstruction,
        "direct_time_vs_affine_D": direct_metric, "affine_delta_s": delta_s.tolist(),
        "substitution": "saved D plus offset*speed times exact linear-in-distance D slope; not reconstructed absolute-time D",
        "physical_time": time.tolist(), "crossing_offset": offset.tolist(),
        "ca_evaluation_time": ca_time.tolist(), "ca_evaluation_s": ca_s.tolist(),
        "shifted_evaluation_s": shifted_s.tolist(), "delta_s": (shifted_s-ca_s).tolist(),
        "D_delta": (shifted_D-D).tolist(),
        "time_source": "saved physical_time in natural effect-slot order; not ordering_time or native captured time"}


def verify_manifest(directory, metadata):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    require(manifest.get("schema_version") == 1, "unsupported trace schema")
    require(manifest.get("completed") is True and manifest.get("trace_passivity_exact") is True,
            "requires complete passive saved trace")
    require(not manifest.get("environment"), "requires default saved F1 environment")
    require(manifest["check_sha256"] == metadata["check_sha256"], "native gate differs")
    require(manifest["inputs"] == metadata["inputs"], "saved native evidence identities/hashes differ")
    require(manifest["trace_sha256"] == digest(directory / "trace.npz"), "trace hash changed")
    require(manifest["time_model"] == "stock" and manifest["reference_model"] == "previous"
            and manifest["straw_geometry"] == "approximate" and manifest["crossing_time"] == "current",
            "unsupported saved fit policy")
    require(manifest.get("native_time") is True, "requires certified native constraint time")
    for source in (manifest["conditions"], manifest["alignment"], *manifest["ann"].values()):
        require(digest(source["path"]) == source["sha256"], "trace configuration hash changed: " + source["path"])
    require(bool(manifest["ann"]), "missing saved ANN provenance")
    root = Path(__file__).resolve().parents[1]
    provenance = {}
    for name, saved_hash in manifest["source"].items():
        source = root / name
        require(not Path(name).is_absolute() and source.resolve().is_relative_to(root), "invalid source path")
        current = digest(source) if source.is_file() else None
        provenance[name] = {"saved_sha256": saved_hash, "current_sha256": current,
                            "source_equal": saved_hash == current,
                            "equality_required": name in REQUIRED_SOURCES}
    for name in REQUIRED_SOURCES:
        require(name in provenance and provenance[name]["source_equal"], "used conversion/material source changed: " + name)
    return manifest, provenance


def audit_one(directory, metadata, iterations, pieces, effects):
    manifest_hash = digest(directory / "manifest.json")
    manifest, provenance = verify_manifest(directory, metadata)
    with np.load(directory / "trace.npz", allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    key, hids, xids = data["key"], data["hit_ids"], data["crossing_ids"]
    require(key.shape == (4,) and key.dtype.kind in "iu" and np.all(key >= 0)
            and np.array_equal(key, manifest["key"]), "trace full key differs/invalid")
    require(len(data["stage_index"]) == manifest["n_iterations"] > 0
            and data["stage_index"][0] == 0, "missing first saved stage/count differs")
    require(data["valid"].shape == data["stage_index"].shape and data["valid"].all(), "invalid saved executed records")
    nh, nx = len(hids), len(xids)
    t = {name: data[name][0] for name in fittrace.shapes(nh, nx)}
    select = lambda table: np.all(keys(table, unique=False) == key, axis=1) & (_col(table, "iteration_ordinal") == 0)
    ni = iterations[select(iterations)]
    old = pieces[select(pieces) & (_col(pieces, "owner") == 0)]
    ne = effects[select(effects)]
    comparison = compare_first(t, ni, old, ne, hids, xids, data["seed_hit_mask"], data["crossing_partner"])
    for name in ("seed_old_owner", "effect_membership", "reference", "hit_active", "bounds_priors"):
        require(comparison["phases"][name]["status"] == "equal", "unresolved first-iteration precondition: " + name)
    material = comparison["phases"]["material"]
    require(not material["unsupported"], "unsupported native material payload")
    require("participation" in material["metrics"], "native material participation unavailable")
    slots = np.asarray(material["compared_f1_slots"], dtype=int)
    indices = np.asarray(material["compared_native_effect_indices"], dtype=int)
    require(len(slots) > 0 and np.all((slots >= nh) & (slots < nh+nx)), "empty/invalid material intersection")
    rows = _match(indices, _col(ne, "effect_index"))
    require(rows >= 0, "missing native effect join")
    native = ne[rows]
    natural = {name: t[name][np.argsort(t["order"])] for name in t if name.startswith("eff_") or name == "is_material"}
    active = natural["is_material"][nh:nh+nx]
    require(np.array_equal(active, t["crossing_active"] & t["crossing_inbounds"]), "saved material participation differs")
    all_native = ne[_col(ne, "kind") == 2]
    constructed = _match(_col(all_native, "strawid"), xids)
    require(constructed >= 0, "native material identity absent from saved crossings")
    payload = _col(all_native, "has_change", "b")
    f1_participates = active[constructed]
    native_participates = _col(all_native, "active", "b") & _col(all_native, "in_fit", "b")
    intersection = payload & f1_participates & native_participates
    require(np.array_equal(np.sort(constructed[intersection]), np.sort(slots-nh)),
            "material participation intersection differs")
    participation = metric(f1_participates[payload], native_participates[payload])
    participation.update(compared_crossing_indices=constructed[payload].tolist(),
        compared_native_effect_indices=_col(all_native, "effect_index")[payload].tolist(),
        f1_participates=f1_participates[payload].tolist(), native_participates=native_participates[payload].tolist(),
        n_constructed=len(all_native), n_payload_available=int(payload.sum()),
        n_native_participating=int(native_participates.sum()), n_intersection=int(intersection.sum()),
        n_f1_only=int((f1_participates & ~native_participates).sum()),
        n_native_only=int((native_participates & ~f1_participates).sum()),
        n_neither=int((~native_participates & ~f1_participates).sum()))
    scale = initial_scale(manifest)
    scalars = N.scattering_variants(natural["eff_xref"][nh:nh+nx, 5], t["crossing_wall"], t["crossing_gas"], scale)
    rebuilt = np.where(active, scalars["default"], 0.)
    saved_sv = natural["eff_sv"][nh:nh+nx]
    reconstruction = metric(rebuilt, saved_sv)
    tight = np.isclose(rebuilt, saved_sv, rtol=RECONSTRUCTION_RTOL, atol=0.)
    reconstruction["tighter_scalar_diagnostic"] = {"rtol": RECONSTRUCTION_RTOL, "atol": 0.,
        "passed": bool(tight.all()), "mismatches": int((~tight).sum()),
        "failing_crossing_indices": np.flatnonzero(~tight).tolist()}
    require(reconstruction["passed"] and tight.all(), "default scalar does not reconstruct saved sv: " + json.dumps(reconstruction))
    crossings = slots-nh
    q = _dense(native, "change_covariance", (6, 6))
    mean = _dense(native, "change_parameters", (6,))
    recovered = N.native_scalars(q, scale)
    shifted_D, kink_time = kink_time_diagnostic(natural["eff_xref"][slots], natural["eff_D"][slots],
        t["physical_time"][slots], t["crossing_offset"][crossings])
    variants = substitutions(natural["eff_xref"][slots], natural["eff_D"][slots],
        natural["eff_sv"][slots], scalars["p109"][crossings], natural["eff_dm"][slots], natural["eff_dmv"][slots], mean, q, shifted_D=shifted_D)
    # Every failure position maps through these explicit lists; no truncated failure inventory.
    result = {"key": key.tolist(), "native_seed": manifest["native_seed"],
        "trace_manifest": str(directory / "manifest.json"), "trace_manifest_sha256": manifest_hash,
        "trace_sha256": manifest["trace_sha256"], "trace_passivity_exact": True,
        "source_provenance": provenance, "saved_schedule": manifest["schedule"],
        "temperature": manifest["schedule"]["T"][0], "varscale": scale,
        "n_crossings": nx, "n_f1_participating": int(active.sum()), "n_compared_effects": len(slots),
        "compared_f1_slots": slots.tolist(), "compared_crossing_indices": crossings.tolist(),
        "compared_strawids": xids[crossings].tolist(), "compared_native_effect_indices": indices.tolist(),
        "participation": participation,
        "excluded_nonparticipating_native_changes": material["excluded_nonparticipating_native_changes"],
        "first_iteration_comparison": comparison, "default_reconstruction": reconstruction,
        "kink_time_diagnostic": kink_time, "variants": variants, "scalars": {name: value[crossings].tolist() for name, value in scalars.items()},
        "native_scalars": {name: value.tolist() for name, value in recovered.items()},
        "fixed_operands": {"momentum": natural["eff_xref"][slots, 5].tolist(),
            "wall": t["crossing_wall"][crossings].tolist(), "gas": t["crossing_gas"][crossings].tolist(),
            "dm": natural["eff_dm"][slots].tolist(), "dmv": natural["eff_dmv"][slots].tolist()},
        "full_closure": False}
    require(digest(directory / "manifest.json") == manifest_hash
            and digest(directory / "trace.npz") == manifest["trace_sha256"], "saved trace changed during audit")
    return result


def run(args):
    require(not args.out.exists(), "refusing existing audit output")
    require(not any(name.startswith("F1_") for name in os.environ), "requires default F1 environment")
    require(1 <= len(args.trace_dir) <= 3, "audit accepts one to three explicit target/configuration traces")
    require(len(set(d.resolve() for d in args.trace_dir)) == len(args.trace_dir), "duplicate trace directory")
    root = Path(__file__).resolve().parents[1]
    files = ("scripts/run_f1_operand_audit.sh", "scripts/f1_material_audit.py", "scripts/f1_trace_compare.py", "scripts/f1_native_decision_audit.py",
             "trkfastsim/reco/nativematerial.py", *REQUIRED_SOURCES)
    sources = {name: digest(root / name) for name in files}
    native_hashes = {str(args.native_source / name): digest(args.native_source / name) for name in NATIVE_SOURCES}
    report = {"schema_version": 1, "rtol": RTOL, "atol": ATOL, "source": sources,
        "D_reconstruction_rtol": RECONSTRUCTION_RTOL, "D_reconstruction_atol": D_RECONSTRUCTION_ATOL,
        "native_source": native_hashes, "limits": LIMITS, "full_closure": False,
        "full_numerical_validation": False, "stock_exact_fit": False, "scale_fit": False,
        "fractions": {"solid": N.SOLID_FRACTION, "saved_default_gas": N.DEFAULT_GAS_FRACTION,
                      "explicit_p109_gas": N.P109_GAS_FRACTION},
        "relative_residual_definition": "unfloored abs(actual-native)/abs(native); zero denominators counted separately"}
    expanded = args.check.parent/"capture.expanded.fcl"
    import re
    configuration = expanded.read_text()
    fractions = {}
    for name, expected in (("GasScatteringFraction", N.P109_GAS_FRACTION), ("SolidScatteringFraction", N.SOLID_FRACTION)):
        values = re.findall(r"\b"+name+r"\s*:\s*([0-9eE.+-]+)", configuration)
        require(values and all(float(v) == expected for v in values), "unsupported captured " + name)
        fractions[name] = values
    report["expanded_configuration"] = {"path": str(expanded), "sha256": digest(expanded), "fractions": fractions,
        "scope": "recorded fractions only; not full FHiCL equivalence"}
    iterations, pieces = load_evidence(args.check, report)
    report["all_captured_iterations_included"] = False
    with uproot.open(report["inputs"]["capture"]["path"]) as f:
        effects = read_tree(f, "kktrace", "effects", report["highest_cycles"]["capture"])
    report["targets"] = [audit_one(d, report, iterations, pieces, effects) for d in args.trace_dir]
    require(sources == {name: digest(root / name) for name in files}, "diagnostic source changed during audit")
    require(native_hashes == {name: digest(name) for name in native_hashes}, "native source changed during audit")
    require(digest(args.check) == report["check_sha256"] and digest(expanded) == report["expanded_configuration"]["sha256"],
            "native gate/configuration changed during audit")
    report["completed"] = True
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # One JSON report, no unsolicited stdout progress or alternate output artifacts.
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
