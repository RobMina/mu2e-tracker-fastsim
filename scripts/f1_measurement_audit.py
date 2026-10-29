"""Isolate source-derived row policies on saved first-iteration F1 operands; never refit."""
import argparse
import json
import os
from pathlib import Path

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import uproot

from f1_native_decision_audit import load_evidence, read_tree, keys, digest, require
from f1_trace_compare import compare_first, _col, _dense, _metric, _match
from trkfastsim.reco import fittrace, hitreco, kalman_jax as K, nativecompare, nativemeasurement as N, nativeref, vr_f1
from trkfastsim.reco.linefit_jax import HIT_KEYS, DRIFT_KEYS


LIMITS = [
    "Only saved actual iteration zero; no refit, later-history or decision-causality claim.",
    "Source-policy rows use analytic CA at saved F1 references, not captured native CA internals.",
    "Residuals and diagonal weights are held fixed to saved F1 values in every substitution.",
    "No native individual residual rows/variances captured; aggregate information agreement is bounded evidence.",
    "Orthogonal-normal DOCA variant is a stability diagnostic, not the native delta-unit-vector arithmetic.",
    "No acceptance/fidelity/ROOT arithmetic or all-discordance closure; defaults unchanged.",
]
NATIVE_SOURCES = ("KinKal/Trajectory/KinematicLine.cc", "KinKal/Trajectory/ClosestApproach.hh",
                  "KinKal/Trajectory/ClosestApproachData.hh", "Offline/Mu2eKinKal/inc/KKStrawHit.hh",
                  "KinKal/Detector/Residual.cc", "KinKal/Detector/ResidualHit.hh")


def substitutions(q, actual_H, source_H, r, w, native_W, native_b):
    """Keep values/variances/references fixed; independently replace each derivative family."""
    variants = {}
    for label, components in (("saved", ()), ("time_only", (1,)), ("long_only", (2,)),
                               ("time_long", (1, 2)), ("all_source_rows", (0, 1, 2))):
        H = actual_H.copy()
        H[..., components, :] = source_H[..., components, :]
        W, b = N.information(q, H, r, w)
        wm, bm = _metric(W, native_W), _metric(b, native_b)
        badW = ~np.all(np.isclose(W, native_W, rtol=1e-5, atol=1e-7), axis=(-1, -2))
        badb = ~np.all(np.isclose(b, native_b, rtol=1e-5, atol=1e-7), axis=-1)
        variants[label] = {"weight_matrix": wm, "weight_vector": bm,
                           "failing_hit_indices": np.flatnonzero(badW | badb).tolist(),
                           "per_hit_matrix_maxabsolute": np.max(np.abs(W-native_W), axis=(-1, -2)).tolist(),
                           "per_hit_vector_maxabsolute": np.max(np.abs(b-native_b), axis=-1).tolist()}
    return variants


def audit_one(directory, metadata, iterations, pieces, effects, Hinput, ref, drift, allow_loader_reconstruction=False,
              diagnostic_context=None):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("completed") is True and manifest.get("trace_passivity_exact") is True,
            "requires complete passive saved trace")
    require(not manifest.get("environment"), "requires default F1 environment")
    require(manifest["check_sha256"] == metadata["check_sha256"], "native gate differs")
    require(manifest["trace_sha256"] == digest(directory / "trace.npz"), "trace hash changed")
    require(manifest["time_model"] == "stock" and manifest["reference_model"] == "previous"
            and manifest["straw_geometry"] == "approximate" and manifest["crossing_time"] == "current",
            "unsupported saved fit policy")
    root = Path(__file__).resolve().parents[1]
    loader_provenance = {}
    for name in ("kalman_jax.py", "linefit_jax.py", "linefit.py", "hitreco.py", "vr_f1.py",
                 "cli.py", "combo.py", "strawaxis.py", "nativegeometry.py", "nativecompare.py"):
        rel = "trkfastsim/reco/" + name
        current, saved_hash = digest(root / rel), manifest["source"][rel]
        if name == "vr_f1.py" and current != saved_hash:
            require(allow_loader_reconstruction, "loader source differs; explicit reconstruction option required")
            loader_provenance = {"saved_sha256": saved_hash, "current_sha256": current,
                "source_equal": False, "validation": "explicit reloading; saved identities/midpoints/membership exact and rows checked below"}
        else:
            require(current == saved_hash, "trace input/row source changed: " + rel)
    require(manifest["inputs"]["baseline"] == metadata["inputs"]["baseline"], "baseline differs")
    for label in ("conditions", "alignment"):
        require(digest(manifest[label]["path"]) == manifest[label]["sha256"], label + " hash changed")
    with np.load(directory / "trace.npz", allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    key = data["key"]
    require(np.array_equal(key, manifest["key"]), "trace key differs")
    require(data["stage_index"][0] == 0, "missing first stage")
    hids, xids = data["hit_ids"], data["crossing_ids"]
    nh = len(hids)
    t = {name: data[name][0] for name in fittrace.shapes(nh, len(xids))}
    select = lambda table: np.all(keys(table, unique=False) == key, axis=1) & (_col(table, "iteration_ordinal") == 0)
    ni, old, ne = iterations[select(iterations)], pieces[select(pieces) & (_col(pieces, "owner") == 0)], effects[select(effects)]
    comparison = compare_first(t, ni, old, ne, hids, xids, data["seed_hit_mask"], data["crossing_partner"])
    for name in ("seed_old_owner", "effect_membership", "reference", "hit_active", "bounds_priors"):
        require(comparison["phases"][name]["status"] == "equal", "unresolved first-iteration precondition: " + name)
    require(not comparison["phases"]["measurement"]["unsupported"], "unsupported native measurement payload")
    ids = np.column_stack([ref[f] for f in ("run", "subrun", "event")] + [np.zeros(len(ref["run"]), np.int64)])
    target = nativeref.match_rows(key[None], ids)[0]
    require(target >= 0, "target absent from eligible F1 inputs")
    require(int(Hinput["valid"][target].sum()) == nh, "target hit count changed")
    require(np.array_equal(Hinput["shloc"][target, :nh], hids), "source hit identities changed")
    require(np.array_equal(Hinput["mid"][target, :nh], data["hit_mid"]), "saved midpoint changed")
    require(np.array_equal(Hinput["inseed"][target, :nh], data["seed_hit_mask"]), "seed membership changed")
    natural = {name: t[name][np.argsort(t["order"])] for name in t if name.startswith("eff_")}
    slots = np.flatnonzero(data["seed_hit_mask"])
    hits = np.flatnonzero(_col(ne, "kind") == 1)
    match = _match(hids[slots], _col(ne, "shindex")[hits])
    require(match >= 0, "missing native hit join")
    native = ne[hits[match]]
    h = {name: jnp.asarray(Hinput[name][target, :nh][slots]) for name in HIT_KEYS}
    h.update(active=jnp.asarray(t["hit_active"][slots]), lr=jnp.asarray(t["hit_lr"][slots]))
    require(np.all(np.asarray(h["lr"]) == 0), "audit presently certifies first null-hit rows only")
    x, r, w = (natural[name][slots] for name in ("eff_xref", "eff_r", "eff_w"))
    q, _, inverse = nativecompare._reference(x)
    actual_H = natural["eff_H"][slots] @ inverse
    dj = {name: jnp.asarray(drift[name], dtype=jnp.float64) for name in DRIFT_KEYS}
    rebuilt = jax.jit(K.hit_rows)(jnp.asarray(x), h, dj)
    rebuilt_r, rebuilt_H = np.asarray(rebuilt[0]), np.asarray(rebuilt[1])
    reconstruction = {"residual": _metric(rebuilt_r, r), "local_H": _metric(rebuilt_H, natural["eff_H"][slots])}
    require(all(m["passed"] for m in reconstruction.values()), "reconstructed F1 rows do not match saved rows")
    source = N.rows(q, np.asarray(h["mid"]), np.asarray(h["wdir"]), np.asarray(h["esign"]), np.asarray(h["vsig"]))
    algebra_checks = {"doca_row": _metric(source["H"][:, 0], actual_H[:, 0]),
        "time_plus_omission": _metric(source["H"][:, 1]+source["omitted_time_direction"], actual_H[:, 1]),
        "long_plus_delta_term": _metric(source["H"][:, 2]+source["full_minus_native_long"], actual_H[:, 2])}
    require(all(algebra_checks[k]["passed"] for k in ("time_plus_omission", "long_plus_delta_term")),
            "time/long derivative decomposition failed: " + json.dumps(algebra_checks, allow_nan=False))
    native_W, native_b = _dense(native, "weight_matrix", (6, 6)), _dense(native, "weight_vector", (6,))
    results = substitutions(q, actual_H, source["H"], r, w, native_W, native_b)
    for item in results.values():
        item["failing_source_shindex"] = hids[slots][item["failing_hit_indices"]].tolist()
    result = {"key": key.tolist(), "trace_manifest": str(manifest_path), "trace_manifest_sha256": digest(manifest_path),
              "trace_sha256": manifest["trace_sha256"], "native_seed": manifest["native_seed"],
              "n_hits": len(slots), "source_shindex": hids[slots].tolist(), "loader_provenance": loader_provenance,
              "native_effect_indices": _col(native, "effect_index").tolist(),
              "preconditions": {name: comparison["phases"][name]["status"] for name in comparison["phase_order"][:5]},
              "reconstructed_rows": reconstruction, "derivative_decomposition": algebra_checks,
              "source_vs_saved_rows": {label: _metric(source["H"][:, i], actual_H[:, i])
                                       for i, label in enumerate(("doca", "time", "long"))},
              "variants": results, "full_closure": False,
              "ca_diagnostics": {"doca": source["doca"].tolist(),
                  "delta_dot_particle": source["delta_dot_particle"].tolist(),
                  "delta_dot_sensor": source["delta_dot_sensor"].tolist(),
                  "particle_time_vs_native": _metric(source["particle_time"], _col(native, "time")),
                  "orthogonal_doca_vs_saved": _metric(source["orthogonal_doca_H"], actual_H[:, 0])},
              "residuals": r.tolist(), "weights": w.tolist()}
    stable_H = source["H"].copy()
    stable_H[:, 0] = source["orthogonal_doca_H"]
    result["orthogonal_doca_diagnostic"] = substitutions(q, actual_H, stable_H, r, w, native_W, native_b)["all_source_rows"]
    if diagnostic_context is not None:
        diagnostic_context.update(q=q, actual_H=actual_H, h={k: np.asarray(v) for k, v in h.items()},
            residual=r, weight=w, native=native, native_W=native_W, native_b=native_b,
            hit_ids=hids[slots], event_index=int(ref["event_index"][target]), native_mass=float(_col(old, "mass")[0]))
    return result


def run(args):
    require(not args.out.exists(), "refusing existing audit output")
    require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
    metadata = {"schema_version": 1, "full_closure": False, "full_numerical_validation": False,
                "rtol": 1e-5, "atol": 1e-7, "limits": LIMITS}
    iterations, pieces = load_evidence(args.check, metadata)
    metadata["all_captured_iterations_included"] = False
    manifests = [json.loads((d / "manifest.json").read_text()) for d in args.trace_dir]
    for label in ("conditions", "alignment"):
        require(all(m[label] == manifests[0][label] for m in manifests), "trace input configurations differ")
        require(digest(manifests[0][label]["path"]) == manifests[0][label]["sha256"], label+" changed")
    metadata.update({label: manifests[0][label] for label in ("conditions", "alignment")})
    cond = hitreco.load_conditions(manifests[0]["conditions"]["path"])
    _, H, _, ref, drift, _ = vr_f1.read_inputs(metadata["inputs"]["baseline"]["path"], cond, "mc",
        straw_axis=manifests[0]["alignment"]["path"], fitok_only=False)
    with uproot.open(metadata["inputs"]["capture"]["path"]) as f:
        effects = read_tree(f, "kktrace", "effects", metadata["highest_cycles"]["capture"])
    metadata["native_source"] = {str(args.native_source / p): digest(args.native_source / p) for p in NATIVE_SOURCES}
    metadata["targets"] = [audit_one(d, metadata, iterations, pieces, effects, H, ref, drift,
                                      args.allow_loader_reconstruction) for d in args.trace_dir]
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "scripts/f1_trace_compare.py", root / "scripts/f1_native_decision_audit.py"]
    sources += sorted((root / "trkfastsim/reco").glob("*.py"))
    sources.append(root / "scripts/run_f1_measurement_audit.sh")
    metadata["source"] = {str(p.relative_to(root)): digest(p) for p in sources}
    require(metadata["native_source"] == {str(args.native_source / p): digest(args.native_source / p)
                                          for p in NATIVE_SOURCES}, "native source changed during audit")
    metadata["completed"] = True
    with args.out.open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for target in metadata["targets"]:
        print(json.dumps({"key": target["key"], "native_seed": target["native_seed"], "n_hits": target["n_hits"],
            "variants": {name: {field: item[field]["mismatches"] for field in ("weight_matrix", "weight_vector")}
                         for name, item in target["variants"].items()},
            "orthogonal_doca": {field: target["orthogonal_doca_diagnostic"][field]["mismatches"]
                                for field in ("weight_matrix", "weight_vector")},
            "time_long_maxabsolute": {field: target["variants"]["time_long"][field]["max_absolute_residual"]
                                      for field in ("weight_matrix", "weight_vector")},
            "doca_abs_range": [float(np.min(np.abs(target["ca_diagnostics"]["doca"]))),
                               float(np.max(np.abs(target["ca_diagnostics"]["doca"])))],
            "worst_time_long_hits": {field: {"shindex": int(np.asarray(target["source_shindex"])[
                np.argmax(target["variants"]["time_long"][field])]),
                "maxabsolute": max(target["variants"]["time_long"][field])}
                for field in ("per_hit_matrix_maxabsolute", "per_hit_vector_maxabsolute")},
            "native_time": target["ca_diagnostics"]["particle_time_vs_native"],
            "orthogonal_doca_vs_saved": target["ca_diagnostics"]["orthogonal_doca_vs_saved"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-loader-reconstruction", action="store_true",
                        help="permit changed input-loader source only, with recorded hashes and saved-row checks")
    run(parser.parse_args())
