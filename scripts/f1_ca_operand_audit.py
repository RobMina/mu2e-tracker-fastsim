"""Reconstruct native local CA arithmetic on saved first-iteration hits; not a fit."""
import argparse
import json
import os
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from f1_native_decision_audit import load_evidence, read_tree, digest, require
from f1_measurement_audit import audit_one, substitutions, NATIVE_SOURCES
from f1_trace_compare import _col, _dense, _metric
from trkfastsim.reco import hitreco, vr_f1, nativeca as C


LIMITS = [
    "Local single-piece CA reconstruction, no full native piece/hit-z retry, unusable-hint fallback or captured prior hints.",
    "Nonfinite/overflow arithmetic fails the audit; native failure-state propagation is not reproduced.",
    "One-ULP sensitivity compares each fixed perturbation to its reconstruction, never selects a best native match.",
    "Dumped wire endpoints and measured hit corrected times seed an explicit constructor then update replay.",
    "NumPy source-formula arithmetic is not ROOT/C++ bit parity; native CA vectors and sensor times are absent.",
    "Captured-reference diagnostic is explicitly distinguished from the saved-F1-reference arm.",
    "Both saved-value row isolation and reconstructed-value variants hold saved weights fixed.",
    "No fit rerun, tolerance/default change, material/decision/acceptance or full closure claim.",
]


def endpoint_inputs(source, context):
    """Join complete dumped wire endpoints and measured source-hit records without event loops."""
    with uproot.open(source) as f:
        g = f["trkdump/conditions"].arrays(["sid", "wcalx", "wcaly", "wcalz", "whvx", "whvy", "whvz"], library="np")
        e = context["event_index"]
        events = f["trkdump/events"].arrays(["s_sid", "s_time", "s_edep"], entry_start=e, entry_stop=e+1)
    require(len(events) == 1, "missing target event")
    geom_sid = g["sid"][0].astype(np.int64)
    require(len(np.unique(geom_sid)) == len(geom_sid), "duplicate endpoint straw identity")
    ids = np.asarray(context["hit_ids"])
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and np.all(ids >= 0)
            and np.all(ids < len(events.s_sid[0])) and len(np.unique(ids)) == len(ids), "invalid source-hit indices")
    sid = ak.to_numpy(events.s_sid[0])[ids].astype(np.int64)
    require(np.array_equal(sid, _col(context["native"], "strawid")), "native/source hit straw differs")
    from f1_trace_compare import _match
    loc = _match(sid, geom_sid)
    require(loc >= 0, "missing wire endpoint")
    cal = np.column_stack([g[k][0][loc] for k in ("wcalx", "wcaly", "wcalz")])
    hv = np.column_stack([g[k][0][loc] for k in ("whvx", "whvy", "whvz")])
    sign = context["h"]["esign"]
    early, late = np.where(sign[:, None] > 0, cal, hv), np.where(sign[:, None] > 0, hv, cal)
    corrected = ak.to_numpy(events.s_time[0])[ids].astype(float)
    energy = ak.to_numpy(events.s_edep[0])[ids].astype(float)
    return early, late, corrected, energy, sid


def ca_variant(q, line, corrected, context, initial=None, mass=C.N.MASS, arithmetic=False):
    """Explicit constructor/update sequence; capture-time starts are separately labelled by caller."""
    if initial is None:
        ph, sh = C.constructor_hint(q, line, corrected, mass)
        built = C.solve(q, line, ph, sh, mass=mass)
    else:
        built = C.solve(q, line, *initial, mass=mass)
    state = C.solve(q, line, built["particle_time"], built["sensor_time"], mass=mass)
    rows = C.rows_at_state(q, line, state, mass)
    r, w = context["residual"], context["weight"]
    result = substitutions(q, context["actual_H"], rows["H"], r, w,
                           context["native_W"], context["native_b"])["all_source_rows"]
    h = context["h"]
    values = np.stack((-rows["doca"], state["sensor_time"]-state["particle_time"]-h["totdt"],
        rows["longitudinal"]-h["wdist"]*np.where(h["esign"]*h["udir_dot_w"] < 0, -1., 1.)), -1)
    changed_values = substitutions(q, context["actual_H"], rows["H"], values, w,
                                  context["native_W"], context["native_b"])["all_source_rows"]
    sensitivity = {}
    if mass == context["native_mass"]:
        for field in ("particle_time", "sensor_time"):
            for label, toward in (("minus", -np.inf), ("plus", np.inf)):
                shifted = dict(state)
                shifted[field] = np.nextafter(state[field], toward)
                shifted["particle_poca"] = C.particle(q, shifted["particle_time"], mass)[0]
                shifted["sensor_poca"] = C.sensor_position(line, shifted["sensor_time"])
                shifted["delta"] = shifted["sensor_poca"]-shifted["particle_poca"]
                Hshift = C.rows_at_state(q, line, shifted, mass)["H"]
                Wshift, bshift = C.N.information(q, Hshift, r, w)
                Wbase, bbase = C.N.information(q, rows["H"], r, w)
                sensitivity[field+"_"+label+"_one_ulp"] = {
                    "time_delta": (shifted[field]-state[field]).tolist(),
                    "row_vs_unshifted": _metric(Hshift, rows["H"]),
                    "weight_matrix_vs_unshifted": _metric(Wshift, Wbase),
                    "weight_vector_vs_unshifted": _metric(bshift, bbase)}
    arithmetic_result = arithmetic_diagnostic(q, line, state, rows, context, mass) if arithmetic else None
    return {"row_only": result, "rows_and_values": changed_values, "one_ulp_sensitivity": sensitivity,
        "fixed_state_arithmetic": arithmetic_result,
        "value_vs_saved": _metric(values, r), "time_vs_native": _metric(state["particle_time"], _col(context["native"], "time")),
        "time_vs_native_exact": _metric(state["particle_time"], _col(context["native"], "time"), exact=True),
        "particle_z_distance_from_wire_mid": (state["particle_poca"][:, 2]-line["mid"][:, 2]).tolist(),
        "hit_z_retry_trigger_count": int(np.count_nonzero(np.abs(state["particle_poca"][:, 2]-line["mid"][:, 2]) > 100.)),
        "constructor_updates": built["updates"].tolist(), "update_updates": state["updates"].tolist(),
        "status": state["status"].tolist(), "doca": rows["doca"].tolist(),
        "delta_dot_particle": rows["delta_dot_particle"].tolist(),
        "delta_dot_sensor": rows["delta_dot_sensor"].tolist(),
        "particle_time": state["particle_time"].tolist(), "sensor_time": state["sensor_time"].tolist(),
        "delta": state["delta"].tolist(), "H": rows["H"].tolist()}


def arithmetic_diagnostic(q, line, state, rows, context, mass):
    """Fixed CA state; two predeclared derivative/information grouping controls."""
    from trkfastsim.reco import nativearith as A
    ordered = A.rows_at_state(q, line, state, mass)
    r, w = context["residual"], context["weight"]
    result = {"scope": A.ARITHMETIC_DIAGNOSTIC, "CA_state_changed": False,
              "values_or_weights_changed": False,
              "rows_vs_previous": _metric(ordered["H"], rows["H"]), "variants": {}}
    for name, H, info in (("source_rows_only", ordered["H"], C.N.information),
                           ("source_information_only", rows["H"], A.information),
                           ("source_rows_and_information", ordered["H"], A.information)):
        W, b = info(q, H, r, w)
        bad = (~np.all(np.isclose(W, context["native_W"], rtol=1e-5, atol=1e-7), axis=(-2, -1))
               | ~np.all(np.isclose(b, context["native_b"], rtol=1e-5, atol=1e-7), axis=-1))
        result["variants"][name] = {"weight_matrix": _metric(W, context["native_W"]),
            "weight_vector": _metric(b, context["native_b"]), "failing_hit_indices": np.flatnonzero(bad).tolist(),
            "per_hit_matrix_maxabsolute": np.max(np.abs(W-context["native_W"]), axis=(-2, -1)).tolist(),
            "per_hit_vector_maxabsolute": np.max(np.abs(b-context["native_b"]), axis=-1).tolist()}
    return result


def root_arithmetic_sources(native_source):
    """Bind the actual ROOT include path recorded in the frozen KinKal build commands."""
    build = native_source/"kinkal_build_p109/compile_commands.json"
    commands = json.loads(build.read_text())
    import shlex
    includes = {token[2:] for row in commands for token in shlex.split(row["command"])
                if token.startswith("-I") and token.endswith("/include/root")}
    require(len(includes) == 1, "ambiguous frozen ROOT include provenance")
    include = Path(includes.pop())
    files = [build, include/"Math/GenVector/DisplacementVector3D.h", include/"Math/GenVector/Cartesian3D.h"]
    return {str(p): digest(p) for p in files}


def run(args):
    require(not args.out.exists(), "refusing existing CA audit output")
    require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
    require(1 <= len(args.trace_dir) <= 3, "CA audit accepts one to three explicit traces")
    require(len({p.resolve() for p in args.trace_dir}) == len(args.trace_dir), "duplicate trace directory")
    metadata = {"schema_version": 1, "full_closure": False, "full_numerical_validation": False,
                "limits": LIMITS, "precision": 1e-4, "max_updates": 100, "targets": []}
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), root/"scripts/f1_measurement_audit.py", root/"scripts/f1_trace_compare.py",
                    root/"scripts/f1_native_decision_audit.py", root/"scripts/run_f1_operand_audit.sh"]
    source_files += sorted((root/"trkfastsim/reco").glob("*.py"))
    metadata["source"] = {str(p.relative_to(root)): digest(p) for p in source_files}
    paths = NATIVE_SOURCES + ("KinKal/Trajectory/KinematicLine.hh", "KinKal/Trajectory/SensorLine.hh",
                             "KinKal/Trajectory/ConstantDistanceToTime.hh", "KinKal/Trajectory/ConstantDistanceToTime.cc",
                             "KinKal/Trajectory/SensorLine.cc", "KinKal/Geometry/LineSegment.hh",
                             "KinKal/Geometry/Ray.hh", "Offline/Mu2eKinKal/inc/KKFit.hh",
                             "Offline/Mu2eKinKal/src/KKFitUtilities.cc")
    metadata["native_source"] = {str(args.native_source/p): digest(args.native_source/p) for p in paths}
    metadata["root_arithmetic_source"] = root_arithmetic_sources(args.native_source)
    expanded = args.check.parent/"capture.expanded.fcl"
    import re
    precision_values = re.findall(r"\bTPOCAPrecision\s*:\s*([0-9eE.+-]+)", expanded.read_text())
    require(precision_values and all(float(v) == 1e-4 for v in precision_values), "unsupported captured CA precision")
    metadata["expanded_configuration"] = {"path": str(expanded), "sha256": digest(expanded),
        "TPOCAPrecision_values": precision_values, "scope": "recorded precision only, not full FHiCL equivalence"}
    iterations, pieces = load_evidence(args.check, metadata)
    metadata["all_captured_iterations_included"] = False
    manifest = json.loads((args.trace_dir[0] / "manifest.json").read_text())
    cond = hitreco.load_conditions(manifest["conditions"]["path"])
    source = metadata["inputs"]["baseline"]["path"]
    _, hinput, _, ref, drift, _ = vr_f1.read_inputs(source, cond, "mc", straw_axis=manifest["alignment"]["path"], fitok_only=False)
    with uproot.open(metadata["inputs"]["capture"]["path"]) as f:
        effects = read_tree(f, "kktrace", "effects", metadata["highest_cycles"]["capture"])
    for directory in args.trace_dir:
        manifest_hash = digest(directory/"manifest.json")
        current = json.loads((directory / "manifest.json").read_text())
        require(all(current[k] == manifest[k] for k in ("conditions", "alignment")), "target inputs differ")
        context = {}
        base = audit_one(directory, metadata, iterations, pieces, effects, hinput, ref, drift,
                         args.allow_loader_reconstruction, diagnostic_context=context)
        early, late, corrected, energy, sid = endpoint_inputs(source, context)
        h = context["h"]
        line = C.sensor(early, late, h["tmeas"], h["vsig"])
        geometry = {"midpoint": _metric(line["mid"], h["mid"]),
                    "direction": _metric(line["direction"], h["wdir"]*h["esign"][:, None]),
                    "half_length": _metric(.5*line["length"], h["hlen"]), "energy": _metric(energy, h["edep"])}
        native_q = _dense(context["native"], "reference_parameters", (6,))
        variants = {"f1_reference_endpoints": ca_variant(context["q"], line, corrected, context),
                    "native_reference_endpoints": ca_variant(native_q, line, corrected, context),
                    "native_reference_mass": ca_variant(native_q, line, corrected, context, mass=context["native_mass"], arithmetic=True),
                    "f1_reference_native_mass": ca_variant(context["q"], line, corrected, context, mass=context["native_mass"])}
        from trkfastsim.reco import nativearith as A
        root_line = A.sensor(early, late, h["tmeas"], h["vsig"])
        variants["native_reference_mass_root_sensor"] = ca_variant(
            native_q, root_line, corrected, context, mass=context["native_mass"], arithmetic=True)
        variants["f1_reference_native_mass_root_sensor"] = ca_variant(
            context["q"], root_line, corrected, context, mass=context["native_mass"], arithmetic=True)
        geometry["root_sensor_direction_vs_previous_exact"] = _metric(root_line["direction"], line["direction"], exact=True)
        captime = _col(context["native"], "time")
        _, u, _ = C.particle(native_q, captime)
        pp, _, _ = C.particle(native_q, captime)
        distance = np.sum((line["end"]-pp)*line["direction"], -1)
        sensor_hint = line["measurement_time"]-distance/line["speed"]
        variants["native_time_hint_diagnostic"] = ca_variant(native_q, line, corrected, context, (captime, sensor_hint))
        metadata["targets"].append({"key": base["key"], "trace_manifest_sha256": base["trace_manifest_sha256"],
            "trace_sha256": base["trace_sha256"], "native_seed": base["native_seed"],
            "loader_provenance": base["loader_provenance"], "source_shindex": context["hit_ids"].tolist(),
            "strawid": sid.tolist(), "endpoint_geometry": geometry, "native_mass": context["native_mass"],
            "f1_mass": C.N.MASS,
            "baseline_time_long": base["variants"]["time_long"], "variants": variants})
        require(digest(directory/"manifest.json") == manifest_hash and
                digest(directory/"trace.npz") == current["trace_sha256"], "trace changed during CA audit")
    require(metadata["source"] == {str(p.relative_to(root)): digest(p) for p in source_files}, "audit source changed")
    require(metadata["native_source"] == {str(args.native_source/p): digest(args.native_source/p) for p in paths},
            "native source changed")
    require(metadata["root_arithmetic_source"] == root_arithmetic_sources(args.native_source), "ROOT arithmetic source changed")
    require(digest(args.check) == metadata["check_sha256"] and digest(expanded) == metadata["expanded_configuration"]["sha256"],
            "native evidence/configuration changed")
    metadata["completed"] = True
    with args.out.open("x") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
        f.write("\n")
    for t in metadata["targets"]:
        print(json.dumps({"key": t["key"], "native_seed": t["native_seed"], "endpoint_geometry": t["endpoint_geometry"],
            "variants": {name: {label: {k: m[label][k]["mismatches"] for k in ("weight_matrix", "weight_vector")}
                                for label in ("row_only", "rows_and_values")}
                         for name, m in t["variants"].items()}}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-loader-reconstruction", action="store_true")
    run(parser.parse_args())
