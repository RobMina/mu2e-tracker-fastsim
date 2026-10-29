"""Compare validated native CA snapshots with saved F1/local-CA operands; never refit."""
import argparse
import json
import os
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from f1_native_decision_audit import digest, require, load_evidence, read_tree, keys, match_rows
from f1_hit_operand_check import HIT_KEYS, ORIGINAL_KEYS, NATIVE_CA_PRECISION
from f1_measurement_audit import audit_one, NATIVE_SOURCES
from f1_ca_operand_audit import endpoint_inputs, root_arithmetic_sources
from f1_trace_compare import _col, _dense, _metric
from trkfastsim.reco import nativeca as C, nativearith as A, hitreco, vr_f1

FROZEN_EXPANDED_SHA256 = "03488bb9812f101e11e8d571882432c17b9a1c89927c6b5a2f01c5024e127e1d"
LIMITS = [
    "Selected saved first iterations only; no refit, full piece search, retry, extension or decision replay.",
    "Captured-reference/state/row substitutions are diagnostics, not independently generated F1 operands.",
    "Geometry exactness is reported separately from unchanged rtol=1e-5, atol=1e-7 numerical comparisons.",
    "No ULP search, fitted correction, regularization, tolerance/default change or full closure claim.",
    "Only active null hits with TOT and longitudinal residuals and usable CA are supported here.",
    "Source-grouped NumPy derivatives do not certify ROOT SMatrix/compiler/transcendental parity.",
    "Frozen expanded FCL is hash-pinned to the prior audit; no original capture-generation FCL hash receipt exists.",
]


def receipt(path, check, frozen_capture):
    """Bind the successful full sidecar gate, including every before/after input hash."""
    path = Path(path)
    report = json.loads(path.read_text())
    for flag in ("passed", "hashes_unchanged", "all_four_evidence_hashes_verified", "capture_off_hit_operands_empty"):
        require(report.get(flag) is True, "sidecar gate missing: " + flag)
    require(report.get("check_sha256") == digest(check), "sidecar/new gate differs")
    require(report.get("reference_capture", {}).get("sha256") == digest(frozen_capture), "sidecar/frozen reference differs")
    require(set(report.get("original_tree_passivity", {})) == set(ORIGINAL_KEYS), "missing exact original-tree gate")
    require(all(v.get("passed") is True for v in report["original_tree_passivity"].values()), "original-tree gate failed")
    require(all(type(report.get(k)) is int for k in ("n_hits", "n_expected_hits", "n_excluded_hits")), "missing integer sidecar census")
    require(report["n_hits"] > 0 and report["n_excluded_hits"] == 0 and report["n_hits"] == report["n_expected_hits"], "sidecar census differs")
    require(report.get("rtol") == 1e-5 and report.get("atol") == 1e-7, "sidecar tolerances differ")
    require(report.get("end_sha256"), "missing sidecar hash manifest")
    watched = dict(report["end_sha256"])
    watched[str(path.resolve())] = digest(path)
    require(all(digest(p) == h for p, h in watched.items()), "sidecar evidence/source hash changed")
    return report, watched


def join_null_rows(rows, native):
    """Full-key snapshot join and explicit native prediction to F1 residual convention."""
    index = match_rows(keys(native, HIT_KEYS), keys(rows, HIT_KEYS))
    selected = rows[index]
    for field in ("shindex", "strawid", "input_index", "miter", "iter", "active", "in_fit"):
        require(np.array_equal(_col(selected, field), _col(native, field)), "snapshot/effect differs: " + field)
    for field in ("active", "ca_usable", "constrain_tot", "constrain_long"):
        require(_col(selected, field), "unsupported first-hit flag: " + field)
    require(_col(selected, "constrain_absdrift") == 0, "absolute drift time unsupported")
    require(_col(selected, "hit_state") == 0, "requires native null hits")
    require(_dense(selected, "row_active", (3,)) == 1, "requires all three residual rows active")
    require(_col(selected, "precision") == NATIVE_CA_PRECISION, "native precision differs")
    g = _dense(selected, "row_derivative", (3, 6))
    value = _dense(selected, "row_value", (3,))
    mvar = _dense(selected, "row_mvar", (3,))
    scale = _col(selected, "variance_scale")
    require(np.isfinite(g), "nonfinite native derivative")
    require(np.isfinite(value), "nonfinite native residual")
    require(np.isfinite(mvar) & (mvar > 0), "invalid native measurement variance")
    require(np.isfinite(scale) & (scale > 0), "invalid native scale")
    signs = np.array([-1., 1., 1.])
    H, r, w = -signs[None, :, None]*g, signs[None, :]*value, 1./(mvar*scale[:, None])
    require(np.isfinite(w) & (w > 0), "nonfinite native inverse variance")
    require(np.array_equal(g[:, 0], _dense(selected, "dDdP", (6,))), "null row differs from cached dDdP")
    require(np.array_equal(g[:, 1], _dense(selected, "dTdP", (6,))), "TOT row differs from cached dTdP")
    require(np.array_equal(value[:, 0], _col(selected, "doca")), "null value differs from cached doca")
    return selected, H, r, w


def snapshot_state(rows):
    pp, sp = (_dense(rows, field, (4,)) for field in ("particle_poca", "sensor_poca"))
    return dict(particle_time=pp[:, 3], sensor_time=sp[:, 3], particle_poca=pp[:, :3], sensor_poca=sp[:, :3],
                delta=sp[:, :3]-pp[:, :3], direction=_dense(rows, "particle_direction", (3,)),
                sensor_direction=_dense(rows, "sensor_direction", (3,)), usable=_col(rows, "ca_usable"))


@A._checked
def fixed_state_rows(q, line, state, mass):
    """Source derivatives at supplied vectors/directions and particle time; no CA solve."""
    require(np.asarray(state["usable"]).all(), "unusable fixed state")
    _, _, speed, dx, dm = A.kinematics(q, state["particle_time"], mass)
    u, s, delta = (np.asarray(state[k], dtype=float) for k in ("direction", "sensor_direction", "delta"))
    require(u.shape == s.shape == delta.shape == q.shape[:-1]+(3,), "fixed vector shape differs")
    require(np.isfinite(u) & np.isfinite(s) & np.isfinite(delta), "nonfinite fixed state")
    dot, norm = A._dot3(s, u), np.sqrt(A._dot3(delta, delta))
    den = 1.-dot*dot
    require((den >= 1e-5) & (norm > 0), "undefined fixed CA derivative")
    sign = np.copysign(1., A._dot3(np.cross(s, u), delta))
    dD = A._contract3(sign[:, None]*(delta*(1./norm)[:, None]), dx)
    beta = (1./(speed*den))[:, None]*(s*dot[:, None]-u)
    gamma = (1./(line["speed"]*den))[:, None]*(s-dot[:, None]*u)
    dT = A._contract3(beta-gamma, dx)
    ds, dp, ddm = A._contract3(s, dx), A._contract3(u, dx), A._contract3(delta, dm)
    Hlong = ds+((dot[:, None]*ds-dp-ddm)/den[:, None])*dot[:, None]
    H = np.stack((dD, -dT, Hlong), axis=1)
    require(np.isfinite(H), "nonfinite fixed derivative")
    longitudinal = A._dot3(state["sensor_poca"]-line["mid"], s)
    require(np.isfinite(norm) & np.isfinite(longitudinal), "nonfinite fixed position/norm")
    return dict(H=H, doca=sign*norm, lsign=sign, longitudinal=longitudinal)


def metrics(actual, native):
    return {"criterion": _metric(actual, native), "exact": _metric(actual, native, exact=True)}


def information_report(q, H, r, w, context):
    W, b = A.information(q, H, r, w)
    bad = (~np.isclose(W, context["native_W"], rtol=1e-5, atol=1e-7).all(axis=(1, 2))
           | ~np.isclose(b, context["native_b"], rtol=1e-5, atol=1e-7).all(axis=1))
    return dict(weight_matrix=_metric(W, context["native_W"]), weight_vector=_metric(b, context["native_b"]),
                failing_source_shindex=context["hit_ids"][bad].tolist())


def compare_target(context, rows, line, corrected):
    """Fixed controls over every initial hit, without selecting or tuning against native W/b."""
    native = context["native"]
    rows, capH, capr, capw = join_null_rows(rows, native)
    cap = snapshot_state(rows)
    q = _dense(native, "reference_parameters", (6,))
    mass = context["native_mass"]
    require(np.array_equal(cap["particle_time"], _col(native, "time")), "particle time/effect differs")
    report = dict(n_hits=len(rows), source_shindex=context["hit_ids"].tolist(),
                  hit_keys=keys(rows, HIT_KEYS).tolist(), native_precision=NATIVE_CA_PRECISION,
                  native_reference_vs_saved=metrics(context["q"], q), captured_state={k: v.tolist() for k, v in cap.items()},
                  captured_H=capH.tolist(), captured_residual=capr.tolist(), captured_weight=capw.tolist(),
                  saved_vs_captured={"H": metrics(context["actual_H"], capH), "residual": metrics(context["residual"], capr),
                                     "weight": metrics(context["weight"], capw)}, variants={})
    for label, ref, precision in (("native_reference_legacy_precision", q, 1e-4),
                                  ("native_reference_native_precision", q, NATIVE_CA_PRECISION),
                                  ("saved_f1_reference_native_precision", context["q"], NATIVE_CA_PRECISION)):
        hints = C.constructor_hint(ref, line, corrected, mass)
        built = C.solve(ref, line, *hints, mass=mass, precision=precision)
        state = C.solve(ref, line, built["particle_time"], built["sensor_time"], mass=mass, precision=precision)
        generated = fixed_state_rows(ref, line, state, mass)
        geom = {field: metrics(state[field], cap[field]) for field in cap}
        item = dict(scope="local constructor/update using measured inputs and native mass; reference origin named explicitly",
                    geometry=geom, state={k: state[k].tolist() for k in cap}, H=generated["H"].tolist(),
                    H_vs_captured=metrics(generated["H"], capH), doca_vs_captured=metrics(generated["doca"], _col(rows, "doca")),
                    constructor_updates=built["updates"].tolist(), reference_updates=state["updates"].tolist(),
                    information_saved_values_weights=information_report(ref, generated["H"], context["residual"], context["weight"], context),
                    information_captured_values_weights=information_report(ref, generated["H"], capr, capw, context))
        report["variants"][label] = item
        if label == "native_reference_native_precision":
            local, localH = state, generated["H"]
    controls = {}
    for label, particle, sensor, directions, times in (
            ("captured_times_recomputed_positions", False, False, False, True),
            ("captured_particle_poca_only", True, False, False, False),
            ("captured_sensor_poca_only", False, True, False, False),
            ("captured_both_pocas", True, True, False, False),
            ("captured_complete_state", True, True, True, True)):
        state = dict(local)
        if times:
            state.update(particle_time=cap["particle_time"], sensor_time=cap["sensor_time"])
            state["particle_poca"] = C.particle(q, state["particle_time"], mass)[0]
            state["sensor_poca"] = C.sensor_position(line, state["sensor_time"])
        if particle:
            state["particle_poca"] = cap["particle_poca"]
        if sensor:
            state["sensor_poca"] = cap["sensor_poca"]
        if directions:
            state.update(direction=cap["direction"], sensor_direction=cap["sensor_direction"])
        state["delta"] = state["sensor_poca"]-state["particle_poca"]
        H = fixed_state_rows(q, line, state, mass)["H"]
        controls[label] = dict(H_vs_captured=metrics(H, capH), H=H.tolist(),
                              information=information_report(q, H, capr, capw, context))
    H = localH.copy()
    H[:, 0] = capH[:, 0]
    for label, H in (("captured_dDdP_only", H), ("captured_all_rows", capH)):
        controls[label] = dict(H_vs_captured=metrics(H, capH),
                              information=information_report(q, H, capr, capw, context))
    report["captured_operand_controls"] = controls
    report["captured_rows_saved_values_weights"] = information_report(q, capH, context["residual"], context["weight"], context)
    report["saved_rows_captured_values_weights"] = information_report(q, context["actual_H"], capr, capw, context)
    report["reconstructed_sensor_direction_vs_captured"] = metrics(line["direction"], cap["sensor_direction"])
    return report


def source_speed_configuration(path):
    """Read only the simple expanded strawResponse table; reject unsupported calibration modes."""
    import re
    text = Path(path).read_text()
    starts = list(re.finditer(r"\bstrawResponse\s*:\s*\{", text))
    require(len(starts) == 1, "ambiguous expanded strawResponse table")
    begin = starts[0].end()
    end = text.find("}", begin)
    require(end >= begin and "{" not in text[begin:end], "nested/unterminated strawResponse table unsupported")
    block = text[begin:end]

    def scalar(name):
        values = re.findall(r"(?m)^\s*"+name+r"\s*:\s*([^\s]+)\s*$", block)
        require(len(values) == 1, "ambiguous strawResponse scalar " + name)
        return values[0]

    def sequence(name):
        values = re.findall(r"\b"+name+r"\s*:\s*\[([^\]]*)\]", block)
        require(len(values) == 1, "ambiguous strawResponse sequence " + name)
        return np.array([float(v.strip()) for v in values[0].split(",") if v.strip()])

    require(scalar("useDb") == "false", "database strawResponse speed unsupported")
    require(len(sequence("strawHalfPropVelocity")) == 0, "per-straw speed overrides unsupported")
    width, mean, bins = float(scalar("eBinWidth")), float(scalar("defaultHalfPropVelocity")), int(scalar("eBins"))
    scale = sequence("eHalfPVScale")
    require(bins >= 2 and len(scale) == bins and np.isfinite(scale).all(), "invalid speed grid")
    require(np.isfinite(width) and width > 0 and np.isfinite(mean) and mean > 0, "invalid speed constants")
    return dict(path=str(path), sha256=digest(path), bin_width=width, mean_half_velocity=mean, scale=scale.tolist())


@A._checked
def source_signal_speed(energy, config):
    """Native PieceLine slope-first arithmetic at measured ComboHit energy, without response inversion."""
    energy = np.asarray(energy, dtype=float)
    require(energy.ndim == 1 and np.isfinite(energy).all() and np.all(energy >= 0), "invalid measured energy")
    scale = np.asarray(config["scale"], dtype=float)
    grid = np.arange(len(scale), dtype=float)*config["bin_width"]
    width = (grid[-1]-grid[0])/(len(grid)-1)
    x = 1000.*energy
    ib = np.clip(np.floor((x-grid[0])/width), 0, len(grid)-2).astype(np.int64)
    slope = (scale[ib+1]-scale[ib])/width
    interpolated = scale[ib]+(x-grid[ib])*slope
    speed = 2.*(interpolated*config["mean_half_velocity"])
    require(np.isfinite(speed) & (speed > 0), "invalid source signal speed")
    return speed


def speed_controls(context, rows, line, corrected, energy, sid, cond, config):
    """Measured-energy/configuration controls, never native-POCA-fitted speed corrections."""
    rows, capH, capr, capw = join_null_rows(rows, context["native"])
    cap = snapshot_state(rows)
    q = _dense(context["native"], "reference_parameters", (6,))
    us = hitreco.unique_straw(sid)
    grid = np.arange(len(cond["halfvpscale"]))*hitreco.EBIN
    from_dump = 2.*hitreco._pieceline(grid, cond["halfvpscale"], 1000.*energy)*cond["straw"]["halfvp"][us]
    from_source = source_signal_speed(energy, config)
    source_grid = np.arange(len(config["scale"]))*config["bin_width"]
    source_legacy_group = 2.*hitreco._pieceline(source_grid, np.asarray(config["scale"]), 1000.*energy)*config["mean_half_velocity"]
    normalized_config = dict(config, scale=cond["halfvpscale"], bin_width=hitreco.EBIN, mean_half_velocity=1.)
    normalized_source_group = source_signal_speed(energy, normalized_config)*cond["straw"]["halfvp"][us]
    out = dict(scope="measured-energy/FCL speed; captured reference/mass unless labelled saved reference; no captured CA hints or POCAs in solves",
               measured_energy_vs_loader=metrics(energy, context["h"]["edep"]), variants={})
    for label, speed, ref in (("measured_energy_dump_normalized_conditions", from_dump, q),
                              ("normalized_conditions_source_grouping", normalized_source_group, q),
                              ("source_fcl_conditions_legacy_grouping", source_legacy_group, q),
                              ("measured_energy_source_fcl_conditions", from_source, q),
                              ("source_fcl_speed_saved_f1_reference", from_source, context["q"])):
        sensor = dict(line, speed=speed)
        hints = C.constructor_hint(ref, sensor, corrected, context["native_mass"])
        built = C.solve(ref, sensor, *hints, precision=NATIVE_CA_PRECISION, mass=context["native_mass"])
        state = C.solve(ref, sensor, built["particle_time"], built["sensor_time"], precision=NATIVE_CA_PRECISION, mass=context["native_mass"])
        generated = fixed_state_rows(ref, sensor, state, context["native_mass"])
        h = context["h"]
        residual = np.stack((-generated["doca"], state["sensor_time"]-state["particle_time"]-h["totdt"],
            generated["longitudinal"]-h["wdist"]*np.where(h["esign"]*h["udir_dot_w"] < 0, -1., 1.)), -1)
        out["variants"][label] = dict(speed=speed.tolist(), speed_vs_loader=metrics(speed, line["speed"]),
            geometry={field: metrics(state[field], cap[field]) for field in cap},
            sensor_poca_delta=(state["sensor_poca"]-cap["sensor_poca"]).tolist(),
            H_vs_captured=metrics(generated["H"], capH), residual_vs_captured=metrics(residual, capr),
            information_captured_values_weights=information_report(ref, generated["H"], capr, capw, context),
            information_saved_values_weights=information_report(ref, generated["H"], context["residual"], context["weight"], context),
            information_reconstructed_values_saved_weights=information_report(ref, generated["H"], residual, context["weight"], context))
    return out


def run(args):
    with args.out.open("x") as stream:
        report = dict(completed=False, full_closure=False, full_numerical_validation=False,
                      all_captured_iterations_compared=False, limits=LIMITS, targets=[])
        watched = {}
        try:
            require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
            require(1 <= len(args.trace_dir) <= 3 and len(set(args.trace_dir)) == len(args.trace_dir), "invalid explicit trace list")
            root = Path(__file__).resolve().parents[1]
            source_files = sorted((root/"trkfastsim/reco").glob("*.py")) + [Path(__file__),
                *[root/"scripts"/f for f in ("f1_measurement_audit.py", "f1_trace_compare.py", "f1_ca_operand_audit.py",
                                             "f1_native_decision_audit.py", "f1_hit_operand_check.py", "run_f1_operand_audit.sh")]]
            watched.update({str(p.resolve()): digest(p) for p in source_files})
            report["source_sha256"] = dict(watched)
            old = {}
            iterations, pieces = load_evidence(args.check, old)
            gate, receipt_hashes = receipt(args.operand_check, args.operand_check.parent/"check.json", old["inputs"]["capture"]["path"])
            watched.update(receipt_hashes)
            new = {}
            load_evidence(Path(gate["check"]), new)
            require(old["inputs"]["baseline"] == new["inputs"]["baseline"] and old["inputs"]["targets"] == new["inputs"]["targets"],
                    "baseline/targets changed between captures")
            report.update(frozen_gate=old, operand_gate=new, sidecar_receipt=str(args.operand_check))
            for meta in (old, new):
                watched[meta["check"]] = meta["check_sha256"]
                watched.update({v["path"]: v["sha256"] for v in meta["inputs"].values()})
            speed_config = source_speed_configuration(args.check.parent/"capture.expanded.fcl")
            require(speed_config["sha256"] == FROZEN_EXPANDED_SHA256,
                    "expanded FCL differs from frozen prior-audit checkpoint")
            speed_config["provenance_scope"] = "hash pinned in ca_operands_v6; not a capture-generation hash receipt"
            report["speed_configuration"] = speed_config
            watched[speed_config["path"]] = speed_config["sha256"]
            paths = NATIVE_SOURCES + ("Offline/Mu2eKinKal/inc/KKFitSettings.hh", "Offline/Mu2eKinKal/inc/KKFit.hh",
                "Offline/Mu2eKinKal/src/KKFitUtilities.cc", "Offline/TrackerConditions/src/StrawResponse.cc",
                "Offline/TrackerConditions/src/StrawResponseMaker.cc", "Offline/TrackerConfig/inc/StrawResponseConfig.hh",
                "KinKal/Trajectory/SensorLine.cc", "KinKal/Trajectory/ConstantDistanceToTime.cc", "KinKal/Geometry/LineSegment.hh")
            report["native_source_sha256"] = {str(args.native_source/p): digest(args.native_source/p) for p in paths}
            watched.update(report["native_source_sha256"])
            report["root_source_sha256"] = root_arithmetic_sources(args.native_source)
            watched.update(report["root_source_sha256"])
            with uproot.open(old["inputs"]["capture"]["path"]) as f:
                effects = read_tree(f, "kktrace", "effects", old["highest_cycles"]["capture"])
            with uproot.open(new["inputs"]["capture"]["path"]) as f:
                rows = read_tree(f, "kktrace", "hit_operands", new["highest_cycles"]["capture"])
            require(len(rows) == gate["n_hits"], "sidecar highest-cycle count changed")
            manifests = [json.loads((d/"manifest.json").read_text()) for d in args.trace_dir]
            for d, manifest in zip(args.trace_dir, manifests):
                watched[str(d/"manifest.json")] = digest(d/"manifest.json")
                watched[str(d/"trace.npz")] = manifest["trace_sha256"]
                for label in ("conditions", "alignment"):
                    require(manifest[label] == manifests[0][label], "trace input configurations differ")
                    watched[manifest[label]["path"]] = manifest[label]["sha256"]
            cond = hitreco.load_conditions(manifests[0]["conditions"]["path"])
            _, inputs, _, ref, drift, _ = vr_f1.read_inputs(old["inputs"]["baseline"]["path"], cond, "mc",
                straw_axis=manifests[0]["alignment"]["path"], fitok_only=False)
            for directory in args.trace_dir:
                context = {}
                base = audit_one(directory, old, iterations, pieces, effects, inputs, ref, drift,
                                 args.allow_loader_reconstruction, diagnostic_context=context)
                early, late, corrected, energy, sid = endpoint_inputs(old["inputs"]["baseline"]["path"], context)
                line = A.sensor(early, late, context["h"]["tmeas"], context["h"]["vsig"])
                target = compare_target(context, rows, line, corrected)
                target["source_speed_controls"] = speed_controls(context, rows, line, corrected, energy, sid, cond, speed_config)
                target.update(key=base["key"], native_seed=base["native_seed"], trace_dir=str(directory),
                              trace_sha256=base["trace_sha256"], loader_provenance=base["loader_provenance"],
                              preconditions=base["preconditions"])
                report["targets"].append(target)
            report["completed"] = True
        except (ValueError, KeyError, OSError, TypeError, IndexError, AttributeError, OverflowError) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                end = {p: digest(p) for p in watched}
                report["hashes_unchanged"] = end == watched
                report["end_sha256"] = end
                require(end == watched, "input/source changed during audit")
            except (OSError, ValueError) as exc:
                report.update(completed=False, integrity_error=str(exc))
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--operand-check", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-loader-reconstruction", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({k: v for k, v in result.items() if k not in ("targets", "end_sha256")}, indent=2))
    for t in result["targets"]:
        print(json.dumps({"key": t["key"], "native_seed": t["native_seed"], "controls": {
            name: {kind: value["information"][kind]["mismatches"] for kind in ("weight_matrix", "weight_vector")}
            for name, value in t["captured_operand_controls"].items()}}))
    raise SystemExit(0 if result["completed"] else 1)
