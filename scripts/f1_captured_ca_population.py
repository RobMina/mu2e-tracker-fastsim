"""ALL147 initial-hit captured CA audit; measured-input reconstruction, never a fit."""
import argparse
import json
import os
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

from f1_native_decision_audit import (KEYS, array, column, digest, keys, load_evidence,
                                     match_rows, read_tree, require)
from f1_ca_population_audit import NATIVE_PATHS, hit_inputs, select_initial
from f1_captured_ca_audit import (A, C, NATIVE_CA_PRECISION, NATIVE_SOURCES, HIT_KEYS, FROZEN_EXPANDED_SHA256,
                                  fixed_state_rows, join_null_rows, receipt,
                                  root_arithmetic_sources, snapshot_state,
                                  source_signal_speed, source_speed_configuration)
from f1_trace_compare import _dense, _metric
from trkfastsim.reco import hitreco

RTOL, ATOL = 1e-5, 1e-7
EXPECTED_TARGETS, EXPECTED_HITS = 147, 2538
SPEED_CONTROLS = ("measured_energy_dump_normalized_conditions", "measured_energy_source_fcl_conditions")
STATE_FIELDS = ("particle_time", "sensor_time", "particle_poca", "sensor_poca", "delta",
                "direction", "sensor_direction", "usable")
INFORMATION_SCOPE = ("Generated H with captured native reference, residual values and measurement weights; "
                     "NOT independent residual, variance, information-operand or fit closure.")
LIMITS = [
    "All 147 frozen initial targets / 2538 straw hits; exact constructor/reference/single-piece membership, no success selection.",
    "Native seed/reference and mass are captured inputs; generated CA vectors and H are not F1-generated fit operands.",
    "Two predeclared measured-energy speed controls only: dumped-normalized conditions and original expanded FCL.",
    "Constructor seed solve then retained-hint reference solve at float32(1e-4) promoted to double; no captured CA hints.",
    "Local single-piece CA only; no hit-z retry, fallback, piece search, refit, build or native capture.",
    INFORMATION_SCOPE,
    "Captured complete-state rows are a labelled diagnostic, not independent CA reconstruction.",
    "NumPy source-grouped arithmetic does not certify ROOT/SMatrix/compiler/transcendental bit parity.",
    "Expanded FCL is pinned to prior-audit hash, not an original capture-generation hash receipt.",
    "No optimization, ULP search, regularization, default/tolerance changes, masks dropping failures or full closure claim.",
    "Unsupported non-null, unusable or non-three-row hits and nonfinite arithmetic fail the entire audit closed.",
]


def endpoint_measurements(events, geometry, hit_keys, raw_sid):
    """Direct full-event/source-hit join, including BOTH endpoints; never 2*mid-end.

    Repeated wanted source identities are legal across different seed targets.
    Available event and geometry identities must be unique, even if unselected.
    """
    hit_keys, raw_sid = np.asarray(hit_keys), np.asarray(raw_sid)
    require(hit_keys.ndim == 2 and hit_keys.shape[1] == 4 and hit_keys.dtype.kind in "iu",
            "invalid event/source-hit keys")
    require(np.all(hit_keys >= 0), "negative source-hit identity")
    require(np.all(hit_keys <= np.iinfo(np.int64).max), "event/source-hit identity outside int64")
    require(raw_sid.shape == (len(hit_keys),) and raw_sid.dtype.kind in "iu"
            and np.all((raw_sid >= 0) & (raw_sid <= np.iinfo(np.int64).max)), "invalid raw straw identity")
    event = match_rows(hit_keys[:, :3], keys(events, KEYS[:3]))
    counts = ak.to_numpy(ak.num(events.s_sid))
    require(hit_keys[:, 3] < counts[event], "source-hit index outside event")
    flat = np.r_[0, np.cumsum(counts)][event]+hit_keys[:, 3]
    measured = {}
    for field in ("sid", "time", "et0", "et1", "edep"):
        require(np.array_equal(ak.to_numpy(ak.num(events["s_"+field])), counts), "measured hit lengths differ")
        measured[field] = array(ak.flatten(events["s_"+field]))[flat]
        require(np.isfinite(measured[field]), "nonfinite measured " + field)
    require(measured["sid"].dtype.kind in "iu" and np.array_equal(measured["sid"], raw_sid),
            "native/source raw straw differs")
    require(measured["edep"] >= 0, "invalid measured energy")
    fields = ("sid", "wcalx", "wcaly", "wcalz", "whvx", "whvy", "whvz")
    require(all(len(geometry[f]) == 1 for f in fields), "requires one dumped geometry row")
    sid = np.asarray(geometry["sid"][0])
    require(sid.ndim == 1 and sid.dtype.kind in "iu" and np.all(sid >= 0), "invalid geometry straw identities")
    where = match_rows(raw_sid[:, None], sid[:, None])
    for f in fields[1:]:
        require(np.asarray(geometry[f][0]).shape == sid.shape, "geometry endpoint lengths differ")
    cal = np.column_stack([geometry[f][0][where] for f in fields[1:4]]).astype(float)
    hv = np.column_stack([geometry[f][0][where] for f in fields[4:7]]).astype(float)
    require(np.isfinite(cal) & np.isfinite(hv), "nonfinite measured endpoint")
    hv_early = measured["et1"] < measured["et0"]  # Equal times choose cal, as native.
    return dict(early=np.where(hv_early[:, None], hv, cal), late=np.where(hv_early[:, None], cal, hv),
                measurement_time=np.where(hv_early, measured["et1"], measured["et0"]).astype(float),
                corrected_time=measured["time"].astype(float), energy=measured["edep"].astype(float),
                et0=measured["et0"].astype(float), et1=measured["et1"].astype(float))


def measured_sensors(events, geometry, hit_keys, raw_sid, conditions, configuration):
    measured = endpoint_measurements(events, geometry, hit_keys, raw_sid)
    # Reuse the population loader for the current dumped-normalized conditions control.
    dumped, corrected = hit_inputs(events, geometry, hit_keys, raw_sid, conditions)
    require(np.array_equal(corrected, measured["corrected_time"]), "corrected-time loader differs")
    require(np.array_equal(dumped["end"], measured["early"])
            and np.array_equal(dumped["measurement_time"], measured["measurement_time"]), "endpoint loader differs")
    speeds = (dumped["speed"], source_signal_speed(measured["energy"], configuration))
    sensors = {label: A.sensor(measured["early"], measured["late"], measured["measurement_time"], speed)
               for label, speed in zip(SPEED_CONTROLS, speeds)}
    return measured, sensors


def comparison(actual, captured, identities):
    """Bounded component examples, but complete per-hit failing full identities."""
    a, b = np.asarray(actual), np.asarray(captured)
    require(a.shape == b.shape and a.ndim >= 1 and a.shape[0] == len(identities), "comparison shape differs")
    require(np.isfinite(a) & np.isfinite(b), "nonfinite comparison operand")
    result = {}
    for label, exact in (("criterion", False), ("exact", True)):
        item = _metric(a, b, exact=exact)
        require(item["nonfinite"] == 0, "nonfinite comparison arithmetic")
        good = (a == b) if exact else np.isclose(a, b, rtol=RTOL, atol=ATOL)
        counts = (~good).reshape(len(a), -1).sum(axis=1)
        item.update(n_failing_hits=int(np.count_nonzero(counts)),
                    failing_hit_keys=identities[counts > 0].tolist(), per_hit_failed_components=counts.tolist())
        result[label] = item
    return result


def information(qref, H, residual, weight, native, identities):
    W, b = A.information(qref, H, residual, weight)
    return dict(scope=INFORMATION_SCOPE,
                weight_matrix=comparison(W, _dense(native, "weight_matrix", (6, 6)), identities),
                weight_vector=comparison(b, _dense(native, "weight_vector", (6,)), identities))


def compare_population(native, rows, qseed, qref, mass, constructor_time, sensors, corrected, output=None):
    """Two fixed batch solves per control, across EVERY selected hit; no event/hit loops."""
    out = {} if output is None else output
    identities = keys(native, HIT_KEYS)
    require(len(native) > 0, "empty initial population")
    out.update(n_hits=len(native), n_targets=len(np.unique(identities[:, :4], axis=0)), n_excluded_hits=0,
               hit_key_fields=list(HIT_KEYS), hit_keys=identities.tolist(),
               source_hit_key_fields=[*KEYS, "shindex"], source_hit_keys=keys(native, KEYS+("shindex",)).tolist(),
               strawid=column(native, "strawid").tolist(), input_index=column(native, "input_index").tolist(),
               variants={})
    require(tuple(sensors) == SPEED_CONTROLS, "requires exactly the two declared speed controls")
    rows, capH, capr, capw = join_null_rows(rows, native)
    cap = snapshot_state(rows)
    for name in STATE_FIELDS:
        require(np.isfinite(cap[name]), "nonfinite captured state: " + name)
    require(np.array_equal(cap["particle_time"], column(native, "time")), "particle time/effect differs")
    require(np.array_equal(qref, _dense(native, "reference_parameters", (6,))), "reference/effect differs")
    require(np.shape(qseed) == np.shape(qref) == (len(native), 6), "seed/reference shapes differ")
    require(np.shape(mass) == (len(native),) and np.all(np.isfinite(mass)) and np.all(mass > 0), "invalid native mass")
    out.update(captured_state={k: cap[k].tolist() for k in STATE_FIELDS}, native_mass=np.asarray(mass).tolist(),
               native_precision=NATIVE_CA_PRECISION,
               native_seed_vs_reference=comparison(qseed, qref, identities),
               captured_row_convention="H = -diag(-1,+1,+1)*native_prediction_derivative; r = diag(-1,+1,+1)*native_value",
               captured_information_control=information(qref, capH, capr, capw, native, identities))
    for label, line in sensors.items():
        item = out["variants"][label] = dict(scope="Independently reconstructed CA and H from measured endpoints/times/energy, captured seed/reference/mass.",
                                             speed=np.asarray(line["speed"]).tolist())
        hints = C.constructor_hint(qseed, line, corrected, mass)
        built = C.solve(qseed, line, *hints, precision=NATIVE_CA_PRECISION, mass=mass)
        # Reject the entire population rather than recovering/masking unusable constructor lanes.
        item.update(constructor_status=built["status"].tolist(), constructor_updates=built["updates"].tolist(),
                    constructor_unusable_hit_keys=identities[~built["usable"]].tolist())
        require(built["usable"], "unusable constructor CA; no lane exclusion or hint fallback")
        state = C.solve(qref, line, built["particle_time"], built["sensor_time"], precision=NATIVE_CA_PRECISION, mass=mass)
        item.update(reference_status=state["status"].tolist(), reference_updates=state["updates"].tolist(),
                    reference_unusable_hit_keys=identities[~state["usable"]].tolist())
        require(state["usable"], "unusable reference CA; no lane exclusion")
        item["constructor_time"] = comparison(built["particle_time"], constructor_time, identities)
        item["geometry"] = {field: comparison(state[field], cap[field], identities) for field in STATE_FIELDS}
        item["state"] = {k: state[k].tolist() for k in STATE_FIELDS}
        dz = state["particle_poca"][:, 2]-line["mid"][:, 2]
        require(np.isfinite(dz), "nonfinite z distance")
        item.update(z_distance_mm=dz.tolist(), reconstructed_z_retry_hit_keys=identities[np.abs(dz) > 100.].tolist())
        generated = fixed_state_rows(qref, line, state, mass)
        item["H_vs_captured"] = comparison(generated["H"], capH, identities)
        item["row_metrics"] = {name: comparison(generated["H"][:, i], capH[:, i], identities)
                               for i, name in enumerate(("null_doca", "tot_time", "longitudinal"))}
        item["doca_vs_captured"] = comparison(generated["doca"], column(rows, "doca"), identities)
        item["lsign_vs_captured"] = comparison(generated["lsign"], column(rows, "lsign"), identities)
        item["information_captured_values_weights"] = information(qref, generated["H"], capr, capw, native, identities)
        fixed = fixed_state_rows(qref, line, cap, mass)
        item["captured_complete_state_control"] = dict(
            scope="Captured vectors/directions/times substituted; derivative diagnostic, NOT independent CA.",
            H_vs_captured=comparison(fixed["H"], capH, identities),
            information_captured_values_weights=information(qref, fixed["H"], capr, capw, native, identities))
        item["agreement"] = {
            kind: dict(all_state=all(m[kind]["passed"] for m in item["geometry"].values()),
                       generated_H=item["H_vs_captured"][kind]["passed"],
                       information_with_captured_values_weights=all(
                           item["information_captured_values_weights"][f][kind]["passed"]
                           for f in ("weight_matrix", "weight_vector")))
            for kind in ("criterion", "exact")}
    return out


def population_guard(iterations, hkeys):
    targets = np.unique(hkeys[:, :4], axis=0)
    require(len(hkeys) == EXPECTED_HITS and len(targets) == EXPECTED_TARGETS,
            "requires frozen ALL147 / 2538 initial-hit population")
    captured = np.unique(keys(iterations, unique=False), axis=0)
    require(np.array_equal(targets, captured), "first-iteration hit target coverage differs")
    return targets


def local_sources():
    root = Path(__file__).resolve().parents[1]
    helpers = ("f1_captured_ca_population.py", "f1_captured_ca_audit.py", "f1_ca_population_audit.py",
               "f1_ca_operand_audit.py", "f1_measurement_audit.py", "f1_trace_compare.py",
               "f1_native_decision_audit.py", "f1_hit_operand_check.py", "run_f1_operand_audit.sh")
    return [root/"scripts"/f for f in helpers] + sorted((root/"trkfastsim/reco").glob("*.py"))


def native_sources(directory):
    paths = set(NATIVE_PATHS) | set(NATIVE_SOURCES) | {
        "Offline/Mu2eKinKal/inc/KKFitSettings.hh", "Offline/TrackerConditions/src/StrawResponse.cc",
        "Offline/TrackerConditions/src/StrawResponseMaker.cc", "Offline/TrackerConfig/inc/StrawResponseConfig.hh"}
    return [directory/p for p in sorted(paths)]


def run(args):
    """Exclusive JSON on success AND failure; all known input/source hashes rechecked."""
    with Path(args.out).open("x", encoding="utf-8") as stream:
        report = dict(schema_version=1, completed=False, full_closure=False, full_numerical_validation=False,
                      f1_generated_operands=False, independent_residual_variance_closure=False,
                      all_captured_iterations_compared=False, n_excluded_hits=0, rtol=RTOL, atol=ATOL,
                      limits=LIMITS, speed_controls=list(SPEED_CONTROLS), population={},
                      completion_definition="completed means supported census audited with unchanged hashes, NOT numerical agreement or closure")
        watched, old, new = {}, {}, {}

        def watch(path, expected=None):
            path = str(Path(path).resolve())
            value = digest(path) if expected is None else expected
            require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                    "invalid input/source SHA256: " + path)
            require(path not in watched or watched[path] == value, "conflicting input/source hash: " + path)
            watched[path] = value
            return value

        try:
            report["source_sha256"] = {str(p.resolve()): watch(p) for p in local_sources()}
            for p in (args.check, args.operand_check, args.conditions):
                watch(p)
            require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
            iterations, pieces = load_evidence(args.check, old)
            report["frozen_gate"] = old
            # Track the receipt-declared files even if its validation fails partway.
            # These declarations are not trusted evidence until receipt() succeeds.
            declaration = json.loads(args.operand_check.read_text())
            for p, sha in declaration.get("end_sha256", {}).items():
                watch(p, sha)
            # The receipt itself binds the new gate to exact old trees and the frozen capture hash.
            gate, receipt_hashes = receipt(args.operand_check, args.operand_check.parent/"check.json",
                                           old["inputs"]["capture"]["path"])
            for p, sha in receipt_hashes.items():
                watch(p, sha)
            report["sidecar_receipt"] = dict(path=str(args.operand_check.resolve()), sha256=digest(args.operand_check))
            load_evidence(Path(gate["check"]), new)
            report["operand_gate"] = new
            require(old["inputs"]["baseline"] == new["inputs"]["baseline"]
                    and old["inputs"]["targets"] == new["inputs"]["targets"], "baseline/targets changed between captures")
            expanded = args.check.parent/"capture.expanded.fcl"
            watch(expanded)
            precision = re.findall(r"\bTPOCAPrecision\s*:\s*([0-9eE.+-]+)", expanded.read_text())
            require(precision and all(float(v) == 1e-4 for v in precision), "unsupported native CA precision")
            config = source_speed_configuration(expanded)
            require(config["sha256"] == FROZEN_EXPANDED_SHA256, "expanded FCL differs from frozen prior-audit checkpoint")
            config["provenance_scope"] = "hash pinned in ca_operands_v6; not a capture-generation hash receipt"
            report["speed_configuration"] = config
            report["precision_configuration"] = dict(FCL_values=precision, promoted=NATIVE_CA_PRECISION)
            report["conditions"] = dict(path=str(args.conditions.resolve()), sha256=digest(args.conditions))
            report["native_source_sha256"] = {str(p.resolve()): watch(p) for p in native_sources(args.native_source)}
            report["root_source_sha256"] = root_arithmetic_sources(args.native_source)
            for p, sha in report["root_source_sha256"].items():
                watch(p, sha)
            with uproot.open(old["inputs"]["capture"]["path"]) as f:
                effects = read_tree(f, "kktrace", "effects", old["highest_cycles"]["capture"])
                seeds = read_tree(f, "kktrace", "seed_contract", old["highest_cycles"]["capture"])
            hits, qseed, qref, mass, ctime, hkeys = select_initial(iterations, pieces, effects, seeds)
            report["population"].update(hit_key_fields=list(HIT_KEYS), hit_keys=keys(hits, HIT_KEYS).tolist(),
                                        n_hits=len(hits), n_targets=len(np.unique(hkeys[:, :4], axis=0)))
            report["initial_target_keys"] = population_guard(iterations, hkeys).tolist()
            with uproot.open(new["inputs"]["capture"]["path"]) as f:
                rows = read_tree(f, "kktrace", "hit_operands", new["highest_cycles"]["capture"])
            require(len(rows) == gate["n_hits"], "sidecar highest-cycle count changed")
            with uproot.open(old["inputs"]["baseline"]["path"]) as f:
                events = read_tree(f, "trkdump", "events", old["highest_cycles"]["baseline"],
                                   [*KEYS[:3], "s_sid", "s_time", "s_et0", "s_et1", "s_edep"])
                geometry = read_tree(f, "trkdump", "conditions", old["highest_cycles"]["baseline"],
                                     ["sid", "wcalx", "wcaly", "wcalz", "whvx", "whvy", "whvz"])
                geometry = {name: ak.to_numpy(geometry[name]) for name in ak.fields(geometry)}
            cond = hitreco.load_conditions(args.conditions)
            measured, sensors = measured_sensors(events, geometry, hkeys[:, [0, 1, 2, 4]], column(hits, "strawid"), cond, config)
            report["measured_inputs"] = {k: v.tolist() for k, v in measured.items()}
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                compare_population(hits, rows, qseed, qref, mass, ctime, sensors,
                                   measured["corrected_time"], report["population"])
            report["completed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # Preserve gate-declared hashes even when validation stopped midway.
            integrity_errors = []
            for label, meta in (("frozen_gate", old), ("operand_gate", new)):
                if meta:
                    report[label] = meta
                try:
                    if "check_sha256" in meta:
                        watch(meta["check"], meta["check_sha256"])
                    for source in meta.get("inputs", {}).values():
                        watch(source["path"], source["sha256"])
                except Exception as exc:
                    integrity_errors.append(f"{type(exc).__name__}: {exc}")
            end = {}
            for p in watched:
                try:
                    end[p] = digest(p)
                except OSError:
                    end[p] = None
            if end != watched:
                integrity_errors.append("input/source changed during audit")
            report.update(before_sha256=dict(watched), end_sha256=end, hashes_unchanged=not integrity_errors)
            if integrity_errors:
                report.update(completed=False, integrity_error="; ".join(integrity_errors))
            try:
                payload = json.dumps(report, indent=2, allow_nan=False)
            except (ValueError, TypeError) as exc:
                # Never emit NaN/Infinity, even if malformed metadata caused the failure.
                report = dict(completed=False, full_closure=False, error="invalid JSON report: " + str(exc),
                              before_sha256=watched, limits=LIMITS)
                payload = json.dumps(report, indent=2, allow_nan=False)
            stream.write(payload+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("check", "operand-check", "conditions", "native-source", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report[k] for k in ("completed", "error", "integrity_error", "hashes_unchanged") if k in report}, allow_nan=False))
    raise SystemExit(0 if report["completed"] else 1)


if __name__ == "__main__":
    main()
