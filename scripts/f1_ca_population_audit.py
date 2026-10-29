"""Vectorized native-reference first-iteration CA census; no fit or inferred native rows."""
import argparse
import json
import os
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

from f1_native_decision_audit import (load_evidence, read_tree, keys, match_rows, dense,
                                     column, digest, require, KEYS, ITER_KEYS)
from f1_trace_compare import _metric
from trkfastsim.reco import nativeca as C, hitreco


LIMITS = [
    "Captured-reference and captured-mass local CA control, not F1-generated operands or fit closure.",
    "All captured first-iteration straw hits are retained in the census; no later history alignment.",
    "Constructor seed then retained-hint reference update, without full piece search or hit-z retry.",
    "Unusable and z-retry lanes are reported, never relabelled native retries or silently dropped.",
    "Native sensor TOCA, POCA vectors, individual rows/variances and arithmetic internals are absent.",
    "No aggregate information comparison without saved weights, and no rows inferred from native W.",
    "No new fit, native capture, production policy, threshold or tolerance changes.",
]
NATIVE_PATHS = (
    "KinKal/Trajectory/KinematicLine.hh", "KinKal/Trajectory/KinematicLine.cc",
    "KinKal/Trajectory/ClosestApproach.hh", "KinKal/Trajectory/ClosestApproachData.hh",
    "KinKal/Trajectory/SensorLine.cc", "KinKal/Trajectory/SensorLine.hh",
    "KinKal/Trajectory/ConstantDistanceToTime.hh", "KinKal/Trajectory/ConstantDistanceToTime.cc",
    "KinKal/Geometry/LineSegment.hh", "KinKal/Geometry/Ray.hh",
    "Offline/Mu2eKinKal/inc/KKFit.hh", "Offline/Mu2eKinKal/inc/KKStrawHit.hh",
    "Offline/Mu2eKinKal/src/KKFitUtilities.cc", "TrkFastSim/inc/IterationCaptureWriter.hh",
)


def hit_inputs(events, geometry, hit_keys, raw_sid, conditions):
    """Full event/source-index joins into jagged measured hit arrays, never event loops."""
    require(hit_keys.ndim == 2 and hit_keys.shape[1] == 4 and hit_keys.dtype.kind in "iu",
            "invalid event/source-hit keys")
    require(np.all(hit_keys >= 0), "negative source-hit identity")
    event = match_rows(hit_keys[:, :3], keys(events, KEYS[:3]))
    counts = ak.to_numpy(ak.num(events.s_sid))
    require(hit_keys[:, 3] < counts[event], "source-hit index outside event")
    offsets = np.r_[0, np.cumsum(counts)]
    flat = offsets[event]+hit_keys[:, 3]
    data = {}
    for field in ("sid", "time", "et0", "et1", "edep"):
        require(np.array_equal(ak.to_numpy(ak.num(events["s_"+field])), counts), "measured hit lengths differ")
        data[field] = ak.to_numpy(ak.flatten(events["s_"+field]))[flat]
    require(np.array_equal(data["sid"], raw_sid), "native/source raw straw differs")
    sid = np.asarray(geometry["sid"][0], dtype=np.int64)
    where = match_rows(np.asarray(raw_sid, dtype=np.int64)[:, None], sid[:, None])
    cal = np.column_stack([geometry[k][0][where] for k in ("wcalx", "wcaly", "wcalz")])
    hv = np.column_stack([geometry[k][0][where] for k in ("whvx", "whvy", "whvz")])
    require(np.isfinite(data["et0"]) & np.isfinite(data["et1"]), "nonfinite measured endpoint time")
    hv_early = data["et1"] < data["et0"]
    early, late = np.where(hv_early[:, None], hv, cal), np.where(hv_early[:, None], cal, hv)
    energy = data["edep"].astype(float)
    us = hitreco.unique_straw(np.asarray(raw_sid, dtype=np.int64))
    halfvp = np.asarray(conditions["straw"]["halfvp"])
    require((us >= 0) & (us < len(halfvp)), "straw conditions index outside table")
    scale = np.asarray(conditions["halfvpscale"])
    speed = 2*hitreco._pieceline(np.arange(len(scale))*hitreco.EBIN, scale, 1000.*energy)*halfvp[us]
    measurement = np.where(hv_early, data["et1"], data["et0"]).astype(float)
    line = C.sensor(early, late, measurement, speed)
    corrected = data["time"].astype(float)
    require(np.isfinite(corrected) & np.isfinite(energy), "nonfinite measured hit")
    return line, corrected


def census(qseed, qref, mass, line, corrected, native_time, constructor_time, identities):
    """Run two masked local solves across the entire hit population."""
    ph, sh = C.constructor_hint(qseed, line, corrected, mass)
    built = C.solve(qseed, line, ph, sh, mass=mass)
    state = C.solve(qref, line, built["particle_time"], built["sensor_time"], mass=mass)
    dz = state["particle_poca"][:, 2]-line["mid"][:, 2]
    retry = ~state["usable"] | (np.abs(dz) > 100.)
    distance = np.linalg.norm(state["delta"], axis=-1)
    exact = state["particle_time"] == native_time
    close = np.isclose(state["particle_time"], native_time, rtol=1e-5, atol=1e-7)
    delta = state["particle_time"]-native_time
    return {"n_hits": len(qref), "n_targets": len(np.unique(identities[:, :4], axis=0)),
        "constructor_time": _metric(built["particle_time"], constructor_time),
        "constructor_time_exact": _metric(built["particle_time"], constructor_time, exact=True),
        "updated_time": _metric(state["particle_time"], native_time),
        "updated_time_exact": _metric(state["particle_time"], native_time, exact=True),
        "native_seed_vs_reference": _metric(qseed, qref, exact=True),
        "identities": identities.tolist(), "identity_fields": [*KEYS, "shindex"],
        "native_time": native_time.tolist(), "particle_time": state["particle_time"].tolist(),
        "sensor_time_reconstructed_only": state["sensor_time"].tolist(),
        "time_delta_ns": delta.tolist(), "native_time_ulp_ns": np.abs(np.spacing(native_time)).tolist(),
        "exact_mismatch_keys": identities[~exact].tolist(), "tolerant_mismatch_keys": identities[~close].tolist(),
        "constructor_updates": built["updates"].tolist(), "reference_updates": state["updates"].tolist(),
        "constructor_status": built["status"].tolist(), "reference_status": state["status"].tolist(),
        "unusable_keys": identities[~state["usable"]].tolist(),
        "reconstructed_z_retry_keys": identities[retry].tolist(), "z_distance_mm": dz.tolist(),
        "distance_mm": distance.tolist(), "zero_distance_keys": identities[distance == 0].tolist(),
        "max_abs_time_delta_ns": float(np.max(np.abs(delta))) if len(delta) else None,
        "max_abs_z_distance_mm": float(np.max(np.abs(dz))) if len(dz) else None,
        "time_delta_quantiles_ns": np.quantile(delta, [0, .16, .5, .84, 1]).tolist() if len(delta) else [],
        "delta_dot_particle_mm": np.sum(state["delta"]*state["direction"], -1).tolist(),
        "delta_dot_sensor_mm": np.sum(state["delta"]*line["direction"], -1).tolist()}


def select_initial(iterations, pieces, effects, seeds):
    """Require one old piece and exact constructor membership for every captured target."""
    first = iterations[column(iterations, "iteration_ordinal") == 0]
    require(len(first) > 0, "no captured first iterations")
    target_keys = keys(first)
    require(column(first, "iter") == 0, "first ordinal is not initial iteration")
    require(column(first, "capture_valid") & column(first, "effects_captured"), "invalid initial capture")
    require(column(first, "nold") == 1, "initial owner is not one piece")
    old = pieces[(column(pieces, "iteration_ordinal") == 0) & (column(pieces, "owner") == 0)]
    old = old[match_rows(target_keys, keys(old))]
    require(column(old, "piece_index") == 0, "initial piece index differs")
    seed = seeds[match_rows(target_keys, keys(seeds))]
    require(column(seed, "valid"), "invalid constructor seed")
    qseed, qold = dense(seed, "parameters", 6), dense(old, "parameters", 6)
    require(np.array_equal(qseed, qold), "initial old parameters differ from constructor")
    hits = effects[(column(effects, "iteration_ordinal") == 0) & (column(effects, "identity_kind") == 1)]
    require(len(hits) > 0, "no initial straw hits")
    require(column(hits, "identity_valid") & column(hits, "has_reference") & column(hits, "payload_supported"),
            "unsupported initial straw hit")
    hkeys = keys(hits, KEYS+("shindex",))
    parent = match_rows(hkeys[:, :4], target_keys)
    require(column(hits, "iter") == 0, "hit initial iteration differs")
    expected_count = ak.to_numpy(ak.num(seed.sh_shindex))
    require(np.array_equal(expected_count, np.bincount(parent, minlength=len(seed))), "initial hit census differs")
    constructor_keys = np.column_stack([np.repeat(target_keys, expected_count, axis=0),
                                        ak.to_numpy(ak.flatten(seed.sh_shindex))])
    loc = match_rows(hkeys, constructor_keys)
    require(np.array_equal(column(hits, "strawid"), ak.to_numpy(ak.flatten(seed.sh_strawid))[loc]),
            "constructor raw straw differs")
    qref = dense(hits, "reference_parameters", 6)
    require(np.array_equal(qref, qold[parent]), "initial reference differs from old piece")
    mass = column(old, "mass")[parent]
    require(np.array_equal(mass, column(seed, "mass")[parent]), "constructor/reference mass differs")
    return hits, qseed[parent], qref, mass, ak.to_numpy(ak.flatten(seed.sh_time))[loc], hkeys


def run(args):
    require(not args.out.exists(), "refusing existing CA population output")
    require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/"scripts/f1_native_decision_audit.py", root/"scripts/f1_trace_compare.py"]
    sources += sorted((root/"trkfastsim/reco").glob("*.py"))
    native = [args.native_source/p for p in NATIVE_PATHS]
    hashes = lambda paths: {str(p): digest(p) for p in paths}
    report = {"schema_version": 1, "full_closure": False, "full_numerical_validation": False,
              "f1_generated_operands": False, "limits": LIMITS,
              "source": hashes(sources), "native_source": hashes(native),
              "conditions": {"path": str(args.conditions), "sha256": digest(args.conditions)}}
    expanded = args.check.parent/"capture.expanded.fcl"
    precision = re.findall(r"\bTPOCAPrecision\s*:\s*([0-9eE.+-]+)", expanded.read_text())
    require(precision and all(float(v) == 1e-4 for v in precision), "unsupported native CA precision")
    report["expanded_configuration"] = {"path": str(expanded), "sha256": digest(expanded),
        "TPOCAPrecision_values": precision, "scope": "precision only, not full FHiCL equivalence"}
    iterations, pieces = load_evidence(args.check, report)
    with uproot.open(report["inputs"]["capture"]["path"]) as f:
        effects = read_tree(f, "kktrace", "effects", report["highest_cycles"]["capture"])
        seeds = read_tree(f, "kktrace", "seed_contract", report["highest_cycles"]["capture"])
    hits, qseed, qref, mass, ctime, hkeys = select_initial(iterations, pieces, effects, seeds)
    with uproot.open(report["inputs"]["baseline"]["path"]) as f:
        events = f["trkdump/events"].arrays([*KEYS[:3], "s_sid", "s_time", "s_et0", "s_et1", "s_edep"])
        geometry = f["trkdump/conditions"].arrays(["sid", "wcalx", "wcaly", "wcalz", "whvx", "whvy", "whvz"], library="np")
    cond = hitreco.load_conditions(args.conditions)
    line, corrected = hit_inputs(events, geometry, hkeys[:, [0, 1, 2, 4]], column(hits, "strawid"), cond)
    report["census"] = census(qseed, qref, mass, line, corrected, column(hits, "time"), ctime, hkeys)
    report["captured_first_iteration_targets"] = np.unique(hkeys[:, :4], axis=0).tolist()
    require(len(report["captured_first_iteration_targets"]) == len(np.unique(keys(iterations, unique=False), axis=0)),
            "first-iteration hit target coverage differs")
    require(report["source"] == hashes(sources) and report["native_source"] == hashes(native), "source changed during audit")
    require(report["conditions"]["sha256"] == digest(args.conditions) and report["check_sha256"] == digest(args.check)
            and report["expanded_configuration"]["sha256"] == digest(expanded), "inputs changed during audit")
    report["completed"] = True
    with args.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    c = report["census"]
    print(json.dumps({k: c[k] for k in ("n_hits", "n_targets", "constructor_time_exact", "updated_time_exact",
        "updated_time", "max_abs_time_delta_ns", "max_abs_z_distance_mm")}, sort_keys=True))
    print(json.dumps({"unusable": len(c["unusable_keys"]), "z_retry_triggers": len(c["reconstructed_z_retry_keys"]),
                      "zero_distance": len(c["zero_distance_keys"])}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--native-source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
