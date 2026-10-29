"""Initial residual/measurement-variance reconstruction; captured reference and null-state scope."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import awkward as ak
import numpy as np
import uproot

import f1_captured_ca_population as P
from f1_captured_ca_audit import (A, C, HIT_KEYS, NATIVE_CA_PRECISION, FROZEN_EXPANDED_SHA256,
                                  fixed_state_rows, join_null_rows, snapshot_state)
from f1_native_decision_audit import (KEYS, array, column, digest, keys, match_rows, read_tree, require)
from f1_trace_compare import _dense
from f1_initial_measurements import response_configuration, reconstruct

CA_REPORT_SHA256 = "ef6261c648b7c992319761c15b2ef63e5bafc18e5c4ee6d43bc2aad63e444449"
RTOL, ATOL = 1e-5, 1e-7
EVENT_FIELDS = ("sid", "time", "et0", "et1", "edep", "tot0", "tot1", "dtime", "wdist", "timevar", "uvar", "eend")
GEOMETRY_FIELDS = ("sid", "wcalx", "wcaly", "wcalz", "whvx", "whvy", "whvz", "dirx", "diry", "dirz", "hlen")
NATIVE_PATHS = (
    "Offline/Mu2eKinKal/inc/KKStrawHit.hh", "Offline/Mu2eKinKal/inc/WireHitState.hh",
    "Offline/Mu2eKinKal/inc/KKSHFlag.hh", "Offline/Mu2eKinKal/inc/CADSHU.hh",
    "Offline/Mu2eKinKal/src/CADSHU.cc", "Offline/Mu2eKinKal/inc/KKFitSettings.hh",
    "Offline/TrackerConditions/inc/DriftInfo.hh", "Offline/TrackerConditions/src/DriftInfo.cc",
    "Offline/TrackerConditions/src/StrawResponse.cc", "Offline/TrackerConditions/src/StrawResponseMaker.cc",
    "Offline/TrkHitReco/src/StrawHitRecoUtils.cc", "Offline/RecoDataProducts/inc/ComboHit.hh",
    "Offline/DataProducts/inc/TrkTypes.hh", "KinKal/Fit/MetaIterConfig.hh",
    "KinKal/Detector/Hit.hh", "KinKal/Detector/ResidualHit.hh", "KinKal/Detector/Residual.hh",
    "TrkFastSim/src/StrawHitTableDumper_module.cc")
LIMITS = [
    "All 147 initial targets / 2538 hits; no success selection or exclusions, no later iterations.",
    "Captured seed/reference parameters and mass remain inputs; not F1-generated reference or full fit closure.",
    "Null/TOT/LongVal branch is prescribed from source configuration and verified against capture, NOT independently selected.",
    "CAD state eligibility uses unbiased CA and DriftInfo; those operands are not reconstructed by this audit.",
    "Two predeclared measurement arms: independently dumped ComboHit measurements and source-FCL regeneration from measured energy/end times/TOT.",
    "Neither arm consumes sidecar residual values or measurement variances to generate residuals or weights.",
    "Wire direction is the independently dumped straw direction cast to ComboHit float32, not captured CA direction.",
    "Full source calibration regeneration does not regenerate raw electronics energy or endpoint times.",
    "Expanded FCL is pinned to the prior audit, not an original capture-generation hash receipt.",
    "NumPy source arithmetic is not universal ROOT/compiler parity; unchanged rtol=1e-5, atol=1e-7.",
    "No fit/build/native capture, parameter-variance denominators, regularization or default changes.",
]


def verified_ca_receipt(path, watched=None):
    watched = {} if watched is None else watched
    watched[str(Path(path).resolve())] = CA_REPORT_SHA256
    require(digest(path) == CA_REPORT_SHA256, "captured CA report hash differs")
    data = json.loads(Path(path).read_text())
    declarations = data.get("before_sha256")
    if isinstance(declarations, dict):
        watched.update(declarations)
    require(data.get("completed") is True and data.get("hashes_unchanged") is True, "incomplete CA receipt")
    before, end = data.get("before_sha256"), data.get("end_sha256")
    require(isinstance(before, dict) and before and before == end, "missing/unequal CA hash manifest")
    require(data.get("population", {}).get("n_hits") == 2538
            and data["population"].get("n_targets") == 147
            and data["population"].get("n_excluded_hits") == 0, "CA receipt census differs")
    require(data.get("rtol") == RTOL and data.get("atol") == ATOL, "CA receipt tolerances differ")
    for p, sha in before.items():
        require(isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) is not None,
                "invalid receipt SHA256")
        require(digest(p) == sha, "prior CA input/source changed: " + p)
    return data, dict(before) | {str(Path(path).resolve()): digest(path)}


def executing_sources(watched, modules=None):
    """Bind executing inherited modules to receipt paths; track newly imported core helpers."""
    modules = sys.modules if modules is None else modules
    result = {}
    inherited = {Path(p).name for p in watched if p.endswith(".py")}
    for name, module in list(modules.items()):
        if not (name.startswith("f1_") or name.startswith("trkfastsim.reco.") or name.startswith("mu2efastsim")):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or not filename.endswith(".py"):
            continue
        path = str(Path(filename).resolve())
        sha = digest(path)
        if not name.startswith("mu2efastsim") and Path(path).name in inherited:
            require(path in watched and watched[path] == sha, "executing inherited module differs: " + path)
        watched[path] = sha
        result[name] = dict(path=path, sha256=sha)
    return result


def initial_policy(path, drift_source):
    """Parse the identical expanded FitSettings blocks; no extension-policy substitution."""
    text = Path(path).read_text()
    blocks = re.findall(r"(?ms)^([ ]*)FitSettings\s*:\s*\{(.*?)^\1\}", text)
    require(bool(blocks) and len({b for _, b in blocks}) == 1, "ambiguous initial FitSettings")
    block = blocks[0][1]
    cad = re.findall(r"CADSHUSettings\s*:\s*\[\s*\[([^\]]*)\]", block)
    meta = re.findall(r"MetaIterationSettings\s*:\s*\[\s*\[([^\]]*)\]", block)
    require(len(cad) == len(meta) == 1, "missing initial updater/schedule")
    config, schedule = json.loads("["+cad[0]+"]"), json.loads("["+meta[0]+"]")
    require(config == [400, 800000, 50, 50, "TOT:LongVal", "Null:Inactive", "", 0],
            "unsupported initial CADSHU policy")
    require(schedule == [5, "CADSHU"], "unsupported initial schedule")
    require(not re.search(r"(?:BkgANNSHU|DriftANNSHU|PanelDiagSHU|Chi2SHU)Settings", block),
            "additional initial state updater unsupported")
    source = Path(drift_source).read_text()
    require(re.search(r"double\s+DriftInfo::maxdvar_\s*\(2\.4\*2\.4/3\.0\)", source) is not None,
            "null distance variance source changed")
    temp = float(np.float32(schedule[0]))
    return dict(cad=config, temperature=temp, variance_scale=(1.+temp)*(1.+temp),
                distance_variance=2.4*2.4/3.0, hit_state=0, constrain_tot=True, constrain_long=True,
                constrain_absdrift=False, null_drift_variance=False,
                state_eligibility_reconstructed=False,
                scope="Conditional null-row branch; unbiased CAD eligibility remains untested")


def measured_inputs(events, geometry, hkeys, raw_sid):
    """Join full event/source keys and independent measured ComboHit fields in one batch."""
    geometry = {k: array(v) for k, v in geometry.items()}
    measured = P.endpoint_measurements(events, geometry, hkeys, raw_sid)
    parent = match_rows(hkeys[:, :3], keys(events, KEYS[:3]))
    counts = ak.to_numpy(ak.num(events.s_sid))
    flat = np.r_[0, np.cumsum(counts)][parent]+hkeys[:, 3]
    for field in EVENT_FIELDS:
        require(np.array_equal(ak.to_numpy(ak.num(events["s_"+field])), counts), "measured lengths differ: " + field)
        values = array(ak.flatten(events["s_"+field]))[flat]
        require(np.isfinite(values), "nonfinite measured " + field)
        if field not in ("sid", "eend"):
            require(values.dtype == np.dtype("float32"), "ComboHit field is not float32: " + field)
        measured[field] = values.astype(float)
    require(np.isin(measured["eend"], [0, 1]), "invalid measured early end")
    require(measured["timevar"] > 0, "nonpositive measured time variance")
    require(measured["uvar"] > 0, "nonpositive measured longitudinal variance")
    sid = np.asarray(geometry["sid"][0])
    index = match_rows(raw_sid[:, None], sid[:, None])
    for field in ("dirx", "diry", "dirz", "hlen"):
        require(len(geometry[field]) == 1 and np.shape(geometry[field][0]) == sid.shape, "geometry shape differs")
    direction = np.column_stack([geometry[f][0][index] for f in ("dirx", "diry", "dirz")]).astype(float)
    hlen = np.asarray(geometry["hlen"][0][index], dtype=float)
    require(np.isfinite(direction), "nonfinite straw direction")
    require(np.abs(A._dot3(direction, direction)-1.) < 1e-10, "nonunit straw direction")
    require(np.isfinite(hlen) & (hlen > 0), "invalid straw half length")
    measured["udir"] = direction.astype(np.float32).astype(float)
    return measured, dict(direction=direction, hlen=hlen)


@A._checked
def residual_operands(generated, state, measured, policy):
    """Conditional null/TOT/LongVal formula; no captured value or variance arguments."""
    n = len(generated["doca"])
    require(policy["hit_state"] == 0 and policy["constrain_tot"] is True
            and policy["constrain_long"] is True and policy["constrain_absdrift"] is False
            and policy["null_drift_variance"] is False, "unsupported residual policy")
    for name in ("dtime", "wdist", "timevar", "uvar"):
        require(np.shape(measured[name]) == (n,) and np.all(np.isfinite(measured[name])), "invalid measurement " + name)
    require(np.shape(measured["udir"]) == (n, 3) and np.all(np.isfinite(measured["udir"])), "invalid measured direction")
    sign = A._dot3(state["sensor_direction"], measured["udir"])
    require(np.abs(sign) > .99, "sensor/ComboHit orientation ambiguous")
    longitudinal = np.where(sign < 0, generated["longitudinal"]+measured["wdist"],
                            generated["longitudinal"]-measured["wdist"])
    value = np.stack((generated["doca"], state["sensor_time"]-state["particle_time"]-measured["dtime"], longitudinal), -1)
    mvar = np.column_stack((np.full(n, policy["distance_variance"]), measured["timevar"], measured["uvar"]))
    scale = np.full(n, policy["variance_scale"])
    require(np.isfinite(value) & np.isfinite(mvar) & (mvar > 0), "invalid generated residual/variance")
    require(np.isfinite(scale) & (scale > 0), "invalid generated variance scale")
    weight = 1./(mvar*scale[:, None])
    require(np.isfinite(weight) & (weight > 0), "invalid generated weight")
    return dict(value=value, residual=value*np.array([-1., 1., 1.]), mvar=mvar, scale=scale,
                weight=weight, orientation_dot=sign)


def audit(hits, rows, qseed, qref, mass, ctime, measured, regenerated, policy):
    identities = keys(hits, HIT_KEYS)
    selected, capH, capr, capw = join_null_rows(rows, hits)
    cap = snapshot_state(selected)
    line = A.sensor(measured["early"], measured["late"], measured["measurement_time"], regenerated["signal_speed"])
    hints = C.constructor_hint(qseed, line, measured["corrected_time"], mass)
    built = C.solve(qseed, line, *hints, precision=NATIVE_CA_PRECISION, mass=mass)
    require(built["usable"], "unusable constructor; no exclusions")
    state = C.solve(qref, line, built["particle_time"], built["sensor_time"], precision=NATIVE_CA_PRECISION, mass=mass)
    require(state["usable"], "unusable reference; no exclusions")
    require(np.abs(state["particle_poca"][:, 2]-line["mid"][:, 2]) <= 100., "z-retry outside supported initial scope")
    generated = fixed_state_rows(qref, line, state, mass)
    compare = lambda a, b: P.comparison(a, b, identities)
    result = dict(n_hits=len(hits), n_targets=len(np.unique(identities[:, :4], axis=0)), n_excluded_hits=0,
                  hit_keys=identities.tolist(), hit_key_fields=list(HIT_KEYS),
                  constructor_time=compare(built["particle_time"], ctime),
                  geometry={k: compare(state[k], cap[k]) for k in P.STATE_FIELDS},
                  H=compare(generated["H"], capH), measurement_reconstruction={}, variants={})
    for name in ("dtime", "timevar", "wdist", "uvar", "udir"):
        result["measurement_reconstruction"][name] = compare(regenerated[name], measured[name])
    result["measurement_reconstruction"]["early_end"] = compare(regenerated["early_end"], measured["eend"])
    for name, inputs in (("dumped_combohit_measurements", measured), ("source_fcl_regenerated_measurements", regenerated)):
        operands = residual_operands(generated, state, inputs, policy)
        W, b = A.information(qref, generated["H"], operands["residual"], operands["weight"])
        metrics = dict(value=compare(operands["value"], _dense(selected, "row_value", (3,))),
                       mvar=compare(operands["mvar"], _dense(selected, "row_mvar", (3,))),
                       scale=compare(operands["scale"], column(selected, "variance_scale")),
                       weight=compare(operands["weight"], capw),
                       weight_matrix=compare(W, _dense(hits, "weight_matrix", (6, 6))),
                       weight_vector=compare(b, _dense(hits, "weight_vector", (6,))))
        result["variants"][name] = dict(metrics=metrics, generated={k: v.tolist() for k, v in operands.items()},
            agreement={kind: all(m[kind]["passed"] for m in metrics.values()) for kind in ("criterion", "exact")})
    result["captured_state_scope_check"] = dict(hit_state=compare(np.full(len(hits), policy["hit_state"]), column(selected, "hit_state")),
        independently_selected=False, reason="CAD uses unbiased CA/DriftInfo; null scope checked, not independently selected")
    return result


def run(args):
    with args.out.open("x", encoding="utf-8") as stream:
        report = dict(completed=False, full_closure=False, f1_generated_operands=False,
                      independent_state_selection=False, conditional_null_residual_variance_audit=True,
                      rtol=RTOL, atol=ATOL, limits=LIMITS)
        watched = {}
        try:
            require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
            prior, _ = verified_ca_receipt(args.ca_report, watched)
            here = Path(__file__).resolve().parent
            extra = [Path(__file__), here/"f1_initial_measurements.py", here/"run_f1_initial_residuals.sh"]
            extra += [args.native_source/p for p in NATIVE_PATHS]
            for p in extra:
                resolved, sha = str(p.resolve()), digest(p)
                require(resolved not in watched or watched[resolved] == sha, "source differs from prior receipt")
                watched[resolved] = sha
            report["executing_modules"] = executing_sources(watched)
            report["ca_receipt"] = dict(path=str(args.ca_report), sha256=CA_REPORT_SHA256)
            old, new = prior["frozen_gate"], prior["operand_gate"]
            expanded = Path(prior["speed_configuration"]["path"])
            require(digest(expanded) == FROZEN_EXPANDED_SHA256, "expanded FCL hash differs")
            config = response_configuration(expanded)
            policy = initial_policy(expanded, args.native_source/"Offline/TrackerConditions/src/DriftInfo.cc")
            report.update(configuration=config, policy=policy)
            with uproot.open(old["inputs"]["capture"]["path"]) as f:
                tables = {name: read_tree(f, "kktrace", name, old["highest_cycles"]["capture"])
                          for name in ("iterations", "pieces", "effects", "seed_contract")}
            hits, qseed, qref, mass, ctime, hkeys = P.select_initial(tables["iterations"], tables["pieces"], tables["effects"], tables["seed_contract"])
            P.population_guard(tables["iterations"], hkeys)
            with uproot.open(new["inputs"]["capture"]["path"]) as f:
                rows = read_tree(f, "kktrace", "hit_operands", new["highest_cycles"]["capture"])
            with uproot.open(old["inputs"]["baseline"]["path"]) as f:
                events = read_tree(f, "trkdump", "events", old["highest_cycles"]["baseline"], [*KEYS[:3], *["s_"+k for k in EVENT_FIELDS]])
                geometry = read_tree(f, "trkdump", "conditions", old["highest_cycles"]["baseline"], list(GEOMETRY_FIELDS))
                geometry = {k: array(geometry[k]) for k in ak.fields(geometry)}
            measured, geom = measured_inputs(events, geometry, hkeys[:, [0, 1, 2, 4]], column(hits, "strawid"))
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                regenerated = reconstruct(measured, geom, config)
                report["population"] = audit(hits, rows, qseed, qref, mass, ctime, measured, regenerated, policy)
            report["completed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            end = {}
            for p in watched:
                try:
                    end[p] = digest(p)
                except OSError:
                    end[p] = None
            report.update(before_sha256=watched, end_sha256=end, hashes_unchanged=bool(watched) and end == watched)
            if not report["hashes_unchanged"]:
                report.update(completed=False, integrity_error="missing or changed input/source hashes")
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ca-report", "native-source", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    report = run(parser.parse_args())
    print(json.dumps({k: report[k] for k in ("completed", "hashes_unchanged", "error", "integrity_error") if k in report}))
    raise SystemExit(0 if report["completed"] else 1)


if __name__ == "__main__":
    main()
