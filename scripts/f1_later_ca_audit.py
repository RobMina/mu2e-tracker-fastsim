"""Adjacent retained-hit CA audit on preserved captures; not complete native history replay."""
import argparse
import json
import os
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

import f1_initial_residual_audit as R
import f1_captured_ca_population as P
import f1_native_reference_audit as REF
import f1_later_ca as L
from f1_captured_ca_audit import A, C, NATIVE_CA_PRECISION, snapshot_state
from f1_native_decision_audit import (KEYS, ITER_KEYS, array, column, dense, digest,
                                     keys, match_rows, read_tree, require)
from f1_hit_operand_check import HIT_KEYS, boolean
from trkfastsim.reco import hitreco, nativegeometry, nativeref, strawaxis

EXPECTED_SNAPSHOTS, EXPECTED_HIT_UPDATES = 61622, 31600
CAD_SHA256 = "37c3c844dd3339ab2ddca6208485f2d0ee190adbb8b4415a0e91ae8de78d9afb"
REFERENCE_SHA256 = "b66ff5d3c430cd72c2efb15a735d6f43ea92810d5e858a359d4ec79a93e9ca14"
NATIVE_PATHS = ("KinKal/Trajectory/PiecewiseClosestApproach.hh", "KinKal/Trajectory/PiecewiseTrajectory.hh",
                "KinKal/Trajectory/ClosestApproach.hh", "KinKal/Trajectory/ClosestApproachData.hh",
                "KinKal/Trajectory/SensorLine.hh", "KinKal/Trajectory/SensorLine.cc",
                "KinKal/Trajectory/KinematicLine.cc", "KinKal/Fit/Track.hh", "KinKal/Fit/MetaIterConfig.hh",
                "Offline/Mu2eKinKal/inc/KKStrawHit.hh", "Offline/Mu2eKinKal/inc/KKStrawXing.hh",
                "Offline/Mu2eKinKal/inc/KKFitUtilities.hh", "Offline/Mu2eKinKal/src/KKFitUtilities.cc",
                "Offline/Mu2eKinKal/inc/KKFitSettings.hh", "Offline/TrackerGeom/inc/Straw.hh")
LIMITS = [
    "Same-meta-iteration adjacent hit updates after captured swap only; no extension/meta transition or intervening-history replay.",
    "Previous captured usable CA times supply source-derived next hints; actual hint calls and intermediate piece visits are not captured.",
    "Candidate pieces, native mass and prior CA remain captured inputs; no independently generated F1 trajectory or full-fit claim.",
    "Straw::origin is the wire midpoint, exactly dumped as midz; inferred straw midpoints are only a historical diagnostic control.",
    "Retained hit CA vectors/status are observed; crossings have only reference and offset-inclusive time, no CA sidecar vectors/status/hints.",
    "Initial crossing offsets are captured construction values; extension offsets and constructor history are absent.",
    "Hit fallback and crossing fallback hierarchy have synthetic tests only when their required native states are unobserved or unexercised.",
    "NumPy iterative arithmetic is not universal ROOT/compiler parity; no fitted corrections, regularization or changed tolerances.",
    "Expanded FCL continuity is prior-audit provenance, not an original capture-generation receipt.",
]


def receipt(path, watched, expected=CAD_SHA256):
    path = str(Path(path).resolve())
    require(path not in watched or watched[path] == expected, "conflicting predecessor receipt provenance")
    watched[path] = expected
    require(digest(path) == expected, "later-CA predecessor receipt changed")
    data = json.loads(Path(path).read_text())
    manifest = data.get("before_sha256")
    require(isinstance(manifest, dict) and bool(manifest), "missing predecessor manifest")
    for p, sha in manifest.items():
        require(p not in watched or watched[p] == sha, "conflicting manifest provenance")
        watched[p] = sha
    require(data.get("completed") is True and data.get("hashes_unchanged") is True
            and manifest == data.get("end_sha256"), "incomplete predecessor receipt")
    require(data.get("rtol") == R.RTOL and data.get("atol") == R.ATOL, "predecessor tolerances differ")
    for p, sha in watched.items():
        require(isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) is not None, "invalid manifest digest")
        require(digest(p) == sha, "predecessor input/source changed: " + p)
    return data


def adjacent_hits(effects, iterations, rows):
    """Select explicit adjacent swapped same-stage transitions, retaining every matching hit."""
    hit = effects[column(effects, "identity_kind", "iu") == 1]
    require(len(hit) == len(rows) and len(hit) > 0, "straw sidecar census differs")
    ek, ik = keys(hit, HIT_KEYS), keys(iterations, ITER_KEYS)
    rows = rows[match_rows(ek, keys(rows, HIT_KEYS))]
    for field in ("input_index", "shindex", "strawid", "miter", "iter"):
        require(column(hit, field) == column(rows, field), "sidecar identity differs: " + field)
    parent = match_rows(ek[:, :5], ik)
    identity = np.column_stack((ek[:, :5], column(hit, "input_index", "iu")))
    wanted = identity.copy()
    wanted[:, 4] += 1
    following = nativeref.match_rows(wanted, identity)
    safe = np.maximum(following, 0)
    nextparent = parent[safe]
    same = (column(iterations, "miter")[nextparent] == column(iterations, "miter")[parent])
    same &= column(iterations, "iter")[nextparent] == column(iterations, "iter")[parent]+1
    chosen = (following >= 0) & same & boolean(iterations, "swapped")[parent]
    chosen &= boolean(hit, "has_reference") & boolean(hit, "has_reference")[safe]
    selected = np.flatnonzero(chosen)
    after = following[selected]
    require(column(hit, "shindex")[selected] == column(hit, "shindex")[after], "adjacent source-hit identity changed")
    require(column(hit, "strawid")[selected] == column(hit, "strawid")[after], "adjacent straw identity changed")
    census = dict(n_all_snapshots=len(hit), n_adjacent_updates=len(selected), n_not_selected=int((~chosen).sum()),
                  n_no_next=int((following < 0).sum()), n_next_other_stage=int(((following >= 0) & ~same).sum()),
                  n_same_stage_no_swap=int(((following >= 0) & same & ~boolean(iterations, "swapped")[parent]).sum()),
                  scope="nonselected snapshots are outside the declared adjacency scope, not dropped failed comparisons")
    return hit[selected], hit[after], rows[selected], rows[after], parent[selected], census


def pad_candidates(pieces, iterations):
    candidates = pieces[column(pieces, "owner", "iu") == 1]
    keys(candidates, ITER_KEYS+("piece_index",))
    require(np.isfinite(dense(candidates, "parameters", 6)), "nonfinite candidate parameters")
    require(np.isfinite(dense(candidates, "range", 2)), "nonfinite candidate ranges")
    par, ranges, counts = REF.pad_pieces(candidates, iterations)
    return par, ranges, counts


def inferred_geometry(geometry, conditions, alignment):
    """Same frozen panel reconstruction as prior reference audit; no track-dependent fit."""
    st = conditions["straw"]
    sid = array(geometry["sid"])[0]
    unique = hitreco.unique_straw(sid.astype(np.int64))
    require(np.array_equal(np.sort(unique), np.arange(len(st["midx"]))), "incomplete unique straw geometry")
    cal, hv = (np.zeros((len(unique), 3)) for _ in range(2))
    cal[unique] = np.column_stack([array(geometry[k])[0] for k in ("wcalx", "wcaly", "wcalz")])
    hv[unique] = np.column_stack([array(geometry[k])[0] for k in ("whvx", "whvy", "whvz")])
    for field in ("midx", "midy", "midz", "dirx", "diry", "dirz"):
        require(np.array_equal(array(st[field])[unique], array(geometry[field])[0]), "conditions/dump geometry differs: " + field)
    aligned = array(alignment).astype(np.float32).astype(float)
    require(np.isfinite(aligned), "nonfinite alignment")
    return nativegeometry.straw_axes(st, cal, hv, aligned)


@A._checked
def ztime(parameters, ranges, counts, parent, zpos, hint, mass):
    """Source zTime's at-most-100 piece-index updates; no tolerance convergence rule."""
    parameters, ranges, counts = L._pieces(parameters, ranges, counts)
    parent = array(parent)
    require(parent.ndim == 1, "zTime parent shape")
    parent = L._integer(parent, parent.shape, "zTime parent", 0, len(counts)-1)
    time, zpos, mass = (array(v) for v in (hint, zpos, mass))
    require(time.ndim == 1 and time.shape == zpos.shape == mass.shape == np.shape(parent), "zTime shape")
    require(np.isfinite(time) & np.isfinite(zpos) & np.isfinite(mass) & (mass > 0), "zTime finite/positive operands")
    time = time.copy()
    index = nativeref.nearest(ranges, counts, parent, time)
    active = np.ones(len(time), bool)
    updates = np.zeros(len(time), np.int64)
    for _ in range(100):
        lane = np.flatnonzero(active)
        if not len(lane):
            break
        q = parameters[parent[lane], index[lane]]
        position, direction, speed = C.particle(q, time[lane], mass[lane])
        velocity = direction[:, 2]*speed
        require(velocity != 0, "zTime zero longitudinal velocity")
        time[lane] += (zpos[lane]-position[:, 2])/velocity
        require(np.isfinite(time[lane]), "nonfinite zTime")
        nextindex = nativeref.nearest(ranges, counts, parent[lane], time[lane])
        active[lane] = nextindex != index[lane]
        index[lane] = nextindex
        updates[lane] += 1
    return dict(time=time, index=index, updates=updates, exhausted=active)


def hit_update(parameters, ranges, counts, parent, line, previous, drift, straw_z, mass):
    ph, sh = L.hit_hint(previous, line, drift)
    first = L.piecewise(parameters, ranges, counts, parent, line, ph, sh, mass=mass)
    dz = array(straw_z)-first["particle_poca"][:, 2]
    require(np.isfinite(dz), "nonfinite first hit-z difference")
    retry = ~first["usable"] | (np.abs(dz) > 100.)
    final = {k: v.copy() for k, v in first.items()}
    zinfo = None
    if np.any(retry):
        lane = np.flatnonzero(retry)
        subline = {k: v[lane] for k, v in line.items()}
        mtime = line["measurement_time"][lane]-(.5*line["length"][lane])/line["speed"][lane]
        zinfo = ztime(parameters, ranges, counts, parent[lane], array(straw_z)[lane], mtime-array(drift)[lane], mass[lane])
        second = L.piecewise(parameters, ranges, counts, parent[lane], subline, zinfo["time"], mtime, mass=mass[lane])
        for k in final:
            final[k][lane] = second[k]
    finaldz = array(straw_z)-final["particle_poca"][:, 2]
    require(np.isfinite(finaldz), "nonfinite final hit-z difference")
    return first, final, dict(retry=retry, first_dz=dz, final_dz=finaldz,
                             give_up=~final["usable"] | (np.abs(finaldz) > 100.), ztime=zinfo)


def state_metrics(generated, captured, identities):
    return {k: P.comparison(generated[k], captured[k], identities) for k in P.STATE_FIELDS}


def audit_hits(tables, rows, events, geometry, conditions, inferred, speed_config):
    before, after, row0, row1, parent, census = adjacent_hits(tables["effects"], tables["iterations"], rows)
    require(census["n_all_snapshots"] == EXPECTED_SNAPSHOTS and len(before) == EXPECTED_HIT_UPDATES, "frozen adjacent hit census changed")
    old = snapshot_state(row0)
    cap = snapshot_state(row1)
    for field in P.STATE_FIELDS:
        require(np.isfinite(old[field]) & np.isfinite(cap[field]), "nonfinite prior/following CA state: " + field)
    require(boolean(row0, "ca_usable") & boolean(row1, "ca_usable"), "unusable observed adjacent CA unsupported; no exclusion")
    require(column(row0, "precision") == NATIVE_CA_PRECISION, "prior CA precision differs")
    require(column(row1, "precision") == NATIVE_CA_PRECISION, "following CA precision differs")
    require(old["particle_time"] == column(before, "time"), "prior CA/effect time differs")
    require(cap["particle_time"] == column(after, "time"), "following CA/effect time differs")
    par, ranges, counts = pad_candidates(tables["pieces"], tables["iterations"])
    require(counts[parent] > 0, "missing candidate owner")
    present = counts > 0
    parent = np.cumsum(present)[parent]-1
    par, ranges, counts = par[present], ranges[present], counts[present]
    actual_range = dense(after, "reference_range", 2)
    require(actual_range[:, 1] > actual_range[:, 0], "positive-width reference required for unique range lookup")
    actual = nativeref.nearest(ranges, counts, parent, actual_range.mean(axis=1))
    require(np.all(ranges[parent, actual] == actual_range, axis=1), "following reference range not in candidate")
    require(np.all(par[parent, actual] == dense(after, "reference_parameters", 6), axis=1), "following reference not candidate parameters")
    seedrow = match_rows(keys(before, unique=False), keys(tables["seed_contract"]))
    mass = column(tables["seed_contract"], "mass")[seedrow]
    hkeys = keys(before, KEYS+("shindex",), unique=False)
    measured, _ = R.measured_inputs(events, geometry, hkeys[:, [0, 1, 2, 4]], column(before, "strawid"))
    speed = P.source_signal_speed(measured["energy"], speed_config)
    line = A.sensor(measured["early"], measured["late"], measured["measurement_time"], speed)
    sid = hitreco.unique_straw(column(before, "strawid"))
    straw_z = array(conditions["straw"]["midz"])[sid]
    first, final, policy = hit_update(par, ranges, counts, parent, line, old, measured["dtime"], straw_z, mass)
    ids = keys(before, HIT_KEYS)
    inputids = keys(before, ITER_KEYS+("input_index",))
    result = dict(census=census, n_targets=len(np.unique(ids[:, :4], axis=0)), n_excluded_comparisons=0,
                  previous_key_fields=list(HIT_KEYS), previous_hit_keys=ids.tolist(), following_hit_keys=keys(after, HIT_KEYS).tolist(),
                  source_identity_keys=inputids.tolist(), pointer_index=actual.tolist(),
                  first_pass=dict(pointer=P.comparison(first["index"], actual, ids), geometry=state_metrics(first, cap, ids)),
                  final=dict(pointer=P.comparison(final["index"], actual, ids), geometry=state_metrics(final, cap, ids),
                             status=P.comparison(final["status"], column(row1, "ca_status"), ids)),
                  n_previous_unusable=int((~boolean(row0, "ca_usable")).sum()),
                  n_first_moved=int((first["moves"] > 0).sum()), n_first_exhausted=int(first["exhausted"].sum()),
                  n_final_exhausted=int(final["exhausted"].sum()),
                  n_retry=int(policy["retry"].sum()), n_give_up=int(policy["give_up"].sum()),
                  retry_keys=inputids[policy["retry"]].tolist(), give_up_keys=inputids[policy["give_up"]].tolist(),
                  first_dz_mm=policy["first_dz"].tolist(), final_dz_mm=policy["final_dz"].tolist(),
                  captured_dz_mm=(straw_z-cap["particle_poca"][:, 2]).tolist(),
                  final_pointer_solution_split_keys=inputids[final["index"] != final["solution_index"]].tolist(),
                  retry_ztime=None if policy["ztime"] is None else {k: v.tolist() for k, v in policy["ztime"].items()})
    analytic = nativeref.search(par, ranges, counts, parent, line["mid"], line["direction"], old["particle_time"])
    ap, _, _ = C.particle(par[parent, analytic["solution_index"]], analytic["time"], mass)
    adz = straw_z-ap[:, 2]
    historic_dz = inferred["midpoint"][sid, 2]-ap[:, 2]
    result["analytic_control"] = dict(n_retry=int((np.abs(adz) > 100.).sum()), retry_keys=inputids[np.abs(adz) > 100.].tolist(),
                                      n_inferred_straw_midpoint_retry=int((np.abs(historic_dz) > 100.).sum()),
                                      scope="analytic control with native mass only in position; historic analytic toca uses legacy mass",
                                      pointer=P.comparison(analytic["index"], actual, ids))
    predicted_state = np.where(policy["give_up"], -3, column(row0, "hit_state"))
    result["hit_state_transition"] = dict(
        comparison=P.comparison(predicted_state, column(row1, "hit_state"), ids),
        n_previously_unusable=int((column(row0, "hit_state") == -3).sum()),
        n_new_give_up=int((policy["give_up"] & (column(row0, "hit_state") != -3)).sum()),
        n_persistent_unusable_without_retry=int((~policy["give_up"] & (column(row0, "hit_state") == -3)).sum()))
    result["give_up_vs_observed_state"] = dict(
        n_captured_unusable_hit_state=int((column(row1, "hit_state") == -3).sum()),
        n_predicted_give_up_not_captured_unusable=int((policy["give_up"] & (column(row1, "hit_state") != -3)).sum()),
        n_captured_unusable_without_predicted_give_up=int((~policy["give_up"] & (column(row1, "hit_state") == -3)).sum()),
        scope="same-meta-iteration updateState(first=false) preserves updateReference's unusable hit-state marking")
    result["final_ca_agreement"] = {mode: all(metric[mode]["passed"] for metric in (
        *result["final"]["geometry"].values(), result["final"]["pointer"], result["final"]["status"]))
        for mode in ("criterion", "exact")}
    return result


@A._checked
def crossing_cap_evidence(tables, inferred, reference):
    """Inspect the two declared historical caps using observed references and crossing times only."""
    capkeys = array(reference["moves_10"]["initial_crossings"]["exhausted_keys"])
    require(capkeys.shape == (2, 6) and capkeys.dtype.kind in "iu", "historical crossing-cap keys differ")
    xings = tables["effects"][column(tables["effects"], "identity_kind") == 5]
    xkeys = keys(xings, ITER_KEYS+("input_index",))
    selected = match_rows(capkeys, xkeys)
    following_keys = capkeys.copy()
    following_keys[:, 4] += 1
    following = match_rows(following_keys, xkeys)
    before, after = xings[selected], xings[following]
    ik = keys(tables["iterations"], ITER_KEYS)
    parent = match_rows(capkeys[:, :5], ik)
    nextparent = match_rows(following_keys[:, :5], ik)
    require(boolean(tables["iterations"], "swapped")[parent], "crossing-cap owner did not swap")
    require(column(tables["iterations"], "miter")[parent] == column(tables["iterations"], "miter")[nextparent], "crossing-cap stage changed")
    require(column(tables["iterations"], "iter")[nextparent] == column(tables["iterations"], "iter")[parent]+1, "crossing-cap iteration gap")
    require(column(before, "strawid") == column(after, "strawid"), "crossing-cap straw changed")
    seed = tables["seed_contract"]
    sr = match_rows(capkeys[:, :4], keys(seed))
    ci = capkeys[:, 5]
    counts_offset = array(ak.num(seed.sx_offset))
    require(ci < counts_offset[sr], "cap lacks initial constructor offset")
    flat = np.r_[0, np.cumsum(counts_offset)][sr]+ci
    offset = array(ak.flatten(seed.sx_offset))[flat]
    require(np.isfinite(offset) & (offset > 0), "invalid cap construction offset")
    par, ranges, counts = pad_candidates(tables["pieces"], tables["iterations"])
    require(counts[parent] > 0, "cap missing candidate owner")
    sid = hitreco.unique_straw(column(before, "strawid"))
    mid, direction = inferred["midpoint"][sid], inferred["direction"][sid]
    hint = column(before, "time")-offset
    require(np.isfinite(hint), "nonfinite captured prior crossing time")
    predicted = nativeref.search(par, ranges, counts, parent, mid, direction, hint)
    require(predicted["exhausted"] & predicted["usable"], "historical cap not reproduced")
    actual_range = dense(after, "reference_range", 2)
    actual = nativeref.nearest(ranges, counts, parent, actual_range.mean(axis=1))
    require(np.all(ranges[parent, actual] == actual_range, axis=1), "cap reference range differs")
    require(np.all(par[parent, actual] == dense(after, "reference_parameters", 6), axis=1), "cap reference parameters differ")
    pointer_time, pointer_usable = nativeref.toca(par[parent, predicted["index"]], mid, direction)
    require(pointer_usable & np.isfinite(pointer_time) & np.isfinite(predicted["time"]), "nonfinite cap comparison")
    native_time = column(after, "time")-offset
    require(np.isfinite(native_time), "nonfinite captured following crossing time")
    return dict(scope="analytic legacy-mass/inferred-straw-geometry control; observed native time/reference only, NOT native crossing-vector replay",
                key_fields=[*ITER_KEYS, "input_index"], keys=capkeys.tolist(), constructor_offset_ns=offset.tolist(),
                native_particle_time_subtract_offset_ns=native_time.tolist(),
                analytic_retained_time_ns=predicted["time"].tolist(), analytic_pointer_resolve_time_ns=pointer_time.tolist(),
                retained_minus_native_time_ns=(predicted["time"]-native_time).tolist(),
                pointer_resolve_minus_native_time_ns=(pointer_time-native_time).tolist(),
                pointer_resolve_minus_retained_time_ns=(pointer_time-predicted["time"]).tolist(),
                native_pointer_index=actual.tolist(), analytic_pointer_index=predicted["index"].tolist(),
                analytic_solution_index=predicted["solution_index"].tolist(), analytic_moves=predicted["moves"].tolist(),
                pointer_agreement=bool(np.array_equal(predicted["index"], actual)),
                native_hint_calls_captured=False, native_retained_vectors_compared=False,
                native_offset_time_roundtrip_exact=False)


def safe_manifest(manifest):
    """Keep malformed declarations inspectable without allowing invalid JSON values."""
    result = {}
    for path, value in manifest.items():
        try:
            json.dumps(value, allow_nan=False)
            result[path] = value
        except (TypeError, ValueError):
            result[path] = {"invalid_declaration_repr": repr(value)}
    return result


def run(args):
    with args.out.open("x", encoding="utf-8") as stream:
        result = dict(completed=False, full_closure=False, full_history_closure=False, full_state_policy_closure=False,
                      native_hint_calls_captured=False, crossing_retained_vectors_captured=False,
                      rtol=R.RTOL, atol=R.ATOL, limits=LIMITS)
        watched = {}

        def watch(path):
            path = str(Path(path).resolve())
            sha = digest(path)
            require(path not in watched or watched[path] == sha, "changed input/source: " + path)
            watched[path] = sha
            return sha

        try:
            require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
            cad = receipt(args.cad_report, watched)
            residual = json.loads(Path(cad["residual_receipt"]["path"]).read_text())
            prior = json.loads(Path(residual["ca_receipt"]["path"]).read_text())
            reference_path = args.cad_report.parent/"reference_state_v1.json"
            watched[str(reference_path.resolve())] = REFERENCE_SHA256
            require(digest(reference_path) == REFERENCE_SHA256, "historical reference report changed")
            reference = json.loads(reference_path.read_text())
            for entry in reference["geometry_inputs"].values():
                require(watch(entry["path"]) == entry["sha256"], "reference geometry changed")
            for name in NATIVE_PATHS:
                watch(args.native_source/name)
            for name in ("f1_later_ca_audit.py", "f1_later_ca.py", "run_f1_later_ca.sh"):
                watch(Path(__file__).resolve().parent/name)
            result["executing_modules"] = R.executing_sources(watched)
            result["cad_receipt"] = dict(path=str(args.cad_report), sha256=CAD_SHA256)
            expanded = Path(prior["speed_configuration"]["path"])
            require(watch(expanded) == R.FROZEN_EXPANDED_SHA256, "expanded FCL differs")
            require(not re.search(r"\b(?:HitReferencePiece|XingReferencePiece|MaxPieceIter)\s*:", expanded.read_text()), "nondefault piece policy")
            result["piece_policy"] = dict(max_moves=10, strategy="DOCA", basis="absent FCL overrides; MetaIterConfig source defaults")
            old, new = prior["frozen_gate"], prior["operand_gate"]
            with uproot.open(old["inputs"]["capture"]["path"]) as f:
                tables = {name: read_tree(f, "kktrace", name, old["highest_cycles"]["capture"])
                          for name in ("iterations", "pieces", "effects", "seed_contract")}
            with uproot.open(new["inputs"]["capture"]["path"]) as f:
                rows = read_tree(f, "kktrace", "hit_operands", new["highest_cycles"]["capture"])
            with uproot.open(old["inputs"]["baseline"]["path"]) as f:
                events = read_tree(f, "trkdump", "events", old["highest_cycles"]["baseline"], [*KEYS[:3], *["s_"+k for k in R.EVENT_FIELDS]])
                geometry = read_tree(f, "trkdump", "conditions", old["highest_cycles"]["baseline"], list(dict.fromkeys((*R.GEOMETRY_FIELDS, "midx", "midy", "midz"))))
                geometry = {k: array(geometry[k]) for k in geometry.fields}
            cond = hitreco.load_conditions(reference["geometry_inputs"]["conditions"]["path"])
            alignment = strawaxis.read_trkalignstraw(reference["geometry_inputs"]["straw_axis"]["path"])
            inferred = inferred_geometry(geometry, cond, alignment)
            result["inferred_geometry"] = dict(metrics=inferred["metrics"], metadata=inferred["metadata"])
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                result["hits"] = audit_hits(tables, rows, events, geometry, cond, inferred, prior["speed_configuration"])
            result["historical_crossing_caps"] = dict(
                scope="prior analytic reference audit only; no crossing sidecar retained vectors/hints; not new native search closure",
                report_path=str(reference_path), report_sha256=REFERENCE_SHA256,
                n_crossings=reference["moves_10"]["initial_crossings"]["n"],
                exhausted_keys=reference["moves_10"]["initial_crossings"]["exhausted_keys"],
                pointer_minus_retained_time_ns=reference["analytic_ca_state"]["pointer_resolve_minus_retained_time_ns"])
            result["crossing_cap_observed_time_comparison"] = crossing_cap_evidence(tables, inferred, reference)
            result["completed"] = True
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            end = {}
            for p in watched:
                try:
                    end[p] = digest(p)
                except (OSError, ValueError, TypeError):
                    end[p] = None
            result.update(before_sha256=safe_manifest(watched), end_sha256=end, hashes_unchanged=bool(watched) and watched == end)
            if not result["hashes_unchanged"]:
                result.update(completed=False, integrity_error="missing/changed source/input hashes")
            stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cad-report", "native-source", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    report = run(p.parse_args())
    summary = {k: report[k] for k in ("completed", "hashes_unchanged", "error", "integrity_error") if k in report}
    if "hits" in report:
        hits = report["hits"]
        summary["hits"] = {k: hits[k] for k in ("census", "n_targets", "n_retry", "n_give_up", "n_first_moved", "n_first_exhausted", "n_final_exhausted")}
        summary["final"] = {k: {mode: row[mode]["passed"] for mode in ("criterion", "exact")}
                            for k, row in hits["final"]["geometry"].items()}
        summary["pointer"] = hits["final"]["pointer"]["exact"]["passed"]
        summary["status"] = hits["final"]["status"]["exact"]["passed"]
        summary["final_ca_agreement"] = hits["final_ca_agreement"]
        summary["hit_state_transition"] = {k: v for k, v in hits["hit_state_transition"].items() if k != "comparison"}
        summary["hit_state_exact"] = hits["hit_state_transition"]["comparison"]["exact"]["passed"]
    print(json.dumps(summary, indent=2))
    passed = report["completed"] and report.get("hits", {}).get("final_ca_agreement", {}).get("criterion") is True
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
