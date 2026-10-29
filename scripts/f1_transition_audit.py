"""Retained-hit meta/extension continuity and membership-conditioned new-hit CA; no fit replay."""
import argparse
import json
import os
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

import f1_later_ca_audit as H
import f1_initial_residual_audit as R
import f1_captured_ca_population as P
import f1_transition_census as T
from f1_captured_ca_audit import A, NATIVE_CA_PRECISION, snapshot_state
from f1_native_decision_audit import (KEYS, ITER_KEYS, array, column, dense, digest,
                                     keys, match_rows, read_tree, require)
from f1_hit_operand_check import HIT_KEYS, boolean
from trkfastsim.reco import nativeref, hitreco

LATER_SHA256 = "e0dd9a77bb0ad28351f8d6303bdda79486747294b8dcee2ae1fee30703888ce1"
LABELS = ("seed_meta", "extension_entry", "extension_meta")
NATIVE_PATHS = (*H.NATIVE_PATHS, "KinKal/Fit/Measurement.hh", "KinKal/Fit/Material.hh",
                "KinKal/Fit/Effect.hh", "KinKal/General/TimeRange.hh", "Offline/Mu2eKinKal/inc/KKTrack.hh",
                "Offline/Mu2eKinKal/inc/KKFit.hh", "Offline/Mu2eKinKal/src/KKFitSettings.cc",
                "Offline/Mu2eKinKal/src/KKFitUtilities.cc",
                "TrkFastSim/inc/IterationCaptureWriter.hh", "TrkFastSim/src/KinematicLineFitTrace_module.cc",
                "TrkFastSim/src/StrawHitTableDumper_module.cc")
LIMITS = [
    "Preserved captured candidate trajectories, native mass and prior retained CA remain inputs; not an independently generated F1 fit.",
    "Ordinary meta transitions and retained hits across extension are replayed only under source-established no-end-processing/no-domains/no-cluster configuration and exact owner continuity.",
    "First=true may change hit state/weights through unbiased CA, CAD and ANN; retained ca_ is not assigned by that calculation. This audit does not replay those state decisions.",
    "New extension hits use captured accepted membership plus measured inputs and source constructor/reference updates; no excluded-candidate discovery or selection-efficiency claim.",
    "Crossing identities and partners in effects are resolved from FINAL collections, not iteration-local association snapshots. Stable recorded partners alone do not establish constructor-time association.",
    "Crossing constructor private offsets, intermediate CAs and hints remain absent, especially unassociated material-discovery crossings.",
    "Exact finite sample equality is not universal ROOT/compiler parity; no thresholds, tolerances, fit defaults, regularization or native reruns changed.",
    "Expanded FCL/source continuity is prior-audit evidence, not an original capture-generation receipt.",
]


def configuration(path):
    """Read identical expanded blocks and reject configurations with extra CA-mutating transitions."""
    text = Path(path).read_text()
    blocks = {}
    for name in ("FitSettings", "ExtensionSettings", "KKFitSettings"):
        found = re.findall(r"(?ms)^([ ]*)"+name+r"\s*:\s*\{(.*?)^\1\}", text)
        require(bool(found) and len({b for _, b in found}) == 1, "ambiguous/missing " + name)
        blocks[name] = found[0][1]
    result = {}
    for name in ("FitSettings", "ExtensionSettings"):
        block = blocks[name]
        for field in ("ProcessEnds", "BFieldCorrection"):
            require(re.findall(r"\b"+field+r"\s*:\s*(\w+)", block) == ["false"], "unsupported " + name + "." + field)
        require(not re.search(r"\b(?:HitReferencePiece|XingReferencePiece|MaxPieceIter)\s*:", block), "nondefault reference strategy")
        for updater in ("Chi2SHU", "PanelDiagSHU"):
            declarations = list(re.finditer(r"\b"+updater+r"Settings\s*:\s*", block))
            require(len(declarations) <= 1, "ambiguous inactive updater")
            if declarations:
                setting, _ = json.JSONDecoder().raw_decode(block[declarations[0].end():])
                require(setting == [], "unsupported updater")
        schedules = list(re.finditer(r"\bMetaIterationSettings\s*:\s*", block))
        require(len(schedules) == 1, "ambiguous schedule")
        schedule, _ = json.JSONDecoder().raw_decode(block[schedules[0].end():])
        expected = ([[5, "CADSHU"], [5, "CADSHU"], [2, "CADSHU"], [1, "CADSHU"]] if name == "FitSettings" else
                    [[2, "CADSHU"], [1, "CADSHU"], [.3, "CADSHU"], [0, "CADSHU:BkgANNSHU"],
                     [1, "CADSHU:DriftANNSHU"], [.5, "CADSHU:DriftANNSHU"], [.2, "CADSHU:DriftANNSHU"],
                     [0, "CADSHU:DriftANNSHU:BkgANNSHU"], [0, "CADSHU:DriftANNSHU"]])
        require(isinstance(schedule, list) and bool(schedule) and all(isinstance(row, list) and len(row) == 2
                and type(row[0]) in (int, float) and np.isfinite(row[0]) and isinstance(row[1], str)
                for row in schedule), "invalid " + name + " schedule types")
        require(schedule == expected, "unsupported " + name + " schedule")
        result[name] = dict(schedule=schedule, process_ends=False, bfield_correction=False)
    kk = blocks["KKFitSettings"]
    require(re.findall(r'\bStrawHitClusterLevel\s*:\s*"([^"]+)"', kk) == ["none"], "hit clusters unsupported")
    for field in ("UseCaloCluster",):
        require(re.findall(r"\b"+field+r"\s*:\s*(\w+)", kk) == ["false"], "unsupported " + field)
    for field in ("AddHits", "AddMaterial", "MaterialCorrection"):
        require(re.findall(r"\b"+field+r"\s*:\s*(\w+)", kk) == ["true"], "unsupported " + field)
    for field, expected in (("MaxStrawHitDt", 50.), ("MaxStrawHitDOCA", 10.), ("TPOCAPrecision", 1e-4)):
        found = re.findall(r"\b"+field+r"\s*:\s*([^\s,}\]]+)", kk)
        require(len(found) == 1 and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", found[0]) is not None
                and float(found[0]) == expected, "unsupported " + field)
        result[field] = float(np.float32(float(found[0])))
    result.update(seed_stages=4, extension_stages=9, miter_origin=1, max_piece_moves=10, cluster_level="none",
                  origin="Straw::origin() is dumped wire midpoint, not straw-tube midpoint",
                  lifecycle="processEnds disabled; no domains/replacement; existing hits unaffected by createEffects(new); updateWHS uses separate CA")
    return result


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def hit_links(effects, rows, iterations):
    """All adjacent existing-hit identities, regardless of meta boundary; no state-based selection."""
    hits = effects[column(effects, "identity_kind") == 1]
    require(len(hits) == len(rows) and len(hits) > 0, "hit sidecar census differs")
    hk = keys(hits, HIT_KEYS)
    rows = rows[match_rows(hk, keys(rows, HIT_KEYS))]
    for field in ("input_index", "shindex", "strawid", "miter", "iter"):
        require(column(hits, field) == column(rows, field), "sidecar identity differs: " + field)
    parent = match_rows(hk[:, :5], keys(iterations, ITER_KEYS))
    for field in ("miter", "iter"):
        require(column(hits, field) == column(iterations, field)[parent], "effect/iteration identity differs: " + field)
    identity = keys(hits, ITER_KEYS+("input_index",))
    wanted = identity.copy()
    wanted[:, 4] += 1
    following = nativeref.match_rows(wanted, identity)
    before_rows = np.flatnonzero(following >= 0)
    next_rows = following[before_rows]
    for field in ("shindex", "strawid"):
        require(column(hits, field)[before_rows] == column(hits, field)[next_rows], "adjacent hit identity changed: " + field)
    require(boolean(hits, "has_reference"), "hit without reference")
    return hits, rows, parent, before_rows, next_rows


def compact_candidates(tables):
    par, ranges, counts = H.pad_candidates(tables["pieces"], tables["iterations"])
    present = counts > 0
    remap = np.where(present, np.cumsum(present)-1, -1)
    return par[present], ranges[present], counts[present], remap


def observed_pointer(hits, parameters, ranges, counts, parent):
    actual_range = dense(hits, "reference_range", 2)
    require(actual_range[:, 1] > actual_range[:, 0], "positive-width reference required for unique lookup")
    index = nativeref.nearest(ranges, counts, parent, actual_range.mean(axis=1))
    require(np.all(ranges[parent, index] == actual_range, axis=1), "reference range outside candidate owner")
    require(np.all(parameters[parent, index] == dense(hits, "reference_parameters", 6), axis=1), "reference parameters outside candidate owner")
    return index


def observed_state(hits, rows):
    require(boolean(rows, "ca_usable"), "unusable retained native CA unsupported, no exclusion")
    require(column(rows, "precision") == NATIVE_CA_PRECISION, "native precision differs")
    state = snapshot_state(rows)
    for name in P.STATE_FIELDS:
        require(np.isfinite(state[name]), "nonfinite native state: " + name)
    require(state["particle_time"] == column(hits, "time"), "retained CA/effect time differs")
    status = column(rows, "ca_status", "iu")
    require((status >= 0) & (status < 3), "usable native status differs")
    return state


def measured_for_hits(hits, events, geometry, speed_config):
    hkeys = keys(hits, KEYS+("shindex",), unique=False)
    measured, geom = R.measured_inputs(events, geometry, hkeys[:, [0, 1, 2, 4]], column(hits, "strawid"))
    line = A.sensor(measured["early"], measured["late"], measured["measurement_time"],
                    P.source_signal_speed(measured["energy"], speed_config))
    gi = match_rows(column(hits, "strawid")[:, None], array(geometry["sid"])[0, :, None])
    origin_z = array(geometry["midz"])[0, gi]
    require(np.isfinite(origin_z), "nonfinite wire origin")
    return measured, geom, line, origin_z


def metrics(state, cap, actual, rows, identities):
    result = dict(geometry=H.state_metrics(state, cap, identities),
                  pointer=P.comparison(state["index"], actual, identities),
                  status=P.comparison(state["status"], column(rows, "ca_status"), identities))
    result["agreement"] = {mode: all(item[mode]["passed"] for item in (
        *result["geometry"].values(), result["pointer"], result["status"])) for mode in ("criterion", "exact")}
    return result


def replay_transitions(tables, rows, events, geometry, speed_config, links, continuity):
    hits, rows, hp, before_idx, after_idx = hit_links(tables["effects"], rows, tables["iterations"])
    before_parent = hp[before_idx]
    linkrow = match_rows(before_parent[:, None], array(links["previous"])[:, None])
    label = array(links["labels"])[linkrow]
    require(np.array_equal(hp[after_idx], array(links["following"])[linkrow]), "hit/iteration adjacency differs")
    require(boolean(tables["iterations"], "swapped")[before_parent], "adjacent retained hits after unswapped attempt")
    require(array(continuity["equality"])[linkrow], "owner continuity failed for retained hit; no exclusion")
    par, ranges, counts, remap = compact_candidates(tables)
    out = {}
    for group in LABELS:
        take = label == group
        require(np.any(take), "empty declared transition group: " + group)
        b, a = hits[before_idx[take]], hits[after_idx[take]]
        row0, row1 = rows[before_idx[take]], rows[after_idx[take]]
        parent = remap[before_parent[take]]
        require(parent >= 0, "missing candidate owner")
        old, cap = observed_state(b, row0), observed_state(a, row1)
        actual = observed_pointer(a, par, ranges, counts, parent)
        seedrow = match_rows(keys(b, unique=False), keys(tables["seed_contract"]))
        mass = column(tables["seed_contract"], "mass")[seedrow]
        measured, _, line, origin_z = measured_for_hits(b, events, geometry, speed_config)
        first, final, policy = H.hit_update(par, ranges, counts, parent, line, old, measured["dtime"], origin_z, mass)
        ids = keys(b, HIT_KEYS)
        result = metrics(final, cap, actual, row1, ids)
        out[group] = dict(n_hits=len(b), n_targets=len(np.unique(ids[:, :4], axis=0)), n_excluded=0,
                          previous_keys=ids.tolist(), following_keys=keys(a, HIT_KEYS).tolist(),
                          n_moved=int((first["moves"] > 0).sum()), n_retry=int(policy["retry"].sum()),
                          n_give_up=int(policy["give_up"].sum()), n_exhausted=int(final["exhausted"].sum()),
                          retry_keys=ids[policy["retry"]].tolist(), give_up_keys=ids[policy["give_up"]].tolist(),
                          final=result, hit_state_decisions_replayed=False,
                          n_hit_state_changed=int((column(row0, "hit_state") != column(row1, "hit_state")).sum()),
                          scope="retained CA only; first=true unbiased state updaters not replayed")
    out["census"] = dict(n_all_snapshots=len(hits), n_existing_hit_adjacent=len(before_idx),
                         n_same_meta=int((label == "same_meta").sum()),
                         n_new_boundary_comparisons=int((label != "same_meta").sum()),
                         n_snapshots_without_next_same_hit=int(len(hits)-len(before_idx)))
    return out


def new_extension_hits(tables, rows, events, geometry, speed_config, config, links, continuity):
    """Construct accepted extension-hit CAs; accepted membership is supplied, not rediscovered."""
    hits = tables["effects"][column(tables["effects"], "identity_kind") == 1]
    identity = keys(hits, KEYS+("input_index",), unique=False)
    unique, inverse = np.unique(identity, axis=0, return_inverse=True)
    ordinal = column(hits, "iteration_ordinal")
    first_ordinal = np.full(len(unique), np.iinfo(np.int64).max)
    np.minimum.at(first_ordinal, inverse, ordinal)
    first = hits[ordinal == first_ordinal[inverse]]
    seed = tables["seed_contract"]
    seed_keys, _ = T._seed_family(seed, keys(seed), "sh")
    first_keys = keys(first, KEYS+("input_index",))
    is_extension = nativeref.match_rows(first_keys, seed_keys) < 0
    require(column(first, "iteration_ordinal")[~is_extension] == 0, "initial seed hit first observed after construction")
    new = first[is_extension]
    require(len(new) > 0, "empty extension-hit population")
    require(column(new, "miter") == config["seed_stages"]+config["miter_origin"], "new hit first seen outside extension entry")
    require(column(new, "iter") == 0, "new hit not at first extension iteration")
    following_parent = match_rows(keys(new, ITER_KEYS, unique=False), keys(tables["iterations"], ITER_KEYS))
    linkrow = match_rows(following_parent[:, None], array(links["following"])[:, None])
    require(array(links["labels"])[linkrow] == "extension_entry", "new hit lacks extension-entry link")
    require(array(continuity["equality"])[linkrow], "new hit pre-extension owner continuity failed")
    before_parent = array(links["previous"])[linkrow]
    par, ranges, counts, remap = compact_candidates(tables)
    parent = remap[before_parent]
    require(parent >= 0, "new hit missing pre-extension candidate")
    row = rows[match_rows(keys(new, HIT_KEYS), keys(rows, HIT_KEYS))]
    cap = observed_state(new, row)
    actual = observed_pointer(new, par, ranges, counts, parent)
    seedrow = match_rows(keys(new, unique=False), keys(tables["seed_contract"]))
    mass = column(tables["seed_contract"], "mass")[seedrow]
    measured, geom, line, origin_z = measured_for_hits(new, events, geometry, speed_config)
    eparent = match_rows(keys(new, KEYS[:3], unique=False), keys(events, KEYS[:3]))
    counts_hit = array(ak.num(events.s_sid))
    require(array(ak.num(events.s_z)) == counts_hit, "measured position lengths differ")
    flat = np.r_[0, np.cumsum(counts_hit)][eparent]+column(new, "shindex")
    zvalues = array(ak.flatten(events.s_z))
    require(zvalues.dtype == np.dtype("float32"), "ComboHit position is not float32")
    measured_z = zvalues[flat].astype(float)
    require(np.isfinite(measured_z), "nonfinite measured ComboHit position")
    zt = H.ztime(par, ranges, counts, parent, measured_z, measured["corrected_time"], mass)
    psign = A._dot3(line["direction"], geom["direction"])
    htime = line["measurement_time"]-(geom["hlen"]-psign*measured["wdist"])/line["speed"]
    require(np.isfinite(htime), "nonfinite extension sensor hint")
    built = H.L.piecewise(par, ranges, counts, parent, line, zt["time"], htime, mass)
    require(built["usable"], "unusable extension constructor unsupported; no exclusion")
    first, final, policy = H.hit_update(par, ranges, counts, parent, line, built, measured["dtime"], origin_z, mass)
    ids = keys(new, HIT_KEYS)
    absolute_doca = np.sqrt(A._dot3(built["delta"], built["delta"]))
    require(np.isfinite(absolute_doca), "nonfinite constructor DOCA")
    measured_time = measured["corrected_time"]
    dt = np.abs(measured_time-zt["time"])
    brange_begin = ranges[parent, 0, 0]-config["MaxStrawHitDt"]
    brange_end = ranges[parent, counts[parent]-1, 1]+config["MaxStrawHitDt"]
    eligibility = (measured_time >= brange_begin) & (measured_time < brange_end)
    eligibility &= (dt < config["MaxStrawHitDt"]) & (absolute_doca < config["MaxStrawHitDOCA"])
    return dict(n_hits=len(new), n_targets=len(np.unique(ids[:, :4], axis=0)), n_excluded=0,
                hit_keys=ids.tolist(), source_keys=keys(new, KEYS+("shindex",)).tolist(),
                final=metrics(final, cap, actual, row, ids), membership_supplied=True,
                candidate_discovery_replayed=False, native_constructor_intermediates_compared=False,
                n_constructor_moved=int((built["moves"] > 0).sum()), n_constructor_exhausted=int(built["exhausted"].sum()),
                n_constructor_ztime_exhausted=int(zt["exhausted"].sum()),
                n_reference_moved=int((first["moves"] > 0).sum()), n_retry=int(policy["retry"].sum()),
                n_give_up=int(policy["give_up"].sum()), n_reference_exhausted=int(final["exhausted"].sum()),
                n_measured_kinematic_cuts_failed=int((~eligibility).sum()),
                accepted_membership_kinematic_checks_passed=bool(eligibility.all()),
                kinematic_cut_failed_hit_keys=ids[~eligibility].tolist(),
                kinematic_cut_scope="accepted-membership conditional numeric checks; no missing/rejected candidate or flag selection census",
                constructor_state={k: built[k].tolist() for k in P.STATE_FIELDS},
                final_state={k: final[k].tolist() for k in P.STATE_FIELDS},
                ztime=zt["time"].tolist(), sensor_hint=htime.tolist())


def association_contract(lifetime):
    """Source-conditioned single-extension partner inference, never a claimed association snapshot."""
    hits, cross = lifetime["hits"], lifetime["crossings"]
    hk, ck = array(hits["source_keys"]), array(cross["source_keys"])
    hi, ci = array(hits["initial"]), array(cross["initial"])
    partner = array(cross["partner_hit_rows"])
    require(hk.ndim == ck.ndim == 2 and hk.shape[1] == ck.shape[1] == 5
            and hk.dtype.kind in "iu" and ck.dtype.kind in "iu" and len(hk) > 0 and len(ck) > 0,
            "association source-key shape/type/population")
    require(np.all(hk >= 0) and np.all(ck >= 0) and np.all(hk <= np.iinfo(np.int64).max)
            and np.all(ck <= np.iinfo(np.int64).max), "association source-key bounds")
    require(len(np.unique(hk, axis=0)) == len(hk) and len(np.unique(ck, axis=0)) == len(ck), "duplicate association source key")
    require(hi.shape == (len(hk),) and ci.shape == partner.shape == (len(ck),)
            and hi.dtype.kind == ci.dtype.kind == "b" and partner.dtype.kind in "iu", "association flag/index shape")
    associated = partner >= 0
    require((partner >= -1) & (partner < len(hk)), "association partner index bounds")
    use = partner[associated]
    require(np.all(hk[use, :4] == ck[associated, :4], axis=1), "association partner target differs")
    multiplicity = np.bincount(use, minlength=len(hk))
    first_hit = array(hits["first_ordinal"])
    first_cross = array(cross["first_ordinal"])
    require(first_hit.shape == hi.shape and first_cross.shape == ci.shape and first_hit.dtype.kind in "iu"
            and first_cross.dtype.kind in "iu" and np.all(first_hit >= 0) and np.all(first_cross >= 0),
            "association first-ordinal shape/type/bounds")
    same_first = first_cross[associated] == first_hit[use]
    same_kind = ci[associated] == hi[use]
    initial_paired = bool(np.all(associated[ci]))
    one_each = bool(np.all(multiplicity == 1))
    straw = array(cross["strawid"])
    require(straw.shape == ci.shape and straw.dtype.kind in "iu" and np.all(straw >= 0)
            and np.all(straw <= np.iinfo(np.int64).max), "association straw shape/type/bounds")
    straw_keys = np.column_stack((ck[:, :4], straw))
    _, straw_counts = np.unique(straw_keys, axis=0, return_counts=True)
    inferred = initial_paired and one_each and bool(same_first.all()) and bool(same_kind.all())
    return dict(n_hits=len(hk), n_crossings=len(ck), n_initial_crossings=int(ci.sum()),
                n_initial_crossings_without_partner=int((ci & ~associated).sum()),
                n_hits_without_exactly_one_crossing=int((multiplicity != 1).sum()),
                n_partner_first_appearance_different=int((~same_first).sum()),
                n_initial_extension_partner_class_disagreements=int((~same_kind).sum()),
                n_duplicate_crossing_straw_keys=int((straw_counts > 1).sum()),
                source_conditioned_constructor_partner_agreement=inferred,
                native_iteration_local_partner_snapshots=False, private_constructor_offsets_observed=False,
                scope="one source-scheduled extension only: all initial crossings already paired; addStrawHits constructs new paired crossings; addStraws subsequently constructs unpaired material crossings; final metadata validates this inferred ordering, not a captured constructor snapshot")


def snapshot_coverage(rows, cad, later, retained, new):
    """Join disjoint comparison receipts to every sidecar snapshot without event-count inference."""
    initial = cad["population"]
    require(all(initial["reference_geometry"][field]["exact"]["passed"] is True for field in P.STATE_FIELDS),
            "initial geometry receipt does not pass exact equality")
    require(later["hits"]["final_ca_agreement"]["exact"] is True, "same-meta CA receipt does not pass exact equality")
    require(all(retained[group]["final"]["agreement"]["exact"] is True for group in LABELS)
            and new["final"]["agreement"]["exact"] is True, "transition geometry does not pass exact equality")
    parts = {"initial": initial["hit_keys"], "same_meta": later["hits"]["following_hit_keys"],
             **{group: retained[group]["following_keys"] for group in LABELS}, "new_extension": new["hit_keys"]}
    target = keys(rows, HIT_KEYS)
    joined = []
    counts = {}
    for name, value in parts.items():
        part = array(value)
        require(part.ndim == 2 and part.shape[1] == len(HIT_KEYS) and part.dtype.kind in "iu"
                and len(part) > 0 and np.all(part >= 0) and np.all(part <= np.iinfo(np.int64).max),
                "invalid snapshot coverage keys: " + name)
        part = part.astype(np.int64)
        require(len(np.unique(part, axis=0)) == len(part), "duplicate partition snapshot keys")
        match_rows(part, target)
        joined.append(part)
        counts[name] = len(part)
    combined = np.concatenate(joined)
    require(len(combined) == len(target) and len(np.unique(combined, axis=0)) == len(combined),
            "snapshot coverage is not a disjoint complete partition")
    match_rows(target, combined)
    return dict(n_snapshots=len(target), n_targets=len(np.unique(target[:, :4], axis=0)), by_partition=counts,
                n_uncompared=0, disjoint_full_key_partition=True, retained_geometry_exact=True,
                scope="every captured retained-hit geometry snapshot; captured candidate trajectories and earlier states remain inputs, not autonomous full-history/state-policy closure")


def run(args):
    with args.out.open("x", encoding="utf-8") as stream:
        report = dict(completed=False, full_history_closure=False, full_state_policy_closure=False,
                      full_fit_closure=False, native_crossing_constructor_offsets_compared=False,
                      rtol=R.RTOL, atol=R.ATOL, limits=LIMITS)
        watched = {}

        def watch(path):
            p = str(Path(path).resolve())
            sha = digest(p)
            require(p not in watched or watched[p] == sha, "changed source/input: " + p)
            watched[p] = sha
            return sha

        try:
            require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
            previous = H.receipt(args.later_report, watched, expected=LATER_SHA256)
            cad = json.loads(Path(previous["cad_receipt"]["path"]).read_text())
            residual = json.loads(Path(cad["residual_receipt"]["path"]).read_text())
            prior = json.loads(Path(residual["ca_receipt"]["path"]).read_text())
            for name in NATIVE_PATHS:
                watch(args.native_source/name)
            for name in ("f1_transition_audit.py", "f1_transition_census.py", "run_f1_transition.sh"):
                watch(Path(__file__).resolve().parent/name)
            report["executing_modules"] = R.executing_sources(watched)
            report["later_receipt"] = dict(path=str(args.later_report), sha256=LATER_SHA256)
            expanded = Path(prior["speed_configuration"]["path"])
            require(watch(expanded) == R.FROZEN_EXPANDED_SHA256, "expanded FCL differs")
            config = configuration(expanded)
            report["configuration"] = config
            old, new = prior["frozen_gate"], prior["operand_gate"]
            with uproot.open(old["inputs"]["capture"]["path"]) as f:
                tables = {name: read_tree(f, "kktrace", name, old["highest_cycles"]["capture"])
                          for name in ("iterations", "pieces", "effects", "seed_contract")}
            with uproot.open(new["inputs"]["capture"]["path"]) as f:
                rows = read_tree(f, "kktrace", "hit_operands", new["highest_cycles"]["capture"])
            require(len(tables["iterations"]) == 3449 and len(rows) == 61622, "frozen capture census differs")
            with uproot.open(old["inputs"]["baseline"]["path"]) as f:
                events = read_tree(f, "trkdump", "events", old["highest_cycles"]["baseline"],
                                   [*KEYS[:3], *["s_"+k for k in (*R.EVENT_FIELDS, "z")]])
                geometry = read_tree(f, "trkdump", "conditions", old["highest_cycles"]["baseline"],
                                     list(dict.fromkeys((*R.GEOMETRY_FIELDS, "midx", "midy", "midz"))))
                geometry = {k: array(geometry[k]) for k in geometry.fields}
            require(column(tables["iterations"], "history_index") == column(tables["iterations"], "iteration_ordinal")+1,
                    "unexpected intervening history record")
            report["history_contract"] = dict(construction_record_index=0, first_miter=1, extension_first_miter=5,
                history_index_equals_ordinal_plus_one=True, basis="Track constructor appends unfit Status(0,0); fit increments miter from history")
            links = T.iteration_links(tables["iterations"], config["seed_stages"], config["extension_stages"], config["miter_origin"])
            report["iteration_links"] = jsonable(links)
            continuity = T.owner_continuity(tables["pieces"], tables["iterations"], links)
            report["owner_continuity"] = jsonable(continuity)
            lifetime = T.object_lifetimes(tables["effects"], tables["iterations"], tables["seed_contract"], config["seed_stages"], config["miter_origin"])
            report["object_lifetimes"] = jsonable(lifetime)
            report["association_contract"] = association_contract(lifetime)
            for family in ("hits", "crossings"):
                census = lifetime["census"][family]
                require(census["no_disappearance_or_reappearance"] and census["n_initial_first_seen_after_zero"] == 0
                        and census["n_seed_objects_not_observed"] == 0, "unsupported incomplete object history: " + family)
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                report["retained_hits"] = replay_transitions(tables, rows, events, geometry, prior["speed_configuration"], links, continuity)
                report["new_extension_hits"] = new_extension_hits(tables, rows, events, geometry, prior["speed_configuration"], config, links, continuity)
            report["agreement_scope"] = "following retained CA only; accepted-membership kinematic checks are a separate necessary consistency gate"
            report["accepted_membership_kinematic_checks_passed"] = report["new_extension_hits"]["accepted_membership_kinematic_checks_passed"]
            report["agreement"] = {mode: all(report["retained_hits"][g]["final"]["agreement"][mode] for g in LABELS)
                                   and report["new_extension_hits"]["final"]["agreement"][mode]
                                   for mode in ("criterion", "exact")}
            report["snapshot_coverage"] = snapshot_coverage(rows, cad, previous, report["retained_hits"], report["new_extension_hits"])
            report["completed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            end = {}
            for p in watched:
                try:
                    end[p] = digest(p)
                except (OSError, ValueError, TypeError):
                    end[p] = None
            report.update(before_sha256=H.safe_manifest(watched), end_sha256=end, hashes_unchanged=bool(watched) and watched == end)
            if not report["hashes_unchanged"]:
                report.update(completed=False, integrity_error="missing/changed source/input hashes")
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("later-report", "native-source", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    report = run(p.parse_args())
    summary = {k: report[k] for k in ("completed", "hashes_unchanged", "error", "integrity_error", "agreement") if k in report}
    if "retained_hits" in report:
        summary["retained_hits"] = {g: {k: v for k, v in report["retained_hits"][g].items()
                                       if k.startswith("n_") or k == "scope"} for g in LABELS}
    if "new_extension_hits" in report:
        summary["new_extension_hits"] = {k: v for k, v in report["new_extension_hits"].items() if k.startswith("n_")}
    print(json.dumps(summary, indent=2))
    passed = report["completed"] and report.get("agreement", {}).get("criterion") is True
    passed &= report.get("accepted_membership_kinematic_checks_passed") is True
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
