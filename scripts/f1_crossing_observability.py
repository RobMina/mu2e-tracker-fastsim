"""Preserved crossing-reference/order census and missing-observable boundary; no crossing CA replay."""
import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import uproot

import f1_later_ca_audit as H
import f1_transition_audit as D
import f1_transition_census as T
import f1_initial_residual_audit as R
from f1_native_decision_audit import (KEYS, ITER_KEYS, array, column, dense, digest,
                                     keys, match_rows, read_tree, require)
from f1_hit_operand_check import HIT_KEYS, boolean
from trkfastsim.reco import nativeref

TRANSITION_SHA256 = "e95881049167a98c402680512c8997aa4548a10fedbec47b9dcaaa9b68bdc0a5"
SOURCE_HASHES = {
    "Offline/Mu2eKinKal/inc/KKFit.hh": "d177547b7cb72b03f2729595f0b24a8f2aba52c47b0f597581cd0da4d232720f",
    "Offline/Mu2eKinKal/inc/KKStrawXing.hh": "bd628dc818cc11c1a02fa7c4bbe02fdb48cbd738fb945365bf127b8785b73e59",
    "Offline/Mu2eKinKal/src/KKFitUtilities.cc": "9ac76a6219bf310a7889bc6a6926e8f7a1f44d0841d841b18d0770d0ea8a607e",
    "Offline/TrackerGeom/inc/Straw.hh": "a651a9282959599c0bd7e77441aeed19e2e0ed4a5f311cb184b174f739075fe3",
    "KinKal/Fit/Material.hh": "bc45cc89e1521b47382c67ad42d9476cfe4e5d6b070974e62a601f408136b6d4",
    "TrkFastSim/inc/IterationCaptureWriter.hh": "173a6a922fc9db81ffcb359f90ff635dcee527951cd1bca917649523dcbe9cd0",
    "TrkFastSim/src/StrawHitTableDumper_module.cc": "609871cc2fff273d2b2e6d1c04ad8fec61386b51198afb1bb2f950ce73f574c4",
}
LIMITS = [
    "Reference membership/order consistency is not native crossing CA construction, search, retained-vector or offset closure.",
    "The 81 associated hints are reconstructed hit-constructor outputs from the pinned predecessor, not native observed crossing hints.",
    "Crossing sensor geometry uses straw-tube endpoints; the conditions dump contains wire endpoints only. No inferred-geometry substitution.",
    "Effect time includes a private frozen offset; crossing-minus-hit time is NOT that offset or a CA-time observation.",
    "Final collection partners/input order are source-conditioned evidence, not constructor or iteration-local association snapshots.",
    "Material candidate membership is supplied; missing/rejected candidates, hierarchy and selection efficiency are not replayed.",
    "No captured material changes/lengths or later times are inverted to fit missing constructor operands.",
    "No fitter/default/threshold/tolerance change, native run/build or full-fit closure.",
]


def material_configuration(path):
    """Pin the normal hierarchical discovery branch and its numeric cuts."""
    D.configuration(path)
    text = Path(path).read_text()
    blocks = re.findall(r"(?ms)^([ ]*)KKFitSettings\s*:\s*\{(.*?)^\1\}", text)
    require(bool(blocks) and len({b for _, b in blocks}) == 1, "ambiguous material configuration")
    block = blocks[0][1]
    result = {}
    for name, expected in (("AddStrawMinDz", 0), ("MaxDStraw", 2), ("MaxStrawDOCA", 5),
                           ("MaxStrawDOCAConsistency", 1), ("MaxStrawUposBuffer", 0), ("StrawNBuffer", 0)):
        values = re.findall(r"\b"+name+r"\s*:\s*([^\s,}\]]+)", block)
        require(len(values) == 1 and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", values[0]) is not None
                and float(values[0]) == expected, "unsupported material " + name)
        result[name] = expected
    require(re.findall(r"\bSkipStrawCheck\s*:\s*(\w+)", block) == ["false"], "unsupported SkipStrawCheck")
    result.update(SkipStrawCheck=False, vertical_branch_reachable=False,
                  branch_basis="strict abs(direction.Z()) < AddStrawMinDz=0 cannot be true; this does not establish finite/nonzero direction.Z() at every unobserved discovery step",
                  discovery_replayed=False)
    return result


def source_partition(lifetime):
    """Partition exact source identities and check append order, not chronological effect order."""
    association = D.association_contract(lifetime)
    require(association["source_conditioned_constructor_partner_agreement"] is True
            and association["n_duplicate_crossing_straw_keys"] == 0, "unsupported association history")
    c = lifetime["crossings"]
    ck = array(c["source_keys"])
    initial, partner = array(c["initial"]), array(c["partner_hit_rows"])
    associated = (~initial) & (partner >= 0)
    material = (~initial) & (partner < 0)
    labels = np.where(initial, "initial", np.where(associated, "associated_extension", "material_extension"))
    targets, parent = np.unique(ck[:, :4], axis=0, return_inverse=True)
    last_associated = np.full(len(targets), -1, dtype=np.int64)
    first_material = np.full(len(targets), np.iinfo(np.int64).max, dtype=np.int64)
    np.maximum.at(last_associated, parent[associated], ck[associated, 4])
    np.minimum.at(first_material, parent[material], ck[material, 4])
    mixed = (last_associated >= 0) & (first_material != np.iinfo(np.int64).max)
    order_ok = last_associated[mixed] < first_material[mixed]
    chosen = np.flatnonzero(associated)
    require(len(chosen) > 0 and np.any(material), "empty extension partition")
    sort = np.lexsort((ck[chosen, 4], parent[chosen]))
    chosen = chosen[sort]
    shindex = array(lifetime["hits"]["shindex"])
    require(shindex.shape == array(lifetime["hits"]["initial"]).shape and shindex.dtype.kind in "iu"
            and np.all(shindex >= 0) and np.all(shindex <= np.iinfo(np.int64).max), "invalid partner shindex")
    hitindex = shindex[partner[chosen]]
    same = parent[chosen[1:]] == parent[chosen[:-1]]
    hit_order_ok = (~same) | (hitindex[1:] > hitindex[:-1])
    census = {label: dict(n_objects=int(np.sum(labels == label)),
                         n_targets=len(np.unique(ck[labels == label, :4], axis=0)))
              for label in ("initial", "associated_extension", "material_extension")}
    return dict(labels=labels, association=association, census=census,
                associated_before_material=bool(order_ok.all()),
                bad_order_targets=targets[mixed][~order_ok].tolist(),
                associated_in_combohit_order=bool(hit_order_ok.all()),
                bad_hit_order_source_keys=ck[chosen[1:]][~hit_order_ok].tolist(),
                native_constructor_order_observed=False)


def reference_membership(effects, pieces, iterations):
    """Validate every crossing reference against its same-iteration old owner, with exact full keys."""
    cross = effects[column(effects, "identity_kind", "iu") == 5]
    require(len(cross) > 0, "empty crossing snapshots")
    ids = keys(cross, HIT_KEYS)
    keys(cross, ITER_KEYS+("input_index",))
    require(boolean(cross, "has_reference"), "crossing without reference")
    require(np.isfinite(column(cross, "time")), "nonfinite crossing effect time")
    parent = match_rows(ids[:, :5], keys(iterations, ITER_KEYS))
    for field in ("miter", "iter"):
        require(column(cross, field, "iu") == column(iterations, field, "iu")[parent], "crossing iteration differs")
    old = pieces[column(pieces, "owner", "iu") == 0]
    pk = keys(old, ITER_KEYS+("piece_index",))
    owner_ranges = dense(old, "range", 2)
    require(owner_ranges[:, 1] > owner_ranges[:, 0], "nonpositive old-owner range")
    order = np.lexsort(tuple(pk[:, j] for j in range(pk.shape[1]-1, -1, -1)))
    same_owner = np.all(pk[order[1:], :5] == pk[order[:-1], :5], axis=1)
    require((~same_owner) | (owner_ranges[order[1:], 0] >= owner_ranges[order[:-1], 1]),
            "ambiguous/overlapping old-owner ranges")
    for field, width in (("parameters", 6), ("covariance", 36), ("bnom", 3)):
        dense(old, field, width)
    par, ranges, counts = H.REF.pad_pieces(old, iterations)
    require(counts[parent] > 0, "missing old-owner pieces")
    ref_range = dense(cross, "reference_range", 2)
    require(ref_range[:, 1] > ref_range[:, 0], "nonpositive reference range")
    # Positive nonoverlapping ranges have unique ends; no representable interior midpoint is required.
    idx = nativeref.nearest(ranges, counts, parent, ref_range[:, 1])
    row = match_rows(np.column_stack((ids[:, :5], idx)), pk)
    fields = {}
    for name, width in (("range", 2), ("parameters", 6), ("covariance", 36), ("bnom", 3)):
        equal = np.all(dense(cross, "reference_"+name, width) == dense(old, name, width)[row], axis=1)
        fields[name] = dict(exact=bool(equal.all()), n_unequal=int((~equal).sum()), mismatch_keys=ids[~equal].tolist())
    return dict(n_snapshots=len(cross), n_targets=len(np.unique(ids[:, :4], axis=0)), n_excluded=0,
                snapshot_keys=ids.tolist(), piece_index=idx.tolist(), fields=fields,
                exact=all(v["exact"] for v in fields.values()),
                scope="captured reference belongs to old owner; does not certify solved piece or CA geometry")


def associated_hints(lifetime, effects, predecessor):
    """Expose already reconstructed hit constructor hints without constructing a straw-tube CA."""
    part = source_partition(lifetime)
    c, h = lifetime["crossings"], lifetime["hits"]
    take = part["labels"] == "associated_extension"
    ck = array(c["source_keys"])[take]
    partner = array(c["partner_hit_rows"])[take]
    first_rows = {}
    for family, obj in (("crossings", c), ("hits", h)):
        rows = array(obj["first_rows"])
        require(rows.shape == (len(array(obj["source_keys"])),) and rows.dtype.kind in "iu"
                and np.all(rows >= 0) and np.all(rows < len(effects)), "invalid first-row indices")
        require(len(np.unique(rows)) == len(rows), "duplicate first-row indices")
        first_rows[family] = rows
    first_cross = effects[first_rows["crossings"][take]]
    first_hit = effects[first_rows["hits"][partner]]
    require(np.array_equal(keys(first_cross, KEYS+("input_index",)), ck), "crossing first row identity differs")
    require(np.array_equal(keys(first_hit, KEYS+("input_index",)), array(h["source_keys"])[partner]), "hit first row identity differs")
    require(column(first_cross, "identity_kind") == 5, "first row not crossing")
    require(column(first_hit, "identity_kind") == 1, "partner first row not hit")
    require(column(first_cross, "strawid") == column(first_hit, "strawid"), "first partner straw differs")
    require(np.array_equal(keys(first_cross, ITER_KEYS, unique=False), keys(first_hit, ITER_KEYS, unique=False)), "partner first iteration differs")
    new = predecessor["new_extension_hits"]
    require(new["final"]["agreement"]["exact"] is True and new["accepted_membership_kinematic_checks_passed"] is True,
            "predecessor new-hit geometry/cuts do not pass")
    nk = array(new["hit_keys"])
    require(nk.ndim == 2 and nk.shape[1] == 6 and nk.dtype.kind in "iu" and np.all(nk >= 0)
            and np.all(nk <= np.iinfo(np.int64).max), "invalid reconstructed hit keys")
    hk = keys(first_hit, HIT_KEYS)
    join = match_rows(hk, nk)
    require(len(hk) == len(nk), "incomplete associated constructor coverage")
    values = {}
    for field in ("particle_time", "sensor_time", "usable"):
        v = array(new["constructor_state"][field])
        require(v.shape == (len(nk),), "constructor state shape")
        if field == "usable":
            require(v.dtype.kind == "b" and np.all(v), "invalid/unusable hit constructor state")
        else:
            require(v.dtype.kind in "fiu" and np.all(np.isfinite(v)), "nonfinite constructor hint")
        values[field] = v[join]
    require(column(first_cross, "partner_shindex") == column(first_hit, "shindex"), "first partner shindex differs")
    ct, ht = column(first_cross, "time"), column(first_hit, "time")
    require(np.isfinite(ct) & np.isfinite(ht), "nonfinite first effect times")
    with np.errstate(over="raise", invalid="raise"):
        gap = ct-ht
    require(np.isfinite(gap), "nonfinite effect-time gap")
    return dict(n_crossings=len(ck), source_keys=ck.tolist(), partner_hit_keys=hk.tolist(),
                reconstructed_hit_constructor_particle_hint=values["particle_time"].tolist(),
                reconstructed_hit_constructor_sensor_hint=values["sensor_time"].tolist(),
                straw_line_measurement_time=values["particle_time"].tolist(),
                first_crossing_minus_hit_effect_time=gap.tolist(), effect_time_gap_is_private_offset=False,
                effect_time_gap_min=float(gap.min()), effect_time_gap_max=float(gap.max()),
                n_nonpositive_effect_time_gaps=int((gap <= 0).sum()),
                constructor_hints_natively_observed=False, crossing_constructor_ca_replayed=False,
                private_offset_compared=False, retained_crossing_ca_compared=False,
                scope="source-prescribed inputs from reconstructed hit constructor; missing tube endpoints block independent crossing construction")


def run(args):
    with args.out.open("x", encoding="utf-8") as stream:
        report = dict(completed=False, full_history_closure=False, full_state_policy_closure=False,
                      full_fit_closure=False, native_crossing_constructor_offsets_compared=False,
                      rtol=R.RTOL, atol=R.ATOL, limits=LIMITS)
        watched = {}
        def watch(path):
            p = str(Path(path).resolve())
            sha = digest(p)
            require(p not in watched or watched[p] == sha, "changed input/source: "+p)
            watched[p] = sha
            return sha
        try:
            require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
            previous = H.receipt(args.transition_report, watched, expected=TRANSITION_SHA256)
            require(previous["agreement"]["exact"] is True and previous["snapshot_coverage"]["disjoint_full_key_partition"] is True,
                    "incomplete transition geometry coverage")
            for name, expected in SOURCE_HASHES.items():
                require(watch(args.native_source/name) == expected, "frozen native source changed: "+name)
            for name in ("f1_crossing_observability.py", "run_f1_crossing_observability.sh"):
                watch(Path(__file__).resolve().parent/name)
            report["executing_modules"] = R.executing_sources(watched)
            report["transition_receipt"] = dict(path=str(args.transition_report), sha256=TRANSITION_SHA256)
            chain = previous
            for field in ("later_receipt", "cad_receipt", "residual_receipt", "ca_receipt"):
                path = str(Path(chain[field]["path"]).resolve())
                require(path in watched and digest(path) == watched[path], "unbound predecessor receipt")
                chain = json.loads(Path(path).read_text())
            old = chain["frozen_gate"]
            expanded = Path(chain["speed_configuration"]["path"])
            require(watch(expanded) == R.FROZEN_EXPANDED_SHA256, "expanded FCL differs")
            report["material_configuration"] = material_configuration(expanded)
            with uproot.open(old["inputs"]["capture"]["path"]) as f:
                tables = {name: read_tree(f, "kktrace", name, old["highest_cycles"]["capture"])
                          for name in ("iterations", "effects", "pieces", "seed_contract")}
                report["capture_trees"] = f.keys(cycle=True)
            with uproot.open(old["inputs"]["baseline"]["path"]) as f:
                tree_key = old["highest_cycles"]["baseline"]["trkdump/conditions"]
                report["condition_fields"] = list(f[tree_key].keys())
            report["effect_fields"] = tables["effects"].fields
            lifetime = T.object_lifetimes(tables["effects"], tables["iterations"], tables["seed_contract"], 4, 1)
            report["object_census"] = lifetime["census"]
            for family in ("hits", "crossings"):
                census = lifetime["census"][family]
                require(census["no_disappearance_or_reappearance"] is True
                        and census["n_initial_first_seen_after_zero"] == 0
                        and census["n_seed_objects_not_observed"] == 0, "incomplete crossing/partner history")
                obj = lifetime[family]
                first = tables["effects"][obj["first_rows"][obj["extension"]]]
                require(column(first, "miter") == 5, "extension first miter differs")
                require(column(first, "iter") == 0, "extension object first seen after iter0")
            partition = source_partition(lifetime)
            report["partition"] = D.jsonable(partition)
            report["references"] = reference_membership(tables["effects"], tables["pieces"], tables["iterations"])
            report["associated_constructor_inputs"] = associated_hints(lifetime, tables["effects"], previous)
            cross = tables["effects"][column(tables["effects"], "identity_kind") == 5]
            source = keys(cross, KEYS+("input_index",), unique=False)
            label = partition["labels"][match_rows(source, array(lifetime["crossings"]["source_keys"]))]
            report["snapshots_by_partition"] = {name: int(np.sum(label == name)) for name in partition["census"]}
            require(len(cross) == 102949 and len(lifetime["crossings"]["source_keys"]) == 5090
                    and report["associated_constructor_inputs"]["n_crossings"] == 81, "frozen crossing census differs")
            report["consistency_passed"] = (report["references"]["exact"] and partition["associated_before_material"]
                                            and partition["associated_in_combohit_order"])
            report["completed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            end = {}
            for path in watched:
                try:
                    end[path] = digest(path)
                except (OSError, ValueError, TypeError):
                    end[path] = None
            report.update(before_sha256=H.safe_manifest(watched), end_sha256=end,
                          hashes_unchanged=bool(watched) and watched == end)
            if not report["hashes_unchanged"]:
                report.update(completed=False, integrity_error="missing/changed source/input hashes")
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("transition-report", "native-source", "out"):
        p.add_argument("--"+name, required=True, type=Path)
    report = run(p.parse_args())
    print(json.dumps({k: report[k] for k in ("completed", "hashes_unchanged", "consistency_passed", "error",
                                           "snapshots_by_partition") if k in report}, indent=2))
    raise SystemExit(0 if report["completed"] and report.get("consistency_passed") is True else 1)


if __name__ == "__main__":
    main()
