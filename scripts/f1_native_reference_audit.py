"""Audit target coverage and adjacent captured reference updates without refitting a baseline."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from trkfastsim.reco import hitreco, nativeref
from trkfastsim.reco.nativetime import digest
from trkfastsim.reco.vr_f1 import _straw_axes

KEYS = ("run", "subrun", "event", "iseed")


def keys(table, names=KEYS):
    return np.column_stack([ak.to_numpy(table[k]) for k in names]).astype(np.int64)


def highest(parent, name):
    candidates = [k for k in parent.keys(cycle=True, recursive=False) if k.split(";")[0] == name]
    return max(candidates, key=lambda k: int(k.rsplit(";", 1)[1]))


def pad_pieces(pieces, iterations):
    parent = nativeref.match_rows(keys(pieces, KEYS + ("iteration_ordinal",)),
                                 keys(iterations, KEYS + ("iteration_ordinal",)))
    if np.any(parent < 0):
        raise ValueError("piece without iteration")
    counts = np.bincount(parent, minlength=len(iterations))
    width = max(1, int(counts.max()))
    par = np.zeros((len(iterations), width, 6))
    ranges = np.zeros((len(iterations), width, 2))
    index = ak.to_numpy(pieces.piece_index)
    if np.any((index < 0) | (index >= counts[parent])):
        raise ValueError("piece index outside owner")
    if len(np.unique(np.column_stack([parent, index]), axis=0)) != len(index):
        raise ValueError("duplicate or noncontiguous piece indices")
    par[parent, index] = ak.to_numpy(pieces.parameters)
    ranges[parent, index] = ak.to_numpy(pieces.range)
    return par, ranges, counts


def coverage(ids, seed_keys, target_keys, run):
    wanted = np.column_stack([np.full(len(ids["good"]), run), ids["subrun"], ids["event"],
                              np.zeros(len(ids["good"]), dtype=np.int64)]).astype(np.int64)
    if len(np.unique(wanted, axis=0)) != len(wanted):
        raise ValueError("duplicate baseline identity")
    if np.any(nativeref.match_rows(wanted, seed_keys) < 0):
        raise ValueError("baseline identity absent from seed capture")
    found = nativeref.match_rows(wanted, target_keys) >= 0
    groups = {"all": np.ones(len(wanted), bool), "gains": ids["good"] & ~ids["stockok"],
              "losses": ~ids["good"] & ids["stockok"]}
    return {name: {"n": int(mask.sum()), "captured": int((mask & found).sum()),
                   "missing_keys": wanted[mask & ~found].tolist()} for name, mask in groups.items()}


def state_census(parameters, parent, mid, wdir, straw_mid, ishit, result, effect_keys):
    """Bound retry triggers and capped-state loss using the analytic solved piece, not its final pointer."""
    from trkfastsim.reco.linefit import C, MMU
    q = parameters[parent, result["solution_index"]]
    d0, phi, z0, theta, t0, mom = q.T
    origin = np.column_stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0])
    direction = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    position = origin + direction * ((result["time"] - t0) * C * mom / np.hypot(mom, MMU))[:, None]
    dz = straw_mid[:, 2] - position[:, 2]
    pointer_time, pointer_usable = nativeref.toca(parameters[parent, result["index"]], mid, wdir)
    delta = pointer_time - result["time"]
    split = result["index"] != result["solution_index"]
    retry = ishit & (~result["usable"] | (np.abs(dz) > 100.0))
    return {"scope": "analytic solved-state geometry; native sensor hints/status and retry results are not captured",
            "hit_retry_trigger_count": int(retry.sum()), "hit_retry_trigger_keys": effect_keys[retry].tolist(),
            "nonfinite_hit_dz": int((ishit & ~np.isfinite(dz)).sum()),
            "max_abs_hit_dz_mm": float(np.max(np.abs(dz[ishit & np.isfinite(dz)])))
                if np.any(ishit & np.isfinite(dz)) else None,
            "pointer_solution_split_count": int(split.sum()), "pointer_solution_split_keys": effect_keys[split].tolist(),
            "pointer_resolve_minus_retained_time_ns": delta[split & pointer_usable & np.isfinite(delta)].tolist(),
            "split_pointer_unusable_count": int((split & ~pointer_usable).sum())}


def run(args):
    gate = json.loads(args.check.read_text())
    flags = ("passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage",
             "recorded_capture_equivalent")
    if not all(gate.get(k) is True for k in flags) or gate.get("smoke_only") is not False:
        raise ValueError("requires full capture passivity gate")
    for name in ("control", "capture", "baseline", "targets"):
        source = gate["inputs"][name]
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("evidence hash changed: " + name)
    captured = gate["inputs"]["capture"]["path"]
    baseline = gate["inputs"]["baseline"]["path"]
    with uproot.open(captured) as f:
        parent = f["kktrace"]
        cycles = {name: highest(parent, name) for name in ("seed_contract", "iterations", "effects", "pieces")}
        seed = parent[cycles["seed_contract"]].arrays()
        it = parent[cycles["iterations"]].arrays()
        effects = parent[cycles["effects"]].arrays()
        pieces = parent[cycles["pieces"]].arrays()
    with uproot.open(baseline) as f:
        events = f["trkdump/events"].arrays(["run", "subrun", "event"], library="np")
    runs = np.unique(events["run"])
    if len(runs) != 1 or len(events["run"]) != gate["n_events"]:
        raise ValueError("unexpected frozen event census")
    target_keys = np.unique(keys(it), axis=0)
    report = {"check": str(args.check), "check_sha256": digest(args.check), "inputs": gate["inputs"],
              "trees": cycles, "n_events": len(events["run"]), "n_seeds": len(seed),
              "n_iterations": len(it), "n_effects": len(effects), "n_targets": len(target_keys),
              "full_numerical_validation": False, "stock_exact_fit": False,
              "coverage": coverage(np.load(args.ids), keys(seed), target_keys, int(runs[0])),
              "ids": str(args.ids), "ids_sha256": digest(args.ids)}
    candidates = pieces[pieces.owner == 1]
    par, ranges, counts = pad_pieces(candidates, it)
    ik = keys(it, KEYS + ("iteration_ordinal",))
    ep = nativeref.match_rows(keys(effects, KEYS + ("iteration_ordinal",)), ik)
    if np.any(ep < 0):
        raise ValueError("effect without iteration")
    family = np.where(ak.to_numpy(effects.kind) == 2, 1, 0)
    ek = np.column_stack([keys(effects, KEYS + ("iteration_ordinal",)), family,
                          ak.to_numpy(effects.input_index)])
    nextkey = ek.copy()
    nextkey[:, 4] += 1
    nxt = nativeref.match_rows(nextkey, ek)
    safe = np.maximum(nxt, 0)
    npart = ep[safe]
    epart = ep
    same_stage = (ak.to_numpy(it.miter)[npart] == ak.to_numpy(it.miter)[epart])
    same_stage &= ak.to_numpy(it.iter)[npart] == ak.to_numpy(it.iter)[epart] + 1
    selected = (nxt >= 0) & same_stage & ak.to_numpy(it.swapped)[epart]
    selected &= ak.to_numpy(effects.has_reference) & ak.to_numpy(effects.has_reference)[safe]
    seedrow = nativeref.match_rows(keys(effects), keys(seed))
    if np.any(seedrow < 0):
        raise ValueError("effect without native seed")
    sx_index = ak.to_numpy(effects.input_index)
    initial_xing = (ak.to_numpy(effects.identity_kind) == 5) & (sx_index >= 0)
    initial_xing &= sx_index < ak.to_numpy(ak.num(seed.sx_offset))[seedrow]
    offsets = np.zeros(len(effects))
    row = np.flatnonzero(initial_xing)
    offsets[row] = ak.to_numpy(ak.flatten(seed.sx_offset[seedrow[row]][ak.from_regular(sx_index[row, None])]))
    hit = ak.to_numpy(effects.identity_kind) == 1
    selected &= hit | initial_xing
    selected &= counts[epart] > 0
    rows = np.flatnonzero(selected)
    e, following = effects[rows], effects[nxt[rows]]
    parent = ep[rows]
    cond = hitreco.load_conditions(args.conditions)
    smid, sdir = _straw_axes(baseline, cond, str(args.straw_axis))
    geometry_mode = getattr(args, "straw_geometry", "approximate")
    if geometry_mode == "panel":
        from trkfastsim.reco import nativegeometry, strawaxis
        with uproot.open(baseline) as f:
            ends = f["trkdump/conditions"].arrays(["sid", "wcalx", "wcaly", "wcalz", "whvx", "whvy", "whvz"], library="np")
        us = hitreco.unique_straw(ends["sid"][0].astype(np.int64))
        if not np.array_equal(np.sort(us), np.arange(len(smid))):
            raise ValueError("panel inference requires exactly one geometry row per straw")
        wc, wh = np.zeros_like(smid), np.zeros_like(smid)
        wc[us] = np.column_stack([ends[k][0] for k in ("wcalx", "wcaly", "wcalz")])
        wh[us] = np.column_stack([ends[k][0] for k in ("whvx", "whvy", "whvz")])
        old_mid, old_dir = smid, sdir
        alignment = strawaxis.read_trkalignstraw(args.straw_axis).astype(np.float32).astype(np.float64)
        inferred = nativegeometry.straw_axes(cond["straw"], wc, wh, alignment)
        smid, sdir = inferred["midpoint"], inferred["direction"]
        closure = dict(inferred["metrics"], metadata=inferred["metadata"],
                       alignment_precision="CSV parsed through float32, matching native std::stof/float storage")
        closure["max_midpoint_change_mm"] = float(np.linalg.norm(smid - old_mid, axis=1).max())
        closure["max_direction_change"] = float(np.linalg.norm(sdir - old_dir, axis=1).max())
        report["panel_geometry_closure"] = closure
    report["straw_geometry"] = geometry_mode
    st = cond["straw"]
    wire_mid = np.column_stack([st[k] for k in ("midx", "midy", "midz")])
    wire_dir = np.column_stack([st[k] for k in ("dirx", "diry", "dirz")])
    with uproot.open(baseline) as f:
        geometry = f["trkdump/conditions"].arrays(["sid", "midx", "midy", "midz", "dirx", "diry", "dirz"], library="np")
    geometry_sid = hitreco.unique_straw(geometry["sid"][0].astype(np.int64))
    for field in ("midx", "midy", "midz", "dirx", "diry", "dirz"):
        if not np.array_equal(st[field][geometry_sid], geometry[field][0]):
            raise ValueError("conditions wire geometry differs from frozen dump: " + field)
    sid = hitreco.unique_straw(ak.to_numpy(e.strawid).astype(np.int64))
    ish = hit[rows]
    mid = np.where(ish[:, None], wire_mid[sid], smid[sid])
    wdir = np.where(ish[:, None], wire_dir[sid], sdir[sid])
    hint = ak.to_numpy(e.time) - offsets[rows]
    actual_range = ak.to_numpy(following.reference_range)
    actual = nativeref.nearest(ranges, counts, parent, actual_range.mean(axis=1))
    membership = np.all(ranges[parent, actual] == actual_range, axis=1)
    membership &= np.all(par[parent, actual] == ak.to_numpy(following.reference_parameters), axis=1)
    if not membership.all():
        raise ValueError(f"{int((~membership).sum())} following references are not prior candidate pieces")
    local_time, local_usable = nativeref.toca(ak.to_numpy(following.reference_parameters), mid, wdir)
    local_delta = local_time - (ak.to_numpy(following.time) - offsets[rows])
    report["captured_reference_local_ca"] = {
        label: {"n": int(mask.sum()), "unusable": int((mask & ~local_usable).sum()),
                "time_difference_quantiles_ns": np.quantile(local_delta[mask & local_usable], [0, .16, .5, .84, 1]).tolist()
                if (mask & local_usable).any() else []}
        for label, mask in (("hits", ish), ("initial_crossings", ~ish))}
    report["adjacent_updates"] = len(rows)
    report["analytic_ca_state"] = state_census(par, parent, mid, wdir, smid[sid], ish,
        nativeref.search(par, ranges, counts, parent, mid, wdir, hint),
        np.column_stack([keys(e, KEYS + ("iteration_ordinal",)), ak.to_numpy(e.input_index)]))
    report["crossing_scope"] = "initial native crossings only, captured construction offset; extension offsets excluded"
    report["geometry_scope"] = "wire geometry exact against frozen dump; reconstructed straw axes, not certified native straw geometry"
    report["geometry_inputs"] = {"conditions": {"path": str(args.conditions), "sha256": digest(args.conditions)},
                                 "straw_axis": {"path": str(args.straw_axis), "sha256": digest(args.straw_axis)}}
    report["limits"] = ["Analytic straight-line CA, not native iterative floating-point CA arithmetic",
                        "Hit z-retry, unusable-CA fallback and cluster snapshots are not replayed",
                        "Only same-meta-iteration adjacent updates after a swap; no extension transition",
                        "No fit, acceptance, material or decision-closure claim"]
    for moves in (0, 10):
        result = nativeref.search(par, ranges, counts, parent, mid, wdir, hint, max_moves=moves)
        arm = {}
        for label, mask in (("hits", ish), ("initial_crossings", ~ish), ("all", np.ones(len(ish), bool))):
            equal = result["index"] == actual
            dt = result["time"] - (ak.to_numpy(following.time) - offsets[rows])
            fail = mask & ~equal
            arm[label] = {"n": int(mask.sum()), "reference_agreement": int((mask & equal).sum()),
                          "reference_mismatch": int(fail.sum()), "moved": int((mask & (result["moves"] > 0)).sum()),
                          "exhausted": int((mask & result["exhausted"]).sum()),
                          "exhausted_keys": np.column_stack([keys(e, KEYS + ("iteration_ordinal",)),
                                                               ak.to_numpy(e.input_index)])[mask & result["exhausted"]].tolist(),
                          "unusable": int((mask & ~result["usable"]).sum()),
                          "nonfinite_time_differences": int((mask & ~np.isfinite(dt)).sum()),
                          "time_difference_quantiles_ns": np.quantile(dt[mask & np.isfinite(dt)], [0, .16, .5, .84, 1]).tolist()
                          if (mask & np.isfinite(dt)).any() else [],
                          "mismatch_keys": np.column_stack([keys(e, KEYS + ("iteration_ordinal",)),
                                                               ak.to_numpy(e.input_index)])[fail][:30].tolist(),
                          "mismatch_details": {
                              "native_index": actual[fail][:30].tolist(),
                              "replay_index": result["index"][fail][:30].tolist(),
                              "solution_index": result["solution_index"][fail][:30].tolist(),
                              "moves": result["moves"][fail][:30].tolist(),
                              "exhausted": result["exhausted"][fail][:30].tolist(),
                              "native_reference_range": actual_range[fail][:30].tolist(),
                              "replay_time_ns": np.where(np.isfinite(result["time"][fail][:30]),
                                                          result["time"][fail][:30], None).tolist(),
                              "native_time_ns": (ak.to_numpy(following.time) - offsets[rows])[fail][:30].tolist()}}
        report[f"moves_{moves}"] = arm
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(encoded)
    summary = {"adjacent_updates": len(rows), "coverage": {
        k: {f: v for f, v in row.items() if f != "missing_keys"} for k, row in report["coverage"].items()}}
    summary.update({k: {label: {f: v for f, v in row.items() if f not in ("mismatch_keys", "mismatch_details")}
                       for label, row in value.items()} for k, value in report.items() if k.startswith("moves_")})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--straw-axis", type=Path, required=True)
    p.add_argument("--straw-geometry", choices=["approximate", "panel"], default="approximate")
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
