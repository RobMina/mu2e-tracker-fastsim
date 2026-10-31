"""Saved guarded-producer full-denominator, stage-split and refusal-evidence audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ordered_steps_audit import audit as audit_ledgers
from scripts.transport_capture_audit import digest
from trkfastsim import orderedhandoff, tables

KEY = tables.KEY + ["sp"]


def identity_frame(frame, label):
    if not all(key in frame for key in KEY) or frame.duplicated(KEY).any():
        raise ValueError(label + " has incomplete or duplicate primary identities")
    if not np.isfinite(frame[KEY].to_numpy(float)).all() or not np.equal(frame[KEY].to_numpy(), frame[KEY].to_numpy(np.int64)).all():
        raise ValueError(label + " primary identities must be finite integers")
    return pd.MultiIndex.from_frame(frame[KEY])


def audit_handoff(inventory, handoff, plane):
    keys = identity_frame(inventory, "input inventory")
    if not keys.equals(identity_frame(handoff, "handoff")):
        raise ValueError("handoff lost/reordered an input primary")
    live = handoff.alive_world.to_numpy(bool)
    accepted = handoff.accepted_forward.to_numpy(bool)
    if not np.array_equal(live, accepted) or not handoff.disposition.isin(["not-delivered-by-world", "forward-handoff"]).all():
        raise ValueError("producer completed with a refused handoff lane")
    if ((handoff.disposition == "forward-handoff").to_numpy() != live).any():
        raise ValueError("handoff disposition contradicts world-delivered flag")
    values = handoff.loc[live, ["x", "y", "z", "ux", "uy", "uz", "p_handoff_MeV_c", "path_offset_mm", "world_dt_ns"]].to_numpy(float)
    if not np.isfinite(values).all() or (handoff.loc[live, "world_dt_ns"] < 0).any() or (handoff.loc[live, "path_offset_mm"] < 0).any():
        raise ValueError("nonfinite/negative delivered handoff state")
    if (handoff.loc[live, "uy"] >= 0).any():
        raise ValueError("nonforward handed-off primary")
    error = np.abs(handoff.loc[live, "y"].to_numpy() - plane)
    if (error > handoff.loc[live, "plane_roundoff_guard_mm"].to_numpy()).any():
        raise ValueError("interior handoff position differs from world endpoint")
    return dict(input_primaries=len(inventory), world_delivered=int(live.sum()), world_undelivered=int((~live).sum()),
                max_plane_residual_mm=float(error.max(initial=0.)))


def audit_queries(directory, handoff, plane):
    active = handoff.loc[handoff.accepted_forward, KEY].to_numpy(np.int64)
    reports = sorted(Path(directory).glob("navigation_*/receipt.json"))
    calls = []
    for path in reports:
        receipt = json.loads(path.read_text())
        with np.load(path.parent / "rays.npz", allow_pickle=False) as rays:
            if not all(key in rays for key in ("primary_keys", "interior_lanes", "origin", "direction", "lengths")):
                raise ValueError("fresh query lacks full primary/lane addressing")
            lane = rays["interior_lanes"]
            if lane.dtype.kind not in "iu" or (lane < 0).any() or (lane >= len(active)).any() or len(np.unique(lane)) != len(lane):
                raise ValueError("fresh query lane map invalid")
            if not np.array_equal(rays["primary_keys"], active[lane]):
                raise ValueError("fresh query primary identities disagree with handoff")
            if receipt["status"] == "complete":
                orderedhandoff.validate_forward_query(rays["origin"], rays["direction"], rays["lengths"], plane)
                from trkfastsim.guardednavigation import replacement_domain
                cells = pd.read_csv(path.parent / "cells.csv")
                allowed, strict, contacts = replacement_domain(cells, rays["origin"], rays["direction"], plane,
                    allow_start_contact=receipt.get("allow_start_contact", False))
                if not allowed or receipt["replacements_strictly_below_plane"] != strict:
                    raise ValueError("replacement plane evidence/domain disagrees with saved rays")
                if receipt.get("allow_start_contact", False) and receipt.get("replacements_within_forward_domain") is not True:
                    raise ValueError("native exact start contact lacks explicit domain admission")
                for arm in ("inner", "outer"):
                    rows = pd.read_csv(path.parent / f"refined.{arm}.csv")
                    if len(rows) and not rows.lane.isin(np.arange(len(lane))).all():
                        raise ValueError("retained refinement uses remapped rather than query-local lanes")
                if not receipt["envelope_summary"]["all_pairs_admitted"]:
                    raise ValueError("advanced a refused paired geometry query")
            calls.append(dict(index=path.parent.name, status=receipt["status"], stage=receipt["stage"], rays=len(lane)))
    return calls


def audit_forward_refusal(path, handoff, receipt):
    if not path.is_file() or handoff is None:
        raise ValueError("declared forward refusal lacks ray/primary evidence")
    active = handoff.loc[handoff.accepted_forward, KEY].to_numpy(np.int64)
    with np.load(path, allow_pickle=False) as saved:
        lanes, keys = saved["interior_lanes"], saved["primary_keys"]
        if (lanes < 0).any() or (lanes >= len(active)).any() or not np.array_equal(active[lanes], keys):
            raise ValueError("forward refusal primary/lane mapping differs from handoff")
        if int(saved["iteration"]) != receipt["forward_refusal_iteration"] or keys.tolist() != receipt["forward_refusal_primary_keys"]:
            raise ValueError("forward refusal iteration/keys differ from receipt")
        try:
            orderedhandoff.validate_forward_query(saved["origin"], saved["direction"], saved["lengths"], float(saved["plane_y_mm"]))
        except ValueError:
            return dict(iteration=int(saved["iteration"]), primary_keys=keys.tolist())
    raise ValueError("declared forward refusal does not violate domain")


def run(directory, out):
    directory, out = Path(directory), Path(out)
    receipt = json.loads((directory / "complete.json").read_text())
    paths = [p for p in directory.rglob("*") if p.is_file()]
    hashes = {str(p): digest(p) for p in paths}
    inventory = pd.read_csv(directory / "input_primaries.csv")
    events = pd.read_csv(directory / "generated_events.csv")
    if len(events) != receipt["generated_events"] or events.duplicated(tables.KEY).any():
        raise ValueError("independent requested generated-event coverage failed")
    handoff = pd.read_csv(directory / "handoff_primaries.csv") if (directory / "handoff_primaries.csv").exists() else None
    result = dict(scope="computational split/denominator/refusal evidence, not G4 or world material ownership", status=receipt["status"],
        requested_events=len(events), input_primaries=len(inventory), physics_emitted=(directory / "ordered_steps.root").exists())
    identity_frame(inventory, "inventory")
    if receipt["status"] in ("refused", "implementation-error"):
        if result["physics_emitted"]:
            raise ValueError("refused producer published physics sample")
        result.update(refusal_reason=receipt["reason"], handoff_evidence_status=receipt["handoff_evidence_status"])
        if handoff is not None:
            if not identity_frame(inventory, "inventory").equals(identity_frame(handoff, "handoff")):
                raise ValueError("refusal inventory lost primary keys")
            result["world_delivered"] = int(handoff.alive_world.sum())
    elif receipt["status"] == "bounded_full_chain_complete":
        if handoff is None:
            raise ValueError("successful producer missing handoff")
        result["handoff"] = audit_handoff(inventory, handoff, receipt["handoff"]["plane_y_mm"])
        mu, segments, vertices = [pd.read_csv(directory / (name + ".csv")) for name in ("muons", "segments", "vertices")]
        if not identity_frame(inventory, "inventory").equals(identity_frame(mu, "muon ledger")):
            raise ValueError("muon ledger excludes input primary population")
        result["ledger"] = audit_ledgers(mu, segments, vertices)
        generated = tables.read(directory / "ordered_steps.root", "gen")
        if not pd.MultiIndex.from_frame(generated[tables.KEY].drop_duplicates()).equals(pd.MultiIndex.from_frame(events)):
            raise ValueError("published generated-event denominator changed")
        live = handoff.accepted_forward.to_numpy(bool)
        active_mu = mu.loc[live]
        if not np.allclose(active_mu.t_interior_start_ns.to_numpy() - handoff.loc[live, "world_dt_ns"].to_numpy(),
                           steps_times(generated, inventory)[live], rtol=0, atol=1e-9):
            raise ValueError("interior start clock subtracts an untraversed backprojection")
    else:
        raise ValueError("recognized completed or refused producer receipt required")
    if "forward_refusal_iteration" in receipt:
        result["forward_refusal"] = audit_forward_refusal(directory / "forward_refusal.npz", handoff, receipt)
    if handoff is not None:
        result["queries"] = audit_queries(directory, handoff, receipt["handoff"]["plane_y_mm"])
        if len(result["queries"]) != receipt.get("navigation_queries", 0):
            raise ValueError("navigation-query evidence missing")
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("saved producer products changed during audit")
    result.update(input_sha256=hashes, source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2, allow_nan=False))
    return result


def steps_times(gen, inventory):
    from trkfastsim import steps
    primary = steps._muons(gen, True)
    if not np.array_equal(primary[tables.KEY + ["mukey"]].to_numpy(), inventory[KEY].to_numpy()):
        raise ValueError("generation time inventory differs from handoff identities")
    return primary.t.to_numpy(float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.out)
