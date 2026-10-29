"""Pinned all-record crossing CA history on supplied native trajectory evolution."""
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uproot

import f1_initial_crossing_audit as J
import f1_crossing_history as H
from f1_initial_crossing_audit import D, I, A, B, C, L, N

INITIAL_SHA = "100542dfd9b0f6287c06b79c4a3a775883c1d602580418f703112fbc5a2acbf7"
PINS = {"scripts/f1_initial_crossing.py": "3523bbeecbd4674fcb45f7e82aa75021cbbeba609062c200b647b61d75210c2b",
        "scripts/f1_initial_crossing_audit.py": "b7ffbe665b73aa5155e08fb9ac09152ec9df9cd20ab20d6218efba2aa7a0df26"}
LIMITS = [
    "All213617 crossing records/5090objects/147targets; no excluded updates or seven exhausted searches.",
    "Initial2538 and associated81 states/hints/offsets inherited from independently generated pinned constructor receipts.",
    "Material2471 constructor CA state, line anchor and fixed private offset are EXPLICIT CAPTURED INPUTS until independent discovery replay.",
    "Raw geometry, native trajectory evolution, record membership/order and constructor/setHit partner identities are supplied inputs, not autonomous fitting/association discovery.",
    "Later crossing hints and CA state recursively use generated predecessors, never captured later times/indices/hints/status.",
    "Observed own-usable hint branch only; any generated unusable state fails the entire replay. No native partner-hit/midpoint fallback claim.",
    "Per-piece incoming hints/solved times/status/usability are compared; intermediate internal local-solver updates, cached derivatives/variances, activity/material policy are not replayed.",
    "Full fit/F1/numerical/independent discovery/complete partner lifecycle remain open; no fitter/default/tolerance/regularization change.",
]


def helpers(watched):
    result = J.helpers(watched)
    root = Path(__file__).resolve().parents[1]
    for name, sha in PINS.items():
        D.watch(root/name, watched, sha)
    modules = {"scripts/f1_crossing_history.py": H, "scripts/f1_crossing_history_audit.py": __import__(__name__)}
    required = {"scripts/f1_crossing_history.py": ("pack", "schedule", "signed_partner", "partners", "bootstrap", "solve_trace", "replay", "compare"),
                "scripts/f1_crossing_history_audit.py": ("helpers", "census", "population", "run")}
    for name, module in modules.items():
        path = root/name
        A.require(Path(module.__file__).resolve() == path, "module path differs: "+name)
        D.watch(path, watched)
        for field in required[name]:
            D.code_origin(getattr(module, field, None), path)
        result[name] = str(path)
    A.require(H.B is B and H.L is L and H.N is N and H.C is C, "history module binding differs")
    for name in ("array", "column", "dense", "match_rows", "require"):
        A.require(getattr(H, name) is getattr(A, name), "history imported alias differs: "+name)
    for field in ("helpers", "bound_input"):
        D.code_origin(getattr(J, field), root/"scripts/f1_initial_crossing_audit.py")
    for field in ("crossing_hint", "_state_times", "_usable", "_midpoint"):
        D.code_origin(getattr(L, field), root/"scripts/f1_later_ca.py")
    return result


def census(records, trajectories, plan):
    stage = A.column(records, "stage", "iu")
    category = A.column(records, "category", "iu")
    updates = records[plan["update_rows"]]
    hint = A.column(updates, "hint_kind", "iu")
    cap = B.boolean(records, "search_exhausted")
    return dict(n_records=len(records), n_objects=len(plan["constructors"]),
                n_targets=len(np.unique(B.integer_keys(records, B.KEYS, unique=False), axis=0)),
                stage_counts={str(s): int(np.sum(stage == s)) for s in range(5)},
                update_hint_counts={str(h): int(np.sum(hint == h)) for h in range(3)},
                captured_unusable=int(np.sum(~B.boolean(records, "ca_usable"))),
                n_waves=plan["waves"].shape[1],
                cap_keys=B.integer_keys(records[cap], B.KEYS+("sequence",)).tolist(),
                cap_categories=category[cap].tolist(),
                trajectory_categories={str(c): int(np.sum(trajectories["category"] == c)) for c in range(6)})


def population(args, initial, associated, report):
    cycles = report["highest_cycles"] = {}
    with uproot.open(args.capture, object_cache=None, array_cache=None) as root:
        read = lambda name, fields=None: A.read_tree(root, "kktrace", name, cycles, fields)
        records = read("crossing_records")
        trajectories = H.pack(read("crossing_trajectories"))
        plan = H.schedule(records)
        report["census"] = census(records, trajectories, plan)
        c = report["census"]
        A.require(c["n_records"] == 213617 and c["n_objects"] == 5090 and c["n_targets"] == 147, "full history census differs")
        A.require(c["captured_unusable"] == 0 and c["update_hint_counts"]["0"] == c["update_hint_counts"]["2"] == 0,
                  "captured fallback branch not in declared replay")
        A.require(len(c["cap_keys"]) == 7, "exhausted search census differs")
        bindings, tracker = read("crossing_geometry_bindings"), read("crossing_geometry_tracker")
        straws = read("crossing_geometry_straws", ["geometry_index", "fullsid", "tube_cal", "tube_hv"])
        cal, hv, radius = B.geometry_inputs(plan["constructors"], bindings, tracker, straws)
        geometry = dict(cal=cal, hv=hv, radius=radius)
        del bindings, tracker, straws
    seed = H.bootstrap(plan["constructors"], geometry, trajectories, initial, associated)
    out, trace = H.replay(records, plan, trajectories, seed)
    report["comparison"] = H.compare(records, plan, seed, out, trace)
    rows = plan["update_rows"]
    exhausted = rows[out["exhausted"][rows]]
    ids = B.integer_keys(records, B.KEYS+("sequence",))
    report["caps"] = dict(keys=ids[exhausted].tolist(),
        categories=A.column(records, "category")[exhausted].tolist(),
        index=out["index"][exhausted].tolist(), solution_index=out["solution_index"][exhausted].tolist(),
        particle_time=out["particle_time"][exhausted].tolist(), sensor_time=out["sensor_time"][exhausted].tolist(),
        status=out["status"][exhausted].tolist(), visits=out["solution_history"][exhausted].tolist(),
        traces={k: v[exhausted].tolist() for k, v in trace.items()})
    report["n_compared_piece_solves"] = int(np.sum(out["moves"][rows]+1))
    cap_parent = A.match_rows(B.integer_keys(records[exhausted], B.TRAJECTORY, unique=False), trajectories["keys"])
    report["caps"]["reference_parameters"] = out["reference_parameters"][exhausted].tolist()
    report["caps"]["last_solved_parameters"] = trajectories["q"][cap_parent, out["solution_index"][exhausted]].tolist()
    report["caps"]["fixed_offset"] = seed["offset"][plan["parent"][exhausted]].tolist()
    report["caps"]["effect_time"] = out["effect_time"][exhausted].tolist()
    report["by_category"] = {}
    for category in range(3):
        take = A.column(records, "category")[rows] == category
        report["by_category"][str(category)] = dict(n_updates=int(take.sum()), n_moved=int(np.sum(out["moves"][rows][take] > 0)),
                                                   n_exhausted=int(out["exhausted"][rows][take].sum()))
    report["bootstrap"] = dict(independently_reconstructed_constructors=2619, captured_material_constructors=2471,
                                material_constructor_independence=False, association_discovery_independence=False)


def run(args):
    args = SimpleNamespace(**{k: Path(getattr(args, k)).resolve() for k in
                              ("out", "full_check", "initial", "associated", "capture", "native_source")})
    A.require(args.out not in [args.full_check, args.initial, args.associated, args.capture]
              and not args.out.is_relative_to(Path(__file__).resolve().parents[1])
              and not args.out.is_relative_to(args.native_source), "protected report path")
    with args.out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = dict(completed=False, passed=False, full_fit_closure=False, full_f1_closure=False, full_history_closure=False,
                      full_numerical_validation=False, independent_discovery_replayed=False, rtol=B.RTOL, atol=B.ATOL, limits=LIMITS)
        try:
            report["executing_modules"] = helpers(watched)
            full = D.receipt(args.full_check, D.FULL_SHA, watched,
                             ("passed", "full_target_capture_valid", "hashes_unchanged", "full_target_coverage"))
            initial = D.receipt(args.initial, INITIAL_SHA, watched, ("completed", "passed", "exact", "hashes_unchanged"))
            associated = D.receipt(args.associated, J.ASSOCIATED_SHA, watched, ("completed", "passed", "hashes_unchanged"))
            A.require(associated["comparison"]["exact"] is True, "associated certificate not exact")
            J.bound_input(args.capture, full["before_sha256"], watched)
            A.require(full["before_sha256"][str(args.capture)] == D.CAPTURE_SHA, "wrong full capture")
            for relative, sha in (("KinKal/Fit/Material.hh", "bc45cc89e1521b47382c67ad42d9476cfe4e5d6b070974e62a601f408136b6d4"),
                                  ("KinKal/Fit/MetaIterConfig.hh", "1ec563ca47b69166ea11a7f4048299459e3cf00fa1d69132fdd12619eeafa2be")):
                D.watch(args.native_source/relative, watched, sha)
            expanded = J.bound_input(args.capture.parent/"capture.expanded.fcl", full["before_sha256"], watched)
            A.require("XingReferencePiece" not in expanded.read_text(), "nondefault crossing strategy unsupported")
            for relative in ("KinKal/Fit/Track.hh", "Offline/Mu2eKinKal/inc/KKStrawXing.hh",
                             "KinKal/Trajectory/PiecewiseClosestApproach.hh"):
                J.bound_input(args.native_source/relative, full["before_sha256"], watched)
            report["inputs"] = {k: str(getattr(args, k)) for k in ("capture", "full_check", "initial", "associated", "native_source")}
            population(args, initial, associated, report)
            report.update(completed=True, passed=report["comparison"]["passed"], exact=report["comparison"]["exact"])
        except Exception as error:
            report.update(error=f"{type(error).__name__}: {error}", passed=False)
        finally:
            end, errors = {}, []
            for path in watched:
                try:
                    end[path] = A.digest(path)
                except Exception as error:
                    errors.append(f"{path}: {type(error).__name__}: {error}")
            report.update(before_sha256=watched, end_sha256=end, hashes_unchanged=bool(watched) and end == watched and not errors, hash_errors=errors)
            if not report["hashes_unchanged"]:
                report.update(passed=False, integrity_error="Evidence/source changed during audit")
            report, text = D.serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-check", "initial", "associated", "capture", "native-source", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "passed", "exact", "hashes_unchanged", "error", "census")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
