"""Recursive crossing CA history with independently generated constructors for all origins."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uproot

import f1_material_discovery_audit as K
from f1_material_discovery_audit import J, D, A, B, H, C, L, N

MATERIAL_SHA = "762f61184eec3cdf849457b1c86728fddbbd2d207bf16bec497587b00c8d707b"
PINS = {"scripts/f1_material_discovery.py": "c7d0010d6c3ea152c02c84087bac07f54e434f2aa83949f97a0476e84ffe6107",
        "scripts/f1_material_discovery_compare.py": "12b3e330276fab07a1dae313d55eba437eecb848a8010d8975bf7e61275c22e9",
        "scripts/f1_material_discovery_audit.py": "676dc1d099bd9d6210c0e302bd649434c65c17b23e1643e348b390ae701e4d91"}
LIMITS = [
    "All initial2538/associated81/material2471 constructor CA states/anchors/offsets come from independently generated pinned receipts; zero captured constructor CA bootstrap.",
    "Material candidate discovery includes the full supplied geometry universe and all rejected/reached stages; initial and associated membership remain conditioned.",
    "Native trajectory evolution, raw geometry, object chronology and partner identities remain supplied inputs, not autonomous fitting or association discovery.",
    "All105578 later searches/213617 records including seven caps; generated recursive hints and per-piece traces, no exclusions.",
    "Observed own-usable branch only; partner fallback, midpoint fallback, setHit and domain replacement not observed/data-certified.",
    "Fixed constructor offsets; final reference versus retained last-solved state preserved. No effect-time subtraction or material inversion.",
    "Later cached derivatives/variances, internal local100-step arithmetic, activity/material effects, seed/algebra/decision/acceptance/full-fit closure remain outside scope.",
]


def helpers(watched):
    result = K.helpers(watched)
    root = Path(__file__).resolve().parents[1]
    for name, sha in PINS.items():
        D.watch(root/name, watched, sha)
    path = root/"scripts/f1_discovered_history_audit.py"
    D.watch(path, watched)
    module = __import__(__name__)
    A.require(Path(module.__file__).resolve() == path, "discovered history module path differs")
    for name in ("helpers", "bootstrap", "population", "run"):
        D.code_origin(getattr(module, name, None), path)
    result["scripts/f1_discovered_history_audit.py"] = str(path)
    return result


def bootstrap(constructors, geometry, trajectories, initial, associated, material):
    """Constructor outcome columns are never consumed; only full-key/category topology."""
    ids = B.integer_keys(constructors, B.OBJECT)
    cat = A.column(constructors, "category", "iu")
    A.require(np.isin(cat, [0, 1, 2]), "invalid constructor category")
    n = len(ids)
    state = {name: np.zeros((n, 3) if name in ("particle_poca", "sensor_poca", "direction", "sensor_direction") else n,
                           dtype=bool if name == "usable" else np.int64 if name == "status" else float) for name in H.STATE}
    anchor, offset = np.zeros(n), np.zeros(n)
    index = np.zeros(n, dtype=np.int64)
    for category, receipt in ((0, initial), (1, associated), (2, material)):
        take = np.flatnonzero(cat == category)
        wanted = B.integer_matrix(receipt["identities"]["object_keys"], 5, "generated constructor keys")
        A.require(len(wanted) == len(take), "generated constructor census differs")
        join = A.match_rows(ids[take], wanted)
        generated = receipt["generated"]
        source = generated["crossing"] if category == 0 else generated["state"]
        for name in H.STATE:
            if name == "usable":
                continue
            shape = (len(wanted), 3) if state[name].ndim == 2 else (len(wanted),)
            value = L._integer(source[name], shape, name, 0, 6) if name == "status" else L._real(source[name], shape, name)
            state[name][take] = value[join]
        state["usable"][take] = state["status"][take] < 3
        anchor[take] = L._real(generated["particle_hint"], (len(wanted),), "generated anchor")[join]
        offset[take] = L._real(generated["offset"], (len(wanted),), "generated offset")[join]
        index[take] = L._integer(source["index"], (len(wanted),), "generated reference index", 0, trajectories["q"].shape[1]-1)[join]
        if "usable" in source:
            A.require(np.asarray(source["usable"]).dtype.kind == "b", "generated usable must be boolean")
            A.require(np.asarray(source["usable"])[join] == state["usable"][take], "generated usable/status differs")
    A.require(state["usable"], "unusable generated bootstrap; no exclusions")
    A.require(np.isfinite(offset) & (offset > 0), "invalid generated offset")
    parent = A.match_rows(B.integer_keys(constructors, B.TRAJECTORY, unique=False), trajectories["keys"])
    A.require(index < trajectories["counts"][parent], "generated constructor index outside trajectory")
    A.require(trajectories["category"][parent] == cat, "constructor context category differs")
    line = N.sensor(geometry["cal"], geometry["hv"], anchor, np.full(n, N.LIGHT))
    return dict(state=state, line=line, line_start=L._real(geometry["hv"], (n, 3), "tube start"), offset=offset, index=index,
                reference_parameters=trajectories["q"][parent, index], reference_range=trajectories["ranges"][parent, index])


def population(args, initial, associated, material, report):
    cycles = report["highest_cycles"] = {}
    with uproot.open(args.capture, object_cache=None, array_cache=None) as root:
        read = lambda name, fields=None: A.read_tree(root, "kktrace", name, cycles, fields)
        records = read("crossing_records")
        trajectories = H.pack(read("crossing_trajectories"))
        plan = H.schedule(records)
        report["census"] = J.census(records, trajectories, plan)
        c = report["census"]
        A.require(c["n_records"] == 213617 and c["n_objects"] == 5090 and c["n_targets"] == 147, "full history census differs")
        A.require(c["captured_unusable"] == 0 and c["update_hint_counts"]["0"] == c["update_hint_counts"]["2"] == 0, "unsupported fallback population")
        A.require(len(c["cap_keys"]) == 7, "exhausted search census differs")
        bindings, tracker = read("crossing_geometry_bindings"), read("crossing_geometry_tracker")
        straws = read("crossing_geometry_straws", ["geometry_index", "fullsid", "tube_cal", "tube_hv"])
        cal, hv, radius = B.geometry_inputs(plan["constructors"], bindings, tracker, straws)
        geometry = dict(cal=cal, hv=hv, radius=radius)
        del bindings, tracker, straws
    seed = bootstrap(plan["constructors"], geometry, trajectories, initial, associated, material)
    out, traces = H.replay(records, plan, trajectories, seed)
    report["comparison"] = H.compare(records, plan, seed, out, traces)
    rows = plan["update_rows"]
    report["n_compared_piece_solves"] = int(np.sum(out["moves"][rows]+1))
    caps = rows[out["exhausted"][rows]]
    report["caps"] = dict(keys=B.integer_keys(records[caps], B.KEYS+("sequence",)).tolist(),
        category=A.column(records[caps], "category").tolist(),
        index=out["index"][caps].tolist(), solution_index=out["solution_index"][caps].tolist(),
        visits=out["solution_history"][caps].tolist(), traces={k: v[caps].tolist() for k, v in traces.items()},
        fixed_offset=seed["offset"][plan["parent"][caps]].tolist(), effect_time=out["effect_time"][caps].tolist())
    report["bootstrap"] = dict(independently_reconstructed_constructors=5090, captured_material_constructors=0,
                                material_discovery_replayed=True, initial_associated_membership_conditioned=True)


def run(args):
    names = ("out", "full_check", "initial", "associated", "history", "material", "capture", "native_source")
    args = SimpleNamespace(**{k: Path(getattr(args, k)).resolve() for k in names})
    A.require(args.out not in [getattr(args, k) for k in names[1:]] and not args.out.is_relative_to(Path(__file__).resolve().parents[1])
              and not args.out.is_relative_to(args.native_source), "protected report path")
    with args.out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = dict(completed=False, passed=False, exact=False, full_fit_closure=False, full_f1_closure=False,
                      full_history_closure=False, full_numerical_validation=False, generated_constructor_ca_history=False,
                      rtol=B.RTOL, atol=B.ATOL, limits=LIMITS)
        try:
            report["executing_modules"] = helpers(watched)
            full = D.receipt(args.full_check, D.FULL_SHA, watched, ("passed", "full_target_capture_valid", "hashes_unchanged", "full_target_coverage"))
            initial = D.receipt(args.initial, J.INITIAL_SHA, watched, ("completed", "passed", "exact", "hashes_unchanged"))
            associated = D.receipt(args.associated, J.J.ASSOCIATED_SHA, watched, ("completed", "passed", "hashes_unchanged"))
            A.require(associated["comparison"]["exact"] is True, "associated certificate not exact")
            D.receipt(args.history, K.HISTORY_SHA, watched, ("completed", "passed", "exact", "hashes_unchanged"))
            material = D.receipt(args.material, MATERIAL_SHA, watched, ("completed", "passed", "exact", "hashes_unchanged", "independent_discovery_replayed"))
            J.J.bound_input(args.capture, full["before_sha256"], watched)
            A.require(full["before_sha256"][str(args.capture)] == D.CAPTURE_SHA, "wrong capture")
            for relative in ("KinKal/Fit/Track.hh", "Offline/Mu2eKinKal/inc/KKStrawXing.hh", "KinKal/Trajectory/PiecewiseClosestApproach.hh"):
                J.J.bound_input(args.native_source/relative, full["before_sha256"], watched)
            report["inputs"] = {k: str(getattr(args, k)) for k in names[1:]}
            population(args, initial, associated, material, report)
            report.update(completed=True, passed=report["comparison"]["passed"], exact=report["comparison"]["exact"],
                          generated_constructor_ca_history=report["comparison"]["passed"])
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
                report.update(passed=False, generated_constructor_ca_history=False, integrity_error="Evidence/source changed during audit")
            report, text = D.serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-check", "initial", "associated", "history", "material", "capture", "native-source", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "passed", "exact", "hashes_unchanged", "error", "bootstrap", "n_compared_piece_solves")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
