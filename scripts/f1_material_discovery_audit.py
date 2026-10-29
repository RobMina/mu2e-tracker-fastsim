"""Pinned full-universe normal-branch material discovery audit, no native rerun."""
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uproot

import f1_crossing_history_audit as J
import f1_material_discovery as M
import f1_material_discovery_compare as V
from f1_crossing_history_audit import D, A, B, H, C, L, N

HISTORY_SHA = "d46ac5ae41ce5c708c9e4751be27b0f58106be99e7b23f381fe43e7873621049"
PINS = {"scripts/f1_crossing_history.py": "0168b273a0a76d962b946403c45b94a15bc241475ce992d1d7a362649815b57f",
        "scripts/f1_crossing_history_audit.py": "82bbd1c81e6a8e6b90a6aa2065d48a6dd45932e11fced607a61fa726cbe3430b"}
SOURCES = {"KinKal/Trajectory/PiecewiseTrajectory.hh": "d7f91b8b54a9895141c7de21476e7dea78afbe77acba7b5d9d3039eafcbaa81d",
           "KinKal/Trajectory/ClosestApproach.hh": "ba54d8d91036d0085fed44ce432ff8b1926352ed8a88287b476f8eb4706d66d1",
           "KinKal/Trajectory/ParticleTrajectory.hh": "0da161186e3a73d3ad18d7d48781f5c29efda8eba9c374f806ee4d2fcf705645",
           "Offline/Mu2eKinKal/inc/KKFitUtilities.hh": "3f931da21e5944b71833162eb01dff02b1cf3a49466159fbde55070ccb526172",
           "Offline/TrackerGeom/inc/Straw.hh": "a651a9282959599c0bd7e77441aeed19e2e0ed4a5f311cb184b174f739075fe3"}
LIMITS = [
    "Raw complete target geometry and native material-entry trajectories/covariances/mass are supplied; not autonomous fitting or conditions generation.",
    "Single material extension per target; initial/associated existing membership comes from pinned constructor receipts, not discovery outcomes.",
    "All normal-branch plane/panel/window/exclusion/PCA/strict-cut stages generated including rejected and parent-skipped candidates; captured outcomes comparison-only.",
    "Frozen MinDz0/MaxDStraw2/DOCA5/consistency1/Ubuffer0/Nbuffer0/Skipfalse only; vertical branch unreachable and not certified.",
    "CA derivatives/variance use last-solved piece; fixed constructor offset uses final-reference speed. No effect-time subtraction/material inversion.",
    "Fixed source-grouped Similarity is not universal ROOT/compiler parity. Exactness is reported separately without changing tolerances or selection cuts.",
    "Existing sealed history v2 still uses captured material initialization until a separate new history certificate; activity/material effects remain outside scope.",
    "No full-fit/F1/seed/numerical/decision/acceptance or absent partner-fallback/setHit/domain-replacement closure.",
]


def helpers(watched):
    result = J.helpers(watched)
    root = Path(__file__).resolve().parents[1]
    for name, sha in PINS.items():
        D.watch(root/name, watched, sha)
    modules = {"scripts/f1_material_discovery.py": M, "scripts/f1_material_discovery_compare.py": V,
               "scripts/f1_material_discovery_audit.py": __import__(__name__)}
    required = {"scripts/f1_material_discovery.py": ("configuration", "pack", "geometry", "exclusions", "transform", "tzero", "ztime", "similarity", "material_cut", "generate"),
                "scripts/f1_material_discovery_compare.py": ("stage_keys", "compare"),
                "scripts/f1_material_discovery_audit.py": ("helpers", "population", "run")}
    for name, module in modules.items():
        path = root/name
        A.require(Path(module.__file__).resolve() == path, "module path differs: "+name)
        D.watch(path, watched)
        for field in required[name]:
            D.code_origin(getattr(module, field, None), path)
        for value in vars(module).values():
            if inspect.isfunction(value) and value.__module__ == module.__name__:
                D.code_origin(value, path)
        result[name] = str(path)
    A.require(M.H is H and M.B is B and M.L is L and M.N is N and M.C is C and V.M is M, "material module binding differs")
    for module in (M, V):
        for name in ("array", "column", "dense", "match_rows", "require"):
            A.require(getattr(module, name) is getattr(A, name), "material imported alias differs: "+name)
    for name in ("kinematics", "rows_at_state", "_contract3", "_matvec6"):
        D.code_origin(getattr(N, name), root/"trkfastsim/reco/nativearith.py")
    return result


def population(args, initial, associated, report):
    cycles = report["highest_cycles"] = {}
    with uproot.open(args.capture, object_cache=None, array_cache=None) as root:
        read = lambda name, fields=None: A.read_tree(root, "kktrace", name, cycles, fields)
        t = M.pack(read("crossing_trajectories"))
        A.require(len(t["keys"]) == 142, "material context census differs")
        records = read("crossing_records")
        constructors = records[(A.column(records, "stage") == 0) & (A.column(records, "category") == 2)]
        A.require(len(constructors) == 2471, "material constructor census differs")
        del records
        discovery = read("crossing_discovery")
        A.require(len(discovery) == 38174, "discovery census differs")
        tables = {name: read("crossing_geometry_"+name) for name in ("bindings", "tracker", "planes", "panels")}
        fields = ["geometry_index", "tracker_straw_ordinal", "fullsid", "tube_cal", "tube_hv", "wire_cal", "wire_midpoint", "wire_direction", "half_length", "straw_ordinal"]
        tables["straws"] = read("crossing_geometry_straws", fields)
        g = M.geometry(tables)
        A.require(len(g["gid"]) == 147, "full geometry target census differs")
        del tables
    old = M.exclusions(t, g, initial, associated)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = M.generate(t, g, old)
    report["generated_census"] = {str(k): dict(n_rows=len(v["parent"]), n_passed=int(v["passed"].sum())) for k, v in out["stages"].items()}
    report["universe"] = out["universe"]
    report["searches"] = dict(tzero_max_tries=int(out["tzero_tries"].max()), ztime_max_tries=int(out["ztime_tries"].max()),
                              ztime_caps=int(out["ztime_caps"].sum()), ca_moved=int((out["generated"]["state"]["moves"] > 0).sum()),
                              ca_caps=int(out["generated"]["state"]["exhausted"].sum()))
    compared = V.compare(out, t, discovery, constructors)
    report["comparison"] = {k: v for k, v in compared.items() if k not in ("generated", "object_keys")}
    generated = compared["generated"]
    report["identities"] = dict(object_keys=compared["object_keys"].tolist(), strawid=out["stages"][8]["straw"].tolist())
    report["generated"] = {k: generated[k].tolist() for k in ("offset", "effect_time", "reference_speed", "particle_hint", "sensor_hint")}
    report["generated"]["state"] = {k: generated["state"][k].tolist() for k in (*H.STATE, "index", "solution_index", "moves", "exhausted", "solution_history")}
    report["conditioning"] = dict(native_trajectories=True, native_covariances=True, raw_geometry=True,
                                    initial_associated_membership=True, captured_material_constructor_states=False,
                                    captured_discovery_outcomes=False, autonomous_fit=False)


def run(args):
    names = ("out", "full_check", "initial", "associated", "history", "capture", "native_source")
    args = SimpleNamespace(**{k: Path(getattr(args, k)).resolve() for k in names})
    A.require(args.out not in [getattr(args, k) for k in names[1:]] and not args.out.is_relative_to(Path(__file__).resolve().parents[1])
              and not args.out.is_relative_to(args.native_source), "protected report path")
    with args.out.open("x", encoding="utf-8") as stream:
        watched = {}
        report = dict(completed=False, passed=False, exact=False, full_fit_closure=False, full_f1_closure=False,
                      full_history_closure=False, full_numerical_validation=False, independent_discovery_replayed=False,
                      rtol=B.RTOL, atol=B.ATOL, limits=LIMITS)
        try:
            report["executing_modules"] = helpers(watched)
            full = D.receipt(args.full_check, D.FULL_SHA, watched, ("passed", "full_target_capture_valid", "hashes_unchanged", "full_target_coverage"))
            initial = D.receipt(args.initial, J.INITIAL_SHA, watched, ("completed", "passed", "exact", "hashes_unchanged"))
            associated = D.receipt(args.associated, J.J.ASSOCIATED_SHA, watched, ("completed", "passed", "hashes_unchanged"))
            A.require(associated["comparison"]["exact"] is True, "associated certificate not exact")
            D.receipt(args.history, HISTORY_SHA, watched, ("completed", "passed", "exact", "hashes_unchanged"))
            J.J.bound_input(args.capture, full["before_sha256"], watched)
            A.require(full["before_sha256"][str(args.capture)] == D.CAPTURE_SHA, "wrong capture")
            for relative in ("Offline/Mu2eKinKal/inc/KKFit.hh", "Offline/Mu2eKinKal/inc/KKStrawXing.hh", "Offline/Mu2eKinKal/src/KKFitUtilities.cc", "KinKal/Trajectory/PiecewiseClosestApproach.hh"):
                J.J.bound_input(args.native_source/relative, full["before_sha256"], watched)
            for relative, sha in SOURCES.items():
                D.watch(args.native_source/relative, watched, sha)
            expanded = J.J.bound_input(args.capture.parent/"capture.expanded.fcl", full["before_sha256"], watched)
            report["configuration"] = M.configuration(expanded)
            report["inputs"] = {k: str(getattr(args, k)) for k in names[1:]}
            population(args, initial, associated, report)
            report.update(completed=True, passed=report["comparison"]["passed"], exact=report["comparison"]["exact"],
                          independent_discovery_replayed=report["comparison"]["passed"])
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
                report.update(passed=False, independent_discovery_replayed=False, integrity_error="Evidence/source changed during audit")
            report, text = D.serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-check", "initial", "associated", "history", "capture", "native-source", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report.get(k) for k in ("completed", "passed", "exact", "hashes_unchanged", "error", "generated_census", "searches")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
