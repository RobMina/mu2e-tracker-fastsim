"""Saved first-iteration state/variance/bound diagnostics; no fitter execution."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import uproot

import f1_momentum_trace as M
import f1_hit_operand_check as H
from f1_momentum_trace import A, B, C, D, R, T, fittrace

TRACE_SHA = "42675e2f6328344db511a6de839691cef8b983cb3274e24dc00c3b36dd771078"
LIMITS = [
    "One saved compatible fixed-state momentum trace, iteration zero only; no refit or replacement of sealed outputs.",
    "Fixed final stock states differ intentionally from native initial states; mismatches do not establish the cause of final momentum error.",
    "All state/row participation disagreements retained; variance/weight metrics use explicitly counted common-active rows only.",
    "Crossed-mask bounds are labelled diagnostics using captured times/masks; not an independently generated fit or correction model.",
    "No residual/derivative equality, later trajectory, matrix algebra, decision/acceptance or full numerical closure claim.",
]


def bounds(times, active):
    times, active = np.asarray(times), np.asarray(active)
    A.require(times.ndim == 1 and active.shape == times.shape and active.dtype.kind == "b", "invalid bound operands")
    A.require(np.isfinite(times).all() and active.any(), "nonfinite/empty bound operands")
    lo = int(np.argmin(np.where(active, times, np.inf)))
    hi = int(np.argmax(np.where(active, times, -np.inf)))
    return dict(values=[float(times[lo]), float(times[hi])], source_slots=[lo, hi])


def state_variance(t, identities, iteration, effects, rows, scale):
    check = H.validate(rows, effects, iteration)
    A.require(check["passed"], "captured native row algebra failed")
    hids, xids, seed = identities["hit_ids"], identities["crossing_ids"], identities["seed_hit_mask"]
    nh, nx = len(hids), len(xids)
    A.require(len(np.unique(hids)) == nh and len(np.unique(xids)) == nx, "ambiguous source identities")
    slots = np.flatnonzero(seed)
    hits = effects[A.column(effects, "kind") == 1]
    A.require(len(hits) == len(slots), "hit membership differs")
    join = A.match_rows(hids[slots, None], A.column(hits, "shindex")[:, None])
    hits = hits[join]
    rjoin = A.match_rows(A.keys(hits, H.HIT_KEYS), A.keys(rows, H.HIT_KEYS))
    native = rows[rjoin]
    order = np.asarray(t["order"])
    A.require(np.array_equal(np.sort(order), np.arange(nh+nx+1)), "invalid effect order")
    natural = {k: np.asarray(v)[np.argsort(order)] for k, v in t.items() if k.startswith("eff_") or k == "is_material"}
    actual_w = natural["eff_w"][slots]
    actual_var = natural["eff_mvar"][slots]
    native_active = A.dense(native, "row_active", 3).astype(bool)
    native_var = A.dense(native, "row_mvar", 3)
    native_scale = A.column(native, "variance_scale")
    A.require(np.isfinite(scale) and scale > 0 and np.all(native_scale == scale), "native/F1 variance scale differs")
    A.require(np.isfinite(actual_w).all() and np.all(actual_w >= 0), "invalid saved F1 weights")
    actual_active = actual_w > 0
    A.require(np.all(actual_active == t["hit_active"][slots, None]), "F1 row/hit activity inconsistent")
    common = actual_active & native_active
    native_w = np.divide(1., np.where(native_active, native_var, 1.)*native_scale[:, None],
                         out=np.zeros_like(native_var), where=native_active)
    metrics = {}
    for i, name in enumerate(("distance", "time", "longitudinal")):
        use = common[:, i]
        metrics[name] = dict(n_common=int(use.sum()), n_f1_only=int((actual_active[:, i] & ~native_active[:, i]).sum()),
            n_native_only=int((~actual_active[:, i] & native_active[:, i]).sum()), n_neither=int((~actual_active[:, i] & ~native_active[:, i]).sum()),
            source_shindex=hids[slots[use]].tolist(), mvar=T._metric(actual_var[use, i], native_var[use, i]),
            weight=T._metric(actual_w[use, i], native_w[use, i]))
    native_hit_active = H.boolean(native, "active")
    states = dict(source_shindex=hids[slots].tolist(), f1_active=t["hit_active"][slots].tolist(),
        native_active=native_hit_active.tolist(), activity_different_shindex=hids[slots[t["hit_active"][slots] != native_hit_active]].tolist(),
        f1_lr=t["hit_lr"][slots].tolist(), native_hit_state=A.column(native, "hit_state").tolist(),
        f1_row_active=actual_active.tolist(), native_row_active=native_active.tolist(),
        row_mismatch_indices=np.argwhere(actual_active != native_active).tolist(),
        f1_mvar=C.safe_numeric(actual_var), native_mvar=C.safe_numeric(native_var),
        f1_weight=C.safe_numeric(actual_w), native_weight=C.safe_numeric(native_w), variance_scale=scale,
        row_comparisons=metrics)
    constraints = effects[A.column(effects, "kind") == 4]
    A.require(len(constraints) == 1 and H.boolean(constraints, "active").all(), "unsupported constraint activity")
    A.require(np.isin(A.column(effects, "kind"), [1, 2, 4]).all(), "unsupported measurement kinds")
    measurement_slots = np.r_[slots, nh+nx]
    ft = t["physical_time"][measurement_slots]
    nt = np.r_[A.column(hits, "time"), A.column(constraints, "time")]
    fa, na = np.r_[t["hit_active"][slots], True], np.r_[native_hit_active, True]
    native_bounds = np.array([A.column(iteration, field)[0] for field in ("range_begin", "range_end")])
    br = dict(source_shindex_or_constraint=np.r_[hids[slots], -1].tolist(), f1_times=ft.tolist(), native_times=nt.tolist(),
              f1_mask=fa.tolist(), native_mask=na.tolist(), saved_f1=t["bounds"].tolist(), saved_native=native_bounds.tolist(),
              f1_times_f1_mask=bounds(ft, fa), native_times_native_mask=bounds(nt, na),
              diagnostic_native_times_f1_mask=bounds(nt, fa), diagnostic_f1_times_native_mask=bounds(ft, na))
    br["f1_reconstruction"] = T._metric(br["f1_times_f1_mask"]["values"], t["bounds"])
    br["native_reconstruction"] = T._metric(br["native_times_native_mask"]["values"], native_bounds)
    A.require(br["f1_reconstruction"]["passed"] and br["native_reconstruction"]["passed"], "bound reconstruction failed")
    crossings = effects[A.column(effects, "kind") == 2]
    xpos = A.match_rows(A.column(crossings, "strawid")[:, None], xids[:, None])
    native_xactive, native_infit = H.boolean(crossings, "active"), H.boolean(crossings, "in_fit")
    f1_xactive, f1_inbounds = t["crossing_active"][xpos], t["crossing_inbounds"][xpos]
    f1_participates = natural["is_material"][nh+xpos]
    A.require(np.array_equal(f1_participates, f1_xactive & f1_inbounds), "F1 material participation inconsistent")
    xr = dict(strawid=xids[xpos].tolist(), partner_shindex=A.column(crossings, "partner_shindex").tolist(),
        f1_active=f1_xactive.tolist(), native_active=native_xactive.tolist(),
        f1_inbounds=f1_inbounds.tolist(), native_in_fit=native_infit.tolist(),
        f1_participates=f1_participates.tolist(), native_participates=(native_xactive & native_infit).tolist(),
        disagreement_strawids=xids[xpos[f1_participates != (native_xactive & native_infit)]].tolist(),
        f1_time=t["physical_time"][nh+xpos].tolist(), native_time=A.column(crossings, "time").tolist(),
        note="F1 crossing_inbounds includes activity/validity; not a pure geometric-bounds flag.")
    return dict(native_row_algebra=check, hit_states=states, bounds=br, crossings=xr)


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    A.require(not out.is_relative_to(Path(__file__).resolve().parents[1]), "protected checkout output")
    watched = {}
    report = dict(completed=False, passed=False, full_fit_closure=False, full_f1_closure=False,
                  full_numerical_validation=False, limits=LIMITS)
    with out.open("x", encoding="utf-8") as stream:
        try:
            M.helpers(watched)
            for module, names in ((sys.modules[__name__], ("bounds", "state_variance", "run", "main")),
                                  (H, ("boolean", "metric", "validate"))):
                path = Path(module.__file__).resolve()
                A.require(path.parent == Path(__file__).resolve().parent, "operand module checkout differs")
                D.watch(path, watched)
                for name in names:
                    D.code_origin(getattr(module, name, None), path)
                for value in vars(module).values():
                    if inspect.isfunction(value) and value.__module__ == module.__name__:
                        D.code_origin(value, path)
            path = data/"momentum_trace_7033_fixed_v1/manifest.json"
            manifest = D.receipt(path, TRACE_SHA, watched, ("completed", "passed", "controls_passed", "hashes_unchanged"))
            A.require(manifest["trace_passivity"]["exact"] and manifest["population_compatibility"]["passed"], "incompatible/nonpassive trace")
            artifact = manifest["artifacts"]["trace"]
            D.watch(artifact["path"], watched, artifact["sha256"])
            with np.load(artifact["path"], allow_pickle=False) as source:
                saved = {k: source[k] for k in source.files}
            A.require(np.array_equal(saved["key"], manifest["key"]) and saved["valid"].all()
                      and len(saved["valid"]) == manifest["n_iterations"] and saved["stage_index"][0] == 0, "saved identity/validity differs")
            capture = data/"crossing_3k_20261028_v2/capture.root"
            D.watch(capture, watched, C.CAPTURE_SHA)
            cycles = report["highest_cycles"] = {}
            tables = M.native_first(capture, saved["key"], cycles)
            with uproot.open(capture, object_cache=None, array_cache=None) as root:
                rows = A.read_tree(root, "kktrace", "hit_operands", cycles)
                rows = rows[np.all(A.keys(rows, unique=False) == saved["key"], axis=1) & (A.column(rows, "iteration_ordinal") == 0)]
            nh, nx = len(saved["hit_ids"]), len(saved["crossing_ids"])
            first = {k: saved[k][0] for k in fittrace.shapes(nh, nx)}
            scale = float((1+manifest["schedule"]["T"][0])**2)
            report["diagnostic"] = state_variance(first, saved, tables["iterations"], tables["effects"], rows, scale)
            report.update(completed=True, passed=True, key=saved["key"].tolist(), trace_manifest=str(path))
        except Exception as error:
            report.update(error=f"{type(error).__name__}: {error}", passed=False)
        finally:
            end, errors = {}, []
            for path in watched:
                try:
                    end[path] = A.digest(path)
                except Exception as error:
                    errors.append(f"{path}: {type(error).__name__}: {error}")
            report.update(before_sha256=watched, end_sha256=end, hash_errors=errors,
                          hashes_unchanged=bool(watched) and watched == end and not errors)
            if not report["hashes_unchanged"]:
                report["passed"] = False
            report, text = C.serialize(report)
            stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.data, args.out)
    print(json.dumps({k: report.get(k) for k in ("completed", "passed", "hashes_unchanged", "error")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
