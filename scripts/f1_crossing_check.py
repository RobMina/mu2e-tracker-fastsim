"""Independent bounded crossing-sidecar/passivity gate, not physics or full-fit closure."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

import f1_crossing_geometry_check as G
import f1_crossing_record_check as R
import f1_hit_operand_check as H
import f1_native_decision_audit as A

SIDECARS = ("crossing_records", "crossing_trajectories", "crossing_discovery", "crossing_capture_status",
            "crossing_geometry_bindings", "crossing_geometry_tracker", "crossing_geometry_planes",
            "crossing_geometry_panels", "crossing_geometry_straws")
MANIFEST_SHA = "ce3762ab9a5610e5838a5939ccff47c67d362812a41ddb1428eca13cc269cd5d"
PREPARATION_SHA = "9b25e7f783b0949f8a7e1c5c27548a0928a59eb45f1d1090d63a6fe082967c84"
REFERENCE_CHECK_SHA = "683ed29ea38ed004993acf9786e4c2ad4db269e6ce76f8ad4d88ffc7311b9209"
LIMITS = ["100-event smoke only, not full selected-target or full-population closure.",
          "Recorded field equality is not independent crossing/PCA/material-discovery generation.",
          "No native allocation-failure, nested/thread scope or arbitrary-configuration certification.",
          "No fitter, default, threshold, tolerance or acceptance changes."]


def exact_table(a, b, names, label):
    A.require(np.array_equal(A.keys(a, names), A.keys(b, names)), label + ": keys/order differ")
    A.require(set(ak.fields(a)) == set(ak.fields(b)), label + ": fields differ")
    for field in ak.fields(a):
        x, y = a[field], b[field]
        A.require(ak.type(x) == ak.type(y), label + "/" + field + ": type differs")
        for axis in range(x.ndim):
            A.require(not ak.any(ak.is_none(x, axis=axis), axis=None) and
                      not ak.any(ak.is_none(y, axis=axis), axis=None), label + ": missing value")
        for axis in range(1, x.ndim):
            A.require(ak.almost_equal(ak.num(x, axis=axis), ak.num(y, axis=axis), rtol=0, atol=0),
                      label + ": jagged shape differs")
        xx, yy = A.array(ak.flatten(x, axis=None)), A.array(ak.flatten(y, axis=None))
        A.require(xx.dtype == yy.dtype and np.array_equal(xx, yy, equal_nan=xx.dtype.kind in "fc"),
                  label + "/" + field + ": values differ")
    return {"passed": True, "n_rows": len(a), "n_fields": len(ak.fields(a))}


def receipt(path, watched):
    path = Path(path).resolve()
    watched[str(path)] = A.digest(path)
    gate = json.loads(path.read_text())
    for field in ("passed", "recorded_capture_equivalent", "smoke_only"):
        A.require(gate.get(field) is True, "requires literal smoke passivity gate: " + field)
    A.require(gate.get("final_capture_gate") is False, "smoke cannot be final gate")
    A.require(type(gate.get("n_events")) is int and gate["n_events"] == 100, "requires 100-event smoke")
    for field in ("control", "capture", "baseline", "targets"):
        item = gate["inputs"][field]
        source = str(Path(item["path"]).resolve())
        actual = A.digest(source)
        A.require(actual == item["sha256"], "gate input hash mismatch: " + field)
        A.require(source not in watched or watched[source] == actual, "conflicting input hash")
        watched[source] = actual
    return gate


def source_manifest(area, preparation, watched):
    area, preparation = Path(area).resolve(), Path(preparation).resolve()
    manifest = area / "capture-source-manifest.sha256"
    A.require(A.digest(manifest) == MANIFEST_SHA, "crossing source manifest changed")
    A.require(A.digest(preparation) == PREPARATION_SHA, "preparation receipt changed")
    watched[str(manifest)], watched[str(preparation)] = MANIFEST_SHA, PREPARATION_SHA
    paths = []
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        source = (area / name).resolve()
        A.require(source.is_relative_to(area), "manifest path escapes area")
        paths.append(str(source))
        A.require(A.digest(source) == digest, "native source changed: " + name)
        watched[str(source)] = digest
    A.require(len(paths) == len(set(paths)) == 50, "source manifest population")


def expanded_equal(current, reference):
    a, b = Path(current), Path(reference)
    x, y = a.read_text(), b.read_text()
    ax, by = f'fileName: "{a}"', f'fileName: "{b}"'
    A.require(x.count(ax) == y.count(by) == 1, "ambiguous debug output filename")
    A.require(x.replace(ax, 'fileName: "__DEBUG_OUTPUT__"') == y.replace(by, 'fileName: "__DEBUG_OUTPUT__"'),
              "expanded configuration differs beyond debug output")


def configuration(gate, previous, watched):
    directory = Path(gate["inputs"]["capture"]["path"]).parent
    old = Path(previous["inputs"]["capture"]["path"]).parent
    A.require(Path(gate["inputs"]["control"]["path"]).parent == directory, "pair directories differ")
    for arm, expected in (("control", "0ebf6d0c47b5a625439544349fd3fa27b52e6a1e95d96c89ec0a61018410a28a"),
                          ("capture", "68123c10e62928395122f2f7d8e081d4def024f0d867a31d4e9eeb93c070eecc")):
        a, b = directory / (arm + ".expanded.fcl"), old / (arm + ".expanded.fcl")
        A.require(A.digest(b) == expected, "predecessor expanded configuration changed")
        watched[str(a)], watched[str(b)] = A.digest(a), expected
        expanded_equal(a, b)
        exit_path = directory / (arm + ".exit")
        watched[str(exit_path)] = A.digest(exit_path)
        A.require(exit_path.read_text().strip() == "0", "native arm exit failure")
    manifest, old_manifest = directory / "manifest.txt", old / "manifest.txt"
    expected = "43dd21c3fadfde36b73bbba501fa9d48e2cb8ae40eed56efb1c4ca3ca87ed31e"
    A.require(A.digest(old_manifest) == expected, "predecessor launch manifest changed")
    watched[str(manifest)], watched[str(old_manifest)] = A.digest(manifest), expected
    text, old_text = manifest.read_text().splitlines(), old_manifest.read_text().splitlines()
    for field in ("input", "events", "config"):
        a = [line for line in text if line.startswith(field + "=")]
        b = [line for line in old_text if line.startswith(field + "=")]
        A.require(len(a) == len(b) == 1 and a == b, "frozen launch differs: " + field)
    return {"passed": True, "allowed_difference": "debug-config output filename only"}


def read(path, names, cycles):
    with uproot.open(path) as root:
        return {name: A.read_tree(root, "kktrace", name, cycles) for name in names}


def selected(capture, events, targets, gate):
    target = np.asarray(targets)
    A.require(target.ndim == 2 and target.shape[1] == 4 and target.dtype.kind in "iu", "target key shape/type")
    A.require(len(target) > 0 and np.all(target >= 0) and np.all(target <= np.iinfo(np.int64).max), "target key bounds")
    A.require(len(np.unique(target, axis=0)) == len(target), "duplicate targets")
    seed = capture["seed_contract"]
    sk, ek = A.keys(seed), A.keys(events, A.KEYS[:3])
    A.require(len(events) == 100 and len(seed) == gate["n_seeds"], "highest-cycle event/seed census")
    dtype = np.dtype([(f"k{i}", np.int64) for i in range(3)])
    event_view = np.ascontiguousarray(ek).view(dtype).ravel()
    target_view = np.ascontiguousarray(target[:, :3], dtype=np.int64).view(dtype).ravel()
    expected = target[np.isin(target_view, event_view)].astype(np.int64)
    A.require(len(expected) > 0 and len(expected) == gate["n_targets_found"], "selected target census")
    wanted = np.zeros(len(seed), dtype=bool)
    wanted[A.match_rows(expected, sk)] = True
    A.require(np.array_equal(H.boolean(seed, "capture_requested"), wanted), "seed capture requests differ")
    actual = np.unique(A.keys(capture["iterations"], A.ITER_KEYS)[:, :4], axis=0)
    A.require(len(actual) == len(expected), "iteration target population")
    A.match_rows(expected, actual)
    A.require(len(capture["iterations"]) == gate["n_iterations"], "iteration count differs from gate")
    return expected


def run(args):
    out = Path(args.out).resolve()
    protected = [Path(p).resolve() for p in (args.check, args.reference_check, args.preparation)]
    A.require(out not in protected and not out.is_relative_to(Path(args.native_source).resolve()), "protected report path")
    with out.open("x", encoding="utf-8") as stream:
        watched, report = {}, {"passed": False, "smoke_only": True, "full_fit_closure": False, "limits": LIMITS}
        try:
            sources = [Path(__file__).resolve()]
            for module in (G, R, H, A):
                path = Path(module.__file__).resolve()
                A.require(path.parent == sources[0].parent, "executing module from other checkout")
                sources.append(path)
            watched.update({str(path): A.digest(path) for path in sources})
            source_manifest(args.native_source, args.preparation, watched)
            A.require(A.digest(args.reference_check) == REFERENCE_CHECK_SHA, "frozen predecessor receipt changed")
            gate, previous = receipt(args.check, watched), receipt(args.reference_check, watched)
            current_path = gate["inputs"]["capture"]["path"]
            reference_path = previous["inputs"]["capture"]["path"]
            A.require(Path(current_path).resolve() != Path(reference_path).resolve(), "independent predecessor required")
            for name in ("baseline", "targets"):
                A.require(gate["inputs"][name]["sha256"] == previous["inputs"][name]["sha256"], "predecessor input mismatch")
            cycles = {name: {} for name in ("capture", "control", "predecessor")}
            common = tuple(H.ORIGINAL_KEYS) + ("hit_operands",)
            capture = read(current_path, common + SIDECARS, cycles["capture"])
            control = read(gate["inputs"]["control"]["path"], common + SIDECARS, cycles["control"])
            reference = read(reference_path, common, cycles["predecessor"])
            with uproot.open(current_path) as root:
                events = A.read_tree(root, "trkdump", "events", cycles["capture"])
            for label, path in (("control", gate["inputs"]["control"]["path"]), ("predecessor", reference_path)):
                with uproot.open(path) as root:
                    other = A.read_tree(root, "trkdump", "events", cycles[label])
                exact_table(events, other, A.KEYS[:3], label + " events")
            A.match_rows(A.keys(capture["seed_contract"])[:, :3], A.keys(events, A.KEYS[:3]))
            report["configuration"] = configuration(gate, previous, watched)
            targets = json.loads(Path(gate["inputs"]["targets"]["path"]).read_text())["keys"]
            expected = selected(capture, events, targets, gate)
            report.update(highest_cycles=cycles, n_events=len(events), n_targets=len(expected))
            for name in SIDECARS + ("hit_operands", "iterations", "pieces", "effects"):
                A.require(len(control[name]) == 0, "capture-off tree not empty: " + name)
                A.require(set(ak.fields(control[name])) == set(ak.fields(capture[name])), "capture-off partial schema: " + name)
            report["capture_off_sidecars_empty"] = True
            report["original_tree_passivity"] = H.exact_originals(capture, reference)
            report["hit_operand_passivity"] = exact_table(capture["hit_operands"], reference["hit_operands"], H.HIT_KEYS, "hit_operands")
            report["hit_operand_consistency"] = H.validate(capture["hit_operands"], capture["effects"], capture["iterations"])
            A.require(report["hit_operand_consistency"]["passed"] is True, "hit operand information mismatch")
            report["geometry"] = G.validate(capture, expected)
            report["crossings"] = R.validate(capture, capture["effects"], expected)
            A.require(report["geometry"].get("passed") is True and report["crossings"].get("passed") is True, "sidecar validation failed")
            report["passed"] = True
        except (ValueError, KeyError, OSError, TypeError, IndexError, AttributeError, OverflowError) as error:
            report.update(passed=False, error=f"{type(error).__name__}: {error}")
        finally:
            report["before_sha256"] = watched
            try:
                end = {path: A.digest(path) for path in watched}
                report.update(end_sha256=end, hashes_unchanged=end == watched)
                if end != watched:
                    report.update(passed=False, integrity_error="Evidence/source changed during checker")
            except (ValueError, OSError) as error:
                report.update(passed=False, hashes_unchanged=False, integrity_error=str(error))
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("check", "reference-check", "native-source", "preparation", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
