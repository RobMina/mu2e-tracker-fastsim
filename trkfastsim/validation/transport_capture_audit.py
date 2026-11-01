"""Audit saved capture terminal/restart discrepancies without rerunning transport.

python3 scripts/transport_capture_audit.py CAPTURE_RUN_DIR --out NEW_JSON
No causal rate attribution, physical stop position or time closure is inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["run", "subrun", "event", "sp"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_capture(directory):
    directory = Path(directory)
    report = json.loads((directory / "complete.json").read_text())
    if report.get("identical") is not True or digest(directory / "capture.npz") != report["capture_sha256"]:
        raise ValueError("capture identity/completion gate failed")
    with np.load(directory / "capture.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["manifest_json"]))
        if metadata["schema"] != "trkfastsim.transport-observations.v1":
            raise ValueError("unsupported capture schema")
        if metadata["source_sha256"] != report["captured_source_sha256"]:
            raise ValueError("source identity mismatch")
        phases = {}
        for name, description in metadata["phases"].items():
            prefix = f"{name}_summary_"
            summary = pd.DataFrame({key[len(prefix):]: saved[key] for key in saved.files if key.startswith(prefix)})
            if summary.duplicated(KEY).any():
                raise ValueError("duplicate phase summary identity")
            records = {}
            for record in description["records"]:
                arrays = {key: saved[value] for key, value in record["columns"].items()}
                records.setdefault(record["kind"], []).append(arrays)
            phases[name] = summary, records
    return report, phases


def flatten(records, kind):
    batches = records.get(kind, [])
    if not batches:
        return {}
    return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}


def keyed(arrays):
    return pd.DataFrame(arrays["identity"], columns=KEY)


def audit(phases):
    interior, ir = phases["interior"]
    exit_summary, er = phases["exit"]
    restart = flatten(ir, "legacy_restart")
    saved = keyed(restart).assign(restart_local_path_mm=restart["path"],
        restart_ke_MeV=restart["ke"], restart_thin_MeV=restart["lost_thin"],
        restart_support_MeV=restart["lost_support"], legacy_stopped=restart["stopped"])
    joined = interior.merge(saved, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not (joined._merge == "both").all():
        raise ValueError("restart/summary identity coverage mismatch")
    path_gap = joined.last_path_mm - joined.phase_path_offset_mm - joined.restart_local_path_mm
    thin_gap = joined.last_lost_thin_MeV - joined.restart_thin_MeV
    support_gap = joined.last_lost_support_MeV - joined.restart_support_MeV
    joint = interior[KEY + ["disposition"]].merge(exit_summary, on=KEY, how="outer",
        suffixes=("_interior", "_exit"), validate="one_to_one", indicator=True)
    if not (joint._merge == "both").all():
        raise ValueError("phase identity coverage mismatch")
    transitions = joint.groupby(["disposition_interior", "disposition_exit"]).size().reset_index(name="lanes")
    result = dict(scope="algorithm restart/attempt audit; no physical stop or rate attribution",
        interior_lanes=len(interior), restart_behind_last_observed=int((path_gap > 1e-9).sum()),
        max_path_gap_mm=float(path_gap.max()) if len(joined) else None,
        final_entry_missing_thin_MeV_sum=float(thin_gap.sum()),
        max_final_entry_missing_thin_MeV=float(thin_gap.max()) if len(joined) else None,
        final_entry_support_gap_MeV_sum=float(support_gap.sum()),
        transitions=transitions.to_dict("records"), stop_attempts=[])
    attempts = flatten(ir, "stop_attempt")
    if attempts:
        stop = keyed(attempts).assign(attempt_local_path_mm=attempts["attempted_local_path_mm"],
            lower_local_path_mm=attempts["lower_local_path_mm"], raw_model_ke_MeV=attempts["model_ke_after_MeV"],
            committed_candidate=attempts["committed_candidate"])
        if stop.duplicated(KEY).any():
            raise ValueError("more than one first stop attempt per lane")
        stop = stop.merge(interior[KEY + ["disposition", "last_path_mm", "phase_path_offset_mm"]],
                          on=KEY, how="left", validate="one_to_one")
        if not (stop.disposition == "unresolved_stop").all():
            raise ValueError("stop attempt without stop disposition")
        exit_knots = flatten(er, "restart")
        if exit_knots:
            last_exit = keyed(exit_knots).assign(exit_knot_path_mm=exit_knots["path_mm"]).groupby(KEY).max().reset_index()
        else:
            last_exit = pd.DataFrame(columns=KEY + ["exit_knot_path_mm"])
        stop = stop.merge(last_exit, on=KEY, how="left", validate="one_to_one")
        stop["exit_knot_past_attempt"] = stop.exit_knot_path_mm > (stop.attempt_local_path_mm + stop.phase_path_offset_mm)
        result["stop_attempts"] = stop.astype(object).where(pd.notna(stop), None).to_dict("records")
    exit_attempts = flatten(er, "piece_attempt")
    failed = ~exit_attempts["accepted"] if exit_attempts else np.zeros(0, bool)
    result["exit_failed_attempts"] = int(failed.sum())
    result["exit_negative_raw_ke_attempts"] = int((exit_attempts["model_ke_after_MeV"][failed] < 0).sum()) if failed.any() else 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    inputs = [Path(args.directory) / name for name in ("complete.json", "capture.npz")]
    before = {str(p): digest(p) for p in inputs}
    report, phases = load_capture(args.directory)
    result = audit(phases)
    result.update(arm=report["arm"], processed_events=report["processed_events"], inputs=before,
                  audit_source_sha256=digest(__file__))
    if before != {str(p): digest(p) for p in inputs}:
        raise ValueError("saved capture changed during audit")
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
