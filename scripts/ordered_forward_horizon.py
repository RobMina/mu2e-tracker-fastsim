"""Saved requested-leg forward crossings versus physical stepping horizon; no continuation waiver."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest


def run(directory, outdir):
    directory, outdir = Path(directory), Path(outdir)
    path = directory / "forward_refusal.npz"
    receipt = json.loads((directory / "complete.json").read_text())
    with np.load(path, allow_pickle=False) as saved:
        o, d, lengths = saved["origin"], saved["direction"], saved["lengths"]
        keys, plane = saved["primary_keys"], float(saved["plane_y_mm"])
        up = d[:, 1] > 0
        crossing = np.divide(plane - o[:, 1], d[:, 1], out=np.full(len(o), np.inf), where=up)
        end = o[:, 1] + d[:, 1] * lengths
        guard = 64 * np.finfo(float).eps * np.maximum(1., np.maximum(np.abs(o[:, 1]), abs(plane)))
        violates = (o[:, 1] > plane + guard) | (end > plane + guard)
        frame = pd.DataFrame(keys, columns=["run", "subrun", "event", "sp"]).assign(
            y_mm=o[:, 1], uy=d[:, 1], requested_mm=lengths, plane_crossing_mm=crossing,
            requested_crosses=violates, crosses_within_max_step=violates & (crossing <= receipt["effective_settings"]["max_step_mm"]))
    outdir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(outdir / "lanes.csv", index=False)
    result = dict(scope="requested straight lookahead only; no stopping-range proof, physical continuation or domain exit disposition",
        producer_status=receipt["status"], iteration=receipt["forward_refusal_iteration"], rays=len(frame),
        upward_rays=int(up.sum()), violating_rays=int(violates.sum()),
        crossing_within_max_step=int(frame.crosses_within_max_step.sum()),
        min_positive_crossing_mm=float(crossing[violates & (crossing > 0)].min()) if (violates & (crossing > 0)).any() else None,
        input_sha256={str(p): digest(p) for p in (path, directory / "complete.json")},
        source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.outdir)
