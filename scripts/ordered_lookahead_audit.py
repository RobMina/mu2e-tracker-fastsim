"""Saved bounded-lookahead queries retain positive in-domain geometry without shortening terminal arc."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest


def audit_queries(rows, receipt):
    if not rows.columns.is_unique or not {"lane", "iteration", "full_requested_mm", "query_length_mm", "crossing_mm",
            "x", "y", "z", "ux", "uy", "uz"} <= set(rows):
        raise ValueError("complete bounded query evidence required")
    values = rows.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("finite bounded query evidence required")
    step = receipt["effective_settings"]["max_step_mm"]
    plane = receipt["handoff"]["plane_y_mm"]
    if ((rows.crossing_mm <= step) | (rows.query_length_mm <= 0) | (rows.query_length_mm >= rows.crossing_mm) |
            (rows.query_length_mm > rows.full_requested_mm) | (rows.query_length_mm > 100.) | (rows.uy <= 0)).any():
        raise ValueError("bounded query violates physical-step or forward-domain horizon")
    expected = np.minimum(rows.full_requested_mm, np.minimum(100., np.maximum(step, .5 * rows.crossing_mm)))
    if not np.array_equal(expected, rows.query_length_mm):
        raise ValueError("bounded query policy differs from recorded length")
    crossing = (plane - rows.y.to_numpy()) / rows.uy.to_numpy()
    if not np.array_equal(crossing, rows.crossing_mm.to_numpy()):
        raise ValueError("bounded query crossing does not match saved ray")
    return dict(queries=len(rows), lanes=int(rows.lane.nunique()), minimum_crossing_mm=float(rows.crossing_mm.min()),
        maximum_query_mm=float(rows.query_length_mm.max()))


def run(directory, out):
    directory, out = Path(directory), Path(out)
    receipt = json.loads((directory / "complete.json").read_text())
    if receipt.get("lookahead_policy") != "bounded-in-domain-query-not-path-v1" or receipt["status"] != "bounded_full_chain_complete":
        raise ValueError("completed explicitly bounded lookahead producer required")
    paths = [directory / n for n in ("complete.json", "lookahead_queries.csv", "traversed.npz")]
    hashes = {str(p): digest(p) for p in paths}
    rows = pd.read_csv(paths[1], float_precision="round_trip")
    summary = audit_queries(rows, receipt)
    with np.load(paths[2], allow_pickle=False) as saved:
        stopped, path = saved["final_stopped"], saved["final_path"]
        limit = receipt["effective_settings"]["max_path_mm"]
        if ((~stopped) & (path != limit)).any():
            raise ValueError("query horizon became nonstopped terminal path")
        if stopped.sum() != receipt["stats"]["stopped"]:
            raise ValueError("bounded query audit stop count differs")
        summary.update(stopped=int(stopped.sum()), full_arc_nonstopped=int((~stopped).sum()))
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("lookahead audit inputs changed")
    result = dict(scope="query/terminal contracts only, not physical world exit", summary=summary, input_sha256=hashes,
        source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.out)
