"""Retain full-key replacement-plane contacts from a refused producer; no geometry/domain waiver."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest


def run(directory, outdir):
    directory, outdir = Path(directory), Path(outdir)
    receipt = json.loads((directory / "complete.json").read_text())
    plane = receipt["handoff"]["plane_y_mm"]
    outdir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for query in sorted(directory.glob("navigation_*")):
        cells = pd.read_csv(query / "cells.csv")
        with np.load(query / "rays.npz", allow_pickle=False) as saved:
            o, d, keys = saved["origin"], saved["direction"], saved["primary_keys"]
        lane = cells.lane.to_numpy(np.int64)
        cells["y0_mm"] = o[lane, 1] + d[lane, 1] * cells.s0_mm
        cells["y1_mm"] = o[lane, 1] + d[lane, 1] * cells.s1_mm
        contact = cells[np.maximum(cells.y0_mm, cells.y1_mm) >= plane].copy()
        for j, key in enumerate(("run", "subrun", "event", "sp")):
            contact[key] = keys[contact.lane.to_numpy(np.int64), j]
        contact.to_csv(outdir / (query.name + ".csv"), index=False)
        summaries.append(dict(query=query.name, nonstrict_cells=len(contact), lanes=int(contact.lane.nunique()),
            sources=contact.source.unique().tolist(), start_zero_cells=int((contact.s0_mm == 0).sum()),
            above_plane_cells=int((np.maximum(contact.y0_mm, contact.y1_mm) > plane).sum()),
            max_y_excess_mm=float((np.maximum(contact.y0_mm, contact.y1_mm) - plane).max()) if len(contact) else 0.,
            downward_cells=int((d[contact.lane.to_numpy(np.int64), 1] < 0).sum())))
    result = dict(scope="plane localization only, not world/interior ownership", queries=summaries,
        input_sha256={str(p): digest(p) for p in directory.rglob("*") if p.is_file()},
        source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_physics=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.outdir)
