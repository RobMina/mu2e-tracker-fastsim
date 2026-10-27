"""Local-only stock-conditioned F1 companion; never a tracker_point evaluator.

Reuses a saved F1 fit when supplied, otherwise runs the existing own-state F1
harness on the stock seed/hit/crossing inputs. No generated-event efficiency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .kitpoint import code_identity, file_identity, inspect_written
from .reco import hitreco, ntuple, tracktable


def check_saved_source(tracks, dump):
    with np.load(tracks, allow_pickle=False) as saved:
        meta = json.loads(str(saved["meta"]))
    if Path(meta["input"]).resolve() != Path(dump).resolve():
        raise ValueError("saved F1 source path does not match the stock dump; refusing index-only pairing")
    if not meta["own_states"]:
        raise ValueError("F1 companion requires own-state saved fits, not stock final-state conditioning")
    return meta


def run(dump, conditions, outdir, tracks=None, max_tracks=1000, workdir=None):
    if isinstance(max_tracks, bool) or not isinstance(max_tracks, int) or max_tracks < 1:
        raise ValueError("max_tracks must be a positive bounded diagnostic population")
    paths = {"dump": dump, "conditions": conditions, **({"tracks": tracks} if tracks else {})}
    if any(not Path(p).is_file() for p in paths.values()):
        raise ValueError("F1 companion inputs must be regular local files")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("F1 companion requires a new empty output directory")
    if workdir:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    inputs = {k: file_identity(p) for k, p in paths.items()}
    with tempfile.TemporaryDirectory(prefix="tracker-f1-", dir=workdir) as tmp:
        if tracks is None:
            from .reco import vr_f1
            vr_f1.run(str(dump), tmp, str(conditions), settings="mc", max_tracks=max_tracks,
                      own_states=True, save_tracks=True)
            tracks = str(Path(tmp) / "f1_tracks.npz")
            inputs["tracks"] = file_identity(tracks)
        saved = check_saved_source(tracks, dump)
        table = tracktable.from_f1(str(dump), hitreco.load_conditions(conditions), str(tracks), settings="mc")
        product = out / "EventNtuple_F1_stock_conditioned.root"
        writer = ntuple.write(table, str(product))
        keys = np.column_stack([table["events"][k] for k in ("run", "subrun", "event")])
        counts = inspect_written(product, keys, int(np.count_nonzero(table["goodfit"])))
        evidence = {"schema": "tracker-stock-f1-diagnostic-v1", "stock_conditioned": True,
                    "not_generated_event_efficiency": True, "conditioning": table["meta"]["conditioning"],
                    "inputs": inputs, "code": code_identity(), "saved_fit_settings": saved,
                    "candidate_tracks": len(table["ev"]), "persisted": counts, "writer": writer,
                    "product": file_identity(product), "wall_s": time.perf_counter() - start,
                    "limits": ["single-seed single-fit events", "stock selected hits and crossings",
                               "saved fits bind a source path and event indices, not historical input content hashes",
                               "not a tracker_point task or an autonomous F1 chain"]}
        (out / "f1_diagnostic.json").write_text(json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return evidence


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True)
    p.add_argument("--conditions", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tracks", help="reuse matching reco-vr2f1 --own-states --save-tracks output")
    p.add_argument("--max-tracks", type=int, default=1000, help="bound a new fit; saved-fit coverage takes precedence")
    p.add_argument("--workdir")
    return run(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    main()
