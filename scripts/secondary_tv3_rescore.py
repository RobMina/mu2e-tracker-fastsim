"""Rescore a saved comparison manifest without generation, reconstruction or fetching.

A fresh output directory retains the new result, source/input-manifest hashes,
and every changed saved-result field. Existing products are never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from trkfastsim.analyses import tv2compare as TV


def differences(old, new, path=""):
    """Report exact changed JSON leaves, treating matching NaNs as equal."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(old.keys() | new.keys()):
            name = f"{path}.{key}".lstrip(".")
            if key not in old or key not in new:
                out.append(name)
            else:
                out.extend(differences(old[key], new[key], name))
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [path]
        return [name for i, (a, b) in enumerate(zip(old, new)) for name in differences(a, b, f"{path}[{i}]")]
    if isinstance(old, float) and isinstance(new, float) and np.isnan(old) and np.isnan(new):
        return []
    return [] if old == new else [path]


def run(comparison, outdir):
    outdir.mkdir(parents=True, exist_ok=False)
    data = json.loads(comparison.read_text())
    paths = data["input_files"]
    t0 = time.perf_counter()
    TV.run(paths["ntuple_fast"], str(outdir / "comparison"), compare=paths["ntuple_g4"],
           steps=paths["steps_fast"], steps_ref=paths["steps_g4"], hits=paths["hits_fast"],
           hits_ref=paths["hits_g4"], paired=data["paired"])
    result_path = outdir / "comparison" / "tv2compare.json"
    result = json.loads(result_path.read_text())
    root = Path(TV.__file__).resolve().parents[2]
    sources = [Path(__file__), Path(TV.__file__), root / "trkfastsim/analyses/occupancy.py",
               root / "trkfastsim/analyses/clusterband.py"]
    audit = {"source_comparison": str(comparison),
             "source_comparison_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
             "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
             "changed_fields": differences(data, result), "elapsed_s": time.perf_counter() - t0,
             "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (outdir / "rescore.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison, args.outdir)
