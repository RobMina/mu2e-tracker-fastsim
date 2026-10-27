#!/usr/bin/env python3
"""Inspect the persisted generator census and stock drift-calibration domain before a tracker point."""
import argparse
import json

import numpy as np
import uproot

from mu2efastsim import tables
from trkfastsim.kitpoint import generated_keys
from trkfastsim.reco.cli import _drift_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen", required=True)
    p.add_argument("--drift-source", required=True)
    a = p.parse_args()
    g = tables.read(a.gen, "gen")
    with uproot.open(a.gen) as f:
        trees = {k: int(f[k].num_entries) for k in f.keys(recursive=True) if hasattr(f[k], "num_entries")}
    d = _drift_table(a.drift_source)
    print(json.dumps({"generator_trees": trees, "generated_events": len(generated_keys(g)),
                      "gen_rows": len(g), "primary_muons": int((g.pdg.abs() == 13).sum()),
                      "drift": {k: {"n": len(v), "min": float(np.min(v)), "max": float(np.max(v)),
                                    "first": np.asarray(v[:5]).tolist(), "finite": bool(np.isfinite(v).all())}
                                for k, v in d.items()}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
