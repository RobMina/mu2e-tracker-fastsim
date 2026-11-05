"""Saved upstream CE material-count and symmetric migration diagnostics, with no new transport.

Only fast material encounter counts are available; these are not measured Geant4 path lengths or a causal material
comparison. Lost and gained primary-step events are described symmetrically on the same unfiltered population.
Local axial-helix radial reach is a truth-state diagnostic, not a measured-variable response/correction model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import ehelix, tables
from trkfastsim.validation import ceaudit, cerootcause

KEY = tables.KEY


def axial_rmax(x, pv, bz, axis):
    """Maximum orbit radius for a uniform local axial field, not an actual mapped trajectory maximum."""
    pt = np.linalg.norm(pv[:, :2], axis=1)
    radius = pt / (ehelix.K * np.abs(bz))
    direction = pv[:, :2] / np.maximum(pt[:, None], 1e-30)
    offset = np.stack([-direction[:, 1], direction[:, 0]], axis=1) * (np.sign(bz) * radius)[:, None]
    return np.linalg.norm(x[:, :2] - axis + offset, axis=1) + radius


def saved_chain(path):
    """Require the saved chain's exact complete birth identities before attaching material counters."""
    with np.load(path, allow_pickle=False) as z:
        d = pd.DataFrame({k: z[k] for k in KEY})
        d["fast_fate"] = z["fate"]
        for k in ("nfoil", "nipa", "nopa", "nring", "path", "T"):
            if k in z:
                d[f"upstream_{k}"] = z[k]
    return ceaudit.unique(d, "saved CE chain")


def augment(events, chain_dir, field, axis):
    d = ceaudit.unique(events, "saved root-cause events")
    chain_dir = Path(chain_dir)
    saved = saved_chain(chain_dir / "cechain.npz")
    pd.testing.assert_frame_equal(d[KEY], saved[KEY], check_exact=True)
    d = cerootcause.attach(d, saved, "chain material counters")
    birth = ceaudit.unique(tables.read(chain_dir / "states.root", "gen"), "births")
    p0 = np.linalg.norm(birth[["px", "py", "pz"]].to_numpy(float), axis=1)
    b = birth[KEY].copy()
    b["birth_z"] = birth.z
    b["birth_r"] = np.hypot(birth.x - axis[0], birth.y - axis[1])
    b["birth_costheta"] = birth.pz / p0
    b["birth_p"] = p0
    d = cerootcause.attach(d, b, "birth state summaries")
    for label, tree in (("g4", "g4entries"), ("fast", "entries")):
        s = ceaudit.unique(tables.read(chain_dir / "states.root", tree), label)
        x, pv = s[["x", "y", "z"]].to_numpy(float), s[["px", "py", "pz"]].to_numpy(float)
        bz = ehelix.field_at(field, x)[:, 2]
        if not np.all(np.isfinite(bz) & (np.abs(bz) > 0.01)):
            raise ValueError("local axial reach requires a nonzero Bz")
        v = s[KEY].copy()
        v[f"{label}_local_axial_rmax"] = axial_rmax(x, pv, bz, axis)
        d = cerootcause.attach(d, v, f"{label} axial reach")
    d["fast_entry_loss"] = d.birth_p - d.chain_p
    d["g4_entry_loss"] = d.birth_p - d.entry_p
    d["entry_p_fast_minus_g4"] = d.chain_p - d.entry_p
    return d


def report(d, inner_radius):
    g, f = d.g4_steps > 0, d.chain_steps > 0
    masks = {"both_primary_steps": g & f, "g4_only_primary_steps": g & ~f,
             "fast_only_primary_steps": ~g & f, "neither_primary_steps": ~g & ~f,
             "lost_good_track": d.g4_track & ~d.chain_track, "gained_good_track": ~d.g4_track & d.chain_track}
    cols = ["birth_z", "birth_r", "birth_costheta", "birth_p", "fast_entry_loss", "g4_entry_loss",
            "entry_p_fast_minus_g4", "fast_local_axial_rmax", "g4_local_axial_rmax"]
    cols += [k for k in d if k.startswith("upstream_")]
    out = {"events": len(d), "classes": {}, "inner_gas_radius_mm": inner_radius,
           "caveat": "fast-only material counts; no symmetric G4 material path ledger; axial reach assumes uniform local Bz"}
    for label, mask in masks.items():
        r = cerootcause.describe(d, mask, cols)
        valid = mask & d.chain_has_vd13 & d.entry_has_vd13
        r["both_vd13_events"] = int(valid.sum())
        r["g4_local_axial_reaches_gas_radius"] = int((valid & (d.g4_local_axial_rmax >= inner_radius)).sum())
        r["fast_local_axial_reaches_gas_radius"] = int((valid & (d.fast_local_axial_rmax >= inner_radius)).sum())
        r["fast_fate_counts"] = d.loc[mask, "fast_fate"].value_counts().sort_index().to_dict()
        out["classes"][label] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--axis", nargs=2, type=float, default=(-3904.0, 0.0))
    ap.add_argument("--inner-radius", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    d = augment(pd.read_csv(a.events), a.chain_dir, ehelix.grid_field(*ehelix.read_g4bl(a.map)), np.asarray(a.axis))
    result = report(d, a.inner_radius)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "events.csv.gz", index=False)
    (out / "ceupstream.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
