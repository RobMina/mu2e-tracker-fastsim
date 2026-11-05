"""Saved CE efficiency migration and zero-step endpoint diagnostics, without tuning physics.

The symmetric tracker-term flag counts only the injected primary, not the production dts OR-filter, which also
accepts calorimeter and other-particle response. Conditioning on G4 good-track acceptance is tautological and is
reported only as migration bookkeeping, never evidence by itself for the cause of the selected-dts deficit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim import geometry, tables as tracker_tables
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import ceaudit
from trkfastsim.validation.fieldreco import _me

KEY = tables.KEY


def attach(d, data, label):
    """Attach unique exact event identities and reject source rows outside the original denominator."""
    data = ceaudit.unique(data, label)
    check = data[KEY].merge(d[KEY], on=KEY, how="left", indicator=True, validate="one_to_one")
    if (check._merge != "both").any():
        raise ValueError(f"{label} outside original denominator")
    return d.merge(data, on=KEY, how="left", validate="one_to_one")


def step_flags(s):
    """Symmetric primary step delivery and production tracker-term thresholds, retaining event zeros later."""
    s = ceaudit.integer_keys(s, "primary steps")
    if not s.empty and not (s.sp == 1).all():
        raise ValueError("expected sole renumbered primary key 1")
    count = s.groupby(KEY).size().rename("steps").reset_index()
    flags = tracker_tables.trk_filter(s).reset_index().rename(columns={"nmax": "good_steps", "pass_trk": "tracker_term"})
    count = count.merge(flags, on=KEY, how="left", validate="one_to_one")
    return count


def migration(g4, fast):
    """Four cells over the same complete event population with the paired difference error."""
    g4, fast = np.asarray(g4, bool), np.asarray(fast, bool)
    return {"events": len(g4), "both": int((g4 & fast).sum()), "g4_only": int((g4 & ~fast).sum()),
            "fast_only": int((~g4 & fast).sum()), "neither": int((~g4 & ~fast).sum()),
            "g4_fraction": float(g4.mean()) if len(g4) else None,
            "fast_fraction": float(fast.mean()) if len(fast) else None,
            "paired_fast_minus_g4_error_n": _me(fast.astype(float) - g4)}


def migrations(d):
    """Do not conflate primary step delivery, primary tracker-term selection and stock track acceptance."""
    acceptance = {"any_primary_steps": (d.g4_steps > 0, d.chain_steps > 0),
                  "primary_tracker_term": (d.g4_tracker_term, d.chain_tracker_term),
                  "stock_good_track": (d.g4_track, d.chain_track)}
    out = {k: migration(g, f) for k, (g, f) in acceptance.items()}
    out["good_track_by_reference_condition"] = {}
    for name, mask in (("all_events", np.ones(len(d), bool)), ("g4_has_primary_steps", d.g4_steps > 0),
                       ("g4_primary_tracker_term", d.g4_tracker_term),
                       ("g4_fails_primary_tracker_term", ~d.g4_tracker_term),
                       ("fast_primary_tracker_term", d.chain_tracker_term),
                       ("fast_fails_primary_tracker_term", ~d.chain_tracker_term),
                       ("both_primary_tracker_terms", d.g4_tracker_term & d.chain_tracker_term),
                       ("either_primary_tracker_term", d.g4_tracker_term | d.chain_tracker_term),
                       ("g4_good_track", d.g4_track)):
        out["good_track_by_reference_condition"][name] = migration(d.g4_track[mask], d.chain_track[mask])
    out["selection_caveat"] = "tracker term is primary-only, not the full dts OR-filter; G4-good-track conditioning is tautological"
    return out


def gas_inner_radius(geom, axis):
    """Smallest transverse radius of a finite straw gas cylinder, from the loaded sample geometry."""
    o = geom.mid[:, :2] - axis
    u = geom.dir[:, :2]
    h = np.clip(-(o * u).sum(axis=1) / np.maximum((u * u).sum(axis=1), 1e-30), -geom.halflen, geom.halflen)
    return max(0.0, float(np.linalg.norm(o + h[:, None] * u, axis=1).min() - geom.radius))


def endpoint_labels(final, envelope, inner_radius):
    """Classify only observed endpoints; a reverse endpoint is not proof of a particular curl history."""
    r = np.hypot(final.x - envelope.axis[0], final.y - envelope.axis[1])
    front, back = final.z < envelope.z0, final.z > envelope.z1
    hole = r < inner_radius
    labels = np.select([final.status == 1, final.status == 2, (final.status == 0) & front & hole,
                        (final.status == 0) & back & hole, (final.status == 0) & front,
                        (final.status == 0) & back, (final.status == 0) & (r > envelope.r), final.status == 0],
                       ["stopped", "path_or_chord_cap", "front_exit_inside_gas_inner_radius",
                        "back_exit_inside_gas_inner_radius", "front_exit", "back_exit", "outer_exit", "other_exit"],
                       default="unexpected_status")
    return labels


def describe(d, mask, columns):
    out = {"events": int(np.asarray(mask).sum())}
    for k in columns:
        x = d.loc[mask, k].to_numpy(float)
        x = x[np.isfinite(x)]
        out[k] = {"finite": len(x), "q10_q50_q90": np.quantile(x, [0.1, 0.5, 0.9]).tolist() if len(x) else []}
    return out


def assemble(entry_events, chain_events, entry_dir, chain_dir, geom, origin):
    """Join saved arms, handoff states, final states and primary-only steps on audited event identities."""
    d = ceaudit.join_events(entry_events, chain_events)
    env = fs.Envelope(geom, axis_xy=np.asarray(origin)[:2])
    inner = gas_inner_radius(geom, env.axis)
    for arm, path in (("entry", Path(entry_dir)), ("chain", Path(chain_dir))):
        g = ceaudit.unique(tables.read(path / "states.root", "gen", KEY), f"{arm} starts")
        pd.testing.assert_frame_equal(d[KEY], g[KEY], check_exact=True)
        state = ceaudit.state_summary(tables.read(path / "states.root", "entries"), arm)
        state[f"{arm}_has_vd13"] = True
        d = attach(d, state, f"{arm} VD13")
        d[f"{arm}_has_vd13"] = d[f"{arm}_has_vd13"].fillna(False).astype(bool)
        final = ceaudit.unique(tables.read(path / "states.root", "final"), f"{arm} final")
        if not np.array_equal(final[KEY].to_numpy(), state[KEY].to_numpy()):
            raise ValueError(f"{arm} final/entry identity mismatch")
        final[f"{arm}_endpoint"] = endpoint_labels(final, env, inner)
        final[f"{arm}_end_r"] = np.hypot(final.x - env.axis[0], final.y - env.axis[1])
        final[f"{arm}_end_p"] = np.linalg.norm(final[["px", "py", "pz"]].to_numpy(float), axis=1)
        names = {k: f"{arm}_end_{k}" for k in ("x", "y", "z", "px", "py", "pz", "path", "chords", "status")}
        final = final.rename(columns=names).drop(columns=["sp", "t"])
        d = attach(d, final, f"{arm} endpoints")
    for arm, path in (("g4", Path(entry_dir) / "g4prim_field.root"),
                      ("entry", Path(entry_dir) / "fasttrk_field.root"),
                      ("chain", Path(chain_dir) / "fasttrk_field.root")):
        s = step_flags(tables.read(path, "sgsteps"))
        s = s.rename(columns={k: f"{arm}_{k}" for k in s if k not in KEY})
        d = attach(d, s, f"{arm} steps")
        for col in ("steps", "good_steps"):
            d[f"{arm}_{col}"] = d[f"{arm}_{col}"].fillna(0).astype(int)
        d[f"{arm}_tracker_term"] = d[f"{arm}_tracker_term"].fillna(False).astype(bool)
    for arm in ("entry", "chain"):
        d[f"{arm}_reversed_at_end"] = (d[f"{arm}_costheta"] > 0) & (d[f"{arm}_end_pz"] < 0)
        d[f"{arm}_endpoint"] = d[f"{arm}_endpoint"].fillna("no_vd13")
    return d, {"r": env.r, "z0": env.z0, "z1": env.z1, "gas_inner_radius_mm": inner}


def report(d):
    """Exact saved no-step decomposition; physical mechanisms require further controls."""
    out = {"events": len(d), "migration": migrations(d), "no_step_autopsy": {},
           "scope": "saved endpoints and truth states are diagnostics, not a response/correction model"}
    cols = ["chain_p", "entry_p", "chain_r", "entry_r", "chain_costheta", "chain_end_p", "chain_end_z",
            "chain_end_r", "chain_end_path", "chain_end_chords"]
    total = 0
    for name, mask in (("no_chain_vd13", ~d.chain_has_vd13),
                       ("vd13_no_steps", d.chain_has_vd13 & (d.chain_steps == 0)), ("has_steps", d.chain_steps > 0)):
        x = describe(d, mask, cols)
        x.update(g4_tracks=int(d.g4_track[mask].sum()), chain_tracks=int(d.chain_track[mask].sum()))
        x["signed_track_difference"] = x["chain_tracks"] - x["g4_tracks"]
        total += x["signed_track_difference"]
        out["no_step_autopsy"][name] = x
    if total != int(d.chain_track.sum() - d.g4_track.sum()):
        raise ValueError("non-exhaustive yield partition")
    zero = d.chain_has_vd13 & (d.chain_steps == 0)
    out["zero_step_by_endpoint"] = {}
    for label in sorted(d.loc[zero, "chain_endpoint"].unique()):
        mask = zero & (d.chain_endpoint == label)
        x = describe(d, mask, cols)
        x.update(g4_tracks=int(d.g4_track[mask].sum()), reversed_at_end=int(d.chain_reversed_at_end[mask].sum()))
        out["zero_step_by_endpoint"][label] = x
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry-events", required=True)
    ap.add_argument("--chain-events", required=True)
    ap.add_argument("--entry-dir", required=True)
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--origin", nargs=3, type=float, default=(-3904.0, 0.0, 10171.0))
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    geom = geometry.from_gdml(a.gdml)
    d, envelope = assemble(pd.read_csv(a.entry_events), pd.read_csv(a.chain_events), a.entry_dir, a.chain_dir, geom, a.origin)
    result = report(d)
    receipt = json.loads((Path(a.chain_dir) / "handoff.json").read_text())
    result.update(envelope=envelope, reference_selection=receipt.get("reference_selection", "selected-dts"),
                  conditional_on_dts_selection=receipt.get("conditional_on_dts_selection", True))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "events.csv.gz", index=False)
    (out / "cerootcause.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
