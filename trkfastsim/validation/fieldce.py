"""CE hand-off: generated electron or Geant4 VD13 state -> field-on tracker -> injector tables.

Inputs are StrawGasStepTableDumper tables of CeEndpoint dts events. The chain arm transports their generated
starts through beamfastsim.cechain; the entry arm restarts at the primary's first Geant4 VD13. Input selection is
explicit: selected dts is conditional, while unfiltered input requires an externally audited complete count.
No secondaries or return passages are generated. Event keys retain starts that miss the fast entry plane.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from mu2efastsim import ehelix, tables
from trkfastsim import geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import cecontract

KEY = tables.KEY
STATE = ["x", "y", "z", "px", "py", "pz", "t"]
STEP_COLS = KEY + ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len",
                   "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]


def starts(paths):
    """One generated electron per event, in the Mu2e frame, with the original event identity."""
    g = tables.read_many(paths, "gen")
    g = g[g.pdg == 11].copy()
    if g.empty:
        p = tables.read_many(paths, "particles")
        g = p[(p.pdg == 11) & ((p.isprim == 1) | (p.ismcprim == 1))].rename(
            columns={"sx": "x", "sy": "y", "sz": "z", "spx": "px", "spy": "py", "spz": "pz", "st": "t"})
        g = g[KEY + ["pdg"] + STATE].copy()
    if g.duplicated(KEY).any():
        raise ValueError("CE hand-off requires exactly one generated electron per event")
    if g.empty:
        raise ValueError("no generated electrons")
    return cecontract.unique(g, "CE starts")


def primary_entries(paths):
    """Primary electron's first VD13 crossing; no matching by distance or time to a different particle."""
    p = tables.read_many(paths, "particles", KEY + ["key", "pdg", "isprim", "ismcprim"])
    p = p[(p.pdg == 11) & ((p.isprim == 1) | (p.ismcprim == 1))].rename(columns={"key": "sp"})
    v = tables.read_many(paths, "vdsteps")
    p = cecontract.integer_keys(p, "primary particles")
    v = cecontract.integer_keys(v, "VD steps")
    v = v[v.vdid == 13].merge(p[KEY + ["sp", "pdg"]], on=KEY + ["sp"], validate="many_to_one")
    v = v.sort_values("t").drop_duplicates(KEY + ["sp"])
    if v.duplicated(KEY).any():
        raise ValueError("more than one primary electron at VD13 in an event")
    return v[KEY + ["sp", "pdg"] + STATE].sort_values(KEY).reset_index(drop=True)


def result_entries(g, result, plane_fate):
    """Attach event keys by row position, never by a float-valued event id or by a phase-space match."""
    if len(g) != len(result.fate):
        raise ValueError("CE result and generated-start lengths differ")
    ok = result.fate == plane_fate
    out = g.loc[ok, KEY].copy().reset_index(drop=True)
    p = np.sqrt(result.T[ok] * (result.T[ok] + 2 * fs.eloss.ME))
    out[["x", "y", "z"]] = result.x[ok]
    out[["px", "py", "pz"]] = result.u[ok] * p[:, None]
    out["t"] = result.t[ok]
    out["sp"], out["pdg"] = 1, 11
    return out[KEY + ["sp", "pdg"] + STATE]


def injector(path, steps, entries):
    """A single primary per event, renumbered to key 1 for the stock injector's integer key contract."""
    s = steps.copy()
    s["sp"] = 1
    p = entries[KEY].assign(key=1, pdg=11, parent=-1, isprim=1, ismcprim=1)
    tables.write(str(path), {"sgsteps": s[STEP_COLS], "particles": p}, as_ttree=True)


def reference_contract(g, selection, expected_starts=None):
    """Do not relabel a selected reference without an externally audited complete event count."""
    return cecontract.selection_contract(len(g), selection, expected_starts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--mode", choices=("entry", "chain"), required=True)
    ap.add_argument("--offline-root")
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_common.txt")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--max-events", type=int)
    ap.add_argument("--reference-selection", choices=("selected-dts", "unfiltered"), default="selected-dts")
    ap.add_argument("--expected-starts", type=int, help="externally audited complete event count, required for unfiltered")
    ap.add_argument("--gas-state", choices=("solid", "gas"), default="solid")
    ap.add_argument("--chord-mm", type=float, default=10.0)
    ap.add_argument("--origin", nargs=3, type=float, default=(-3904.0, 0.0, 10171.0))
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = tables.expand(a.inputs)
    g = starts(paths)
    if a.max_events is not None:
        if a.max_events <= 0:
            ap.error("--max-events must be positive")
        if a.reference_selection == "unfiltered" and a.max_events < len(g):
            ap.error("--max-events cannot truncate an unfiltered complete-birth population")
        g = g.iloc[:a.max_events]
    original = primary_entries(paths).merge(g[KEY], on=KEY, validate="one_to_one")
    report = dict(mode=a.mode, generated_starts=len(g), seed=a.seed, geom=a.geom,
                  gas_state=a.gas_state, reference_vd13=len(original), inputs=[Path(p).name for p in paths],
                  birth_keys_sha256=cecontract.fingerprint(g),
                  **reference_contract(g, a.reference_selection, a.expected_starts))
    if a.mode == "chain":
        from beamfastsim import cechain
        if not a.offline_root:
            ap.error("--offline-root is required for --mode chain")
        x = g[["x", "y", "z"]].to_numpy(float)
        pv = g[["px", "py", "pz"]].to_numpy(float)
        p = np.linalg.norm(pv, axis=1)
        cg = cechain.from_offline(a.geom, a.offline_root)
        tick = time.perf_counter()
        r = cechain.transport(np.random.default_rng(np.random.SeedSequence([a.seed, 0])), cg,
                              cechain.field_maps(a.maps), x, pv / p[:, None], p, g.t.to_numpy(float), h=10.0)
        report["ce_transport_s"] = time.perf_counter() - tick
        report["ce_fates"] = {name: int((r.fate == i).sum()) for i, name in enumerate(cechain.FATES)}
        ent = result_entries(g, r, cechain.PLANE)
        np.savez_compressed(out / "cechain.npz", **{k: v for k, v in r._asdict().items() if v is not None},
                            **{k: g[k].to_numpy() for k in KEY})
    else:
        ent = original.copy()
    geom = geometry.from_gdml(a.gdml)
    org = np.array(a.origin)
    env = fs.Envelope(geom, axis_xy=org[:2])
    support = fs.interior(a.gdml, geom, env, origin=org)
    field = ehelix.grid_field(*ehelix.read_g4bl(a.maps[0]))
    bnom = ehelix.field_at(field, org[None])[0]
    tick = time.perf_counter()
    fast, final = fs.transport(np.random.default_rng(np.random.SeedSequence([a.seed, 1])), ent, geom, field, support,
                               chord_mm=a.chord_mm, sagitta_mm=0.02, envelope=env, origin=org, bnom=bnom,
                               loss=fs.electron_loss, tail=False,
                               gas=fs.STRAW_GAS_G4STATE if a.gas_state == "gas" else None)
    report.update(tracker_transport_s=time.perf_counter() - tick, entries=len(ent), steps=len(fast),
                  events_with_steps=len(fast[KEY].drop_duplicates()), first_passage_only=True)
    injector(out / "fasttrk_field.root", fast, ent)
    g4 = tables.read_many(paths, "sgsteps")
    g4 = g4.merge(original[KEY + ["sp", "t"]].rename(columns={"t": "entry_t"}), on=KEY + ["sp"])
    g4 = g4[g4.t >= g4.entry_t - 1e-6]
    injector(out / "g4prim_field.root", g4, original)
    tables.write(str(out / "states.root"), {"gen": g, "entries": ent, "g4entries": original, "final": final},
                 as_ttree=True)
    (out / "handoff.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
