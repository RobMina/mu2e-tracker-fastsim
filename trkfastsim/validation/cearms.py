"""Geant4 reference arm of the CE tracker comparison: the primary electron's own StrawGasSteps from its first VD13 crossing.

Input: StrawGasStepTableDumper tables of Geant4 CeEndpoint dts events (trees dump/particles, dump/vdsteps,
dump/sgsteps). The primary electron is the SimParticle with PDG 11 flagged as the generated or MC primary; its first
crossing of VD13 (TT_FrontHollow, the tracker entry plane) is the truth state at the tracker entrance, and its tracker
steps from that time on (every later passage included) form the reference injector table. Secondaries are left out,
as in the fast arm (beamfastsim.cetracker), so both arms differ only in how the primary reaches and crosses the straws.

  python3 -m trkfastsim.validation.cearms TABLE.root [...] --outdir DIR

Writes DIR/g4prim_field.root (dump/sgsteps, dump/particles; one primary per event, SimParticle key 1, the layout of
the art injector StrawGasStepsFromTable) and DIR/g4entries.root (dump/entries: KEY, x, y, z, px, py, pz, t at VD13).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mu2efastsim import tables

KEY = tables.KEY
STATE = ["x", "y", "z", "px", "py", "pz", "t"]
VD_TRACKER_ENTRY = 13
STEP_COLS = KEY + ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len",
                   "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]


def _keys(df):
    for k in KEY:
        df[k] = df[k].astype(np.uint64)
    return df


def primary_entries(paths):
    """First VD13 crossing of the primary electron of each event (events without one are absent)."""
    p = tables.read_many(paths, "particles", KEY + ["key", "pdg", "isprim", "ismcprim"])
    p = _keys(p[(p.pdg == 11) & ((p.isprim == 1) | (p.ismcprim == 1))].rename(columns={"key": "sp"}))
    if p.duplicated(KEY).any():
        raise ValueError("more than one primary electron in an event")
    v = _keys(tables.read_many(paths, "vdsteps"))
    v = v[v.vdid == VD_TRACKER_ENTRY].merge(p[KEY + ["sp"]], on=KEY + ["sp"], validate="many_to_one")
    v = v.sort_values("t", kind="stable").drop_duplicates(KEY)
    return v[KEY + ["sp"] + STATE].sort_values(KEY).reset_index(drop=True)


def primary_steps(paths, entries):
    """StrawGasSteps of the primary electron at or after its VD13 time, renumbered to SimParticle key 1."""
    s = _keys(tables.read_many(paths, "sgsteps"))
    s = s.merge(entries[KEY + ["sp", "t"]].rename(columns={"t": "t_entry"}), on=KEY + ["sp"])
    s = s[s.t >= s.t_entry - 1e-6].drop(columns="t_entry")
    s["sp"] = 1
    return s[STEP_COLS].reset_index(drop=True)


def write_injector(path, steps, entries):
    p = entries[KEY].assign(key=1, pdg=11, parent=-1, isprim=1, ismcprim=1)
    tables.write(str(path), {"sgsteps": steps[STEP_COLS], "particles": p}, as_ttree=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m trkfastsim.validation.cearms", description=__doc__.split("\n")[0])
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = tables.expand(a.tables)
    ent = primary_entries(paths)
    steps = primary_steps(paths, ent)
    write_injector(out / "g4prim_field.root", steps, ent)
    tables.write(str(out / "g4entries.root"), {"entries": ent}, as_ttree=True)
    rep = dict(inputs=[Path(p).name for p in paths], entries=len(ent), steps=len(steps),
               events_with_steps=int(len(steps[KEY].drop_duplicates())))
    (out / "g4arm.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
