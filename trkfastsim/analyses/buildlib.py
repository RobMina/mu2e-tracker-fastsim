"""buildlib -- build the Geant4 secondary library (trkfastsim.library) from StrawGasStepTableDumper tables.

    buildlib "g4ref_v1_tables/sgsteps.*.root" --out lib/trklib_g4ref_v1.root [--gdml G] [--format 1|2]

--format 2 (default): steps also carry their straw-local placement (library.py, format v2; needs the GDML straws);
--format 1: the v1 tables only (reproduces the earlier libraries).

Writes instraw, instraw_steps, offaxis, events and meta trees, and prints a census. The library must not be built
from the events it is validated on: build from a subset (e.g. shards 1-10) and validate on the others, or from a
separate G4 run.
"""
import os

import numpy as np

from mu2efastsim import plots
from .. import geometry, library, tables as T


def run(inputs, outdir, out=None, gdml=None, format=2, **kw):
    paths = T.expand(inputs)
    geom = None
    if int(format) == 2:
        from .steps import default_gdml
        geom = geometry.from_gdml(gdml or default_gdml())
    lib = library.build(paths, geom)
    out = out or os.path.join(outdir, "trklib.root")
    library.write(out, lib, {"inputs": ";".join(os.path.basename(p) for p in paths), "nfiles": len(paths)})
    ins, st, off, ev = lib["instraw"], lib["instraw_steps"], lib["offaxis"], lib["events"]
    c = {"files": len(paths), "muon_events": int(len(ev)), "muon_crossings": int(len(ins)),
         "crossings_with_secondaries": float((ins.nsec > 0).mean()), "instraw_steps": int(len(st)),
         "offaxis_steps": int(len(off)), "events_with_offaxis": int((ev.noff > 0).sum()),
         "eion_keV": {"muon": float(1e3 * ins.mu_eion.sum()), "instraw_sec": float(1e3 * st.eion.sum()),
                      "offaxis": float(1e3 * off.eion.sum())},
         "crossings_per_pbin_lbin_min": int(ins.groupby(["pbin", "lbin"]).size().min()),
         "format": library.version(lib)}
    if c["format"] == 2:   # frame check: step ends must lie inside the straw gas (r <= 2.5085 mm)
        for lab, d in (("instraw", st), ("offaxis", off)):
            c[f"{lab}_r_q50_q99_max_mm"] = np.quantile(np.r_[d.r0, d.r1], [0.5, 0.99, 1.0]).round(3).tolist()
            c[f"{lab}_absdu_q50_q90_mm"] = np.quantile(np.abs(np.r_[d.du0, d.du1]), [0.5, 0.9]).round(1).tolist()
    for k, v in c.items():
        print(f"{k:30s} {v}")
    plots.write_json(os.path.dirname(out) or ".", os.path.basename(out).replace(".root", ".json"), c)
