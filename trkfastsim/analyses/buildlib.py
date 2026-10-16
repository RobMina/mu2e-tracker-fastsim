"""buildlib -- build the Geant4 secondary library (trkfastsim.library) from StrawGasStepTableDumper tables.

    buildlib "g4ref_v1_tables/sgsteps.*.root" --out lib/trklib_g4ref_v1.root

Writes instraw, instraw_steps, offaxis, events and meta trees, and prints a census. The library must not be built
from the events it is validated on: build from a subset (e.g. shards 1-10) and validate on the others, or from a
separate G4 run.
"""
import os

import numpy as np

from mu2efastsim import plots
from .. import library, tables as T


def run(inputs, outdir, out=None, **kw):
    paths = T.expand(inputs)
    lib = library.build(paths)
    out = out or os.path.join(outdir, "trklib.root")
    library.write(out, lib, {"inputs": ";".join(os.path.basename(p) for p in paths), "nfiles": len(paths)})
    ins, st, off, ev = lib["instraw"], lib["instraw_steps"], lib["offaxis"], lib["events"]
    c = {"files": len(paths), "muon_events": int(len(ev)), "muon_crossings": int(len(ins)),
         "crossings_with_secondaries": float((ins.nsec > 0).mean()), "instraw_steps": int(len(st)),
         "offaxis_steps": int(len(off)), "events_with_offaxis": int((ev.noff > 0).sum()),
         "eion_keV": {"muon": float(1e3 * ins.mu_eion.sum()), "instraw_sec": float(1e3 * st.eion.sum()),
                      "offaxis": float(1e3 * off.eion.sum())},
         "crossings_per_pbin_lbin_min": int(ins.groupby(["pbin", "lbin"]).size().min()),
         "offaxis_u_v_w_p01_p99_mm": {a: np.quantile(off[a], [0.01, 0.99]).tolist() for a in "uvw"} if len(off) else {}}
    for k, v in c.items():
        print(f"{k:30s} {v}")
    plots.write_json(os.path.dirname(out) or ".", os.path.basename(out).replace(".root", ".json"), c)
