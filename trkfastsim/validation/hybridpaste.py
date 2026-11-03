"""hybridpaste -- Geant4 primary muons with library secondaries (diagnostic).

The primary-muon StrawGasSteps of reference tables are kept and every other step is removed; then the library's
off-axis and in-straw secondaries are pasted onto those muons exactly as `steps --library` pastes them onto fast muons
(same samplers, options and per-file random streams (seed, i, 1) and (seed, i, 2)). Compared with the reference
(shardhist, shardstats) such an arm carries the library and the paster but not the fast muon transport, so it splits a
fast-vs-Geant4 difference into a secondary part and a muon part.

--library ROOT: a buildlib library. --library-from TABLES: build the library in memory (format 2) from these reference
tables instead, to test the donor sample (donor and recipient files must then be disjoint).
Output per input file: <outdir>/fasttrkCL.<input name>.root with sgsteps, particles, gen and subruns trees.
"""
import json
import os

import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables
from trkfastsim import geometry, tables as T
from trkfastsim.sim import library as L, steps as S

STEP_COLS = ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len", "width", "t",
             "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]


def primary_muons(path):
    """Steps and particles of the primary muons of one reference file, keyed 1..n per event as the fast muons."""
    s = T.read(path, "sgsteps")
    p = T.read(path, "particles", T.KEY + ["key", "pdg", "isprim"])
    pm = p[(p.isprim == 1) & (p.pdg.abs() == 13)].sort_values(T.KEY + ["key"]).reset_index(drop=True)
    pm["mukey"] = pm.groupby(T.KEY).cumcount() + 1
    s = s.merge(pm[T.KEY + ["key", "mukey"]].rename(columns={"key": "sp"}), on=T.KEY + ["sp"], how="inner")
    s["sp"] = s.mukey.astype(s.sp.dtype)
    s = s.sort_values(T.KEY + ["sp", "t"], kind="stable").reset_index(drop=True)
    s["nsgs"] = s.groupby(T.KEY)["sid"].transform("size")
    steps = s[T.KEY + STEP_COLS].reset_index(drop=True)
    parts = pm[T.KEY].copy()
    parts["key"] = pm.mukey.to_numpy()
    parts["parent"] = -1
    parts["pdg"] = pm.pdg.to_numpy()
    parts["isprim"] = 1
    parts["ismcprim"] = 1
    keep = steps[T.KEY].drop_duplicates()
    return steps, parts.merge(keep, on=T.KEY, how="inner").reset_index(drop=True)


def run(inputs, outdir, gdml=None, library=None, library_from=None, seed=1, no_offaxis=False, library_drop_muons=False,
        offaxis_plane_sign=False, offaxis_shift_from_muon=False, instraw_fine_length=False, **kw):
    from trkfastsim.sim.run import default_gdml
    if (library is None) == (library_from is None):
        raise SystemExit("pass exactly one of --library and --library-from")
    os.makedirs(outdir, exist_ok=True)
    geom = geometry.from_gdml(gdml or default_gdml())
    paths = T.expand(inputs)
    if library_from is not None:
        donors = T.expand(library_from)
        if set(map(os.path.basename, donors)) & set(map(os.path.basename, paths)):
            raise SystemExit("donor and recipient files overlap")
        lib = L.build(donors, geom)
    else:
        lib = L.read(library)
    log = []
    for i, f in enumerate(paths):
        steps, parts = primary_muons(f)
        ins = L.InStrawSampler(lib, np.random.default_rng([int(seed), i, 1]), drop_muons=library_drop_muons,
                               fine_length=instraw_fine_length)
        off = None if no_offaxis else L.OffAxisSampler(lib, np.random.default_rng([int(seed), i, 2]),
                                                         drop_muons=library_drop_muons, plane_sign=offaxis_plane_sign,
                                                         shift_from_muon=offaxis_shift_from_muon)
        stats = {}
        if off is not None:
            steps, parts, stats = S.add_offaxis_secondaries(steps, parts, off, geom)
        steps, parts = S.add_instraw_secondaries(steps, parts, ins, geom)
        name = "fasttrkCL." + os.path.basename(f)
        core_tables.write(os.path.join(outdir, name), {"sgsteps": steps, "particles": parts, "gen": T.read(f, "gen"),
                                                        "subruns": T.read(f, "subruns")}, as_ttree=True)
        log.append({"input": os.path.basename(f), "steps": int(len(steps)), "offaxis": stats})
        print(f"# {f}: {len(steps):,} steps, off-axis {stats}", flush=True)
    meta = {"inputs": [os.path.basename(p) for p in paths], "seed": int(seed),
            "library": os.path.basename(library) if library else None,
            "library_from": [os.path.basename(p) for p in T.expand(library_from)] if library_from else None,
            "options": {"no_offaxis": no_offaxis, "library_drop_muons": library_drop_muons,
                        "offaxis_plane_sign": offaxis_plane_sign, "offaxis_shift_from_muon": offaxis_shift_from_muon,
                        "instraw_fine_length": instraw_fine_length}, "files": log}
    with open(os.path.join(outdir, "hybridpaste.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=int)
    return meta
