"""steps -- run the fast StrawGasStep generator (trkfastsim.steps) on generated primaries, one table file per input.

Input: gen tables of the G4 reference (`gen` + `subruns` trees: crvfastsim's gen.*.root from the gen stream, or the
tracker dumper's gen_tables.*.root). Output: fasttrk.<input>.root with the sgsteps/particles/gen/subruns trees in the
StrawGasStepTableDumper layout, same (run, subrun, event) keys, so `v1compare FAST --compare G4` matches events.

    steps gen.001400_00000001.root [...] --gdml mu2e_extracted_v04.gdml --outdir DIR [--seed N] [--stage A|C]

--stage A: straight from the generated vertex; C: GDML world material and multiple scattering down to the tracker.
--library LIB.root: add the G4 library's secondaries (buildlib): in-straw and off-axis (--no-offaxis: in-straw only);
output prefix fasttrk<stage>L.
"""
import os

import numpy as np

from mu2efastsim import tables as core_tables
from .. import geometry, library as lib_mod, steps as st, tables

GDML_NAME = "crv_fastsim/geometry/mu2e_extracted_v04.gdml"   # under $M2FS_DATA (crv/geometry/README.md)


def default_gdml():
    data = os.environ.get("M2FS_DATA")
    if not data:
        raise SystemExit("pass --gdml or set M2FS_DATA (source site.env)")
    return os.path.join(data, GDML_NAME)


def run(inputs, outdir, seed=1, gdml=None, stage="A", library=None, no_offaxis=False, **kw):
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    world = st.load_world(gdml, geom) if stage == "C" else None
    lib = lib_mod.read(library) if library else None
    for i, f in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([int(seed), i])
        gen = tables.read(f, "gen")
        sub = tables.read(f, "subruns")
        sampler = lib_mod.InStrawSampler(lib, np.random.default_rng([int(seed), i, 1])) if lib else None
        offaxis = lib_mod.OffAxisSampler(lib, np.random.default_rng([int(seed), i, 2])) if lib and not no_offaxis else None
        sgs, parts = st.make_steps(gen, geom, rng, stage=stage, world=world, sampler=sampler, offaxis=offaxis)
        if offaxis is not None:
            print(f"#   off-axis: {getattr(st.make_steps, 'offaxis_stats', {})}", flush=True)
        name = f"fasttrk{stage}{'L' if lib else ''}." + os.path.basename(f).replace("gen_tables.", "").replace("gen.", "")
        core_tables.write(os.path.join(outdir, name), {"sgsteps": sgs, "particles": parts, "gen": gen,
                                                        "subruns": sub}, as_ttree=True)
        print(f"# {f}: {len(parts):,} primary muons, {sgs[tables.KEY].drop_duplicates().shape[0]:,} with straw "
              f"crossings, {len(sgs):,} steps", flush=True)
