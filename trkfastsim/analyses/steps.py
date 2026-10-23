"""steps -- run the fast StrawGasStep generator (trkfastsim.steps) on generated primaries, one table file per input.

Input: gen tables of the G4 reference (`gen` + `subruns` trees: crvfastsim's gen.*.root from the gen stream, or the
tracker dumper's gen_tables.*.root). Output: fasttrk.<input>.root with the sgsteps/particles/gen/subruns trees in the
StrawGasStepTableDumper layout, same (run, subrun, event) keys, so `v1compare FAST --compare G4` matches events.

    steps gen.001400_00000001.root [...] --gdml mu2e_extracted_v04.gdml --outdir DIR [--seed N] [--stage A|C]

--stage A: straight from the generated vertex; C: GDML world material and multiple scattering down to the tracker.
Stage C includes the tracker's support structure (TrackerInterior: loss and scattering per segment between straws)
unless --no-interior; --no-interior-msc keeps the loss but no scattering in it. The single-scattering tail above
the Gaussian core (world column and interior) is on by default (--no-msc-tail: Gaussian only); --tail-k K puts its
threshold at K theta0.
--library LIB.root: add the G4 library's secondaries (buildlib): in-straw and off-axis (--no-offaxis: in-straw only);
output prefix fasttrk<stage>L.
"""
import os

import numpy as np

from mu2efastsim import tables as core_tables
from .. import geometry, library as lib_mod, steps as st, surfaces, tables

GDML_NAME = "crv_fastsim/geometry/mu2e_extracted_v04.gdml"   # under $M2FS_DATA (crv/geometry/README.md)


def default_gdml():
    data = os.environ.get("M2FS_DATA")
    if not data:
        raise SystemExit("pass --gdml or set M2FS_DATA (source site.env)")
    return os.path.join(data, GDML_NAME)


def run(inputs, outdir, seed=1, gdml=None, stage="A", library=None, no_offaxis=False, offaxis_plane_bin=6, tag="",
        sliced_msc=False, no_interior=False, near_margin=st.FOOTPRINT_MARGIN, no_interior_msc=False, msc_tail=True, kink_xx0=st.KINK_XX0,
        tail_k=None, band_spread=False, no_surfsteps=False, interior_msc_cal=False, lever_moments=False, kink_sup_xx0=st.KINK_SUP_XX0, no_calo=False, world_mods=None, exit_scatter=False, **kw):
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    world = st.load_world(gdml, geom, mods=world_mods) if stage == "C" else None
    if world_mods:
        print(f"# world modification: {world_mods}", flush=True)
    interior = st.TrackerInterior(gdml, geom, calo=not no_calo) if stage == "C" and not no_interior else None
    if interior is not None:
        if interior_msc_cal and st.INTERIOR_MSC_CAL is None:
            raise SystemExit("--interior-msc-cal: steps.INTERIOR_MSC_CAL is not set")
        print(f"# tracker interior: {len(interior.shells.coax)} support shells ({interior.shells.n_coaxial} coaxial), "
              f"{len(interior.shells.calo) if interior.shells.calo is not None else 0} calorimeter pieces"
              + (" ; support msc calibrated (INTERIOR_MSC_CAL)" if interior_msc_cal else "")
              , flush=True)
    lib = lib_mod.read(library) if library else None
    for i, f in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([int(seed), i])
        gen = tables.read(f, "gen")
        sub = tables.read(f, "subruns")
        sampler = lib_mod.InStrawSampler(lib, np.random.default_rng([int(seed), i, 1])) if lib else None
        offaxis = (lib_mod.OffAxisSampler(lib, np.random.default_rng([int(seed), i, 2]), plane_bin_width=offaxis_plane_bin)
                   if lib and not no_offaxis else None)
        sgs, parts = st.make_steps(gen, geom, rng, stage=stage, world=world, sampler=sampler, offaxis=offaxis,
                                   sliced_msc=sliced_msc, interior=interior, near_margin=near_margin,
                                   interior_msc=not no_interior_msc, msc_tail=msc_tail, tail_k=tail_k,
                                   kink_xx0=kink_xx0, band_spread=band_spread,
                                   interior_msc_cal=st.INTERIOR_MSC_CAL if interior_msc_cal else None,
                                   lever_moments=lever_moments, kink_sup_xx0=kink_sup_xx0, exit_scatter=exit_scatter)
        if interior is not None:
            print(f"#   interior: {getattr(st.make_steps, 'interior_stats', {})}", flush=True)
        if offaxis is not None:
            print(f"#   off-axis: {getattr(st.make_steps, 'offaxis_stats', {})}", flush=True)
        name = f"fasttrk{stage}{'L' if lib else ''}{tag}." + os.path.basename(f).replace("gen_tables.", "").replace("gen.", "")
        verts = getattr(st.make_steps, "path_vertices", None) if stage == "C" and not no_interior_msc else None
        surf = surfaces.surface_steps(sgs, verts) if not no_surfsteps else None
        core_tables.write(os.path.join(outdir, name), {"sgsteps": sgs, "particles": parts, "gen": gen,
                                                        "subruns": sub, "surfsteps": surf}, as_ttree=True)
        print(f"# {f}: {len(parts):,} primary muons, {sgs[tables.KEY].drop_duplicates().shape[0]:,} with straw "
              f"crossings, {len(sgs):,} steps", flush=True)
