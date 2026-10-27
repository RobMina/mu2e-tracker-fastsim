"""Single CLI entrypoint:  python3 -m trkfastsim <subcommand> <inputs> [...]

Run from the umbrella root with PYTHONPATH=core:crv:tracker (site.env). Common flags: --outdir DIR (default
plots/<cmd>); --emit X.json / --merge for batchable ones.
"""
import argparse
import sys

from .analyses import (buildlib, elosscheck, fatecheck, libplacecheck, msccheck, refcheck, secdiag, slowcheck, stepcheck, steps,
                       slabcheck, surfcheck, survivalcheck, tubecheck, tv2compare, v0check, v1compare, worldcheck)
from .analyses import geomcheck
from .analyses import interiorcensus
from .analyses import seedcheck
from .analyses import rangecheck
from .analyses import arrivalcheck
from .analyses import jaxcheck

ANALYSES = {
    "refcheck":  (refcheck.run,  "content check of StrawGasStepTableDumper tables: census, linkage, PrimaryFilter tracker term"),
    "v0check":   (v0check.run,   "T-V0: straws from the GDML against the G4 reference StrawGasStep entry/exit points"),
    "steps":     (steps.run,     "fast StrawGasSteps for the primary muons of gen tables (straight line, G4-matched eIon)"),
    "buildlib":  (buildlib.run,  "build the G4 secondary library (in-straw and off-axis non-muon StrawGasSteps) from reference tables"),
    "elosscheck": (elosscheck.run, "primary-muon eIon per straw: fast Urban sampler vs G4 per path-length bin (model and population)"),
    "worldcheck": (worldcheck.run, "world model event by event: |p| at the first straw vs generation, delivery and loss, fast vs G4"),
    "msccheck":  (msccheck.run,  "in-tracker scattering per consecutive step pair (p theta, straw-only vs support pairs), fast vs G4"),
    "stepcheck": (stepcheck.run, "diagnose the T-V1 path and eIon checks: crossing-angle population, G4 half-vs-half noise floor, band bins"),
    "geomcheck": (geomcheck.run, "true G4 path vs the fast world for muons one arm delivers (fate, footprint, material along both paths)"),
    "fatecheck": (fatecheck.run, "fate of the muons the fast sim does not deliver (stopped above / missed / few steps), paired with G4"),
    "slowcheck": (slowcheck.run, "slow muons in the tracker per |p| bin: rates by good-step count, entry state, in-tracker loss, fast vs G4"),
    "survivalcheck": (survivalcheck.run, "selected-muon rate by generated |p|, world column and energy margin, fast vs G4 (unpaired)"),
    "tubecheck": (tubecheck.run, "tube/polycone material above the tracker the fast world omits, along G4-delivered lines (needs worldcheck output)"),
    "slabcheck": (slabcheck.run, "stage C world transport on one concrete slab vs Geant4 (art/run_slab.sh): survival, exit |p|, angle, lateral"),
    "surfcheck": (surfcheck.run, "tracker VD crossings (SurfaceSteps, sids 0-4) of the primary muon, fast vs G4 per sid (unpaired, per live time)"),
    "tv2compare": (tv2compare.run, "T-V2/T-V3: stock digi+reco+EventNtuple on fast vs G4 steps (yield, hits, chi2, residuals at TT_Outer; digis with --hits)"),
    "secdiag": (secdiag.run, "why fewer non-muon track hits: secondary steps by place, shared-straw timing, track hits by straw content, fast vs G4"),
    "libplacecheck": (libplacecheck.run, "placement of non-muon steps in their straws (along-wire offset, wire distance), fast vs G4"),
    "arrivalcheck": (arrivalcheck.run, "muon state at the plane above the tracker, fast vs G4 on the same primaries, for every G4 muon crossing it (arrival, |p|, offset, aim) by energy margin"),
    "rangecheck": (rangecheck.run, "energy loss near range-out in thick slabs: G4 slabs vs the interior loss model (transmission, exit KE)"),
    "seedcheck": (seedcheck.run, "is geomcheck's seeded interior-scattering excess real? seed at the ring tops / plane / tracker envelope, all events and the CRV-term subset"),
    "interiorcensus": (interiorcensus.run, "tracker-interior material per volume group: fast shells vs G4 trajectory loss, plane to first straw, and mass vs GDML"),
    "v1compare": (v1compare.run, "T-V1: primary-muon StrawGasSteps, fast vs G4 (straws per muon, path, eIon, entry, time)"),
    "jaxcheck": (jaxcheck.run, "steps --backend jax against numpy on the same primaries: statistics, paired PrimaryFilter, reproducibility, timing"),
}
BATCHABLE = {"refcheck"}


def build_parser():
    p = argparse.ArgumentParser(prog="trkfastsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, (fn, help_) in ANALYSES.items():
        sp = sub.add_parser(name, help=help_, description=sys.modules[fn.__module__].__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        sp.add_argument("inputs", nargs="+", help="input files (globs allowed, quote them)")
        sp.add_argument("--outdir", default=None, help="output dir (default plots/<cmd>)")
        if name in ("v0check", "steps", "buildlib", "libplacecheck", "worldcheck", "msccheck", "stepcheck", "tubecheck", "fatecheck", "survivalcheck"):
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
        if name == "steps":
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, i)")
            sp.add_argument("--stage", default="A", choices=["A", "C"],
                            help="A: straight from the vertex; C: GDML world material + MSC down to the tracker")
            sp.add_argument("--library", default=None, metavar="ROOT", help="G4 secondary library (buildlib) to paste secondaries")
            sp.add_argument("--no-offaxis", action="store_true", help="with --library: in-straw secondaries only")
            sp.add_argument("--offaxis-plane-bin", type=int, default=6,
                            help="off-axis draw: planes per plane-extent bin of the matching key (0 = no plane matching)")
            sp.add_argument("--tag", default="", help="suffix of the output file prefix (fasttrk<stage>L<tag>.)")
            sp.add_argument("--sliced-msc", action="store_true", help="stage C: per-band |p| in the Highland scattering")
            sp.add_argument("--first-muon-only", action="store_true",
                            help="transport only the first generated muon of each event (before 2026-10-24; default: all)")
            sp.add_argument("--near-margin", type=float, default=1500.0,
                            help="stage C: transport through the world only muons whose straight line passes within this "
                                 "many mm of the straws' footprint at the plane (1e9 = every downward muon)")
            sp.add_argument("--no-interior", action="store_true",
                            help="stage C: leave out the tracker's support structure (rings, ribs, beams) between straws")
            sp.add_argument("--no-interior-msc", action="store_true",
                            help="stage C: energy loss in the support structure but no scattering in it (straight tracks)")
            sp.add_argument("--msc-tail", action=argparse.BooleanOptionalAction, default=True,
                            help="stage C: single-scattering (Rutherford) tail on top of the Gaussian MSC core "
                                 "(default on; --no-msc-tail for the Gaussian core only)")
            sp.add_argument("--tail-k", type=float, default=None,
                            help="tail threshold in units of theta0 (default: where tail and Gaussian cross)")
            sp.add_argument("--kink-xx0", type=float, default=1e-3,
                            help="stage C: x/X0 of straw material between in-tracker kinks (default 1e-3)")
            sp.add_argument("--no-surfsteps", action="store_true",
                            help="do not write the surfsteps tree (tracker VD crossings of the primary muon)")
            sp.add_argument("--interior-msc-cal", action=argparse.BooleanOptionalAction, default=False,
                            help="stage C: G4-slab width calibration of the support-structure scattering (steps.INTERIOR_MSC_CAL)")
            sp.add_argument("--lever-moments", action="store_true",
                            help="stage C: interior kink lever arms from the support material's own moments (default: segment midpoint)")
            sp.add_argument("--no-calo", action="store_true",
                            help="stage C: leave the calorimeter disk out of the interior (diagnostic)")
            sp.add_argument("--exit-scatter", action=argparse.BooleanOptionalAction, default=True,
                            help="stage C: scatter the material behind the last straw into the path for the tracker VD (surfsteps) exit crossings (default on; --no-exit-scatter)")
            sp.add_argument("--world-mods", default=None,
                            help="stage C: named world modification (trkfastsim.worldmods, e.g. issue2017) for geometry sensitivity runs")
            sp.add_argument("--kink-sup-xx0", type=float, default=0.25,
                            help="stage C: split support segments into kinks of at most this x/X0 (default 0.25 = steps.KINK_SUP_XX0; 0 = one kink per segment)")
            sp.add_argument("--band-spread", action="store_true",
                            help="stage C: spread each world height band's scattering along its path (default: at the band centre)")
            sp.add_argument("--backend", default="numpy", choices=["numpy", "jax"],
                            help="numpy (default, the reference) or jax: in-tracker transport, crossings and gas eIon on the "
                                 "JAX device (CPU or GPU; statistically equal, different random streams; trkfastsim.steps_jax)")
        if name == "geomcheck":
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
            sp.add_argument("--fast", nargs="+", required=True, metavar="ROOT", help="fast step tables made from these gen tables")
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4 reference sgsteps tables (with trajpts)")
            sp.add_argument("--seed", type=int, default=1, help="base seed of the fast steps run")
            sp.add_argument("--no-tubes", action="store_true", help="skip the tube/polycone column along the G4 paths")
        if name == "arrivalcheck":
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4 reference sgsteps tables (with trajpts)")
            sp.add_argument("--seed", type=int, default=1, help="base seed, as steps (file i uses (seed, i))")
            sp.add_argument("--no-sliced", action="store_true", help="column scattering at one |p| (geometric mean) instead of per band")
            sp.add_argument("--no-msc-tail", action="store_true", help="no single-scattering tail in the world column")
            sp.add_argument("--band-mm", type=float, default=None, help="uniform height bands of this size [mm] (diagnostic)")
            sp.add_argument("--crv", nargs="+", default=None, metavar="ROOT", help="G4 CrvStep tables: keep CRV-term events only")
        if name == "jaxcheck":
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, i), as steps")
            sp.add_argument("--stage", default="C", choices=["A", "C"])
            sp.add_argument("--repeat", type=int, default=2, help="jax passes on the same seed (the first compiles)")
            sp.add_argument("--numpy-repeat", type=int, default=1, help="numpy passes on the same seed")
            sp.add_argument("--no-exit-scatter", action="store_true", help="skip exit_scatter (surfsteps path) in both")
            sp.add_argument("--save", action="store_true", help="also write both step tables")
            sp.add_argument("--concat", action="store_true",
                            help="one make_steps call on all inputs concatenated (throughput at a large batch)")
            sp.add_argument("--lanes", type=int, default=None, help="jax muons per device chunk (default steps_jax.LANES)")
            sp.add_argument("--cross", default=None, metavar="DIR",
                            help="--save outdir of a run on another JAX device: compare its jax tables step by step")
            sp.add_argument("--kernel-times", action="store_true",
                            help="one more jax pass with every steps_jax kernel synchronised and timed")
        if name == "rangecheck":
            sp.add_argument("--nmu", type=int, default=30000, help="fast muons per point")
            sp.add_argument("--seed", type=int, default=1)
        if name == "seedcheck":
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
            sp.add_argument("--crv", nargs="+", required=True, metavar="ROOT", help="G4 reference CrvStep tables (crvsteps.*.root)")
            sp.add_argument("--seed", type=int, default=1)
            sp.add_argument("--lever-moments", action="store_true",
                            help="interior kinks with the material's own lever-arm moments (TrackerInterior.transport lever_moments)")
            sp.add_argument("--kink-sup-xx0", type=float, default=0.25,
                            help="split support segments into kinks of at most this x/X0 (default 0.25 = steps.KINK_SUP_XX0; 0 = one kink per segment)")
            sp.add_argument("--kink-p-mean", action="store_true",
                            help="interior kinks scatter at the mean of the group's entry and exit |p| (default exit |p|)")
        if name == "interiorcensus":
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
            sp.add_argument("--nsub", type=int, default=8, help="phi samples per sector interval (as TrackerInterior)")
        if name == "fatecheck":
            sp.add_argument("--seed", type=int, default=1, help="base seed, as steps (file i uses (seed, i))")
        if name == "elosscheck":
            sp.add_argument("--fast", nargs="+", default=None, metavar="ROOT", help="fast-sim tables for the population check")
            sp.add_argument("--seed", type=int, default=1)
            sp.add_argument("--splits", default="1,2,4", help="sub-step counts to test, comma-separated")
        if name == "buildlib":
            sp.add_argument("--out", default=None, metavar="ROOT", help="library file (default <outdir>/trklib.root)")
            sp.add_argument("--format", type=int, default=2, choices=[1, 2], help="library format (2: straw-local placement)")
        if name in ("v1compare", "worldcheck", "msccheck", "stepcheck", "fatecheck", "slowcheck", "survivalcheck", "surfcheck"):
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4 reference sgsteps tables")
        if name == "libplacecheck":
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4 reference sgsteps tables")
            sp.add_argument("--fast-ref", nargs="+", default=None, metavar="ROOT", help="second fast arm (e.g. v1 library)")
        if name == "slabcheck":
            sp.add_argument("--nbands", type=int, default=4, help="height bands the slab is split into (as stage C)")
            sp.add_argument("--nmu", type=int, default=100000, help="fast muons per point")
            sp.add_argument("--seed", type=int, default=1)
            sp.add_argument("--model", default="world", choices=["world", "interior"],
                            help="fast arm: stage C world transport (default) or the tracker-interior kink (TrackerInterior)")
            sp.add_argument("--material", default=None, help="interior: NIST slab material (default: input's directory name)")
            sp.add_argument("--cal", default=None, metavar="JSON", help="interior: calibration to test (default: built-in)")
            sp.add_argument("--fit", action="store_true", help="interior: fit the per-material width correction")
            sp.add_argument("--kink-p-mean", action="store_true",
                            help="interior only: scatter at mean entry/exit |p| (default exit |p|); diagnostic, not production validation")
        if name in ("tv2compare", "secdiag"):
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4-arm EventNtuples")
            sp.add_argument("--steps", nargs="+", required=True, metavar="ROOT", help="fast step tables of the fast arm")
            sp.add_argument("--steps-ref", nargs="+", required=True, metavar="ROOT", help="G4 sgsteps tables of the G4 arm")
            sp.add_argument("--hits", nargs="+", default=None, metavar="ROOT", help="fast-arm StrawHitTableDumper tables")
            sp.add_argument("--hits-ref", nargs="+", default=None, metavar="ROOT", help="G4-arm StrawHitTableDumper tables")
            sp.add_argument("--tot-breakdown", action="store_true", help="track-hit TOT/edep by true relation and true path length")
        if name in BATCHABLE:
            sp.add_argument("--emit", default=None, metavar="JSON", help="write this batch's counts, no summary")
            sp.add_argument("--merge", action="store_true", help="inputs are --emit JSONs: sum and report")
    return p


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv  # reco-<cmd> -> trkfastsim.reco.cli
    if a and a[0].startswith("reco-"): from .reco import cli as rc; return rc.main([a[0][5:]] + a[1:])
    args = build_parser().parse_args(argv)
    kw = vars(args)
    cmd = kw.pop("cmd")
    inputs = kw.pop("inputs")
    outdir = kw.pop("outdir") or f"plots/{cmd}"
    ANALYSES[cmd][0](inputs, outdir, **kw)


if __name__ == "__main__":
    main()
