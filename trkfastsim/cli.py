"""Single CLI entrypoint:  python3 -m trkfastsim <subcommand> <inputs> [...]

Run from the umbrella root with PYTHONPATH=core:crv:tracker (site.env). Common flags: --outdir DIR (default
plots/<cmd>); --emit X.json / --merge for batchable ones.
"""
import argparse
import sys

from .analyses import buildlib, elosscheck, msccheck, refcheck, steps, v0check, v1compare, worldcheck

ANALYSES = {
    "refcheck":  (refcheck.run,  "content check of StrawGasStepTableDumper tables: census, linkage, PrimaryFilter tracker term"),
    "v0check":   (v0check.run,   "T-V0: straws from the GDML against the G4 reference StrawGasStep entry/exit points"),
    "steps":     (steps.run,     "fast StrawGasSteps for the primary muons of gen tables (straight line, G4-matched eIon)"),
    "buildlib":  (buildlib.run,  "build the G4 secondary library (in-straw and off-axis non-muon StrawGasSteps) from reference tables"),
    "elosscheck": (elosscheck.run, "primary-muon eIon per straw: fast Urban sampler vs G4 per path-length bin (model and population)"),
    "worldcheck": (worldcheck.run, "world model event by event: |p| at the first straw vs generation, delivery and loss, fast vs G4"),
    "msccheck":  (msccheck.run,  "in-tracker scattering per consecutive step pair (p theta, straw-only vs support pairs), fast vs G4"),
    "v1compare": (v1compare.run, "T-V1: primary-muon StrawGasSteps, fast vs G4 (straws per muon, path, eIon, entry, time)"),
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
        if name in ("v0check", "steps", "worldcheck", "msccheck"):
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
        if name == "elosscheck":
            sp.add_argument("--fast", nargs="+", default=None, metavar="ROOT", help="fast-sim tables for the population check")
            sp.add_argument("--seed", type=int, default=1)
            sp.add_argument("--splits", default="1,2,4", help="sub-step counts to test, comma-separated")
        if name == "buildlib":
            sp.add_argument("--out", default=None, metavar="ROOT", help="library file (default <outdir>/trklib.root)")
        if name in ("v1compare", "worldcheck", "msccheck"):
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4 reference sgsteps tables")
        if name in BATCHABLE:
            sp.add_argument("--emit", default=None, metavar="JSON", help="write this batch's counts, no summary")
            sp.add_argument("--merge", action="store_true", help="inputs are --emit JSONs: sum and report")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    kw = vars(args)
    cmd = kw.pop("cmd")
    inputs = kw.pop("inputs")
    outdir = kw.pop("outdir") or f"plots/{cmd}"
    ANALYSES[cmd][0](inputs, outdir, **kw)


if __name__ == "__main__":
    main()
