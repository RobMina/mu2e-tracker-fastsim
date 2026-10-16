"""Single CLI entrypoint:  python3 -m trkfastsim <subcommand> <inputs> [...]

Run from the umbrella root with PYTHONPATH=core:crv:tracker (site.env). Common flags: --outdir DIR (default
plots/<cmd>); --emit X.json / --merge for batchable ones.
"""
import argparse
import sys

from .analyses import refcheck, steps, v0check, v1compare

ANALYSES = {
    "refcheck":  (refcheck.run,  "content check of StrawGasStepTableDumper tables: census, linkage, PrimaryFilter tracker term"),
    "v0check":   (v0check.run,   "T-V0: straws from the GDML against the G4 reference StrawGasStep entry/exit points"),
    "steps":     (steps.run,     "fast StrawGasSteps for the primary muons of gen tables (straight line, G4-matched eIon)"),
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
        if name in ("v0check", "steps"):
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
        if name == "steps":
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, i)")
        if name == "v1compare":
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
