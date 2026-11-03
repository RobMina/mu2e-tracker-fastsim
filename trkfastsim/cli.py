"""Single CLI entrypoint:  python3 -m trkfastsim <subcommand> <inputs> [...]

Run from the umbrella root with PYTHONPATH=core:crv:tracker (site.env). Common flags: --outdir DIR (default
plots/<cmd>); --emit X.json / --merge for batchable ones.
"""
import argparse
import sys

from trkfastsim.sim import buildlib, run as steps
from trkfastsim.validation import elosscheck, fatecheck, msccheck, refcheck, stepcheck, slabcheck, surfcheck, tv2compare, v0check, v1compare, worldcheck
from trkfastsim.validation import seedcheck
from trkfastsim.validation import jaxcheck
from trkfastsim.validation import digicmp
from trkfastsim.digi import fastdigi
from trkfastsim.validation import seccensus
from trkfastsim.validation import shardhist, shardstats
from trkfastsim.validation import fieldcompare, fieldreco

ANALYSES = {
    "refcheck":  (refcheck.run,  "content check of StrawGasStepTableDumper tables: census, linkage, PrimaryFilter tracker term"),
    "v0check":   (v0check.run,   "T-V0: straws from the GDML against the G4 reference StrawGasStep entry/exit points"),
    "steps":     (steps.run,     "fast StrawGasSteps for the primary muons of gen tables (straight line, G4-matched eIon)"),
    "buildlib":  (buildlib.run,  "build the G4 secondary library (in-straw and off-axis non-muon StrawGasSteps) from reference tables"),
    "elosscheck": (elosscheck.run, "primary-muon eIon per straw: fast Urban sampler vs G4 per path-length bin (model and population)"),
    "worldcheck": (worldcheck.run, "world model event by event: |p| at the first straw vs generation, delivery and loss, fast vs G4"),
    "msccheck":  (msccheck.run,  "in-tracker scattering per consecutive step pair (p theta, straw-only vs support pairs), fast vs G4"),
    "stepcheck": (stepcheck.run, "diagnose the T-V1 path and eIon checks: crossing-angle population, G4 half-vs-half noise floor, band bins"),
    "fatecheck": (fatecheck.run, "fate of the muons the fast sim does not deliver (stopped above / missed / few steps), paired with G4"),
    "slabcheck": (slabcheck.run, "stage C world transport on one concrete slab vs Geant4 (art/run_slab.sh): survival, exit |p|, angle, lateral"),
    "surfcheck": (surfcheck.run, "tracker VD crossings (SurfaceSteps, sids 0-4) of the primary muon, fast vs G4 per sid (unpaired, per live time)"),
    "tv2compare": (tv2compare.run, "T-V2/T-V3: stock digi+reco+EventNtuple on fast vs G4 steps (yield, hits, chi2, residuals at TT_Outer; digis with --hits)"),
    "seedcheck": (seedcheck.run, "is the seeded interior-scattering excess real? seed at the ring tops / plane / tracker envelope, all events and the CRV-term subset"),
    "seccensus": (seccensus.run, "T-V1b secondaries per primary-muon step by species, eIon, place, wire distance, muon |p|; step-level panel occupancy with clustered errors (fast vs G4, optionally paired)"),
    "v1compare": (v1compare.run, "T-V1: primary-muon StrawGasSteps, fast vs G4 (straws per muon, path, eIon, entry, time)"),
    "jaxcheck": (jaxcheck.run, "steps --backend jax against numpy on the same primaries: statistics, paired PrimaryFilter, reproducibility, timing"),
    "digi-cond": (fastdigi.run_cond, "fast digitisation conditions (.npz) from the stock job's fhicl-dump, DB CSVs and the GDML"),
    "digi": (fastdigi.run_digi, "fast digitisation (port of StrawDigisFromStrawGasSteps): step tables -> StrawHitTableDumper-layout digi tables"),
    "field-reco": (fieldreco.run, "stock KKCHmu reco on injected field-on steps: arms (NAME=DIR of nts files) paired per event against --ref (yield, hits, fit quality, momentum)"),
    "field-compare": (fieldcompare.run, "field-on primary StrawGasSteps: fast chord transport (sim.fieldsteps) vs G4, restarted from the G4 tracker-VD entry state"),
    "digicmp": (digicmp.run, "fast vs stock digis on the same steps: digis/event, occupancy, TDC/TOT/PMP, dt, hit reco and combo hits, with a reseeded stock floor"),
    "shardhist": (shardhist.run, "per-shard (subrun) histograms of one arm: steps, digis, hit reco, fits; input of shardstats"),
    "shardstats": (shardstats.run, "paired per-shard statistic: batch variance, sign-flip null, family-wise limits, sizes; split-shard null"),
}
BATCHABLE = {"refcheck"}


def _nonnegative_int(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be nonnegative (0 disables batching)")
    return n


def build_parser():
    p = argparse.ArgumentParser(prog="trkfastsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, (fn, help_) in ANALYSES.items():
        sp = sub.add_parser(name, help=help_, description=sys.modules[fn.__module__].__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        sp.add_argument("inputs", nargs="+", help="input files (globs allowed, quote them)")
        sp.add_argument("--outdir", default=None, help="output dir (default plots/<cmd>)")
        if name in ("v0check", "steps", "buildlib", "worldcheck", "msccheck", "stepcheck", "fatecheck", "seccensus"):
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
        if name == "steps":
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, i)")
            sp.add_argument("--stage", default="A", choices=["A", "C"],
                            help="A: straight from the vertex; C: GDML world material + MSC down to the tracker")
            sp.add_argument("--library", default=None, metavar="ROOT", help="G4 secondary library (buildlib) to paste secondaries")
            sp.add_argument("--no-offaxis", action="store_true", help="with --library: in-straw secondaries only")
            sp.add_argument("--offaxis-plane-bin", type=int, default=6,
                            help="off-axis draw: planes per plane-extent bin of the matching key (0 = no plane matching)")
            sp.add_argument("--offaxis-topology-anchor", action="store_true", help="opt-in donor topology anchor; requires buildlib --topology")
            sp.add_argument("--offaxis-edge-anchor", action="store_true",
                            help="off-axis paste: steps outside the library muon's planes anchor on the fast muon's first/last plane (opt-in)")
            sp.add_argument("--library-drop-muons", action="store_true",
                            help="with --library: leave out drawn muon steps (other primary muons of the donor event, which the "
                                 "fast transport already carries; opt-in)")
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
                            help="stage C: named world modification (trkfastsim.sim.worldmods, e.g. issue2017) for geometry sensitivity runs")
            sp.add_argument("--kink-sup-xx0", type=float, default=0.25,
                            help="stage C: split support segments into kinks of at most this x/X0 (default 0.25 = steps.KINK_SUP_XX0; 0 = one kink per segment)")
            sp.add_argument("--scatter-in", action="store_true",
                            help="stage C, NumPy: support kinks also on lines that cross no further straw, so support scattering "
                                 "can move muons onto the straws (TrackerInterior.transport scatter_in; opt-in)")
            sp.add_argument("--band-spread", action="store_true",
                            help="stage C: spread each world height band's scattering along its path (default: at the band centre)")
            sp.add_argument("--backend", default="numpy", choices=["numpy", "jax"],
                            help="numpy (default, the reference) or jax: world/interior transport, crossings and gas eIon on the "
                                 "JAX device (CPU or GPU; different random streams; validate with jaxcheck)")
            sp.add_argument("--navigation", default="legacy", choices=["legacy", "ordered-stored-mean-v1", "ordered-stored-sampled-v1"],
                            help="opt-in ordered stored-material mean or sample-coupled loss, slice clock and angular MSC; primary-only")
            sp.add_argument("--ordered-signed-endpoint-budget-mm", type=float, default=None,
                            help="ordered mode only: explicit bounded signed-source endpoint arm (strict by default)")
            sp.add_argument("--ordered-stopping-evaluation", choices=["entry", "midpoint"], default="entry",
                            help="ordered mode only: explicit midpoint predictor for loss evaluation (entry remains default)")
            sp.add_argument("--ledger-arms", default=None, metavar="PRESET",
                            help="stage C transport-ledger arms (trkfastsim.sim.ledger.PRESETS: legacy, A, E, M, T, AM, AMT; "
                                 "default steps.LEDGER_ARMS_DEFAULT)")
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
        if name == "fatecheck":
            sp.add_argument("--seed", type=int, default=1, help="base seed, as steps (file i uses (seed, i))")
        if name == "elosscheck":
            sp.add_argument("--fast", nargs="+", default=None, metavar="ROOT", help="fast-sim tables for the population check")
            sp.add_argument("--seed", type=int, default=1)
            sp.add_argument("--splits", default="1,2,4", help="sub-step counts to test, comma-separated")
        if name == "buildlib":
            sp.add_argument("--topology", action="store_true", help="retain donor plane topology for placement diagnostics (opt-in)")
            sp.add_argument("--out", default=None, metavar="ROOT", help="library file (default <outdir>/trklib.root)")
            sp.add_argument("--format", type=int, default=2, choices=[1, 2], help="library format (2: straw-local placement)")
        if name in ("v1compare", "worldcheck", "msccheck", "stepcheck", "fatecheck", "surfcheck"):
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4 reference sgsteps tables")
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
        if name == "seccensus":
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4 reference sgsteps tables")
            sp.add_argument("--paired", action="store_true", help="same CRY primaries: match files by subrun, events selected in both arms")
            sp.add_argument("--trees", action="store_true",
                            help="step-tree diagnostic: covariance of secondary eIon along each muon and its same-tree share (opt-in)")
        if name == "tv2compare":
            sp.add_argument("--compare", nargs="+", required=True, metavar="ROOT", help="G4-arm EventNtuples")
            sp.add_argument("--steps", nargs="+", required=True, metavar="ROOT", help="fast step tables of the fast arm")
            sp.add_argument("--steps-ref", nargs="+", required=True, metavar="ROOT", help="G4 sgsteps tables of the G4 arm")
            sp.add_argument("--hits", nargs="+", default=None, metavar="ROOT", help="fast-arm StrawHitTableDumper tables")
            sp.add_argument("--hits-ref", nargs="+", default=None, metavar="ROOT", help="G4-arm StrawHitTableDumper tables")
            sp.add_argument("--tot-breakdown", action="store_true", help="track-hit TOT/edep by true relation and true path length")
            if name == "tv2compare":
                sp.add_argument("--paired", action="store_true",
                                help="both arms on the same CRY primaries: split the yield difference into selection-only and both-selected parts")
        if name == "digi-cond":
            sp.add_argument("--preamp", default=None, metavar="CSV", help="TrkPreampStraw content (dbTool print-run ... --content)")
            sp.add_argument("--align", default=None, metavar="CSV", help="TrkAlignStrawSim content (wire-end offsets of alignedTrackerSim)")
            sp.add_argument("--gdml", default=None, help="GDML world export (default $M2FS_DATA/crv_fastsim/geometry/...)")
            sp.add_argument("--global-constants", default=None, metavar="TXT", help="GlobalConstantsService physicsParams file")
            sp.add_argument("--out", default=None, metavar="NPZ", help="conditions file (default <outdir>/digicond.npz)")
            sp.add_argument("--check-ref", default=None, metavar="ROOT", help="StrawHitTableDumper file: compare the straws with its tracker")
        if name == "digi":
            sp.add_argument("--cond", default=None, metavar="NPZ", help="conditions from digi-cond")
            sp.add_argument("--events", nargs="+", default=None, metavar="ROOT",
                            help="StrawHitTableDumper files: digitise exactly their events (default: PrimaryFilter tracker term)")
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, i)")
            sp.add_argument("--chunk", type=int, default=1000, help="events per vectorised batch")
            sp.add_argument("--backend", choices=["numpy", "jax"], default="numpy",
                            help="digitizer backend (default numpy); JAX uses separate explicit random keys")
        if name == "digicmp":
            sp.add_argument("--ref", nargs="+", required=True, metavar="ROOT", help="stock StrawHitTableDumper tables (same steps)")
            sp.add_argument("--floor", nargs="+", default=None, metavar="ROOT", help="stock tables with another digitisation seed")
            sp.add_argument("--hit-cond", default=None, metavar="JSON", help="hit-reco conditions (reco-conditions); default: from --ref[0]")
            sp.add_argument("--f2", action="store_true", help="also run the fast reco chain (time clusters, seeds, F2 fit; JAX) on every arm")
            sp.add_argument("--f2-chunk", type=_nonnegative_int, default=0, metavar="N",
                            help="events per F2 reco batch, on every arm (default 0: all events at once)")
            sp.add_argument("--f2-fixed-shapes", action="store_true",
                            help="bound finder allocations and pad F2 to fixed track/hit buckets without changing fit physics")
            sp.add_argument("--f2-progress", action="store_true", help="log per-batch F2 timing/shapes as JSON lines")
            sp.add_argument("--require-same-events", action="store_true", help="reject unequal or duplicate event keys instead of intersecting")
            sp.add_argument("--drift-source", default=None, metavar="ROOT",
                            help="F2 drift conditions shared by all arms (default: --ref[0]); permits digi-only references")
        if name == "shardhist":
            sp.add_argument("--arm", required=True, help="arm name (output <outdir>/<arm>.npz)")
            sp.add_argument("--kind", default="steps", choices=["steps", "digis"])
            sp.add_argument("--select-steps", nargs="+", default=None, metavar="ROOT",
                            help="digis: step tables of this arm; their tracker-term events and subruns are used")
            sp.add_argument("--hit-cond", default=None, metavar="JSON", help="digis: hit-reco conditions (default: from --drift-source or inputs[0])")
            sp.add_argument("--stock-hits", action="store_true", help="digis: also the stored stock straw/combo hits (arm@stockreco)")
            sp.add_argument("--f2", action="store_true", help="digis: fast reco chain per shard (fit/*; JAX)")
            sp.add_argument("--drift-source", default=None, metavar="ROOT", help="digis --f2: drift conditions (default inputs[0])")
            sp.add_argument("--kkline", nargs="+", default=None, metavar="ROOT", help="digis: stock EventNtuples of these events (arm@kkline)")
            sp.add_argument("--batches", type=int, default=1, help="shards per subrun, by event number modulo this (default 1)")
        if name == "shardstats":
            sp.add_argument("--pairs", nargs="+", default=None, metavar="REF:CAND", help="paired comparisons (arm names)")
            sp.add_argument("--splits", nargs="+", default=None, metavar="ARM", help="unpaired split-shard null of these arms")
            sp.add_argument("--shape", action="store_true", help="also the shape mode (candidate scaled by the selected-event ratio)")
            sp.add_argument("--keys", nargs="+", default=None, metavar="GLOB", help="histogram keys to use (fnmatch globs)")
            sp.add_argument("--nflip", type=int, default=20000)
            sp.add_argument("--alpha", type=float, default=0.01, help="family-wise rate per comparison")
            sp.add_argument("--detail", nargs="+", default=None, metavar="KEY", help="per-bin ratio and pull of these keys")
            sp.add_argument("--detail-min-pull", type=float, default=0.0, help="print only bins with |pull| >= this")
        if name == "field-reco":
            sp.add_argument("--events", default=None, help="injector table (dump/sgsteps) whose events restrict the denominator")
            sp.add_argument("--file-tables", default=None, help="directory of the per-file dumper tables sgsteps.<stem>.root")
            sp.add_argument("--ref", default="g4", help="reference arm name")
        if name == "field-compare":
            sp.add_argument("--gdml", required=True, help="GDML world export of the sample's geometry (straws, support)")
            sp.add_argument("--species", type=int, default=13, choices=[11, 13], help="MC primary |pdg| (13 muons, 11 electrons)")
            sp.add_argument("--map", dest="field_map", default=None, metavar="HEADER", help="G4BL field map header (e.g. DSMap.header)")
            sp.add_argument("--bz", type=float, default=None, help="constant Bz [T] instead of a map (default 1.0 without --map)")
            sp.add_argument("--no-field", action="store_true", help="straight-line control arm")
            sp.add_argument("--seed", type=int, default=1)
            sp.add_argument("--chord-mm", type=float, default=20.0, help="longest chord of the helix [mm]")
            sp.add_argument("--sagitta-mm", type=float, default=0.02, help="largest chord sagitta [mm]")
            sp.add_argument("--entry-vds", type=int, nargs="+", default=list(fieldcompare.TRACKER_VDS),
                            help="VirtualDetectorIds whose first crossing is the entry state (13 = TT_FrontHollow)")
            sp.add_argument("--entry", choices=["vd", "surface", "first"], default="vd",
                            help="entry state from the first tracker VD crossing (default), the first tracker SurfaceStep "
                                 "(entry vdid = -1 - SurfaceId in the output), or the earlier of the two")
            sp.add_argument("--origin", type=float, nargs=3, default=None,
                            help="detector-frame origin in the Mu2e frame [mm] (default: fitted from the G4 steps)")
            sp.add_argument("--interior", action=argparse.BooleanOptionalAction, default=True,
                            help="support structure from the GDML (default on)")
            sp.add_argument("--max-events", type=int, default=None, help="first N events with a primary (quick checks)")
            sp.add_argument("--delta-census", action="store_true",
                            help="G4 e+- daughters of the primaries above 10 MeV: in straw gas, wall or elsewhere, per crossing")
            sp.add_argument("--offset-bound", action="store_true",
                            help="first-straw entry offsets per entry type and |p| bin vs the Highland spread of G4's own "
                                 "support scattering (first_straw_offset_bound)")
            sp.add_argument("--gas-state", choices=["solid", "gas"], default="solid",
                            help="density-effect branch for the straw gas: solid (as the extracted chain) or Geant4's "
                                 "gas branch (the GDML StrawGas state)")
            sp.add_argument("--mu-radiative", action="store_true",
                            help="muons: also bremsstrahlung, pair and photonuclear losses in the solids (opt-in)")
            sp.add_argument("--save-steps", action="store_true", help="also write the fast steps and final states (fast_steps.root)")
            sp.add_argument("--dump-tracks", type=int, default=0, metavar="N",
                            help="write tracks.json: the step lists of N tracks above 2 GeV/c per entry type, both arms")
            sp.add_argument("--write-tables", action="store_true",
                            help="also write primary-only sgsteps/particles TTrees (fast and G4) for the StrawGasStepsFromTable injector")
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
