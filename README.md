# mu2e-tracker-fastsim

Geant4-free tracker simulation (`StrawGasStep`s) and fast tracker reconstruction for the Mu2e extracted (KPP, no
field) cosmic configuration: stage 2 of the Mu2e fast simulation, beside
[`mu2e-crv-fastsim`](https://github.com/RobMina/mu2e-crv-fastsim). Physics primitives (energy loss, scattering,
materials, crossings, tables) come from [`mu2e-fastsim-core`](https://github.com/RobMina/mu2e-fastsim-core). All
three are submodules of the umbrella repository [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim); work from
a clone of the umbrella (setup there).

This repository holds technical software only: no physics results, no Mu2e-internal conditions, tables or data.

## Status (2026-10-22)

Fast simulation of the primary muon's `StrawGasStep`s is in place. Fast reconstruction has started
(`trkfastsim.reco`, below): on the same stock digis, the straw-hit and combo-hit reco reproduce stock (T-VR1), in numpy
and in JAX twins that are bitwise equal to numpy, and so do the time clusters and line seeds (T-VR1b). The digitisation
prototypes and the track fit are next (`docs/plan.md`). The generator `python3 -m trkfastsim steps` reruns the CRY primaries of the G4 reference sample:

| stage | what it does |
|---|---|
| A | straight line from the generated vertex, air mean loss |
| C (default for validation) | GDML world above the tracker in height bands (building, overburden, CRV, electronics): per-material sliced energy loss and correlated Highland scattering to a plane above the straws (`--sliced-msc`: each band with its own \|p\|); then inside the tracker (`TrackerInterior`): energy loss in the support structure (rings, ribs, beams, manifolds, electronics: coaxial shells from the GDML) and scattering, segment by segment, in that structure and in the straws themselves (Mylar wall, outer Al, gas, W/Au sense wire; gaps are vacuum as in the GDML). The track kinks after every support segment (support segments above 0.25 X0 are split into several kinks, `--kink-sup-xx0`, 0 = off) and every 1e-3 X0 of straw material, and its straw crossings are recomputed |
| single-scattering tail | on by default in stage C (`--no-msc-tail` to switch off): screened Rutherford on nuclei, with an exponential nuclear form factor per material, plus atomic electrons up to m_e/m_mu, added above the angle where it crosses the Gaussian core (`--tail-k K` puts the threshold at K theta0) |
| eIon | Urban fluctuations in the straw gas with the tracker region cut, matched to Geant4 11.3.2 |
| secondaries | `--library LIB.root`: in-straw and off-axis secondary steps pasted from a G4 library (`buildlib`) |

Switches for A/B comparisons: `--no-interior` (no support structure), `--no-interior-msc` (support loss, straight
tracks inside the tracker), `--no-msc-tail`, `--near-margin MM` (which muons get the world transport).

## Package

| path | content |
|---|---|
| `trkfastsim/geometry.py` | straws from the GDML (StrawSD ids, inner and wall radii, sense wire), detector frame, alignment composition |
| `trkfastsim/crossings.py` | vectorised panel-box cull + ray-cylinder straw crossings, point/nearest-straw lookups |
| `trkfastsim/steps.py` | the generator: world transport to the plane, `TrackerInterior` (support loss and in-tracker scattering), StrawGasStep rows, secondary pasting |
| `trkfastsim/steps_jax.py` | opt-in JAX backend of `steps` (`--backend jax`, CPU or GPU): the in-tracker transport, `exit_scatter`, the straight-line crossings and the gas eIon on the device; the world column above the plane, the library pasting and the table assembly stay numpy. Statistically equal to numpy (other random streams), bit-identical per seed on one device |
| `trkfastsim/library.py` | G4 secondary library: build, in-straw and off-axis samplers |
| `trkfastsim/tables.py` | table readers, the production PrimaryFilter tracker term |
| `trkfastsim/analyses/` | one module per CLI subcommand (below) |
| `scripts/stagec_msc_ab.sh`, `scripts/stagec_msc_ab_summary.py` | the stage C scattering A/B on the 20 reference shards (steps + v1compare + worldcheck + msccheck per arm) and its summary |
| `scripts/jax_steps_cpu.sh`, `scripts/uva_steps_gpu.sh`, `scripts/uva_steps_stream.sh`, `scripts/eaf_steps_gpu.sh` | `jaxcheck` on one CPU core (GPVM), on one A100 at UVA (Slurm job fed by the stream script from the laptop) and on an EAF A100 session |
| `art/` | muse satellite area with `StrawGasStepTableDumper` (G4 reference tables), `art/README.md` |
| `trkfastsim/digi.py`, `trkfastsim/digicond.py` | fast digitisation: a vectorised port of `StrawDigisFromStrawGasSteps` and its conditions from the stock job's sources (`docs/fast-digi.md`); `scripts/digi_validate.sh` and `art/run_digi_ref.sh` run its validation |
| `trkfastsim/reco/` | fast reconstruction: `hitreco` (StrawHitReco), `combo` (CombineStrawHits), their JAX twins `hitreco_jax` and `combo_jax`, `linefind` (SimpleTimeCluster, LineFinder); `tracktable`, `ntuple`, `ntschema`, `ntcompare` (EventNtuple output); subcommands `reco-conditions`, `reco-vr1`, `reco-vr1b`, `reco-ntuple`, `reco-ntschema`, `reco-ntcompare` |
| `tests/` | pytest unit tests (synthetic geometry, no data files) |

CLI: `python3 -m trkfastsim <subcommand> -h`.

| subcommand | purpose |
|---|---|
| `refcheck` | content check of the reference tables (census, linkage, PrimaryFilter tracker term) |
| `v0check` | T-V0: straws from the GDML against the G4 step entry/exit points |
| `steps` | the fast generator (above) |
| `buildlib` | G4 secondary library from reference tables |
| `elosscheck` | primary-muon eIon per straw: the Urban sampler on G4 kinematics, and the fast population against G4 |
| `worldcheck` | world model event by event: \|p\| at the first straw vs generation, delivery and loss, fast vs G4 |
| `v1compare` | T-V1: primary-muon StrawGasSteps fast vs G4 (rate per live time, straws per muon, path, eIon with the MPV from a Moyal fit, first-straw state, first-to-last-straw deflection per \|p\|) with the T-V1 thresholds |
| `msccheck` | in-tracker scattering per consecutive step pair (p theta), straw-only vs support pairs, tail shares, fast vs G4 |
| `stepcheck` | diagnose the T-V1 path and eIon checks: crossing-angle population, G4 half-vs-half noise floor, band bins |
| `jaxcheck` | `steps --backend jax` against numpy on the same primaries (straws per muon, eIon MPV and spectrum, time, entry points, occupancy, paired PrimaryFilter tracker term), per-seed reproducibility, cross-device comparison (`--cross`), timing per stage; `--concat --lanes N` for the throughput at a large batch |
| `digi-cond` | fast digitisation conditions (`.npz`) from the stock job's `fhicl-dump`, the DbService CSVs and the GDML |
| `digi` | fast digitisation: StrawGasStep tables to digi tables in the `StrawHitTableDumper` layout the fast hit reco reads |
| `digicmp` | fast against stock digis on the same steps, with a reseeded stock floor: digis per event, occupancy, TDC, TOT, ADC/PMP, dt, hit reco and the F2 fit yield |
| `reco-conditions` | the straw conditions a fast reco job needs, read from the conditions tree of a `StrawHitTableDumper` output |
| `reco-vr1` | T-VR1: fast straw-hit and combo-hit reco against the stock hits dumped from the same digis (`--jax` for the JAX twins) |
| `reco-vr1b` | T-VR1b: fast time clusters and line seeds against stock `SimpleTimeCluster` and `LineFinder` (`DumpTracks` in the dumper, `art/TrkFastSim/fcl/dump_tracks_mc.fcl`) |
| `reco-ntuple` | EventNtuple-compatible `evtinfo`, `trk`, `trksegs`, `trkhits`, `trksegpars_kl` (and `trkmats` when the fit has straw crossings) from one `DumpTracks` file: `--fit stock` (the dump's own KKLine fits, the writer closure), `--fit f2` (the `reco-vr3` chain from digis), `--fit f1` (`reco-vr2f1 --save-tracks` output, stock-conditioned). See "EventNtuple output" below |
| `reco-ntschema` | field names, types and nesting of a stock EventNtuple (JSON); the stored stock schemas are in `trkfastsim/reco/data/` |
| `reco-ntcompare` | `trk`/`trksegs`/`trkhits` summary of an EventNtuple, or with `--compare STOCK.root` the event waterfall, shared-hit track matching and matched-track, matched-hit and reference-crossing differences; `--report` prints the digest of a saved `ntcompare.json` |

## EventNtuple output

`reco-ntuple` (`trkfastsim/reco/ntuple.py`, track-table contract in `tracktable.py`) writes `EventNtuple/ntuple` with
the stock branch names, field order, types, nesting and units (mm, ns, MeV/c, detector frame), driven by a stored
stock schema. Every event of the input is written (also those without tracks); by default only good fits, as stock
(`--all-fits` adds failed fits with `goodfit` 0, `status` -1).

- Layout: uproot cannot write ROOT split-class TTree branches or `std::vector<std::vector<T>>`, so the tree is an
  RNTuple. `tree["evtinfo"].array()`, `tree["evtinfo/run"]`, `tree["trk"].array()` (a record of `trk.<leaf>` lists)
  and `tree["trksegs"|"trkhits"|"trksegpars_kl"|"trkmats"].array()` give the same awkward types as on a stock file.
- Split-branch paths: `tree["trk/trk.<leaf>"].array()` (crvextrap `load_quality` and most trk readers) returns the
  stock jagged array because the `trk` record carries one extra trailing field `trk`, the same per-track lists under
  the bare leaf names. `ntuple.without_alias` drops it; `--no-split-alias` writes the bare stock layout.
- Not supported (uproot 5.7 RNTuple reading): `tree.arrays(["trk", "trksegs", ...])`, as pyutils `Importer` calls it,
  returns empty records for every record field (`tree.arrays(filter_name="trk*")` works); `tree.arrays(["trk.nactive"])`
  nests the result under `trk`; `"trk/trk.status" in tree.keys()` is False. ROOT C++ reading of these files (field
  names containing dots) is not checked.
- Not written: the MC branches (`trkmcsim`, `trksegsmc`, `trkhitsmc`, `evtinfomc`), CRV and DS surfaces in `trksegs`
  (stock has the ToCRV extrapolation, sid 201-277), `trk.nseg` (KinKal piecewise segment count, sentinel).
- Unfilled fields hold -9999.0 (floats), -9999 (integers) or False (booleans). The per-field coverage (filled,
  approx, constant, sentinel), the fit type, the stock conditioning, the package commit and a content hash are in
  `EventNtuple/fastreco_provenance` (JSON string).
- `trksegs`: every intersection of the fitted line with TT_Front, TT_Back (r < 950 mm) and TT_Outer; no CRV surfaces.
  The momentum is the fit momentum at every crossing (no energy loss along the line): stock is lower by about
  0.1 MeV/c at the TT_Outer entry.
- `--fit f1` is stock-conditioned: stock LineFinder seed, stock hit list and stock straw-crossing list, single-seed
  single-fit events only. `--fit f2` runs from digis but has a fixed momentum (2000 MeV/c) and no material.
- `scripts/ntuple_validate.sh` runs the three fits and `reco-ntcompare` against the stock EventNtuple of the same digis.

## Reproducing the validation on a Mu2e GPVM

For Mu2e collaborators with an account on the Mu2e interactive machines (`mu2egpvm01`-`07`). Everything runs in
Python from the `ana` environment on CVMFS; nothing is built. Follow the Mu2e GPVM rules: one job per machine, outputs
in your `/exp/mu2e/data/users/$USER` area (150 GiB quota), never in the app area.

### 1. Code and environment (once)

```bash
cd /exp/mu2e/app/users/$USER
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git
cd mu2e-fastsim
cp site.env.example site.env        # then edit the first block: at least M2FS_CODE and M2FS_DATA
```

In `site.env` set `M2FS_CODE=/exp/mu2e/app/users/$USER/mu2e-fastsim` and `M2FS_DATA=/exp/mu2e/data/users/$USER`; the
ALCF and grid lines can stay as they are for this. Every new shell then needs:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana                            # numpy, pandas, uproot, matplotlib, pytest
cd /exp/mu2e/app/users/$USER/mu2e-fastsim && source site.env   # PYTHONPATH=core:crv:tracker
```

Check the install (about a minute; run each suite on its own, `crv/tests` and `tracker/tests` share a file name):

```bash
python3 -m pytest -q -p no:cacheprovider core/tests
python3 -m pytest -q -p no:cacheprovider tracker/tests
```

### 2. Inputs

The validation needs three inputs. They are Mu2e-internal and are not in this repository; the copies used for the
published comparisons are readable on `/exp` and can be linked instead of regenerated:

| input | what | where (readable copy) | how it is made |
|---|---|---|---|
| GDML world | v04 extracted geometry export | `/exp/mu2e/data/users/ram2aq/crv_fastsim/geometry/mu2e_extracted_v04.gdml` | `crv/geometry/` Offline job |
| gen tables | CRY primaries and live time of every generated event, 20 shards x 50,000 (`gen.001400_*.root`) | `/exp/mu2e/data/users/ram2aq/crv_fastsim/g4ref_v1_tables/` | G4 reference grid run `crv/grid/` (`g4ref_v1`), dumped with `crv/art/dump_g4ref.sh` |
| G4 reference StrawGasSteps | tracker truth of the accepted events of the same run (`sgsteps.*.root`) | `/exp/mu2e/data/users/ram2aq/trk_fastsim/g4ref_v1_tables/` | `art/dump_ref.sh` on the `g4ref_v1` dts files (needs the muse area in `art/`, built on `mu2ebuild02`) |

```bash
mkdir -p $M2FS_DATA/crv_fastsim/geometry $M2FS_DATA/crv_fastsim $M2FS_DATA/trk_fastsim
R=/exp/mu2e/data/users/ram2aq
ln -s $R/crv_fastsim/geometry/mu2e_extracted_v04.gdml $M2FS_DATA/crv_fastsim/geometry/
ln -s $R/crv_fastsim/g4ref_v1_tables $M2FS_DATA/crv_fastsim/g4ref_v1_tables
ln -s $R/trk_fastsim/g4ref_v1_tables $M2FS_DATA/trk_fastsim/g4ref_v1_tables
```

The `steps` and `msccheck` subcommands find the GDML at `$M2FS_DATA/crv_fastsim/geometry/mu2e_extracted_v04.gdml`
(override with `--gdml`). `CRVFS_DATA` (set by `site.env`) is `$M2FS_DATA/crv_fastsim`.

### 3. One shard (smoke test, about 30-40 s)

```bash
mkdir -p /tmp/$USER/trkfs
python3 -m trkfastsim steps $CRVFS_DATA/g4ref_v1_tables/gen.001400_00000011.root \
    --outdir /tmp/$USER/trkfs --seed 1 --stage C --sliced-msc
```

It prints the support-structure census, the in-tracker transport counters (kinks, tail scatters, wire hits) and
`36,777 primary muons, ~360 with straw crossings, ~5,600 steps`.

### 4. The stage C validation (20 shards)

```bash
tracker/scripts/stagec_msc_ab.sh default      # current defaults: in-tracker scattering + single-scattering tail
python3 tracker/scripts/stagec_msc_ab_summary.py $M2FS_DATA/trk_fastsim c1SIsD
```

`steps` takes about 70 s for the 20 shards on an idle GPVM; the three comparisons add a few minutes.
`stagec_msc_ab.sh ARM [ARM ...]` runs `steps` on the 20 gen shards (seed 1, `--stage C --sliced-msc` plus the arm's
flags), then `v1compare`, `worldcheck` and `msccheck` against the G4 tables. Outputs, for arm tag `TAG`:

| output | content |
|---|---|
| `$M2FS_DATA/trk_fastsim/fast_TAG/fasttrkC.*.root` | fast tables (`sgsteps`, `particles`, `gen`, `subruns`), same keys as the G4 tables; logs `steps.log`, `v1.log`, `world.log`, `msc.log` |
| `$M2FS_DATA/trk_fastsim/plots/v1compare_TAG/v1compare.json` | T-V1 comparison and threshold decisions |
| `$M2FS_DATA/trk_fastsim/plots/worldcheck_TAG/` | delivery and energy loss per generated \|p\| and column, per-event CSV |
| `$M2FS_DATA/trk_fastsim/plots/msccheck_TAG/` | step-pair scattering tables, the largest-angle G4 pairs (CSV) |
| `$M2FS_DATA/trk_fastsim/stagec_msc_ab.log` | one line per finished arm (wall time of `steps`) |

The other arms reproduce steps of the scattering study, each with seed 1. An arm reproduces its published numbers
only with the code it was run with: check out that tracker commit, with the core commit pinned beside it in the
umbrella history, and run the script from that checkout.

| arm | tag | configuration | code |
|---|---|---|---|
| `base` | `c1SIs` | `--no-interior-msc --no-msc-tail`: support loss, straight tracks in the tracker | any; the script only rescores an existing `fast_c1SIs`, made with `steps ... --outdir $M2FS_DATA/trk_fastsim/fast_c1SIs --seed 1 --stage C --sliced-msc --no-interior-msc --no-msc-tail` |
| `vac` / `vactail` | `c1SIsV` / `c1SIsVT` | straws and support scatter, no sense wire; without / with the nuclear-only tail | tracker `e6e16ed` |
| `wire` / `wiretail` | `c1SIsX` / `c1SIsXT` | + sense wire; without / with the nuclear + electron tail | tracker `52a7522` to `bff9271` (later code adds the kink fix) |
| `default` | `c1SIsD` | no flags: `wiretail` + the kink fix (a kink no longer cuts the next straw's chord short) | `b1f77f2` and later |
| `rescore_<SUFFIX>` | `c1SIs<SUFFIX>` | reruns only the comparisons on an existing `fast_c1SIs<SUFFIX>` (`rescore_base` for `fast_c1SIs`) | any |

The `msc`, `tail`, `wall` and `walltail` arms in the script are intermediate states of 2026-10-19 (support-only
scattering; air in the gaps) that were never committed on their own; they are kept for the record and cannot be
reproduced from the history.

Expected result of `default` (2026-10-20, seed 1): every T-V1 threshold passes; rate per live time -1.2 % +- 1.9 %,
straws per muon -0.7 %, path mean -0.44 %, eIon mean +0.7 %, eIon MPV +1.6 % +- 1.6 %; first-to-last-straw
deflection medians within 0.1 mrad of G4 from 300 MeV/c to 5 GeV/c.

Single comparisons can also be run by hand, e.g.

```bash
D=$M2FS_DATA/trk_fastsim
python3 -m trkfastsim v1compare "$D/fast_c1SIsD/fasttrkC.*.root" --compare "$D/g4ref_v1_tables/sgsteps.*.root" --outdir $D/plots/v1compare_c1SIsD
python3 -m trkfastsim msccheck  "$D/fast_c1SIsD/fasttrkC.*.root" --compare "$D/g4ref_v1_tables/sgsteps.*.root" --outdir $D/plots/msccheck_c1SIsD
python3 -m trkfastsim stepcheck "$D/fast_c1SIsD/fasttrkC.*.root" --compare "$D/g4ref_v1_tables/sgsteps.*.root" --outdir $D/plots/stepcheck_c1SIsD
```

`stepcheck` explains a path or eIon difference: path per crossing-angle, impact-parameter and |p| bin, the fast mean
path reweighted to the G4 angle population, every T-V1 statistic for G4 against itself (alternating shards) as the
noise floor, and the per-bin ratios behind the band checks.

Quote the globs: the subcommands expand them.

### 5. Tracker surface crossings (surfsteps)

`steps` also writes the tree `dump/surfsteps` (switch off with `--no-surfsteps`): the primary muon's crossings of the
tracker virtual detectors, sids 0-4 (TT_Front, TT_Mid + TT_MidInner, TT_Back, TT_Inner, TT_Outer; surfaces from the v04
GDML, `trkfastsim/surfaces.py`), in the columns of `StrawGasStepTableDumper`'s surfsteps tree, so path A
(`StrawGasStepsFromTable`) fills the SurfaceStepCollection that EventNtuple's `trksegsmc` reads. It is computed from the
muon's own StrawGasSteps (no random numbers: `sgsteps` is bit-identical with and without it). Check against G4:

```bash
python3 -m trkfastsim steps "$CRVFS_DATA/g4ref_v1_tables/gen.001400_*.root" --outdir $D/fast_c1SIsDss --seed 1 --stage C --sliced-msc
python3 -m trkfastsim surfcheck "$D/fast_c1SIsDss/fasttrkC.*.root" --compare "$D/g4ref_v1_tables/sgsteps.*.root" --outdir $D/plots/surfcheck_c1SIsDss
```

Expected (2026-10-21): crossings per selected event within 1 % of G4 on every sid; positions on the surfaces; |p|, angle,
position and time bands 0-3.5 % after the 2-sigma allowance. 52-72 % of G4 surfsteps come from particles other than the
primary muon (delta rays crossing the VDs): the fast sim does not write those.

## Interior slab momentum diagnostic

`slabcheck --model interior --kink-p-mean` tests the arithmetic mean of group entry/exit momentum for the
Highland core and single-scattering tail. Without the flag it retains exit momentum. The optional support
calibration is evaluated at exit momentum in both modes, matching `TrackerInterior.transport`.

Run the two arms against the **same existing G4 slab files**, seed and sample count, with different output
directories. For example, from the umbrella environment:

```bash
python3 -m trkfastsim slabcheck "$M2FS_DATA/trk_fastsim/slab/G4_Al/slab.*.root" \
    --model interior --seed 1 --nmu 100000 --outdir "$M2FS_DATA/trk_fastsim/slab/check_exit_TAG"
python3 -m trkfastsim slabcheck "$M2FS_DATA/trk_fastsim/slab/G4_Al/slab.*.root" \
    --model interior --kink-p-mean --seed 1 --nmu 100000 --outdir "$M2FS_DATA/trk_fastsim/slab/check_mean_TAG"
```

These are diagnostic entry points, not evidence that either comparison has run. Repeat for the Cu/Pb inputs.
`slabcheck_interior.json` includes the effective momentum conventions, calibration, seed and sample count.
The flag is rejected for `--model world` and is not exposed by production `steps`. This helper models **one
unsplit support group**; it does not validate the production multi-kink traversal, straw losses, timing or rate.
Do not promote a production default or refit the calibration solely from a synthetic unit-test pass.

Summarize a candidate against a saved baseline without regenerating it:

```bash
python3 -m trkfastsim.analyses.slabcompare BEFORE/slabcheck_interior.json AFTER/slabcheck_interior.json \
    --references "$M2FS_DATA/trk_fastsim/slab/G4_Al/slab.*.root" \
    "$M2FS_DATA/trk_fastsim/slab/G4_Cu/slab.*.root" "$M2FS_DATA/trk_fastsim/slab/G4_Pb/slab.*.root" \
    --tracker-ref TRACKER_SHA --core-ref CORE_SHA --out AFTER/comparison.json
```

The comparator requires matching saved G4 observables, audits actual primary-muon events from the highest-cycle
particle trees, records input JSON hashes, and refuses to overwrite its output. Per-material summaries use the
common fit domain (median exit momentum at least 60% of entrance momentum). They are unweighted point summaries,
not significance tests. Historical outputs without settings remain explicitly unidentified in that field; matching
G4 observables does not establish matched random seeds or code for the historical fast arm.

## Entry-energy writer diagnostic

[Entry-ledger diagnostics](docs/entry-ledger.md) isolate the air-energy subtraction and vacuum convention on
fixed primary-muon trajectories. Run `python3 -m trkfastsim.analyses.ledgercheck INPUTS --gdml GDML
--outdir NEW_DIR --seed N --tracker-ref SHA --core-ref SHA`. This writes summaries, source/input hashes and
identity-paired counts, not surface/reco-ready ROOT products. The normal generator retains its defaults;
full transport-energy and integrated-flight-time closure remain separate work.

## License

Apache-2.0 (`LICENSE`, `NOTICE`).
