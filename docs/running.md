# Tracker: running

Work from a clone of the umbrella repository [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim); its
[`docs/running.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/docs/running.md) covers the environment,
`site.env` and the input release. This page holds the tracker commands.

## Minimum site configuration

```bash
export M2FS_CODE=/path/to/mu2e-fastsim                 # the umbrella clone
export M2FS_INPUTS=/path/to/verified/input-release     # read-only
export M2FS_DATA=/path/to/writable/outputs
export PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker"
export CRVFS_DATA="$M2FS_INPUTS/crv_fastsim"           # the gen tables and the GDML are shared with the CRV stage
```

Python 3.10+ with numpy, pandas, uproot, scipy (`pip install -e core -e tracker`); pyarrow for the Parquet outputs
(`pip install -e "tracker[parquet]"`); JAX only for `--backend jax`. On a Mu2e GPVM:
`source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && pyenv ana`, then `source site.env` (no JAX there; the JAX
tests skip). One job per GPVM; outputs in your data area, never in the app area.

## One shard: primary-muon StrawGasSteps (default)

```bash
python3 -m trkfastsim steps "$CRVFS_DATA/g4ref_v2_tables/gen.001400_00000022.root" \
    --gdml "$CRVFS_DATA/geometry/mu2e_extracted_v04.gdml" \
    --stage C --sliced-msc --seed 22 --outdir "$M2FS_DATA/tracker-smoke"
```

`fasttrkC.*.root` holds the `sgsteps`, `particles`, `gen`, `subruns` and `surfsteps` trees with the same keys as the
Geant4 tables; the log prints the support-structure census and the in-tracker transport counters. Pass `--gdml`
explicitly (the fallback resolver looks under `$M2FS_DATA`). Add `--library "$M2FS_INPUTS/trk_fastsim/lib/trklib_mdc2025as_20f_v2.root"`
for the Geant4 secondary steps. This is primary transport only; it does not exercise digitisation or reconstruction.

## Digitisation and reconstruction

```bash
python3 -m trkfastsim digi-cond -h        # conditions (.npz) from the stock job's fhicl-dump, DbService CSVs and the GDML
python3 -m trkfastsim digi -h             # StrawGasStep tables -> StrawHitTableDumper-layout digis (NumPy; --backend jax opt-in)
python3 -m trkfastsim reco-ntuple -h      # EventNtuple-compatible output (--fit stock | f2 | f1), docs/eventntuple.md
```

These need matched digitisation and reco conditions ([fast-digi.md](fast-digi.md)); the kit task `tracker_point`
chains steps, digis, hits, seeds and the F2 fit for one shard ([kit-task.md](kit-task.md)).

## Validation campaigns

### Stage C T-V1 validation (20 shards)

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

### Surface crossings (surfsteps)

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

### Secondaries, panel occupancy and paired T-V2/T-V3

The per-panel occupancy and the TDC/TOT shapes are sums over events in which one muon puts several digis into each
panel it crosses: a Poisson chi2 over panels is about the mean cluster size (3-4) for two samples of the same physics.
`tv2compare` therefore reports a full event-clustered panel covariance and its rank-based chi2
(`trkfastsim/analyses/occupancy.py`, tested in `tests/test_occupancy.py`), with marginal pulls,
diagonal/Poisson diagnostic scores and G4-against-G4 floors from disjoint files. Paired occupancy
includes cross-arm event covariance, conditional on the fixed secondary library. Input coverage,
checked event-ID bounds, matched paired width blocks and yield-factorization checks prevent
incomplete outputs or unmatched groups from becoming apparent physics losses or zero errors.
See [secondary validation](secondary-validation.md) for the full saved-default-L paired result,
exact generator-content audit, event-clustered shape limitations and opt-in edge/topology diagnostics.
The saved sample's yield is −2.038 ± 0.366%, mostly selection-only; no placement change is promoted.

Paired campaign on the `g4ref_v2` primaries (10 M generated, 200 subruns; same CRY events in both arms), from the
umbrella environment on a GPVM:

```bash
O=$M2FS_DATA/trk_straw_occ_TAG
tracker/scripts/straw_occ_steps.sh $O g4ref_v2            # fast steps + library on 20 merged gen groups, paired seccensus
tracker/scripts/straw_occ_floors.sh $O                     # G4 even/odd floors of every T-V2/T-V3 row; unpaired census
tracker/scripts/tv2_group_lists.sh g4ref_v2 $O/fast_g4ref_v2_L $O tv2p   # grid lists (10 subruns per job)
DUMP=1 TABLES_DIR=$O/fast_g4ref_v2_L tracker/grid/submit_tv2.sh --arm fast --list $O/list_tv2p_fast.txt --tarball --use-staged --cluster --tag tv2p_fast
DUMP=1 tracker/grid/submit_tv2.sh --arm g4 --list $O/list_tv2p_g4.txt --tarball --use-staged --cluster --tag tv2p_g4
python3 -m trkfastsim tv2compare "$TRKFS_PNFS/tv2p_fast/nts.*.root" --compare "$TRKFS_PNFS/tv2p_g4/nts.*.root" \
  --steps "$O/fast_g4ref_v2_L/fasttrkCL.*.root" --steps-ref "$M2FS_DATA/trk_fastsim/g4ref_v2_tables/sgsteps.*.root" \
  --hits "$TRKFS_PNFS/tv2p_fast/hits/*.root" --hits-ref "$TRKFS_PNFS/tv2p_g4/hits/*.root" --paired --outdir $O/tv2p
```

`grid/grid_tv2.sh` takes comma-separated inputs on one list line (one art job over several files, one fast table holding
all their events). Expanding the pnfs globs above lists dCache: copy the outputs to the data area first, or pass the
known file names.

### Interior slab momentum diagnostic

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
