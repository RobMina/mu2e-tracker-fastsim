# mu2e-tracker-fastsim

Geant4-free tracker simulation (`StrawGasStep`s) and fast tracker reconstruction for the Mu2e extracted (KPP, no
field) cosmic configuration: stage 2 of the Mu2e fast simulation, beside
[`mu2e-crv-fastsim`](https://github.com/RobMina/mu2e-crv-fastsim). Physics primitives (energy loss, scattering,
materials, crossings, tables) come from [`mu2e-fastsim-core`](https://github.com/RobMina/mu2e-fastsim-core). All
three are submodules of the umbrella repository [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim); work from
a clone of the umbrella (setup there).

This repository holds technical software only: no physics results, no Mu2e-internal conditions, tables or data.

## Status (2026-10-20)

Fast simulation of the primary muon's `StrawGasStep`s is in place; fast reconstruction is not started
(`docs/plan.md`). The generator `python3 -m trkfastsim steps` reruns the CRY primaries of the G4 reference sample:

| stage | what it does |
|---|---|
| A | straight line from the generated vertex, air mean loss |
| C (default for validation) | GDML world above the tracker in height bands (building, overburden, CRV, electronics): per-material sliced energy loss and correlated Highland scattering to a plane above the straws (`--sliced-msc`: each band with its own \|p\|); then inside the tracker (`TrackerInterior`): energy loss in the support structure (rings, ribs, beams, manifolds, electronics: coaxial shells from the GDML) and scattering, segment by segment, in that structure and in the straws themselves (Mylar wall, outer Al, gas, W/Au sense wire; gaps are vacuum as in the GDML). The track kinks after every support segment and every 1e-3 X0 of straw material, and its straw crossings are recomputed |
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
| `trkfastsim/library.py` | G4 secondary library: build, in-straw and off-axis samplers |
| `trkfastsim/tables.py` | table readers, the production PrimaryFilter tracker term |
| `trkfastsim/analyses/` | one module per CLI subcommand (below) |
| `scripts/stagec_msc_ab.sh`, `scripts/stagec_msc_ab_summary.py` | the stage C scattering A/B on the 20 reference shards (steps + v1compare + worldcheck + msccheck per arm) and its summary |
| `art/` | muse satellite area with `StrawGasStepTableDumper` (G4 reference tables), `art/README.md` |
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
| `v1compare` | T-V1: primary-muon StrawGasSteps fast vs G4 (rate per live time, straws per muon, path, eIon, first-straw state, first-to-last-straw deflection per \|p\|) with the T-V1 thresholds |
| `msccheck` | in-tracker scattering per consecutive step pair (p theta), straw-only vs support pairs, tail shares, fast vs G4 |

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
| `wire` / `wiretail` | `c1SIsX` / `c1SIsXT` | + sense wire; without / with the nuclear + electron tail | tracker `52a7522` and later (`wire` = `--no-msc-tail`) |
| `default` | `c1SIsD` | no flags; identical to `wiretail` | tracker `94e53a8` and later |

The `msc`, `tail`, `wall` and `walltail` arms in the script are intermediate states of 2026-10-19 (support-only
scattering; air in the gaps) that were never committed on their own; they are kept for the record and cannot be
reproduced from the history.

Single comparisons can also be run by hand, e.g.

```bash
D=$M2FS_DATA/trk_fastsim
python3 -m trkfastsim v1compare "$D/fast_c1SIsD/fasttrkC.*.root" --compare "$D/g4ref_v1_tables/sgsteps.*.root" --outdir $D/plots/v1compare_c1SIsD
python3 -m trkfastsim msccheck  "$D/fast_c1SIsD/fasttrkC.*.root" --compare "$D/g4ref_v1_tables/sgsteps.*.root" --outdir $D/plots/msccheck_c1SIsD
```

Quote the globs: the subcommands expand them.

## License

Apache-2.0 (`LICENSE`, `NOTICE`).
