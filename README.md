# mu2e-tracker-fastsim

Geant4-free tracker simulation (`StrawGasStep`s) and fast tracker reconstruction for the Mu2e extracted (KPP, no
field) cosmic configuration: transport of the primary muon through the world and the tracker's support structure and
straws, a Geant4 secondary library, a port of the stock digitisation, and ports of the stock hit reco, time
clustering, line finding and line fit, with an EventNtuple-compatible output. NumPy, with JAX twins for GPUs.

Stage 2 of the umbrella [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim) (submodule `tracker/`), beside
[`mu2e-crv-fastsim`](https://github.com/RobMina/mu2e-crv-fastsim); physics primitives come from
[`mu2e-fastsim-core`](https://github.com/RobMina/mu2e-fastsim-core). Work from a clone of the umbrella. This
repository holds code only: no Mu2e-internal conditions, tables, data or results.

## Status

| Part | Default | State |
|---|---|---|
| geometry | straws from the v04 GDML | T-V0 pass |
| primary-muon steps | stage C (world + tracker interior + single-scattering tail) | T-V1 pass; any-primary-muon rate -0.82 +- 0.26 % (open), +0.14 +- 0.26 % with the opt-in `--scatter-in` |
| secondary steps | opt-in Geant4 library (`--library`) | eIon per straw 12.4 % vs 5 % (open) |
| digitisation | NumPy port of `StrawDigisFromStrawGasSteps`; JAX opt-in | agrees with stock within the reseed floor |
| hits, combos, clusters, seeds | NumPy, JAX twins | identical to stock on the same digis (T-VR1, T-VR1b) |
| line fit | F2 (fixed momentum, statistical); F1 KinKal port opt-in | F1 frozen below its target: acceptance 94.2 % (default), 95.6 % (opt-in time model) vs 99.5 % |
| full chain vs Geant4 + stock | | occupancy pass; reco yield -2.04 +- 0.37 % (T-V3 fail) |

Numbers, samples, thresholds and commits: [`docs/validation.md`](docs/validation.md). What is missing or not
validated: [`docs/limitations.md`](docs/limitations.md).

## Quickstart

```bash
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
cp site.env.example site.env        # fill M2FS_INPUTS (verified input release), M2FS_DATA (outputs); source it
source site.env
python3 -m pytest -q -p no:cacheprovider tracker/tests      # JAX tests skip without JAX
python3 -m trkfastsim steps "$CRVFS_DATA/g4ref_v2_tables/gen.001400_00000022.root" \
    --gdml "$CRVFS_DATA/geometry/mu2e_extracted_v04.gdml" --stage C --sliced-msc --seed 22 --outdir "$M2FS_DATA/tracker-smoke"
```

Digitisation, reconstruction and the validation campaigns: [`docs/running.md`](docs/running.md). The inputs are
Mu2e-internal and not yet published as a collaborator release (umbrella `docs/running.md`).

## Map

| Path | Content |
|---|---|
| `trkfastsim/` | the package and its CLI (`python3 -m trkfastsim <subcommand> -h`); geometry, crossings and tables at the top |
| `trkfastsim/sim/` | step generator (world and tracker-interior transport, NumPy and JAX), secondary library, ordered navigation, ledgers |
| `trkfastsim/digi/` | fast digitisation (`digitize`, its JAX twins), conditions, digi tables |
| `trkfastsim/reco/` | fast reconstruction: hit reco, combos, line finding, F1/F2 fits, EventNtuple writer |
| `trkfastsim/validation/` | one module per comparison subcommand, and the report/audit libraries behind `scripts/` |
| `trkfastsim/kit/` | endpoint-kit tasks (`kit_task`, `kit_uva`, `kitpoint`, `kitf1`); `trkfastsim.kit.kit_task`, `kit_uva` and `worldmods` are aliases for one release |
| `art/` | muse satellite area: StrawGasStep/digi/hit dumpers, table-to-art injectors, payload audit |
| `grid/` | gen-only and T-V2 grid jobs |
| `scripts/` | campaign drivers (stage C A/B, occupancy, digi and ntuple validation, UVA/EAF device runs) |
| `tests/` | pytest suite (synthetic geometry, no data files) |

Docs: [`architecture.md`](docs/architecture.md), [`running.md`](docs/running.md), [`validation.md`](docs/validation.md),
[`limitations.md`](docs/limitations.md), [`fast-digi.md`](docs/fast-digi.md), [`eventntuple.md`](docs/eventntuple.md),
[`secondary-validation.md`](docs/secondary-validation.md), [`kit-task.md`](docs/kit-task.md),
[`device-world.md`](docs/device-world.md); opt-in features below. [`docs/plan.md`](docs/plan.md) is the original plan
(historical).

### Subcommands

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
| `digi` | fast digitisation: StrawGasStep tables to `StrawHitTableDumper`-layout digis; NumPy default, experimental explicit-key `--backend jax` ([status/limits](docs/fast-digi.md#opt-in-jax-backend)) |
| `digicmp` | fast against stock digis on the same steps, with a reseeded stock floor: digis per event, occupancy, TDC, TOT, ADC/PMP, dt, hit reco and the F2 fit yield |
| `seccensus` | T-V1b: secondary steps per primary-muon step and their eIon share, by species, eIon, path, place relative to the muon, wire distance, along-wire position, muon \|p\|, G4 creation process; eIon per straw with a G4 half-vs-half floor; step-level panel occupancy with event-clustered errors; `--paired` on the same CRY primaries |
| `tv2compare` | T-V2/T-V3 on path A EventNtuples (and `StrawHitTableDumper` tables with `--hits`): yield, hits, chi2, TT_Outer residual widths (jackknife errors, entry/exit split), digis, TDC/TOT, panel occupancy over the 216 uniquePanels with Poisson and event-clustered errors and G4 even/odd-file floors; `--paired` splits the yield difference into selection-only and both-selected parts; every row against its threshold |
| `reco-conditions` | the straw conditions a fast reco job needs, read from the conditions tree of a `StrawHitTableDumper` output |
| `reco-vr1` | T-VR1: fast straw-hit and combo-hit reco against the stock hits dumped from the same digis (`--jax` for the JAX twins) |
| `reco-vr1b` | T-VR1b: fast time clusters and line seeds against stock `SimpleTimeCluster` and `LineFinder` (`DumpTracks` in the dumper, `art/TrkFastSim/fcl/dump_tracks_mc.fcl`) |
| `reco-ntuple` | EventNtuple-compatible `evtinfo`, `trk`, `trksegs`, `trkhits`, `trksegpars_kl` (and `trkmats` when the fit has straw crossings) from one `DumpTracks` file: `--fit stock` (the dump's own KKLine fits, the writer closure), `--fit f2` (the `reco-vr3` chain from digis), `--fit f1` (`reco-vr2f1 --save-tracks` output, stock-conditioned). See "EventNtuple output" below |
| `reco-ntschema` | field names, types and nesting of a stock EventNtuple (JSON); the stored stock schemas are in `trkfastsim/reco/data/` |
| `reco-ntcompare` | `trk`/`trksegs`/`trkhits` summary of an EventNtuple, or with `--compare STOCK.root` the event waterfall, shared-hit track matching and matched-track, matched-hit and reference-crossing differences; `--report` prints the digest of a saved `ntcompare.json` |

### Opt-in features (defaults unchanged)

| Feature | Doc |
|---|---|
| JAX step generator (`steps --backend jax`) | [device-world.md](docs/device-world.md) |
| JAX digitiser (`digi --backend jax`) | [fast-digi.md](docs/fast-digi.md#opt-in-jax-backend) |
| ordered stored-material navigation (`steps --navigation ordered-*`) | [ordered-stored-path.md](docs/ordered-stored-path.md) |
| detector-only F1 prototype (`reco.autonomous`) | [autonomous-f1.md](docs/autonomous-f1.md), [autonomous-reference-inputs.md](docs/autonomous-reference-inputs.md) |
| F1 numerical-validity diagnostics | [f1-validity.md](docs/f1-validity.md) |
| entry-ledger, transport-ledger and transport-capture diagnostics | [entry-ledger.md](docs/entry-ledger.md), [transport-ledger.md](docs/transport-ledger.md), [transport-capture.md](docs/transport-capture.md), [rate-ledger.md](docs/rate-ledger.md) |
| relocated saved-product validation | [saved-validation.md](docs/saved-validation.md) |
| kit task `tracker_point` | [kit-task.md](docs/kit-task.md) |

## Tests

`python3 -m pytest -q -p no:cacheprovider tests` from this directory with core, crv and tracker on `PYTHONPATH`
(`crv/tests` and `tracker/tests` share a file name: run them in separate processes). Large-sample checks are
`@pytest.mark.slow` (`--runslow`). CI runs the suite with and without JAX on every pull request.

## Studies kept off main

| Branch | Contents |
|---|---|
| `park/f1-native-closure` | F1 native-closure tooling: crossing/operand capture modules, native arithmetic/CA/material/measurement/reference ports, `f1_*` audits and drivers, the per-step f1-* records |
| `park/ordered-navigation` | ordered-navigation geometry probes and producers (guarded/native/mixed/inclusive envelopes, exact planar, wide torus/convex, segment transport) and the `ordered_*` audit scripts; the `--navigation ordered-*` options stay on main |
| `park/rate-ledger` | rate-ledger history state/surfaces and the ledger job drivers |
| `park/rate-diagnostics` | the earlier rate-deficit diagnostics `geomcheck`, `interiorcensus`, `tubecheck`, `arrivalcheck`, `slowcheck`, `survivalcheck`, `rangecheck`, `secdiag`, `libplacecheck` (untested CLIs, superseded by `ratearms`) |
| `park/tv3-rescoring` | the saved T-V3 secondary-bank replicas, response and summaries |

About 90 unreferenced one-off scripts were removed; they are in the history and on every park branch.

## License

Apache-2.0 (`LICENSE`, `NOTICE`).
