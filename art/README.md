# art helpers (muse satellite area)

`TrkFastSim/` is a muse package (a `.muse` file plus `src/SConscript`) built in its own work area, backed by the CVMFS
Musing `SimJob MDC2025ay`, so no Offline checkout is touched. Same layout as `crv/art/CrvFastSim` of the umbrella.

| module | role |
|---|---|
| `StrawGasStepTableDumper` | the tracker truth of a dts file -> flat TTrees in `dump/`: `sgsteps`, `surfsteps`, `vdsteps`, `trajpts`, `particles`, `gen`, `subruns`. uproot cannot read these art products directly (memberwise `SimParticle`/`GenParticle`, the `MCTrajectory` map) |
| `StrawDigiTableDumper` | lossless detector-only `trkdigi/events`; optional `compareTo` checks every original/replayed digi field and ADC sample in the same art event |
| `StrawDigisFromTable` | ordered StrawDigi and aligned ADC waveform collections from a lossless table, matched by full event identity; no redigitisation or invented truth |

## Trees

All trees carry `run`, `subrun`, `event`. `sp` is the `SimParticle` key (`particles.key`).

| Tree | Product (default tag) | Columns | Frame |
|---|---|---|---|
| `sgsteps` | `StrawGasStep` (`compressDetStepMCs`) | `nsgs` (steps in the event), `sid` (`StrawId::asUint16`), `plane`, `panel`, `straw`, `upanel`, `shape` (line/arc/curl/point), `ion` (minion/highion/neutral), `sp`, `eion` (MeV), `len` (path in gas, mm), `width` (mm), `t` (ns, gas entry), `x0..z1` (entry/exit), `px..pz` (mean momentum, MeV/c) | Mu2e |
| `surfsteps` | `SurfaceStep` (`compressDetStepMCs`) | `surf` (SurfaceId enum, as EventNtuple `sid`), `sindex`, `sp`, `edep`, `t`, `x0..z1`, `px..pz` | detector |
| `vdsteps` | `StepPointMC` (`compressDetStepMCs:virtualdetector`) | `vdid` (VirtualDetectorId), `sp`, `x..z`, `px..pz`, `t` | Mu2e |
| `trajpts` | `MCTrajectory` (`compressDetStepMCs`) | `sp`, `ipt` (point index), `x..z`, `t`, `ke` (kinetic energy, MeV) | Mu2e |
| `particles` | `SimParticle` (`compressDetStepMCs`), `PrimaryParticle` (`FindMCPrimary`) | the `CrvStepTableDumper` columns (`key`, `parent`, `pdg`, `ccode`, `scode`, `isprim`, `genidx`, start/end position, 4-momentum, time) + `ismcprim` | Mu2e |
| `gen` | `GenParticle` (`cosmicTimeOffset`) | `idx`, `pdg`, `genid`, `x..z`, `px..pz`, `e`, `t` | Mu2e |
| `subruns` | `CosmicLivetime` (`generate`) | `livetime` (s), `primaries` | |

Missing optional products (`SurfaceStep`, VD steps, `MCTrajectory`, `PrimaryParticle`) leave their tree empty rather
than failing; `StrawGasStep`, `SimParticle` and `GenParticle` are required.

## Build and run

```bash
source site.env                                        # umbrella root: sets TRKFS_CODE=$M2FS_CODE/tracker
$TRKFS_CODE/art/setup_area.sh                          # backing link only
# user, on mu2ebuild02:
cd $TRKFS_CODE/art && source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && setup muse && muse setup && muse build -j 32
# run (GPVM):
mu2e -c TrkFastSim/fcl/dump_sgsteps.fcl -s dts....art -T sgsteps.root
$TRKFS_CODE/art/dump_ref.sh files.txt $M2FS_DATA/trk_fastsim/tables
```

The grid campaign of CRY primaries without Geant4 (gen tables for fast runs of a chosen live time) is in `../grid/`
(`grid/README.md`); its dump step uses `dump_sgsteps_gen.fcl` from this area.

## StrawDigi table bridge (opt-in)

`StrawDigiTableDumper` writes one `trkdigi/events` row per input event, including empty events. All columns are
unsigned 32-bit: scalar `run, subrun, event`, jagged `d_sid, d_tdc0, d_tdc1, d_tot0, d_tot1, d_pmp, d_flags,
d_nadc, d_adc`. `d_sid` is the packed `StrawId`, not a contiguous index; end 0 is cal, end 1 is HV.
`d_nadc` is one waveform length per digi; `d_adc` concatenates the waveforms in digi order. Repeated straws,
zero-length waveforms, every flag bit and full uint32 TDC values are preserved. A missing ADC collection is an
error, not an invented waveform. The importer accepts ROOT `vector<int>` / `vector<unsigned int>` branches and
uproot int32/uint32 counted arrays; it rejects other types. Input must be a TTree, not an RNTuple.

`StrawDigisFromTable` takes `tableFile` and optional `directory` (`trkdigi`). It produces `StrawDigiCollection` and
`StrawDigiADCWaveformCollection` with no instance label. Missing/duplicate event identities, invalid straw IDs,
negative/out-of-range integers, unequal digi-field lengths and per-event ADC count mismatches throw. A missing event
is distinct from a present zero-digi event and never silently becomes empty. The table is indexed once, data are
read for the requested full identity, and digi/ADC ordering is unchanged. Unrequested table events are not rejected:
a full-population audit must still check finished entry counts. Other event products come from RootInput.

### Build gate

Use an isolated tracker worktree and its `art/` satellite, never copy modules into a shared Offline checkout.
Prepare `art/backing` to a recorded current Offline CI build (for example, the build selected at preparation was
`/cvmfs/mu2e-development.opensciencegrid.org/museCIBuild/main/807d948f_94a4d81f_2b90c646`). Do not run the older
`setup_area.sh` for this current-head validation: it selects a historical release. On **mu2ebuild02**, the user runs:

```bash
cd /path/to/isolated/tracker/art
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
muse setup
muse build -j 32
```

`src/SConscript` explicitly links ROOT `TreePlayer` for `TTreeReader`/`TTreeReaderArray`, in addition to the
existing RecoDataProducts, art, TFileService, fhiclcpp/types, canvas, cetlib_except and ROOT base libraries.
There is no CMakeLists file in this satellite.
Wait for successful build completion, verify plugin timestamps/linkage and the resolved backing/envset before art.
The new C++ readers, vector/count-array interoperability and FCLs are not certified merely by Python tests.

### Round trip and reconstruction checks

On a compute node with that already-built satellite (the `container/offline/offline_uva.sh` runner of the umbrella
can mount CVMFS on Slurm), use a **new output directory** and explicit staged inputs. Expand each FCL first, inspect
it, then run without `--debug-config` (expansion alone exits without processing events). Supply input tag overrides
if the original products are not under `makeSD`. Do not redigitise or regenerate the baseline.

```bash
# In the configured runtime; INPUT.art is the exact unfiltered digi population.
mu2e -c TrkFastSim/fcl/dump_digis.fcl -s INPUT.art -T NEW_DIR/digis.root
# Local top-level replay.fcl includes replay_digis.fcl and sets tableFile to NEW_DIR/digis.root.
mu2e -c replay.fcl -s INPUT.art -T NEW_DIR/replayed.root
python3 -m trkfastsim.digitable compare NEW_DIR/digis.root NEW_DIR/replayed.root --report NEW_DIR/digi_identity.json
```

`replay_digis.fcl` also enables the C++ dumper's `compareTo: makeSD`: it throws on any original/replayed field or
ADC-sample mismatch, with event/digi location. Run first on a small explicit population, then on **all** original MC
and KPP events. With Offline CI `807d948f_94a4d81f_2b90c646` / p109, full extracted-v04 validation passes for
2,866 MC events (63,384 digis, 950,760 ADC samples, six empty-digi events) and 500 KPP events
(332,098 digis, 4,981,470 ADC samples). Every digi field, flags and waveform sample round-trips exactly, and all
exported stock-reco/conditions branches match at zero tolerance. This certifies those populations/configuration,
not arbitrary files or all persistent reco members; the complete-payload check below remains separate.

For stock reconstruction, `reco_digis_table_mc.fcl` / `reco_digis_table_data.fcl` change only the digi inputs of
PBTFSD/makeSH and the dumper; the unfiltered SaveAllFits reference paths and calibration stay unchanged. Supply
`tableFile` in a local wrapper. Compare original-vs-replayed `trkdump/events` **and conditions**, all highest-cycle
entries and every branch, at zero tolerance (the umbrella `container/offline/compare_root.py` is available).
That checks exported values, not every persistent KalSeed member. `TrackerRecoPayloadDumper` supplies the stricter
check: append `reco_payload.fcl` to both configurations, preserving identical process/module labels, and compare
every `payload/events` byte array at zero tolerance. It uses ROOT dictionaries to stream the complete persistent
products (PBT, straw/single/panel/stereo/flagged hits, intensity products, clusters, line seeds, KalSeeds and
KalLineAssns, plus saved background clusters/hits), including full covariances, intersections and associations.
Missing active products/dictionaries fail; no members or floating-point bits are omitted. The no-field reference
configures `flagPH` as `NullProducer`: explicit `flag_ph_active=0` / `backgrounds_active=0` markers require its four
nonexistent payloads to remain empty, rather than pretending they were compared. With `FlagBkgHits`, its configured
payloads are required; other module types are rejected. The FCL takes that type and the PBT consumed by makeSH from
the expanded producer configuration; both PBTFSD and the consumed PBT are streamed. Every active product, including
an empty collection, must have nonempty serialized bytes, and payload event identities must match the reco dump.
Transient KKTrk pointers, ROOT file metadata and event provenance are not persistent reco payloads and are not
compared. A byte difference is a failure to investigate, not a tolerance to loosen. The corrected audit plugin
builds and runs on both ten-event MC/KPP smoke populations, but **literal full persistent equality fails** only for
`CosmicTrackSeedCollection`: nine seeded MC events / 610 bytes, one seeded KPP event / 76 bytes. All other audited
payloads, including full KalSeeds and associations, are byte-identical on these smoke populations only. A separate
stock-only rerun on the same original MC digis also changes seed bytes in all nine seeded events (675 bytes), with
every other audited payload exact: the discrepancy is reproducible without the table producer. Source inspection
finds unset legacy `TrackParams.delta*`, `CosmicTrack.niters` and private kinematic members; dictionary-based
inspection points to those fields but reports art::RefCore decoding warnings, so it is not a certified full decoder.
No fields are zeroed/omitted and no stock reco initialization is changed. Full-population persistent certification
is blocked; it requires a separately reviewed stock initialization fix or an explicit gate decision. Reduced-dump
and full detector replay results above remain valid, not a substitute for the failed literal gate.
Retain/reuse an existing baseline only when backing, conditions and config match; otherwise explicitly label the
new current-build reference. KPP requires the matching `reco_data_vst.fcl` include tree and extracted v04 geometry.
It is the uncalibrated reference, not daqana3.

Stock PBTFSD, StrawHitReco, time clustering, LineFinder and KKLine on these paths use detector products and
conditions, not StrawDigiMC truth. The bridge therefore does not manufacture StrawDigiMC/SimParticles from
`d_step0/1`: those are only partial step-row references. Original truth remains in the input event but is **not**
truth-matched to replacement fast digis. MC diagnostics needing that association must be disabled, or supplied a
separately validated truth bridge. No truth-matching or MC-efficiency claim follows from this adapter.

### Fast-digi tables

The existing fast writer's `trkdump/events` format has no `d_flags` branch. Conversion requires an explicit
zero-flag assumption (the fast digitiser makes no digi flags), records that assumption in a mandatory receipt,
and never calls it a lossless original-art round trip:

```bash
python3 -m trkfastsim.digitable convert FAST.root CANONICAL.root --tree trkdump/events \
  --assume-zero-flags --report CONVERSION.json
```

The canonical file includes every fast input event, including zeros. Negative/overflow values and malformed
ADC segments are rejected rather than narrowed. Output/report overwrite is refused. `digitable compare` allows
event-row reorder only; the full unique event population and ordered digi/ADC content must match exactly.
The persisted canonical file is re-read and checked during conversion; the receipt hashes inputs/outputs.

Only after the two art round trips pass may stock reconstruction on injected **fast** digis be compared with
`trkfastsim.reco.autonomous` on the **injected stock dump** (`trkdump/events` contains those same digis plus the
PBT actually used by stock reco). The minimal canonical table intentionally has no PBT/conditions and is not a
drop-in input to the existing fast-reco reader. Supply `--conditions`, `--drift-source` and data/MC settings from
the same stock job. Use `reco-ntuple --fit stock` on that injected stock dump and
`python3 -m trkfastsim.reco.autonomous_compare FAST_NTUPLE.root STOCK_NTUPLE.root --same-digis --out NEW.json`.
The same-digis option is a caller assertion, not an automatic proof. Keep every event/multiple candidate and
report paired yields and conditional residual uncertainties; do not intersect mismatched populations.

On the validated 2,866-event fast-digi shard, injected stock accepts 2,584 tracks versus the unchanged autonomous
prototype's 2,591: paired **+0.271 ± 0.457%**, with **73 gained / 66 lost** accepted-track events. All 2,591 accepted
autonomous final states pass existing numerical checks. Identical input digis were verified field/sample by field/sample
before shared-hit evaluation. Conditional matched-track widths are not true residuals or a native-closure certificate;
the prototype's candidate/continuation and writer limitations remain. This is not the old independent-digitisation
comparison or a generated-event efficiency. No reconstruction code, acceptance cut or default changes.

### Slurm reproducibility helpers

`scripts/prepare_digitable_runtime.py UMBRELLA/container/offline/offline_uva.sh NEW_RUNTIME_DIR` creates private
runner adapters that additionally mount the CI-backing CVMFS repository and invoke the sequence scripts after setup.
It refuses unsupported runner versions and existing output files, and does not change the original runner or build
anything. Put the matching `reco_data_vst.fcl` include in `NEW_RUNTIME_DIR/fcl/`, and the two `art/run_*reco.sh` /
`art/run_digitable_sequence.sh` scripts in the already-built satellite snapshot. Reuse the saved EL8 cvmfsexec client;
never share its mutable node-local cache. Supply the runtime adapters under `ROOT/runtime/`, satellite under
`ROOT/built_area/`, and Python snapshots under `ROOT/code/{tracker,core}`. Data/includes/ANN weights must match the
pinned backing and inputs, with SHA256 manifests; a directory name is not provenance.

- `uva_digitable_art.sh`: ROOT, MODE (`mc`/`kpp`), TAG (new), INPUT_ROOT, OFFLINE_EVENTS; four sequential jobs and
  a `sequence.done` marker. Its named fixture layout contains the original MC digi shard and KPP full-map input;
  it is a validation wrapper, not data discovery. `PAYLOAD=1` additionally requires the newly built audit plugin.
  Optional `AREA` and `RUNNER` override the default `ROOT/built_area` and `ROOT/runtime/offline_sequence.sh`, so a new
  verified snapshot can be audited without overwriting an earlier one. Both must contain the matching payload support.
- `digitable_report.py DIRECTORY --expected-events N --out NEW.json [--payload]`: checks completed sequences,
  highest-cycle full event counts, canonical ordered detector fields and all exported reco/conditions branches.
  `--payload` requires every documented persistent product branch and byte equality; missing products fail closed.
- `digitable_payload_diff.py REF.root CAND.root --out NEW.json`: vectorized full-population byte localization,
  with event IDs and offsets, never normalization or an equality waiver. Its JSON helper diagnostics do not change
  the strict audit result. `TrkFastSim/macros/payload_diagnostic.C` additionally inspects decoded seed scalars using
  compiled ROOT dictionaries (interpreter only, no ACLiC); art::RefCore warnings limit that decoder's scope.
- `run_payload_control.sh` and generated `offline_payload_control.sh`: explicit ten-event MC stock-only
  reproducibility control on a saved original-reco FCL, not a replacement production baseline.
- `uva_fastdigis_art.sh`: ROOT, TAG, INPUT_ROOT, CVMFSEXEC_DIST; reads `ROOT/fast_input/digis.root` and the original
  MC art event context, writes stock reco plus the lossless injected digi dump. No stale original MC association is used.
- `uva_digitable_autonomous.sh`: ROOT, DUMP, CANONICAL, OUT (new), ANN, SIF; proves injected detector equality first,
  then extracts the stock job's conditions, runs the unchanged autonomous path and evaluates all events with
  `--same-digis`. ANN is the staged Offline root with matching `Offline/Mu2eKinKal/data` files, not a private environment.

Submit each with an explicit account/partition and bounded resources, for example `sbatch -A ACCOUNT -p standard
--cpus-per-task=1 --mem=8G --time=00:30:00 --export=ALL,... scripts/uva_digitable_art.sh`. Actual validation used one
core and ~2.4 GiB aggregate Slurm RSS for the art sequence; the autonomous comparison used ~5.1 GiB. These are sample
measurements, not arbitrary-input memory bounds. Select tests with the shared `testselect` on the exact snapshot;
the project Python image has no git, so pass the selected file list and run isolated pytest processes inside it.
