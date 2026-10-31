# Passive crossing capture — full selected-target capture gates pass

Status 2026-10-28: **user-built area verified; 100-event smoke and full 3,000-event capture-off/on pair pass**. This
implements the proposal in [f1-crossing-observability.md](f1-crossing-observability.md) for the frozen p109
experiment. Existing fitter, thresholds, defaults, immutable receipts and both previous native areas remain unchanged.
This is not full F1 closure.

## Full-capture result after space release

The user authorized resumption after verified historical-file archival and release. The unchanged prepared driver ran
serially on GPVM03 into new data directory `trk_f1native_r2_20261027/crossing_3k_20261028_v2/`; both native arms
exited0. Source/repo/binary/helper pins, local linkage, expanded configurations and the 4GiB output +4GiB reserve
policy passed without any relaxed gate. No baseline was regenerated.

Standard `stock_iteration_check.py --expected-events 3000` (no subset flag) and
`f1_crossing_full_check.py` both exited0 on GPVM02. Finished highest-cycle counts:
**3,000events,2,319seeds,147/147targets,3,449iterations**, no missing targets.
All3,000event rows/112fields match control, predecessor and frozen baseline exactly.
Exact predecessor equality also covers2,319seed rows/33fields,3,449iterations/44fields,
130,273pieces/16fields,168,020effects/43fields and61,622hit operands/37fields.
Off-mode sidecars are empty with complete schemas; before/end watched hashes agree.

New capture census:213,617records,5,090objects,4,093trajectories/72,826pieces,
38,174discovery rows,115,085searches, **7exhausted searches retained**, all102,949
prefilter crossings/noexclusions. Captured constructor quotient and constant stored
offsets pass; these are not independent CA/offset generation. Geometry is complete
for all147targets:5,292planes,31,752panels,3,048,192straw rows. Hit consistency includes
all61,622hits (7,235inactive),163,161active residual rows, no exclusions or variance
arithmetic failures; existing tolerances unchanged.

Measured control/capture files:17,293,326 /817,646,554bytes. Native wall times 67.86/78.70s, maximum
RSS589,152/928,892KiB; standard checker8.81s/581,964KiB; full checker33.01s/**2,314,444KiB**. These are observed run
resources, not runtime projections or a speed benchmark. Validation moved to idle GPVM02 because GPVM03 had only
about1.35GiB available RAM after native completion. No physics/data change was involved.

Receipts under the new output directory:
- `check.json`: SHA256 `3fce8fae281f064f6264891b3f0b2c6f9cc4b18f3bd717d2f8e5803ab946353c`.
- `crossing_full_check_v1.json`: SHA256 `c8e1225b1cc71ad82ff7055c77cd382c1b564956a1470522136d48575d53f8ec`.
- `manifest.txt`: SHA256 `161282af06390c78e6ba8e61d9d780dd39b88b93bcaec64c6f32cedb128da5c2`.
- `completion_receipts_v1.sha256`: SHA256 `89f3289c2b09a6cab046453e3035a2e8d0787514938b395a786ec43f28e031bf`.
- `summary_v1.json`, `completion_v1.txt` and checker time logs retain the census/resources.

Shared data free was approximately26.509GiB at16:24EDT, subject to other writers. The older GPVM01 archive marker
remains pending that host's recovery/state check; it does not indicate a running capture.

**Full selected-target capture validity and passivity pass; full numerical/physics/F1
closure remains false.** The subsequent [associated-constructor audit](f1-crossing-constructor.md)
now reconstructs all81 associated-extension constructors exactly from tube geometry,
supplied trajectories and independently reconstructed upstream hit hints. Initial and
material-only construction, partner/retained history and physical discovery remain
open before native-seed/numerical/decision/full-population closure.

## Historical full-capture preparation: blocked on disk space before native launch

After the user's `Continue`, new `scripts/run_f1_crossing_full_pair.sh` and
`scripts/f1_crossing_full_check.py` were prepared without modifying any receipt-bound
smoke helper, native source, template, preparer, previous area or baseline.
The full checker requires literal full-gate flags, fixed3,000events/2,319seeds/
147targets/3,449iterations, highest-cycle event/seed coverage, complete selected
status/geometry/prefilter populations and exact original/hit fields against the
saved `operand_3k_20261028_v1`. It pins historical smoke/full-hit/full-standard
receipts and the executing helper paths/hashes, checks current launch provenance,
rechecks hashes on failure/success and writes exclusive strict JSON. Per-target
geometry counts must be36planes/216panels/20,736straws, not merely self-consistent
empty geometry. Records/effects are released before loading full geometry; peak
RSS has not been measured. Raw PNFS input is bound by the frozen list/path, not a
new content hash or a namespace metadata query.

The new driver is fixed3,000events, oneCPU and the same frozen input/config. It
requires unchanged plugin/all seven KinKal binaries against authenticated smoke/
preflight evidence, local linkage, source hashes/pins and exact expanded config
per arm except the unique debug-output filename. It refuses existing outputs.
Review fixed an unpinned smoke-manifest dependency and separate predecessor `stat`
failures; the new checker binds required manifest hashes and rejects incomplete
geometry. Focused suite **855passed** (224newfull-checker/driver tests plus631smoke
checker tests), log `local_data/trk_f1native_r2_20261027/crossing_full_tests_v1.txt`.
No new full/slow-suite certificate or full-data checker result exists yet.

Space policy: the observed15,152,564byte smoke capture for2targets, scaled only as a **space proxy** to147targets and
augmented by the saved full pair's134,521,951bytes, gives1,248,235,405bytes. The driver budgets **4GiB output plus4GiB
reserve**, checks before output creation and each arm, and makes no runtime extrapolation. Fresh free space
initially8.82GiB fell to **6.51GiB** during preparation. Actual state checked: no full output directory and no
mu2e/python3 processes. Nothing was deleted and the safety budget was not lowered.

Preserved remote data evidence:
- `crossing_3k_20261028_v1.driver.txt`, SHA
  `0043e9b5a80935128c2e4a2abd1642cdac6fffc43d8b9031f97548d23c484d3e`.
- Full driver SHA `2314e9aecbfa8fbc3626241a9bc31fc35d4c4ec57e72f8a0e6fde985d384568f`;
  full checker SHA `6b8f93b945831d7fe040e8b805a0b2311bcb720f70ae5025b660038d84c39e90`.

Remote diffcheck and fitterhash pass. Reuse existing baselines. Then standard full gate
and new full checker; only afterwards independent crossing physics.

## Executed post-build checks and smoke

The user's build/install confirmation is accepted. A nested SSH request to read the
build pane failed authentication; the user explicitly reaffirmed completion and
asked to proceed. The working gpvm01 route required no credential intervention.
No build-marker re-confirmation was required. A build verifier (an earlier campaign script, not kept in the
repository) checked
all50source hashes, five repository pins, plugin/Fit-library timestamps newer than
all manifest sources, local runtime KinKal/Mu2eKinKal linkage, and exact `.text`
section equality for all seven installed versus locally built KinKal libraries.
SONAME-patched installed ELFs are not claimed whole-file-identical to build ELFs.

Data prefix: `$M2FS_DATA/trk_f1native_r2_20261027/`.
- `crossing_build_preflight_20261028_v1/verification.txt`, SHA256
  `c79577c3aeea7e18216091e431f60bd4f19ac11522ec3a06e3f8b871cecc6da0`;
  terminal `CROSSING_BUILD_VERIFIED` and section dumps retained.
- `crossing_smoke_20261028_v1/check.json`, SHA256
  `2f35c694ae1fc31cc019535cbb77e25e8e5f65ab82a5a67b7f21cec22437275a`:
  standard frozen-baseline passivity/structure smoke gate passes.
- `crossing_smoke_20261028_v1/crossing_check_v1.json`, SHA256
  `9b9983124f8df4f48b253158f414f7aa9712d03d1eb0b22e250ea47810f12eb1`:
  independent crossing/geometry/status gate passes, all watched hashes unchanged.

Both native arms processed100events and exited0. Highest-cycle census:77seeds,
2/147selected targets (`1400:0:3700:0`, `1400:0:7033:0`),49iterations,3,481effects.
Exact NaN-aware predecessor equality covers all33seed fields,44iteration fields,
16piece fields over2,379pieces,43effect fields, and all37hit-sidecar fields over942rows.
The saved `operand_smoke_20261028_v2` is reused, never rerun. All100event identities
and event payloads match control and predecessor, including seedless events.
Expanded configs match the corresponding frozen predecessor arm exactly after
replacing only its unique debug-output filename; launch input/config/events are bound.

New-sidecar census:5,169records,166constructor objects,58trajectories/1,307pieces,
1,076discovery rows,2,895searches, **all2,490prefilter crossings with zero exclusions**.
Private offsets equal the captured constructor radius/speed quotient exactly and
stay constant. Current-partner lifecycle, search-vector/cap-pointer/reference and
usable-payload checks pass; final-collection partner stamps are not substituted.
Both targets have complete attempt/completion status and fresh geometry bindings:
2trackers,72planes,432panels,41,472straw rows; float halfLength preserved.
All new off-mode sidecars are empty with complete schemas. No unusable CA or
exhausted search occurs in this smoke; those paths are not native-certified.
Hit W/b consistency also passes the unchanged criterion (942hits,128inactive,
2,442active residual rows, no exclusions/nonfinite arithmetic failures).

Measured outputs:control2,714,374bytes;capture15,152,564bytes. Individual wall times 15.55s/5.88s and maximum
RSS591,468/614,720KiB (off/on) are measurements, not a speed benchmark or a basis for linear runtime extrapolation.

New checker implementation: `f1_crossing_check.py`, `f1_crossing_geometry_check.py`,
`f1_crossing_record_check.py`; **631focused synthetic tests pass** (31integration,
468geometry,132record), log `local_data/trk_f1native_r2_20261027/crossing_checker_tests_v1.txt`.
No rerun of the full/slow or previously selected suite is claimed. Review added
frozen launch/expanded-config checks, full highest-cycle event comparisons and
canonical output containment in the new 100-event-only driver (an earlier campaign driver, not kept in the
repository), which was separate because the preserved operand driver fixes the operand area.

Captured quotient and payload consistency are **not** independent generation, full-history or full-fit closure.

The source-preparation and user build handoff below are historical provenance,
not a new build request. The immutable preparation receipt keeps its original
no-build/no-physics flags because those describe its creation time.

## Area and reproducibility

New cluster-only area:
`$TRKFS_MUSE_AREAS/fastreco_crossings_p109_20261028`

Validated, read-only input:
`$TRKFS_MUSE_AREAS/fastreco_operands_p109_20261028`

The older `fastreco_capture_p109_20261026` also remains untouched. This is isolated
instrumentation of the preserved p109 evidence, not a reconstruction arm on current
Offline head. Five detached worktrees use the exact operand-area pins:

| Repository | Commit |
|---|---|
| Offline | `9b3d463371f34731f8515d1a63914ccf070536a8` |
| KinKal | `bb23c26348fc6e5fcbeefb5c7138919ef21f7abc` |
| Production | `4479f3fad37a85ecd9485394198658237961dcc3` |
| mu2e-trig-config | `d3ec315c7e8a626e8cf9c417ab6522a8e9a8f669` |
| EventNtuple | `76ecb8af80c60e98d80a62a29f0082f36c20add5` |

`scripts/prepare_f1_crossing_capture.py` checks exact source hashes, all repo pins,
the expected inherited KinKal dirty files and no other repo edits. It hashes every
TrkFastSim package file and the new templates, rejects package symlinks, checks
unique exact source anchors before creating anything, refuses an existing area and
uses exclusive JSON reports outside all source/destination trees. Failed or partial
areas are preserved for inspection. It contains no environment/build/install command.
The templates live under `art/crossing_capture/`; no frozen helper was edited.

Preparation evidence under remote data `trk_f1native_r2_20261027/`:

- `crossing_prepare_check_v1.json`, `crossing_prepare_check_v2.json`: pure source
  validation; no area/build/physics creation. Both retained.
- **`crossing_preparation_v1.json`**, SHA256
  **`9b25e7f783b0949f8a7e1c5c27548a0928a59eb45f1d1090d63a6fe082967c84`**:
  completed, all before/end input/template hashes unchanged, build/physics/passivity
  flags false. This is the authoritative preparation report in data, not a copied
  or self-hashed receipt inside the app area.
- Area `capture-source-manifest.sha256`, SHA256
  **`ce3762ab9a5610e5838a5939ccff47c67d362812a41ddb1428eca13cc269cd5d`**,
  verifies all 50 listed source/package/header files. `source-pins.txt` records repo
  pins. No `build/` or `kinkal_build_p109/` existed at handoff.

Changes to inherited native files are limited to:

- `KinKal/Fit/Track.hh`: optional observer/sink in existing IterationCapture scope,
  value trajectory copies and wrapper/prefilter/reference capture points.
- `KinKal/Trajectory/PiecewiseClosestApproach.hh`: optional final trace argument,
  actual hint/solve/pointer history; native branch/order/arithmetic retained.
- `Offline/Mu2eKinKal/inc/KKFit.hh`: constructor search binding, pre-extension
  trajectory copies and ordered material discovery observation points.
- `Offline/Mu2eKinKal/inc/KKStrawXing.hh`: private offset/operand capture,
  constructor/reference/partner-state records.
- `TrkFastSim/src/KinematicLineFitTrace_module.cc`: selected-target sidecar wiring.

New headers are `PiecewiseCATrace.hh`, `CrossingTrace.hh`, `CrossingCapture.hh`,
`CrossingCaptureWriter.hh`, `CrossingCaptureStatus.hh`, `CrossingGeometryWriter.hh`.
All are header-only using existing dependencies; no CMake/SConscript library edit.
The inherited hit sidecar and original ROOT schemas remain unchanged.

## Capture boundaries and source safety

### Scope and ownership

The existing thread-local `Track::IterationCapture` pointer reaches the optional
crossing sink and prefilter callback. No new thread-local/process-global observer
or mutable Track instance member is added. Null-sink scopes disable the observers;
existing nested scope restoration remains in force. The module-owned sink holds
value copies, object IDs and a runtime-only address-to-ID map. No pointer/address
is serialized. A duplicate live object address or missing constructor is an error,
not an implicit identity reuse.

Capture calls catch diagnostic exceptions into the sink without changing fit
status. ROOT initialization and serialization failures are isolated from physics.
Writers require complete successful initialization before accepting writes. The
independent status/binding trees must be checked: **the previous capture gate alone
cannot detect missing/failed new diagnostics**, by design. Missing completion is
failure even if an attempted failure-status Fill also fails. Serialization is not
transactional; partial trees never constitute success.

The radius and speed are evaluated once in the native constructor expression, then
passed to `captureOffset(radius,speed)`, which records the actual values and returns
`radius/speed`. This preserves the quotient and native evaluated operands, but it
is a source restructuring: compiler/runtime passivity still requires off/on and
predecessor tests. The stored private `toff_` is copied directly, never reconstructed
as `time()-particleToca()`.

### Constructor/search/reference records

All selected initial and extension crossings get constructor records before the
Material wrapper updates them. Explicit-CA constructors are linked to the actual
PCA trace by source-order append, not by final collections. Each record stores
contemporaneous partner identity, exact reference q/C/range/bnom/mass/charge,
SensorLine start/end/direction/length/time anchor, stored offset and activity.
Usable CA geometry, status and cached derivatives are copied without solving.
Unusable CA derived fields remain empty/NaN and are not read. Effect time is copied
through the accessor only for usable CA; unusable rows have explicit validity.

PCA instrumentation records incoming hints/precision/maxmoves, every solved piece
and its actual hints/status/usability/times, plus the final reference pointer and
last solved piece separately. Pointer assignment is observed before native
stable/usability/cap tests; cap exhaustion is not inferred after the fact. Solve
history statuses precede cap demotion; the enclosing record's final CA status is
post-demotion. No new search/evaluation is performed to obtain the trace. Allocation
failure leaves the trace invalid rather than affecting the native search.

Trajectory copies bind initial construction, hit-extension construction,
material discovery, wrapper construction and successful-candidate reference updates.
The candidate hook occurs **inside `if(status().usable())` immediately before
updateReference**, then native swaps. Rejected candidates do not replace this
binding. The frozen configuration excludes domain/end mutations; the domain
replacement copy is an additional explicit hook, not a certification of those paths.

### Material discovery

The normal branch records decisions after the native operands exist. Observers
copy locals and repeat only cached accessors/boolean predicates; they do not rerun
zTime, PCA, coordinate transforms, weight or material calculations. Predicates
repeated for diagnostics do not replace native control flow. Stage-unreached
computations are not fabricated. Material search captures excluded existing straws,
plane/radius cuts, panel windows/length cuts, attempted PCA/variance/longitudinal
operands and acceptance/object links.

**Vertical discovery is deliberately unsupported and marks capture failed** if
encountered. Frozen `AddStrawMinDz=0` makes its strict `abs(dir.z)<0` branch unreachable.
Reconstitution/implicit-hit constructors lacking explicit PCA binding also fail the
new constructor gate rather than silently claiming coverage. This preparation targets
the established full-schedule, single-extension, no-cluster/no-domain/no-processEnds
experiment, not arbitrary configurations.

## New sidecar schema (version 1)

Every target is keyed by `(run,subrun,event,iseed)`; all object/trajectory/sequence
indices are target-local, zero-based. Existing effect full keys remain
`target + iteration_ordinal + effect_index`, never miter alone.

### `crossing_records`

`object,sequence,stage,category,strawid,partner,trajectory_index`, reference state,
CA status/usability and usable vectors/derivatives, line definition, private offset,
constructor radius/speed, optional search history and effect time.

| stage | Meaning | Additional identity |
|---:|---|---|
| 0 | Constructor, before wrapper mutation | creation object ordinal; actual radius/speed |
| 1 | Ordinary updateReference | preceding iteration ordinal and supplied trajectory |
| 2 | Material wrapper updateReference | supplied wrapper trajectory |
| 3 | Post-updateState/prefilter | iteration_ordinal, effect_index, generic input_index |
| 4 | setHit partner mutation | contemporaneous partner; not final metadata |

`hint_kind`: 0 midpoint fallback, 1 own usable CA, 2 associated usable hit CA, -1
not applicable. Categories 0 initial, 1 associated extension, 2 material-only
extension are assigned at construction and retained across later records.

### `crossing_trajectories`

One row per native piece of each supplied trajectory: trajectory_index, category,
piece_index and q/C/range/bnom/mass/charge. Categories 0–2 name the construction
contexts above; 3 wrapper, 4 successful candidate, 5 domain replacement.
A record's trajectory index is not interchangeable with the old/candidate owner
index in the existing tree; joins must compare full payload and lifecycle keys.

### `crossing_discovery`

Ordered sequence, trajectory index, plane/panel/straw full IDs, object link, stage,
passed, integer/scalar operand vectors, and optional PCA/state. Full geometry
contains the ordered candidate universe; skipped descendants are identified by
failed parent stages and window bounds, not emitted as fabricated evaluated PCAs.

| stage | Scalars (`values`) | Integers / decision |
|---:|---|---|
| 0 | t0pos xyz, t0dir xyz, MinDz, radius, rmin/rmax, ymin/ymax, spitch, DOCAmax/consistency/Ubuffer/Nbuffer | MaxDStraw, SkipStrawCheck, then existing straw IDs; passed=normal branch |
| 1 | none | plane existence |
| 2 | plane z, zTime, position xyz, rho | radial cut |
| 3 | direction xyz, panel-plane dz, corrected position xyz, panel position xyz, dirv | ibuffer, istraw, nStraws; window cut |
| 4 | first window straw halfLength | istrmin; panel longitudinal cut |
| 5 | none | istrmin, istrmax; visited window |
| 6 | none | straw index; not-already-present decision |
| 7 | du, absdoca, dsig, native docaVar, halfLength | actual attempted PCA and strict acceptance decision |
| 8 | none | accepted constructor object link |

### Geometry and completion

`crossing_geometry_{bindings,tracker,planes,panels,straws}` copies **fresh geometry
for every selected target**, including exact tube endpoints and dsToPanel inverse
rotation/translation. This deliberately avoids pointer caching and assumptions
about IOV/address reuse. Binding status0 attempted then1 complete (−1 failure), with
exact plane/panel/straw counts and contiguous hierarchy indices, is mandatory.
Float halfLength stays float; double geometry is not renormalized. Ordered hierarchy
and flat collection pointer consistency are validated. Native API/copy-rebinding
contracts were checked against the pinned source, not just the local mirror.

This may produce roughly three million straw rows for147targets; that is an object count, **not a measured
file-size/runtime estimate**.

`crossing_capture_status` similarly requires status0 then1 per selected target and
exact n_records/n_objects/n_trajectories/n_discovery. Off-mode sidecars must be empty.
Original capture-validity fields intentionally remain unchanged even on a diagnostic
failure; new status and geometry-binding gates are necessary.

## Tests and limits of preparation evidence

- **71 new Python/static tests passed, plus5subtests**:18PCA patch,7geometry,
  37preparer,9capture wiring/guards. These validate exact anchors, scope/wiring,
  key structure, no native commands, exclusive paths, hashes and failure handling.
- Selected isolated suite **2,756passed,1skipped,54/54files**, pinnedcore233771b/ANN,
  CPU/single-thread; `local_data/trk_f1native_r2_20261027/isolated_crossing_prepare_v1.txt`.
  No full/slow-tier certificate.
- Review fixed unguarded diagnostic beginJob failures and unchecked status branches;
  readiness gates prevent partial initialization from appearing successful.
- A preparer test caught self-hashing a report inside its input package; report paths
  are now rejected before opening inside source/template/destination trees.
- One read-only review delegate exhausted its turn budget without a conclusion;
  it is not counted as approval. A replacement bounded review supplied the issues
  above. API/hash validation, pure Python tests and source review are not compilation.

Unusable CA/fallback branches, cap semantics, nested/threaded scopes, allocation
failures and compiler effects still need native/synthetic C++ runtime validation.
No native-exact geometry/constructor/discovery or full-fit closure follows from
source preparation. The acceptance99.5% and per-parameter99% gates are unchanged.

## User build/install handoff

On **mu2ebuild02**, in a fresh shell:

```bash
cd $TRKFS_MUSE_AREAS/fastreco_crossings_p109_20261028
sha256sum -c capture-source-manifest.sha256
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
muse setup -q prof:p109
muse build -j 32
```

Then install the matching local KinKal, also user-run on **mu2ebuild02**:

```bash
AREA=$TRKFS_MUSE_AREAS/fastreco_crossings_p109_20261028
KINKAL_BLD="$AREA/kinkal_build_p109" \
KINKAL_DST="$AREA/build/al9-prof-e29-p109/Offline/lib" \
  bash /path/to/build_kinkal_into_area.sh "$AREA"
```

Require `scons: done building targets.` and `INSTALL_KINKAL_DONE`. Do not run against
this area until the user confirms both. Afterwards verify source manifests,
source/plugin/local-KinKal timestamps and actual library linkage before any smoke.

## Next runtime work after the completed smoke

1. Host01 is not reserved by this note.
2. Retry with new v2 output/log only after at least8GiB is free. The receipt-bound smoke checker still
   accepts100events only. The earlier operand driver is hardwired to its old area; no predecessor driver/checker is
   modified.
3. Require complete147target status/schema/constructor/search/geometry/prefilter
   coverage and exact old-tree/hit-sidecar equality against the saved3,000event
   operand capture. Reuse that baseline, never refit it.
4. Keep exclusive versioned reports/directories and preserve all failed attempts.
5. Independently reconstruct tube lines and constructor CA/offsets for associated
   crossings first, then retained crossing history and material discovery. Never use
   captured acceptance/offsets/CA outputs to tune reconstruction inputs.

Fitter SHA stays `988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a`.
