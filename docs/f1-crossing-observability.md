# Crossing observability and the next passive capture boundary

Status 2026-10-28: **preserved-input reference/order consistency passes; crossing
constructor/retained-CA/private-offset closure remains unavailable.** This follows
[meta/extension continuity](f1-transition-continuity.md). No fitter, native C++, default,
threshold or tolerance changes; no refit, native run or build.

## New audit on preserved evidence

`scripts/f1_crossing_observability.py` pins `transitions_v5.json` and its complete
inherited manifest, checks the actual executing modules and frozen source, then
rehashes all known inputs/sources before closing an exclusive strict-JSON report.
Generation does not invert captured material lengths/changes or fit offsets to times.

| Crossing class | Unique objects | Targets | Captured snapshots |
|---|---:|---:|---:|
| Initial | 2,538 | 147 | 60,217 |
| Associated extension | 81 | 50 | 1,405 |
| Material-only extension | 2,471 | 137 | 41,327 |
| **Total** | **5,090** | **147** | **102,949** |

Classes are disjoint by exact target/source keys, not by counts alone. Every snapshot
joins its object, including terminal iterations. The census rejects missing initial
seed objects, late-first-seen initial objects, disappearance/reappearance, duplicate
straws, unsupported partner histories, and extension objects first seen outside
**miter5/iter0**. Final-collection association stamps remain distinct from snapshots.

Every crossing's captured **range, six parameters, 36 covariance entries and nominal
field** exactly matches a piece in **owner0 of that same full iteration key**. All
102,949 snapshots pass, zero exclusions. Positive-width, nonoverlapping owner ranges
make the value lookup unique; use the range's right end rather than a rounded midpoint,
which can land on the previous piece for a one-ULP-wide interval. Effect time is not
used to choose a reference. This is reference-value membership, not a measurement of
which piece solved the CA; capped pointer/solution distinctions remain relevant.

### Associated extension inputs and ordering

Frozen `KKFit::addStrawHits` visits ComboHits in ascending index and appends accepted
crossings. `addStraws` follows it. `KKTrack::convertTypes`, `Track::createEffects`
and derived collection insertion preserve that source order, distinct from the
subsequently time-sorted effects. On all targets:

- Associated extension crossing input indices precede material-only indices.
- Associated partners are strictly ascending in ComboHit `shindex`.
- Every associated crossing has one same-straw new hit, with the same first snapshot.

These are **source-conditioned append-order/association checks**, not captured
constructor call sequence or iteration-local partner observations.

The audit joins all 81 associated first snapshots to the pinned transition report's
**hit-constructor** particle/sensor times, not its later wrapper-reference result or
its earlier zTime/htime inputs. Those reconstructed times are exactly the inputs that
source passes to the crossing PCA; the straw-line measurement-time anchor is the
hit-constructor particle time. They remain supplied-native-trajectory/accepted-membership
reconstructions, not native observed crossing hints or crossing constructor CA results.

For description only, the first-snapshot `crossing.time - partner_hit.time` ranges
from **-0.0028486598894232884 to +0.006057990234694444 ns**; **43/81 are nonpositive**.
This is not an offset test or ordering failure. Each retained CA can have a different
particle time, and the Material wrapper has already updated the crossing reference.
The constructor's positive `toff_` does not imply this observed difference is positive.
Never estimate `toff_` from that subtraction.

## Why the existing capture cannot close construction

Frozen source is authoritative, not the stale local `off` mirror:

1. `strawLine(straw,time)` constructs its SensorLine from native
   **strawEnd(cal), strawEnd(hv)** with light speed. `Straw::strawEnd` uses the tube
   midpoint/direction `_smid/_sdir`, whereas dumped `wireEnd` uses `_wmid/_wdir`.
   `getMidPoint()`/`origin()` are wire midpoints. The conditions tree contains wire
   endpoints/direction, not tube endpoints or exact native panel transforms.
2. Associated construction calls PCA on this tube line with reconstructed hit-CA
   times, then passes that PCA to the explicit-CA `KKStrawXing` constructor.
3. The private stored offset is `wireRadius / constructor_ca.particleTraj().speed(
   constructor_ca.particleToca())`. It is fixed at construction, not recalculated
   on later reference updates. A capped PCA can retain geometry from one piece while
   its reference pointer identifies another; capture both identities explicitly.
4. Material wrapper construction immediately invokes `updateReference` again.
   Crossing updateReference prefers its own usable CA, then its associated hit's
   usable CA, then straw-line midpoint times. The first filter snapshot is therefore
   not the crossing constructor snapshot.
5. Existing effects provide reference payloads and offset-inclusive time, but no
   crossing retained particle/sensor times, POCAs/directions, status, explicit hints
   or private offset. The newer sidecar is hit-only. Initial seed offsets are time
   differences, not private-field reads; all 2,552 extension offsets are absent.

Inferred tube geometry, accepted final membership, scalar effect time and aggregate
material operands cannot be substituted for these missing observations. No independent
crossing CA/offset comparison was run or claimed.

## Material-only discovery: source-established scope, not replay

The frozen expanded configuration is checked explicitly:

| Setting | Value |
|---|---:|
| AddStrawMinDz | 0 |
| MaxDStraw | 2 |
| MaxStrawDOCA | 5 mm |
| MaxStrawDOCAConsistency | 1 |
| MaxStrawUposBuffer | 0 mm |
| StrawNBuffer | 0 |
| SkipStrawCheck | false |

The vertical branch tests strict `abs(direction.Z()) < AddStrawMinDz`; with zero
threshold it cannot be selected. This is source/configuration reasoning, not proof
that every unobserved normal-branch longitudinal direction is finite/nonzero.

All 2,471 accepted material-only objects must therefore come through the normal
plane/panel/straw hierarchy under this configuration. Its missing evidence includes:

- Plane existence/order/origins and zTime on the pre-extension trajectory, radial cuts.
- Exact panel origins, v direction and dsToPanel transform; corrected panel position.
- `fillTrackerInfo` values derived from the first plane/panel: ymin/ymax, rmin/rmax,
  straw pitch and radius; integer/rint window construction and endpoint bounds.
- Exclusions of initial and newly associated crossings before straw PCA.
- Native tube-line operands and PCA, sensor longitudinal projection onto wire geometry,
  DOCA variance from the actual referenced covariance, and strict cuts on
  DOCA, `max(0,abs(DOCA)-radius)/sqrt(max(docaVar,min_double))`, and longitudinal position.
- Rejected and skipped candidates at every stage, not just the final accepted objects.

The source's zero StrawNBuffer removes a finite angular buffer contribution, not the
need to validate the actual computations, panel geometry or candidate universe. There
is no candidate-discovery or selection-efficiency certificate in this report.

## Subsequent source preparation

[Passive crossing capture](f1-crossing-capture.md) implements the following proposal
in isolated `fastreco_crossings_p109_20261028`. Source hashes and Python/static tests
pass; **user build/install and native passivity remain pending**. This does not alter
the observability report or supply its missing native observations retroactively.

## Minimal next passive capture proposal — checkpoint scope

Use a **new isolated p109 instrumentation area**, derived from the validated passive
hit-operand area, preserving both existing native areas and all baseline outputs.
This is a proposal only: no C++ edits, area creation or build command was executed.
A user-built/install-confirmed step is required before any new native runtime test.

### A. Crossing construction and retained state

1. **Native geometry:** independently dump exact tube cal/hv endpoints and wire
   geometry for all straws, with geometry/conditions identity. Retain float/double
   semantics, line time anchor, speed and actual normalization operands/results.
2. **Construction record:** full run/subrun/event/seed plus source object identity,
   creation category (initial/associated extension/material-only), creation ordinal,
   and constructor-time partner identity. Bind to the pre-extension trajectory and
   actual CA reference owner/piece by stable per-capture indices, never raw addresses.
3. **Native PCA record:** actual incoming particle/sensor hints, precision, move limit,
   attempted/solved piece indices, final pointer, move/cap outcome, and retained CA
   status/usability, both times, both POCAs and directions. Keep constructor and wrapper
   states separate. Do not call solve, updateReference or updateState to take a snapshot.
4. **Private offset:** capture stored `toff_` itself through a read-only capture path.
   Record the actual evaluated speed and wire radius at the constructor assignment;
   do not reconstruct the private value as `time()-particleToca()`. If recording these
   operands needs source restructuring, treat its passivity as a new test obligation.
5. **Wrapper/iteration snapshots:** immediately after Material construction's reference
   update and at the existing post-updateState/prefilter boundary, copy the retained
   CA/reference, stored offset, effect time and contemporaneous partner identity.
   Include all crossings, active or inactive; unusable fields carry explicit validity,
   not silent selection. Associate every payload with its constructor object.

A scoped optional observer/value-owned payload can follow the proven hit-sidecar
pattern; design it so disabled capture does not recompute physics or consume RNG.
Default-off instrumentation must not add unscoped global state or mutable Track members.
Empty/missing/duplicate/mismatched payloads and ROOT Fill failures fail the capture gate.
Pointer-before-stop and solved-piece provenance require observing the actual PCA path;
post-hoc lookup from a CA pointer alone is insufficient.

### B. Material discovery, separately identifiable payload

Record the ordered plane/panel/straw universe and every visited or rejected stage,
with exact input geometry/transform/trajectory keys, existing-object exclusions,
computed search positions/windows/hints, CA/variance/longitudinal operands, cut values,
decision and accepted-object link. Record branch-unreached reasons without evaluating
computations that native code skipped. Exact static geometry plus stage records should
support independent replay, not merely repeat saved decisions. Accepted-only records
would improve constructor validation but still not close discovery completeness.

### Required runtime gates after a user build

- Verify source/build/library identities in the new area; do not reuse stale binaries.
- Require exact NaN-aware original-tree equality and unchanged physics against the
  frozen predecessor, with explicit highest-cycle/full-key checks; the off sidecars
  must be empty. Keep constructor, wrapper and iteration populations distinct.
- Count all crossings/objects and failed/missing payloads. Independently compare
  tube geometry, generated constructor inputs/CA and stored offsets before attempting
  the material-discovery replay. No tolerance or threshold relaxation.
- Synthetic tests cover disabled/nested/threaded scopes, failure handling, unusable
  fallback precedence, duplicate partners and the capped solved-piece/pointer split;
  native sample coverage remains explicitly bounded.

Later unbiased CAD/ANN policy, strict ROOT/filter algebra, independent F1 chart/history
and the acceptance/fidelity targets remain open even if this capture succeeds.

## Receipt, tests and operation

Final remote data report:
`trk_f1native_r2_20261027/crossing_observability_v2.json`

SHA256 **`20217e4e7a32791d14291d64418ac2bf5873140b8a5b2930090fb9613e8b6626`**.
Completed, all hashes unchanged and consistency passed; full history/state-policy/fit
and native crossing-constructor-offset comparison remain false.

- Script SHA `3d74d2fe717867de4823cf1e19a8cae9bc10fe13e92698df0307307d61048fd7`.
- Wrapper SHA `09bc8baeaa45bf36908554c4db2016ad2734851abb87d3e70dbb2c18652e6cdd`.
- v1 report retained, SHA `4a91b3c487f73edf98fb2c6b58caedf031552e688d345acd91bab49dd7a08c06`.
  v2 supersedes its executable-source binding with first-row bounds/uniqueness guards
  and range-end lookup; the measured population results are unchanged.
- **231 new tests passed** (207 observability +24 report/guard tests). A proposed
  whitespace-normalization test was corrected: the parser deliberately requires
  identical expanded block text, consistent with the pinned predecessor. It rejects
  differing indentation rather than silently broadening the supported configuration.
- Review caught a one-ULP range midpoint false-negative; the final range-end lookup
  has a dedicated regression. Strict failure JSON, immutable predecessor handling,
  input masks/bounds, full keys, source ordering and output independence are tested.
- Selected isolated suite **2,685 passed, 1 skipped, 50/50 files**, pinned core233771b,
  ANN/CPU/single-thread. Local log `local_data/trk_f1native_r2_20261027/isolated_crossing_v1.txt`.
  No full/slow-tier certificate.

```bash
bash scripts/run_f1_crossing_observability.sh \
  $M2FS_DATA/trk_f1native_r2_20261027/NEW_REPORT.json
```

Fitter SHA remains `988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a`. Remote `git diff --check`
passes.
