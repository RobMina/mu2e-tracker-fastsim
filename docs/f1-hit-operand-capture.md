# Passive native hit-operand capture: preparation and runtime validation

Status (2026-10-28): **bounded native capture/passivity and row-algebra gates pass**.
The user explicitly confirmed both build/install completion steps. Source pins/hashes,
newer plugin/local KinKal timestamps and fully local linkage pass preflight. Seven
installed KinKal libraries match the local build's `.text` dumps; installed ELF names
use the expected 0.0.0 convention (build version 3.9.0). Build IDs are absent. The
build pane/log is unavailable; completion is user-confirmed, not independently read.
Preflight and `user-confirmation.txt` are preserved under remote data
`trk_f1native_r2_20261027/operand_build_preflight_20261028_v1/`.
The fitter and all previous evidence remain unchanged. This is instrumentation of
the fixed p109 experiment, not a reconstruction-physics arm on current Offline head.

## Executed runtime checks

Outputs under `$M2FS_DATA/trk_f1native_r2_20261027/`:

| Gate | Events | Seeds | Targets | Iterations | Effects | Result |
|---|---:|---:|---:|---:|---:|---|
| `operand_smoke_20261028_v2/check.json` | 100 | 77 | 2/147 | 49 | 3,481 | smoke pass |
| `operand_3k_20261028_v1/check.json` | 3,000 | 2,319 | 147/147 | 3,449 | 168,020 | full capture gate pass |

Frozen baseline → new off → new on recorded physics agrees. Expanded off/on FHiCL
differs only in `CaptureIterations` and debug output filename; new versus frozen
capture differs only in that filename. The diffs are saved in the full output directory.
`hit_operands_check_v2.json` verifies exact NaN-aware fields/types/full-key order
against the frozen capture for all four original trees: 2,319 seeds, 3,449 iterations,
130,273 pieces and 168,020 effects. The capture-off sidecar is empty.

All **61,622** straw snapshots join one-to-one, with **7,235 inactive hits**, **0 unusable
CAs**, **163,161 active residual rows**, and **0 excluded hits**. Reconstructed W/b passes
unchanged rtol=1e-5, atol=1e-7: **0/2,218,392 matrix** and **0/369,732 vector** component
failures, no nonfinite/variance-arithmetic failures. Maximum absolute differences are
5.960464477539063e-8 and 3.5762786865234375e-7 respectively (combined relative/absolute
gate, not an absolute-only claim). Input/source hashes remain unchanged through the audit.

Two test-harness corrections are preserved with their failed evidence:

- Smoke v1 exited 90 before events: `trace_f1_candidates.fcl` lives in the publisher's
  `art` include root, not the copied source area. The driver now uses the original
  publisher include path, checks its four-file chain and hashes it; no FCL was edited.
- First sidecar check rejected precision. `KKFitSettings.hh` declares `tpocaPrec` as
  `fhicl::Atom<float>`; `KKFit.hh` promotes it into double `tprec_`. The exact native
  value is **9.999999747378752e-5**, not double literal 1e-4. The checker now requires
  precisely `float(np.float32(1e-4))`; regression tests reject the double literal and
  either adjacent double. This is a source-type correction, not tolerance relaxation.
  `hit_operands_check.json` is retained; v2 checks the same ROOT output without rerunning it.

Focused Python regression tests: **89 passed** in the two capture/checker files; log
kept locally, not in the repository. The prior 805-pass/1-skip isolated
suite was not rerun. Both native arms ran serially on gpvm02 CPU 0; full off/on measured wall times were 21.73/32.50
s, maximum RSS 594,920/786,144 KiB. These are individual measurements, not a speed benchmark.

**Not full F1 closure.** This proves passive capture and consistency of captured
rows with captured cached weights on the selected native histories. Independent
CA/row generation, later state propagation, missing discordant targets, extension
construction, and full numerical/acceptance closure remain open. No fitter/default
changes. Unusable CA handling, nested/threaded scopes and allocation-failure paths
were not exercised by this sample. The [subsequent captured-CA audit](f1-captured-ca-operands.md)
reproduces initial CA vectors and derivatives over all 147 targets using source
signal-speed coefficients at captured references/mass; it preserves the full-fit boundary.

## Why this capture is needed

The captured-reference local CA reconstruction reproduces all 2,538 initial hit
particle times exactly across 147 captured targets. On the two saved actual F1
traces, near-zero-DOCA information discrepancies nevertheless remain on three
hits. Source-grouped derivative/information arithmetic and ROOT reciprocal sensor
normalization do not remove them. Native sensor time, POCA vectors and individual
residual rows were missing from the earlier capture; aggregate information cannot
uniquely provide them. The validated sidecar now supplies those cached operands.

## Isolated area and frozen baseline

New cluster-only area:
`$TRKFS_MUSE_AREAS/fastreco_operands_p109_20261028`.

Frozen, read-only source and built evidence area:
`$TRKFS_MUSE_AREAS/fastreco_capture_p109_20261026`.

`scripts/prepare_f1_operand_capture.py` requires the four exact previous capture
source hashes and all five repository pins. It refuses an existing destination,
creates detached worktrees, copies TrkFastSim, and makes exact-anchor additive
changes. It contains no environment setup, build, or physics command. Shared
publisher and dependency worktrees are not edited. The new area carries
`source-pins.txt`, `capture-source-manifest.sha256`, and `operand-preparation.json`.
A partial preparation is preserved for inspection rather than overwritten.

No new libraries are required: this is a header-only observer and ROOT sidecar in
the existing plugin. Neither CMake nor SConscript dependencies change. A complete
user build and the matching local KinKal install are still required because the
Track template and trace payload have changed. That requirement was fulfilled by
the user for the runtime checks above; these handoff instructions are retained for
reproduction, not a request to rebuild the now-validated area.

## Snapshot boundary and safe reads

`Track::IterationCapture` owns an optional measurement-observer function pointer.
The existing thread-local scope pointer reaches it; no Track instance member or
independent process-global observer is added. A null-sink scope disables its
observer. Nested scopes use the existing save/restore behavior. The observer is
called inside the existing `captureCopy` exception boundary, immediately after
copying the cached measurement weight in `captureEffects`: after state updates
and sorting, before filtering. Domain-growth recapture retains the existing
invalid-capture rule.

The callback receives a const generic Hit and only the new optional payload, not
mutable access to earlier captured operands. The TrkFastSim adapter recognizes a
direct `KKStrawHit`, copies individual cached members into a local value-owned
payload, then attaches it. It never calls updateState, updateReference, CA solving,
position evaluation, drift evaluation, residual evaluation or weight computation.

Inactive native residual derivatives can be uninitialized. They are **not read**;
their destination slots are NaN under a false active mask. Unusable CA geometry
and derivatives are likewise not read, and have empty vectors/NaN scalars. No
whole native CA or Residual object is copied. Exceptions invalidate diagnostic
capture through the existing error path, not fit status. Missing payload on an
expected straw effect, identity disagreement, and ROOT Fill failures are errors.

These source safeguards do not prove arbitrary allocation-failure behavior,
thread/nested-scope runtime behavior, compiler parity, or real-input passivity.
Those remain explicit validation requirements.

## Additive sidecar schema

Old tree schemas stay unchanged. `kktrace/hit_operands`, schema version 1, contains
one row per direct straw measurement effect, including inactive/out-of-fit hits.
Clusters, constraints and other measurement types get no fabricated straw rows.

Primary key:
`(run, subrun, event, iseed, iteration_ordinal, effect_index)`.

Consistency fields include `miter`, `iter`, generic `input_index`, snapshot source
`shindex`, raw `strawid`, and effect active/in-fit flags. History identity is joined
through the full iteration key, not meta-iteration labels alone.

- CA status/usability, precision and wire-hit state/constraint flags.
- Cached particle and sensor POCA `(x,y,z,t)` in mm/ns; cached directions `(x,y,z)`.
- Signed DOCA, DOCA/time variances, lsign, cached dDdP and dTdP when CA is usable.
- Three rows in native order **distance, time, longitudinal**: active, residual
  value, measurement variance, parameter variance and six prediction derivatives.
- Actual `miconfig.varianceScale()` copied at iterate entry; repeated on sidecar
  rows only. Iterations with no straw effects have no sidecar scale record.

Derivatives use native parameter order `(d0,phi0,z0,theta,t0,mom)`. Native dTdP is
the source residual/prediction convention, not a renamed derivative of particle
TOCA alone. Delta is derived from the two captured POCA vectors during analysis,
not an extra geometry calculation during capture. CA iterations/hint history,
private crossing offsets and extension-constructor state are still not captured.

## User build handoff

On **mu2ebuild02**, fresh shell:

```bash
cd $TRKFS_MUSE_AREAS/fastreco_operands_p109_20261028
sha256sum -c capture-source-manifest.sha256
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
muse setup -q prof:p109
muse build -j 32
```

Then install the matching local KinKal (also user-run):

```bash
AREA=$TRKFS_MUSE_AREAS/fastreco_operands_p109_20261028
KINKAL_BLD="$AREA/kinkal_build_p109" \
KINKAL_DST="$AREA/build/al9-prof-e29-p109/Offline/lib" \
  bash /path/to/build_kinkal_into_area.sh "$AREA"
```

Require `scons: done building targets.` and `INSTALL_KINKAL_DONE`. A helper exit code alone is insufficient. Native
tests were run only after user confirmation and local library linkage/source-timestamp verification.

## Required runtime gates after the user build

2. Use `scripts/run_f1_operand_pair.sh AREA INPUT_LIST CONFIG_DIR NEW_OUT N` with
   the preserved input list and `p109_iteration_targets_v2_20261026` configurations.
   Run the original 100-event prefix smoke, then the original 3,000-event prefix.
   The target union and capture-off/on settings stay fixed; no F1 baseline refit.
3. Run the existing publisher's `stock_iteration_check.py` with the frozen
   candidate baseline. Smoke remains smoke-only and permits target subset; full
   interpretation requires the complete 3,000-event gate. Inspect expanded configs.
4. Run `scripts/f1_hit_operand_check.py --check NEW_FULL_CHECK.json
   --reference-capture FROZEN_CAPTURE.root --out NEW_OPERAND_CHECK.json`.
   It requires the normal hash/passivity gate, an empty capture-off sidecar, exact
   old-tree fields/keys/values, complete sidecar-to-straw-effect joins and valid
   guarded payload shapes. Scale must be constant within each iteration; current
   capture cannot serve as its own frozen reference.
5. Reconstruct native information from captured rows without CA recomputation:
   `W = sum(g*g.T/(mvar*scale))`,
   `b = W*qref + sum(g*value/(mvar*scale))`.
   Use measurement variance, not `mvar+pvar`; mask inactive NaNs before arithmetic.
   The unchanged criterion is rtol=1e-5, atol=1e-7. Failures remain in denominators.
6. Only then compare native sensor state and rows to the existing saved F1
   diagnostics. Full fit, material, candidate, decision, later-history and
   population closure remain separate. No default promotion follows automatically.
