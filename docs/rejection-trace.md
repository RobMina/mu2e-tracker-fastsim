# Matched-history rejected-candidate trace (experimental)

This branch requires the companion KinKal branch `thread-f-rejection-trace-v3.7.0`. It is **not built or
physics-validated**. Do not use its trace plugin with the stock backing libraries: the added `Track` members change
its C++ layout and require a coherent rebuild of affected Offline/KinKal consumers in an isolated area.
No installed CVMFS headers or shared reconstruction area have been modified.

## Verified backing, October 25

Read-only `muse setup` in `/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/tracker/art` on `mu2egpvm06` identified:

- `MDC2025ay` -> Offline `v13_39_00`, `al9-prof-e29-p107`, muse `v4_14_01`.
- KinKal `v03_07_00`, Spack package `kinkal-3.7.0-wdoyhvln75lqde5lv32a5koch7ufsoy7`.
- Installed `KinKal/Fit/Track.hh` SHA256:
  `f668f4bc485f943019aef159c077d744032362f7eda2faef65e02e1063a850bc`.
- Git tag `v3.7.0` is `2011f342bb3f233ebc11c4c57dea358e4af8d02f`; its header has the same hash. The earlier
  local source investigation therefore applies exactly to this header, despite the environment initially being
  unverified. The existing trace plugin timestamp is October 23, 21:59; no new binary was produced here.

## Passive capture

KinKal's `captureRejection(bool)` clears the previous snapshot and enables/disables capture, default off. The first
actual `paramsdiverged` decision stores candidate/reference front/back trajectories, exact raw parameter steps,
reference indices, threshold, parameter-change chi2, average gap, delta-chi2 and status. The capture uses copies
of the values already computed at the decision; it does not change acceptance or update the trajectory. Track
cloning starts with capture disabled and no snapshot. Capture is enabled after the seed fit, before extension;
seed-construction failures are outside this initial diagnostic's scope.

The trace module adds `FullSchedule` and `CaptureRejection` (both default false). Set `FullSchedule: true` in both
arms, `PerIteration: false`, and change only `CaptureRejection: false/true`. Enabling capture without full schedule
or combining full schedule with capped iteration reruns throws. Full schedule emits one row per selected seed,
`stage = -1`, `maxit = -1`; actual `miter`, `iter` and history are the stock execution's identifiers. An unusable
seed is recorded without extension. Other seed-selection and good-fit gates remain in place.

The new output is deliberately not compatible with scripts that treat negative stage values as histogram bins.
Use `scripts/rejection_trace_check.py`, not `trace_check.py`, `cov_eigen.py`, or mixed-cap trace comparators.

### Snapshot arrays

- `rejected`: one when an actual parameter-divergence snapshot exists, otherwise zero and all arrays empty.
- `rejected_parameters`: 4 x 6 doubles, flattened; candidate-front, reference-front, candidate-back, reference-back.
- `rejected_covariances`: the same order, 4 x 6 x 6, row-major doubles.
- `rejected_ranges`: 4 x 2 begin/end doubles in the same order.
- `rejected_steps`: front then back, 2 x 6 raw parameter differences (the exact KinKal decision convention).
- `rejected_metrics`: front chi2, back chi2, threshold, average gap, delta fit chi2/ndof.
- `rejected_reference_indices`: two indices in the old trajectory selected by candidate endpoint midpoints.
- `capture_enabled`, `full_schedule`: configuration markers.

Existing retained `pf/pb/cf/cb`, chi2, momentum and chi2 history output is now double precision. These still
represent the retained trajectory, not the candidate. Event identity includes run/subrun/event/seed.

## Validation order

1. Prepare an isolated full Offline v13_39_00 + KinKal v3.7.0 area with these diagnostic branches; do not overlay
   the changed header onto the frozen backing plugin. Rebuild all affected consumers together (user only).
   Reuse the area's existing KinKal build/install protocol; a new header layout must not mix with old binaries.
2. Run the same bounded input once with capture off and once with capture on, same full schedule and settings,
   fresh output tags. This is a diagnostic-capture A/B, not a regenerated physics baseline.
3. `python3 scripts/rejection_trace_check.py CONTROL.root CAPTURED.root` must pass before interpreting projections.
   It requires identical seed populations, unique full event/seed keys, full uncapped rows, and exact stored
   status/chi2 histories, retained endpoints/covariances, hit states/quality and crossing counts. It rejects empty
   files. Equality of stored fields does not prove equality of every unrecorded effect or intermediate state.
4. The checker verifies stored steps, recomputes endpoint parameter-change quadratic forms, requires the recorded
   threshold crossing, and reports scaled-reference weak-direction projections. Do not infer F1/stock causation
   from this stock-only output: a matching F1 decision-state snapshot is a separate next change.

Nine synthetic checker tests pass locally; 26 pass when combined with validity, empty-input and combo tests.
No C++ compilation, syntax check, art run, actual rejected-candidate measurement or capture-on/off equivalence
has been performed. No production acceptance or fidelity threshold is relaxed.
