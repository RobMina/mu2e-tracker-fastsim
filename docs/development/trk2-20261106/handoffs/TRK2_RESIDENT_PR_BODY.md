# Add opt-in GPU-resident measured-hit wire-helix reconstruction

## Purpose

Unblock device consumers with a bounded, failure-preserving reconstruction API,
without waiting for a complete CentralHelix/KinKal replacement. The input boundary
is calibrated measured straw hits plus matching geometry; not truth, stock tracks
or saved CPU associations. Existing extracted-tracker defaults are unchanged.

## Changes

- Pure-device float64 geometric helix seeding and bounded wire-only LM/POCA fit.
- One JIT-composable measured-hit pipeline: time groups → direct measured-position
  seed → original 3D-wire refinement, without host callbacks or NumPy fallback.
- Preserve complete event and source-hit census, every time group and failed state;
  reject whole over-capacity groups rather than fit truncated prefixes.
- Explicit stage-attempt flags and numerical status codes; no momentum/covariance
  export from the integrated path. Document the one-hypothesis/group limitation.
- Separate checked host adapter, reproducible CLI, synthetic tests, and A100 runner
  using an existing runtime image.

## Validation

- Local focused suite: 201 passing tests, warnings-as-errors. This includes three
  optional comparisons to CPU prototypes that are not part of this changeset.
- Independent grouping/census oracles, exact-input preservation, empty and failed
  lanes, overflow rejection, malformed geometry/IDs, finite-wire/POCA failure,
  numerical stationarity, stage statuses and batch/permutation tests.
- Execution under `jax.transfer_guard("disallow_explicit")`; callback-free JAX trace.
- Saved development sample: all 1,000 events (235 empty), 13,492 hits and 773 groups
  retained. CPU: 39 numerical convergences, 23 insufficient-hit groups, 15 seed
  failures and 696 fit failures. Batch 17/32 preserve structure and group statuses;
  continuous results are not claimed bitwise batch-independent.
- A100-SXM4-80GB: 198 tests passed, three optional CPU-prototype comparisons
  skipped in the isolated scope-only snapshot. All independent tests ran.
- A100 saved sample preserves the complete census and all CPU group statuses;
  three GPU repeats are exact across every output array. Maximum absolute final
  objective CPU/GPU difference 4.69e-6 on 287 jointly objective-valid groups;
  parameter/residual cross-device agreement is not certified by this comparison.
- A100 resident runs 1.297/1.199/1.210 s; compilation 12.541 s, upload 0.0186 s,
  download 0.0025–0.0064 s separate. No stock or full-chain speedup claim.
- First isolated GPU test job failed on a test-only optional CPU import; fixed to
  skip when that prototype is absent. No reconstruction threshold/kernel changed.

## Limitations and acceptance scope

This is an experimental geometric reconstruction baseline, **not** stock-equivalent
reconstruction. Direct-position seeds replace neither the richer CPU association
study nor a drift fit. No drift calibration fit, multiple-track separation,
outlier model, charge/direction inference, field-map/material propagation or
uncertainty coverage is implemented. Numerical convergence is not tracking
efficiency, calibrated χ², physical resolution or truth matching. No matched stock
throughput or end-to-end digi-to-track speedup is claimed.

