# Device world transport

`steps --backend jax` selects device world transport before the existing device interior, exit scattering,
crossings and gas-ionization stages. NumPy remains the default and reference. No production ledger convention,
calibration, stopping threshold or physics parameter is changed.

## Boundaries

`world_jax.DeviceWorld` consumes the `BandedWorld` prepared by `steps.load_world`. Raw lists are rejected because
their material insertion order differs from the packed world's sorted order, and sequential slowing depends on
that order. Edges must match the tracker plane, including on a cached device-world reuse. Geometry must remain
immutable while cached; construct a new `BandedWorld` for a geometry change.

Device work:

- Ray/box tests and exact convex-plane clipping in 32-piece device-loop tiles; every piece is considered,
  without a candidate cap. Fixed-point column accumulation is exact across tile boundaries. Tiling bounds
  intermediate shapes and avoids pathological compilation of large candidate/material reduction fusions.
- Per-band, sequential per-material sliced loss using the existing core Urban sampler and the same fractional
  slice, stopping and final-mean-fallback policy; active lanes compact between eight-slice blocks, then mean
  air loss is applied over uncovered path. The dense fixed-lane implementation remains a regression reference.
- Band momenta, material radiation lengths and nuclear-tail coefficients; correlated Highland angle/displacement
  using the same total-column correction, optional band momentum and within-band spread conventions.
- Nuclear-tail Poisson proposals with per-proposal keys and form-factor thinning. This world sampler has no new
  fixed proposal-count cap.

Host work: generator selection, packed geometry construction, bucket decisions, orchestration, primary-table
and path-vertex assembly, and library-secondary pasting. This is not an entirely device-resident chain.
The legacy energy/time/terminal bookkeeping limitations are inherited; device transport does not resolve them.
Nonlegacy ledger and other unsupported JAX options remain rejected.

## Randomness and support slicing

One NumPy RNG draw creates the JAX root key before transport. World keys are indexed by original muon position,
band, material and purpose. World and interior streams are separate. This changes JAX realizations from versions
that ran world transport on the host; old CPU tables must not be used as a cross-device reference.

Support slicing retains each pair's original `fold_in(pair_key, slice_index)` stream. Batch-level early completion
skips only identity rounds after all real lanes finish. Out-of-bounds padding cannot keep the pending-work list
alive; writes for padding are dropped. Slice limits and fallback behavior are unchanged. Tests compare all carry
fields and final losses against the fixed-round implementation, including near-cap continuation.

## Validation

Use a pinned, clean core checkout on both devices. `scripts/jax_steps_cpu.sh OUT [SHARDS...]` supports
`CORE_DIR` and a persistent `JAX_COMPILATION_CACHE_DIR`, uses one CPU core and stops at the first failed shard.
The GPU stream driver accepts extra `EXPORTS`, including `VALIDATION_SHARDS` to distinguish the five cross-device
validation shards from a larger throughput batch. CPU-JAX reference tables must come from matching sources,
configuration, seed and input ordering. Missing optional cross tables do not count as a successful comparison.

`jaxcheck` writes actual generated event counts separately from generator-particle rows. Its legacy step-level
mean errors, histogram errors and independent-bin occupancy chi-square are descriptive: a muon's repeated steps
are correlated. `event_clustered` supplements them with paired event-cluster influence covariance, keeping all
primaries and steps of an event together. For a step-weighted mean, each event contributes
`(sum_event(value) - mean * nsteps_event) / total_steps`; subtract the arms' influences and sum their squares
with the finite-event correction. Normalized panel occupancy uses the analogous full covariance and a
rank-revealing eigendecomposition. Report marginal diagnostics and residuals, not only a successful script exit.
Moyal fit errors and legacy spectrum bands are not promoted to event-bootstrap errors by this addition.

Synthetic tests include geometry, empty/invalid/stopped lanes, loss/scattering statistics, exact slice-loop
regression, chunking/reproducibility, the panel-cut fallback, and event-cluster duplication invariance.
`scripts/world_device_check.py` audits actual-world columns and upstream transport independently of tracker
acceptance. `scripts/numpy_default_check.py` compares the default against a saved base `steps.py`, including
all primary steps, particles, path vertices and final RNG state on a real input.

Measure warm uninstrumented wall time separately from the first compiling pass and from synchronized kernel
timings. Whole-stage throughput includes host work and must be compared with one CPU core on the same event
count. The geometry's lane/tile/plane/material products determine device working memory; inspect these before
increasing batch sizes or using a new geometry. A fast kernel alone is not the whole-stage performance result.

Saved reports include source/input SHA256s, runtime versions and effective settings. Completed passes print
immediately; an `incomplete` partial JSON preserves timings during long compilation of later passes. The saved
report auditor rejects incomplete evidence, absent transport counters, reported reproducibility failures and
identity/filter mismatches. Numerical cross-device magnitudes still need explicit review; those identity gates
are not a numerical-tolerance or physics-acceptance certificate.
