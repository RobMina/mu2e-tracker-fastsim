# Opt-in resident measured-hit wire reconstruction

This is a deliberately limited integration baseline, not CentralHelix/KinKal or the
CPU stereo-association/drift studies. It reconstructs one constant-axial-field
geometric helix hypothesis per measured time group and refines it against original
3D wires. Defaults and the extracted-tracker line chain are untouched.

## Supported boundary

`reco.resident.reconstruct(event_keys, hits, *, bz, max_hits=128,
group_batch=32, max_iterations=40)` accepts device arrays and returns device arrays.
The entire grouping/seed/refinement chain composes under one outer `jax.jit`.
There is no NumPy fallback, host callback, host membership calculation or
intermediate device download. Float64 is required. This measured-hit entry point
is not yet a fused digi-to-track chain; the existing public straw-hit wrapper
still downloads its output.

`event_keys` is lossless integer `[E,3]` (run, subrun, event), unique and nonnegative.
`hits` contains exactly these flat padded columns:

| Column | Shape/type | Meaning |
|---|---|---|
| `ev`, `sid`, `flag` | `[N]` integers | event-row index, packed straw ID, measured SH flags |
| `time` | `[N]` real | measured corrected time, ns, used for grouping only |
| `wdist`, `uvar` | `[N]` real | measured along-wire position mm and variance mm² |
| `mid`, `direction` | `[N,3]` real | original wire midpoint mm and full 3D direction |
| `hlen` | `[N]` real | finite half-length mm |
| `valid` | `[N]` bool | actual input row versus padding, not a fit-quality mask |

Wire direction uses the measured hit's along-wire sign convention (HV→cal for the
Run1A exporter). It is normalized without discarding alignment tilt. `mid +
wdist*unit_direction` supplies the measured seed position. No truth, stock-track
membership, saved association, drift tables or preset momentum enter this API.
Unsigned 64-bit integer arrays are rejected at the device boundary; the explicit
host adapter range-checks and converts losslessly to signed int64. Shapes/config
raise at trace time; invalid runtime numbers remain failure statuses.

```python
import jax
import numpy as np
from trkfastsim.reco.residentio import prepare
from trkfastsim.reco.resident import reconstruct

keys, hits = prepare(event_keys, measured_sh, matching_conditions)
dkeys, dhits, bz = jax.device_put((keys, hits, np.float64(1.0117)), device)
compiled = reconstruct.lower(dkeys, dhits, bz=bz).compile()
with jax.transfer_guard("disallow_explicit"):
    result = compiled(dkeys, dhits, bz=bz)
    jax.block_until_ready(result)
# Download explicitly only where serialization or host analysis needs it.
host_result = jax.device_get(result)
```

`residentio.prepare` is a separate O(N) host input/geometry adapter for unpadded
SH tables; a supplied `valid` field is rejected rather than reactivated. It performs
no time grouping or association. Packed straw IDs require plane <36, panel <6,
straw <96. Resident consumers may provide the device schema
directly. Conditions must match the sample and IOV; there is no conditions service.

## Algorithm and deliberately unsupported cases

1. Stable event/time/straw/source-row ordering; accepted SH flags exclude bit 6,
   dead and noisy, and require tdiv. Adjacent time gaps >20 ns start groups.
2. Entire groups with >100 ns span, repeated straw IDs, invalid geometry or more
   than `max_hits` fail before fitting. No partial fit of an overflow prefix.
3. Centered/scaled algebraic circle followed by phase-versus-z geometric seeding.
   Repeated measured z, singular circles, non-identifiable pitch and phase span
   >=2π fail. Adjacent actual phase gaps must be <π; undersampled turns cannot be
   distinguished. No z jitter, outlier removal or alternative hypothesis search.
4. Five parameters `(x0,y0,tx,ty,omega=dphase/dz)` at fixed `z_ref`; bounded
   positive-Hessian local POCA and diagonally equilibrated LM wire refinement.
   Every accepted trial requires all selected POCAs to converge within both the
   search basin and finite wire. Strict objective decrease is necessary but not
   sufficient: final unconstrained gradient, rank and conditioning determine
   numerical convergence. Failed fits retain last accepted state.

The objective is half the sum of redundant perpendicular xyz residuals squared
with a 2.5 mm engineering scale, plus along-wire pulls squared. It is **not χ²**.
The direct-position seed is not the exhaustive stereo winner used by the CPU
studies: their convergence counts do not transfer to this implementation.

Not supported/validated: drift-time fitting, multiple tracks within a group,
outlier handling, flight direction/charge, field-map propagation, material loss
or scattering, covariance/uncertainty coverage, momentum resolution, efficiency
or fake rate. The resident chain exports no momentum or covariance. The standalone
geometric seed helper has algebraic momentum-magnitude diagnostics only.

## Output and failure contract

Every input row is returned unchanged as `input_*`, including `input_valid` (the
row-presence mask). `batch_valid` is a distinct scalar identity/config validity
flag; it does not certify geometry or any track. Full event keys and event hit,
group and converged-group counts include empty events. `hit_group=-1` means not
grouped. Per-hit status: 0 padding, 1 grouped, 2 flag-rejected, 3 invalid input.

All actual groups occupy the first `n_groups` slots of arrays with capacity N;
`group_valid` distinguishes them from padding. Each has event index, complete hit
count, time span, integer status, raw seed/fit statuses, explicit stage-attempt
flags, objective validity, initial/final score, last local parameters, reference z,
iteration counts, rank, condition and gradient norm.
`hit_doca`, `hit_wire_pull`, `hit_poca_valid` retain source-row alignment.
Failed/unsupported hit diagnostics are zero with false validity, not good fits.

Group codes: **0** padding, **1** numerically converged, **2** insufficient hits,
**3** wide time group, **4** hit-capacity overflow, **5** invalid input,
**6** duplicate straw, **7** seed failure, **8** fit failure. Invalid identity/time
invalidates its whole event; an out-of-census event index or bad full event keys
invalidates the batch. Group failure precedence is invalid input, capacity,
wide time, duplicate straw, insufficient count, seed, fit. Raw seed/fit codes are
interpreted only for stages that ran, not as standalone success flags.

Group arrays use O(N) storage; fit workspace is bounded by
`group_batch * max_hits <= 262144`. Limits: 4..512 hits, 1..2048 groups/batch,
0..120 LM updates. The device loops over group batches, not Python event/hit loops.
This is not a bound on XLA compiler/allocator overhead or the caller's input size.

## Reproduction

No private input data is committed. Reuse the saved development hit table and
matching geometry; no new simulation/digitization or exhaustive association.

```bash
python3 -m trkfastsim.validation.residentcheck \
  --hits SAVED/numpy.npz --conditions SAVED/conditions.json --bz 1.0117 \
  --outdir NEW --require-gpu
```

This refuses an existing output directory and refuses CPU fallback when GPU is
required. It records input/code/output hashes, actual platform/device, x64, full
census and numerical status counts, exact repeats, and separate read/prepare,
upload, compile, synchronized resident execution and download timings. Transfer
guard surrounds execution, not upload/download. This is not digitization timing
or a matched stock reconstruction speedup. `scripts/resident_gpu.sh` runs the
focused synthetic tests and saved development sample inside an existing Apptainer
image in an allocated GPU job; no software/image build is performed.

## Measured validation (2026-11-06)

JAX/jaxlib 0.11.2, float64; saved development hits and matching Run1A conditions.
No thresholds were tuned on held-out data. The focused local suite has **201
passes** (32.04 s). An isolated source snapshot containing only the scoped device
implementation/tests runs on **NVIDIA A100-SXM4-80GB**: **198 passes, 3 skips**
(240.15 s). Skips are optional comparisons to CPU prototypes intentionally absent
from this PR; the independent synthetic/grouping/failure tests all run.

| Development census/status | JAX CPU | A100 |
|---|---:|---:|
| Full events / empty events | 1,000 / 235 | 1,000 / 235 |
| Source hits / time groups | 13,492 / 773 | 13,492 / 773 |
| Numerical wire-fit convergence | 39 | 39 |
| Insufficient hits / seed failure / fit failure | 23 / 15 / 696 | 23 / 15 / 696 |

Original input arrays, full event/group census and hit membership compare exactly;
no group status changes. On the 287 groups with valid objectives on both devices,
maximum absolute final-objective difference is 4.69e-6. This comparison does not
certify parameter/residual agreement: the CLI currently reports structural
identity, group-status transitions and objective differences only. CPU group-batch
17/32 also preserves structure/statuses, but maximum objective difference is
2.65e-6; no bitwise batch-independence claim. Three identical GPU repeats compare
all output arrays exactly at fixed shape/device, including failed lanes.

A100 timings for this sparse sample: read/prepare 0.121 s, upload 0.0186 s,
compile 12.541 s, synchronized resident runs **1.297, 1.199, 1.210 s**, download
0.00639/0.00254/0.00264 s. These are measured-hit-to-wire-output times, excluding
serialization, not digitization/stock/end-to-end or production throughput. No
GPU memory/utilization profile was measured. The 39 numerical convergences are
not tracking efficiency; the limited direct-position seed intentionally trades
capability for a composable failure-preserving device implementation.

Inputs SHA256: hits `9bd666236840e2525eb9fe03ccca741c5ee0a9a44726ad3116e80be2dd4d1b2d`,
conditions `f78baed3c613b4692993621fcbb3917303b073fca502a38742c77782780e770e`.
GPU output SHA256 `724d9d1baa064476304b75b34737a2725cd2e1d2c5e43f0c9636efacd1fe6d07`.
Reports include the exact source hashes; subsequent source changes are comment-only
in the seed module (two comments shortened) and docs, with focused local tests
rerun. A first isolated GPU test job stopped at an unconditional optional CPU import;
that test now skips when the prototype is unavailable, without changing a kernel.
