# Initial unbiased CA, drift calibration and CAD decisions

This follows the [initial residual/variance audit](f1-initial-residual-variance.md), using only
preserved data. New `scripts/f1_initial_cad_audit.py` and `f1_initial_drift.py` reconstruct
CAD eligibility rather than prescribing the captured null state. No fitter, C++, default,
threshold or tolerance changes; no refits, new native jobs, commits or production promotion.

## Scope and source lifecycle

The native initial path is:

1. Fresh `KKStrawHit` is active/null if its constructor CA is usable.
2. `ResidualHit` contains default `Weights`; ROOT `SMatrix` and `SVector` constructors zero
   the stored matrix/vector. This is source-established, not a pre-update weight snapshot.
3. `KKTrack` has no hit clusters in this FCL. `Measurement` construction calls only
   `updateReference`; it does not evaluate weights or hit states.
4. The first `Track::iterate` calls `updateState(first=true)`. `updateWHS` first computes
   `unbiasedClosestApproach`, before the first `updateWeight`.
5. Active-hit unbiased parameters therefore execute **C → W → C′** and **q → Wq → C′Wq**,
   even though the subtracted measurement weight is zero. No endpoint prior deweighting
   belongs in these reference covariances.

The diagnostic joins captured seed, initial old-piece and hit-reference covariances by
full identities and requires exact equality. Captured q, covariance and mass remain inputs.
Generated constructor/reference CA vectors match the sidecar exactly, as do all **2,538
reference DOCA variances**.

## Physical field versus nominal seed vector

The geometry's physical field is zero (`bfgeom_no_field.txt`). The seed nevertheless uses
`bnom=(0,0,0.001)`, and `KinematicLine(other,newBnom,time)` copies the original without
changing this stored vector. All three captured sources retain it. Its unit direction is
used in `Mu2eKinKal::LorentzAngle`; it is **not** the physical field used by `StrawDriftMaker`.

At zero physical B, native `tempGamma=1` for every phi bin: the distance/time tables are
angle-independent. The reconstructed Lorentz angle is retained as a diagnostic, but does
not alter the drift values. There is no nonzero-field claim.

The actual geometry include chain is read and hash-bound (469 include visits); it supplies
outer straw radius **2.5235 mm** and wire radius **0.0125 mm**. The outer radius also exactly
matches the independent conditions dump. Do not substitute the hard-coded 2.5-mm response
radius or 2.4-mm null-variance scale for this geometry radius.

## Source-rebuilt drift calibration

`f1_initial_drift.py` uses source electric-field/velocity data, source drift/response FCL
coefficients and geometry. It preserves sequential distance/time accumulation, and the
native time inversion's **first strictly greater** distance-time index (then upper cap),
not ordinary lower-bracket interpolation. The response uses five-point-or-shorter online
linear regression, including boundary windows and extrapolation.

All **2,001 saved native probes** from −20 to 80 ns are compared for each of:

- calibrated drift radius;
- drift velocity;
- signed drift error;
- unsigned drift error.

**All 8,004 values match exactly.** The probes validate the independently generated model;
they are not used to generate its tables. NumPy online regression and exact probe agreement
do not certify universal GSL/compiler parity or unmeasured intermediate values.

## All initial state decisions agree

Every arm includes **147 targets / 2,538 hits**, with zero exclusions:

| Declared arithmetic arm | Native state/active/TOT/Long/AbsDrift mismatches | Inversions |
|---|---:|---:|
| Existing lower-storage Cholesky-factor diagnostic | **0** | 5,076 successful factors, 0 fallback |
| NumPy LU round trip | **0** | 5,076 |
| Ideal zero-weight identity control | **0** | 0 |

The ideal control is not the native rounding path. The other two are sensitivity
implementations, not ROOT InvertChol/Bunch–Kaufman parity. The same measured hits under
three algorithms are not three independent native populations.

CAD applies the source's strict cuts: calibrated drift radius <50 mm, |DOCA| <400 mm,
and 0 < DOCA variance <800000² mm². Every reconstructed hit becomes active/null with
TOT and LongVal set, AbsDrift unset. Initial configuration and captured decisions are not
used to fit corrections or select successful lanes.

For the Cholesky arm:

| Quantity | Reconstructed range / nearest margin |
|---|---|
| Delta time | −10.963689799624262 to 49.59998348987938 ns |
| Calibrated drift radius | −0.009936921692151879 to 2.9007457138418804 mm |
| Signed DOCA | −9.956462280545548 to 9.989192934799004 mm |
| DOCA variance | 102.22377937605282 to 330402.941407015 mm² |
| Margin below drift-radius limit | **47.09925428615812 mm** |
| Margin below absolute-DOCA limit | **390.010807065201 mm** |
| Margin below variance ceiling | **639999669597.0586 mm²** |

The Cholesky round trip changes 5,859/15,228 q components versus the supplied reference
(maximum 3.637978807091713e-12), while covariance returns exactly. LU changes 1,617 q
components (maximum 2.2737367544323206e-13) and 2,538 covariance components (maximum
5.421010862427522e-20). These are **roundtrip-versus-reference comparisons**, not errors
against a captured unbiased result. All three arms select the same states, far from cuts.

## Evidence boundary and next work

**The native unbiased q/C, unbiased CA and per-hit DriftInfo were not captured.** State
outputs are compared, but intermediate parity is untested. The sample contains only the
initial active/null branch; it does not test inactive, unusable, drift-signed or later-history
policy. Synthetic boundary tests are not a replacement for those populations.

Thus this is independently reconstructed **initial decision agreement with large measured
margins**, not exact native state-policy or full-fit closure. Prior residual/weight agreement
is no longer contingent on simply reading captured states, but the captured seed/reference
and native-arithmetic boundary remain. Next target is later-history retained CA: native
piece moves, hit-z retry, fallback and capped-pointer/retained-geometry semantics, using the
new sidecar where it observes the result and stating where hints/history remain absent.
Extension construction, saved-F1 chart sensitivity, strict filter algebra and final
acceptance/fidelity remain open. No acceptance gain is implied by this diagnostic.

Subsequent [adjacent retained-hit CA audit](f1-later-ca-history.md) reconstructs
31,600 same-meta updates and accounts for the 246 z retries, with explicit limits
on crossing vectors and meta/extension transitions. This initial-CAD report retains
its original scope and immutable evidence.

## Immutable evidence and tests

Data root: `$M2FS_DATA/trk_f1native_r2_20261027/`.

- **`initial_cad_v4.json`**, SHA256
  `37c3c844dd3339ab2ddca6208485f2d0ee190adbb8b4415a0e91ae8de78d9afb`.
- `initial_cad_v3.json` retained, SHA256
  `0e6484208c80848d2481432b7e69f780ced0fb93bab9f6619ab94c2ee6923b74`.
  V3 already passed all criteria; 136 velocity probes differed by at most 6.94e-18.
  V4 uses the source divide/multiply order, producing exact velocity probes. No ULP search,
  calibration fitting or tolerance change.
- Failed v1 (`unsupported geometry/field configuration`) retained: initial parser incorrectly
  included GlobalConstantsService's unrelated `inputFile`. Fixed by scoping GeometryService.
- Failed v2 (`nonzero captured nominal field`) retained: an incorrect zero-bnom guard confused
  nominal seed direction with physical field. Corrected from source and checked against capture.

The pinned residual report and every inherited input/source hash are rechecked; actual
executing inherited module paths/hashes must match. New source hashes cover lifecycle,
ROOT zero initialization, drift generation, typed configuration, geometry and conditions.
Expanded-FCL/source hash continuity remains **prior-audit continuity**, not an original
capture-generation hash receipt. Old helpers/captures are unchanged.

Review strengthened quadratic mask/shape checks and clarified reconstructed-policy metadata;
synthetic tests demonstrate captured decisions/weights cannot influence generated CAD results.
New tests: **194 drift + 163 audit + 15 population = 372 passed**.
Selected isolated suite: **1,612 passed, 1 skipped, 39/39 files**, pinned core 233771b/ANN;
log kept locally, not in the repository.
No full/slow-tier claim.

```bash
bash scripts/run_f1_initial_cad.sh \
  $M2FS_DATA/trk_f1native_r2_20261027/NEW_REPORT.json
```

Wrapper pins one CPU and single-thread libraries; exclusive JSON on completion or failure. Data usage
141543221373/161061273600 bytes (~18.18 GiB free). Fitter SHA256 remains
`988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a`.
