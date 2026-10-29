# Initial residual and measurement-variance reconstruction

The next boundary after [initial CA reconstruction](f1-captured-ca-operands.md) is tested by
`scripts/f1_initial_residual_audit.py` and `f1_initial_measurements.py`. No fitter/default,
C++ or tolerance changes, refits, new native jobs, commits or production promotion.

Subsequent [initial CAD reconstruction](f1-initial-cad-state.md) independently reproduces
all initial state decisions in three declared arithmetic arms and validates source drift
calibration. Native unbiased intermediate parity remains unobserved; this page records
the earlier conditional-null residual audit.

## Result: all 147 targets, 2,538 initial hits

Both predeclared arms retain every hit and compare against the validated passive sidecar:

1. Independently dumped ComboHit `dtime`, `wdist`, `timevar`, `uvar` measurements.
2. Source-FCL reconstruction of those quantities from measured ComboHit energy, endpoint
   times and TOT, plus independently dumped geometry. This arm does not consume the first
   arm's derived measurements, or sidecar residual values/variances, to generate operands.

| Quantity | Components | Exact mismatches | Tolerance failures |
|---|---:|---:|---:|
| Native residual values | 7,614 | **0** | **0** |
| Measurement variances | 7,614 | **0** | **0** |
| Annealing scale | 2,538 | **0** | **0** |
| Inverse measurement variances | 7,614 | **0** | **0** |
| W | 91,368 | 11,672 | **0** |
| b | 15,228 | 2,341 | **0** |

Results are identical for both arms. All regenerated `dtime`, `wdist`, `timevar`, `uvar`,
float32 straw directions and early-end choices exactly equal the independently dumped
measurements. All generated CA vectors and 45,684 H components remain exact. W/b maximum
absolute differences are **1.4551915228366852e-11 / 4.656612873077393e-10**, at unchanged
**rtol=1e-5, atol=1e-7**. These are tolerance-level W/b results, not bitwise agreement.
Exact means finite elementwise numeric equality on this sample, not universal compiler parity.

## Native arithmetic and policy

The initial `FitSettings` has CAD settings `[400,800000,50,50,"TOT:LongVal","Null:Inactive","",0]`
and first temperature **5**, stored through the native float configuration type. Thus the
variance multiplier is `(1+5)*(1+5)=36`. The prescribed null branch has no `NullDriftVar` or
absolute-drift flag: `DriftInfo::maxdvar_ = 2.4*2.4/3.0 = 1.92 mm²`. Measurement denominators
never include parameter variance.

Regeneration follows `StrawHitRecoUtils::createComboHit`, `StrawResponse::PieceLine`,
`wireDistance`, `wpRes`, and TOT lookup:

- Float endpoint subtraction occurs **before** promotion to double (`TDCTimes` is float).
- Half signal velocity uses original source coefficients; interpolation is slope-first.
- Wire resolution applies central/slope dependence, the 30-mm floor, and configured
  longitudinal truncation before storing wire distance and squared resolution as float32.
- Earlier end is HV only for `et1 < et0`; ties choose cal. Table indexing uses promoted
  measured floats and source double bin widths, with upper saturation.
- Drift time subtracts 0.1 ns before float32 storage; time variance squares the double
  table error before float32 storage. Straw direction is float32, not re-normalized.

Native residual order is distance, time, longitudinal. For the null branch,
`v=(doca, sensorTime-particleTime-driftTime, calong +/- wireDist)`; the longitudinal sign
comes from the sensor/ComboHit direction dot product. F1 convention remains
`r=diag(-1,+1,+1)*v`, `H=-diag(-1,+1,+1)*g`.

## Deliberate remaining boundary: hit-state selection

**The null branch is prescribed from the source configuration and checked against capture;
it is not independently selected.** `KKStrawHit::updateWHS` calls `unbiasedClosestApproach`,
which passes through parameter-to-weight-to-parameter inversion, even before subtracting a
nonzero hit weight. CAD then tests unbiased DOCA, its variance, and calibrated drift radius.
This audit does not reconstruct those unbiased operands or the CAD eligibility decision.

Captured seed/reference parameters and mass remain inputs. Measured energy/end-times/TOT
are inputs rather than regenerated electronics. Initial agreement does not close later
non-null states, CA retry/fallback/capped-piece semantics, extensions/private offsets, saved
F1-reference roundtrips, strict native filter algebra, missing gain/loss coverage or final
acceptance. Existing 95.594% acceptance and 89.808% momentum fidelity are not improved by
this diagnostic; no fitter execution changed.

## Evidence, integrity and tests

Remote data root: `/exp/mu2e/data/users/ram2aq/trk_f1native_r2_20261027/`.

- Final **`initial_residuals_v2.json`** SHA256
  `1539f808dd7aeab96042dac29023aed33e6688349686d6abb2c804a305bd3219`.
- Initial `initial_residuals_v1.json` retained, SHA256
  `721d4e3f38875ba2ca03ba32b17e40bf749ff00b9a492a622fbf9423c22ea69a`.
- V2 population content is exactly equal to v1. V2 adds review hardening: true-mask
  rejection, executing-module path/hash binding, and preserving declared hashes even
  when receipt validation fails. No numerical or tolerance change.
- The prior `captured_ca_population_v2.json` receipt is hard-pinned and its entire source/input
  manifest rechecked. All original scripts and native captures remain untouched. Source-FCL
  hash continuity is to the earlier audit, not a retrospective capture-generation receipt.
- Executing inherited helper paths must equal the receipt's paths, preventing a different
  checkout from silently supplying unverified implementation code.
- New native source hashes include residual/policy definitions, typed ComboHit/TDCTimes,
  measurement reconstruction and independent dumper source; all are rehashed after analysis.

New tests: **128 measurement tests + 142 residual/provenance tests = 270 passed**.
Final selected isolated regression: **1,240 passed, 1 skipped, 36/36 files** with pinned core
`233771b` and ANN search path. Log:
`~/mu2e/local_data/trk_f1native_r2_20261027/isolated_initial_residuals_v2.txt`.
No full/slow-tier claim. Early synthetic failures exposed an empty-list `np.all` guard and
masked measured lanes; fixed before the first population run. Later review guards were
fixed and the population rerun to a new filename.

```bash
bash scripts/run_f1_initial_residuals.sh \
  /exp/mu2e/data/users/ram2aq/trk_f1native_r2_20261027/NEW_REPORT.json
```

Wrapper pins one CPU and single-thread numerical libraries; outputs are exclusive. Next: independent unbiased
CA/DriftInfo/CAD state selection.
