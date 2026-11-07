# Run 1A saved analytic-candidate failure diagnosis

This read-only study follows [the analytic candidate](run1a-stabledrift.md). It does **not**
refit a track, change LR/support/direction, promote a failed row, or replace the production POCA.
The saved candidate still has307 checked hypotheses and203 failures out of510 attempted directions.
No acceptance threshold, calibration, baseline array or historical report is changed.

## Controlled numerical experiment

`reco.driftfailure.diagnose` evaluates every saved final state, including failed and empty lanes.
The original evaluator, analytic Jacobian and all certificate fields are replayed. A second,
explicitly diagnostic evaluator starts from the original POCA at every center and FD probe and
tries exactly three Newton polishing rounds. Each accepted root must reduce absolute g/H,
retain positive Hessian, stock stationarity and finite-wire/timing validity, remain in the stock
basin, and stay within four original corrections plus arithmetic slack of the original root.
An originally invalid row cannot be rescued. All downstream timing, response and normalized
residuals are recomputed. Zero-round reconstruction is tested array-exact against the original
state; it is not merely a pass-through for valid rows.

Polishing changes numerical evaluation, not fit parameters. It is neither an independent exact
root reference nor a new fitter. Arithmetic-floor flags and g/H are evidence, not rigorous error
bounds. Independent scalar Brent-root/calibration and noisy SciPy tests check the implementation;
no independent-root campaign over real data has yet been run.

The boundary inventory reproduces the same64-epsilon arithmetic guards as the analytic module,
including absolute time operands and coordinate/DOCA angle scales. It checks that those flags
explain the analytic differentiability mask exactly. Native knots remain conservatively guarded;
this does not prove every flagged composed objective is mathematically nonsmooth. The nearest
internal offset edge includes separate coefficient, value-jump and slope-jump evidence.

`reco.driftprobecompare` retains all four original stencils and six coordinates. It separates the
signed normalized gradient discrepancy using one common original residual R and column norm:

```
(J_FD - J_analytic) R = (J_FD - J_polished_FD) R
                       + (J_polished_FD - J_polished_analytic) R
                       + (J_polished_analytic - J_analytic) R
```

The middle term is a remainder, not error against truth. Center/probe branch identity, realized
parameter increments and t0-root invariance are recorded. Endpoint matching is not a path proof.
Native time, phi, offset, signed-error, signed-DOCA and unfolded-angle changes are separate flags.
No blanket smaller stencil, smoothing, sigma floor or relaxed success criterion is used.

## Saved development1000 results

The census remains1000 events,235 empty events,13492 hits,773 groups,344 original seeds and5427
fixed selected hits. There are255 eligible groups/510 direction hypotheses. All historical
candidate and baseline arrays are copied unchanged. The original final-state checks replay exactly.

| Saved candidate category | Hypotheses | Pass after diagnostic polishing only |
|---|---:|---:|
| Already checked |307|291|
| Derivative unstable |134|27|
| Guarded boundary |48|0|
| Budget exhausted |17|0|
| Stalled |4|0|

Thus27 failures change to a diagnostic pass while16 existing passes lose it. **Neither side is
relabelled.** The diagnostic total318 is not a new reconstruction result. Tighter POCA evaluation
alone is not a demonstrated robust fix.

Of134 derivative-unstable states:

-102 originally fail derivative consistency:27 pass the polished diagnostic,72 still fail the FD
 check and3 become analytically nonstationary.
-32 have no admissible adjacent stencil pair, and all32 retain that result after polishing.
-At scales1e-3/1e-4/1e-5/1e-6, compatible rows are0/11/102/131 in both evaluations. Time-cell
 changes occur in134/114/29/3 rows and offset-regression changes in77/19/3/0; these tags overlap.
 Among the32 no-pair rows, time changes occur in32/30/29/3 and offset changes in20/5/3/0.
-Original compatible-stencil gradient disagreement divided by the original threshold has
 median0.277, p900.890 and max23.533. With polishing, median0.144, p900.729 and max2.987.
 This establishes sensitivity to root evaluation, not that all remaining discrepancies are roundoff.
-For same-center/same-probe-branch coordinate checks, the signed decomposition closes to
 <9e-15 of the stopping threshold. Probe-root contributions can be large, but cancellation and
 center derivative changes matter; maxima of separate components need not belong to the same row.
-Pure t0 probes leave original and polished roots exactly unchanged. Their original gradient
 discrepancy is at most6.76e-5 of threshold in compatible checks, a useful control against blaming
 all six columns on perturbation-dependent POCA stopping.

The48 guarded states comprise **45 offset-regression boundaries and3 native time knots**, with
one flagged hit-hypothesis entry per row. No phi, signed-error, geometric-fold or zero-DOCA guard
is implicated in these centers. All45 nearest offset switches have nonzero value jumps:
absolute jump median6.46e-5 mm, range8.85e-6 to2.428e-4 mm. Distances to those edges range
4.04e-15 to8.64e-11 mm; this is guarded proximity, not an equality-to-edge assertion.
Polishing changes center branch in5 of48;46 remain guarded and2 become smooth but nonstationary.
The largest boundary-row objective change is+0.004777. No boundary state is declared a minimum.

For the134 consistency-failure rows, final absolute POCA correction p90 decreases from8.95e-9 mm
to7.57e-14 mm, maximum from9.75e-8 to1.41e-13 mm. Their objective changes range-1.38e-6 to+7.22e-6,
median-1.96e-12. Small corrections/objective differences do not by themselves establish reliable
derivatives or improved physics. The17 budget and4 stall states remain explicit failures.

## Reproduction, validation and provenance

```bash
python3 -m trkfastsim.validation.run1a_drift_failure \
  --saved SAVED_DEFAULT40_JOINT_DIRECTORY --candidate SAVED_ANALYTIC_CANDIDATE_DIRECTORY \
  --outdir NEW_DIAGNOSTIC_DIRECTORY
# Repeat with --group-batch 17 --compare-saved NEW_DIAGNOSTIC_DIRECTORY/diagnostics.npz
```

Accepted cache: `driftfailure-1000-20261106-final`, array-exact against both `-2` and
`driftfailure-1000-batch17-20261106`. The earlier `-1` is retained; it predates expanded replay,
branch attribution, raw-invalid-vector preservation and missing-bin refinements. Final payload
SHA256: `97cb861ec4c9b8ddeddd50f98b38f9da2943403720fb9b11b08b27cfa480f13c`.
Final processing33.40 s, load/oracle0.192 s; batch17 processing38.72 s. Output is excluded.
These are laptop diagnostic timings, not fitting throughput or stock/GPU comparisons.

The driver binds saved baseline/input hashes, reruns the oracle/configuration/IOV checks and lists
exactly which final-state fields it replays. Objective-valid hit fields, raw linear diagnostics and
all certificate fields are checked; traces/status histories are preserved, not optimizer-replayed.
Current candidate/output hashes do not retrospectively authenticate the old unbound candidate NPZ,
original association inputs or baseline output. Historical source hashes are retained separately.
Review caught sentinel masking, omitted replay fields, invalid-vector sanitization and unselected
bin sentinels; focused regressions cover these fixes. A synthetic eigensolve-failure fixture exposed
an existing candidate-report max-float division warning; its setup alone suppresses that overflow.
The original candidate implementation remains unchanged; that reporting edge case is still open.

Focused validation: **1078 passes in22 selected files**, warnings-as-errors, including72 new tests
(37 kernel,4 decomposition,31 driver). This selection differs from the preceding1010-test selection;
no CI/full-suite claim. Real-data batch32/17 comparison includes every array and original copy.

Next: an independent bounded root/arithmetic reference for unresolved smooth failures, plus
explicit one-sided branch-aware treatment rather than smoothing calibration jumps. The separate
[measured LR continuation](run1a-lrdrift.md) now implements monotone sign blocks/refits from saved
candidate finals; no updates were applied to this read-only study. Direction calibration, multiple
tracks/outliers, material, field-map propagation, covariance and saved-stock CentralHelix
physics/throughput comparisons remain open. Digitiser optimization stays deferred.
