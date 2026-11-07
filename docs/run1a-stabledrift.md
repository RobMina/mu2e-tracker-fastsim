# Run 1A analytic-derivative fit candidate

This separate **opt-in numerical candidate** follows the [finite-difference diagnosis](run1a-drift-diagnostic.md).
It is not a replacement for stock CentralHelix, a tracking efficiency, a momentum-resolution result,
or a demonstrated reconstruction speedup. The original `helixdriftfit` and its saved outputs remain
unchanged. No material, field-map propagation, covariance, outlier removal, LR updates or direction
assignment is introduced.

## What changes, and what does not

`reco.driftjacobian` differentiates the existing conditional residual analytically within the
current branches. It includes stable local helix position/tangent partials, implicit stationary
POCA derivatives, signal propagation, measured-curvature beta/TOF, signed DOCA, unnormalized
Lorentz-angle geometry, native bilinear T2D interpolation, local offset regression, and signed-error
normalization derivatives. Stock `driftVelocity` is not substituted for this full derivative.

For the positive-z TOF derivative, the equivalent algebraic expression is

```
TOF = (z-zref)/c * sqrt(1 + tx^2 + ty^2 + (mass*omega/(0.299792458*abs(Bz)))^2)
```

The objective values still come from the unchanged evaluator, not that algebraic rewrite. No
response smoothing, clipping of negative drift radii, sigma floors or truth quantities enter.
The analytic derivative describes the stationary local POCA root evaluated at the returned
approximate root, **not the discrete stopping/root-selection algorithm**. The existing1e-7 mm POCA
tolerance is not a uniform derivative-error bound. Independent tests use scalar Brent roots and
quadrature references; production retains the same POCA policy.

`reco.stabledriftfit` retains the baseline40 outer-update budget, scales, damping and full-rank
condition policy. Each update now tests up to8 fixed half-step backtracks, accepting the first
all-hit-valid strict decrease of the actual objective. Trials may cross calibration branches;
branches are never smoothed. Damping grows by10 on rejection and decreases by3 on acceptance.
Forty updates with this search are **not equal compute** to40 baseline updates.

All candidates start from the **same original measured-wire parameters and profiled t0**, with
saved initial LR/scales and identical support. They do not start from old successful final fits.
Both flight directions and all original failure/empty lanes remain represented.

## Meaning of a passing candidate

The output status `converged_certified` means only **analytic stationary, adjacent-stencil checked**:

1. Valid full support, rank6 and equilibrated normal-matrix condition <=1e12.
2. Analytic normalized gradient <= `1e-6*(1+sqrt(2*objective))`, the unchanged baseline threshold.
3. No guarded native knot, nonidentical regression boundary, zero-DOCA or geometric angle fold
   at the center. Arithmetic-scaled boundary slack accounts for absolute coordinates/time operands;
   it is not a new physical hit cut. Identical neighboring regressions are not artificial branches.
4. At least two adjacent stencils from `[1e-3,1e-4,1e-5,1e-6]` pass independently. Every +/- probe
   is valid, with matching time/phi cells, regression coefficients, signed DOCA and unfolded angle
   signs. Endpoint branch identity is conservative evidence, not a proof about the whole path.
5. Each FD gradient passes the same threshold in its own and the **common analytic column norm**;
   analytic/FD gradient disagreement is <=one quarter of that threshold. Each Jacobian column must
   agree within relative L2 error1e-3. The latter prevents residual-orthogonal derivative errors
   from hiding behind a small gradient. These are declared engineering consistency policies,
   not fitted uncertainties or calibrated statistical tests.

All stencil outcomes, column discrepancies and the first passing adjacent pair are retained.
This is not an independent exact derivative, local-minimum proof, or physics validation. The
smallest stencil is not privileged; it often amplifies POCA stopping/roundoff effects. Numerical
termination (`analytic_stationary`, `max_iterations`, `stalled`) and final consistency status are
separate fields. Failed fits retain the last accepted finite parameters and no physical momentum.

## Same-original-seed development1000 comparison

Input census is unchanged:1000 events,235 empty events,13492 hits,773 groups,344 original seeds,
5427 selected-hit identities;255 eligible wire/profile groups give510 direction hypotheses.
Earlier89 invalid wire seeds and429 empty group lanes remain in each direction.

| Numerical criterion | Saved baseline final states | New candidate |
|---|---:|---:|
| Original baseline criterion (historical labels only) | 195 | Not used for acceptance |
| Identical analytic + adjacent-stencil check | **20** | **307** |
| Groups with at least one passing direction under that check | 20 | 213 |
| Groups with both directions passing | 0 | 94 |

The baseline's20 passes comprise13 originally labelled converged and7 originally stalled. The
same criterion is applied **read-only** to saved baseline finals; their original statuses are not
changed. It is therefore misleading to compare307 directly with195 as though the criteria matched.
Neither20 nor307 is a tracking-efficiency numerator.

Candidate pass split:151 negative-z and156 positive-z; among255 eligible groups,213 have any,
94 both and42 neither. Remaining203 hypotheses stay failures:

-134 analytically stationary but fail adjacent-stencil checks:102 consistency failures and32
 with no admissible adjacent pair;
-48 at guarded nondifferentiable/boundary states;
-17 still smooth/nonstationary after the iteration budget;
-4 still smooth/nonstationary after damping stalls.

The raw termination census is441 analytic-stationary,48 budget exhausted and21 damping stalled;
the boundary classification overlaps those termination reasons. Full transitions are in the report.
Of282 old stalls,153 now pass;13 of33 old budget failures pass. Of195 old nominal successes,141
pass and54 fail the new checks. No status is repaired by deleting hits or choosing a direction.

All510 candidate objectives decrease from their original seeds. Relative to baseline final values,
411 are lower and99 higher; independent trajectories can reach different local branches. Median
candidate-minus-baseline objective is-9.99e-9, minimum-2.828, maximum+0.01579. Most changes are tiny:
this result primarily improves **numerical stationarity evidence**, not demonstrated physics quality.
No residual or momentum-resolution improvement is claimed.

There are6212 logical analytic checks,19392 logical trial attempts and22176 eligible-row certificate
attempts, with3887 accepted updates. All tested optimization trials are valid in this run;947 have
unchanged floating-point parameters. These counts are not total batched operations or throughput.
The final POCA correction magnitude across valid hit-hypothesis entries has median1.37e-13 mm,
p906.48e-9 mm and maximum9.998e-8 mm. Same-branch Jacobian column discrepancies remain small
(max4.20e-5 among compatible probes), yet gradient disagreement can exceed the stringent stopping
threshold; such rows remain failures rather than lowering the standard.

## Reproduction and validation

```bash
python3 -m trkfastsim.validation.run1a_stabledrift \
  --saved SAVED_DEFAULT40_JOINT_DIRECTORY --outdir NEW_CANDIDATE_DIRECTORY
python3 -m trkfastsim.validation.run1a_stabledrift \
  --saved SAVED_DEFAULT40_JOINT_DIRECTORY --group-batch 17 --outdir NEW_REPEAT_DIRECTORY \
  --compare-saved NEW_CANDIDATE_DIRECTORY/diagnostics.npz
```

The driver checks the prior input hashes, reference endpoints/configuration, oracle/IOV, original
seeds/LR/scales and initial objectives. It replays old final objectives/default gradients and runs
both old-final consistency checks and new fits. All baseline NPZ arrays are copied unchanged;
current implementation hashes and historical hashes remain separate. These checks cannot restore
the missing original association-input or prior output-NPZ hash binding.

Final cache `stabledrift-1000-20261106-final` is array-exact versus
`stabledrift-1000-20261106-2` and `stabledrift-1000-batch17-20261106`, including every trace,
failure, per-hit diagnostic and baseline check. The first candidate run is preserved separately;
it predates the added read-only baseline consistency arrays, with the same307 candidate passes.
Final default candidate processing31.85 s; separate old-final diagnostic replay4.74 s,
old-final adjacent-stencil check4.31 s, load/oracle0.17 s, output writing excluded. Other measured
candidate runs30.26/44.72/30.53 s illustrate laptop variability. No stock/GPU speedup claim.

Focused validation: **1010 passes across19 files**, warnings treated as errors, including63 analytic
Jacobian,24 candidate kernel and51 candidate-driver tests. Tests include independent scalar
calibration/Brent-root geometry, trajectory quadrature, nonconstant-error normalization, noisy
SciPy fits, knots/jumps/folds, adjacent-pair requirements, rejection traces, full empty/unselected
censuses, exact singleton and campaign batches, and provenance/tamper checks. Reviews caught invalid
singleton sentinel differences and per-hit POCA mapping; both are fixed. No CI or C++ build.

Follow-on [read-only failure study](run1a-drift-failure.md) identifies45 offset-switch and3 native-time
boundary states. Diagnostic POCA polishing changes27 consistency failures to a diagnostic pass but
also loses16 existing passes; all saved labels stay unchanged. The32 no-adjacent-pair cases persist.
No criterion was relaxed and no LR update applied in that study. A separate [LR continuation](run1a-lrdrift.md)
now starts from these saved finals:90→336 under the combined continuous+LR check,384 objectives lower,
126 unchanged; all historical arrays are preserved. It adds compute and is not an original-seed A/B.
Independent root/arithmetic checks, branch-aware handling, flight-direction calibration, multiple
tracks/outliers, material/field-map propagation, covariance and saved-stock CentralHelix comparisons
remain open. Shared digitiser optimization is still deferred.
