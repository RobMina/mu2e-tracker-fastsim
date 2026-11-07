# Run 1A joint-fit numerical diagnosis

This is a **read-only final-state investigation**, not a new fitter or an improvement in tracking
performance. `reco.driftfitdiagnostic` and `validation.run1a_drift_diagnostic` use the saved
[default40 joint fit](run1a-timing.md), frozen selected hits, original LR signs, and original parameter
scales. Neither solver settings nor recorded statuses change. No wire fit, association search,
simulation, digitisation or heldout tuning is repeated.

## What is evaluated

The diagnostic refinds the same local POCA and recomputes measured propagation, signed TOF, Lorentz
angle, drift radius and signed error. It replays the saved objective, residuals and default gradient
on all510 attempted direction hypotheses. Invalid seed/empty lanes stay present. All original NPZ
arrays are copied unchanged, including failures,1000 events,235 empty events,773 groups,13492 hits
and5427 original selected-hit identities.

Four dimensionless central-difference half-steps are tested: `1e-3,1e-4,1e-5,1e-6`, multiplying the
**saved initial parameter scales**. The first is the production setting; the others are diagnostic,
not replacements. The actual normalized residual is differentiated, including refound POCA and
calibration. Each stencil records both one-sided objective slopes and validity, native time/angle
cells, offset/error regression bins, coefficient changes, and signed-DOCA flips. Counts are
hit/perturbation incidences; a bin crossing alone is not evidence that it caused a stall.

For the first and last stencils, the normalized diagonally preconditioned negative gradient is
probed at dimensionless steps `1e-2..1e-8` in decade increments. Saved deltas are `Ftrial-Fcenter`;
no trial is accepted. The report also screens decreases against `128*epsilon*max(1,abs(Fcenter))`.
That is a **heuristic arithmetic screen**, not a proven bound on POCA/calibration numerical error.
Absence of sampled descent is not a proof of local optimality. The smallest stencil is not presumed
correct merely because it is smaller.

The existing stock response is unchanged. Offset calibration has140 bins of width0.025 mm and
adjacent local regressions do not generally join: absolute boundary jumps have median0.000045 mm,
p900.00020181 mm and maximum0.000870 mm. Native time/angle interpolation is continuous, with
0.05 ns and pi/38 spacing, but has derivative kinks. Signed errors are constant0.16 mm in this
configuration, so error-regression jumps cannot explain these results. Tests distinguish a bin
change with identical coefficients, distinct coefficients that join continuously, and an actual
response discontinuity.

## Saved development1000 evidence

All510 final default linearizations have rank6. Stalled rows have median equilibrated condition46.35
and maximum22870, far below the1e12 limit. They are not final rank/condition failures. Tags below
**overlap** and are not a causal partition:

| Final-state observation | Recorded converged | Stalled | Budget exhausted |
|---|---:|---:|---:|
| Hypotheses | 195 | 282 | 33 |
| Default stencil changes offset-regression coefficients | 32 | 231 | 27 |
| Default stencil crosses native time cells | 193 | 279 | 31 |
| Smallest stencil crosses no recorded branch | 195 | 273 | 14 |
| Sampled default-gradient decrease above arithmetic screen | 119 | 25 | 20 |
| Sampled smallest-gradient decrease above arithmetic screen | 193 | 271 | 15 |
| Pass same gradient threshold at smallest stencil | 13 | 8 | 0 |
| Neither sampled direction gives a decrease above screen | 2 | 11 | 7 |

Of282 stalls,246 show a screened decrease using the smaller-stencil direction but not the default
one;267 show smaller-stencil descent while the smallest derivative stencil crosses no recorded
branch. This establishes substantial **finite-difference scale sensitivity** and available sampled
improvement, not that every stalled row is a calibration-boundary minimum. The median maximum
sampled gain among the271 improving stalled rows is6.37e-7 objective units, p904.60e-5, max0.0176.
Terminal iteration history was not recorded by the original fitter: invalid trial sequences,
arithmetic stagnation at high damping, and POCA stopping effects are not fully separated here.

**The195 earlier convergence labels also require qualification.** Only13 pass the same gradient
threshold at the smallest stencil;182 do not. Median gradient/threshold changes from0.240 to9.78.
The last two stencils differ by median0.233 threshold units, so a smaller-stencil result is not an
independent exact derivative certificate.193 have sampled decreases, but their median best gain is
only2.65e-9 objective units (p908.15e-7,max0.00515). Tiny decreases alone do not contradict a finite
convergence tolerance. Preserve all195 recorded labels, but do not describe them as stencil-stable
stationarity certificates or physics-validated reconstructed tracks.

## Measured left/right evidence

At fixed geometry/time, the lower-cost sign for the existing squared residual is
`sign(r_drift*signed_doca)`, retaining the original LR on an exact tie. In particular, negative
calibrated radii must **not** be absolutized or forced to the geometrical sign. The old-minus-flipped
per-hit objective difference is `-2*initial_LR*r_drift*signed_doca/signed_error**2`.

This diagnostic finds beneficial flip candidates in135 of195 recorded-converged hypotheses,
222 of282 stalled, and27 of33 exhausted. The respective counts are286,453 and76 **hit-hypothesis
entries**, not distinct hits; the same source hit can appear in both direction hypotheses. These
are pointwise conditional improvements, not accepted LR changes or an ambiguity solution. Support,
LR and direction outputs remain unchanged. The255 eligible groups still split36 with both directions
converged,123 with one and96 with neither. Single-direction convergence is not direction evidence.

## Reproduction and verification

```bash
python3 -m trkfastsim.validation.run1a_drift_diagnostic \
  --saved SAVED_JOINT_DIRECTORY --outdir NEW_DIAGNOSTIC_DIRECTORY
python3 -m trkfastsim.validation.run1a_drift_diagnostic \
  --saved SAVED_JOINT_DIRECTORY --group-batch 17 --outdir NEW_REPEAT_DIRECTORY \
  --compare-saved NEW_DIAGNOSTIC_DIRECTORY/diagnostics.npz
```

The original report supplies paths for measured inputs, conditions/reference endpoints and the
validated oracle. Current data hashes must match that report and the wire-fit input hashes. Service
configuration/IOV/oracle checks are repeated; original timing geometry/LR and parameter scales are
replayed. The diagnostic hashes its input NPZ and current implementation separately from historical
implementation hashes; it does not silently rewrite the original report after comment-only edits.
Original association-input and saved-output hash-binding gaps remain explicitly acknowledged.

The final cache is `drift-diagnostic-1000-20261106-final`; it is array-exact versus the initial run,
second default run and `drift-diagnostic-1000-batch17-20261106`. All original joint arrays survive
unchanged. Final default processing was5.58 s plus0.19 s load/oracle checks, excluding output writing;
this is laptop diagnostic cost, not reconstruction throughput or a GPU/stock speedup.

Focused validation: **872 passes across16 files**, warnings treated as errors;44 diagnostic-kernel
and35 diagnostic-driver tests. New tests include independent affine-response analytic t0 slopes and
normalized gradients, negative-radius LR arithmetic, frozen support/no mutation, invalid probes,
continuous/discontinuous branch distinctions, exact batches, all-empty/zero-group censuses, and
input/replay tampering. Reviews prompted complete failure-category reporting, hit-validity guards
and stronger initial-LR replay. No CI or C++ rebuild.

## Next numerical step

A separate [analytic candidate](run1a-stabledrift.md) now implements branch derivatives and adjacent-FD
checks:307/510 passes versus20 saved baseline finals under the identical new criterion;203 failures remain.
The original baseline/defaults and this diagnostic output are unchanged. Continue with the remaining
consistency/boundary failures, not a globally smaller FD step or relaxed gradient threshold.
A proposed adaptive derivative or trial strategy needs independent smooth/kink/jump tests and a
separate measured-only comparison with this unchanged baseline. Measured LR candidate updates can
then be tested with monotone full-objective checks and explicit failure states; no score-ordering
shortcut assigns flight direction. Material, field-map propagation, covariance, multiple tracks,
outliers and saved-stock CentralHelix validation remain open; digitiser optimisation stays deferred.
