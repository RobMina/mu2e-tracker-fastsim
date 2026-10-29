# Saved first-iteration measurement-row audit

`scripts/f1_measurement_audit.py` tests derivative-policy differences against actual
saved F1 operands and captured native information weights. It never reruns a fit or
changes the fitter. Component criteria remain `rtol=1e-5, atol=1e-7`; full closure is
always false. This is separate from native-input algebra replay.

## Source formulas and conventions

`nativemeasurement.py` implements the frozen formulas from
`KinematicLine::{dXdPar,dMdPar}`, `ClosestApproach::findTCA`,
`KKStrawHit::setResiduals` and `Residual::weight`. Let `s,u` be unit sensor/particle
directions, `c=s.u`, `D=1-c²`, `delta=sensorPOCA-particlePOCA`, `X=dx/dq` at fixed
particle time, and `U=du/dq`. Native coordinates are
`q=(d0,phi0,z0,theta,t0,momentum)`, with the full momentum-dependent velocity.
The native line's nominal field does not rotate this parameterization.

F1 residual convention is `r+H*(q-qref)`, while native `dRdP` is a prediction
derivative: `H=-dRdP`. A null native DOCA row has both value and derivative opposite
to F1's `-doca` row; flipping both preserves its information matrix and vector.
The audit currently requires first-iteration null hits and frozen drift radius.

At exact analytic line-line CA, differentiating the two orthogonality conditions gives:

```
H_time(full) - H_time(native) = (c/vsensor - 1/vparticle) / D * (delta.U)
H_long(full) - H_long(native) = 2*c / D * (delta.U)
```

The first term corresponds to the direction-change dependence explicitly omitted
in native `ClosestApproach`. The second follows from the minus `delta.dMdPar/p`
term in the frozen native longitudinal formula. Neither formula alone establishes
how final fit decisions change. Independent tests solve CA as a linear system and
use autodiff for its exact derivatives, separately checking the source formulas
and both difference terms. Both sensor signs, several momenta, nonplanar wire
orientations, vanishing coefficients and near-zero DOCA are covered.

## Isolation and evidence guards

Five variants hold saved references, residual values, weights and active-row masks
fixed: saved rows, time-only replacement, long-only replacement, both, and all
source rows. Each comparison checks both `W=H.T diag(w) H` and
`b=H.T diag(w) (H qref-r)`. Per-hit residual maxima and source-hit identities are
retained. The constraint is excluded from these hit-only denominators; the initial
first-iteration comparison still checks it.

The driver verifies the four-hash native gate, saved trace hash/passivity, full
identities, initial membership/references/activity/priors, input hashes and relevant
source hashes. It reconstructs the F1 hit rows from the original hashed inputs and
checks them against saved rows before substitution. Reconstructed analytic CA time
is compared with captured native hit time, without claiming native CA-state equality.

By default, any changed loader source is refused. `--allow-loader-reconstruction`
permits only an input-loader source mismatch, explicitly recording both hashes and
requiring exact saved identities/midpoints/membership and strict row agreement.
This is an explicit reconstruction caveat, not source equality; other source guards
cannot be waived. Historical gain traces require it after loader changes.

```bash
bash scripts/run_f1_measurement_audit.sh NEW_OUTPUT.json \
  --trace-dir SAVED_GAIN_TRACE --trace-dir SAVED_LOSS_TRACE \
  --allow-loader-reconstruction
```

The shell wrapper pins one CPU and single-threaded CPU JAX. Native-source hashes and audit sources accompany the JSON;
outputs are exclusive. Saved traces, native capture and previous comparisons are never modified.
Malformed/inconsistent inputs fail rather than produce a result.

## DOCA conditioning limitation

The source formula uses `delta/|delta|`. Reconstructing position from large absolute
times can leave small nonorthogonality; dividing it by very small DOCA amplifies it.
An additional, explicitly non-native arithmetic diagnostic uses the analytic normal
`(s cross u)/sqrt(D)`. On the measured targets this reproduces saved F1 DOCA rows at
the existing tolerance, whereas direct delta normalization need not. This does not
recover the native iterative CA's retained vector or prove the cause of its error.

Two saved native-seed targets show partial improvement when only time/long rows
change: matrix failures 291→107 of 900 hit components on the gain, 171→23 of 468 on
the loss; vector failures 85→56 of 150 and 48→10 of 78. The loss's remaining failures
are confined to two near-zero-DOCA hits. The gain still has failures on 22/25 hits;
its largest matrix/vector residuals remain essentially unchanged. These are bounded
operand diagnostics, not measurement closure, all-target attribution, candidate or
acceptance closure. Material noise and later-history comparison remain separate.

The subsequent [CA/material audit](f1-ca-material-operands.md) explicitly tests
native mass, local time-update arithmetic, ULP sensitivity, per-material scattering
fractions and the material-Jacobian evaluation point. Its bounded results do not
replace or retroactively alter this row-only isolation.
