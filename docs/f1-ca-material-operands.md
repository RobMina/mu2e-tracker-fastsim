# Local CA and material operand isolation

These diagnostics extend the [saved measurement-row audit](f1-measurement-rows.md).
They consume immutable actual-F1 first-iteration traces and the native evidence
gate. They do not modify the fitter, rerun a track, change defaults or certify
acceptance, candidate arithmetic, later histories or all-target closure.

## Local CA reconstruction

`nativeca.py` implements a vectorized KL/SensorLine local CA reconstruction using
the coupled particle/sensor time updates in `ClosestApproach::findTCA`. The loop
is over at most 100 updates, never individual hits. It keeps converged lanes
stopped and exposes statuses, counts, retained times, positions and derivatives.
Parallel lanes are explicit; nonfinite/overflow arithmetic aborts rather than
claiming native failure-state propagation. No piece moves, capped pointer/state
split, unusable-hint fallback, hit-z retry or ROOT arithmetic equivalence is ported.

`scripts/f1_ca_operand_audit.py` uses the dumped double-precision wire endpoints
and measured ComboHit corrected times to reconstruct the constructor z-time hint,
then one reference update with the retained hints. Endpoint joins use source hit
indices and raw straw identity. Exact and tolerant particle-time comparisons are
reported separately. A reconstructed `|particlePOCA.z-wireMid.z|>100 mm` census
is diagnostic, not evidence of native retries. The captured expanded configuration
binds `TPOCAPrecision=1e-4`; native code applies that number to time updates despite
the old configuration comment calling its units mm.

Variants distinguish saved F1 reference parameters from captured native references,
and F1 muon mass (105.6583755 MeV) from captured mass (105.658367 MeV). This small
mass difference matters numerically when dividing a tiny reconstructed CA vector
by its magnitude. Passing the seed loader's mass compatibility tolerance does not
mean bit-identical velocity or derivative arithmetic. Residual values and weights
are held fixed for row-only variants; a separately labelled variant reconstructs
values too. Neither variant has captured native sensor times or delta vectors.

With native references and mass, reconstructed particle times exactly match all
25 gain/13 loss hit times in the tested first iteration. Measurement component
failures remain: gain 4/900 W and 3/150 b (one hit), loss 6/468 W and 0/78 b (two
hits). Using saved F1 references instead leaves 14 W/10 b failures on the loss.
Thus even parameter roundtrip errors below the ordinary tolerance can matter here;
the native-reference result is not a claim that F1-generated operands close.

Four fixed sensitivity diagnostics perturb particle or sensor time by one ULP,
positive and negative. Each compares only against the unperturbed reconstruction,
not against a native target or an optimized correction. They deliberately move
away from a recomputed CA. Large changes demonstrate conditioning, not the source
of the last native discrepancy or a permissible tolerance relaxation.

## All-captured-target timing census and arithmetic controls

`scripts/f1_ca_population_audit.py` extends the local CA timing check to all
**147 captured first iterations / 2,538 initial straw hits**, using vectorized
full event/seed/source-hit joins. Constructor membership, raw straw identities,
initial single-piece ownership, seed/reference parameters and mass are checked
before reconstruction. Endpoint times must both be finite, even when only one is
selected as the measurement end. All hits remain in the reported denominator.

Using captured references/mass and measured endpoint inputs, constructor and first
reference-update **particle times match exactly for every hit**. Every constructor
solve takes two updates and every reference solve one. There are no reconstructed
unusable states, zero-distance states or 100-mm z-retry triggers; the largest
absolute z distance is **9.663233981304458 mm**. The smallest reconstructed CA
distance is **4.0477564519426165e-9 mm**. These are reconstructed vectors, not
captured native sensor times or POCA vectors. This census does not expand W/b or
material agreement beyond the two saved targets, nor resolve later-iteration
retries and capped retained states. It does not cover uncaptured discordances.

`nativearith.py` supplies separate, predeclared arithmetic controls. At a fixed CA
state it follows native scalar grouping for speed, gamma, dXdPar and dMdPar/p,
reciprocal beta/gamma factors, and a separately labelled per-row information
assembly. A second sensor variant uses ROOT's Cartesian3D Unit expression:
reciprocal multiplication rather than componentwise division. The actual ROOT
include directory is taken from the frozen KinKal `compile_commands.json`; that
file and both GenVector headers are hash-bound. SMatrix/compiler/transcendental
bit parity is not certified, and no arithmetic variant is selected by optimizing
agreement with native results.

None of these controls removes the remaining failures: gain **4/900 W, 3/150 b**;
loss **6/468 W, 0/78 b**, including the combined sensor/row/information control.
The altered sensor directions differ by at most **1.1102230246251565e-16** from
the prior reconstruction, while particle times remain exact. Fixed-state row
changes are at most **6.821210263296962e-13** on the gain and
**1.1368683772161603e-13** on the loss. Saved-F1-reference controls remain worse
on the loss and native-time-only gain. This bounds the tested hypotheses without
attributing the residual to an uncaptured quantity.

The next decisive evidence is native retained sensor time, both POCA vectors,
directions and per-row residual/derivative/variance payloads, or an independently
certified arithmetic replay. Aggregate W/b alone cannot supply those fields.
No C++ source change, capture build, fitter edit or refit accompanies this result.

## Material policy and evaluation point

`nativematerial.py` and `scripts/f1_material_audit.py` isolate three variants:

1. Saved `sv,D,dm,dmv` operands.
2. Explicit per-material gas scattering fraction, keeping saved D and other inputs.
3. That fraction plus a change in D evaluation point from CA to the offset crossing.

The frozen native `DetMaterial` uses per-material fractions: solid `.999999`, gas
`.9999999`. F1's default retains the legacy static solid fraction for both; the
existing optional gas policy is not activated by this audit. The expanded native
configuration values and source hashes are bound in the report. The scalar formula
is independently reconstructed with NumPy and the material mixture constants.
Saved `sv` reconstruction must pass `rtol=1e-10, atol=0`, as well as the existing
comparison criterion. Native `Q[theta,theta]` and `Q[momentum,momentum]` independently
isolate scaled angular and momentum variances. Relative residuals are reported
without flooring; zero denominators have explicit counts.

Native `ElementXing::parameterChange` evaluates `dPardM(time())` at
`CAparticleToca+constructorOffset`. F1's saved D is at the CA position although its
crossing ordering time includes that offset. D is affine in path distance, so the
third diagnostic uses

```
D_offset = D_saved + (D(s=1)-D(s=0)) * speed * saved_crossing_offset
```

This avoids subtracting large absolute times to regenerate the saved operand.
Absolute-time reconstruction and direct shifted-time evaluation are still checked
against the established `rtol=1e-5, atol=1e-7` gate. An additional, stricter
`1e-10/1e-12` reconstruction diagnostic fails on near-zero entries in the loss
trace and remains in the output. It is not silently called exact or repaired.
All covariance and mean comparisons retain the established criterion. No fitted
scale, geometry path change, native-reference substitution or variance floor is used.

On the two tested native-seed first iterations, participating covariance failures
fall from 336/864 (gain) and 183/468 (loss), to 68 and 52 after gas-only substitution,
then to zero after the offset-D substitution. Participating means pass throughout.
The gain native-time control also passes material operands after both substitutions.
This is a bounded material-operand result, not full native fit closure. Native
extension-constructor snapshots and later-iteration offset semantics remain open.

## Commands and safeguards

```bash
bash scripts/run_f1_operand_audit.sh ca NEW_CA.json \
  --trace-dir SAVED_GAIN --trace-dir SAVED_LOSS --allow-loader-reconstruction
bash scripts/run_f1_operand_audit.sh material NEW_MATERIAL.json \
  --trace-dir SAVED_GAIN --trace-dir SAVED_LOSS
bash scripts/run_f1_operand_audit.sh population NEW_POPULATION.json \
  --conditions FROZEN_CONDITIONS.json
```

Each output is exclusive. Both drivers validate full identities and native evidence/passivity before operand
comparisons. The CA driver inherits the explicit loader-reconstruction caveat from the measurement audit. The material
driver never reloads hit inputs and records unused source differences honestly. Material participation is checked
before comparing the intersection; out-of-fit cached native changes are counted, not mistaken for zero filter
operands. Sources, trace/input hashes, selected configuration values and all variants remain in the reports. Original
traces, baseline results and failed diagnostic logs stay unchanged. No fitter option is added or promoted.

The [passive hit-operand capture](f1-hit-operand-capture.md) now passes native
runtime passivity and row-algebra checks in a separate built area. The subsequent
[captured CA comparison](f1-captured-ca-operands.md) isolates the last sensor-position
differences to signal-speed coefficient representation and reproduces all initial
CA vectors/derivatives on 2,538 hits at captured references/mass. It does not replace
these saved results or establish later-history/F1 closure.
