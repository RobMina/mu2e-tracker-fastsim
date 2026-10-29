# Captured CA operand comparison and signal-speed representation

This diagnostic consumes the [validated passive hit sidecar](f1-hit-operand-capture.md)
and the original saved F1 first-iteration traces. No fit rerun, C++ change, native
capture, fitter/default change or tolerance relaxation accompanies these results.

Subsequent [initial residual/measurement-variance reconstruction](f1-initial-residual-variance.md)
removes the captured values/variances dependency for all 2,538 initial hits, conditional on
the null branch and captured seed/reference/mass. Independent CAD state selection remains open.

## Evidence bridge and sign contract

`scripts/f1_captured_ca_audit.py` requires both complete ordinary capture gates and
the successful sidecar receipt: exact four-original-tree passivity against the
frozen reference, complete sidecar coverage, empty off-sidecar and unchanged hashes.
The saved trace still uses its original gate hash; it is not rewritten to claim the
new capture generated it. New snapshots join on all six iteration/effect keys and
recheck source-hit/straw identity, activity and null/TOT/longitudinal policy.

Native prediction derivatives `g` use `b=Wq+sum(g*value*w)`. For null hits, convert
to the F1 residual convention with `S=diag(-1,+1,+1)`, `r=S*value`, `H=-S*g`.
Thus distance H is **+dDdP**, time H is **−dTdP**, and longitudinal H is **−g_long**.
The weights use only `1/(measurementVariance*varianceScale)`, never parameter variance.

Fixed-state controls explicitly consume the supplied particle AND sensor directions,
positions and times. They never solve CA or silently regenerate captured positions.
All 25 gain and 13 loss initial hits remain in the comparison, including the three
previously discrepant source hits (gain 16; loss 19/31). A third trace of the same
gain with native time but without native seed injection is a separate reference control,
not 25 additional independent detector hits.

## Localization: sensor position, not particle or sensor time

With the measured endpoints, captured native reference/mass, and ROOT-style reciprocal
sensor normalization, both times, particle POCA and both directions are exactly equal
to capture on these targets. Sensor POCA differs in 10 components per target, at most
1.1368683772161603e−13 mm. This tiny difference matters when normalizing near-zero delta.

- Captured particle position alone, or captured times followed by position reevaluation,
  leaves gain 4 W/3 b and loss 6 W/0 b component failures.
- Captured sensor position alone removes those failures; captured full state and captured
  dDdP controls also pass. These are operand substitutions, not independent generation.
- Replacing the solver's double 1e−4 with exact native float32-promoted precision changes
  none of these failures on the saved targets.

Read-only disassembly of the frozen installed `SensorLine::position3` shows separate
multiply/subtract instructions, not fused multiply-add in that function. This does not
certify all compiler arithmetic. Its source and binary hashes are preserved.

## Cause on this sample: calibration representation

The saved conditions extraction normalizes sampled `halfPropV` knots and reconstructs
per-straw mean velocity from another sampled point. This preserves ordinary numerical
agreement but not exact native products. The original expanded configuration has
`strawResponse.useDb=false`, no per-straw overrides, `defaultHalfPropVelocity=100`,
59 `eHalfPVScale` coefficients and 0.1-keV spacing. These are read directly, not inferred
from captured POCA, residuals or weight matrices.

`source_signal_speed` uses measured ComboHit energy, the FCL coefficients, and the
source `StrawResponse::PieceLine` expression (slope first), followed by `2*halfPropV`.
This changes 11/25 gain and 7/13 loss speeds by at most 2.842170943040401e−14 mm/ns.
All CA vectors, 450 gain / 234 loss derivative components, and 75 / 39 reconstructed
residual components then match captured values exactly at native references/mass.
Reusing saved weights gives zero W/b failures at unchanged rtol=1e−5, atol=1e−7.

A fixed two-by-two check distinguishes coefficient representation from interpolation
grouping: normalized coefficients with source grouping retain the failures; original
coefficients with either grouping remove them on these targets. Reloading measured
energy through the normalized conditions reproduces the previous speed exactly; the
loader's overwritten energy is not the explanation on these targets.

**The saved-F1-reference controls do not all pass:** gain/native-seed passes, loss keeps
14 W/10 b failures, and native-time-only gain keeps 124 W/67 b. Native-reference agreement
must not be reported as F1-generated operand or fit closure.

## All 147 captured first iterations

`scripts/f1_captured_ca_population.py` requires the complete frozen population of
2,538 initial straw hits across 147 targets. It validates exact constructor membership,
raw identity, one old piece, seed/reference/mass, and joins measured event/source-hit
records and both wire endpoints directly. No `2*mid-end` endpoint reconstruction,
per-event loops, success selection, ULP search or fitted correction.

| Metric | Dump-normalized speed | Source-FCL speed |
|---|---:|---:|
| Particle / sensor time exact mismatches | 0 / 0 | 0 / 0 |
| Particle position exact mismatches | 0 / 7,614 | 0 / 7,614 |
| Sensor position exact mismatches | 925 / 7,614 | **0 / 7,614** |
| Both direction vectors exact mismatches | 0 | **0** |
| Derivative exact mismatches | 9,112 / 45,684 | **0 / 45,684** |
| Derivative tolerance failures | 71 (all distance row) | **0** |
| W tolerance failures | 288 / 91,368 (78 hits) | **0 / 91,368** |
| b tolerance failures | 231 / 15,228 (58 hits) | **0 / 15,228** |

Maximum normalized-control sensor-position difference is 2.2737367544323206e−13 mm.
The source-FCL arm's maximum absolute W/b differences are 1.4551915228366852e−11 and
4.656612873077393e−10. All hits retained: zero unusable/zero-distance states or reconstructed
100-mm z-retry triggers. Both constructors and first updates use the explicit native
precision. Exact here means finite elementwise numeric equality, not a universal bit-parity
certificate. W/b are tolerance-level comparisons, not exact equality.

**Population W/b still use captured residual values and measurement variances.** The
independently generated part is the local CA and derivative rows at captured references
and mass. Residual values were reconstructed only on the saved targets, whose saved
weights were separately found exactly equal to captured weights; no independent
all-target variance-policy reconstruction is claimed. This does not address later histories, non-null hits, native retries/fallback/capped pieces,
extension constructors, missing discordant targets, strict filter algebra or final acceptance.

## Provenance, tests and commands

Remote data root: `/exp/mu2e/data/users/ram2aq/trk_f1native_r2_20261027/`.

- `captured_ca_v4.json`: final three-trace comparison; v1–v3 retained.
  SHA256 `80f70074881547ae4b4ada83a5f40d0d8d54e7a9ece900502eb059edc1c9f2ae`.
- `captured_ca_population_v2.json`: final ALL147 comparison; v1 retained.
  SHA256 `ef6261c648b7c992319761c15b2ef63e5bafc18e5c4ee6d43bc2aad63e444449`.
- `sensor_position_disassembly_20261028.txt` and `captured_ca_checkpoint_20261028.txt`.

Sources, input files, sidecar receipt, old/new gates and source/configuration headers are
hash-bound before/after. The expanded FCL hash is pinned to the previous `ca_operands_v6`
checkpoint (`03488bb9…e127e1d`); no original capture-generation expanded-FCL hash receipt
exists. Current integrity and continuity with that earlier audit are established, not a
new retrospective build receipt. Native sources retain the already-verified frozen pins.

Synthetic/review guards include exact null sign conversion, full-key joins, no captured
CA solver hints, supplied directions, exclusive failure JSON, positive integer census,
integer absolute-drift rejection, finite-overflow rejection, and clipping interpolation
segments before integer conversion. Reviews found no sign/full-key defect; the boundary
and census robustness findings were fixed without changing physical results.

Selected isolated regression suite: **970 passed, 1 skipped, 34/34 files**, pinned core
233771b and ANN path. Log `~/mu2e/local_data/trk_f1native_r2_20261027/isolated_captured_ca_v1.txt`.
Seven-file focused suite: 398 passed. Full/slow tier not rerun. No fitter changes.

```bash
bash scripts/run_f1_operand_audit.sh captured NEW_TARGET_REPORT.json \
  --operand-check FULL_CAPTURE/hit_operands_check_v2.json \
  --trace-dir SAVED_GAIN --trace-dir SAVED_LOSS --trace-dir SAVED_GAIN_TIME \
  --allow-loader-reconstruction
bash scripts/run_f1_operand_audit.sh captured-population NEW_POPULATION_REPORT.json \
  --operand-check FULL_CAPTURE/hit_operands_check_v2.json \
  --conditions FROZEN_CONDITIONS.json
```

