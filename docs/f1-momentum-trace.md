# Momentum-isolated outlier coverage and actual-F1 tracing

## Scope and preparation

This continuation follows the completed [seed population decomposition](f1-seed-population.md).
It adds new diagnostics only; all previous receipt-bound sources/results and the fitter remain unchanged.
First establish the full `(run, subrun, event, iseed)` intersection of the 218 remaining fixed-state momentum
failures and 12 fixed-state fit rejections with the 147 native-history targets. Missing histories stay explicit.
Own-state gain/loss coverage is reported separately; no acceptance denominator is reduced to captured targets.

`f1_seed_outlier_coverage.py` reads the sealed momentum populations and the highest-cycle iteration identities
in the preserved full crossing capture. It performs no fits. Invalid pulls remain fidelity failures.

`f1_momentum_trace.py` is a bounded, single-target diagnostic, not another population campaign. It reuses
sealed population preparation, baseline seed arithmetic, momentum selector, and the unchanged fitter trace API.
Baseline q/C/range are computed in the same caller JIT; only fitter p is replaced, including its coupled prior
and momentum-constraint-target consequences. All native constraint times, stock schedules, policies and
thresholds remain unchanged. Fixed states are **final saved stock hit states**, not native iteration-local states:
an initial activity difference can be intentional conditioning rather than a newly discovered fitter defect.
Crossing activity is not frozen.

## Controls and interpretation

The compact unbatched execution differs from the production vmapped/padded compiler context. The cap is chosen
from the saved population cumulative iteration history: cap10 if any stage used at least3 iterations, otherwise
cap3. This is recorded and checked against the saved result, not inferred native chronology.

1. Ordinary versus explicit baseline injection must match the complete returned tree exactly.
2. Momentum-only untraced versus traced must match the complete returned tree after removing only `cf.trace`.
3. Projected momentum output must pass the unchanged saved-population compatibility checks, including discrete
   states and six fidelity predicates. Exact floating agreement is reported separately.
4. Only `trace.valid` records are retained; their count must equal the executed fitter iteration count. Rejected
   executed iterations remain present. Stage histograms are not evidence that speculative iterations executed.

All complete control outputs and the trace are saved to exclusive new files before scientific interpretation.
The 64MiB trace allocation bound is enforced; it is not a bound on compiler RSS. Native first-iteration
`iterations`, `pieces`, and `effects` are read from the existing authenticated capture. The existing comparator
reports q/C/ranges, membership, references, activity, priors and aggregate measurement/material operands.
Its phase order is report order, **not proven causal precedence**. Later histories, row-level derivatives,
actual ordering, candidate/decision/status precedence remain outside that comparison. No new native capture,
build, fitter modification, regularization, tolerance change or default promotion is authorized or performed.

## Completed coverage and bounded traces

Coverage audit completed with stable input/source hashes. All147 native targets/3449 contiguous iterations are
accounted for. Fixed population coverage is71/2151 native-good targets, including67/2139 both-good tracks.
Of the218 remaining momentum failures, **20 have history and198 do not**. Of12 fixed fit rejections,
**4 have history and8 do not**. Other fixed parameter failure coverage is d0=3/49, phi0=4/65, z0=6/74,
theta=8/74, t0=0/2. Own-state coverage remains52/59 gains and19/43 losses. No denominator was narrowed.
Receipt `seed_outlier_coverage_v1.json`, SHA
`e3a68e4fcd67c86c385900b97950b92f30506906991fe09c827740ca1bde7984`.
The coverage module/test are now sealed;57tests pass locally and remotely. Measured39.48s/131328KiB.

Two deterministic first-covered targets were traced with momentum-only substitution and fixed final stock states. Both
use inferred cap3; this is not proof of the original production retry decision. Ordinary/control and momentum
traced/untraced complete50-leaf trees are exactly equal in both cases; no nonfinite control leaves. Raw traced trees
are preserved **before** validity checks, alongside filtered executed traces and complete control results. Strict-JSON
failure preserves the artifact paths/hashes/leaf maps. The earlier three fixture failures after adding the raw
artifact were corrected in exact inventory/event expectations, not by weakening controls.

### Momentum failure:1400:0:7033:0

Completed26 executed iterations, population discrete outputs and six fidelity predicates match exactly.
Floating projection compatibility passes the existing1e-5/1e-7 criterion but is **not bitwise equality**:
maximum |delta xt0|1.011880499390827e-7, |delta Ct0|1.2742339663418534e-6.
The saved population momentum pull is-1.0156604708612724 (outside0.1sigma).

First-iteration seed/reference q/C/range and membership pass the established criterion. The first differing
reported phase is hit activity:4/19 initial hits are inactive in fixed F1 but active in native. The lower bound
differs by0.579937090868043ns; measurement W/b have397/720 and88/120 failing components. Material participation
differs on5/19 initial crossings; on the14 common participating crossings, mean changes pass and196/504 noise
components fail. These are measured operand differences, **not a causal explanation of final momentum error**:
fixed F1 uses final native states whereas native initial fitting does not. Later iterations remain uncompared.
Receipt `momentum_trace_7033_fixed_v1/manifest.json`, SHA
`42675e2f6328344db511a6de839691cef8b983cb3274e24dc00c3b36dd771078`.
Measured59.66s/1808948KiB.

### Fit rejection:1400:0:75388:0 — compatibility gate failed

Two executed iterations. Exact passivity controls pass, and population good/status/iteration/NDOF/hit/LR/crossing
states and fidelity predicates match. **Chi2 fails saved-population compatibility**:
absolute difference0.0007922073404210694. q/C pass relative/absolute criterion but are not exact (large rejected
state scales: maximum absolute q difference0.5115264867199585, C difference5564931.515625).
The driver correctly stops **before native first-operand comparison**. No cause is attributed from this compact
trace; it is not representative evidence for the saved failed fit until compiler/batching compatibility is
resolved. All7NPZ artifacts, failed manifest, source snapshot and log remain unchanged.
ReceiptSHA`4bc4c6b9872ed8b8fe39382b5c01a32524776dd641e6bd58d45695f55532c9f7`.
Measured52.58s/1503660KiB. No tolerance relaxation or rerun of the population baseline.

## Completed saved-operand diagnostic

`f1_momentum_first_operands.py` completed on the compatible7033 trace, no fit. Native sidecar row algebra passes
on19hits/57active rows with zero exclusions. **Native starts all19 hits in state0 (null); fixed F1 starts14
signed-drift hits,1null hit and4inactive hits**. Inactive source-hit indices are32,33,35,38. On the15common-active
hits,14 distance variances differ: F1=0.0256 vs native=1.92, with common variance scale36, hence **75x F1 row
weight** (1.0850694444444444 vs0.014467592592592591). The remaining common null hit has the same1.92 variance.
Time and longitudinal measurement variances **and inverse variances are exact** on all15common-active rows.
Four native-only rows per coordinate remain explicitly counted. No residual/derivative agreement is inferred.

Bounds reconstructed from each side's saved times/active masks are exactly its saved bounds. Native lower-bound
source hit38 (18097.621345250634ns) is replaced by hit34 in F1 (18098.201282341503ns). The labelled diagnostic
using native times with the F1 mask gives18098.20128234079ns; this identifies the mask contribution at the
saved-operand level, not a causal refit or a correction. Upper-bound source hit14 is shared.

Native material participates on all19 initial crossings; F1 only14. Missing strawids23809,23711,23709,23707,
23705 partner with hits38,35,34,33,32 respectively. F1 crossing-active flags are false for partners38/33/32;
partners35/34 remain active but are below the F1 lower bound. `crossing_inbounds` includes activity/validity,
not just the geometric interval, and is not mislabeled as a pure bound predicate.

Receipt `momentum_first_operands_v1.json`, SHA
`497614944a1f9f0527e265c004b41c0cdaf662b837c500e672852accc4dc7dc0`;
measured7.38s/865128KiB.74new tests pass locally/remote (remote5.44s), including real native-row analytic fixtures,
full-key/permutation/activity/nonfinite/bounds/provenance controls. Total this phase214tests=57+83+74, including
2actual-fitter toys. No full-suite or independent native-binary certificate.

## Sealing and resource release

Three new modules and three tests are now receipt-bound; future modifications require new versions.
`momentum_trace_receipts_v1.sha256`:30entries verified, SHA
`3fb7b3d78737b764f4a9c4863080f10cb7fa7c68322c556798f867548a439a46`.
`momentum_first_operands_receipts_v1.sha256`:8entries verified including that seal, SHA
`0b53df4f3d23f6741c87effbdf49fe25f70811b7d3300e76e70a2743a5b75789`. Both successful and failed
receipts,14trace/control NPZ, logs and source/test snapshots preserved. Fitter SHA988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a unchanged.

Next: distinguish final-state conditioning from genuine F1 numerical/state evolution discrepancies, using a
separately controlled own-state momentum trace or a clearly labelled native-iteration-state diagnostic.
For75388 first restore saved-population compiler/batching compatibility before native attribution; retain the
failed compact trace. No arbitrary later-history ordinal pairing, missing-history inference, threshold/default
promotion, build, new native capture, cleanup, git writes or R7handoff. Acceptance95.594% and fixed momentum
89.808% gates remain unmet; this diagnostic does not change those population measurements.
