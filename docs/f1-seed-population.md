# Seed-component populations — versioned driver and compatibility gates

## Design

This phase follows the measured [seed reconstruction and target probes](f1-seed-results.md). The original
receipt-bound helpers, fitter and captures remain unchanged. New modules:

- `f1_seed_population_kernel.py`: eight kernel choices, including momentum-only and geometry/time-only q splits.
- `f1_seed_population_compare.py`: vectorized NumPy full-key comparison, acceptance and stock-sigma fidelity.
- `f1_seed_population.py`: input/baseline authentication, bounded controls and production-shaped batching.
- `f1_seed_population_run.py`: reauthenticate a completed bounded-control receipt, then run only new populations.

Population execution intentionally excludes ordinary/control and the already finished combined-full arm.
Those preserved results are reused, not regenerated. Only `parameters`, `momentum`, `geometry_time`,
`covariance`, `range` run as new populations. The kernel retains ordinary/control/full for bounded tests.

| New arm | Fitter parameters | Covariance | Initial owner range |
|---|---|---|---|
| parameters | all6 transformed captured native | ordinary | ordinary |
| momentum | ordinary first5; native p | ordinary | ordinary |
| geometry_time | native first5; ordinary p | ordinary | ordinary |
| covariance | ordinary | transformed native | ordinary |
| range | ordinary | ordinary | native |

The first5 coordinates are `(a0,b0,a1,b1,time_at_y0)`, not KL coordinates. Their native-to-fitter transformation
can itself involve momentum-dependent time conversion. These are explicitly fitter-coordinate ablations, not
claims to isolate every underlying physical cause. Momentum substitution also changes the initial prior mean
and seed-derived momentum constraint target; covariance changes prior uncertainty/material consequences.
No covariance or range is silently recomputed from substituted q. Baseline operands are computed inside the
same caller JIT as the fit. The exact float32 covariance supplement is required evidence but does not modify
or replace the captured-native transformation used by these diagnostic arms.

All arms retain explicit native constraint time, stock full schedule, reference=`previous`, crossing=`current`,
activity=`hit_or_doca`, and unchanged material/default/tolerance settings. Fast cap3 followed by cap10 refits
matches the production orchestration: retry if any cumulative iteration difference >=3. Three size classes
(32/64,64/128,file maximum),512-row chunks, repeated-last-row lane padding to multiples of256. No per-track
Python fitter loop. Compiler caches are cleared between arms, not shared unsafely across modified settings.

## Populations and baseline recovery

- Own states: all **2,315 eligible single-seed/single-KL events**, including164 native-rejected fits.
- Fixed states: all **2,151 native-FitOK events**, including F1-rejected outcomes. The164 native-rejected events
  are explicitly outside this fidelity population; the earlier3700fixed probe remains a separate diagnostic.
- Neither is the2,319 seed-audit population or147 captured-history population.

The saved native-time-only baseline `ids.npz` omits run and seed. Recover full `(run,subrun,event,iseed)` using
its unique event_index joined to the hash-bound input; require matching saved subrun/event/KLstatus, exact
population census, candidate masks/partners/states and stock references. Never infer run from a filename.
All paths/digests, original seed plus covariance receipts, ANN files and effective runtime are authenticated.
Saved baseline source hashes differ from current source: bounded numerical compatibility is checked rather
than pretending source identity. Original/current metadata are retained.

Acceptance includes every eligible row. Fidelity uses both-good denominators, with invalid/nonfinite pulls
counted as failures, not removed. Report common baseline/arm/stock-good intersections separately, wrap phi,
retain per-population status/nonfinite counts and paired changes. Original99.5% acceptance and99% fidelity
targets remain unchanged. Full fit/F1/numerical closure flags remain false.

## Bounded compatibility controls

Choose the first input-order member of every nonempty `(size class, fast/retry, good/rejected)` stratum of the
saved baseline, plus captured3700/7033 where present. This is deterministic bounded coverage, not a population
ordinary rerun or proof of bitwise equivalence for unrerun rows. Each class uses256 lanes including padding.

At cap3 and cap10, ordinary and explicit-baseline control must match **the complete50-leaf returned tree**,
including dtype and nonfinite positions. Then select the ordinary fast/refit outcome and compare the saved
baseline's t0 q/C, good/status/iterations/chi2/NDOF, valid-hit activity/LR and valid-crossing **actual fitter
activity**, not the final-line proxy. Discrete values are exact; floating metrics report exactness separately
from unchanged rtol1e-5/atol1e-7. Additionally require identical0.1-stock-sigma fidelity predicates, with finite
positive stock errors. A failure blocks all population arms and preserves its artifacts.

Local tests also caught foreign-callable origin checks missing from the runner; explicit required-function checks were
added. No physics tolerance was adjusted.

## Local validation

**331 new tests passed locally**, including six explicitly selected actual-fitter slow tests, in13.14s:
128 orchestration/validation tests,105 comparator tests,98 kernel tests. Fixtures cover full-key permutations,
nonfinite denominators, exclusive/hash-stable failure outputs, control-before-ablation, repeated padding and
1,031-row multi-class fast/refit restoration. Local execution is not native population certification.
Log: `local_data/trk_f1native_r2_20261027/seed_population_tests_v1.txt`.
The separate receipt-reuse launcher has its own tests, recorded with its eventual execution snapshot.

## Execution checkpoints

**Own-state bounded gate completed:** `seed_population_own_controls_v1/manifest.json`, SHA
`21bbc55d828bda8457c2d6d9c97db5d396b365f38f2cb1ad149c648607314cc0`.
All13controls across12nonempty strata and3sizeclasses match saved baseline **exactly in every compared field**;
all complete ordinary/control trees exact at bothcaps; no nonfinite leaves. Measured390.38s wall/2,174,564KiB RSS
on oneCPU03. This binds the three new modules at the versions in `seed_population_v1_sources/`; no subsequent
edits to those modules are permitted without a new version.

**Fixed-state bounded gate completed:**12controls across3sizeclasses; every compared baseline field and all
complete ordinary/control trees exact, no nonfinite leaves. ReceiptSHA
`20b66ea0757a3bd7a7853d53c0c76aaa5fde272ff9e6ee82cf4dcbe09eda544a`.
Measured731.49s wall/2,168,832KiB RSS on oneCPU03. The laptop600s wait expired; immediate host inspection
confirmed the same job still running. It completed normally; no relaunch or output overwrite.

## Completed population results

Both modes completed allfive new component arms with successful inherited controls and stable before/end
hashes. The own-state run used oneCPU02 and fixed-state oneCPU03, never two jobs on a host. No nonfinite values
occur in any saved array of any arm, including rejected rows. Both finished baseline populations were reused.

**Own-state acceptance is identical across all five arms and the preserved baseline:**
2,315eligible,2,151native-good,2,167F1-good,2,108both-good,59gains,43losses,105neither.
Agreement **95.59395% ±0.42654%**, still below99.5%. There are zero acceptance flips relative to baseline.
Fixed mode retains2,139F1-good and12F1-rejected out of2,151native-good, identical identities across arms; its
conditional99.44212% retention is not the all-eligible acceptance metric.

### Fixed-state fidelity (all2,139 common both-good tracks)

Fractions are within0.1native sigma, with nonfinite outcomes retained as failures if present.

| Arm | Momentum passing | Momentum fraction | Other five parameter passing counts |
|---|---:|---:|---|
| Preserved ordinary baseline |911/2139|42.5900%|2090,2074,2065,2065,2137|
| Parameters (all6 q) |1921/2139|89.8083%|same|
| **Momentum only** |**1921/2139**|**89.8083% ±0.65415%**|same|
| Geometry/time only |911/2139|42.5900%|same|
| Covariance only |911/2139|42.5900%|same|
| Range only |911/2139|42.5900%|same|
| Preserved combined-full seed |1921/2139|89.8083%|same|

The ordered other coordinates are d0,phi0,z0,theta,t0, with fractions
97.7092%,96.9612%,96.5404%,96.5404%,99.9065%. Only t0 clears99%; momentum still misses it.

The separate hash-bound `f1_seed_population_report.py` checks **per-track predicates**, not just equal totals.
Momentum-only versus baseline gains1,012 momentum passes and loses2 (net+1,010); no other coordinate changes
its pass/fail predicate. Momentum-only versus all-q **and versus the reused combined-full result has zero
predicate disagreements for every parameter and identical acceptance identities**. Floating fitted values are
not bit-identical. This establishes that momentum substitution, with its prior/constraint-target consequences,
is sufficient for the observed combined-seed fidelity gain on this sample. It does not distinguish the prior
from the constraint-target mechanism or imply full numerical closure.

### Own-state fidelity diagnostic (2,108 common both-good tracks)

Momentum passing goes749→1729 (35.5313%→82.0209%) for momentum-only or all-q; geometry/time,C and range stay749.
Paired momentum-only change:991new passes,11lost passes. Momentum-only,all-q and preserved full have identical
per-track six-parameter pass/fail predicates. Own-state fidelity is diagnostic, not the fixed-state99% gate.
Range-only has no predicate changes but is not bitwise identical in every transformed KL coordinate; do not
promote the earlier one-piece expectation to universal returned-tree equality.

## Validation, receipts and resources

**470 new tests pass locally and remotely:**128driver+105comparator+98kernel+132receipt-reuse-launcher+7paired
report. The463-test remote run (including6slow actual-fitter toys) passed in105.87s;7report tests in1.11s.
Read-only reviews addressed crossing-activity and invalid-error controls; tests addressed origin checks.
No independent review of the final paired report's native execution is claimed.

All under `/exp/mu2e/data/users/ram2aq/trk_f1native_r2_20261027/`:

| Result | SHA-256 |
|---|---|
| seed_population_own_v1/manifest.json |3131e7f6b8781604e8823b11bc0685ef676b0182f429a3dc0be307c4908d766b|
| seed_population_fixed_v1/manifest.json |72de580e575bc7d653d5697ffb7af0bfcd4fbffd6fe7b54ba3bf3029819e1ea0|
| seed_population_report_v1.json |c93682a2d4d1096944e603fe838ac7104d93d527fb8c3001c77a87a2b0102bdc|
| seed_population_receipts_v1.sha256 (62 entries, verified) |14651f5cfe54d344bd8d78449fb5d8df34fa2fa610693dc9b577559d969bda94|

Source snapshots: `seed_population_v1_sources/`, `seed_population_run_v1_sources/`,
`seed_population_report_v1_sources/`. All five modules and tests are now receipt-bound and remain unchanged. Measured
population walltime/RSS: own756.31s/2,003,884KiB;fixed901.24s/1,850,140KiB. No relaunches. Paired
report9.43s/511,748KiB. No runtime extrapolation.

No cleanup, build/native recapture, git writes, fitter/default/tolerance change or R7handoff. Fitter hash stays
`988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a`.

**Next:** momentum correction has explained the prior improvement, not removed the remaining218fixed-state
momentum failures or12fixed-state fit failures. Investigate first differing actual-F1 operands with this known
seed effect separated; strict algebra862vectors/1508components, later state/CAD/activity/material and missing
7gains/24losses native-history coverage remain open. No default promotion is authorized.
