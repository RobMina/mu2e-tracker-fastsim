# Run 1A measured-only left/right continuation

This opt-in stage continues the saved [analytic fixed-LR candidate](run1a-stabledrift.md).
It is NOT another independent fit from the original wire/profile seeds, an equal-compute A/B,
or a replacement for stock CentralHelix. The [remaining numerical limitations](run1a-drift-failure.md)
still apply. The baseline optimizer, POCA, response, derivative and certificate code are unchanged.

## Declared contract

Start from every saved candidate final, including objective-valid numerical failures. Retain every
event, group, hit, direction and invalid/empty lane. Fixed support, original parameter scales,
measured early-end times, field/mass hypotheses and response calibration remain unchanged. Neither
truth nor stock-track membership enters. No direction winner, momentum, covariance or outlier cut.

At fixed continuous parameters (including t0), the LR-dependent hit objective is

```
q(L) = 0.5*((L*r_drift - signed_doca)/signed_error)^2, L in {-1,+1}
q(L) - q(-L) = -2*L*r_drift*signed_doca/signed_error^2
```

The pointwise preferred sign is `sign(r_drift)*sign(signed_doca)` for nonzero finite operands;
otherwise retain the current LR. Negative drift radius is NOT made positive. Sign factors avoid
product underflow/overflow hiding a nonzero preference. Gain magnitudes are diagnostics, never
acceptance criteria. Since sigma, radius, geometry and timing are independent of LR at fixed
parameters, simultaneous preferred signs minimize this discrete block in exact arithmetic.
This does not establish a joint continuous/discrete minimum or a physical ambiguity probability.

The default **three outer LR rounds**, with **up to40 unchanged analytic-fit iterations per accepted
LR block**, are declared before the saved-data run. Both budgets are bounded, not tuned to this
sample's resulting pass count. Per round:

1. Propose all beneficial measured LR changes at the current accepted parameters.
2. With no beneficial changes, stop that row without granting extra continuous-fit iterations.
3. Evaluate the whole support with the proposed signs. Accept only an all-hit-valid, finite,
   strictly lower actual objective. A rounded tie rejects the proposal; it is not LR stationarity.
4. Refit only rows with an accepted LR change, using the unchanged candidate solver and saved scales.
   Retain only further strict, all-hit-valid continuous improvements. Numerical failure does not
   discard a verified improvement, but remains explicit and cannot itself supply success.
5. Recheck LR at the retained post-refit state, including after the last permitted round.

Every certificate describes the retained parameters and CURRENT LR, never a rejected trial or the
previous fixed-LR state. Final success requires the unchanged analytic/adjacent-stencil check,
no remaining beneficial pointwise LR flips and no unresolved algorithm error. Outer termination,
LR stationarity, refit outcomes and final certificate are distinct. A recurring LR pattern at changed
continuous parameters is not an exact state cycle; it is not rejected merely for repeating signs.

## Saved development1000 result

Full census is unchanged:1000 events,235 empty events,13492 hits,773 groups,344 original seeds,
5427 fixed selected hits,255 eligible groups and510 attempted direction hypotheses. All historical
candidate/baseline arrays and1036 empty/invalid direction lanes remain unchanged.

| Check on saved/current state | Before continuation | After continuation |
|---|---:|---:|
| Original continuous-only analytic/adjacent check |307|348|
| Combined check: continuous + LR stationary + no continuation error |90|336|
| Groups with either direction passing combined check |63|222|
| Groups with both directions passing combined check |27|114|

The336 combined passes are168 per direction, with33 eligible groups having neither. All90 previous
combined passes remain;246 are gained. This is not a tracking efficiency or a direction classifier.
The saved307 label counts use a weaker criterion and must not be presented as307→336 under an
identical criterion. This continuation uses the saved finals and additional compute, not an
independent original-seed comparison.

Of510 hypotheses,384 have beneficial initial LR flips and receive updates;126 receive **no refit**
and stay unchanged. All384 updated objectives decrease and none increase. Candidate-minus-saved
objective has median-4.327, minimum-199.114, maximum0. Across the same8790 valid hit-hypothesis
entries, including numerical failures, absolute drift residual median falls0.19254→0.14588 mm,
p900.59875→0.46256 mm. These are conditional fit residuals, not independent resolution or physics
validation. Optimization of the same objective naturally reduces its residuals.

LR blocks accepted by round:384/124/26, totaling534 refits and1050 accepted hit-sign flip operations.
Final net changes are846 hit-hypothesis entries over513 distinct source hits; operations and net
changes differ because some signs recur after geometry/timing changes. Continuous improvements are
accepted after384/122/26 blocks; two refits retain identical post-LR states. There are no invalid
proposals, rounded-tie rejections, objective-increase acceptances or contract failures in this run.
Summed objective reductions split3400.32 from fixed-geometry LR blocks and5563.55 from continuous
refits; this is bookkeeping, not a causal comparison against an equal-compute fixed-LR control.

Final174 failures are explicit:110 derivative-unstable,17 guarded-boundary,6 analytically
nonstationary,3 pending-LR labels and38 sticky refit-numerical-failure labels. These labels have
precedence: the independent final certificate census includes41 boundary states, and **five**
hypotheses still have13 beneficial hit-sign flips, two also carrying sticky failures.

A sticky numerical flag records any raw continuous refit termination of `stalled`,
`linearization_failed`, `ill_conditioned`, or invalid/empty termination. It remains a veto even if
a later retained state passes the continuous check. All38 affected rows here experienced a damping
stall; no assertion says that a passed final derivative check erases that history. Of348 final
continuous passes,3 have pending flips and11 have sticky flags, with2 in both sets, leaving336.
This conservative policy was declared before the saved run and is not relaxed to raise the count.
Intermediate `max_iterations`, boundary or derivative-check failures are retained in the trace;
a later LR round may move away from them without making the earlier failed state successful.

The534 refit raw terminations are454 analytic-stationary,37 budget and43 stalled; final per-refit
statuses are342 checked,112 derivative-unstable,74 boundary,3 budget and3 stalled. Final row status,
raw termination and per-refit history must not be conflated. Logical continuous work is4653 update
rounds,2961 accepted updates and21706 trial evaluations. The22512 final-certificate attempts exclude
inner-refit checks and saved replay; these are not hardware operation or throughput counts.

## Evidence and reproduction

The driver replays saved final objectives, all certificate fields and selected objective-valid hit
fields before continuation. It copies all historical arrays unchanged and records current hashes
separately. This does not restore missing original association-input hashes or historical output
bindings. Both continuous-only and combined continuous+LR criteria are reported on the SAVED final
states, so unlike criteria are not silently compared.

```bash
python3 -m trkfastsim.validation.run1a_lrdrift \
  --saved SAVED_DEFAULT40_JOINT_DIRECTORY --candidate SAVED_ANALYTIC_CANDIDATE_DIRECTORY \
  --outdir NEW_LR_CONTINUATION_DIRECTORY
# Repeat with --group-batch 17 --compare-saved NEW_LR_CONTINUATION_DIRECTORY/diagnostics.npz
```

Accepted cache `lrdrift-1000-20261106-final` is array-exact against `lrdrift-1000-20261106-1`
and `lrdrift-1000-batch17-20261106`, including all traces, failures and historical arrays. Payload
SHA256: `d1aee6c77c43086d5cc34138cac7f9ba5b096278a9cf39c290482cea2801bcb3`.
The first report predates added overlap/raw-termination accounting; absent fields there are not zero.
Final local continuation25.37 s, separate saved-final replay5.02 s and load/oracle0.254 s, output
excluded. Earlier continuation timings29.40/57.82 s show laptop variability, not a batch speed claim.
This stage adds work after the saved fixed-LR fit. Its timing is neither full reconstruction
throughput nor a stock/GPU speedup.

Focused validation: **1176 passes in24 selected files**, warnings-as-errors, including53 kernel
and45 driver tests for this stage. The new tests cover independent exhaustive small LR blocks,
negative radii, zero/underflow signs, actual-objective versus diagnostic gain, strict post-LR
refit acceptance, retained numerical-failure improvements, fresh final certificates, pending
last-round flips, recurring sign patterns, noncontiguous changed rows, unequal group widths,
source-hit traces, empty/invalid/unselected lanes and saved-state/hash tampering. No CI/full-suite
claim. Review caught a diagnostic-gain finiteness veto, incomplete saved-field checks and ambiguous
combined-criterion text; these were fixed before the accepted report. The inherited candidate
max-float reporting edge case and independent-root/arithmetic work remain documented separately.

Next: independent root/arithmetic and branch-aware handling of remaining numerical failures, then
multi-track/outlier and physical propagation treatment. The LR stage is an opt-in local alternating
method, not a global ambiguity solution. Material, field-map propagation, uncertainty coverage and
saved-stock CentralHelix physics comparisons remain unfinished. Digitiser optimization stays deferred.
