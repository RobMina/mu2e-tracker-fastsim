# Seed-component fitter experiments

## Execution update (2026-10-28, 18:58 EDT)

After the successful independent seed audit, own/fixed probes on1400:0:3700:0 and1400:0:7033:0 completed allsix
arms with exact50-leaf baseline controls, stable hashes and no nonfinite leaves. Good/status/iteration outcomes
are unchanged between arms. The rejected3700 fixed-state range arm differs only in retained initial owner range.
These are bounded diagnostics, not population fidelity/acceptance. See [results and receipts](f1-seed-results.md).

The preparation record below is historical; population infrastructure and comparisons remain pending.

## Preparation status, 2026-10-28

The independent seed audit described in [f1-seed-contract.md](f1-seed-contract.md) is still pending a free GPVM.
**No new native-data seed result, acceptance result, or fixed-state fidelity result exists.** Preparing these
experiments does not bypass the ordered seed audit. The fitter and every prior sealed helper remain unchanged.

New files:

- `scripts/f1_seed_arms.py`: traceable baseline initialization and component selection around unchanged
  `K.fit_track`, plus a host-side input validator.
- `scripts/f1_seed_arm_probe.py`: exclusive, receipt-gated **single-target** experiment runner.
- `tests/test_f1_seed_arms.py`, `test_f1_seed_arm_probe.py`, `test_f1_seed_arms_fit.py`.

A full-population campaign driver/comparator is **not** implemented by this single-target runner. It is intended
first to verify that the experiment itself preserves the ordinary baseline before interpreting ablations.

## Arms and couplings

Baseline q/C/range are calculated inside the same JIT as the fit, from reconstructed seed inputs and that call's
hit/crossing vectors. No host-materialized covariance or KL round trip supplies the baseline. The expressions
preserve `s_crossing = straw_poca + wire_radius` before dividing by speed, seed membership masks, inactive seed
hits, and the associated-only initial crossing selection.

| Arm | q | C | Initial owner range |
|---|---|---|---|
| ordinary | ordinary initialization | ordinary initialization | ordinary initialization |
| control | baseline | baseline | baseline |
| parameters | transformed captured native | baseline | baseline |
| covariance | baseline | transformed captured native | baseline |
| range | baseline | baseline | captured native |
| full | transformed captured native | transformed captured native | captured native |

All arms use the **same explicit native constraint time**, stock full schedule, maximum 10 iterations,
`reference_model=previous`, `crossing_time=current`, and `xing_activity=hit_or_doca`. The experiment does not
change thresholds, deweighting, material defaults, or regularization. Mixed tuples are synthetic fitter-coordinate
ablations, not individually native-exact seeds.

- q also changes initial reference/unbiased/prior means and the seed-derived momentum-constraint target.
- C also changes prior covariance and downstream uncertainty-dependent activity/material operands.
- Explicit constraint time is separate from its target momentum.
- Range-only is expected to leave the fitted numbers unchanged when the initial owner has one piece. If no
  update is accepted, its altered initial range can remain in returned owner bookkeeping. Do not demand that
  intentionally different range bookkeeping be equal or infer general range irrelevance.

The current public fitter API cannot independently freeze prior means/covariances or the constraint target while
changing the corresponding seed inputs. These experiments make no such causal claim.

## Runner gates and evidence

Before fitting, the runner requires a completed/passed/hash-stable independent seed receipt and an explicitly
supplied SHA-256, verifies the complete inherited manifest and target membership, and binds native loader inputs
by **both canonical path and digest** to that receipt. Once the real seed audit exists, choose and record its
immutable hash; no successful future receipt hash is invented here.

It verifies fitter source hash, executing helper origins, effective import-time fitter settings, CPU/x64 execution,
material scattering/geometry settings and cached material constants. `F1_*` overrides are rejected. An explicit
native constraint time is mandatory. Invalid seed/geometry/association operands fail rather than being repaired,
filtered out or regularized. Native transformed covariance may retain tiny multiplication asymmetry; it is not
symmetrized. This is not a complete validator of all downstream hit/ANN/conditions physics.

The ordered six-arm inventory is checked. Ordinary executes first, then explicit baseline control. Every returned
array, history, status and good flag must agree exactly (with matching nonfinite positions retained as diagnostic
outcomes); **a failed control stops before any component arm**. Nonfinite output is separately enumerated and
preserved in NPZ, not converted to valid physics. All arm artifacts use exclusive creation, per-leaf path metadata,
hashes, and an end-of-execution artifact recheck. Source/input hashes are rechecked at the end; serialization
fallback explicitly retains false control/full-numerical flags. Earlier artifacts survive failures.

`controls_passed` means the runner controls passed, **not** successful reconstruction or numerical closure.
`full_fit_closure`, `full_f1_closure`, and `full_numerical_validation` remain false.

## Local validation

**237 new tests pass:**

- 129 helper tests: component isolation, five-/six-component inputs, seed masks, finite padding, crossing
  associations, single-hit/degenerate ranges, same-JIT baseline expressions, input rejection, and mocked routing.
- 104 probe tests: full-key parsing, nonfinite/exact whole-tree comparison, source origins, exclusive outputs,
  successful/failed mocked receipt preparation, own/fixed routing, control-before-ablation, failure artifact
  retention, hash drift, serialization fallback, altered arm inventory and stale import-time settings.
- Four explicitly selected slow tests using the actual unchanged fitter and locally available ANN files:
  full-schedule **own-state and fixed-state accepted toys** reproduce the ordinary complete tree exactly through
  baseline injection and range-only substitution; all six arms on a synthetic no-active-hit rejection retain
  the selected initial operands; a three-lane batch (including a repeated padding lane) preserves exact control.
  These tests do not constitute a native-data or Linux/compiler certificate. Rejected-C comparison checks use
  tight test-only roundoff tolerance for the separate expected calculation, not changed physics tolerances.

Measured final slow-test invocation: **4 passed in 17.33s**, not a runtime projection. Final unit/probe invocation:
**233 passed in 2.72s**. Logs under laptop `local_data/trk_f1native_r2_20261027/`:
`seed_arms_unit_v1.txt`, `seed_arms_probe_v1.txt`, `seed_arms_fit_v1.txt`, `seed_arms_fit_v2.txt`,
`seed_arms_fit_v3.txt`. The first two-test slow run lacked the shared core hooks and emitted an unknown-marker
warning; later runs explicitly included the existing `deps-core-233771b` and selected only this slow module.
No installation or changes to shared test configuration were made.

They were addressed locally; the review did not run native data. Local development found dropped probe flags in the
old shared serializer fallback; the new runner restores them without editing that receipt-bound serializer. Tests also
prompted mask-before-conversion and zero-wire guards.

## Execution boundary

Only after the independent seed audit succeeds, on a freshly checked idle/unlocked GPVM, using the existing
single-thread JAX environment and run-and-capture with one allowed CPU:

```bash
python3 scripts/f1_seed_arm_probe.py \
  --seed-contract /path/to/completed_seed_contract.json \
  --seed-contract-sha RECORDED_64_HEX_SHA \
  --check /exp/mu2e/data/users/ram2aq/trk_fastreco/p109_iteration_3k_20261027/check.json \
  --conditions /exp/mu2e/data/users/ram2aq/trk_fastreco/p109_prior_all_20261025/conditions.json \
  --straw-axis /exp/mu2e/data/users/ram2aq/trk_fastreco/align/TrkAlignStraw.csv \
  --key RUN:SUBRUN:EVENT:0 --out /path/to/new_exclusive_probe_directory
```

Use `--fixed` in a separate exclusive output for fixed-state probing. Native-data execution, population control
against preserved baseline outputs, q momentum-versus-geometry/time subdivision, and full acceptance/fidelity
comparisons are still pending. Never rerun a finished full baseline merely to supply a comparison.

06 broken `/exp`. Original 99.5% acceptance / 99% per-parameter fidelity gates remain unmet and the previous measured
results are unchanged.
