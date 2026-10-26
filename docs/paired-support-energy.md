# Paired support-energy sensitivity

## Scope and separation

`paired-support-envelope-mean-sensitivity-v1` consumes the reviewed, saved inner/outer support paths from
[support-envelope.md](support-envelope.md). It is a separate experimental runner, not a production switch.
No geometry is regenerated, and no tracker straw crossings, scattering, upstream state, exit continuation,
secondary library, response, digi/reco or G4 reference is run. Both full seed populations must be admitted
as a geometric pair before execution; the runner never intersects accepted subsets or drops stopped lanes.
The original geometric artifacts retain `admitted_for_transport: false`. This explicitly named experiment
permits a sensitivity calculation, not a retroactive production-geometry approval.

Two distinctions matter:

1. `ordered-support-mean-replace-v1` remains a **sampled** model; only overshooting samples use the mean.
2. New `ordered-support-deterministic-mean-v1` invokes core `transport_segment(loss_model="mean")`.
   It proposes the already computed `dedx_restricted * step`, with no sampler or RNG access. The existing
   adaptive integration, KE update, absorbing stop and residual cutoff conventions are reused. There is
   no new loss formula or duplicated integrator. Core defaults remain `loss_model="sampled"`.

The mean model is not the expectation of nonlinear stochastic stopping. The ledger schema's
`sampled_loss_MeV` field stores the full deterministic proposal for this mode; metadata explicitly labels it.
Positive-material deposition and escaping energy remain unmodeled/NaN, and timing is unknown. Only primary
KE debit closes. Cutoff residual KE is retained, not deposited. The inherited modeled vacuum (including
loader air-name exclusions) is unchanged; this does not isolate the separate air-versus-vacuum physics arm.

## Input approval and exact identities

The geometric envelope manifest hashes inputs and sources, but does not hash its own emitted CSVs. The
energy tool therefore has a separate **pin** operation that hashes the exact selected six files:

- envelope summary and original saved seeds;
- inner and outer complete interval CSVs;
- paired decisions and refusals for the explicit endpoint budget.

The pin is a review/approval snapshot, not an independent proof that files were unmodified before approval.
`run` requires its SHA256 supplied independently, verifies pinned bytes before parsing, and parses those same
bytes. The saved seeds must match the audit seed bytes. Every inherited audit/control file and the GDML are
rehash-checked; ROOT entries/cycles/reference counts are inherited evidence, not freshly inspected ROOT data.
No existing pin or output directory is overwritten. Only a completed all-pair envelope is eligible; coarse
or zero-budget partial-admission products cannot silently substitute for the fine inclusive sample.

Full `(run,subrun,event,sp)` and lane columns are parsed as exact nonnegative decimal integers compatible
with int64. No floating-point route can round an identity above `2**53`. Joins require all identities to
agree with the original seed lane; duplicate/missing/extra lanes, false admission flags, truncated tails,
gaps and overlaps are errors. Seed provenance columns (`reference_surface_row`, `surf`, `t`) are retained
in final tables. These seeds are tracker-envelope SurfaceStep midpoints, not certified generated/entrance
states; no upstream energy/time closure is implied.

Material names map literally to the existing core registry with the same effective unrestricted cut
`1e9 MeV`. Seeds whose core kinematics has `Tmax > 1e9 MeV` are rejected so the unrestricted label remains
valid over the modeled slowing history. Unknown material keys fail; no fallback or new aliases are invented. The full core
Material descriptions, cut and loss basis are stored in each history with their digest.

## Canonical path representation

Probe subdivision is not an additional transport interaction. The runner first validates exact ordered
coverage `[0, requested_length]` for every positive-length lane, then merges only exactly touching equal
material/cut/loss-identity rows (including vacuum). This prevents restarting the adaptive step schedule at
arbitrary probe-cell boundaries. No tolerance, reordering, material priority or geometry endpoint correction
is introduced. Original geometry files remain untouched; a source-row-to-canonical-row table preserves
membership. Refining same-material CSV rows must leave canonical paths and mean transport unchanged.

For the reviewed 391 rays, 9,717 inner and 9,816 outer input rows both reduce to **7,497 canonical intervals**,
with **19 nonvacuum core material keys**. The two positive endpoint-budget geometries are identical; using one
explicit budget is not an additional independent physics validation.

## Run, outputs and interpretation

```bash
cd ~/mu2e/remote/analysis
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025
~/mu2e/.venv/bin/python3 -m trkfastsim.analyses.supportenergy pin REVIEWED_ENVELOPE_DIR \
  --pin-path NEW_PIN.json --endpoint-budget-mm 1e-10
~/mu2e/.venv/bin/python3 -m trkfastsim.analyses.supportenergy run NEW_PIN.json \
  --pin-sha256 APPROVED_PIN_SHA256 --outdir NEW_ENERGY_OUTPUT \
  --tracker-ref FULL_TRACKER_BASE_COMMIT --core-ref FULL_CORE_BASE_COMMIT \
  --max-step-mm 1 --fractional-loss .05 --stop-ke-MeV 1 --max-steps 10000
```

The supplied commit fields identify the worktree base; current source SHA256 manifests identify the actual
code, including any uncommitted changes. Both arm histories embed these identities, material/cut descriptions,
the pin hash, inherited geometry evidence and explicit numerical settings. No RNG is created for mean mode.

Both arms must complete and validate before writing output. Each history is validated by SegmentHistory;
per-identity first/final KE, summed primary debit and path are checked against final states. Transmitted rays
must reach their full requested extent; stopped rays remain present with retained residual energy. Outputs
include two NPZ histories, both canonical interval/membership tables, full final seed tables and one paired
final table with signed **outer-minus-inner** KE/path differences and status transitions. Completion is written
last and hashes every emitted data artifact; no completion is written after an arm failure.

These differences are **sensitivity results, not energy or range bounds**. Material ordering, adaptive
integration, stopping and any later stochastic/scattering coupling can break monotonicity. No deposited
energy, physical stopping probability or corrected detector acceptance follows from this experiment.

## Reviewed result and verification

Authoritative local output: `~/mu2e/remote-data/trk_fastsim/ledger/support_energy_v04_20261025_reviewed/`
(9.5 MB, not auto-mirrored); the earlier pilot is retained. Input pin:
`~/mu2e/remote-data/trk_fastsim/ledger/support_energy_input_v04_20261025.json`, SHA256
`cc8aafb3b53894d56bc4f4d05231c3b932db7eb0cca3ec6768b601721439c390`.
Reviewed execution was 1.41 s locally excluding output serialization, not an extrapolated runtime.
The local run on the exact 391 saved rays gives 377 transmitted and 14 model-stopped identities in each
arm, with no terminal-status changes. Both histories have 63,422 rows and zero per-row primary-debit residual.
Exactly eight identities have changed final KE; maximum absolute outer-minus-inner difference is
**1.7100842342188116e-6 MeV**, and the final-path difference is zero. These are the deterministic mean policy's
results, not a physical precision claim; integration-step error has not been bounded by this number.

Final full local tracker non-JAX suite: **1,044 passed, 5 strict legacy xfails**, with published-source capture
regression enabled. Focused core segment/trace/transport/world-energy/loss suites: **157 passed**. New tests
include 125 paired-core-layer cases, 24 runner cases and 49 deterministic core cases. An existing mode-parameterized
test also exercises the new mode. All new tests pass; no new expected failure is left. Tests cover exact large
integers, malformed inputs/flags/headers, unknown materials, gaps/overlap/tails, arbitrary subdivision invariance,
empty/zero/multiple primaries, unequal-arm stops, no RNG, cutoff/exhaustion, primary closure, hashes and arm-failure
atomicity. Independent review fixes reject duplicate raw CSV headers, non-Boolean JSON flags and momenta outside
the effective unrestricted-cut domain. Follow-up review found no obvious critical issue in those fixes.

## Next gate

Before any physics interpretation, vary adaptive/max-step resolution on the same paired paths and separate
that integration sensitivity from geometric uncertainty. Then test explicitly named stochastic-stop arms
with recorded RNG policies (equal seeds alone do not ensure paired draws). Ordered wall/gas/wire, consistent
restricted-loss/secondary accounting and energy-coupled crossing/scattering updates remain missing. Full
Tasks 2/3, integrated upstream/exit flight time and stock TV2/TV3 gates remain open; production defaults stay
frozen. No C++ build is required for this local Python experiment.
