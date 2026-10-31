# Native seed contract — independent reconstruction

## Execution update (2026-10-28, 18:58 EDT)

The audit has now run: all2,319 seed q vectors and all147 target ranges/2,538 constructors match exactly.
The original covariance comparison passes tolerance but remains inexact; a separate source-typed float32-square
supplement matches all seed/construction covariances exactly without editing this audit's sealed code or receipt.
Four own/fixed single-target component probes also completed with exact baseline controls. Population gates
remain unchanged. See [measured results and immutable receipts](f1-seed-results.md).

The preparation record below is historical; its pending statements describe the checkpoint before execution.

## Preparation status (2026-10-28)

New `scripts/f1_seed_contract.py` and `scripts/f1_seed_contract_audit.py` implement source-derived seed
reconstruction without importing JAX or editing any sealed helper. **The real-data audit has not run.**
Local synthetic/provenance tests pass; this is not a native agreement certificate or a fitter result.

The prior combined captured-seed substitution remains the only measured seed fit arm: acceptance agreement 95.594% and
momentum fixed-state fidelity 89.808%, below the unchanged 99.5% / 99% gates. Nothing here updates those numbers. or
default changes.

## Source findings and independent operands

The frozen `TrkFastSim/src/KinematicLineFitTrace_module.cc:263–271` constructs a seed as follows:

1. Position `(MinuitParams.A0, 0, MinuitParams.B0, hseed._t0._t0)`.
2. **Float32** `XYZVectorF(MinuitParams.A1, -1, MinuitParams.B1)`, `Unit()`, then multiplication by
   `SeedMomentum=2000`. ROOT Unit uses reciprocal multiplication in the vector's scalar type. Promote the
   resulting momentum components to double only when constructing `MOM4`.
3. KinematicLine computes the actual momentum magnitude; this need not equal exactly 2000 MeV. The hypothesis
   that float32 normalization explains the observed small momentum departures is now directly testable, not
   yet certified on the capture.
4. KinematicLine converts the position/momentum to its six parameters using a POCA to the z axis. POCAUtil
   normalizes the direction again; its near-parallel cut remains `1e-8`. Keep source contraction grouping,
   sign, speed, and flight-time arithmetic; no angle clipping or covariance regularization.
5. `seedcov_` is the configured diagonal KL covariance from errors `[15,.05,10,.02,2,1]`, not the
   LineFinder covariance.

**FitParams versus MinuitParams:** the event dump stores FitParams. This is not generally the native
constructor's input. However, the inspected frozen `Offline/CosmicReco/src/LineFinder_module.cc:289–298`
assigns FitParams and MinuitParams identically from the same seed intercept/direction, with both time fields
copied from `_t0`. No drift refit follows `findLine` in that producer. The audit explicitly depends on this
producer-specific source equivalence; it does not assume equivalence for arbitrary CosmicTrackSeed producers.
New upstream source/header pins are inspected-version provenance, not a retrospective binary certificate.

## Range, constraint, and membership decomposition

- **Construction range:** min/max measured float32 corrected ComboHit times of the source seed membership,
  before hit/crossing creation (`KKFitUtilities::timeBounds`). Source ordering and full keys are preserved.
- **Fit seed range:** `KKFit::range` then replaces that range with extrema of all initial hit times and all
  crossing effect times. There is no activity filter. The prior crossing audit already established that
  all 2,538 target hit rows have a different post-construction seed range.
- **ParameterHit:** copy the generated seed q; reset all covariance entries, with diagonal sigma squared for
  positive configured constraints and 1 otherwise; use the corresponding mask. Only momentum is constrained,
  sigma `1e-4` MeV. Its immutable time is the fit range midpoint, not the LineFinder time or KL t0.
- **Associations:** compare ordered source hit indices, initial hit and crossing straw IDs, partner source
  indices, and contiguous collection indices. This checks supplied membership, not independent LineFinder
  hit discovery.

## Intended audit populations (not yet executed)

| Population | Generation | Comparison/conditioning boundary |
|---|---|---|
| All 2,319 captured seeds | Raw saved LineFinder inputs → float32 momentum → double KL q; configured full C; source-hit identities | Native mass supplied; no fit-status/eligible filtering |
| All 2,319 seed ranges and constraint times | Extrema from captured initial hit/crossing times, then constraint construction | Explicitly conditional range check, not independent CA generation |
| 147 fully instrumented targets / 2,538 initial constructors | Generated q and measured construction ranges → unchanged initial hit/tube CA helper → generated fixed offsets/effect times → fit range and constraint time | Full supplied raw geometry and accepted membership remain conditioned; captured q/C/ranges/CA outcomes are comparison-only |

All numeric comparisons retain `rtol=1e-5`, `atol=1e-7`; exact equality is separately reported.
Discrete comparisons require exact values and compatible integer/boolean types. The audit uses exclusive
strict-JSON output, canonical input paths, before/end hashes, executing-helper origins and pinned prior receipts.
All full-fit/F1/numerical and independent-seed-finder flags remain false.

## Local tests and review

- `tests/test_f1_seed_contract.py`: 124 passing tests; analytic cases, float32 arithmetic, jagged full-key
  inputs, bounds, constraints, configuration rejection, and columnar-production guards.
- `tests/test_f1_seed_contract_audit.py`: 92 passing tests; independent analytical four-seed/three-event
  capture, mutation isolation for every all-seed comparison family, ordered membership/shuffles, provenance,
  protected paths, exclusive output, and hash drift.
- `tests/test_f1_seed_targets.py`: 16 passing adapter tests over a vectorized 147-target / 2,538-constructor
  fixture; real CA helpers execute, with independent table shuffles, wrong trajectory rejection, and captured
  q/range/time mutation independence. Constructor expectations use a separate generation, so this is adapter
  coverage rather than independent numerical validation.
- **232 new tests pass locally**; ten separately run existing/new modules plus the target adapter total
  **667 selected passing tests**. Logs: `local_data/trk_f1native_r2_20261027/seed_contract_isolated_v1.txt` and
  `seed_contract_targets_v1.txt`. No full/slow-suite or remote test certificate.
- The review did not execute the audit or independently validate native data.

Early local development failures were validation/parser/interface issues, not observed native mismatches:
removed an unsupported `unique=False` argument to the immutable integer-key helper, rejected commented/dotted
configuration overrides, and fixed a shadowed module-name variable in the provenance result map. One synthetic
expected range was corrected to match its explicit source membership. Existing receipt-bound modules unchanged.

## Resource checkpoint

The preparation-only `seed_contract_v1_sources` snapshot is retained; its existence is not evidence that a job ran. No
additional archive cleanup, builds, native recapture or git writes.

## Reproduction once a host is available

```bash
D=$M2FS_DATA/trk_f1native_r2_20261027
C=$D/crossing_3k_20261028_v2
PYTHONPATH=.:scripts python3 scripts/f1_seed_contract_audit.py \
  --full-check "$C/crossing_full_check_v1.json" \
  --initial "$D/initial_crossing_v2.json" \
  --capture "$C/capture.root" --expanded "$C/capture.expanded.fcl" \
  --native-source $TRKFS_MUSE_AREAS/fastreco_crossings_p109_20261028 \
  --out "$D/seed_contract_v1.json"
```

Drive through run-and-capture with taskset, versioned source/test snapshots and logs. Verify output nonexistence
before running; preserve every failed attempt. The script never launches native reconstruction or a build.

## Subsequent actual-fitter arms

The existing seed option bundles q, C, and range. The new locally tested single-target preparation in
[f1-seed-arms.md](f1-seed-arms.md) first checks an ordinary-versus-explicit `(qB,CB,RB)` control, then isolates
`(qN,CB,RB)`, `(qB,CN,RB)`,
`(qB,CB,RN)` and the full tuple. Keep constraint time explicitly identical. These mixed tuples are synthetic
operand ablations, not native seeds. q necessarily changes initial prior means and the seed-derived momentum
constraint target; C changes prior covariance and later uncertainty/material operands. The current fitter API
cannot independently freeze those coupled consequences. Do not claim prior-independent or constraint-target-
independent q effects. Reuse existing finished baseline outputs; report own-state acceptance and fixed-state
fidelity separately with all required denominators. The full-population driver and native-data fit runs remain
pending; the single-target runner is receipt-gated and cannot proceed before a successful independent seed audit.
