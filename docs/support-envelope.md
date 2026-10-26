# Quantified paired support-navigation envelopes

## Decision and scope

The separate candidate **`support-navigation-torus-envelope-v1`** admits a **pair of geometric envelope
partitions**, not an exact trajectory or a nominal midpoint boundary. It is the next geometry-only gate
following [support-navigation.md](support-navigation.md). The refusal-first candidate remains unchanged and
is reproduced as a control on the same saved rays. Neither candidate is wired to support energy transport,
`SegmentHistory`, scattering, `steps.py`, flight time, core, production defaults or stock reconstruction.

- **Inner envelope:** only certified omitted-leaf cells replace their immediate parent.
- **Outer envelope:** certified cells plus eligible possible torus cells replace their immediate parent.
- **Original probe cells:** unchanged, including `state=unresolved` and their reasons and provenance.
- **Admission flag:** `bounded_pair_admitted`, never `candidate_resolved` for the new policy;
  `admitted_for_transport` is always false.

The inner leaf occupancy is contained in the guarded actual-solid occupancy, which is contained in the
outer leaf occupancy, subject to the existing engineering float64 guards. Neither envelope need be a
realizable torus boundary; neither is a certified G4 geometry or physical range/time history. A positive
admission means that uncertainty is bounded **within the explicitly chosen stored support model**.
It does not bound its tapered-polycone/Boolean/polygon approximations or the earlier stored-endpoint
coalescence relative to unmodified GDML. The omission bounds are not machine-verified interval arithmetic.

## Explicit geometric policy

`supportenvelope.compose` requires all three resolution values as keyword arguments:
`min_cell_mm`, `max_component_mm`, `max_ray_unknown_mm`, plus the separate `endpoint_budget_mm`.
The replay supplies:

| Quantity | Value | Meaning |
|---|---:|---|
| Probe terminal cell width | 1e-7 mm | Existing probe subdivision termination, not a crossing accuracy claim |
| Maximum connected unknown component | 1e-6 mm (1 nm) | Per ray, exact-touch union across placements |
| Maximum unknown union length per ray | 1e-5 mm (10 nm) | Sum of disjoint unknown components, not per-cell or sample-total allowance |
| Stored source endpoint budgets | 0, 1e-10, 1e-9 mm | Existing isolated numerical policy, unchanged |

The 1 nm / 10 nm limits are **declared engineering geometry resolutions**, not tolerances derived from a
physics comparison, energy-loss model or detector-resolution argument. No code default installs them into
production; both limits are required CLI arguments. Zero limits give the refusal control on these saved
rays. They are not increased for coarser probes to make them pass. Failing either budget refuses both
partitions for that lane. Every budget comparison uses an outward upper bound; a nominal width equal to
a budget may therefore conservatively fail at the last ULP.

Unknown components are formed per ray from the union of **all** unresolved cells, across placements.
Only exact overlap/touch joins cells; no tolerance bridges a known gap. Original cell/source membership
is preserved. Overlapping possible leaf cells are independently a refusal even though the budget union
counts their length only once. Thus splitting one wide uncertain band into many tiny cells cannot evade
the component limit. Disconnected components cannot evade the per-ray total limit.

Only torus `reason="min_width"` cells are eligible. The complete torus reason histogram, inside/unknown
counts, lane counts and an explicit Boolean cap flag must match the retained evidence. A cap in any
placement refuses every lane, including lanes without retained cells; this deliberately avoids interpreting
partial computation as complete exclusion. Roundoff/stall or other unresolved reasons remain refused.
Unknown xtru cells remain refused even if small. Evaluated non-torus misses require explicit uncapped
metadata. Cap reasons contradicting a false cap flag and conflicting cap aliases raise; an alleged
`min_width` cell must have a strictly interior binary64 midpoint under the probe's formula. This rejects
stalled brackets mislabeled as small cells, without using that midpoint as a boundary. `min_width` is not a
proof that coordinate/elementary-function roundoff is negligible; all original guards remain in force.

## Parent replacement on the entire possible footprint

The unchanged refusal control is run first. All its refusals remain except `unresolved-solid-boundary`,
which this policy can replace by a bounded-pair decision. Missing/unevaluated placements, invalid signed
stored occupancy, failed ancestor/leaf verification, positive overlap, and parent/ownership ambiguity
are not waived.

Parent material and coalesced raw-source ownership are checked over **inside plus unresolved cells**, not
only the known inner footprint. A mismatch or negative/multiple-positive owner anywhere in a possible
boundary band refuses the lane. These comparisons use the endpoint-adjusted raw terms, not untouched raw
coordinates or reconstructed physical placement identity. They retain the conservative ambiguity refusal
introduced by the preceding candidate for same-material sibling cancellation.

Each envelope has explicit `+leaf/-nonvacuum-parent` terms and is independently strictly partitioned,
including vacuum. No terms change certification status. Both arms are emitted atomically for the same
accepted lanes; any unexpected sweep inconsistency aborts the whole call rather than exposing one successful
arm or a prefix. The loader's existing vacuum-name convention, including air names, is unchanged and is not
the separate transport air/vacuum correction arm.

## Material bounds must not cancel across placements

The minimum and maximum of each material's **whole-arm** path totals are not conservative in general.
For example, one unknown band replaces A by B while another replaces B by A: inner and outer totals can
be equal even though the material amount is uncertain.

Instead, `atoms` is the common refinement of both complete partitions and all unknown-component endpoints.
For each material `m`:

- Lower bound sums atoms where **both** envelope labels are `m`.
- Upper bound sums atoms where **either** label is `m`.

Endpoint subtraction and grouped summation are outward-padded in binary64 using nextafter and a conservative
summation-error factor. These are marginal enclosures, not necessarily attainable extrema. Their sums need
not equal the path length; each individual complete partition does cover the full path. Marginal bounds
are capped by the requested ray length, not by a material's rounded nominal total.

Geometric dispute remains flagged even when parent and leaf have the same material: both labels then agree
and there is no material-label uncertainty beyond arithmetic enclosure, but the placement boundary is still
unknown. An independently computed `all-nonvacuum-path` bound applies the nonvacuum predicate **before**
aggregation. Replacing one nonvacuum material with another preserves that aggregate; summing their separate
marginal upper bounds would lose this correlation.

No lower/upper **energy or time** claim follows. Changing where energy is lost can alter stopping, subsequent
sampling and scattering. A future paired energy test would initially be a sensitivity comparison, not a
proven physical error enclosure, unless those additional monotonicity/accounting contracts are established.

## Reproduction and provenance

```bash
cd ~/mu2e/remote/analysis
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m trkfastsim.analyses.supportenvelope \
  ~/mu2e/remote-data/trk_fastsim/ledger/navigation_audit_v04_20261025_final \
  --candidate-dir ~/mu2e/remote-data/trk_fastsim/ledger/support_navigation_v04_20261025_reviewed \
  --gdml ~/mu2e/remote-data/crv_fastsim/geometry/mu2e_extracted_v04.gdml \
  --outdir NEW_UNIQUE_OUTPUT --max-component-mm 1e-6 --max-ray-unknown-mm 1e-5 \
  --budgets 0 1e-10 1e-9 --min-cell-mm 1e-7 --compare-cell-mm 1e-6
```

The runner uses the same saved seeds/hits/census and v04 geometry without rereading/regenerating G4 references.
It requires a completed refusal-first replay, identical audit/GDML hashes and matching probe settings.
Current census/probe evidence and each strict control's intervals/decisions/refusals must reproduce the saved
products exactly. It hashes all consumed input/control files and current core/tracker sources, retains base
commits and full `(run,subrun,event,sp)` identities, and never truncates excess rows. Unique outputs and a
completion manifest written last preserve failed/previous results. Data outputs stay local unless separately
configured for mirroring.

Per-budget fine and coarse tables include both intervals, common atoms, marginal material/nonvacuum bounds,
decisions/refusals, unknown cells/components, full-footprint parent/ownership comparisons and signed terms.
Shared probe cells/reports, saved seeds and all effective settings remain explicit. Reversed source/cell/report
order and +/- one-ULP stored-source endpoint stress are checked separately. Probe coordinate guards depend on
the full saved batch domain; arbitrary lane-batch invariance remains unclaimed.

## Measured replay

Authoritative local output: `~/mu2e/remote-data/trk_fastsim/ledger/support_envelope_v04_20261025_reviewed/`
(15 MB; not auto-mirrored). The earlier `..._pilot/` remains intact. Reviewed computation was **5.12 seconds**,
excluding serialization; no scaling estimate is implied. Full input/source hashes are in the manifest.
The 391 saved rays and 76 unsupported placements reproduce the preceding omission evidence exactly.
There are **99 unknown cells in 30 connected components**. At the 1e-7 mm probe width:

- Largest component upper bound: **7.411150590996847e-7 mm** (0.741 nm).
- Largest per-ray unknown union-length upper bound: **2.0933277937729183e-6 mm** (2.093 nm).
- All ancestors are verified; no cap, possible-leaf overlap or parent/ownership mismatch is found.

| Stored endpoint budget | Bounded pairs | Refused rays | Strict-control resolved rays | Inner / outer rows |
|---|---:|---:|---:|---:|
| 0 mm | 241 | 150 | 235 | 4,977 / 5,050 |
| 1e-10 mm | 391 | 0 | 383 | 9,717 / 9,816 |
| 1e-9 mm | 391 | 0 | 383 | 9,717 / 9,816 |

Both positive-budget arms admit all eight previously refused torus-crossing rays **as bounded pairs only**.
All 1,059 possible-cell parent comparisons agree, including the 99 unknown cells; 349 raw-ownership comparisons
are unambiguous. Zero unknown budgets reproduce the refusal decisions, and the saved strict control remains
unchanged. The 150 zero-endpoint-budget refusals are still invalid stored occupancy, not waived by torus policy.

The 1e-6 mm coarse probe uses the **same** 1 nm / 10 nm budgets. At positive endpoint budgets it admits 384 pairs
and refuses seven rays. Its largest component is 9.882 nm and largest per-ray unknown sum is 26.680 nm.
All marginal bounds overlap on the 384 common accepted rays; omission length bounds also overlap for all 76
placements. This is a resolution stress check, not permission to widen limits. Fine/coarse comparisons must
state their common domain rather than quietly dropping the failed coarse rays.

## Verification

Full local isolated non-JAX suite: **894 passed, 5 strict legacy xfails**, including the enabled published-source
capture regression. New suites contain **128 envelope tests and 20 replay tests**. The unchanged refusal-first
sources and tests pass as part of this run. Full-suite invocation is in `transport-ledger.md`; focused command:

```bash
cd ~/mu2e/remote/analysis
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  trk-segment-history-20261025/tests/test_supportenvelope.py \
  trk-segment-history-20261025/tests/test_supportenvelope_replay.py
```

Coverage includes actual synthetic GDML tori and analytic chords, tangency/sector/ULP perturbations, refinement,
unknown-only parent/ownership faults, material-role cancellation, nonvacuum correlations, exact threshold and
adjacent-float budgets, caps, stalled/forged reason metadata, empty/zero inputs, permutations, input immutability,
provenance and no-overwrite. Independent review exposed the cap-evidence and stalled-bracket validation gaps;
both were fixed with 13 additional regressions. Follow-up static review found no new bypass in those fixes.

All positive-budget fine tables are byte-identical across the two stored endpoint budgets, not just their
admission counts. Reversing evidence and +/- one-ULP stored-endpoint perturbations preserve decisions at all
three budgets. The zero-unknown-budget control is checked explicitly against the unchanged strict candidate.
No energy/time run, source/core build, G4 regeneration, commit, push or production-default change occurred.

## Next gate

The geometric pair makes an inclusive bounded support-energy sensitivity comparison possible to design,
but does not itself authorize production transport or complete Tasks 2/3. Before coupling, specify a separate
energy runner that consumes these exact paired paths with immutable geometry/cut/material identities, retains
the same full particle identities and both arms, and calls the existing core loss/segment transport routines.
Keep mean/stochastic stopping and timing choices separate. Do not label energy differences as a rigorous bound
or infer physical deposition without restricted-loss/secondary accounting. Ordered straw wall/gas/wire and
energy-coupled crossings/scattering remain later gates; no material retuning or default promotion is justified.
