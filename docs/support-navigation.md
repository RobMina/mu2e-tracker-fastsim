# Refusal-first support-navigation candidate

## Scope and explicit name

`support-navigation-torus-replace-refuse-v1` is a separately named geometry-only candidate in
`trkfastsim/supportnavigation.py`. It does not change `interiorpath`, the strict navigators,
`TrackerInterior.transport_support_history`, `steps.py`, core, production defaults or loss sampling.
The prior [resolution](navigation-resolution.md) remains a diagnostic; this candidate makes the admission
policy explicit and exposes complete partitions **only for candidate-resolved whole rays**.

Every output still states `admitted_for_transport: false`. A candidate-resolved ray means consistency of the
stored approximate support model plus guarded omission evidence, not exact GDML/G4 navigation. Straw wall,
gas and wire ordering, energy-coupled crossings/scattering, world/exit history, secondary accounting and
final stock validation remain absent. This work does not complete transport-energy or integrated-time tasks.

## Two distinct numerical policies

1. **Stored source endpoints:** the caller must supply `endpoint_budget_mm`. Existing
   `boundarypolicy.coalesce` retains raw endpoints and applied shifts; its bounded canonicalization followed
   by a strict signed sweep is unchanged. Zero budget is the strict numerical control. Genuine sub-budget
   gaps/overlaps cannot be distinguished from roundoff. A collapse/transitive-chain error aborts the call;
   invalid signed occupancy is localized to affected lanes before independent strict partitioning.
2. **Omitted-solid boundaries:** `refuse-whole-lane-on-any-unresolved-cell`. Any positive-width unresolved
   torus/xtru cell refuses the entire requested ray. No bracket midpoint is used as a crossing, no bracket
   is assigned parent/vacuum material, no resolved prefix is offered for transport, and no tolerance waives
   a small unknown width. Stored endpoint coalescence is never applied to the omission cells.

Whole-ray refusal is intentional even when the uncertainty is tiny. To admit crossing torus rays later
requires a separately justified quantified boundary policy or stronger boundary solver; reducing probe
cell width alone does not certify an exact boundary.

## Parent replacement and conservative refusal

`navigate(gdml, census, origin, direction, lengths, hits, *, endpoint_budget_mm, ...)` calls the existing
actual-solid omission probes on every unsupported census placement and every supplied ray. It validates
census class counts, unique physical paths, placement assumptions and the bounded selection domain. The
replay runner additionally regenerates the full census and requires exact agreement with the saved census.
The selection-box roundoff guard is recorded; it is not material, a general geometric clearance or a new
boundary-resolution allowance.

`compose(...)` is the lower-level trusted-evidence compositor. It requires exactly one report for every
unsupported placement and checks source/material/scope/leaf metadata, recognized states, actual Boolean
verification flags, positive finite in-domain cells, and report length bounds/counts against the retained
cells. A missing/duplicate/orphan report or inconsistent evidence raises. An unevaluated placement blocks
**all** rays because its absent cells do not establish exclusion. Standalone `compose` assumes the supplied
probe tables belong to these rays; use `navigate` plus the runner's immutable provenance for a bound replay.

For certified inside cells, whole-cell ancestor containment and leaf verification are required. Intersections
against the strict complete stored partition must agree with the immediate parent material everywhere.
A mismatch refuses the lane even when signed cancellation would hide it. Possible-cell overlaps are checked
independently of material names, including two siblings with the same material and duplicate source cells.
An overlap of conservative boundary bands can therefore over-refuse rather than silently choose priority.
Raw stored terms are also intersected with each certified cell before replacement: any negative term,
simultaneous positive terms, or different raw material refuses the lane as ambiguous stored-parent ownership.
This catches retained same-material siblings that otherwise cancel against parent subtraction. It can also
refuse valid signed Boolean decompositions: placement identity is not recovered from the loader's lossy labels,
so ambiguous ownership is not guessed. Raw ownership comparisons are persisted separately.

Eligible certified inside cells produce explicit signed terms:

- `+1` leaf material;
- `-1` immediate nonvacuum parent material (no subtraction for the explicit vacuum convention).

The signed replacement is swept again. Only lanes passing every gate enter the complete output, including
exact vacuum gaps. All cells and parent comparisons remain evidence, including those on refused rays.
`replacement_terms` is a **partial diagnostic**, never a complete partition; its source/cell identity and
operation are retained. `resolved=True` in an omission report means evaluated, not boundary-resolved.
The inherited loader/probe vacuum-name convention (including air names) is unchanged; this is not the
separate transport air-versus-vacuum physics arm.

No per-ray Python loop is used. Event/ray rows, sweeps, overlaps and parent comparisons are vectorized;
geometry-placement loops reuse the existing probes. A pair-count bound prevents an uncontrolled parent
cross-product. Empty and zero-length requests have explicit decisions and no positive interval rows.

## Replay and immutable provenance

```bash
cd ~/mu2e/remote/analysis
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m trkfastsim.analyses.supportnavigation \
  ~/mu2e/remote-data/trk_fastsim/ledger/navigation_audit_v04_20261025_final \
  --resolution-dir ~/mu2e/remote-data/trk_fastsim/ledger/navigation_resolve_v04_20261025_reviewed \
  --gdml ~/mu2e/remote-data/crv_fastsim/geometry/mu2e_extracted_v04.gdml \
  --outdir NEW_UNIQUE_OUTPUT --budgets 0 1e-10 1e-9 \
  --min-cell-mm 1e-7 --compare-cell-mm 1e-6
```

No ROOT reference is regenerated or reread: the runner consumes exactly the saved seeds/raw hits/census
and prior summary, with round-trip CSV floats. It binds the reviewed resolution to the same input hashes,
requires identical re-probed cells and reports, hashes every consumed file and tracker/core Python source,
and retains the original commit identities and retained-reference counts. It checks raw-hit full identities
against lane identities. Outputs retain `(run,subrun,event,sp)` even for multiple primaries in one event.
Existing output directories are refused; completion is written last. Numerical/input errors abort without
a completion product. A completed replay may still have refused rays and is never transport admission.

Output tables per endpoint budget: `intervals`, `decisions`, `refusals`, `adjusted_hits`, `endpoints`,
`replacement_terms`, `parent_comparisons`, `ownership_comparisons`; shared `seeds`, `omitted_cells`,
`omitted_placements` and a manifest.
Reversed raw-source/cell/report order must preserve decisions, refusals and complete intervals exactly.
Separate one-ULP positive/negative stored-endpoint perturbations keep exact path anchors and record whether
lane decisions change; these are numerical stress tests, not new geometric rays. Omission guards depend on
the full ray batch's coordinate scale; lane-batch invariance of those probes is **not** claimed.
Independent coarse/fine probe bounds remain reported as in the previous resolution.

## Measured v04 replay

Authoritative local output: `~/mu2e/remote-data/trk_fastsim/ledger/support_navigation_v04_20261025_reviewed/`
(9.8 MB; not configured for automatic remote mirroring). The earlier `..._pilot/` is retained, not overwritten.
Reviewed computation was **2.17 seconds**, excluding output serialization; no scaling estimate is implied.
Source hashes are in its manifest and the ledger handoff. All 391 saved rays, 5,896 raw source rows and 76 unsupported placements are retained. The new census,
omission cells and placement reports match the previous reviewed result exactly.

| Stored endpoint budget [mm] | Resolved whole rays | Refused whole rays | Complete interval rows |
|---|---:|---:|---:|
| 0 | 235 | 156 | 4,146 |
| 1e-10 | 383 | 8 | 8,521 |
| 1e-9 | 383 | 8 | 8,521 |

At zero budget, 150 lanes have invalid stored occupancy; eight have unresolved omission boundaries, with
two in both groups. At either positive budget, only those eight torus-crossing lanes remain refused.
All 99 unresolved cells are retained; summed per-cell width upper bound is **7.455355131469334e-6 mm**.
This is a geometric uncertainty-width sum, not a deposition or range-error estimate. No cap was reached.

Both positive-budget arms have 960 matching parent comparisons, zero mismatches and 1,275 signed replacement
terms (960 additions, 315 nonvacuum subtractions). All 315 raw ownership comparisons are unambiguous.
These terms belong to certified inner cells on refused rays. Thus the 383 complete partitions contain **no newly admitted torus crossing**. They must not be used
to claim a torus-corrected transport comparison or an unbiased acceptance sample by dropping the eight rays.
The prior per-placement material length bounds (about 29.4166 mm total, including about 7.149 mm replacing
honeycomb) are unchanged. Both external xtru probes remain excluded on the bounded rays.

Reversing evidence order and applying either one-ULP stored-endpoint perturbation preserve lane decisions
at all three budgets. The two positive-budget complete interval tables agree exactly. This is numerical
consistency evidence, not proof of exact shared-face identity or global geometry completeness.

## Verification

Full isolated local non-JAX tracker suite: **746 passed, 5 strict legacy xfails**, with published-source
capture regression enabled. New suites contribute 134 compositor/navigation tests and 16 runner tests.
Use the full-suite command in `transport-ledger.md`; targeted verification is:

```bash
cd ~/mu2e/remote/analysis
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  trk-segment-history-20261025/tests/test_supportnavigation.py \
  trk-segment-history-20261025/tests/test_supportnavigation_replay.py
```

Tests include actual synthetic GDML tori, analytic chord bounds, tangent/sector +/- and ULP perturbations,
cap preservation, empty/zero/multiple-primary cases, genuine overlaps, material and raw-ownership checks,
report coverage/metadata, immutability, source permutation and no-overwrite/hash mismatch. Independent review
found the retained same-material sibling cancellation hole, unsigned-sign wrapping and partial identity-schema
ambiguity; all now have regressions. Follow-up static review found no new material-safety defect in those fixes.
No GPVM computation was added while all seven hosts had existing user sessions; computation/tests were local.

## Next gate

Keep the eight rays in every denominator and report their refusal, not zero material or transmission.
Before energy coupling, decide whether to retain this strict refusal policy for a deliberately bounded
resolved-ray test, or implement and validate a separately named quantified boundary policy that can admit
the eight crossings. Any physics comparison must state the selection induced by refusal. Then add ordered
wall/gas/wire intervals and restricted-loss/secondary accounting without double-charging gas deposition.
No build, production run, G4 baseline generation, commit, push or default promotion is part of this replay.

## Quantified-envelope follow-up

The separately named paired candidate is now documented in [support-envelope.md](support-envelope.md).
It leaves this refusal-first API and its saved outputs unchanged. At the same fine probe resolution, all
391 rays pass explicit 1 nm connected-band / 10 nm per-ray uncertainty limits at either positive stored
endpoint budget. The eight torus crossings are admitted only as paired inner/outer occupancy envelopes,
not exact boundaries or energy histories. Both parent-material and raw-ownership checks cover the entire
possible footprint, including the unknown bands; uncertainty never becomes a guessed midpoint crossing.
