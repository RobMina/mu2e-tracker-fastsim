# Saved-ray boundary and omitted-solid diagnostics

## Scope

This is the next bounded gate after [navigation-audit.md](navigation-audit.md). It replays saved source
intervals and identical reference seeds, without G4 generation, reconstruction, energy/time transport or
production-default changes. No physical `SegmentHistory` is admitted. The strict production and experimental
support navigators remain unchanged. All new helpers live in the isolated tracker worktree, not core.

The two questions are separate:

1. Can an explicit bounded endpoint-resolution policy produce a consistent partition of the stored model?
2. Do the previously omitted solids actually intersect these rays, and which parent material do they replace?

Neither question proves the stored model equals GDML/G4 truth. Full geometry coverage, straw ordering,
energy-coupled scattering/crossings, upstream/exit timing and final stock validation remain open.

## Endpoint policy

`boundarypolicy.coalesce(hits, lengths, max_shift_mm=...)` operates on raw signed rows with source provenance.
It groups connected endpoints within each lane, including exact path-start/end anchors. The entire group
must fit within the caller's absolute millimetre budget; a chain of individually small gaps cannot absorb
a larger extent. The canonical point is the group midpoint, or its exact anchor. Distinct anchors in a group,
out-of-domain coordinates and collapsing any positive-width source interval are errors. All raw endpoints,
signs, row positions, group bounds, canonical points and applied shifts are retained in evidence.

The subsequent `boundarypolicy.partition` is strict: per-material signed occupancy must be zero or one,
different materials cannot overlap, and the output partitions each positive-length ray exactly with explicit
vacuum gaps. No negative-column clipping or implicit material priority is used. Coalescence is opt-in;
zero budget reproduces the original strict behavior. Source permutation and independent lane-batch replay
are checked on the saved data. Actual shift and signed/unsigned source-length changes are reported.

**Sub-budget physical gaps/overlaps are indistinguishable from roundoff using these endpoints alone.** Passing
two nearby budgets is numerical consistency evidence, not proof that a mismatch was roundoff, and is not a
license to introduce that tolerance into production. Source geometry boundary identity remains a stronger
future contract. The same absolute budget must not be silently scaled with world coordinates or tuned to
make a physics comparison pass.

## Actual-solid probes

`omittedpath.PlacementIndex` verifies the original census physical-placement paths, records every ancestor
including the world, and rejects unresolved placement-reference/unit/scale assumptions. Non-leaf omissions
are unevaluated rather than assigned their full solid as material. All unsupported census placements are
probed against the full bounded ray set, not only the earlier unguarded AABB candidate pairs.

### Torus

`torusprobe.torus_intervals` handles a ring torus in local coordinates with full, wrapped or reflex phi sectors.
The distance to its circular centreline, `hypot(hypot(x,y)-R,z)`, is 1-Lipschitz. Outward-guarded AABB clipping
and vectorized adaptive path cells give inside/outside certificates and explicit unresolved intervals.
Sector-plane intervals use AND for sectors up to pi and OR for reflex sectors. Plane crossings are subdivision
hints only, never automatically accepted boundaries. No fragile quartic-root discard determines a miss.

Minimum cell width, evaluated-node count, frontier size and iteration caps are explicit. A cap retains all
pending cells as unresolved; it never declares transmission or zero material. Rays on sector planes can
remain unresolved over positive length. Lower/upper lengths include unresolved cells and directed endpoint
subtraction/summation margins. Standalone per-lane bounds and the integration's filtered/grouped sums both
use outward aggregation.

An explicit caller coordinate-transform guard supplements the local arithmetic guard. The integration uses
an ancestry-depth-scaled 128-epsilon world-coordinate bound; exact formulas and effective values are in source
and per-placement reports. **These are documented engineering float64 guards, not machine-verified interval
arithmetic.** No exact boundary, certified G4 equivalence or universal numerical proof is claimed.

### Nonconvex xtru

For two ordered, unscaled, unshifted mm sections, reuse core ear-clipping and prism halfspaces. Validate that
the polygon is simple, has no zero-length edges, triangulation is complete and areas agree. The union is over
triangles of one solid, not distinct material volumes. Contracted/expanded halfspaces provide an inner/outer
guarded envelope; expanded AABBs include acute-corner amplification. Unsupported section details or
ill-conditioned corners are refused. Internal triangle seams can conservatively remain unknown. A broad
bounding box crossing alone never becomes a material crossing.

### Hierarchy and parent replacement

For retained cells, verify whole-segment containment in every ancestor with supported conservative predicates:
boxes, exactly full-circle tubes, constant-radius polycone sections, and conservative Boolean union/intersection
containment. Near-full sectors are not silently widened. Unsupported ancestry means unverified, not outside
or inside. Parent material is recorded explicitly; a metal pipe in honeycomb replaces honeycomb rather than
adding metal on top of it.

For certified inside cells only, compare the parent against the endpoint-adjusted stored partition. A separate
partial trial adds the leaf material and subtracts its nonvacuum parent, then reruns strict occupancy checks.
This tests sibling/stored overlap only on those cells. Unresolved boundary brackets are not filled, assigned
a material, or promoted to a complete corrected geometry. Reported summed lengths are per-placement values,
not automatically net material columns or energy-loss changes. Unevaluated placements have no numerical
upper bound and must remain visible; sums for evaluated placements are not global omitted-geometry bounds.

## Runner and provenance

```bash
cd ~/mu2e/remote/analysis
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m trkfastsim.analyses.navigationresolve SAVED_AUDIT_DIRECTORY \
  --gdml EXISTING_V04.gdml --outdir NEW_OUTPUT \
  --budgets 0 1e-10 1e-9 --min-cell-mm 1e-7 --compare-cell-mm 1e-6
```

The prior audit must be completed and match the GDML SHA256. CSV uses round-trip binary64 parsing; no seed
coordinates are regenerated, re-rounded deliberately, or back-projected. Saved seed/row bounds reject excess
rather than truncating. The runner hashes every consumed saved file and all current tracker/core sources;
base commits and prior reference counts are retained. These remain retained-reference seed counts, never an
acceptance denominator. Existing output directories are refused and completion is written last.

Outputs include per-budget adjusted hits, endpoint evidence, successful stored partitions and partial parent
replacement comparisons; omitted-solid inside/unresolved cells with full particle identity; per-placement
geometry/hierarchy/cap/guard summaries; and a final manifest. `--compare-cell-mm` independently re-probes the
same rays at coarser resolution and reports interval overlap, width refinement and stable hit counts.
`--skip-omitted` isolates boundary work and explicitly writes no omission result. All outputs state
`admitted_for_transport: false`.

## Validation and next gate

Final local isolated non-JAX tracker suite: **596 passed, 5 strict legacy xfails**, with published-source
capture regressions enabled. New suites contain 114 boundary-policy, 61 torus, 15 omission-integration and
7 replay-runner tests. No core or production tracker source changed in this follow-up.

Synthetic suites cover endpoint chains, genuine overlaps, thin-interval collapse, signed material accounting,
empty/multimuon cases, permutation/batching, torus analytic chords/sectors/tangencies/caps, conservative prism
bounds, parent containment and replacement. Review fixes include directed integration aggregation, exact-full
parent-sector admission, preservation of repeated-z polycone plane order, world-to-local uncertainty guards
and expanded-prism broad-phase bounds.

Exact measured v04 results, final test counts, hashes and local output locations are retained in the internal
tracker ledger handoff wiki. Next turn the bounded missing-solid evidence into a separately named navigation
candidate with explicit unresolved-boundary refusal (or a justified quantified endpoint policy), parent
replacement and immutable provenance. Recheck real rays and boundary perturbations before support transport;
then add ordered wall/gas/wire intervals with the restricted-loss/secondary convention. Do not charge gas loss
twice or treat this diagnostic as completion of Tasks 2/3 or TV2/TV3.

## Refusal-first candidate follow-up

The separate candidate and same-ray replay are now documented in [support-navigation.md](support-navigation.md).
It preserves the endpoint diagnostic and omission probes unchanged, adds parent replacement with conservative
raw-ownership checks, and refuses whole rays on any unresolved solid cell. At either positive endpoint budget,
383/391 rays have complete candidate partitions; eight retain explicit refusal. No energy history is admitted.
The prior partial replacement trial alone did not exclude retained same-material sibling cancellation; the
candidate now checks raw stored terms and refuses that ambiguity independently of net material occupancy.
