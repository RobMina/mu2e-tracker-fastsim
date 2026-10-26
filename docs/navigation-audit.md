# Bounded stored-geometry navigation audit

## Purpose

`python3 -m trkfastsim.analyses.navigationaudit` checks the ordered-support navigation prerequisite using an
existing GDML export and one retained G4 reference table. It does not run G4, energy transport, scattering,
flight-time integration, stock digitisation/reconstruction or production acceptance. An audit completion is
not navigation admission: the manifest always says `admitted_for_transport: false`.

The experimental support history still refuses loader omissions and positive-width interval overlaps.
This audit preserves those failures and their contributors; it does not install a tolerance, clip negative
occupancy, silently union materials or change production defaults.

## Inputs and selection

Read explicit highest cycles of `dump/gen`, `dump/particles`, and `dump/surfsteps`. Record object names,
actual entries, input SHA256 and GDML SHA256. Only required branches are read, with an explicit per-tree row
bound (default 2,000,000, excess rejected). The `gen` table in a retained reference is not the full generated
sample; neither its row count nor the selected seeds is a generated-event acceptance denominator.

Select primary muons using `ismcprim == 1` and `abs(pdg) == 13`; preserve `(run,subrun,event,sp)`. Duplicate
primary identities or nonfinite relevant surface times are errors. Pick the first time-ordered tracker-envelope
SurfaceStep (surface IDs 0, 2, 4), breaking time ties by its source row number. Use its endpoint midpoint in the
Mu2e world frame and stored momentum. This is the existing envelope diagnostic convention, not proof of an
inward boundary crossing or a new exact G4 entry point. Only finite, nonzero-momentum seeds inside the support
selection box are used. Counts explicitly report invalid states, outside-box states and zero-length rays.

The selection box is `panel_boxes ± 300 mm`, matching `TrackerInterior`. Straight rays are limited to the
requested length and clipped at their first box exit. Seeds are ordered by full identity; at most `--max-rows`
(default 256) are selected, with eligible, selected and bounded-out counts. This is a deterministic bounded
subset, not random sampling or a rate estimate. The output seed table retains original surface row, surface
ID and reference time as well as full identities and world-frame positions/momenta.

## Geometry evidence

`navigationcensus.census` walks physical placements, pruning straw/wire subtrees and classifying tube,
calorimeter, intentional exclusions, unsupported shapes and existing approximations. It inventories tracker
and calorimeter descendants and possibly relevant external placements. Physical placement paths include sibling
indices; geometry loops are allowed and ray calculations remain vectorized.

Conservative AABBs cover supported primitive bounds plus torus, cone, xtru, arb8, polyhedra and transformed
Boolean union envelopes. They are not a replacement solid navigator. Unresolved references, nonidentity
scales or unsupported units/bounds remain explicitly unknown. Identity scales do not invalidate bounds.
The census lists the following loader limitations instead of hiding them:

- Tube-loader Booleans retain their first operand; cutTube cut planes are ignored.
- Tapered polycones use enclosing cylindrical sections; coaxial packing rounds radii and snaps axes.
- Calorimeter round solids use the existing equal-area polygonal approximation.
- Unsupported solids can be omitted. `calo_skipped` is global to the subtree, not already bbox-filtered.
- Tube-only support has no general mother/daughter material subtraction or material precedence.

`omitted_aabb_candidates.csv` intersects bounded rays with conservative boxes of **unsupported** solids only.
These are candidate pairs, not proven solid intersections or missing material lengths. Approximation discrepancies
are a separate problem. Large nonconvex external solids can have AABBs spanning the entire tracker while their
actual material does not. Unknown bounds cannot be cleared by a no-candidate result.

`materialpath.tube_hits` exposes the existing analytic stored-tube intersections without vacuum or overlap
admission. `tube_intervals` uses the same helper and retains its strict partition check. Calorimeter diagnostics
use the existing convex clipping primitives and packed signed pieces. The audit sweeps exact boundaries both
per material and over all materials, retaining multiplicities outside 0/1. Fault tables include component,
full identity, endpoints, signed contributors and world-frame witness positions. Tube ordinals join to the
catalogue of loader paths; calo labels are volume groups plus packed bucket/piece indices, not unique placement
names. Material and total fault rows can describe the same physical defect and must not be summed as unique
physical overlaps. Component summaries likewise repeat combined evidence.

Tube interval sums are compared to existing sampled columns for `nsub=8,32,128`. These are stored-model numerical
comparisons, not independent GDML truth. Raw sums remain available even when strict history admission fails.
Neither sampled-sector convergence nor very small occupancy spans qualify a tolerance policy automatically.

## Reproduction

Use isolated tracker/core sources and a new output directory. No build is necessary.

```bash
cd ~/mu2e/remote/analysis
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m trkfastsim.analyses.navigationaudit EXISTING_REFERENCE.root \
  --gdml EXISTING_EXTRACTED_V04.gdml --outdir NEW_OUTPUT \
  --tracker-ref FULL_TRACKER_COMMIT --core-ref FULL_CORE_COMMIT \
  --max-rows 512 --length-mm 2000
```

The reference/GDML may be fetched once with documented `remote-sync fetch`; no reference is regenerated.
Outputs include `summary.json`, `loader_census.json`, `seeds.csv`, `raw_hits.csv`, `faults.csv`,
`fault_witnesses.csv` and `omitted_aabb_candidates.csv`. Existing directories are never overwritten.
The final summary records all loaded tracker/core Python source hashes in addition to caller-supplied base
commits, effective settings and command line. Completion is written last. Output remains local unless its
data path is explicitly configured for mirroring; the source-tree mirror is separate.

## Tests and remaining gate

Local isolated non-JAX tracker suite: **399 passed, 5 strict legacy xfails**, including 18 audit tests and
51 census tests. Published-source capture regressions remain enabled. Existing tracker/core transport sources
were not changed in this audit session. Test command is the non-JAX command in `transport-ledger.md`.

The first actual-v04 runs exposed strict boundary refusals and unsupported-solid candidates. Exact inputs,
measured counts, output locations and source hashes are recorded in the internal tracker ledger handoff wiki.
Next resolve boundary identity/roundoff with a documented, bounded policy and replayable witnesses, retaining
rejection of genuine overlaps. Resolve unsupported torus and external nonconvex candidates using real solid
occupancy/hierarchy or an explicit proven exclusion, not AABB lengths. Then repeat this bounded gate before
ordered wall/gas/wire work and energy-coupled scattering/crossings. Do not infer transport-energy/time closure,
full-geometry coverage or final TV2/TV3 acceptance from this audit.

The follow-up [navigation-resolution.md](navigation-resolution.md) replays these saved rays with an explicit
endpoint-resolution diagnostic and actual torus/nonconvex-prism probes. Production and strict admission remain
unchanged; the internal handoff contains its measured results and reviewed output tag.
