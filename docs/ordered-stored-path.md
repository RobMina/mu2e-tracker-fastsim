# Ordered stored-material path candidate

## Scope and explicit API

`trkfastsim.orderedpath.transport(..., mode="ordered-stored-straw-mean-v1")` is a separately named, opt-in,
straight-ray diagnostic. It does not change `make_steps`, NumPy/JAX production defaults, RNG order, the
entry writer, digitisation, reconstruction or secondary placement. The production terminal, stopped-path
extension and step/vertex clock xfails remain required gates; candidate tests are not replacements for them.

The candidate composes analytic intervals of the **stored** tube/signed-convex support representation with
finite concentric straw cylinders: outer aluminium, Mylar, gas, gold plating and tungsten core. Intervals
include vacuum and are ordered in distance for each full `(run,subrun,event,sp)` identity. Positive overlaps
are rejected, including equal-material overlaps. No parent precedence or average material is invented.
Straw layers use the stored gas half length; end plugs are not inferred. `wire_r` is the outer plated radius,
matching the production transport convention. Stored support loaders can omit solids or approximate them;
strict navigation of these arrays is not complete GDML navigation.

The tracker-local `segmenttransport`/`orderedtransport` helpers use the existing core restricted mean and
Urban sampler. The public candidate deliberately selects deterministic mean evolution. Each live lane
advances one ordered material interval at a time, with sequential momentum updates and bounded adaptive
slices. Maximum step, fractional loss, model cutoff and iteration cap are explicit. A cap is an error, not
transmission. Initially sub-cutoff lanes end at zero path. Stops retain residual kinetic energy and are
absorbing; no downstream extension is generated. The preserved isolated sampled modes are experimental
helpers, not a production stochastic range certificate. Deterministic mean stopping is not the expectation
of stochastic stopping.

## One state and clock

The result contains a `SegmentHistory` in detector coordinates, terminal arrays in supplied lane order, and
the requested material partition. Positive material slices close only **primary kinetic debit**; deposited
and escaping transfers remain explicitly unmodeled/NaN. In particular there is no independent gas-response
sample or pasted-secondary debit added to force closure.

Time is the exact linear-in-slice kinetic integral supplied by `flighttime.linear_kinetic_dt_ns`.
`orderedpath.at_entries` and `orderedpath.surface_steps` query that same history, including partial slices.
There is no entry-to-entry smearing across vacuum/material boundaries, separate vertex clock, fabricated
world history, displacement flight time, approach segment or exit continuation. Stops are available terminal
states but not live transmissions. This resolves analogous candidate bookkeeping contracts only, not the
production xfails or physical deposition/time validation.

## Bounded real-geometry probe

```bash
python3 scripts/orderedpath_probe.py SAVED_CAPTURE_DIRECTORY \
  --gdml EXTRACTED_V04_GDML --tracker-ref FULL_TRACKER_HASH --core-ref FULL_CORE_HASH \
  --max-lanes 16 --outdir NEW_DIRECTORY
```

The probe consumes initial states from a hash-verified completed passive capture; it never regenerates
primaries or infers an acceptance denominator. It attempts the first 16 lanes as one vectorized batch over
2500 mm and records strict geometry/integration refusals. Omitted calorimeter solids are listed even if the
stored path succeeds. No result admits the actual geometry for production. Input/source hashes bind the
report; existing output directories are refused.

## Outstanding production gates

Complete geometry/ownership including omitted solids; coupled scattering and crossing recomputation;
world/interior/exit continuity and physical stopping policy; gas/secondary partition; stochastic range and
same-identity G4 endpoint/time validation; current-code primary rate; stock digi/reco. A terminal-array patch
alone would change exit propagation while leaving these contracts unresolved. Do not promote this candidate
or reinterpret a passing synthetic test as a repair to `TrackerInterior.transport`.

## Integrated NumPy opt-in

`make_steps(..., stage="C", interior=..., navigation="ordered-stored-mean-v1")`
(or `steps --navigation ordered-stored-mean-v1`) now exercises a separate ordered
policy through the actual interior/writer/surface path. Legacy remains the default.
It refuses JAX, library secondaries, passive legacy capture and legacy ledger-arm
combinations. It does not assert GDML completeness or production certification.

The policy advances deterministic unrestricted mean loss in at most 10 mm slices,
limited to 5% of current KE, stopping absorbingly at 0.5 MeV residual KE. Every
material slice uses current energy, receives an angular Highland/tail kick at its
end and recomputes downstream crossings after a changed direction. There is no
transverse-displacement approximation or reused stale restart. The default finite
requested length is 4000 mm; it is not a geometry world-exit certificate. A cap
raises instead of being interpreted as transmission. Gas transport can have
multiple physical slices within one straw, including a truncated final slice. The
writer groups by primary/straw, sums eIon and actual gas arc length, and retains
the earliest entry and final exit: integration slices cannot inflate PrimaryFilter
step counts. A curved in-straw path is represented by that endpoint chord while
len retains the actual arc. Downstream digi/reco compatibility and convergence
still need validation.

Same-medium stored support is an exact set union (`stored-same-medium-union-v1`):
no endpoint tolerance, no cross-medium priority, and genuine same-medium overlaps
are charged once under this declared material-field interpretation. This resolves
numerical same-medium double columns without claiming physical placement ownership.
Support/straw and cross-medium overlaps still refuse the entire batch.

Changed directions at world-coordinate material faces expose a separate numerical
resolution policy: a leading navigation span no wider than 1e-9 mm is relabeled to
the following span. Its path and time are retained, not skipped. The cumulative
relabeled extent is stored per lane as `boundary_relabel_mm` and must not exceed
1e-7 mm; a chain cap refuses. These are engineering resolutions, not a proved
GDML error bound. Genuine sub-resolution layers can be relabeled. This is distinct
from same-medium union; positive gaps there remain exact. Sensitivity to these
limits and real-geometry refusals must be reported before any promotion.

One slice state supplies entry energy/time, terminal arrays and surface partial-KE
integrals. Surface projection recognizes explicit `kinetic_MeV`/`time_ns` vertices;
legacy vertices take exactly the existing branch. No untraversed approach/tail
extension or separate legacy exit-scatter pass is added in ordered mode.

Restricted gas samples are capped by the slice's primary mean debit. The unused
transfer is `unassigned_gas_MeV`, **not** demonstrated escaping secondary energy;
`escaping_gas_MeV` stays NaN. This is an explicitly biased experimental response
policy, not Urban closure. Support deposition/escape also remain unmodeled. The
primary energy sum closes algebraically; the full energy partition does not.

Ordered CLI products use a distinct `fasttrkC_orderedmean` prefix and a JSON
navigation/source/geometry receipt. Existing ordered products are refused; legacy
names and overwrite behavior are unchanged. The deterministic final slice ends
at exactly the model cutoff by its local linear stopping-power law, avoiding
step-end overshoot of that cutoff. This is still not a validated stochastic range.

### Optional signed-convex endpoint arm

`ordered_options={"signed_endpoint_budget_mm": ...}` explicitly invokes bounded
endpoint canonicalization **before** signed occupancy. Default `None` leaves
strict signed navigation unchanged. Equal endpoints, absolute-budget connected
groups, exact path anchors and maximum shifts are recorded; any transitive group
wider than the budget or collapse of a positive-width source interval refuses.
No material takes precedence over another. Changed raw endpoints/signs/lanes and
iteration are retained in `interior.last_endpoint_evidence`. The arm cannot
restore omitted geometry, prove GDML hierarchy or distinguish genuine sub-budget
features from floating-point slivers. Do not increase the budget or drop thin
source intervals merely to pass a geometry sample.

`ordered_steps_smoke.py` runs a bounded whole-generated-event sample through the
real world/interior/writer/surface chain and preserves a complete refusal report.
`orderedinterior_probe.py` instead consumes saved entrance states (or an explicitly
named full muon identity); it is not an inclusive generator acceptance test.
`ordered_probe_compare.py` requires identical ordered identities and input hashes.
Different boundary/step counts can realign shared random streams, so stochastic
paired differences are not deterministic error bounds or convergence certificates.

When the signed-endpoint arm is explicitly enabled, the same declared leading
resolution applies **before** signed occupancy too: source intervals beginning
exactly at the current ray origin and ending within the leading resolution are
retained in `last_leading_source_evidence`, excluded from signed source terms,
and their unsigned widths count toward the per-lane boundary relabel budget.
The physical partition still covers the full path/time; the tiny leading field
uses the resolved continuation rather than a guessed cross-material priority.
Interior thin-source collapse remains a refusal. Both signs are retained in the
evidence, and unsigned budget accumulation prevents cancellations hiding extent.
This is an approximate engineering resolution arm, not an exact GDML navigator.

## Sample-coupled experimental arm

`navigation="ordered-stored-sampled-v1"` uses the same navigator, absorbing cutoff,
state clock and merged-straw writer. Support/wall/wire use unrestricted Urban
proposals at current momentum. Gas uses its restricted Urban proposal directly as
primary energy debit plus a deterministic nonnegative unrestricted-minus-restricted
mean transfer. The latter stays **unassigned**, not assumed secondary escape.
Thus ordinary gas samples are not clipped to an independent mean budget.

A proposal reaching the available KE above the positive cutoff shortens its slice
by the available/proposed fraction. Gas deposition is scaled by the same fraction;
final residual KE equals the cutoff. This is an explicit constant-rate-within-slice
model, not an expectation-preserving truncation or validated G4 range law. Adaptive
step lengths still use the unrestricted mean. The mode records its identity, gas
sample/applied sums and clipping count. Mean and sampled products have distinct
`_orderedmean` and `_orderedsampled` prefixes. Neither replaces legacy defaults.

The optional endpoint arm applies to both raw tube intervals and signed-convex
sources with the same caller-supplied budget; their subsequent strict occupancy
checks still reject unresolved cross-material overlap. The CLI spelling is
`--ordered-signed-endpoint-budget-mm`; it is rejected with legacy navigation.

## Controlled convergence and reference-entry diagnostics

`ordered_uniform_study.py` calls the actual interior API in a homogeneous finite cylinder, with
independent history batches and an independent quadrature of the existing mean stopping-power model.
`ordered_uniform_summary.py` reports range/time shifts, independent-history errors and descriptive
KS tests; distinct seeds do not imply same-history coupling. The entry-evaluated 10 mm / 5% policy
shows percent-level range sensitivity. A passing primary balance is not a precision range result.

The explicit `ordered_options={"stopping_evaluation": "midpoint"}` arm evaluates stopping power and
loss proposals at a half-mean-loss predictor energy. The entry evaluation remains the default in both
ordered modes. The midpoint proposal uses the same restricted gas-response/unassigned-transfer
partition; it is not a newly validated straggling model. Overshoot scaling still assumes a constant
within-slice rate, and the flight-time integral still assumes linear within-slice KE. Range convergence
against continuous mean loss and stochastic step refinement are separate gates, neither a G4 certificate.

`ordered_g4_state_compare.py` first checks the saved reference generation records, then uses generated
all-particle identity and straw, excluding ambiguous recrossings. It does not replace zero matches by
nearest-time/position matches. `ordered_g4_entry_replay.py` instead starts from the G4 primary's first
straw entrance and can select the lowest-momentum entrances explicitly. That is a **truth-conditioned
diagnostic** which removes world delivery from this comparison, not an allowed production response or
correction model. Later shared-straw time/KE errors use event clusters, exclude the tautological initial
entry, and cannot establish an inclusive rate or a common terminal stopping boundary.

## Actual-slice response representation probe

The bounded `ordered_steps_smoke.py` product now retains `traversed.npz`: full primary keys in interior-lane
order, absolute start clocks, terminal arrays and every actual physical slice. This is an additive diagnostic
product; it does not alter production rows, selection or RNG draws. Its unchanged merged ROOT tables can be
checked exactly against an earlier smoke before response work.

`ordered_response_probe.py` checks gas energy/arc sums, first/final endpoints and entry clocks against the merged
writer. It reports curvature, integration-slice multiplicity and non-gas gaps inside each merged primary/straw
span. All whole-generated-event keys, including empty digi/fit events, enter the unchanged NumPy digitizer and
fast F2. The diagnostic arms are merged rows, actual gas slices, and straightened gas slices with identical
slice energy/time but positions placed by cumulative gas arc fraction on the merged endpoint chord. The latter
is an explicit counterfactual geometry, not a response correction. Independent response seeds are retained.

Slice rows must never be used as PrimaryFilter inputs: numerical subdivisions would inflate step multiplicity.
Neither a successful consumer run nor agreement on a bounded primary-only sample validates stock KKLine,
secondary energy partition, physical stopping, omitted geometry or inclusive rates. A sensitivity to slicing
can include cluster-allocation/ion-type/time effects; it is not solely a measurement of chord curvature.

Before a stock-consumer attempt, `ordered_stock_preflight.py` requires exact all-particle source-prefix records
and fast-muon identities. Fast keys count muons within each event, not all generated particles. The existing
art producer's same-PDG generator-order fallback is retained; it is not confused with the G4 all-particle key.
Step and surface keys must resolve to actual table primaries, which must be unique and parentless. A source
prefix dump made with the trusted built module proves coverage of zero-hit generated events as well as hits.

`ordered_geometry_impact.py` further joins actual omission cells to full primary identities, filter-selected
events, gas entry chronology and model stopping states. A cell outside a tracker observation envelope can
precede the first gas entry and alter incoming transport; it must not be dismissed as an irrelevant exit tail.
After the last gas entry is not necessarily after its exit or every VD observation. Per-placement lengths are
not net missing energy. Geometry ownership, the world/interior handoff and re-evolved transport remain required
before attributing a rate shift or constructing a material correction.

## Selected incoming torus and upstream query evidence

`ordered_torus_pair_probe.py` consumes the actual selected incoming prefix and explicitly loaded, hash-recorded
preserved `supportnavigation`/`supportenvelope` algorithms. It retains raw labelled tube/signed-calorimeter
hits and recomputes every unsupported placement on those rays. The existing zero-unknown refusal control and
reviewed 1e-6 mm connected / 1e-5 mm per-ray unknown budgets remain distinct. Both complete geometric arms,
raw ownership comparisons, unresolved cells and parent-replacement terms are saved. This is not wired into
energy transport; it cannot silently substitute a nominal torus boundary or certify the full geometry.

`ordered_world_capture.py` separately replays the actual upstream stage on the exact original generator
prefix, recording the world-column calls, full primary keys, material columns and residual AIR. Off/on state,
integrated time, RNG state and seed-sequence state must be exact. Comparison against the saved CSV plane
momenta separately allows the recorded decimal roundtrip at relative 3e-16; the maximum difference is reported.
The wrapper restores the existing column function and never edits core or draws random numbers. Column sums
are not source ownership or a physical material partition, and residual AIR is the existing computational gap
policy. A successful passive capture alone does not establish a consistent world/interior material handoff.

## Conditional guarded torus evolution candidate

`ordered-guarded-torus-sensitivity-v1:{inner,outer}` is a separately named experimental navigation provider,
not a new default or a physical-world ownership certificate. `orderedinterior.transport` accepts an explicit
`navigation_provider`; without it the existing stored navigation is unchanged. A provider must return complete,
positive, contiguous disjoint partitions with explicit support/thin/vacuum provenance. Loss, exact slice clock,
absorbing cutoff and angular kicks use the same ordered evolution implementation. A changed direction invalidates
the cached partition, so both geometry envelopes are recomputed on the actual new ray before advancing further.
Exact-touch equal support media are merged to avoid numerical probe subdivision creating extra kick boundaries.
Straw walls/gas/wires refine the chosen support arm strictly; even equal-material overlaps refuse, with no priority.

The guarded provider consumes the explicitly hashed reviewed geometry modules, a full unsupported-placement census,
raw labelled signed stored sources and the unchanged cell/component/per-ray unknown budgets (1e-7/1e-6/1e-5 mm).
The explicit 1e-10 mm stored endpoint policy and separately declared leading-source policy are applied once and
recorded; omitted-cell endpoints are never moved. Composition of the adjusted terms uses zero additional shift.
All existing whole-possible-footprint parent/raw-source ownership, caps, overlap and reason-evidence refusals remain.
A failed pair refuses the whole query before any slice evolves. Each call retains both geometric arms and original
cell/reason/ownership evidence. These flags remain geometric admission only, not retroactively transport admission.

`python3 -m scripts.ordered_torus_evolution` runs only the saved selected incoming witness. Its immutable initial
position/direction/KE is the actual saved post-world, post-backprojection interior handoff, not a truth-conditioned
G4 entry. It separately evolves stored control, inner and outer geometry in mean and sampled midpoint modes,
with tail disabled, finite requested arc through the baseline first gas path plus 10 mm. Finite arc is not a world
exit and a deflected arm need not reach gas; no extension is manufactured. Every possible replacement footprint
must lie strictly below the upstream world endpoint plane on each new ray. That is a local stage-separation gate,
not proof that legacy aggregate world columns/backprojection are globally disjoint or exact physical ownership.
Upstream state and time are held fixed, never corrected after transport. Existing aggregate world capture is not
interpreted as source-placement ownership or permission for a posthoc steel debit.

Independently evolved inner/outer arms are sensitivity comparisons, **not energy, range or time bounds**. Changed
kicks and sampled losses can change later crossings, RNG consumption and stopping; equal seed is not a rigorous
common-variates uncertainty model. Approximate retained polycones/Boolean ownership, actual upstream handoff,
full GDML stopping and the historical inclusive rate remain uncertified. This runner adds no secondary-library
coupling and does not change legacy/default physics, core, reconstruction or any stock binary.

The saved-only `scripts.ordered_torus_evolution_audit` validates each completed arm's primary debit,
physical endpoint law, exact linear-KE slice clock, scalar/position continuity, gas deposit/unassigned partition,
absorbing stop and terminal state, and full retained query/control/refinement receipt coverage. It reports
material path/debit totals and first-gas KE/time/path/position sensitivity against the stored control. It never
converts the independent single-draw arm differences into error bars, rates or geometric energy bounds.

## Raised-plane forward producer candidate

A separate explicit handoff `raised-plane-forward-v1` avoids replaying the overlap between the old low world
plane and the legacy backprojected support top. Its declared plane is above every known tracker/calo conservative
placement bound, loaded support envelope and straw box by an explicit clearance. Unknown bounds refuse planning.
The actual world loader, band edges, MSC endpoint and interior initial position share that same plane. This arm
never calls `_back_to_interior_top` and never subtracts a backprojection path/time. The old default handoff stays
unchanged. Full `(run,subrun,event,sp)` handoff states/dispositions include world-undelivered and empty-gas lanes.
An invalid delivered handoff refuses the whole producer rather than altering alive flags or discarding a lane.
Every fresh guarded interior query must stay below this plane; an upward world re-crossing refuses, not clips.

This establishes a **computational stage split**, not exact world-material ownership or a continuous world path:
world loss remains on the generated straight ray with aggregated per-material columns, legacy unknown-material
fallback/air gap policy and aggregate MSC at the plane. Raising the endpoint does not fix upstream omissions or
approximate Boolean ownership. Retained interior geometry approximations and finite requested interior path
also remain explicit. `scripts.ordered_guarded_producer` uses the actual `make_steps`/writer/surface path on all
requested whole generated events with sampled midpoint transport, fixed engineering budgets and inner/outer
sensitivity arms; it emits no partial physics output when navigation refuses. Each guarded call retains fresh-ray
to-interior-lane maps and full primary identities alongside both geometric/straw partitions, raw source terms,
endpoint/unknown-cell/parent/ownership evidence and original refusal stage. Stored control is a separately named
arm using the same raised plane, not the old lower-plane baseline silently repurposed.

### Physical placement ownership normalization (separate explicit arm)

The first full 48-lane query refused one calorimeter lane: an omitted aluminium torus in a vacuum daughter
intersected the stored signed `+CalCableRunOuter/-CalCableRunOuter` terms of a supported parent/subtraction.
Net material vacuum did not prove physical ownership, so the old raw-source refusal was retained, not waived.
The separate `--physical-ownership` arm (`calo-physical-placement-signed-parent-v1`) retains full physical placement
paths and explicit own-shape/direct-daughter operation labels through the same signed convex loader. Every packed
plane/sign/material/bound tuple must match the old loader's multiset exactly, before using different source labels.
This is a provenance refinement of that stored model, not more accurate GDML geometry.

Only within one physical owner does a strict signed sweep cancel its own shape CSG and verified direct-daughter
subtractions. Daughter labels must name a direct physical child; negative or multiple occupancy per owner refuses.
Normalized positive ownership spans must reproduce the original complete material field on every common atom.
Cross-owner cancellation and positive sibling overlap still refuse, including equal-material siblings. On the
whole possible omitted-leaf footprint, every surviving calorimeter owner must equal the leaf's actual immediate
physical parent, not merely have the same material. All original signed terms, operation/daughter/owner provenance,
adjusted terms and exact material comparisons remain retained, as do the reviewed pair's unknown/cap/overlap/
parent/refusal gates. This policy does not authorize generic same-material unions, hidden negative clipping or
changing probe budgets. The original raw-source control remains independently reproducible.

Tube/polycone stored shell indices remain positive passthrough approximations, explicitly excluded from this
calorimeter physical-cancellation policy. They are not relabelled as physical CSG owners. The production runner's
normalizer authenticates each calorimeter term identifier, owner, material, operation, daughter and sign against
the immutable loaded term inventory before normalization. A forged own-shape label cannot disguise a sibling
negative term; own-shape daughter must equal owner even in the standalone algebra helper. Recursive unsupported
solid/Boolean length/angle units or inline/reference transforms refuse rather than rely on raw numeric units.

### Wider numerical torus arm

`stored-placement-wide-torus-v1` is a separate opt-in arithmetic policy, not a relaxed geometry tolerance.
`widetorus` requires `np.longdouble` to have strictly more mantissa bits than binary64; otherwise it refuses.
`wideomission.WideOmission` promotes binary64 world rays and the already accumulated binary64 placement `R,t`
**before** local subtraction/matrix multiplication. It never regenerates, inverts or reorthogonalizes stored
rotations, reparses decimal radii or changes the full-sector sentinel/angle conversion. Wider spatial `hypot`,
plane/interval arithmetic, speed/transform guards and outward rounding use the wider epsilon for genuinely wider
operations. Stored transform rounding relative to exact GDML remains a model limitation, not eliminated error.

Ray-parameter endpoints, subdivision midpoints/stalls and termination stay on a binary64 lattice. Wider AABB
brackets project outward before subdivision; sector cuts are only subdivision hints. Certified inside cells never
expand during export, unknown cells never silently disappear, and binary64 endpoint subtraction/bincount length
reductions retain their binary64 error factors. All caps, reason histograms, whole-footprint parent ownership and
unknown-component/per-ray budgets remain unchanged; ancestor checks use the unchanged reviewed hierarchy code.
This is an engineering stored-model enclosure, not machine-verified interval arithmetic or G4/GDML certification.

Saved actual refused-query comparison `ordered_refusal_refinement --wide` admits 13/13 pairs at
1e-8 and1e-9 mm cell widths, largest unknown component 5.25871e-7/6.57340e-8mm, versus unchanged binary64 control
12/13 (1.53189e-6/1.07603e-6 mm). The 1e-6 mm component and 1e-5 mm ray budgets are identical in all arms; original
refusals/cells are retained. This numerical comparison enables a separately named all-lane producer trial, not
a claim that earlier binary64 refusals were physical misses or that wider arithmetic makes complete GDML.

### Expanded-source and mixed-family gates (not yet larger-producer admission)

The 10k forward producer's initial 592 rays exposed 3 calorimeter honeycomb parents outside the original loader
selection and 11 external concrete xtru uncertain-band cases. Original torus-only policy refused the whole producer.
`RaySelectedSources` can grow stored-source selection to enclose complete actual fresh rays, without clipping them.
It retains per-query selection/census generations and authenticates terms against the matching loader generation.
A separate local-plane numerical arm intersects original stored binary64 primitive planes using promoted stored
placement `R,t`, avoiding already-rounded global plane offsets; it is not silently substituted for global-plane
controls. On the saved 592-ray source gate it restores every parent/owner without changing endpoint budgets.

`mixedenvelope.MixedEnvelope` separately authenticates `xtru-plane-offset-v1` guarded contracted/expanded prism
certificates: exact scheme/method, finite guard, explicit false cap flag/aliases, complete counts and reasons.
Xtru bands are not torus `min_width` cells. Original strict and torus-only controls stay unchanged. Unknown union
components span all placements/families and use identical 1 nm/10 nm budgets. This gate alone admits 589/592 using old
sources; actual expanded source+census must pass too. Refreshing the full ray-domain census exposes 125 unsupported
placements (not 76);18 unimplemented external placements currently refuse the whole mixed batch. No partial 10k
physics sample is emitted and no irrelevant-exterior assumption excuses them.

`csgenvelope` is currently a **diagnostic-only** planar CSG occupancy candidate for the newly exposed leaf Boolean
volumes. For signed piece occupancy f, its lower/upper bounds are positive-inner minus negative-outer and
positive-outer minus negative-inner. Common atoms are inside only when lower>=1, outside only when upper<1,
otherwise unresolved; negative lower or upper>1 are uncertainty, never clipped. Inner endpoints round inward,
outer outward; coplanar uncertainty remains. Supported box/xtru/Boolean units, transforms, triangulation and
construction/work caps are explicit, nonfinite guards/offsets refuse. This does not relax mixed eligibility or
certify hierarchy/parent replacement. Nonleaf boxes proven missed by conservative bounds need a distinct
ray-bound miss certificate, never a fabricated leaf flag. These later gates are not wired into the current
completed 1000-event transport, whose frozen source/outputs remain intact.

### Ray-bound inclusive saved gate (separate, not producer admission)

`support-navigation-ray-bound-miss-planar-csg-v1` binds generated evidence to GDML SHA 256, every census record,
actual query rays, lengths, cells and reports. Unsupported boxes use independently reconstructed outward stored-R,t
bounds with a padded slab exclusion. Only an all-ray miss qualifies; no nonleaf is relabelled as a leaf, no report
is dropped and no possible nonleaf hit is excused. The original unevaluated report is retained, together with an
explicit evaluated-miss certificate. Planar leaf CSG uses the unchanged reviewed xtru validity/decomposition checks,
signed occupancy enclosures and hierarchy checks on the full exported possible footprint. Mutated/stale evidence
refuses. The original strict results before and after CSG evaluation are separately retained. The pair compositor
uses only the precisely named miss-conditioned refusal view; it does not fabricate strict interval products.

Saved 592-ray test: 18 inclusive/CSG tests and 9 mixed tests pass in separate processes, preserving the
reviewed namespace-loader replacement refusal. All 125 unsupported reports remain;16 box misses and 2 leaf CSG
placements are evaluated. No parent mismatch or ambiguous physical ownership is found. **438 pairs qualify but
154 rays still refuse**, all for overlapping possible leaf cells. The original1e-6mm connected/1e-5mm per-ray
unknown budgets remain unchanged (maxima 1.0002077e-7/1.7697312e-7mm). This is not a successful producer gate.

Saved overlap localization retains 320 overlap pairs on 154 rays, no same-source pair. All overlap pairs involve
at least one uncertain band; there is no inside/inside pair. Most are floor/foundation same-material interfaces;
others cross concrete/overburden or alcove sibling placements. Small width does not establish shared-face ownership
or authorize cancellation, material priority or union of separate physical leaves. `ordered_overlap_witness` retains
full primary identities and both source/state/material labels. No actual larger producer is launched from the 438
passing subset. A separately justified shared-boundary/multi-owner uncertainty treatment, not a tolerance waiver,
is required before this saved gate can admit the entire batch.

### Explicit exact-predicate and literal/native geometry arms

`stored-binary64-exact-dyadic-planar-predicates-v1` orders roots using exact rational affine predicates on frozen
local planes, already accumulated binary64 R,t and original world rays. It preserves distinct sub-budget roots
and invalid signed occupancy rather than snapping or clipping them. The saved 154-overlap-ray diagnostic finds
8 positive overlap atoms on7rays and2invalid signed occupancy atoms, despite0certified inside/inside pairs in the
coarser guards. Thus the earlier guard overlaps cannot be waived as exact shared boundaries in that model.

`literal-gdml-native-planar-quarterturn-v1` is a separate, explicit geometry-model sensitivity: positions and
polygon vertices are exact literal rational values; integer quarter-turn rotations use exact signed permutations;
other inline rotations use the explicitly frozen core binary64 coefficients. Native polygon winding and recursive
Boolean membership avoid triangulation seams. The same 154 rays/13 sources give0positive overlaps,0 occupancy faults,
159 exact contacts and0coplanar lanes. This does not certify the old model or reproduce G4's floating-point navigator.

A full-domain candidate `native-planar-joint-root-projection-sensitivity-v1` evaluates all eligible planar leaves,
not only previously failing sources. Root topology is exact before export. For one arm every exact root projects
downward to binary64; for the other every root projects upward. Algebraically equal roots share a parameter while
retaining distinct owner identities; separate roots are never snapped. Distinct roots not resolvable on the lattice,
positive exact leaf overlaps, positive-length coplanar contacts and projection widths>1e-10mm refuse. Full outward
possible footprints retain unchanged hierarchy, physical-parent and raw-ownership checks. Directed root brackets
join remaining torus uncertainty under unchanged1e-6/1e-5 mm component/ray budgets. The two concrete paths are
correlated root-projection sensitivities, NOT inner/outer material or energy bounds. Old strict/mixed/inclusive
refusals remain separate controls; this candidate requires a full saved-domain pass before any transport trial.

### Native actual-producer gates and explicit material constants

The complete saved 592-ray native gate qualifies all rays with all 125 unsupported reports retained,35 native planar
leaves/408 root rows and 0 exact native leaf overlaps. This passes a geometry-model sensitivity gate, not G4 truth.
Fresh-ray integration tests exercise both projection arms through actual ordered loss/scattering/re-navigation.
The original frozen stored-plane model/refusal controls remain authoritative for that separate model.

Actual 10k coupling exposed 10 native cells starting at exacts=0,y=handoff-plane with downward directions and no
above-plane span. `exact-downward-zero-path-plane-contact-v1` is an explicit native-only closed computational
boundary: the starting endpoint may equal the plane only at exactlys=0 with strictly downward direction and far
endpoint below. Any nextafter-above point, positive-path contact, upward or coplanar segment still refuses; older
strictly-below default is unchanged. Per-query plane-contact tables and both strict/domain flags are retained and
recomputed by the saved audit. This does not certify world source placement or remove upstream aggregate losses.

The first actual retry then reached `Polyethylene0935`, absent from the frozen core material registry. The optional
tracker-local `ordered-explicit-gdml-material-constants-v1` resolver takes only explicit requested GDML material
names, mm-compatible density/excitation units and elemental mass fractions; it never mutates core `BY_NAME`,
replaces existing constants, normalizes fractions, aliases to a similar material or adds fallback. Actual v04 values
are density 0.935g/cm3, I56.982601836529eV, H.14/C.86 mass fractions. It uses the existing core condensed-material
formula and frozen elemental constants, with those limits recorded. Unknown names still refuse. The material is
passed only through orderedinterior's explicit optional resolver; default behavior is unchanged.

### Bounded query lookahead is not a physical terminal or domain exit

`bounded-in-domain-query-not-path-v1` is explicit and off by default. A kicked upward ray may request a full
remaining-arc straight leg that would cross the handoff plane kilometres of microscopic steps later; that is not
an actual traversed crossing. If the projected crossing is farther than the existing 10 mm max step, this option
requests at most 100mm/half the plane distance, retains that horizon and re-queries if exhausted without a kick.
The 4 m physical arc, KE/time/stop logic and full primary population are unchanged. A crossing within one 10 mm step
still refuses; no world exit/transmission/drop is inferred and no physical path is clipped to the plane.

Both nativev 97 arms completed 44 valid fresh geometry queries then refused a distant future-leg crossing. The one
violating inner ray crossed only after 2545mm (storedcontrol 2428mm),not within 10mm. The updated stored 10 kcontrol
completes with 54 filter passes/3 model stops/0 caps; 10 bounded queries affect one lane, every 589 nonstopped lane still
ends at the exact4marc. All3stops remain absorbing. Savedlookahead and original ledger audits pass. These are
computational/ledger validations, not a physical stopping or acceptance-equivalence certificate.
