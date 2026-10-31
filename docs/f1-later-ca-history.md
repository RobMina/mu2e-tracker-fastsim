# Adjacent retained-hit closest approach and crossing-cap evidence

This follows the [initial CAD-state audit](f1-initial-cad-state.md). New standalone
`scripts/f1_later_ca.py` and `f1_later_ca_audit.py` use preserved native captures;
no fitter/C++ edits, refits, builds, native reruns, defaults or tolerance changes.

## Declared population and inputs

Of 61,622 straw-hit snapshots, **31,600 adjacent updates across all 147 captured
targets** have a next snapshot in the same meta-iteration after a successful swap.
All 31,600 are compared, without failure exclusions. The other 30,022 snapshots
comprise 2,619 without a following snapshot and 27,403 across another stage or
nonconsecutive algebraic iteration; they are outside this comparison's scope.

Candidate pieces, native seed mass and the previous captured CA remain inputs.
Measured endpoints/end times/energy and original FCL signal-speed coefficients
construct the sensor line. The following captured geometry/status/state does not
supply generated CA; synthetic population perturbations verify that separation.

Although explicit hint-call snapshots are absent, source ordering establishes the
hint for this bounded adjacency: previous usable CA retains its particle/sensor
times through `updateState(first=false)`, and `KKStrawHit::updateReference` uses
those times. This does not infer unobserved extension or meta-boundary histories.

## Source control flow

The helper uses the immutable local `nativeca.solve`, with native float32(1e-4)
promoted to double, at most 100 local updates, and at most ten piece moves. At
exact boundaries nearestIndex chooses the earlier piece (with begin/end clamps).
The reference pointer advances **before** checking stable index, usability, and
the move cap. Thus a capped pointer may identify an unsolved piece while the
retained vectors/times/derivatives still belong to the last solved piece.

Hit fallback is usable own CA, otherwise sensor midpoint time minus measured drift
time for the particle hint and midpoint time for the sensor. The source hit-z
retry fires for unusable CA or strict |dz|>100 mm, recomputes the hint using zTime
(up to 100 piece-index updates), and retries once. A second failure marks the
**wire-hit state** unusable, not necessarily the **CA status**.

Important geometry distinction: `Straw::origin()` returns `_wmid`, the **wire
midpoint**, exactly represented by the independent dump's `midz`. It is not the
aligned straw-tube midpoint `_smid`. The historical analytic audit used inferred
straw midpoints; the final new audit uses exact dumped wire origin for hit retries.
The historic inferred-straw control still gives the same retry count on this sample.

Crossing fallback is own usable CA, then usable associated-hit CA, then sensor
midpoint. It is implemented and tested synthetically, not validated as an observed
native crossing fallback: no crossing CA sidecar exists.

## All selected retained-hit states agree exactly

For **31,600/31,600 updates**, the generated final values equal captured values:

- particle and sensor time;
- both POCA positions and their difference;
- both direction vectors;
- CA usability and status;
- selected reference-piece identity.

**15,461** first searches move pieces; **zero** hit searches exhaust the cap.
Both unchanged numerical criteria (rtol=1e-5, atol=1e-7) and finite elementwise exact
equality pass. This is sample equality, not universal ROOT/compiler parity.

The first pass alone differs on one sensor time by 2.2737367544323206e-13 ns and
two sensor-position components by at most 4.274625098332763e-11 mm. Source retry
removes those exact differences. The previous snapshot key is
`(1400,0,98246,0,18,3)` in run/subrun/event/iseed/iteration_ordinal/effect_index order.

### The 246 z retries are now accounted for

**246** updates trigger retry and still exceed the z cut on the second solve.
All zTime searches finish in **one or two** updates, none exhausts 100. All retained
CAs remain usable, while hit states are unusable as required by the native branch.
Across the batch, first/final dz ranges from -1128.8745781419786 to
665.7904983855551 mm; the retry does not remove these geometrical outliers.

The full same-meta hit-state transition also matches exactly:

| Category | Count |
|---|---:|
| Previously unusable hit states | 110 |
| Newly marked unusable by retry give-up | 167 |
| Already unusable among the 246 give-ups | 79 |
| Previously unusable persisting without new give-up | 31 |
| Finally unusable hit states | **277** |

The prediction is `give_up ? unusable : previous_hit_state`, not a new CAD decision.
This explains why the older sidecar census of **zero unusable CAs** did not exclude
unusable **hit states**. It also resolves the previous 246 analytic trigger question
within this selected same-stage history, without claiming later CAD-policy closure.

## Two caps are crossings, not straw hits

Keys here end in **input_index**, not effect_index:

| Previous crossing key | Captured pointer | Analytic solved piece | Retained minus captured time (ns) | Re-solved pointer minus captured time (ns) |
|---|---:|---:|---:|---:|
| 1400:0:75919:0:8:8 | 10 | 9 | -4.511093720793724e-10 | -4.565954441204667e-7 |
| 1400:0:81667:0:24:35 | 36 | 35 | +3.92901711165905e-10 | -1.0668663890101016e-5 |

Both analytic searches make ten moves and reproduce the captured final pointer.
Times compare against the observed crossing effect time minus its captured initial
constructor offset (4.1753650293685496e-5 / 4.175365756964311e-5 ns). That subtraction
need not recover the original CA time exactly. Retained analytic times are much
closer than solving on the advanced pointer, supporting the source's retained-state
interpretation, **not certifying native retained crossing vectors**.

This remains the historical analytic/legacy-mass, inferred-straw-geometry control.
No captured crossing sensor times, directions, status, actual hints, piece visits,
or extension constructor/private offsets are available. It is not a full crossing
replay or a new tolerance test; absolute differences are reported without changing
existing criteria.

## Provenance, tests and superseded runs

Final report:
`$M2FS_DATA/trk_f1native_r2_20261027/later_ca_v4.json`

SHA256 **`e0dd9a77bb0ad28351f8d6303bdda79486747294b8dcee2ae1fee30703888ce1`**.
`completed=true`, hashes unchanged; final CA exact/criterion agreement true;
full-history, full-state-policy and full-fit closure remain false.

The receipt pins initial_cad_v4 and all inherited inputs/sources, actual executing
module paths, native search/update sources including Straw.hh, the historical
reference report and independent alignment geometry. Expanded-FCL continuity is
prior-audit evidence, not an original generation receipt.

Earlier versions are retained:
- v1, SHA `a8ca7a35ff6091f350b29fd1bd6307279d8f0952e6d8b68134100045c4a89449`,
  used inferred straw midpoint for hit retry. Superseded despite identical final CA
  on this sample; this is not the source's origin definition.
- v2, SHA `217400378b52dab4627acaf1a9ad4b01e2a95fdc2ff03e694410c4806fca7fbc`,
  corrected origin and added give-up versus observed-hit-state counts.
- v3 adds persistent-state accounting, mask/provenance validation and final-CA gate.
- v4 includes the bounded crossing-cap time comparison and final guard hardening.

Review/tests fixed direct zTime mask/owner validation; conflicting provenance
replacement; strict failure JSON for malformed NaN digests/invalid paths; finite
prior-state/crossing-time guards; and separation of execution completion from final
CA agreement. A mocked zTime fixture needed a positive momentum after validation
was strengthened. None of these changes tunes physics criteria or discards failures.

New tests **99 helper + 78 audit + 39 cap + 68 population = 284 passed**.
Final selected isolated suite **1,896 passed, 1 skipped, 43/43 files**, pinned
core233771b/ANN and CPU/single-thread settings; log
kept locally, not in the repository.
No full/slow-tier certificate.

```bash
bash scripts/run_f1_later_ca.sh \
  $M2FS_DATA/trk_f1native_r2_20261027/NEW_REPORT.json
```

Data usage 141711041678/161061273600 bytes (~18.02 GiB free).

Fitter SHA remains `988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a`. No staging, commit, push,
build, grid submission.

## Next boundary

The subsequent [meta/extension continuity audit](f1-transition-continuity.md) validates
the source/configuration conditions below and compares the remaining 27,403 boundary
updates plus 81 new extension hits. A full-key union covers every captured hit-geometry
snapshot; crossing private offsets, autonomous trajectories and full state policy stay open.
This earlier receipt and its same-stage scope are unchanged.

Extend the observed-hit continuity audit across meta-iteration and extension
transitions only after proving intervening processEnds/createEffects/reference
updates and associations. Keep the missing crossing constructor/private-offset and
retained-state evidence explicit; a further native capture would require a separate
source proposal and user build, not an assumption that the present sidecar observes
it. Saved-F1 chart sensitivity, strict native inversion/filter arithmetic and final
acceptance/fidelity remain unresolved. This checkpoint does not change the 99.5%
acceptance or 99% per-parameter fidelity targets.
