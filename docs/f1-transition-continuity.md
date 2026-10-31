# Meta-iteration and extension continuity

This follows [adjacent retained-hit CA](f1-later-ca-history.md). New standalone
`scripts/f1_transition_audit.py` and `f1_transition_census.py` use the preserved
capture only. No fitter/C++ edits, refits, native jobs, builds, default or tolerance
changes. The previous diagnostics and their receipts remain unchanged.

## Source lifecycle and configuration gates

Frozen native sources—not the stale local off mirror—establish:

- The Track constructor appends `Status(0,0,unfit,"Construction")`. `fit()` then
  increments the meta number from history. Consequently captured seed meta numbers
  are **1–4**, extension meta numbers **5–13**, not zero-based 0–12.
- Every captured iteration has **history_index = iteration_ordinal + 1**. This
  checks the constructor record and absence of extra history records in this sample.
- Both FitSettings and ExtensionSettings have **ProcessEnds=false** and
  **BFieldCorrection=false**. There is no processEnds mutation, domain growth or
  domain replacement between the compared snapshots.
- **StrawHitClusterLevel=none**, UseCaloCluster=false; no cluster-side effects.
  Reference strategy is default DOCA, ten piece moves, native float precision.
- At a meta boundary `updateState(first=true)` computes an unbiased local CA and
  runs CAD/ANN state updaters; it does **not assign retained `ca_`**. Stored reference
  CA geometry therefore remains the preceding usable iteration's updateReference
  result. This does not imply unchanged hit state or weights.
- Extension calls `addStrawHits`, then `addStraws`, then `extendTrack` once.
  With field corrections disabled, `createEffects` references **new objects only**;
  existing Measurement/Material updateConfig methods are empty. New Measurement
  construction calls updateReference again after hit construction.
- `Chi2SHUSettings: []` is an unused empty declaration, not an active updater.
  makeConfig selects actual updaters from the exact checked schedule.

The audit checks identical expanded blocks and the full four/nine-stage schedules.
Unexpected end processing, domains, clusters, reference strategy, schedule or native
history fails closed. Expanded-FCL continuity is inherited prior-audit evidence,
not a reconstructed original generation receipt.

## Exact owner continuity across all iteration links

All **3,302 adjacent links / 3,449 iterations / 147 targets** follow a successful
swap. Their full-key and piece-index comparisons give exact equality between the
previous candidate owner and next iteration's old owner:

- piece count and order;
- six parameters and all 36 covariance entries;
- ranges and nominal field vectors;
- mass and charge.

No unequal or missing owners. The link census is **1,770 same-meta**, **433 seed-meta**,
**142 extension entry**, and **957 extension-meta** links; 147 terminal iterations.
Every owner, including terminal and unswapped terminal attempts, receives structural
validation. Actual captured trajectories remain inputs; this is continuity, not an
independent filter construction.

## Retained-hit geometry at the previously excluded boundaries

| Transition | Compared hit updates | Targets | First searches moving pieces | Retry / give-up | Caps |
|---|---:|---:|---:|---:|---:|
| Seed meta → seed meta | 7,520 | 147 | 2,214 | 25 / 25 | 0 |
| Seed → extension entry | 2,478 | 142 | 238 | 5 / 5 | 0 |
| Extension meta → extension meta | 17,405 | 142 | 2,453 | 90 / 90 | 0 |
| **Total** | **27,403** | — | **4,905** | **120 / 120** | **0** |

All compared final times, both POCA vectors, delta, both directions, CA usability
and status, and reference-piece identity match **exactly**, with zero exclusions.
The previous captured CA supplies legitimate source-derived hints. Native explicit
hint calls/piece histories remain unobserved; no claim of compiler-universal parity.

At these first=true boundaries, captured hit state changes **193 / 46 / 3,191** times
in the three groups. Those CAD/ANN decisions are **not replayed** here; unlike the
same-meta checkpoint, previous hit state plus give-up is not the full transition
rule. Retained CA agreement must not be promoted to state-policy closure.

## All 81 accepted new extension hits

Captured object membership identifies **81 new hits in 50 targets**, first observed
at extension miter5/iter0. Exact seed sh_index membership distinguishes them from
initial objects; a late-first-seen initial object is rejected, not treated as new.

The independent arithmetic chain, conditional on that accepted membership, is:

1. Use the captured pre-extension trajectory, whose identity with the next old
   owner is verified, and measured ComboHit float32 z/corrected time.
2. `zTime` provides the particle hint.
3. Measured endpoints/end times/energy and original source-FCL coefficients define
   the SensorLine. Compute `psign = line.direction dot straw.wireDirection`, then
   `htime = measurement_time - (halfLength - psign*wireDist)/speed`.
4. Source-style piecewise CA constructs the hit.
5. Measurement wrapper construction invokes hit updateReference using the
   constructed CA; its retry origin is the exact dumped **wire** midpoint.
6. Compare the resulting retained CA with the first sidecar snapshot.

All **81 final retained CAs, statuses and references agree exactly**. Thirty
constructor searches move pieces; none exhausts. No wrapper-reference piece move,
z retry or give-up. All accepted-membership numeric checks pass: corrected time in
`[range_begin-50, range_end+50)`, strict |correctedTime-zTime|<50 ns, strict |DOCA|<10 mm.
These checks are separately reported and gate success, but **not** candidate discovery:
excluded candidates and input flag selection are not audited. Native constructor
intermediate snapshots are absent, so final agreement is not intermediate parity.

## Complete, disjoint snapshot coverage

A new exact full-key join combines the pinned prior receipts and this report:

| Partition | Straw-hit snapshots |
|---|---:|
| Initial constructor/reference | 2,538 |
| Same-meta adjacency (prior checkpoint) | 31,600 |
| Seed-meta boundary | 7,520 |
| Retained extension entry | 2,478 |
| Extension-meta boundary | 17,405 |
| New extension hits | 81 |
| **Total** | **61,622** |

All partitions are disjoint, equal the entire sidecar population, and have exact
retained-geometry agreement. **Zero uncompared captured hit-geometry snapshots.**
This is all **147 captured targets**, not all 2,319 seeds or the missing gain/loss
population. It remains supplied-native-trajectory/previous-state reconstruction,
not an autonomous trajectory history or complete numerical/state-policy fit replay.

## Crossing construction and partner evidence

The captured target population has **2,619 unique hits** (2,538 initial +81 extension)
and **5,090 unique crossings** (2,538 initial +2,552 extension). No object disappears
or reappears within the captured history. Initial seed objects are all observed for
the 147 captured targets; 2,172 other seed targets are outside iteration capture,
not thousands of missing captured objects.

The extension crossings divide into **81 associated** and **2,471 unassociated**.
Each hit has exactly one same-straw crossing; no duplicate crossing straw keys,
initial/extension partner-class disagreements, or differing first observations.
Initial pairs also match the seed contract exactly.

**Identity caveat:** IterationCaptureWriter fills crossing partner_shindex from
**final collections** when writing, not from iteration-local/constructor snapshots.
Stable stamped partners alone are not a history measurement. The stronger bounded
inference here uses frozen source order and census: one extension, all initial
crossings already paired, addStrawHits makes the new associated crossings, and
addStraws subsequently makes the unassociated material crossings. No additional
extension can attach later hits in this configuration. This is source-conditioned
constructor-partner agreement, **not a captured association snapshot**.

All **2,552 extension private offsets and crossing constructor CAs remain unobserved**.
The 81 associated constructor hints can be further investigated using the hit
construction result, but native straw endpoints and retained crossing vectors are
still absent. The 2,471 unassociated material crossings require hierarchical
plane/panel/straw candidate-discovery history. Do not infer private offsets by
subtracting unrelated later CA times or fitting captured material operands.

## Evidence, superseded runs, tests and cleanup

Final report:
`$M2FS_DATA/trk_f1native_r2_20261027/transitions_v5.json`

SHA256 **`e95881049167a98c402680512c8997aa4548a10fedbec47b9dcaaa9b68bdc0a5`**.
Completed, before/after hashes unchanged, retained CA criterion/exact agreement true,
accepted-membership numeric cuts pass. Full history/state-policy/full-fit closure false.

Preserved versions:
- v1 fails `ValueError: unsupported updater`: empty Chi2SHUSettings was mistaken
  for an enabled updater. Corrected by allowing only absent/empty declarations and
  checking the actual schedule/source makeConfig.
- v2 fails `ValueError: miter outside configured stages`: the constructor's initial
  history record makes captured miter start at1. Origin is now explicit and checked,
  not inferred from whichever rows happen to be present.
- v3 succeeds numerically, but its seed absence counter included noncaptured targets.
- v4 scopes the seed catalog correctly and proves disjoint snapshot coverage.
- v5 adds source-conditioned association inference and final type/counter guards.

Review/testing also fixed boolean temperatures accepted via Python numeric equality,
malformed numeric-prefix FCL tokens, origin-offset extension counter, strict literal
receipt pass flags, exact seed membership, misleading terminal-hit terminology, and
unnecessary derivative evaluation merely to obtain |DOCA|. No tolerances or cut
thresholds changed. New tests perturb following captured geometry without altering
generated new-hit results, and cover missing/masked identities and strict failure JSON.

New tests **64 census +131 audit +74 boundaries +262 coverage +27 associations =558 passed**.
Selected isolated suite **2,454 passed, 1 skipped, 48/48 files**, pinned core233771b/ANN,
CPU/single-thread; log
kept locally, not in the repository.
No full/slow-tier certificate.

```bash
bash scripts/run_f1_transition.sh \
  $M2FS_DATA/trk_f1native_r2_20261027/NEW_REPORT.json
```

Data usage 147202348616/161061273600 bytes (~12.91 GiB free), measured after other concurrent jobs; no data deleted.
Fitter SHA remains `988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a`. No
staging/commit/push/build/grid/handoff.

## Subsequent crossing checkpoint

[Crossing observability](f1-crossing-observability.md) now accounts for all 102,949 crossing
snapshots and verifies exact same-iteration old-owner reference membership, source-conditioned
extension order and the 81 reconstructed associated constructor inputs. Native tube geometry,
constructor/private offsets and crossing retained CA remain absent. That checkpoint contains
the minimal passive capture proposal; it does not claim crossing geometry/discovery closure.

## Next boundary

Continue crossing construction and retained-state observability, starting with the
81 associated extensions while keeping the larger unassociated material-discovery
population explicit. Decide whether source reconstruction can be compared with
existing observations or a narrowly scoped passive constructor/private-offset
capture is required; any native source/build work needs a separate user-build step.
Later unbiased CAD/ANN inputs, saved-F1 chart representation, strict ROOT/filter
arithmetic and final acceptance/fidelity remain open. All-hit geometry agreement
alone does not meet the 99.5% acceptance or 99% per-parameter fidelity targets.
