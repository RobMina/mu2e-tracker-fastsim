# Run 1A fast reconstruction development

The user expanded `TRK2-run1a-digitization` to include field-on fast tracker reconstruction.
The validated digitiser dependency remains unchanged. This work is opt-in; the extracted-tracker
line chain and TRK1's conversion-electron drivers are not modified.

**Delivery priority:** the independent [resident wire-reconstruction API](resident-reco.md)
provides bounded GPU grouping, direct measured-position seeding and wire-only refinement.
It does not consume this page's saved CPU associations or drift fits; their success counts
must not be attributed to it. Further CPU diagnostics are subordinate to landing that limited API.

## Development boundaries

1. **Hits on the same saved digis:** compare NumPy/JAX single-straw hits and supported panel-hit
   columns against stock `makeSH`/`makePH`, with the actual Run 1A reconstruction conditions.
2. **Curved seeds from measured hits:** introduce a batched constant-Bz geometric helix seed,
   with curvature-derived momentum magnitude rather than F2's fixed 2 GeV/c. This is a seed,
   not a drift-time fit or a replacement for CentralHelix/KinKal.
3. **Field-on fitting and validation:** add angle-dependent drift, wire residuals, hit association,
   timing/direction, covariance and material treatment; compare to the saved stock CentralHelix
   tracks before claiming momentum resolution, tracking efficiency or end-to-end speedup.

The parked F1 code is a field-free KinKal port below its own closure targets, not a ready-made
field-on fit. The existing dumper's `driftInfo(dt,phi=0)` table is insufficient for field-on fitting.
Neither truth grouping nor true momentum may be used as reconstruction inputs.

## Hit reference without new C++

`run1a_reco_reference.fcl` includes stock `recoMC/OffSpill.fcl`, then runs only PBTFSD, makeSH,
makePH and the existing `StrawHitTableDumper` (`DumpTracks:false`). It reads saved digis: no
Geant4, redigitisation, track fitting or output art. The dumper consumes the same PBT and EWM
products as makeSH, rather than assuming its extracted-data default.

```bash
bash scripts/run1a_reco_export.sh BUILT_AREA SAVED_DIG_ART NEW_OUTPUT 1000
python3 -m trkfastsim.validation.run1a_reco --reference NEW_OUTPUT/hits.root \
  --outdir NEW_COMPARISON --jax
```

The runtime smoke (10 events) and development export (1,000 events) completed on the already-built
benchmark area. Geometry `geom_run1_a`, field `bfgeom_no_tsu_ps_v01`, DB `Sim_best/v1_1`, run 1430:0
match the benchmark. The resolved configuration confirms FitType 1, FilterHits false, no calorimeter
selection or cross-talk, zero-to-100000 ns hit window, the existing panel grouping settings, and the
current longitudinal/TOT response parametrisation. The first-event conditions export is valid for
this one IOV; it is not a multi-IOV conditions service.

The comparison retains the full event census and reports hit identity, every supported hit quantity,
flags and panel constituents before comparing aligned numerical columns. It reports differences,
not an automatic physics acceptance gate. NumPy/JAX timings separate first call from repeat and
are not hardware-matched stock speedups. A JAX CPU run is not a GPU measurement.

## Measured hit results

The first 1,000 events and the next disjoint 10,000 events use the same saved cosmic parent as the
digitisation study. Full `(run, subrun, event)` checks find no overlap; this is not an independent-parent
test. No reco-kernel settings were tuned between blocks. All rows, including empty events, are retained.

| Observable | Development 1,000 | Disjoint 10,000 |
|---|---:|---:|
| Straw hits, stock = NumPy = JAX CPU | 13,492 | 137,649 |
| Panel hits, stock = NumPy = JAX CPU | 7,192 | 73,099 |
| Zero-digi events | 235 | 2,207 |
| Straw identities / flags / panel constituents | exact | exact |
| Largest straw-hit time difference from stock (ns) | 0.0078125 | 0.0078125 |
| Largest panel-hit time difference from stock (ns) | 0.015625 | 0.015625 |
| Largest straw-position component difference (mm) | 0.00003052 | 0.00006104 |
| Largest panel-position component difference (mm) | 0.0001221 | 0.0002441 |
| Largest SH / PH longitudinal variance difference (mm2) | 0.001953 / 0.04443 | 0.01123 / 0.05615 |

All compared NumPy and JAX CPU columns are identical, as are repeats. Both stock and fast hit energies
are nonnegative on these blocks, and all exported numerical hit fields are finite. This checks the existing
ports under this configuration; it does not establish arbitrary negative-energy/TOT-index behaviour.
The constant SH transverse variance differs by 2.384e-7 mm2: stock multiplies by float `1/3`, while the
port divides by double 3. The SH time expression also differs in intermediate float rounding. Neither
kernel was altered to tune away these differences. Calibrated end times, TOTs, early end, TOT drift time
and time variance are exact; hit energy differences are at most 4.657e-10 MeV.

On the laptop, the 10,000-event SH+PH first/repeat times were NumPy 0.0430/0.0387 s and JAX CPU
0.8366/0.1126 s. ROOT reading took 0.1368 s, condition extraction/loading and packing are separate,
and each compressed numerical output took about 0.5 s to write. These are sparse hit-stage timings,
not GPU measurements or same-host stock speedups.

## Geometric seed API

`reco.helixseed.fit_helix_seed(positions, valid, bz=..., weights=...)` accepts padded measured
positions and masks. Centered/scaled algebraic circle fitting and z-versus-phase regression return
radius, pitch and momentum magnitudes with explicit failure/conditioning diagnostics, preserving
empty lanes. The 65 analytic tests cover signs, phase wraps, translations, masking, degeneracy and
short arcs. The measured-only association below now connects this primitive to stereo alternatives;
repeated-z quartets still fail rather than receiving invented jitter or truth ordering. It has no JAX
implementation yet.

## Remaining scope

Panel-hit representation does not yet carry the complete stock covariance; its combination assumes
zero wire-direction z. The geometric seed uses scalar position weights rather than straw anisotropic
errors and cannot resolve drift ambiguity. Constant Bz, monotonic-z phase ordering and sufficient phase
sampling are seed approximations. Charge and flight direction are not inferred from positions alone.
Multiple-track separation, outlier handling and nearly straight high-momentum tracks need further work.
No new field-on track efficiency, momentum-resolution or GPU-throughput result is established yet.

## Measured stereo control checkpoint

Optimisation is deferred until reconstruction works, per the user's request. New `reco.stereoseed`
and `validation.run1a_seed` connect saved measured hits to bounded stereo-candidate diagnostics.
The geometry uses normalized 3D wire directions, including alignment tilts: projected intersections
are solved in actual along-wire mm, and z is evaluated on both wires. Flattening aligned wires is
not allowed. Local track slope and drift radius are still ignored at this seed stage.

On the first 1,000 development events only, 12,748 of 13,492 hits pass the candidate flags;
773 time groups yield 7,392 candidate intersections in 624 events. All events, including 235 empty
ones, remain in the output. The conservative unambiguous-only driver attempts no helix seeds:
all 554 groups with at least four candidates contain shared-hit alternatives. This is retained as a
strict control, not a geometry failure or measured tracking-efficiency result. The association below
handles these alternatives; it does not discard shared-hit groups as the final solution. No held-out
events were used for these development decisions.

The stereo/driver/helix suites pass 188 tests. The saved development report is produced by:

```bash
python3 -m trkfastsim.validation.run1a_seed --hits COMPARISON/numpy.npz \
  --conditions COMPARISON/conditions.json --bz 1.0117 --outdir NEW_SEED_OUTPUT
```

## Measured-only association and 3D wire scoring

`reco.seedassociation.associate_wires` enumerates all four-candidate combinations within each time
group. A defining quartet must use eight distinct physical straws; repeated z is not jittered.
`reco.helixwire.wire_residuals` scores its fitted helix against the ORIGINAL normalized 3D wires,
not a cloud of dependent stereo intersections. Local Newton closest approaches start at measured
wire-point z, use bounded steps/backtracking and a bounded phase basin, and require a strict positive
Hessian plus an unclipped convergence test. This is not a global-POCA guarantee or finite-endpoint fit.

Fixed engineering support limits are DOCA <=10 mm and longitudinal pull <=3, with convergence and
finite-wire membership required. Each straw contributes at most once, selecting its lowest-cost
compatible pulse. Rank by support count, then mean squared normalized DOCA/pull; at least eight
supported straws are required. This geometric score is NOT chi-square. Defining wires participate
in scoring, so support is not independent validation. Retain the best and runner-up hypotheses,
with original candidate/hit indices; the runner can represent the same trajectory, not a second track.
No helix refit, timing/direction, angle-dependent drift, covariance or material treatment is supplied.

Memory is bounded by batch/score-cell limits and incremental top-two-per-group reduction. All
combinations are enumerated, never randomly sampled or truncated; >256 candidates/group or >2M
raw combinations raises. The present exhaustive implementation is development scaffolding, not
production-speed pattern recognition. Fixed score-row width gives bit-exact reduction order across
hypothesis batch sizes; an earlier variable-width version changed only floating-point scores when
changing batches. This was an implementation correction, not a physics-cut adjustment.

On the same **first 1,000 development events only**:

| Diagnostic | Result |
|---|---:|
| Full event census / zero-hit events | 1,000 / 235 |
| Original time groups / stereo candidates | 773 / 7,392 |
| Raw quartets / eight-distinct-straw quartets | 1,856,308 / 619,480 |
| Successful geometric hypothesis fits | 619,463 in 367 groups |
| Hypotheses with >=8 supported straws | 498,519 |
| Winning supported seeds / events with a seed | 344 / 344 |
| Selected distinct source straws (summed across groups) | 5,427 |
| >=4-candidate groups without an independent quartet | 187 |
| Fitted groups without sufficient wire support | 23 |
| Groups with runner-up / same-support runner-up | 334 / 258 |

All source-hit/event identities and rejected groups survive. The support census is independently
recounted from distinct `(group, straw)` identities. Supported-wire DOCA median/p90 is **3.844/8.488 mm**,
so this is visibly still a coarse seed, not a drift-quality fit. Circle-condition median/p90 is
**1.56e3/5.07e4**, a numerical-conditioning diagnostic, not momentum uncertainty. No momentum-resolution,
tracking-efficiency, purity, stock-CH agreement or GPU claim follows from these counts. One seed per
time group also cannot resolve multiple tracks; an explicit two-track synthetic test records this limit.

```bash
python3 -m trkfastsim.validation.run1a_association --hits COMPARISON/numpy.npz \
  --conditions COMPARISON/conditions.json --bz 1.0117 --outdir NEW_ASSOCIATION
# Same inputs, alternate batch; every saved array must agree, including scores.
python3 -m trkfastsim.validation.run1a_association --hits COMPARISON/numpy.npz \
  --conditions COMPARISON/conditions.json --bz 1.0117 --hypothesis-batch 257 \
  --compare-saved NEW_ASSOCIATION/diagnostics.npz --outdir NEW_REPEAT
```

Final development outputs: local `TRK2/run1a-reco/association-1000-fixedwidth-20261106` and
`association-1000-fixedwidth-batch257-20261106`. Every NPZ array agrees exactly, including NaNs.
Measured laptop processing is **141.30 s** (batch1024) and **188.00 s** (batch257); load/write are
separate. These are CPU exhaustive-development timings, not a stock speedup or a GPU measurement.
The second process outlived a local tool timeout; its actual state and completed report were checked,
not relaunched. Earlier variable-width development outputs are retained, not silently overwritten.

Validation: refreshed remote affected selector finds 27 files; their isolated local run gave **648
passes**, followed by four additional association regressions and 23 existing reco/site/layout tests:
**675 unique passing tests**, no skips. The final focused run is **356 passes**. The 105 wire tests
include analytic intersections/normal offsets, tilted/reversed/scaled directions and an independent
SciPy scalar minimization oracle. Association tests cover shared alternatives, duplicate pulses,
multi-event/multi-group provenance, empty/rejected events, bounds, permutations and exact batch
invariance. Read-only review identified and prompted fixes to winner-storage growth, derived-geometry
finite checks and report scalar serialization. No CI run, C++ edit, build, commit or publication.

Next: refit against measured wires and angle-dependent drift/T2D, resolve timing/direction and
ambiguity, then compare to saved stock CentralHelix. Digitiser optimisation remains deferred.

## Fixed-membership measured-wire refit

`reco.helixfit.fit_wire_helix` uses local `(x0,y0,tx,ty,omega=dphase/dz)` parameters at a fixed z
reference, stable through the straight limit. Batched damped least squares minimizes perpendicular
3D wire displacement scaled by **2.5 mm** plus measured longitudinal pulls. This engineering
objective is **not chi-square**, and wire-only fitting is not a drift fit. Three perpendicular
components are redundant after projection/POCA; no covariance or statistical errors are inferred.
Numerically unidentifiable curvature yields no momentum even if the local fit converges.

Membership is frozen to the saved association's distinct-straw support: no outlier dropping,
hit additions or candidate-search rerun. Every accepted trial must have converged local POCA and
finite-wire membership for every selected hit. Convergence requires the unconstrained gradient,
full rank and bounded conditioning; clipped steps, exhausted budgets and damping stalls remain
failures. Failed lanes retain last finite local diagnostics, not successful physical estimates.

`validation.run1a_wirefit` checks full event/straw identities, group membership, distinct-straw
support and replays selected seed residuals before fitting. This is **not complete original-input
provenance**: the earlier association did not save originating input hashes. The new runner hashes
this invocation's files, and explicitly reports that limitation. Alternate local-parameter seeds
cannot override the validated helix seed. Support/index dtypes are normalized after validation.

On first1000 only, retaining 773 groups, 235 empty events and all 5427 selected hits:

| Same fixed-support fit | Default40 iterations | Budget diagnostic120 |
|---|---:|---:|
| Input seed groups | 344 | 344 |
| Converged groups/events | 255 | 314 |
| Iteration-exhausted / stalled | 89 / 0 | 20 / 10 |
| Hits in converged groups | 4395 | 5086 |
| Median initial DOCA, converged groups (mm) | 3.631 | 3.814 |
| Median final DOCA, **same** groups (mm) | 0.966 | 0.992 |
| p90 final DOCA (mm) | 2.044 | 2.068 |
| Initial measured laptop processing (s) | 6.358 | 10.446 |

All344 objectives decreased, but improvement alone does not mean convergence. Default40 is
unchanged;120 is an explicit numerical-budget diagnostic, not a relaxed physics selection.
Some final DOCA and longitudinal pulls exceed the association cuts: membership was deliberately
not pruned again. Quoted residual improvement is on identical hits within each column, not a
comparison of differently selected seed/fit populations. No heldout or stock-track inputs were used.

```bash
python3 -m trkfastsim.validation.run1a_wirefit --hits COMPARISON/numpy.npz \
  --conditions COMPARISON/conditions.json --association ASSOCIATION/diagnostics.npz \
  --outdir NEW_WIREFIT
```

Local cache `TRK2/run1a-reco/wirefit-1000-20261106-final` is the final default report; its payload
agrees exactly with `wirefit-1000-20261106-1`. The repeat `wirefit-1000-batch31-20261106-1` is also
array-exact versus group batch128. `wirefit-1000-budget120-20261106` retains the diagnostic separately.
Earlier default reports used a broader residual-replay field name; final reports narrow it and
explicitly disclaim original-input hash binding. No original outputs were overwritten.

Refreshed affected selection: **29 files/743 passes**, plus **23 existing reco/site/layout tests**:
**766 unique passes**, no skips, no CI. New kernel/driver suites contribute66/25 tests, including
independent noisy SciPy minimization, tilted wires, straight limit, failures and exact batch invariance.
Review prompted seed-representation and unsigned-index fixes; no fitted cut was tuned to this sample.

[The drift/timing contract](run1a-drift-contract.md) records
why the phi-zero export and TOT-corrected hit time cannot be substituted. The user-built stock oracle
now passes 1- and 10-event saved-digi runs; all393276 queries are finite and repeat array-exactly.

`reco.fielddrift` now constructs the native angle-dependent T2D/calibrated response from resolved
FHiCL, nominal radii and full-precision origin field. It does not interpolate the final oracle values.
All393276 stock queries agree within fixed numerical thresholds: max raw/calibrated-radius errors
**1.3785e-12 / 6.8390e-14 mm**, velocity **1.2559e-15 mm/ns**, both errors and public CC exact.
The source-level evaluator and its scalar-reference tests are separate from the wire-only fit;
shared digitisation is unchanged. No new simulation/digitisation or association was performed.

## Timing and joint drift fit

[Timing details and reproduction](run1a-timing.md): all255 converged wire-fit groups pass fixed-geometry
t0 profiling in both flight directions, using explicit cal/HV endpoints and original calibrated
early-end times. Signals point **late to early**; propagation is subtracted from the early-end time.
The muon-mass/unit-charge beta comes from measured curvature in the same constant axial field.
No TOT-derived radius/time error, truth input or stock-track membership enters these residuals.

The opt-in six-parameter geometry+t0 fit preserves support and initial LR. Of510 direction
hypotheses, **195 converge** (105 negative-z,90 positive-z); **282 stall and33 exhaust** the40-iteration
budget. All510 objectives improve, which alone is not success.159 groups/events have at least one
converged direction;36 have both. No direction is assigned. All1000 events/773 groups and5427
original selected-hit identities remain represented, including89 prior wire-fit failures and429
empty lanes per direction. These counts are numerical diagnostics, not tracking efficiency.

On the same hits within converged hypotheses, median absolute drift residual changes
**0.368->0.184 mm** (negative-z) and **0.386->0.182 mm** (positive-z); no post-fit residual cut.
All saved arrays repeat exactly at group batches32/17. Final cache `driftfit-1000-20261106-final`;
timing-only cache `timing-1000-20261106-1` plus exact batch7 repeat. The focused13-file run gives
**673 passes**, including80 timing/62 joint-fit/43 driver tests; no CI. Shared digitisation unchanged.

[Final-state diagnostics](run1a-drift-diagnostic.md) now replay all510 objectives/default gradients
and retain every original array.231/282 stalled hypotheses cross offset-regression coefficients
in the default stencil,279/282 cross time cells;271 have sampled smaller-stencil descent versus25
with the default direction above an arithmetic screen. These tags overlap, not causal assignments.
Only13/195 recorded successes pass the same gradient threshold at the smallest stencil: **the
original convergence labels are not derivative-scale-stable certificates**. No status is changed.
Pointwise measured LR candidates are recorded without updates or direction decisions. Diagnostics
repeat array-exactly at batches32/17;872 focused tests pass, no CI or new C++ build.

The separate [analytic candidate](run1a-stabledrift.md) now gives307/510 passing analytic+adjacent-FD
checks from the same original seeds/LR/support; the saved baseline gives20 under that identical
criterion, not195. Candidate213groups have any passing direction,94both;134 consistency failures,
48 boundary states,17 budget failures and4 stalls remain. Baseline labels/arrays are preserved.
Most objective changes are tiny;411 candidate finals are below baseline and99 above. No efficiency,
resolution, LR/direction assignment or stock-relative speedup claim. Complete arrays repeat exactly
at batches32/17;1010 focused tests pass, no CI/build/newphysics. Baseline optimizer stays unchanged.

The [read-only failure study](run1a-drift-failure.md) resolves the48 guarded centers into45 offset
switches and3 time knots. Three diagnostic POCA-polishing rounds gain27 checks but lose16;32
no-adjacent-pair cases persist. Saved307/203 labels remain unchanged, not a new fit result.
All arrays repeat at batches32/17;1078 focused tests in22 selected files, no CI or new physics.
The separate [measured LR continuation](run1a-lrdrift.md) adds up to3 LR/refit rounds from those saved
finals:90→336 combined continuous+LR checks,222groupsany114both;384lower objectives126unchanged.
Same8790 valid hit-hypothesis entries have median absolute drift residual0.193→0.146mm, not resolution.
174failures remain (including38 sticky numerical histories); both directions stay unassigned.
Arrays are exact at batches32/17;1176 focused tests in24files pass. No original-seed/equal-compute A/B.
The new [read-only stock census/overlap](run1a-stockcompare.md) preserves every recorded stock
track/hit and all same-event group/track pairs without ranking or feeding membership into reco.
Saved stock output is CHPath-selected: absent events stay unknown, never zero-track failures.
The25 MiB saved CH ntuple was fetched to nonsynced laptop cache and analyzed locally:1888recorded
CHevents/tracks,62204hits;38events overlap development1000. Of these28have a combined-passing group,
3lack seed support and7have valid objectives but neither direction passes. All35nonempty supports
are subsets of their stock straw sets; passing-group median stock coverage0.931, not pulse purity.
962absent events remain unknown, including194with fast passes, not demonstrated gains or fakes.
Arrays repeat exactly with4096/17-entry read chunks;1407tests in26files pass (236new), noCIclaim.
The [saved-stage/pair audit](run1a-stockattrition.md) now finds2zero-stereo-candidate CH groups and
1with no independent quartet; none of the7supported nonpassing groups has a continuous check.
All7392stereo source pairs replay exactly across121499considered pairs;1730focusedtests pass.
Next: slope-aware/direct-wire seeding, independent root/arithmetic and branch-aware treatment,
multiple tracks/outliers, direction and material/covariance. No complete field-on validation yet.
Digitiser optimisation remains deferred.
