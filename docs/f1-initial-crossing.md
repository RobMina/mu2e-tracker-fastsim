# Initial hit and tube-crossing constructors

## Result and scope

The standalone `scripts/f1_initial_crossing.py` and `f1_initial_crossing_audit.py` reconstruct **all 2,538 initial crossings across 147 targets**, with **exact agreement and zero exclusions** on the preserved full 3,000-event capture. Both initial hit and tube searches have zero piece moves and zero exhausted searches. All 2,538 reconstructed hit-constructor particle times also exactly match the immutable initial-hit snapshot.

The prior [associated-extension constructor certificate](f1-crossing-constructor.md) remains unchanged: 81 constructors / 50 targets. Together these two explicitly conditioned comparisons cover **2,619 initial-or-associated constructors**. The remaining 2,471 material-only constructors, later reference/wrapper/partner lifecycle, and seven exhausted later searches are not certified here. Full fit/F1/history/numerical and independent-discovery flags remain false.

## Independent generation contract

1. Select stage-0/category-0 constructor identities. Join each contemporaneous partner to immutable seed hit membership by `(run, subrun, event, iseed, shindex)`, checking bijective coverage and straw identity. No final-collection partner stamps are used.
2. Join the category-0 construction trajectory by full target/trajectory identity. All 147 contexts contain one piece; native parameters and mass equal the later seed-contract values. Native trajectory/mass and accepted membership are explicit inputs, not autonomous seed or hit-selection closure.
3. Read measured ComboHit endpoint times, corrected time and energy from `trkdump/events` in the full capture. Full event/source-index joins preserve different seeds in one event. Fields retain float32 storage; equal endpoint times select cal, and the saved early-end flag must agree.
4. Use target-specific raw **wire** cal/hv endpoints for the hit SensorLine. Reconstruct signal speed from the pinned expanded-FCL coefficients and measured energy, using native PieceLine slope-first arithmetic. No captured CA direction, time, speed or residual is used.
5. Starting from corrected hit time, reconstruct the native z-adjusted particle hint with `nativeca.constructor_hint`; solve the hit CA through unchanged `f1_later_ca.piecewise` at `float(np.float32(1e-4))`, ten moves and the unchanged 100-update local solver.
6. Use separate raw **tube** cal/hv endpoints for the crossing SensorLine, cal=end and hv=start, speed=c_light. Its anchor and both incoming hints are the **generated hit-CA solution times**, not the raw z hint or a later reference state.
7. Generate the tube CA, final pointer and retained solution. Compute the private offset from geometry wire radius divided by final-reference-piece speed at the particle TOCA; never subtract effect time or invert material effects.

Generation is separate from comparison. Captured crossing line, hints, CA, reference selection, radius/speed operands, private offset and effect time are comparison-only. Captured initial hit time is likewise comparison-only. Geometry is supplied raw geometry, not independently regenerated conditions.

Exact comparisons cover line start/end/direction/length/anchor; incoming hints; both CA times, positions and directions; status/usability; initial/final-reference/last-solved indices; solved-piece history/count; reference parameters/range; radius; reference speed; private offset; and effect time. Cached derivatives/variances and every intermediate local-solve scalar trace are outside this comparison.

## Construction range is not the later seed-contract range

The first data attempt correctly failed its then-declared equality assertion before numerical comparison, but that assertion described the wrong lifecycle. Fresh native source inspection establishes:

`makeSeedTraj -> pseedtraj -> makeStrawHits -> seedtraj.range() = KKFit::range(sh,ch,sx) -> seedContract -> KKTrack`

The crossing trajectory snapshot occurs at `makeStrawHits` entry; the seed-contract snapshot follows range replacement. **All 2,538 hit rows have different ranges at those two stages.** The reconstruction must use the construction trajectory's range, not overwrite it with the later seed range. Parameter and mass equality remain required. The final report retains both ranges per hit and the full mismatch census. This correction changes neither numerical arithmetic nor tolerances; failed v1 and its source snapshot remain immutable.

## Provenance and execution

Data prefix: `$M2FS_DATA/trk_f1native_r2_20261027/`.

- Final **`initial_crossing_v2.json`** SHA256 `100542dfd9b0f6287c06b79c4a3a775883c1d602580418f703112fbc5a2acbf7`.
- `completed=true`, `passed=true`, `exact=true`, `hashes_unchanged=true`.
- Helper SHA256 `3523bbeecbd4674fcb45f7e82aa75021cbbeba609062c200b647b61d75210c2b`.
- Audit SHA256 `b7ffbe665b73aa5155e08fb9ac09152ec9df9cd20ab20d6218efba2aa7a0df26`.
- Receipt seal **`initial_crossing_receipts_v2.sha256`** SHA256 `10cc9a90dbb7e522dffd395e66c2bd090e78f7817d2086f188dcfe2f67fe0ca1`.
- Failed `initial_crossing_v1.json` SHA256 `84e8c77e4bc0a747eb48e2ebb213669ffd9d293c71907aa1dbf64b55c7d4eacb`.
- Full capture gate SHA256 `c8e1225b1cc71ad82ff7055c77cd382c1b564956a1470522136d48575d53f8ec`.
- Inherited associated v4 receipt SHA256 `8f13afc822ef99590b36c47764251f1fdefb994117cb0bdeb6b48a98818bfe2e`.

Full-capture and associated-receipt input manifests are authenticated before and after execution. Canonical paths, actual required function origins/imported aliases, and numerical decorator execution closures are checked. Reports are exclusive strict JSON with failure fallback. All bound prior modules, native sources and fitter remain unchanged.

Measured successful run on one CPU of GPVM02 in `pyenv ana`: **21.56 s wall, 1,360,356 KiB maximum RSS, exit 0**.
These are measurements, not performance extrapolations.

## Tests and preserved failures

**93 new tests pass locally and remotely**: 73 helper + 20 audit. Selected ten-module isolated regression coverage totals **1,035 passed**: the prior 942 plus these 93. This is not a new full/slow-suite certificate.

- Local `local_data/trk_f1native_r2_20261027/initial_crossing_isolated_v1.txt` contains 567 passes in the first nine modules, followed by an import-path collection failure in the geometry module. That module requires `scripts` explicitly on PYTHONPATH.
- Separate `initial_crossing_geometry_isolated_v2.txt` records all 468 geometry tests passing with `PYTHONPATH=.:scripts`; no test/source change was needed.
- Remote `initial_crossing_tests_v4.txt` contains all 93 new passes.
- Initial remote test collection failed because importing the old general CA audit transitively imported unavailable JAX. No dependency was installed. The new helper now contains the small standalone source-speed parser/arithmetic without importing the fitter. Old helpers remain untouched; the failed log and its sources are retained.
- `initial_crossing_tests_v1_sources/`, `initial_crossing_v1_sources/`, and `initial_crossing_v2_sources/` retain the respective source states; the seal binds them and all versioned reports/logs.

Read-only review led to pre-coercion seed-index typing, distinguishable per-target synthetic fixtures, and actual wrapper-closure binding checks. Counterfactual tests perturb captured outcomes and seed hit times without affecting generation. Other tests cover all full-key permutations/missing/duplicate identities, measured float32/nonfinite guards, endpoint ties/reversal, source-speed modes/constants, separated range stages, immutable input binding and exclusive failure output.

## Reproduction and next step

Use **new exclusive output and log names**:

```bash
python3 scripts/f1_initial_crossing_audit.py \
  --full-check $M2FS_DATA/trk_f1native_r2_20261027/crossing_3k_20261028_v2/crossing_full_check_v1.json \
  --associated $M2FS_DATA/trk_f1native_r2_20261027/crossing_constructor_v4.json \
  --capture $M2FS_DATA/trk_f1native_r2_20261027/crossing_3k_20261028_v2/capture.root \
  --expanded $M2FS_DATA/trk_f1native_r2_20261027/crossing_3k_20261028_v2/capture.expanded.fcl \
  --native-source $TRKFS_MUSE_AREAS/fastreco_crossings_p109_20261028 \
  --out $M2FS_DATA/trk_f1native_r2_20261027/NEW_initial_crossing.json
```

Next is reference/wrapper/retained-CA/partner-history reconstruction, including all seven capped stage-1 searches and final-pointer-versus-last-solved semantics. Material-only construction/discovery still requires the full geometry universe and rejected/skipped branches. Native seed, strict algebra, state policy, decision, acceptance and full-population closure remain open. Do not integrate the diagnostic into the fitter or promote defaults from conditional agreement.
