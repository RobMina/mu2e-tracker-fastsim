# F1 stock-conditioned closure campaign

The F1 native-closure tooling this record uses (capture modules, `f1_*`/`run_f1_*` scripts, the per-step f1-* records) is on branch `park/f1-native-closure`; the F1 fit itself (`reco/kalman_jax.py`, `reco/vr_f1.py`) is on main.

## Executed result — 2026-10-27

**Not closed; do not promote a new default.** All 20 CPU arms finished with exit 0. The report independently
verified the highest-cycle input has 3,000 events (run 1400), all 2,315 eligible single-seed/single-fit events are
represented in every own-state arm, and the stock-good denominator is 2,151. Source manifests match across arms.
The legacy acceptance mask exactly reproduces the saved baseline on all 2,315 identities. Existing baseline
products were preserved; the new baseline replay adds source manifests, crossing-state diagnostics and identities.

| Own-state arm | Good | Gains / losses against stock | Net relative yield | Acceptance agreement |
|---|---:|---:|---:|---:|
| Legacy default | 2,243 | 113 / 21 | +4.277 ± 0.538% | 94.212% |
| Diagnostic time model | 2,167 | 59 / 43 | +0.744 ± 0.470% | 95.594% |
| Time model, revert constraint placement | 2,141 | 67 / 77 | -0.465 ± 0.558% | 93.780% |
| Time model, stock final crossing flags | 2,167 | 60 / 44 | +0.744 ± 0.474% | 95.508% |
| Time model, p109 gas fraction | 2,167 | 59 / 43 | +0.744 ± 0.470% | 95.594% |

The near-zero net yield is **not** fit closure: gain/loss exchanges remain far above the 99.5% acceptance-agreement
target. Against the saved prior-only arm (66 gains / 77 losses), the time model recovers 47 of the 77 losses but
introduces 13 new losses; it retains 52 prior gains and creates seven others. Reverting its constraint placement
reproduces that prior arm on 2,314/2,315 decisions (one extra gain).

Leave-one-out versus the full time model: constraint reversion changes 80 decisions (27 gains / 53 losses), prior
one loss, bounds three gains, status reference one gain / one loss, seed-crossing rule none, transit one loss.
The p109 gas-fraction change gives identical acceptance masks. DOCA-only activity gives 85 / 33 against stock,
but contradicts the native rule and is not recommended. These are sensitivities of an approximation, not native
causal attribution.

| Fixed stock hit states, both-good tracks | d0 | phi0 | z0 | theta | t0 | Momentum |
|---|---:|---:|---:|---:|---:|---:|
| Legacy (2,137) | 97.801% | 97.052% | 96.631% | 96.631% | 99.906% | 6.130% |
| Diagnostic time model (2,139) | 97.709% | 96.961% | 96.540% | 96.540% | 99.906% | 42.590% |
| + stock crossing flags (2,139) | 98.738% | 98.364% | 98.457% | 98.457% | 99.906% | 46.517% |
| + stock scattering paths (2,140) | 98.738% | 98.364% | 98.411% | 98.411% | 99.907% | 46.589% |

Entries are fractions within 0.1 stock sigma; target 99% per parameter. Neither final crossing flags nor final
scattering paths close geometry. For the time-model arm, 69/97 joint-geometry outliers have a crossing-activity
mismatch inside the final-line proxy; forcing stock flags leaves 45 outliers. This demonstrates association and
sensitivity, not which actual iteration crossings caused the disagreement.

Momentum difference q16/50/84 [MeV/c] changes from `[-0.02620, 0.00003635, 0.07446]` (fixed legacy) to
`[-0.0001088, 0.000003367, 0.0001260]` (fixed time model); pull quantiles change from
`[-10.35, 0.305, 16.81]` to `[-0.587, 0.00247, 0.676]`. Large improvement, still far from the 0.1-sigma target.
Only 92.89–96.31% of geometry errors are within 2% in the time-model arm; the fraction rises to 97.62–98.18%
with stock crossing flags. A near-unit median error ratio alone would conceal these tails.

Own-time-model losses are mostly extension ParamsDiverged (31), plus one seed ParamsDiverged. The saved front
marginal momentum term exceeds the stage threshold for 17/32 such losses, but is not a joint-statistic
decomposition. All accepted tracks in all 14 own-state arms pass the existing final-effect/t0 numerical-validity
checks. This does not certify rejected candidates, intermediate states or physical correctness.

### Evidence and handoff

- Worktree: `analysis/trkfs_wt/f1`, branch `trk-f1-closure`; all changes unstaged/uncommitted.
- Data root: `$M2FS_DATA/trk_f1_closure_20261026/p109_arms_v4/`.
- `report_v4.json`: full metrics, paired exchanges, numerical validity, configuration and input-count gates.
- `report_v4.md`, `report_v4_detailed.md`: readable tables; `report_v4_*_exchange.csv`: individual gain/loss keys.
- Each arm has `source.sha256`, `command.txt`, `vr2f1.json`, `ids.npz`, and the existing fit diagnostics.
- Hosts: gpvm05 yield, gpvm03 fidelity/gas/report; both batches and the queued report ended cleanly, locks released.
- Tests: 51 fitter/validity/reco regressions plus five report tests passed; shell syntax and `git diff --check` pass.
- No C++ build/run, commit, push, or production default change. The exact native capture remains blocked on the
  user build. At the time of this record the capture area had no `build/` or `kinkal_build_p109/`; app free space was about
  5.2 GiB, below its 6 GiB pre-build reserve. Recheck before building, do not delete outputs without permission.

**Build-area relocation (2026-10-27):** the user builds muse areas directly under the app root. The prepared
capture worktrees now live at `$TRKFS_MUSE_AREAS/fastreco_capture_p109_20261026`, cluster-only. All five
pins, four manifest hashes and Git back-links were verified after `git worktree move`; the capture sources and
unstaged KinKal patch are unchanged. The setup script and runtime guard use this new path. App free space at
relocation was about 17.7 GiB, clearing the earlier space blocker. No build or environment setup was run.

Next: build and passively validate native capture, then isolate exact constraint placement and reference-piece
search using captured operands. Do not tune divergence thresholds to manufacture agreement. The autonomous F1
outline below remains an implementation plan, not a delivered stock-independent fitter.

## Native capture and isolated constraint-time arm — 2026-10-27

The user completed the relocated muse build and matching local KinKal install. Source hashes and local library
resolution passed; 100-event smoke and full 3,000-event capture-off/on runs completed on gpvm01. The full gate
passes on 2,319 native seeds, all 147 target identities, 3,449 iterations and 168,020 effects; no unsupported
payloads. Recorded physics is equal to the frozen baseline and between off/on. Independent prior-weight checks
pass on 6,856 endpoints. This is a structural/passivity gate, not full fit-algebra or decision certification.

The opt-in `--native-time-check CHECK.json` now loads only the captured immutable ParameterHit time. It requires
the successful complete gate, rechecks all four evidence-file hashes and the fit-input hash, joins full
run/subrun/event/seed identities, requires one valid momentum-only constraint, and checks its exact native range
midpoint. Seed parameters/covariances, reconstructed owner range, candidates, priors, material, reference policy
and thresholds remain unchanged. It cannot be combined with legacy timing or the `cons` reversion. All six ANN
weight hashes match the preserved p109 manifest despite the source-area relocation.

**Result: no change on this sample.** For both the 2,315-track own-state and 2,151-track fixed-state populations,
`ids.npz`, `validity.npz`, `xings.npz` and the own-state `yield.npz` are byte-identical to their saved v4 time-model
baselines. The baseline was not refitted. `audit.json` records all file hashes and exact decoded-array checks.
Native times differ on 2,314/2,315 own-state seeds, but native-minus-reconstructed q16/50/84 is
`[-1.3980e-7, -6.2573e-10, 1.3442e-7] ns`, range `[-2.2457e-6, 2.5899e-6] ns`; no seed constraint-order changes.
The missing native midpoint precision is therefore not responsible for the remaining closure failures here.
This does not rule out differences from reference-piece search, construction-fixed offsets, native seed
parameters or object associations, nor does it compare an autonomous reconstruction.

Evidence:

- Native capture: `$M2FS_DATA/trk_fastreco/p109_iteration_3k_20261027/check.json` and paired ROOT files.
- F1 candidate: `$M2FS_DATA/trk_f1_closure_20261026/p109_native_time_v2_20261027/`.
- `report.json` compares own/fixed native-time arms to preserved v4 outputs; `audit.json` verifies exact products
  and time differences; per-arm `native_time.npz`, `ann.sha256`, `source.sha256` retain provenance.
- The first attempt in `p109_native_time_20261027` stopped before fitting because a NumPy masked-array wrapper
  could not be passed to JAX. The loader now returns a plain dense array after requiring exactly one time;
  a regression test covers this. The failed directory remains intact and is not a physics result.
- Eight native-time tests cover evidence hashes, complete-gate guards, identities and isolated scalar use.
  The toy check verifies equal supplied/reconstructed time preserves all fit outputs exactly and changing only
  time leaves the reconstructed initial owner untouched.

No C++ source or compiled binary changed after the user build. No additional build is needed for these Python
results. Defaults remain unchanged and F1 is still not closed: 59 gains / 43 losses, 95.594% acceptance agreement.

## Scope and unchanged default

These arms test the fitter on the frozen p109 3,000-event flat dump. They retain stock LineFinder seeds,
stock hit candidates and stock crossing candidates. They are **not autonomous reconstruction efficiencies**.
The legacy fitter, `hit_or_doca` crossing activity and static solid scattering fraction remain the defaults.
No thresholds are loosened and no stock-good or F1-good flags are redefined.

`--time-model stock` is a historical option name for a **stock-inspired diagnostic approximation**, not a
stock-exact implementation. Its JSON explicitly records `stock_exact=false` and reconstructed seed-time
provenance. Never label an improvement from this arm as native fit closure.

The two relevant populations are:

- `own_*`: every eligible single-seed/single-fit event, including stock-failed fits; evaluate paired gains/losses,
  yield and acceptance agreement against the stock FitOK flag.
- `fixed_*`: stock FitOK tracks, with stock final hit states imposed at each stage; evaluate parameter fidelity
  on tracks both accept. This is a fit-only diagnostic and is not an efficiency measurement.

A zero net yield difference does not imply closure. Report the gain/loss exchange and acceptance agreement
(target 99.5%) separately. Geometry parameters must individually be within 0.1 stock sigma for at least 99%.
The errors-within-2% fraction is also reported; a median error ratio near unity is not that criterion.

## Preserved diagnostic arms

`scripts/run_f1_closure.sh OUT ARM...` runs sequentially on one pinned CPU using the existing analysis environment.
It refuses existing arm directories/logs and saves a source manifest and command record per arm. New outputs must
be under the data area, not the source worktree. Run it on an otherwise idle host: check for other users and active
tmux jobs before launching it.

| Arm | Difference from the indicated base |
|---|---|
| `own_legacy`, `fixed_legacy` | unchanged default fit |
| `own_stock`, `fixed_stock` | time ordering, reconstructed seed-time constraint, owned old-piece priors and status references, seed-crossing membership, transit ranges |
| `own_stock_Lcons` | revert constraint to legacy local-s placement |
| `own_stock_Lprior` | revert to retained endpoint priors |
| `own_stock_Lbounds` | exclude constraint from active bounds |
| `own_stock_Lstatusref` | query status references at candidate bounds rather than end-piece midpoints |
| `own_stock_Lseedx` | keep extension-hit crossings in seed fit as legacy does |
| `own_stock_Ltransit` | omit transit extension of last piece range |
| `*_xdoca` | diagnostic DOCA-only crossing activity, **not** the p109 rule |
| `*_xstock` | impose final stock crossing activity at all stages, diagnostic only |
| `*_pstock` | additionally use final stock wall/gas paths for scattering, not energy loss; diagnostic only |
| `*_gasp109` | per-material gas scattering fraction instead of static first-material solid fraction |

The native `KKStrawXing::active()` rule is `active_ || associated_hit.active()`. Removing the hit term does not
restore native p109 behavior. Leave-one-out arms measure sensitivity of this approximation; correlated changes
and different fit histories preclude adding their effects as independent causal contributions.

## Known non-native semantics (source audit, 2026-10-27)

1. **Missing exact seed operands.** Flat inputs omit the constructed native seed parameters/covariance, initial
   hit/crossing identities and times, and persistent ParameterHit time/constraint/mask. Reconstructing from
   saved slopes and final candidates does not reproduce float-normalized native seeds or construction cuts.
2. **Reference-piece search.** The diagnostic assigns candidate references using closest-approach times on the
   previous reference. Native default `PiecewiseClosestApproach` recomputes TOCA and searches candidate pieces.
   Exact-boundary assignment also differs (`searchsorted(..., side="right")` vs native earlier-piece choice).
3. **Crossing offset.** The diagnostic uses wire radius/current speed each iteration. Native `KKStrawXing`
   captures the time offset at construction, including a separate construction for extension crossings.
4. **Final candidate associations.** Seed membership is inferred from saved hit indices and crossing partners
   from same-straw matches. These are not native object-identity or per-iteration membership snapshots.
5. **Crossing attribution proxy.** Saved `xinb` is a projection on the final t0 line, bounded by stock-final active
   hits. It is not actual time-ordered membership (which includes the constraint). The report labels these bins
   `inside_proxy`/`outside_proxy` and explicitly disables causal interpretation.
6. **Momentum diagnostic.** `mterm` is the front marginal momentum displacement squared over variance at the
   maximum endpoint-chi2 iteration. The maximum can be at the back and the joint statistic has correlations.
   It is a diagnostic, not a decomposition or proof that momentum alone causes the rejection.

Relevant native source: `Offline/Mu2eKinKal/inc/KKStrawXing.hh`, `KKStrawHit.hh`, `KKFit.hh`,
`KinKal/Trajectory/PiecewiseClosestApproach.hh`, `KinKal/Fit/{Track,MetaIterConfig}.hh`.
The frozen reference is Offline `9b3d463371` and KinKal `bb23c26`; inspect the matching source/build, not current head.

## Report and gates

`scripts/run_f1_closure_report.sh OUT.json NAME=DIR...` invokes the tracked report on completed arm products.
`f1_closure_report.py` checks the highest-cycle input tree has the expected 3,000 events, enforces a single run,
requires full uncapped stock histories, rejects duplicate/missing identity joins and metadata count mismatches,
and reports numerical-validity populations without changing acceptance. Nonfinite parameter pulls remain in the
both-good fidelity denominator and fail the criterion. Summary outputs preserve absolute data paths as provenance.

A completed `vr2f1.json` is required; log progress counts are not event counts. Historical results without native
capture cannot establish exact seed-time or successful-stock iteration operands.

The isolated native-capture area must first be built by the user, including the matching KinKal install. Do not
run against it until the user confirms completion and linkage/timestamps are verified. The first runtime gate is
frozen baseline → new capture off → new capture on equality on a small prefix and then the complete 3,000 events.
Only then design a separate, isolated exact immutable-time arm. Do not silently combine it with the prior arm.

## Outline: stock-independent F1 (not implemented here)

1. **Detector-only input boundary.** Start from digis, measured conditions and matching geometry; use the existing
   hit/combo reconstruction and LineFinder implementation. Do not read any `sd_*`, `kh_*`, `kst_*` or `kl_*`
   stock-fit products in the autonomous execution path. Keep stock comparison in a separate evaluator.
2. **Seed construction.** Build candidates from measured hits, reproducing seed ranges, native normalization,
   construction cuts and constraints. Record explicit event/seed identities and empty/multi-seed cases.
3. **Candidate generation.** Construct seed crossings from detector geometry; add hitless crossings and extension
   hits from the evolving F1 trajectory, with construction-fixed offsets and actual partner identities. Never
   condition candidate availability on stock acceptance or its final lists.
4. **Iteration contract.** Retain owned piece ranges and implement the native configured reference search, ordering,
   bounds, priors and status tests. Validate operands using the passive capture before claiming faithful closure.
5. **Output boundary.** Adapt to the EventNtuple-compatible writer on current tracker main, preserving the existing
   writer schema/provenance tests; do not duplicate that writer in this old campaign worktree.
6. **Validation.** Keep a same-hit/same-seed unit comparison to isolate the fitter, then run the unconditional
   denominator (all input events, including zero/multiple seeds) on MC and KPP data. Report paired yield exchange,
   fit failures, active hits, parameters/errors and independent truth pulls where truth exists. Use a held-out
   sample for any learned/measured correction; truth-conditioned models are diagnostics only.
7. **Performance only after fidelity.** Measure one CPU/A100 run with fixed precision and explicit compile/I/O
   accounting. No GPU speed claim from the CPU closure campaign.
