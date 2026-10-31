# Associated-extension crossing construction — exact bounded agreement

Status 2026-10-28: **all 81 associated-extension crossing constructors across 50 targets agree exactly** with the native capture. This follows the [full passive crossing capture](f1-crossing-capture.md). It is conditional constructor replay, not autonomous F1, full history, material discovery or full-fit closure.

New vectorized helpers: `scripts/f1_crossing_constructor.py` and `f1_crossing_constructor_audit.py`. The fitter, native capture sources, receipt-bound existing helpers, thresholds and defaults are unchanged. No native job, build, refit, baseline regeneration or grid submission occurred.

## Input separation

Inputs to generation:

- Raw **tube cal/hv endpoints**, native wire radius and target-specific geometry bindings from the validated new geometry sidecar. Wire endpoints are not substituted for tube endpoints.
- The supplied native pre-extension construction trajectories: exact piece order/ranges, native parameters and mass. This is not independently generated fitted trajectory evolution.
- Independently reconstructed upstream **hit-constructor** particle/sensor times from pinned `crossing_observability_v2.json`, which inherits `transitions_v5.json`. These are not the crossing's captured hints, wrapper-reference times, or final effect times.
- Accepted associated membership, object/trajectory context and full-key joins remain source-conditioned inputs. No excluded-hit discovery or acceptance-efficiency claim is made.

Generation constructs SensorLine from cal as the measurement/end point, hv as start, c_light as signal speed and hit-constructor particle time as measurement anchor. Fixed source grouping uses the existing `nativearith.sensor`; the unchanged `f1_later_ca.piecewise` performs native-style iterative local CA and piece search at `float(np.float32(1e-4))`, up to ten piece moves and 100 local updates. It preserves earlier-piece ties and pointer-before-stop/cap behavior.

The private constructor offset is generated as **wire radius / final reference-piece speed at retained particle TOCA**. At a cap, this reference pointer can differ from the last solved piece, so the solver's retained solution speed is not blindly reused. Offset is not inferred by subtracting effect time, and no material length or parameter change is inverted.

Recorded crossing endpoints, direction, length, time anchor, incoming hints, CA results, search indices/history, radius/speed operands, private offset and effect time are **comparison-only**. Synthetic outcome perturbations leave generated outputs unchanged and fail the comparison. Exact comparison is reported separately from the existing rtol=1e-5, atol=1e-7 criterion; neither was tuned.

## Result and population boundary

Every declared constructor agrees exactly for:

- Tube-line start/end/direction/length and time anchor.
- Incoming particle/sensor hints.
- Particle/sensor times, both POCA positions and both direction vectors.
- Usability/status; initial, final reference and retained solution indices; solved-piece sequence and number of solves.
- Final reference parameters/range, native radius/speed operand, private stored offset and offset-inclusive effect time.

**81/81 constructors, 50 targets, zero exclusions, zero moved searches and zero exhausted constructor searches.** These exact equalities are observed for this sample/build, not universal compiler/ROOT parity. Cached CA derivatives/variances and all intermediate per-solve scalar trace fields are not numerically regenerated here.

The overall capture census remains 2,538 initial +81 associated-extension +2,471 material-only constructors. Initial and material-only constructors are outside this declared comparison, not failed comparisons dropped. All 142 category-1 trajectories are retained and validated: 50 have an accepted associated constructor; **92 have none**, because native extension entry captures its trajectory before any selection. The audit must not require every construction-context trajectory to create an object.

All seven exhausted searches are **later stage-1 reference updates**: six on initial crossings and one on a material-only crossing. None is a constructor or discovery search. Their full keys are retained in the report. No retained-history/cap-physics validation is claimed by counting them.

## Receipts and measured resources

Data prefix: `$M2FS_DATA/trk_f1native_r2_20261027/`.

Final `crossing_constructor_v4.json` SHA256:
**`8f13afc822ef99590b36c47764251f1fdefb994117cb0bdeb6b48a98818bfe2e`**.
`completed=true`, `passed=true`, `comparison.exact=true`, `hashes_unchanged=true`.
`full_fit_closure`, `full_f1_closure`, `full_history_closure`, `full_numerical_validation` and `independent_discovery_replayed` remain **false**.

- Generation/comparison helper SHA256: `7a1dd18bd18f81a3f7ee69901112c78068cec988a7760752cbc5cc5963f213b5`.
- Audit driver SHA256: `f3268c65ef0083dbaacd615d0dbb8bcca060d3c53617d84077c7c99515000bd7`.
- Full capture receipt: `crossing_3k_20261028_v2/crossing_full_check_v1.json`, SHA256 `c8e1225b1cc71ad82ff7055c77cd382c1b564956a1470522136d48575d53f8ec`.
- Upstream hints receipt: `crossing_observability_v2.json`, SHA256 `20217e4e7a32791d14291d64418ac2bf5873140b8a5b2930090fb9613e8b6626`.
- **Measured final audit:12.47s wall,1,087,044KiB maximum RSS**, one CPU, GPVM02 `pyenv ana`. No runtime extrapolation.

Every inherited full/hint receipt input is rehashed before and after the audit. Actual module paths, required function code origins/wrapper chains and imported aliases are checked. Evidence paths are resolved once before use; outputs are exclusive and outside source/native/input paths. Strict failure JSON retains integrity evidence; residual overflow fails rather than serializing nonfinite statistics.

## Tests, review and preserved attempts

**136 new synthetic tests pass** (100 helper,36 audit). The selected **eight-file isolated suite:942 passed**; includes later-CA/search-cap, native CA/arithmetic, crossing record and geometry regressions. No new full/slow-tier certificate. Laptop log: `local_data/trk_f1native_r2_20261027/constructor_isolated_v1.txt`.

Tests cover full-key permutations/missing/duplicate identities, signed/unsigned overflows, masks, unsupported flags, counterfactual recorded-output independence, reversed endpoints, geometry-sourced radius, zero-duration pieces, no-object extension contexts, cap reference-versus-solution speed, receipt/path/hash binding, imported/foreign function rejection, canonical-path symlink handling and strict failure serialization.

A read-only interface review established the cap-speed distinction. A first implementation review exhausted its budget without findings and is not counted as approval. A replacement review identified canonical-path, function-binding, boolean typing and overflow/failure-JSON holes; all four were addressed with regressions, without modifying existing bound helpers.

Preserved reports/logs and source snapshots:

- **v1** stopped before numerical comparison because the initial audit incorrectly demanded every category-1 trajectory have an accepted associated constructor. Inspected native `extendTrack` captures before selection; the corrected audit retains and counts all 142 contexts, including 92 without an object. v1 SHA256 `4d09ae7ccd1202559384a5c20d96daf9572264b2fce9ed2ee416ae5dddacb464`.
- **v2** first exact81/81 result, before hardening; SHA256 `e59cbb7fa929d09df8190b237937e763ae2f4dd32f5e762bded112557eae0144`.
- **v3** failed the new discrete-type gate because padding native `vector<unsigned long long>` with -1 promoted the array to float. The fix validates bounds then explicitly converts to int64 **before** padding, preserving integer comparisons; no tolerance is relaxed.
- **v4** final hardened result; versioned source snapshots under `crossing_constructor_v{1,2,3,4}_sources/` preserve the code used by earlier attempts. Earlier receipts are never rewritten to point to a new source hash.

## Reproduction and next boundary

On a freshly checked, owned idle GPVM in `pyenv ana`, set `PYTHONPATH` to this worktree and all BLAS/OMP thread counts to1; use one CPU and a new report/log:

```bash
python3 scripts/f1_crossing_constructor_audit.py \
  --full-check $M2FS_DATA/trk_f1native_r2_20261027/crossing_3k_20261028_v2/crossing_full_check_v1.json \
  --observability $M2FS_DATA/trk_f1native_r2_20261027/crossing_observability_v2.json \
  --capture $M2FS_DATA/trk_f1native_r2_20261027/crossing_3k_20261028_v2/capture.root \
  --native-source $TRKFS_MUSE_AREAS/fastreco_crossings_p109_20261028 \
  --out $M2FS_DATA/trk_f1native_r2_20261027/NEW_constructor_report.json
```

Next: independently reconstruct initial hit-constructor hints and all2,538 initial crossing constructors, then reference/wrapper/partner history including the seven caps. Material-only discovery requires its complete geometry-universe and rejection-stage replay, not only accepted2,471 constructors. Native seed/state-policy/numerical/decision/acceptance closure remains subsequent work. Do not integrate this diagnostic into the fitter or change defaults based on conditional agreement.
