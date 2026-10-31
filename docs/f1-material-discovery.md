# Material discovery and generated-constructor crossing history

## Result — 2026-10-28

**Material discovery is exact for all 38,174 records / 142 material-entry contexts**, including every reached rejection/exclusion and all **4,417 attempted CAs / 5,900 per-piece solves**. All **2,471 accepted material-only constructors** agree exactly. **1,483** attempted CAs move pieces; none exhausts the crossing move cap. `zTime` uses at most two solves in this sample; no 100-step cap occurs.

The separate new history audit removes the captured-material initialization used by [history v2](f1-crossing-history.md): **all 5,090 constructors are independently generated**, with **zero captured constructor CA states/anchors/private offsets** consumed. All **105,578 later searches / 213,617 records / 139,202 per-piece solves**, including **33,560 moved searches and seven exhausted searches**, remain exact with zero exclusions and 32 batched dependency waves.

This is conditional discovery/construction/CA-history evidence, **not autonomous F1 or full-fit closure**. Native trajectory evolution and covariance, raw target geometry, initial/associated accepted membership, and supplied object chronology/partners remain conditioning inputs.

## Complete geometry universe and stage census

The pinned full capture contains 147 complete target geometries. All 142 material-entry contexts are retained, whether or not they produce an accepted object. Each traverses a complete universe of 36 planes, 216 panels and 20,736 straws: **5,112 planes / 30,672 panels / 2,944,512 straw candidates** in the material-context universe.

The replay does not evaluate fabricated PCAs for descendants of failed parent gates. It starts from the full hierarchy, generates every gate/window, and compares the complete reached stream and its order; excluded descendants are determined by those generated gates. It never takes a captured candidate list as input.

| Stage | Meaning | Reached | Passed | Rejected/excluded |
|---:|---|---:|---:|---:|
| 0 | Normal branch and tracker limits | 142 | 142 | 0 |
| 1 | Plane exists | 5,112 | 5,112 | 0 |
| 2 | Plane radial gate | 5,112 | 2,002 | 3,110 |
| 3 | Panel straw-number window | 12,012 | 1,921 | 10,091 |
| 4 | Panel longitudinal gate | 1,921 | 1,191 | 730 |
| 5 | Candidate window bounds | 1,191 | 1,191 | 0 |
| 6 | Not already present | 5,796 | 4,417 | 1,379 |
| 7 | CA and strict physical cuts | 4,417 | 2,471 | 1,946 |
| 8 | Accepted constructor link | 2,471 | 2,471 | 0 |

All per-stage integer/scalar operands and booleans, full-key memberships, target-local chronology, and accepted constructor object links agree exactly. Initial/associated exclusion membership is reconstructed from their pinned receipt identities, not from captured discovery integer vectors. The single material context per target and unique straw visits are enforced; reflecting/multiple-extension generalization is not claimed.

## Source semantics preserved

- `KKFit::fillTrackerInfo` uses the first panel's **wire** origins/midpoints for `ymin/ymax`, the outer **wire cal endpoint** magnitude for `rmax`, outer straw radius for buffers, and `(96-1)/(ymax-ymin)` for pitch. Cached limits must be identical across supplied target geometries or the audit stops.
- `HepTransform` applies `displacement + rotation * position`, with row-major captured inverse coefficients and fixed three-term source grouping. No geometry normalization/reconstruction from transformed basis differences.
- `PiecewiseTrajectory::t0` performs at most ten self-consistency pointer transitions; `KKFitUtilities::zTime` starts at the trajectory range begin and performs up to 100 transitions. The panel position uses the nearest piece at the returned `zTime`, including the native cap-pointer semantics.
- Frozen settings are `AddStrawMinDz=0`, `MaxDStraw=2`, `MaxStrawDOCA=5`, `MaxStrawDOCAConsistency=1`, `MaxStrawUposBuffer=0`, `StrawNBuffer=0`, `SkipStrawCheck=false`. Strict `abs(dir.z)<0` makes the uninstrumented vertical branch unreachable. No threshold changes.
- Panel rough straw number uses nearest-even `rint`; buffer integer conversion and clipped inclusive window follow source order. Native float32 `halfLength` is preserved before promotion in comparisons.
- Each fresh straw uses **tube endpoints**, a SensorLine anchored at generated plane `zTime`, and both hints equal to that time. The unchanged piecewise/local CA implementations perform ten piece moves / 100 local updates, with promoted float32 precision `9.999999747378752e-5`.
- `du` is computed with the **wire midpoint and wire direction**, not the tube midpoint. Strict acceptance is `abs(doca)<5`, `max(0,abs(doca)-radius)/sqrt(max(docaVar,minDouble))<1`, and `du<halfLength`.
- CA DOCA/derivative/variance generation uses the **last-solved piece** and its full covariance. The constructor offset instead uses **final-reference-piece speed**: `wireRadius/referenceSpeed`, fixed throughout history. No subtraction of captured effect times or inversion of material effects.

Every attempted CA matches retained times/positions/directions/status/usability, line, hints, reference parameters/range, DOCA, DOCA variance, DOCA/time derivatives, piece indices and visits, and every per-piece input hint/solved time/status/usability. Source-grouped Similarity gives exact agreement for this sample; this is not universal compiler/ROOT parity. CA time variance and every internal local-solve arithmetic step are not separately certified here.

## Generated history and limits

`scripts/f1_discovered_history_audit.py` uses a new bootstrap rather than calling the old captured-material bootstrap. It joins the initial 2,538, associated 81 and material 2,471 generated receipts by full target/object identity, then calls unchanged history replay/comparison. Constructor outcome columns can be removed in tests without changing bootstrap; changing a generated material input changes its later recursive hint.

All later searches still use own usable retained CA. No partner-hit fallback, midpoint fallback, setHit mutation or
category-5 domain replacement occurs. These absent branches are not data-certified. Later cached CA
derivatives/variances, activity/material lengths/effects, native seed generation, strict filtering/inversion/algebra,
decisions and acceptance remain outside this certificate. The 99.5% acceptance and 99% parameter-fidelity goals remain
unmet; no fitter/default promotion follows.

`independent_discovery_replayed=true` in the material receipt means **discovery on supplied native trajectories, geometry and initial/associated membership**. `generated_constructor_ca_history=true` in the new history receipt means no captured CA bootstrap remains. `full_history_closure`, `full_fit_closure`, `full_f1_closure`, and `full_numerical_validation` remain **false**.

### Cap-key documentation correction

The immutable old history-v2 JSON and new discovered-history JSON agree. An earlier prose/table transposition assigned the wrong index pair to two event86901 sequences. Correct keyed values (run1400/subrun0/seed0) are:

| Event | Sequence | Origin | Final reference | Last solved |
|---:|---:|---|---:|---:|
| 75919 | 499 | Initial | 10 | 9 |
| 81667 | 3134 | Initial | 36 | 35 |
| 86901 | 1638 | Material | **10** | **9** |
| 86901 | 3050 | Initial | **18** | **17** |
| 176008 | 1308 | Initial | 17 | 16 |
| 176008 | 1952 | Initial | 8 | 7 |
| 176008 | 1953 | Initial | 9 | 8 |

All seven have eleven visits/ten moves and final status1. No numerical evidence was changed by correcting documentation.

## Files, receipts and tests

New standalone modules, now receipt-bound:

| File | SHA256 |
|---|---|
| `scripts/f1_material_discovery.py` | `c7d0010d6c3ea152c02c84087bac07f54e434f2aa83949f97a0476e84ffe6107` |
| `scripts/f1_material_discovery_compare.py` | `12b3e330276fab07a1dae313d55eba437eecb848a8010d8975bf7e61275c22e9` |
| `scripts/f1_material_discovery_audit.py` | `676dc1d099bd9d6210c0e302bd649434c65c17b23e1643e348b390ae701e4d91` |
| `scripts/f1_discovered_history_audit.py` | `f2ef7085094956999f3c56d42f1d9b17b25d44c40b5a24959767302b751c996a` |

Data prefix `$M2FS_DATA/trk_f1native_r2_20261027/`:

- **`material_discovery_v1.json`**, SHA256 **`762f61184eec3cdf849457b1c86728fddbbd2d207bf16bec497587b00c8d707b`**.
- **`discovered_history_v1.json`**, SHA256 **`d92044981ee84e31b0647bf0705b69f729c0806903fec0707b02c89cb9703852`**.
- Both completed/passed/exact/hashes_unchanged true; both runners exit0. No real-data failure attempt in this continuation.
- Seal **`material_discovered_history_receipts_v1.sha256`**, SHA256 **`3c5c562658bf8d2cabffe6ac70ea5527528b2726872a0091131d6523b2481a68`**. Verified all13 entries; binds both reports/logs, remote test log and versioned source/test snapshots.
- Existing full-capture, initial-v2, associated-v4 and history-v2 receipts and helper sources are unchanged. Inputs and executing function/alias/wrapper bindings checked; canonical paths, before/end hashes, exclusive strict-JSON outputs.

**122 new tests pass locally and remotely:** 41 discovery helper,25 comparison,21 material audit,35 generated-history
audit. Selected **16-module isolated regression suite: 1,237 passed**; not a full/slow-tier certificate. Local log
`local_data/trk_f1native_r2_20261027/material_discovered_history_isolated_v1.txt`; remote
`material_discovered_history_tests_v1.txt` (5.34s). Tests include full synthetic hierarchies, strict-boundary/float32
behavior, recursive independence, no-alias corruption comparisons, malformed keys/receipts, protected/exclusive
outputs and hash drift. The generated-history permutation fixture has one object per category, not multiple
within-category permutations.

Measured oneCPU GPVM02/ana, not benchmarks/projections: material **16.45s wall /1,771,124KiB maxRSS**; generated
history **18.82s /967,592KiB**. OldGPVM01 archive marker remains pending host-state verification. No native rerun,
build, tarball, cleanup, git writes, fitter/default/tolerance change.

## Reproduction and next boundary

Do not overwrite the named receipts; choose new output paths. The material CLI requires `--full-check`, `--initial`,
`--associated`, `--history`, `--capture`, `--native-source`, `--out`; the generated-history CLI adds `--material`.
Inputs are the frozen `crossing_3k_20261028_v2/{crossing_full_check_v1.json,capture.root}`,
`initial_crossing_v2.json`, `crossing_constructor_v4.json`, `crossing_history_v2.json`, and the new material receipt,
with native source area `fastreco_crossings_p109_20261028`.

Next boundary: native seed contract and remaining state/CAD/activity/material/numerical/decision closure. Reuse all
baselines and the sealed capture. Full-population claims still require the missing seven gains and24 losses, and
unchanged acceptance/fidelity gates. Existing scripts are immutable evidence; subsequent stages use new
modules/receipts.
