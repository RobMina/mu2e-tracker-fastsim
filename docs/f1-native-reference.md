# F1 reference-selection diagnostics

The default fitter is unchanged. `reco vr2f1 --time-model stock` accepts an
additional opt-in `--reference-model`:

| Value | Reference selection after an accepted candidate |
|---|---|
| `previous` (default) | Existing previous-reference TOCAs and right-sided boundary lookup |
| `boundary` | Earlier-piece selection at exact internal boundaries, using previous-reference TOCAs |
| `search` | Earlier-piece selection followed by analytic straight-line CA recomputation and up to ten piece moves |

These options modify neither candidate construction, seed operands, construction
associations, crossing offsets, material, prior selection nor decision thresholds.
They are stock-conditioned diagnostics, not stock-independent reconstruction or
native-exact fit certification. They require the stock-inspired time model.

## Search contract and limits

`kalman_jax.reference_search` selects the piece at the previous particle TOCA,
computes its local line-to-line closest approach, then selects the piece containing
the new TOCA. Selection stops on a stable piece or unusable CA. At the move cap,
the native algorithm updates the reference pointer without recomputing the solution;
the diagnostic flags this case as unsupported. The analytic calculation does not
reproduce native iterative floating-point CA arithmetic, hit z-retries or the
fallback hints for unusable prior CAs. It must not be called native-exact.

`reference_search_state` additionally exposes `solution_index`, `moves`, `usable`
and `exhausted`; the existing `reference_search` tuple view is unchanged. This makes
the split inspectable but does not propagate retained CA geometry through the fitter.
The audit's `analytic_ca_state` census evaluates hit z-retry triggers on the **solved**
piece and reports final-pointer re-solve time differences. Trigger counts are analytic
geometry diagnostics, not captured native retry counts or a native retry implementation.

A track-level accumulated flag records parallel/nonfinite or exhausted searches on
valid saved candidates following accepted iterations, including extension candidates
that the current stock-conditioned fitter carries before native construction.
Consequently this is a conservative diagnostic census, not proof that a native
constructed effect exhausted its search. The flag does not alter acceptance.
`vr2f1.json` records `reference_model`, `reference_unsupported_tracks`, and
`stock_exact=false`; non-default arms also write `reference_search.npz` with full
single-run subrun/event identities. Unsupported cases must be investigated before
claiming closure, even if acceptance or parameter metrics improve.

## Captured-operand audit

`scripts/f1_native_reference_audit.py` requires:

- a successful full structural/passivity gate, with all four evidence hashes;
- an existing `ids.npz` baseline, without rerunning that fit;
- measured conditions and the alignment table, recorded with their hashes;
- an unused JSON output path.

It reads highest ROOT cycles, checks full seed identities, measures which current
baseline discordances have iteration capture, and compares reference selection
between adjacent same-meta-iteration captures after a successful swap. Native
initial-crossing offsets are taken from the seed contract **only for this operand
audit**; the fit diagnostic continues to use its unchanged crossing construction.
Extension transitions and crossing offsets absent from the seed contract are
excluded explicitly. Conditions wire geometry and reconstructed straw axes are
not independently certified native geometry. The output is not a new passivity
certificate, effect-algebra certificate, or acceptance result.

The independent NumPy implementation in `reco/nativeref.py` uses native line
parameters. Tests compare its selection to the JAX fitter's coordinates, including
boundary ties, multiple moves, independent lanes and move-cap pointer/solution
splitting. Audit joins reject missing parents and malformed piece identities.

## Opt-in shared-panel geometry

`vr2f1 --straw-geometry panel --straw-axis ALIGN.csv` and the capture audit's
`--straw-geometry panel` infer one rigid aligned frame per complete panel.
`nativegeometry.py` fits all 96 wire midpoints to the nominal two-layer constant
pitch grid plus the wire-end alignment shifts. The CSV values first pass through
float32, matching native `TrkAlignStraw::addRow` (`std::stof`) and float storage.
The shifted straw midpoint and direction are then reconstructed in this frame.
The default `approximate` path and production geometry remain unchanged.

The batched fit rejects incomplete/duplicate straw coverage, rank-deficient designs,
nonrigid grids, nonorthonormal frames and failed endpoint/conditions closure. It
reports residuals and limits without fitting a tolerance to observed discrepancies.
This uses measured conditions only, not track truth or captured fitted positions.
It remains inferred geometry: dumped float half-length rounding is retained in the
corrected direction, and exact native straw endpoints are not recorded.

Separate geometry-only and geometry-plus-search arms are available:

```bash
bash scripts/run_f1_closure.sh NEW_OUTPUT \
  own_stock_panelgeom fixed_stock_panelgeom \
  own_stock_refsearch_panelgeom fixed_stock_refsearch_panelgeom
```

Compare geometry-only with preserved default arms, and geometry-plus-search with
preserved search arms. These change geometry operands (and dependent times/material),
not the reference algorithm, construction-offset policy or native-seed choice.

## Running separate arms

```bash
bash scripts/run_f1_closure.sh NEW_OUTPUT \
  own_stock_refboundary fixed_stock_refboundary \
  own_stock_refsearch fixed_stock_refsearch
```

The driver serializes arms on one CPU and refuses existing arm directories. Use
`f1_closure_report.py` with the preserved `own_stock`/`fixed_stock` arms and new
arms, then `f1_reference_summary.py REPORT.json --out NEW_SUMMARY.json` for compact
metrics and exact-array/byte comparisons. Acceptance agreement and per-parameter
fixed-state fidelity are separate gates; net yield alone is insufficient. Source
manifests must agree across the new arms before interpreting their comparison.
