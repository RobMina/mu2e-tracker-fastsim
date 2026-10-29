# Native construction and seed diagnostics

These are separate stock-conditioned experiments. No default, candidate collection,
activity threshold, fit threshold or reference-selection default is changed.

## Construction-time offsets

`--time-model stock --crossing-time constructed` freezes initial crossing offsets
at wire radius / seed speed. At the full schedule's extension gate it selects a
reference on the pre-extension F1-owned trajectory and freezes non-seed offsets at
that speed. All later iterations reuse those offsets. This changes effect times
but not the material kick location, candidates, hit partners or initial owner.

The extension reference is reconstructed using the saved candidate's prior hint.
It is not a native constructor snapshot or an autonomous candidate finder. Initial
hit membership and same-straw partners must be audited independently before use.
The capture records initial offset as a time subtraction rather than the private
constructor field; extension offsets are absent. Do not call this native-exact.
A schedule without its seed-to-extension gate cannot certify extension construction.
The offset diagnostic freezes each valid non-seed crossing once, only when the
extension gate is reached without a stopped/rejected seed. It separately records
`construction_unsupported_tracks`, `construction_unsupported_crossings` and
`constructed_crossings`; `construction_search.npz` carries the per-track counts
and full run/subrun/event identities. Initial seed offsets use seed speed without
a construction search. These counters are distinct from
`reference_unsupported_tracks`, which covers later reference updates. A zero
construction counter does not certify the absent native constructor snapshots.

`scripts/f1_native_contract_audit.py` verifies the full capture gate and hashes,
then checks initial hit/crossing membership, duplicate straw identities, initial
partner links, seed momentum and offset precision against final saved candidates.
Its `--seed-check` additionally validates the native seed loader without fitting.

A later [meta/extension continuity census](f1-transition-continuity.md) covers all
147 captured targets: 81 new associated-hit crossings plus 2,471 material-only
crossings, with no observed disappearance or duplicate crossing straws. Partner
metadata is stamped from final collections; source order supports bounded constructor
association inference, not an iteration-local snapshot. All 2,552 extension private
offsets and constructor CAs remain unobserved. The earlier fit arms are unchanged.

## Native seed operands

`--native-seed-check CHECK --native-time-check CHECK --time-model stock` supplies
captured native seed parameters, full covariance and absolute initial range. The
same explicit native constraint time is required in the comparison arm; the seed
option must not reset it implicitly. Use the preserved native-time-only baseline.

The six-dimensional transformation maps native KinematicLine parameters to the
fitter's y=0 intercept/slopes/time/momentum coordinates, including the momentum
dependence of the time shift. Covariance is `J C J.T`; no clipping, ridge or diagonal
replacement is allowed. All full run/subrun/event/seed identities are joined and
validated. Unsupported schema, mass, direction, covariance, momentum or momentum
constraint fail before fitting.

Seed replacement necessarily changes initial prior values and the seed-derived
momentum-constraint target. The prior policy, material model, ordering algorithm,
thresholds and candidate/association tensors remain unchanged. Propagated values
can change even when those policies do not. `native_seed.npz` records transformed
operands and identity keys; the capture and metadata retain native operands and
hashes. This is not a stock-independent seed builder or a full native contract.

## Separate comparisons

```bash
bash scripts/run_f1_closure.sh NEW_CONSTRUCTION_OUTPUT \
  own_stock_xconstructed fixed_stock_xconstructed
F1C_NATIVE_CHECK=CHECK.json bash scripts/run_f1_closure.sh NEW_SEED_OUTPUT \
  own_stock_native_seednative fixed_stock_native_seednative
```

Do not concatenate these into one changed arm before the isolated comparisons have
been evaluated. Measure acceptance exchange and fixed-state fidelity separately,
retain rejected fits in their required denominators, and retain nonfinite failures.
