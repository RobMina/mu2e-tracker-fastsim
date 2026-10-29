# Actual F1 iteration tracing

This default-off diagnostic records operands produced by `kalman_jax.fit_track`,
not a replay using native measurement weights or material changes. It is separate
from the native-input algebra audit. No fit policy, threshold or default changes.

## Capture and passivity

`fit_track(..., time_model="stock", trace=True)` adds `cf["trace"]` while preserving
ordinary outputs. The trace is supported only with the covariance filter. A static
per-stage buffer covers the full iteration cap; the outer schedule collects stage
buffers. `valid` excludes speculative computation after a stop or failed extension
gate; rejected executed iterations remain valid with `swapped=false`. Consumers must
mask `valid`. There are no fabricated later iterations or provisional FAST_NITER fits.

The supported driver executes **one explicit target**, no vmap or 256-lane padding.
It rejects a conservative payload estimate above 64 MiB before fitting. This bounds
trace buffers, not total JAX compilation/runtime memory. A future batched caller must
check `allocation_bytes(..., batch=actual_batch)` outside vmap; the fitter's static
single-track check cannot enforce a batched allocation. Batch tracing is not supported
by this driver.

```bash
bash scripts/run_f1_trace.sh NEW_OUTPUT 1400:0:75919:0 --native-time
bash scripts/run_f1_trace.sh OTHER_OUTPUT 1400:0:75919:0 --native-seed
```

`--native-seed` includes the same certified native constraint time. Compare it with
`--native-time`, not a silently changed timing baseline. The driver verifies the full
four-hash native evidence gate, requires a captured eligible single-seed identity,
runs the identical ordinary and traced F1 at full cap, and requires exact equality
of every ordinary output leaf and tree structure. It writes only executed records
and checks their count against F1's final iteration count. The output directory must
not exist. With run-and-capture, use a log such as `NEW_OUTPUT.driver.log`, not
`NEW_OUTPUT.log` (the shell driver's overwrite guard reserves that name).

`manifest.json` binds input/check, conditions, alignment, ANN and source hashes,
explicit options, schedule, passivity and output hash. Diagnostic F1 environment
variables are refused, rather than silently changing driver semantics. No population
acceptance/fidelity claim follows from a target trace, nor from single-track passivity
versus the existing differently shaped batch campaign.

## Schema 1

Each saved record has one stage index and local iteration index. `owner_count` and
`candidate_count` are **last valid piece indices**, not counts: zero means one piece.
Owner/candidate arrays have trailing padding; only indices through that value exist.

- `reference_*`, `physical_time`, `ordering_time`, `reference_query_time` use natural
  order: all saved hits, all saved crossings, then the constraint (reference tensors
  omit the constraint). `reference_query_time` captures the actual lookup time,
  including the already-computed crossing-offset subtraction; it is not reconstructed.
- `eff_*` and `is_material` use **sorted kernel order**. `order` maps sorted slots to
  natural slots; `argsort(order)` undoes it. `ordering_time` has padding/excluded
  crossings at the kernel sentinel; `physical_time` retains actual computed times.
- `eff_xref,H,r,w,mvar,D,sv,dm,dmv` are the exact filter input arrays. Hit activity,
  crossing activity/in-bounds, geometry/path/offset/transit values are also retained.
- Old and candidate owners, undeweighted prior parameters/covariances, actual deweight,
  status query/reference operands, endpoint quadratics, previous/current chi2/ndof,
  status and actual swap decision are captured without refitting.
- Hit source indices, raw crossing straw IDs, source-index partners, initial hit
  membership and full run/subrun/event/seed identity accompany the arrays.

The kernel carries final saved candidates before native extension construction.
Initial membership is inferred and checked separately; kernel padding or inactive
candidates must not be mislabeled native constructed effects.

## Coordinate conversion and first-iteration comparison

`nativecompare.py` uses the full coordinate Jacobian, including momentum-dependent
flight time. Measurement conversion includes the affine origin shift:
`Hy = H @ inv(J)`, `W = Hy.T @ diag(w) @ Hy`,
`b = Hy.T @ diag(w) @ (Hy @ native_reference - r)`. Simply transforming the local
information vector is wrong. Material shifts/noise use an explicitly first-order
conversion; an exact finite coordinate shift is a separate diagnostic. Invalid
geometry/nonfinite operands fail without regularization.

The current comparator supports **only the first actual iteration**. It requires
ordinal zero, local iteration zero and a one-piece initial owner, without assuming
native miter numbering. It checks full identities, duplicate/ambiguous semantic
identities, source-hit and unique-straw associations, initial references, hit activity,
bounds/priors and available measurement/material operands. Native cluster effects
are unsupported, not dropped or approximated using final cluster membership.

F1 material kernel operands are masked outside participation; native caches may retain
nonzero changes there. Participation is compared first; numerical material comparison
uses the explicitly counted intersection. Missing or unsupported operands remain
reported. Derived native-coordinate prior weights are comparison quantities; F1's
actual filter receives covariance-form priors, not those derived information weights.

The component criterion remains `rtol=1e-5, atol=1e-7`; identities are exact. Phase
order is reporting order, **not proof of causal precedence**. Passing small seed
differences under this criterion is not bitwise equality. Native CA internals,
ordering/geometry equivalence, candidate/decision closure and later-iteration history
alignment are still untested. No all-discordance attribution follows from two targets.

Recompare saved operands without another fit:

```bash
python3 scripts/f1_trace_compare.py --trace-dir SAVED_TRACE \
  --check CHECK.json --out NEW_COMPARISON.json
```

The recompare verifies trace/manifest/native-gate hashes and retains first mismatching
component indices/values plus matched effect slots. Original manifests/results are
never overwritten when comparison logic is corrected.

A separate [measurement-row audit](f1-measurement-rows.md) substitutes source-derived
time/longitudinal derivatives into saved first-iteration operands, holding values
and weights fixed. It is not a fitter option or a refit, and retains strict failures.
