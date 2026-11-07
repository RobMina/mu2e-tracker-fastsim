# Saved-stage attrition after the stock census

This read-only diagnostic follows the [stock comparison](run1a-stockcompare.md). It changes no
reconstruction code, support, LR assignment, numerical threshold or status. Stock coverage is a
reporting partition only, never an input to recovery. Every event, group, hit and direction lane
in the comparison payload remains present. No exhaustive association or fit is rerun.

## Contract

`validation.run1a_stockattrition` verifies the saved comparison payload hash, its input hashes,
the original LR payload and measured-hit identity binding. It exactly reconstructs the stock
comparison tables from their saved inputs without reading new physics. It binds the saved
association payload to the LR report and checks event/group/hit identities, fixed support,
candidate-source membership, unique source pairs, candidate counts, choose(n,4) totals and
nested recorded independent/fitted/supported counts. Those last three counters are **saved
values**, not independently rerun hypothesis fits or support scoring. Historical association
input authentication gaps are not repaired by current hashes.

Association-stage labels are an exclusive availability hierarchy: unsupported geometry, wide
time group, fewer than four candidates, no independent quartet, no fitted hypothesis, no
supported hypothesis, supported. The earlier unambiguous-control rejection bits are **not**
association vetoes; shared stereo alternatives are legitimate inputs to the separate association.
Counts of groups and counts of direction lanes are explicitly distinct. The full `groups.csv`
contains all773 group rows, not only the stock-covered failures. It is a diagnostic inventory,
not a physics-efficiency denominator. `--compare-saved` compares payload arrays only, not reports
or CSV serialization.

```bash
python3 -m trkfastsim.validation.run1a_stockattrition \
  --comparison SAVED_STOCK_COMPARISON_DIRECTORY --outdir NEW_DIRECTORY
```

## What the saved states say

Across all773 groups,429 lack selected support:219 have fewer than four stereo candidates,
187 have no independent quartet, and23 have fitted hypotheses but none with sufficient support.
There are344 supported seeds. Of122 supported groups with neither final direction passing,
89 have two `invalid_seed` lanes and33 reached objective-valid fitting. These are distinct
upstream and numerical availability limitations; changing the final success label cannot
repair missing seeds.

The three missing-seed groups within the38 recorded stock CH events are all run1430, subrun0:

| Event | Good group hits | Time span (ns) | Stereo candidates | Distinct represented source hits | Saved independent quartets |
|---|---:|---:|---:|---:|---:|
|317|47|30.590|0|0|0|
|8249|66|42.752|0|0|0|
|8400|26|33.914|6|8|0 of15 raw quartets|

They are not empty detector events or wide-time-group rejections. Merely raising the association
hypothesis budget cannot create missing stereo candidates or an independent quartet in the saved
candidate set. The projected-wire stereo approximation ignores local track slope; diagnosing its
pair cuts precedes any proposal to replace that approximation. No relaxed-cut rescue is claimed.

The seven supported CH nonpassing groups are events1733,2051,2131,3013,6837,8563 and8679. Their14
final direction certificates are5 `fd_check_failed`,5 `no_admissible_stencil_pair`,3
`analytic_nondifferentiable` and1 `analytic_nonstationary`: **zero continuous passes**. Therefore
removing sticky numerical-history vetoes or ignoring pending LR flips at the saved final states
would recover none of these seven. This does not prove that a different optimization trajectory
or additional iterations could never help; it shows why relabeling alone is not a solution.

Raw termination, certificate, final status and sticky histories are kept separately. For example,
events2131 and3013 fail the adjacent-stencil check in both directions, while8679 fails the FD
consistency check in both. Event8563 is boundary-guarded in both directions, one also having a
sticky stall. Event1733 has two pending hit-sign flips in the positive-z hypothesis and2051 has
four in the negative-z hypothesis; both also fail the continuous criterion. No flight winner
or common true-particle identity follows from these labels.

## Next reconstruction decisions

Use the measured pair-cut replay to distinguish unavailable local stereo geometry from longitudinal
or timing rejection before designing a different seed representation. For supported fits, preserve
the fixed thresholds and develop independent root/arithmetic and branch-aware checks rather than
calling a smaller stencil exact or smoothing away stock calibration boundaries. Both changes need
measured-only inputs, all-group evaluation and fresh opt-in outputs. Stock-selected events are useful
examples, not a target set on which to tune tracking efficiency.

## Measured stereo-pair replay

`validation.stereopairaudit` independently rebuilds flag/time membership and evaluates the fixed
production pair predicates on **all** eligible saved groups. It does not use stock coverage,
association support, control ambiguity bits or candidate-used masks to choose pairs. This is a
bounded diagnostic replay of pair construction, not another association or fit. Original-dtype
geometry prechecks/time sorting precede float64 pair arithmetic. Unsupported groups remain whole;
no hit is removed to make a geometry check pass.

All773 groups remain,758 eligible and15 singleton. Across121499 considered unordered pairs,
exclusive first-failure counts are39786 parallel,61357 far-z,1987 timing,3505 wire-length and7472
longitudinal wire-distance; none first fail nonfinite or same-z. The7392 accepted original source
pairs match the saved candidates exactly, including multiplicity. Canonical replay holds the whole
accepted-pair set; only evaluation temporaries are pair-batch limited.

| Event | Pairs | Parallel | Far-z | Timing | Wire length | Wire distance | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
|317|1081|559|0|43|236|243|0|
|8249|2145|1200|0|73|872|0|0|
|8400|325|84|229|4|0|2|6|

These are **ordered first-failure counts**, not independent causal contributions. In particular,
the zero wire-distance count in8249 means no pair reached that test; it does not demonstrate sound
longitudinal measurements. Projected XY wire intersections need not represent a sloping particle's
finite-wire crossings. This evidence motivates a slope-aware or direct-wire seed alternative, not
a claim that loosening one cut would recover physical tracks. The8400 association limitation remains
its saved zero independent-quartet count, not a lack of all stereo candidates.

```bash
python3 -m trkfastsim.validation.stereopairaudit \
  --hits SAVED_HITS.npz --conditions SAVED_CONDITIONS.json \
  --association SAVED_ASSOCIATION.npz --outdir NEW_PAIR_AUDIT_DIRECTORY
```

The complete pair counts and candidate arrays are byte-identical at batches65536 and17. Accepted
`stereopairaudit-1000-20261106-final` NPZ SHA256:
`293b3fae001cf8ef8c3e54b17ab5b4ff86f5a3c0231957c9d8b0e63fd69b9510`.
The saved-stage inventory `stockattrition-1000-20261106-final` is array-exact against earlier
runs consuming both default- and17-entry stock comparisons; NPZ SHA256:
`e2a657cf8dcb98c87ea6ab7f793e8aedf3382ba3477696238160cd95cd6d6c7a`.
No statuses or earlier payloads were edited. Input/source hashes and missing historical bindings
are recorded explicitly. Both new modules have synthetic malformed/empty/masked, group/source
alignment, cut-order, exact-repeat and no-overwrite tests. Latest focused run:1730 passes across28
selected files, including268 attrition and55 pair-audit tests (323 new tests). No CI or full-suite
claim. Earlier delegate figures147/175 included existing tests, not147 new pair-audit tests.
