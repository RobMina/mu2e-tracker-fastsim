# Secondary-content and occupancy validation

## Selection and completeness

`seccensus` and `tv2compare` use the tracker filter on **any primary muon**
(`isprim == 1`, `abs(pdg) == 13`). This is deliberately not the `ismcprim`
convention: fast and reference producers need not mark the same set of primary
muons with that flag. Particle identity remains `(run, subrun, event, key)`;
secondary-only straw classification uses `(run, subrun, event, sid)` tuples,
without packing a straw ID into too few bits.

`tv2compare` requires one explicit EventNtuple row for every selected event,
including events with zero tracks. Missing or duplicate rows are errors, not
physics losses. Optional hit tables satisfy the same contract. ROOT readers use
the highest tree cycle. The selected-event denominator is shared by the rate and
conditional reconstruction factor, and their product is checked against the
overall yield ratio.

Paired census inputs must have identical, nonoverlapping `(run, subrun)` coverage.
A reference file cannot span multiple fast-file groups. Paired reconstruction
also checks per-subrun exposure and livetime, but exposure metadata alone does
**not** prove identical generated event content. `primaries` is CosmicLivetime
exposure, not a finished ntuple event count. `tv2compare` explicitly rejects IDs
outside its nonnegative packed domain: run < 2^15, subrun/event < 2^24. It does
not silently truncate or alias larger IDs. The sparse occupancy/clusterband APIs
also accept full integer `(run, subrun, event)` arrays without that packing.

Paired width errors use matched disjoint observed `(run, subrun)` file blocks,
not the arbitrary order of the input file list. Unmatched/differently partitioned
blocks make the error unavailable. Jointly empty residual blocks are excluded;
fewer than two populated blocks or any invalid leave-one-block replicate makes
the width uncertainty unavailable, never zero. Comparably sized independent
blocks are assumed. The metadata records the reference-to-fast group mapping.

## Statistical interpretation

For panel counts `n_ep` in event `e`, the share is `s_p = sum(n_ep)/N`,
where `N` is the total count. The linearized event-clustered covariance is

```
V_pq = sum_e [(n_ep - s_p n_e)(n_eq - s_q n_e)] / N^2.
```

Sparse event-by-panel matrices accumulate the second moments without event
loops. The full-covariance discrepancy uses the covariance pseudoinverse and
its numerical rank. The Poisson and diagonal-clustered squared-pull sums remain
available for comparison, but are not calibrated chi-square statistics. Plane
profiles include cross-panel covariance. An empty or rank-zero comparison is
unavailable rather than a passing result.

For **paired arms**, `tv2compare` and `seccensus` accumulate raw cross moments
`S_fr = sum_e n_f,e n_r,e^T` on the union of shared event identities; an event
absent in one arm contributes zero there. With each arm's share influence
`u_a,e = (n_a,e - s_a n_a,e,total)/N_a`, the covariance is
`sum_e (u_f,e-u_r,e)(u_f,e-u_r,e)^T`, including both random denominators.
Census pairing uses full `(run,subrun,event)` keys; every accumulation call must
contain complete events and file groups must be disjoint. Cross-moment coverage
must match the per-arm counts. Numerical cancellation is judged relative to the
within-arm variance; identical arms have zero rank, not a spurious passing score.

Paired full-covariance occupancy can certify the stated threshold **conditional
on the fixed library and independent event clusters**. Cross-event library reuse
and library estimation uncertainty are not included. A disjoint reference split
supplies an independent-arm noise floor; a missing half is unavailable.

The paired yield decomposition is additive: tracks in events selected only in
one arm, plus the track difference in events selected in both arms, divided by
the total reference yield. Influence residuals include reference-denominator
fluctuations and the covariance between the two contributions. This differs from
an unpaired selected-rate times conditional-reconstruction factorization.

Census breakdown errors use delete-one-nonempty-file-group jackknife replicates;
fewer than two usable replicates cannot give an uncertainty. Widths with no
file-jackknife error and legacy independent-entry bands are explicitly
non-certifying (`pass: null`, point-only decision retained as a diagnostic).

`clusterband.py` supplies event-clustered per-straw energy means and histogram
ratios (`eion_per_straw.event_clustered*`), plus TDC, TOT and track-chi2 shape
summaries in `tv2compare`. It keeps complete event identities with the entries,
uses sparse event/bin moments and the paired ratio influence
`(u_f - ratio*u_r)/share_r`. The means use all finite occupied-straw energies;
histograms normalize each arm to its **random finite in-range count** and retain
underflow, overflow and nonfinite counts. This differs from the legacy core
band's all-entry normalization, which remains available unchanged. Reference
quantile bin edges are treated as fixed for uncertainty; no edge-selection
uncertainty is modeled. No finite-cluster/Bessel correction is applied.

The maximum marginal deviation minus two standard errors is a descriptive
shape diagnostic, **not a simultaneous confidence band or a calibrated test**.
It does not on its own certify a production shape threshold. Empty/rank-zero
comparisons are unavailable. Separate event-clustered G4 floors retain the same
limitations. These errors also do not include cross-event library reuse.

Each momentum bin is an **event-level truth diagnostic** from its earliest
primary-muon straw step, not a measured-variable response model or a per-muon
ancestry assignment.

## Opt-in off-axis anchoring diagnostic

`steps --offaxis-edge-anchor` changes only the placement anchor for off-axis
library steps with nonzero plane offset. Instead of mapping the library anchor's
relative crossing rank into the recipient muon, it chooses a crossing on the
recipient's first or last plane. The default remains unchanged. This tests
compression of the library's plane extent when a recipient crosses a different
number of straws. A geometry gap or multiple muons in one library event may
violate the assumption that a nonzero offset lies beyond the plane extent; do
not promote this diagnostic on one marginal distribution alone.

Validate the plane-distance profile, spatial occupancy, energy per straw and
stock digitization/reconstruction together. The optional anchor can improve one
profile while degrading another. A replacement anchor also changes panel,
straw, time and the local endpoint frame. The lattice mirror, muon-straw veto
and same-particle/straw merging can then change counts. Source exterior status
must be distinguished from an internal gap before applying an edge correction;
the builder-to-paster internal-gap regression deliberately records this limit.
The builder's event-level `ismcprim` anchor join is distinct from the census's
any-primary-muon convention. Multiple MC-primary anchors can duplicate template
rows and classify another primary as secondary; this requires a separate content
audit, not an implicit change to the builder during placement studies.

### Donor-topology candidate

`buildlib --topology` adds six off-axis columns without changing legacy payloads:
`source_anchor_plane`, `source_plane_min`, `source_plane_max`, `source_exterior`
(−1, 0, +1), `source_plane_rank`, and `source_plane_count`. Format 1/2 still describes
the endpoint representation; topology metadata is an independent opt-in extension.
Old libraries cannot infer exterior status from `dpl` alone.

`steps --offaxis-topology-anchor` requires that annotation and is mutually exclusive
with `--offaxis-edge-anchor`. It maps the donor anchor's fractional **plane** position
into the recipient's plane extent, chooses the nearest crossed plane (lower plane
breaks a tie), then maps rank within that plane with integer multiplication/division.
Only truly exterior donor steps force a boundary anchor. Offsets `dpl/dpa/dst/dt`,
endpoint reconstruction, event-wide muon-straw veto, reflection and merging stay
unchanged. An internal gap self-repastes correctly; it need not remain internal in
a recipient with a different or degenerate plane extent. No random numbers are
consumed by anchor mapping. These are truth-conditioned simulation diagnostics,
not response corrections conditioned on measured variables.

The synthetic closure tests check final SID, time, endpoints, parent, payload and
RNG state, including multiple recipient muons and detector edges. Integer arithmetic
is important: floating-point `floor(15/22*22)` can incorrectly return 14. Nearest-plane
distances are compared as integer numerators with a common per-donor denominator,
so an exact tie always selects the lower plane: floating `(9/14)*21` would lie
slightly above 13.5. Recipient planes must be finite integers in [0,35].

A fixed-library, full saved-sample replay has exact original/annotated-default
step and particle identity in all 20 groups. It preserves primary and in-straw
steps and generator/subrun/surface tables, without rerunning transport. In the
38,431 events selected in both arms, same-plane-other discrepancy improves from
+29.82 ± 2.80% to +2.15 ± 2.35%, but the 1–2-plane deficit remains −13.28 ± 2.00%
and nearby-panel deficit is −16.92 ± 1.79%. Secondary-only paired occupancy
chi2/rank worsens from 1.4910 to 1.6755 (rank 215); mean occupied-straw energy
is +0.552 ± 0.242% relative to G4. **Do not promote this candidate.** Plane-profile
improvement alone does not establish spatial, digitization or reconstruction closure.
These are conditional-library errors, not finite-library uncertainties.

### Joint migration and donor-class diagnostics

The shared-draw audit identifies sampled occurrences before rejection/merging,
then checks aggregate lattice, eIon, length, time and particle keys against both
saved arms. In the paired selection, 20,880 exterior donor rows move from
same-plane-other to more distant-plane categories. Moving only planes or only
panel/straw anchors produces nearby-panel deficits of 57.33 ± 0.86% or
52.47 ± 0.99%; the joint lattice mapping cannot be replaced by independent
marginal corrections.

An exterior-only lattice diagnostic retains the default anchor for internal
and same-plane donor rows. Same-plane-other becomes +3.31 ± 2.28%, but the
1–2-plane and nearby-panel deficits remain −13.87 ± 2.15% and −14.46 ± 1.56%.
The same-plane change relative to baseline is −26.51 ± 0.84 percentage points
of the reference rate, from a paired file-group contrast. Non-exterior-only
remapping contributes little to this correction. Neither diagnostic constructs
endpoints or predicts reconstruction; mixed donor classes belonging to one
particle are merged together afresh, not added as independent final outputs.

Per-file marginal and paired cross moments permit a delete-one-file-group
uncertainty on the covariance-score difference, recomputing covariance and
pseudoinverse per replica. With stable reference support and rank 215 throughout,
secondary-only topology-minus-default is +0.1845 ± 0.2626 and exterior-only-minus-
default is −0.0292 ± 0.1833. Neither change is resolved by this diagnostic.
These are conditional-library sensitivity errors, not calibrated model-selection
p-values; comparably sized independent groups are assumed. All variants were
explored on the same sample, not selected then independently certified. The
residual local deficits and missing digi/reco validation still preclude promotion.
The quoted topology/migration results predate the exact rational nearest-plane
tie fix; they remain historical diagnostics, not validation of that corrected
candidate. Default-L reconstruction does not use the topology hook.

## Full paired saved-default-L reconstruction

Twenty ten-subrun groups contain 10 million generated events. There are 55,432
fast and 56,254 G4 tracker-selected events, including 38,431 selected in both.
All selected events have explicit unique highest-cycle ntuple/hit rows, including
zero-track events; the G4 ntuples also retain upstream CRV-selected events which
are excluded from this tracker comparison. The sample is saved default-L, not a
new transport run on current main.

| quantity | fast relative to G4 |
|---|---:|
| total track yield, 50,271 / 51,317 tracks | −2.038 ± 0.366% |
| selection-only additive contribution, −882 tracks | −1.719 ± 0.340 percentage points |
| both-selected content contribution, −164 tracks | −0.320 ± 0.139 percentage points |
| digis per selected event | −1.086 ± 0.326% |
| TT_Outer dx core width | +0.686 ± 0.775% |
| TT_Outer dy core width | −0.134 ± 1.080% |
| paired panel occupancy score / rank | 1.10566 / 215 |
| disjoint G4 occupancy score / rank | 1.00068 / 215 |

The additive contributions use the same fluctuating G4-track denominator and
covariance −3.55172e−8; independent quadrature is not appropriate. Selection-only
means tracks in events selected in only one arm, not a proof that one specific
transport term caused the deficit. Among both-selected events, 2,633 lose their
last track and 2,473 gain a track; multitrack changes also enter the −164 count.
The saved `by_fast_muon_straws`/`by_straw_difference` labels describe the maximum
number of **good StrawGasSteps** for one primary muon, not unique occupied straws.
Their truth-conditioned correlations are diagnostics, not response models or a
causal identification of secondary placement.

The earlier unpaired width excesses are not reproduced: entry/exit dx changes
are +1.544 ± 0.988% / −0.637 ± 0.650%; dy changes are +1.777 ± 1.186% /
−2.211 ± 1.553%. This does not isolate why the historical sample differed.
The paired file-block guards leave every numerical result unchanged for these
populated, matching twenty groups; their grouping metadata is now explicit.

| event-clustered shape | maximum marginal deviation | after 2 marginal errors | disjoint G4 diagnostic after 2 errors |
|---|---:|---:|---:|
| chi2/ndf | 13.074% | 0.000% | not computed |
| TDC (cal end) | 6.456% | 2.554% | 1.215% |
| TOT (cal end) | 1.150% | 0.230% | 0.000% |

These are finite-in-range normalized, fixed-reference-edge diagnostics with
under/overflow accounting, not simultaneous shape certificates. Zero after
error subtraction is not zero discrepancy. Yield remains outside the 1% target;
occupancy meets its configured threshold conditional on the fixed library. No
placement replacement or production-default promotion follows from this result.

Generator provenance is checked beyond exposure: all 10,056,916 saved generator
particles (including 31,915 multimuon events) match the merged and original
all-event generator tables exactly. All 505,330 particles in 499,534 G4-retained
events match the corresponding all-event subset, with full run/subrun/event/index
keys. Both dumpers read `cosmicTimeOffset`; the reference simulation emits gen
and selected dts streams from one job. This saved-table audit does not independently
reread art products or recover G4-rejected payloads. Finite-library uncertainty,
independent candidate holdout and current-main production validation are separate
follow-up work, not implied closure here.

## Drivers

The `straw_occ_*` scripts use the isolated tracker/core paths supplied by `TRK`
and `CORE`; check those before reusing them. They activate `pyenv ana` themselves.
If their parent shell already sourced Mu2e setup, export the shell function with
`export -f pyenv` before invoking a child bash: resourcing an already-initialized
setup can omit that function. Operations setup must precede Python activation.
The scripts fail on analysis errors rather than reporting a successful last
`echo`.

`straw_occ_report.py LABEL=JSON ... --out FILE` summarizes saved results and
records their hashes. It refuses to overwrite its output. A grouped path-A grid
campaign needs a separately approved fast-arm test; reuse completed reference
products and the existing code tarball rather than rebuilding or rerunning them.


`rescore_secondary_pilot.sh DATA SAVED OUT GDML [FIRST]` uses existing grouped
pilot files (default first subrun 21), audits both arms' highest-cycle output
coverage, then reruns `tv2compare` and `seccensus`. It refuses an existing output
directory and performs no fetch, simulation, reconstruction or submission. Supply
an activated Python environment and explicit tracker/core `PYTHONPATH`.

`secondary_tv3_preflight.sh SAVED NEW_AUDIT TRK_PNFS GEN_PNFS` audits the
original twenty-group default-L campaign: queue/output deduplication, exact input
manifest, staged-table hashes, retained pilot worker and equal fast/G4 staged
FCLs. It writes a nineteen-group tail manifest and a `READY` marker, but never
submits. The operator must inspect the queue and submit only after authorization;
use a separately named staged list rather than overwrite the original list/FCLs.

`secondary_tv3_collect.sh SAVED TRK_DATA TRK_PNFS NEW_RESULT` requires all twenty
fast/G4 ntuple, hit and log products before fetching, checks data-area headroom,
reuses the saved pilot, audits highest-cycle event coverage for each arm/group,
and runs the full paired `tv2compare`. A new output directory is required, and
`COMPLETE` is written only after comparison succeeds. Supply operations tools,
an activated Python environment and explicit tracker/core `PYTHONPATH`. Shell
syntax checks and live preflight/collection checks complement the isolated Python
tests; the selector does not supply unit coverage for these shell drivers.
`secondary_tv3_watch.sh AUDIT SAVED TRK_DATA TRK_PNFS NEW_RESULT CLUSTER` polls
an already submitted cluster every two minutes and invokes that collector only
when all twenty groups have ntuple/hit/log products. Set `CORE` and `CRVEXTRAP`
explicitly. It records an exit code and stops on held jobs, incomplete products
without queued jobs, or its bounded monitoring window; it never resubmits.

`secondary_tv3_provenance.py --comparison SAVED_JSON --gen-dir ORIGINAL_GEN
--merged-dir MERGED_GEN --outdir NEW [--max-groups N]` follows the saved paired
step manifest, requires exact per-subrun exposure and full generator row/payload
identity, and hashes canonical full-key-sorted generator columns. It checks
G4-kept events as a subset, including all non-muon particles, and refuses missing,
duplicate or noncontiguous generator indices. `--max-groups` is explicitly partial;
`--expected-events` defaults to 50,000 unique generator events per subrun.

`secondary_tv3_rescore.py SAVED_JSON --outdir NEW` reruns only `tv2compare` on
the exact saved input lists. It records source/result hashes, elapsed time, and
all changed JSON fields. It does not fetch or regenerate physics products.

`secondary_placement_audit.py BASE_JSON EDGE_JSON BASE_LOG EDGE_LOG --out FILE`
compares saved census bins and placement counters, rejects mismatched coverage,
and records input hashes. Its log counters precede analysis selection; it is not
a donor-to-recipient migration matrix and cannot recover covariance between the
two saved summary estimates. Use the separate ROOT identity audit for generator,
primary-step and surface-step equality.

`secondary_source_audit.py G4_TABLES ... --out FILE` reproduces the builder's
`ismcprim` event-anchor join, counts multi-anchor events, duplicated source rows,
and true-primary muon content in the resulting template population. It does not
silently choose one anchor when several exist. It audits the supplied population,
not the training library unless the exact training inputs are supplied. Use
`--library-meta ORIGINAL.meta.json --source-dir DONORS` to select exactly those
inputs. It separates all anchor events from library-eligible events with an anchor
straw crossing; their raw off-axis populations are not interchangeable.

`secondary_library_topology.py --library ORIGINAL --source-dir DONORS --gdml G
--outdir NEW` reads the sidecar's ordered donor manifest, rebuilds one file at a
time, checks row/content alignment, and appends only topology to the **retained
original** payload. Only derived p/ux/uy/uz values may differ during alignment by
at most two dtype eps relative; no such rebuilt values replace original values.
All other payloads must match exactly. The sidecar records those differences,
donor/library/geometry/source hashes and row counts. The original is never overwritten.

`secondary_replay.py BASE_FILES... --start-index I --seed N --library ORIGINAL
--annotated TOPOLOGY --gdml G --outdir NEW` reconstructs saved primaries, then
repeats off-axis → in-straw pasting with the original per-file RNG streams.
Provide explicit contiguous paths in original order; a subset must retain its
original index, not restart at zero. Every original and annotated-default check
must pass before any candidate is written. Candidate checks protect primary and
in-straw payloads (excluding event `nsgs`) and generator/subrun/surface tables.
`counts.json` records exact checks, hashes, counts, dtype differences and timings,
including failures. No transport, stock reconstruction or grid submission runs.
`secondary_replay_remaining.sh` expands a saved campaign's sorted files and
preserves the requested tail's original indices.

`secondary_migration.py --census BASE_JSON --topology-census TOPOLOGY_JSON
--library ANNOTATED --gdml G --outdir NEW` reads the saved full seed-1 campaign
manifest in original order. It records raw state transitions by donor exterior/
internal-gap/anchor-plane class, complete rejection/merging, and paired occupancy
for base, topology, plane-only, transverse-only, exterior-only and non-exterior-only
lattice variants. Full base/topology summaries must close on the saved censuses.
`--max-groups N` audits a prefix without claiming full-sample closure. No event loops
or ROOT output; `occupancy_groups.npz` stores per-file moments for reanalysis.

`secondary_migration_report.py MIGRATION_JSON --out NEW_JSON` reports paired
file-group location contrasts and their interactions. Raw migration counts are
not merged step or unique occupied-straw counts.

`secondary_occupancy_jackknife.py MIGRATION_JSON OCCUPANCY_GROUPS.npz --out NEW_JSON`
checks the archive hash and saved scores, then recomputes every paired score after
deleting each common file group. Rank/support changes, unavailable covariance or
fewer than two groups make the error unavailable. This does not estimate finite-
library uncertainty or establish an independent post-selection validation.
