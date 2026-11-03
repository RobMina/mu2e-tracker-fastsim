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

## Off-axis anchor studies (parked)

Two alternative anchors for off-axis library steps were studied and are on branch `park/offaxis-anchors-20261103`,
with `buildlib --topology` (six donor-topology columns), the shared-draw migration and replay tools and their tests:
the edge anchor (steps with a plane offset anchored on the recipient's first or last plane) and the donor-topology
anchor (the donor anchor's fractional plane position mapped into the recipient's plane extent). On the saved
full sample, in the 38,431 events selected in both arms, the topology anchor moved same-plane-other from
+29.82 +- 2.80 % to +2.15 +- 2.35 %, but the 1-2-plane deficit stayed at -13.28 +- 2.00 %, the nearby-panel deficit
at -16.92 +- 1.79 %, and the secondary-only paired occupancy chi2/rank got worse (1.4910 to 1.6755, rank 215).
Neither anchor was promoted. The placement options below replace them.

## Opt-in placement and in-straw options (paired per-shard)

Four `steps` options change how library secondaries are pasted. Each is off by default; with all of them off the step
tables are identical to the default, table for table. None has a fitted value.

| option | what it changes | motivation (measured upstream of the digis) |
|---|---|---|
| `--library-drop-muons` | leaves out drawn rows with \|pdg\| = 13 (other primary muons of the donor event) | species flag: non-primary muons the fast transport already carries |
| `--offaxis-plane-sign` | mirrors a drawn set's plane offsets when donor and recipient muons travel opposite ways along the planes (sign of u_z) | same-plane excess, 1-2 and >= 3 plane deficits; the self-paste closure (each Geant4 event's own set pasted back) reproduces every place class |
| `--offaxis-shift-from-muon` | an off-axis step landing in a fast muon straw moves up to 3 straws along its straw offset instead of being dropped | about 2.7 % of drawn off-axis steps were dropped; a donor's off-axis step never shares a straw with its muon |
| `--instraw-fine-length` | the in-straw draw key splits the muon path in the straw beyond 8 mm (10, 12, 15, 20, 30 mm) | muon-only straws: path-length population flagged (chords longer than 20 mm +25 %, 8-11 mm -5 to -8 %) while eIon per mm agrees; the 8 mm-and-above key gave long chords along the wire the secondary rate of 8-10 mm crossings |
| `--offaxis-width-match` (branch `park/offaxis-width-match`) | keys off-axis donors on the width of the muon's plane extent | tested and rejected: every placement histogram is worse |

Method: arms are fast steps (seed 1) of the `g4ref_v2` generation (10 M generated, 200 subruns); the statistic is the
paired per-shard one in shape mode ([shard-statistics.md](shard-statistics.md)), against Geant4 steps of the same
events. The options were chosen on subruns 21-120; subruns 121-220 were looked at once for the chosen set; the
row "all" uses every subrun. The T-V2 proxy is the fast digitiser on each arm's steps against the fast digitiser on
the Geant4 steps (the fast digitiser on Geant4 steps against stock: 0 of 16 histograms flagged on all 200 subruns).
The placement and shift options were developed while subruns 121-220 had been seen for an earlier placement
candidate, and the path-length cause was found on the default arm over all 200 subruns, so the 121-220 look is not
blind for them.

Steps, chi2/ndf with the family-wise limit in brackets (27 histograms):

| | default | all four options |
|---|---:|---:|
| flags, subruns 21-120 / 121-220 / all | 5 / 5 / 11 | 4 / 4 / 6 |
| secondary place (all) | 112 (6.9) | 55 (6.5) |
| same plane / 1-2 planes / >= 3 planes (all) | +25 / -27 / -29 % | +18 / -18 / -21 % (each +- 2-3 %) |
| secondary species (all) | 255 (6.1) | not flagged |
| eIon per straw (all) | 1.83 (1.82) | 1.54 (1.83) |
| eIon per muon-only straw (all) | 1.73 (1.68) | 1.20 (1.69) |
| muon path per muon-only straw (all) | 2.21 (1.80) | 1.31 (1.81) |
| secondary-only straws (all) | -4.1 +- 1.6 % | -1.5 +- 1.7 % |
| other steps per event (all) | -3.2 +- 1.1 % | -2.3 +- 1.1 % |

Fast digis on these steps against fast digis on the Geant4 steps (17 histograms):

| | default | all four options |
|---|---:|---:|
| flags, subruns 21-120 / 121-220 / all | 1 / 0 / 7 | 0 / 0 / 4 |
| TOT0 / TOT1 (all) | 4.45 / 4.52 (4.1) | 2.84 / 2.85 (4.4) |
| PMP (all) | 2.33 (2.05) | 1.61 (2.07) |
| hit energy (all) | 2.40 (2.13) | 1.52 (2.18) |
| dt (all) | 1.90 (1.70) | 1.66 (1.71) |
| digis per selected event (all) | -0.70 +- 0.44 % | -0.48 +- 0.43 % |

Still flagged with all four options: secondary placement (about half the default's excess left), plane distance,
other steps per event, secondary path, event eIon; through the digitiser, straw occupancy, wire distance and the
digis-per-event shape. Muon-only straws with chords below 1.25 mm are 3-4 % low with or without the options (not a
library effect). Steps wall time with all four options against the default on the same node: -0.8 % and +1.2 % in two
measurements, each with about 3 % spread between repetitions, i.e. no measurable change (about 10,000 generated
events per second per core for the steps).

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

`secondary_source_audit.py G4_TABLES ... --out FILE` reproduces the builder's
`ismcprim` event-anchor join, counts multi-anchor events, duplicated source rows,
and true-primary muon content in the resulting template population. It does not
silently choose one anchor when several exist. It audits the supplied population,
not the training library unless the exact training inputs are supplied. Use
`--library-meta ORIGINAL.meta.json --source-dir DONORS` to select exactly those
inputs. It separates all anchor events from library-eligible events with an anchor
straw crossing; their raw off-axis populations are not interchangeable.

The anchor-study tools (`secondary_placement_audit`, `secondary_library_topology`, `secondary_replay`,
`secondary_migration` and its report) are on branch `park/offaxis-anchors-20261103`.

`secondary_occupancy_jackknife.py MIGRATION_JSON OCCUPANCY_GROUPS.npz --out NEW_JSON`
checks the archive hash and saved scores, then recomputes every paired score after
deleting each common file group. Rank/support changes, unavailable covariance or
fewer than two groups make the error unavailable. This does not estimate finite-
library uncertainty or establish an independent post-selection validation.
