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
exposure, not a finished ntuple event count.

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

These occupancy covariances assume **independent arms**. Paired `tv2compare`
occupancy results are labelled diagnostic and cannot certify the threshold until
cross-arm covariance is included. The same caveat applies to paired step
occupancy in `seccensus`. A disjoint reference split supplies an independent-arm
noise floor. Reusing a training library can add further dependencies not captured
by within-event covariance.

The paired yield decomposition is additive: tracks in events selected only in
one arm, plus the track difference in events selected in both arms, divided by
the total reference yield. Influence residuals include reference-denominator
fluctuations and the covariance between the two contributions. This differs from
an unpaired selected-rate times conditional-reconstruction factorization.

Census breakdown errors use delete-one-file-group jackknife replicates. The
per-straw energy band retains the historical independent-entry error convention
and is labelled a legacy diagnostic; its G4 split provides an empirical floor.
It is not an event-clustered or paired significance test. Each momentum bin is
an **event-level truth diagnostic** from its earliest primary-muon straw step,
not a measured-variable response model or a per-muon ancestry assignment.

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
profile while degrading another.

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
