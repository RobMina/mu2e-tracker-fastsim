# Experimental ordered support transport

## Scope and comparison contract

This is an **opt-in support-only straight-path experiment**, not the production tracker transporter. It now
constructs a physical `SegmentHistory` from ordered intervals in the stored geometry representation. It does
not include straws, scattering, magnetic fields, upstream world transport, secondary libraries, gas response,
or stock digi/reco. No production defaults or historical outputs change.

Two explicit names are available:

| Mode | Overshoot sample (`raw >= KE`) treatment |
|---|---|
| `ordered-support-mean-replace-v1` | Replace overshoot by existing core restricted mean times proposed step |
| `ordered-support-sampled-stop-v1` | Retain raw sample; exhaustion can occur inside proposed step |

These modes differ **only** in that treatment. A comparison of them isolates this stopping policy within the
candidate. Comparing either against legacy production combines multiple changes (navigation order, adaptive
step schedule, sequential momentum updates, cutoff convention), and must not be labeled an isolated stopping
or energy-versus-momentum comparison. The separate writer energy/air-vacuum diagnostic remains unchanged.
No support-width/calibration retuning is part of this work.

## Navigation

`materialpath.tube_intervals` intersects full/sector coaxial rings and rotated noncoaxial tube tuples using
analytic annulus, z and phi-plane boundaries. It handles wrapped/reflex sectors. Work is vectorized over rays;
loops are over geometry elements and the small fixed number of phi subintervals. The earlier bounded
`support_intervals` remains available with its original full-ring restrictions.

`convexpath.convex_intervals` clips the existing `PieceSet` or `SignedPieceSet` halfspaces. Signed events are
aggregated at coincident boundaries per `(lane,material)` before cumulative occupancy is calculated. Negative
occupancy or multiplicity greater than one is rejected; summed material columns are never globally clipped
to hide invalid regions. `interiorpath.interior_intervals` combines tube and calorimeter solids and inserts
vacuum gaps. Positive-length overlaps are rejected, including same-material overlaps: no material precedence
or hidden union rule is invented. Shared faces/tangencies follow documented boundary policies. Phi inclusion
uses 32 float64 eps radians at both faces to avoid wrapping a tiny negative zero-face error into vacuum.

The result partitions each positive-length requested ray into `lane,s0_mm,s1_mm,material` rows. Zero requested
length gives no navigation rows. The representation uses core's stored rounded radii/axis tolerance, polygonal
convex approximations and selected solids. It is **not a new GDML truth navigator**. Geometry loader omissions,
Boolean approximations, overlap refusals, numerical boundary choices and cost on the actual extracted geometry
remain to be audited. Passing synthetic geometry tests does not qualify full extracted geometry.

## Energy evolution and stops

Core `mu2efastsim.segmenttransport.transport_segment` is a new experimental integration policy around the
**existing** `eloss.dedx_restricted` and `eloss.sample_loss`, not another implementation of material or loss
physics. It leaves production `slab_loss` untouched. Defaults for this explicit candidate are max step 1 mm,
mean fractional loss limit 0.05, model-stop KE 1 MeV and cap 10000 slices per material interval.

A proposed step is the minimum of remaining material extent, max step and fractional-KE/mean-loss limit.
All active lanes are sampled in vectorized calls. KE is updated in energy space after each slice, and the
next material sees that updated state. Momentum is derived with `sqrt(K)*sqrt(K+2*m_mu)`, without a transmitted
momentum floor. Nonfinite/negative model outputs, inability to advance path, unrepresentable positive energy
debit, or iteration-cap exhaustion raise; none silently becomes transmission.

For a proposal that exhausts KE, the candidate explicitly assumes a constant sampled loss rate over the
proposed step and sets used length to `step * KE / proposed_loss`. This locates the terminal point **within this
model**, not a validated G4 stochastic range law. The original raw sample for the full proposal is retained;
it is not the applied debit over the shortened path. If the endpoint crosses the model cutoff without
exhaustion, its residual KE remains in `ke_out_MeV` and the trajectory ends as `model_stop`; that residual is
not silently deposited. Initially below-cutoff lanes similarly get zero-length terminal records, including
in vacuum. Thus the candidate cannot be used to infer physical low-energy vacuum stopping.

`orderedtransport.transport_intervals` processes one interval per live muon at a time, grouping by material
within each traversal round. Histories are subsequently sorted by full identity, preserving each muon's
physical chronology, not sampler batch order. Grouping/active masks may change RNG consumption between modes;
equal seeds do not certify draw-by-draw stochastic pairing after survival diverges. Capture/timing choices
within a given mode do not change the energy samples.

Only primary KE debit is closed. Positive-material deposited and escaping terms stay explicitly `unmodeled`
with NaN values. No delta-ray escape or local deposition is set to zero to force closure. Support uses an
explicit 1e9 MeV effective cut labeled unrestricted. No straw-gas debit is added, and no library deposition is
charged to the primary.

## APIs and shared-crossing state

```python
result = interior.transport_support_history(
    rng, identities, origins, unit_directions, kinetic_MeV, lengths_mm, metadata,
    mode="ordered-support-mean-replace-v1", timing="linear-in-path")
history = result["history"]
```

`TrackerInterior.transport_support_history` is a separate method, not a switch hidden inside production
`transport` or `make_steps`. It returns history plus final KE, position, path and status aligned with the input
identities. The manifest must describe every used core material/cut exactly. Effective numerical policies and
source hashes are persisted. Empty batches and multiple muons in the same event are supported.

Timing defaults to `unknown`. Explicit `timing="linear-in-path"` invokes the analytic flight-time primitive
on each physical slice and persists that assumption. `historystate.at_path(...,
energy_evolution="linear-in-path")` supplies one energy/position/relative-time convention at arbitrary path
crossings. It accepts only physical straight-segment histories with consistent known times. Shared boundaries
select the outgoing segment; a terminal endpoint is available but not alive. Missing identities, negative
paths and queries beyond a stop/end are unavailable/NaN, never extrapolated. Displacement and curved segments
are rejected rather than silently interpolated. No generated/bunch/world offset is added.

This shared query helper is preparation for eventual step/vertex/surface integration; the production writer
and surfaces do not yet use it. The back-to-ring-top/upstream time convention still needs a complete history.

## Bounded runner and provenance

`python3 -m trkfastsim.analyses.supporthistory` reads an explicit seed CSV with exactly:

`run,subrun,event,sp,x,y,z,px,py,pz,length_mm`

Positions/directions are in the Mu2e world frame; momentum components are MeV/c. Seed momenta must be nonzero;
already-stopped lanes can be supplied via the Python API in KE space. It requires a GDML file, full tracker/core
commit hashes and a uniquely named output directory. Default row bound is 1000; excess rows are rejected, not
silently truncated. No G4 generation is invoked. The runner refuses reported skipped calorimeter solids and
navigation overlaps. It writes `history.npz`, requested intervals, final states and a completion summary only
after successful transport. It hashes the seed input, GDML and all loaded tracker/core Python sources.

```bash
python3 -m trkfastsim.analyses.supporthistory SEEDS.csv --gdml EXISTING.gdml \
  --outdir NEW_OUTPUT --tracker-ref FULL_SHA --core-ref FULL_SHA \
  --mode ordered-support-mean-replace-v1 --timing linear-in-path --seed 1
```

This counts supplied seed rows/events, **not a generated-event acceptance denominator**. Seed identity must
remain `(run,subrun,event,sp)`; a future G4 comparison must preserve multiple primaries and verify source tree
cycles/entries independently. The runner has only been tested on synthetic inputs with geometry fixture
injection. No real-data/G4 run, full-scale memory/performance measurement or production acceptance is claimed.

## Validation and next gates

Final local isolated verification: **330 passed, 5 strict legacy xfails** in the non-JAX tracker suite and
**108 passed** in core segmenttransport/transport/slab-trace/world-energy/eloss tests. The same published-source
tracker regression remains enabled. Read-only review led to regression-tested refusal of sub-ULP positive
energy debits and stable phi-face inclusion. Local/cluster source hashes match and tracked diffs pass
`git diff --check`. No remote compute test or actual GDML/reference run was performed.

Synthetic tests cover chronological mixed-material updates, absorbing stops, partial-step exhaustion, residual
cutoff energy, vacuum identity, empty/zero/multi-muon batches, repeatability, closure, timing independence,
shared boundary queries, sectors/noncoaxial tubes, signed solids and invalid overlaps. The non-JAX tracker suite
and focused core suite commands remain in `transport-ledger.md`; append the new core
`tests/test_segmenttransport.py` when running core checks.

The next session executed a bounded actual-v04 navigation audit, documented in
[navigation-audit.md](navigation-audit.md); the earlier counts above remain the pre-audit verification record.
Strict navigation refused tiny boundary overlaps/negative occupancy, and the loader census found unresolved
unsupported-solid candidates. No physical transport history was admitted. Next explicitly resolve these
unsupported/overlapping geometry cases instead of silently dropping them. Then add ordered wall/gas/wire intervals
with a consistent restricted-loss/secondary convention; couple energy updates to subsequent scattering and
crossing recomputation, including upstream and exit state. Keep the isolated energy/air/stopping/timing arms
and provenance separate. Only then compare same identities to existing G4 references below 500 MeV/c and by
range-out margin, followed by library-on stock TV2/TV3. No default promotion is authorized by these tests.

## Deterministic mean option

A third explicitly named mode, `ordered-support-deterministic-mean-v1`, invokes the experimental core
segment integrator with `loss_model="mean"`. It uses the existing restricted mean times the step and makes
no sampler/RNG calls. The two original sampled modes retain their meaning and core defaults stay sampled.
Metadata labels `sampled_loss_MeV` as a deterministic full-step proposal in the new mode. It supports the
paired geometry sensitivity in [paired-support-energy.md](paired-support-energy.md); it is not the expectation
of stochastic stopping or a separate loss model implementation.
