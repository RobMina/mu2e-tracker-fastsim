# Passive material-segment history (schema v1)

## Scope and ownership

**Current continuation:** [experimental ordered support transport](ordered-support-transport.md) adds sector/noncoaxial
and signed-convex navigation, a separate core integration policy, actual support-only SegmentHistory production,
and shared-crossing state queries. The earlier bounded-helper descriptions below are dated development slices,
not the current extent of helper support. None is a production promotion.

`trkfastsim.transportledger` is a prerequisite data contract, not a transport correction. Production transport
does not construct a physical `SegmentHistory`. An opt-in `LegacyTrace` now captures algorithm operations
through `steps` and core `slab_loss`, as detailed below. Defaults, crossings, random draws, gas deposition,
calibration and historical products are unchanged. Work remains confined to the isolated tracker/core trees.

The first implementation captures **supplied** states and samples; it does not discover physical material
boundaries, interpolate stopping positions, sample energy loss or integrate flight time. A successful schema
check is not evidence that the supplied geometry, energy evolution or time is physically correct. Full Task 1
production history capture still requires coordinated producer hooks; energy and timing integration are later
gates. Do not claim the existing lost-muon rate deficit has been explained or corrected by this work.

## Representation

`SegmentHistory(columns, metadata)` owns a copy of each flat NumPy column. Rows must already be sorted
lexicographically by `(run, subrun, event, sp, segment)`. Identity is never event alone. Segment ordinals start
at zero and increase by one per muon, including displacement records. Do not sort the sampler's work arrays
to satisfy this representation: reorder only already-captured copies. Validation uses array operations, not
per-event/per-muon loops.

| Columns | Meaning |
|---|---|
| `run, subrun, event, sp, segment` | Nonnegative integer identity; full muon identity plus ordered segment |
| `kind` | `physical` or `displacement` |
| `medium` | Index in manifest media; 0 = explicitly modeled exact vacuum; -1 = displacement only |
| `path_mm` | Actual physical arc length, not necessarily endpoint chord or requested slab thickness |
| `entry_mm, exit_mm` | `(N,3)` positions in the manifest coordinate frame |
| `ke_in_MeV, ke_out_MeV` | Actual primary kinetic states, including residual kinetic energy at a model stop |
| `sampled_loss_MeV, sample_status` | Raw supplied sample/aggregate, before any cap or replacement; may be unknown |
| `applied_loss_MeV` | Actual primary kinetic debit, not automatically deposited energy |
| `deposited_MeV, deposit_status` | Local deposition attributable to this segment's primary transfer |
| `escaping_MeV, escape_status` | Energy transferred out of the segment accounting volume |
| `dt_ns, dt_status` | Supplied physical flight-time increment; no integration is performed here |
| `terminal` | `none`, `exhausted` (exact zero KE), or `model_stop` (may retain residual KE) |

Status values are `known`, `unknown` (not recorded/established), and `unmodeled` (not modeled by the producer).
Known values are finite and nonnegative. Other values must be NaN; zero is not a substitute for unknown.
A complete empty history has every field present, including position arrays of shape `(0,3)`.

### Physical segments and condensed-scattering jumps

A physical segment has one material/cut identity. Mixed sequential materials require separate ordered rows.
A per-material aggregate column, without ordering and boundary locations, cannot be turned into a physical
history. Curved paths may exceed their endpoint chord; paths shorter than their chord are rejected.

The transverse relocation of a condensed-scattering kink is a separate `displacement` record, with medium
-1, physical length and time exactly zero, and known zero energy transfers. Its entry and exit positions may
differ. Neighboring records must match in position and kinetic energy. This is an explicit approximation,
not a claim of extra physical flight along the jump. Direction and scattering-angle information remain in
the transporter's own state; this ledger does not reconstruct them.

Zero-length physical records are allowed for boundary/stopped-state bookkeeping, with coincident endpoints,
zero time and known zero transfers. Modeled vacuum similarly has known zero transfers; its nonzero physical
length still needs a positive time or an explicit unknown time. Vacuum here is the model convention, not an
automatic alias for every GDML material whose name contains "Vacuum".

### Stops and range-out

A stop is absorbing within a history: no later row of the same muon is accepted. A zero-energy input cannot
traverse a nonzero physical path; an already stopped lane can be represented by one zero-length terminal row.
There is no momentum floor or inferred transmission. A sampled loss may exceed available KE; the actual debit
and remaining energy must be supplied separately. The ledger neither chooses stochastic stopping nor replaces
a tail sample with its mean. A model cutoff may leave residual KE in `ke_out_MeV`; it is not silently deposited.

The producer must supply the actual traversed length, stop position and terminal time. The current core
`slab_loss` returns only a total debit, uses a low-energy cutoff, and substitutes mean loss for some overshoots.
Its return alone does not determine those endpoints or the raw rejected samples. Do not report a complete
physical history by charging its result at the far side of a slab that the muon did not traverse.

## Energy accounting domain

The explicit domain is **primary kinetic energy transferred within each physical segment**. Rest mass, capture,
decay, unrelated shower energy entering the segment, and detector-wide deposition are outside it. Two checks
must not be conflated:

1. Primary debit: `KE_in - KE_out - applied_loss = 0` (always required).
2. Transfer partition: `applied_loss - deposited - escaping = 0` (required only when both terms are known).

`accounting()` returns separate residuals and a `partition_complete` mask. An incomplete partition gives a
NaN residual, never a false zero. When escaping energy subsequently deposits elsewhere, it belongs to a linked
secondary history; adding that deposition again to this primary debit would double-count it.

The existing gas sampler uses `core.eloss.sample_loss` with a restricted delta-electron threshold. Core support
slabs use a very high effective cut. Persist both the numerical cut and `restricted`/`unrestricted` convention.
A restricted primary debit is not proof that all delta-ray energy was modeled. The secondary library pastes
G4 secondary depositions; those values are not an independently established primary energy-transfer ledger.
Until an explicit linked convention is validated, mark the relevant deposited/escaping terms unknown or
unmodeled. Do not set delta escape to zero just to make a partition close, or debit sampled gas eIon twice.

## Provenance, persistence and random ordering

`medium(core_material, cut_MeV=..., loss_basis=...)` serializes the existing core `Material` dataclass, including
composition, density, ionization/fluctuation inputs and cut tables; it defines no new material or loss law.
`medium()` denotes exact modeled vacuum. `provenance(...)` requires full tracker/core commit hashes, geometry
SHA256, ordered input names/SHA256s, coordinate frame, settings and media. Units and accounting domain are fixed
by the schema. The material/cut descriptions have an embedded SHA256 checked on load. These are supplied
identities, not automatic verification against disk: the runner must hash actual inputs and record source-file
hashes for dirty code in settings, along with effective stop policy, time origin, seed, library identity and
transport arm. Synthetic fixtures are labeled as such and do not have real geometry/input provenance.

`history.save(new_path)` writes an NPZ with numeric/string arrays and embedded JSON manifest, without pickle,
and refuses to overwrite an existing path. `SegmentHistory.load(path)` revalidates everything. It is a small
batch diagnostic format, not a production ROOT schema or a transactional campaign artifact store.

`capture(columns, metadata, enabled=False)` returns None without inspecting its arguments. Enabled capture
copies already-produced arrays and never accesses an RNG. Tests draw from the real core loss sampler, capture
the resulting states, and compare both the samples and subsequent RNG stream with an uncaptured control.
No new sampler is introduced. Future hooks must not change sampler call order, grouping, active-lane masks or
random shapes; production capture-off/on bitwise tests remain mandatory when those hooks exist.

Memory is proportional to stored rows, with copy and sorting-validation overhead. Use bounded batches of
**complete muon histories**, assigning each muon to one batch; v1 does not validate identity continuity across
separate files. Do not split a muon into batches and reset its segment counter. Empty batches are supported.
Memory/performance at production scale have not been benchmarked.

## Verification and next integration gates

Development is isolated in `analysis/trk-segment-history-20261025`, branch
`tracker-segment-history-20261025`, based on published tracker
`56e66bfa44a3e5d58fa647ac2e38955c353d26ba`. The separate clean core worktree
`analysis/core-segment-history-20261025` pins `dcde9f0597072fae13c8130b27efafc3bbf27f84`.
Both were the submodule pins of published umbrella `e41827197b8b01878d478c2057f0bceff21a272c`
when work began. Shared worktrees, indexes and other threads' branches are not changed.

From the laptop's `~/mu2e/remote/analysis`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  trk-segment-history-20261025/tests/test_transportledger.py \
  trk-segment-history-20261025/tests/test_entryledger.py \
  trk-segment-history-20261025/tests/test_step_energy_ledger.py \
  trk-segment-history-20261025/tests/test_slab_momentum.py \
  trk-segment-history-20261025/tests/test_surfsteps_vertices.py \
  trk-segment-history-20261025/tests/test_surfsteps.py
```

Final focused run: **71 passed, 5 strict legacy xfails**, including **43 new contract tests**. Read-only review
prompted guards on partial-partition energy bounds, immutable array backing and complete core material
descriptions; regression tests cover all three. The five existing expected failures remain unresolved. Tests cover multi-muon identity, sequential vacuum/Al/displacement/Cu history, near-stop,
exhausted and model-stop lanes, unknown partition/time, empty arrays, persistence, provenance and invalid-state
rejection. These are synthetic contract tests, not transport, G4, TV2 or TV3 validation.

Thread E was paused by the user, authorizing producer-hook work in the isolated `steps.py`. The passive
operation hooks described below are implemented; actual material-boundary history through world/interior/exit,
including resolved stopping locations, is still a separate gate. Then implement individually named opt-in energy
arms and integrated time against that history, reusing core loss primitives. Compare existing same-input G4
references below 500 MeV/c and by range-out margin. Keep library-off primary closure separate from library-on
stock validation. No default promotion until consistent step/vertex/surface state and final TV2/TV3 gates.
No reference regeneration, build, grid submission or production run is needed for this contract slice.

## Follow-up: passive producer instrumentation and bounded helpers

With thread E paused and user permission to proceed, isolated tracker `steps.py` and isolated core
`transport.py:slab_loss` now have optional capture hooks. Their default arithmetic, sampling call order and
outputs are unchanged. The new core callback is a dependency of tracker capture; old core remains usable
when capture is off. No changes were made to shared checkouts or published refs.

### Legacy operation trace

```python
from trkfastsim.legacytrace import LegacyTrace
trace = LegacyTrace(metadata)
sgsteps, particles = make_steps(gen, geom, rng, transport_trace=trace, **existing_settings)
trace.save(new_npz_path)
```

Use a fresh collector per call. Primary-only capture is required; secondary libraries are rejected before
transport or RNG use. The JSON/NPZ stores full `(run,subrun,event,sp)` identities, material descriptions,
effective settings and hashes of the loaded tracker steps/core transport source. Caller provenance still
must identify actual inputs, GDML and other dirty sources. The collector holds copied batches in memory;
use bounded inputs rather than a full campaign. Persistence refuses overwrite. This is a distinct schema,
`trkfastsim.legacy-operation-trace.v1`, **not** `SegmentHistory`.

Recorded operations:

- Stage-C handoff distinguishes footprint-rejected generated states, live plane states and upstream-stopped
  plane proxies. The latter are not the physical upstream stopping position. Ring-top origin/path-offset
  adjustment is recorded separately, without pretending a negative offset is physical flight.
- `loss_to_straws`, interior kink groups and exit groups record material column queries and sampler calls.
  Core events are `init`, `slice`, `mean_remainder`, `final`; lane indices are local to the matching `slab_begin`
  call, whose identity array maps them back to muons. Raw samples, mean replacements, effective accumulator
  increments, accounted/untraversed material-column lengths, cutoff/clamp adjustments and returned losses
  are separate fields. Material-column distance is not a physical coordinate or resolved range-out length.
- Interior candidate states are captured before the crossing filter, including first-stop flags; straight
  mode captures the actual writer kinetic state after the air term and its legacy stop/filter decision.
  Kinks retain both endpoint and displaced restart. Terminal records distinguish model stop, no further
  candidates/material, extent completion, ineligibility and iteration-cap truncation as applicable.
- Thin wall/gas/wire/gap columns record the current absence of transport loss. Writer gas samples retain
  restricted-cut provenance with unmodeled transport-debit/escaping terms. They are not charged a second time.

The legacy calculation can sample multiple material columns at the same group-entry momentum, then sum them;
it may sample later candidate rows that will never be traversed. The trace intentionally retains these
operations instead of sorting them into fictitious physical chronology. Likewise the existing stopped-lane
momentum floor remains observable, not repaired by tracing. World internal slices are not yet hooked; only
its handoff is captured. Capture does not solve the energy or timing defects.

### Exact full-ring navigation, bounded scope

`materialpath.support_intervals(shells, origins, unit_directions, lengths)` returns ordered lane/path/material
intervals, including vacuum gaps. It intersects the stored full coaxial ring annuli and z bounds analytically;
aggregation over its intervals agrees with the existing column routine for supported test geometries. Ray
operations, sorting and gap filling are vectorized; loops are over geometry elements, not muons.

It **rejects** phi sectors, noncoaxial tubes, calorimeter pieces and overlapping supports, even same-material
overlaps. It uses the radii/axis representation already stored by `CoaxialShells`; it is not an independent
GDML oracle. No straw-wall/gas/wire navigator or production wiring exists yet. Full extracted geometry is
not supported by this bounded helper, so it must not silently replace `TrackerInterior._support`.

### Flight-time primitive

`flighttime.linear_kinetic_dt_ns(L, K0, K1)` integrates exactly for explicitly supplied linear-in-path KE:

`dt = L * (E0 + E1) / (c * (p0 + p1))`, with `E=K+m_mu`.

The implementation uses stable half-sums/separate roots, handles constant momentum and a zero-energy exit,
and returns zero for zero-length stopped/displacement rows. A stopped lane cannot traverse positive path.
`history_dt_ns(history, energy_evolution="linear-in-path")` requires the explicit assumption and returns
increments without overwriting the history. Endpoint energies do not establish that this assumption is
physical. No existing generated-beta time, global offset or surface time is changed.

### Follow-up tests and remaining gate

Local isolated non-JAX tracker suite: **244 passed, 5 strict legacy xfails**, with published-source regression
enabled. It includes 43 segment-contract, 52 timing, 22 material-navigation and 16 legacy-producer tests.
The two published-source tests compare current trace-on tables/vertices and subsequent RNG draws against
tracker `56e66bf` loaded directly from git; ordinary tests compare current capture-off/on. Both arms use the
same isolated core; core off/on regression is tested separately. Core trace/transport/world-energy suite:
**37 passed**. Synthetic tests exercise overshoot replacement, residual cutoff closure, iteration caps,
empty/stopped lanes, two muons in one event and mixed support materials. No G4 validation has run.

```bash
cd ~/mu2e/remote/analysis
PYTHONDONTWRITEBYTECODE=1 \
TRACKER_BASELINE_GIT_DIR=$PWD/mu2e-fastsim/.git/modules/tracker \
PYTHONPATH=core-segment-history-20261025:trk-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider trk-segment-history-20261025/tests \
  --ignore=trk-segment-history-20261025/tests/test_kalman_jax.py \
  --ignore=trk-segment-history-20261025/tests/test_reco_jax.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core-segment-history-20261025 \
~/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  core-segment-history-20261025/tests/test_slab_trace.py \
  core-segment-history-20261025/tests/test_transport.py \
  core-segment-history-20261025/tests/test_world_energy.py
```

Next is supported physical navigation and an explicit stopping/energy-evolution policy, then connecting
actual ordered segments to the separately named candidate energy/time arms. Do not turn aggregate legacy
column traces into a physical ledger by inventing ordering, distributing losses uniformly without labeling
the assumption, or placing every stopped muon at a requested slab exit. Production promotion and TV2/TV3
remain gated; no reference outputs need regeneration.
