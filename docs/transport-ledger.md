# Physical segment contract and shared crossing state

## Scope

`transportledger.SegmentHistory` is a passive contract for **supplied** physical states, not a correction to the
production transporter. `flighttime` supplies an explicit linear-in-path kinetic-energy integral. Neither changes
`make_steps`, the existing `ledger.Arms`, the default NumPy/JAX transport, gas deposition, or secondary placement.
The richer contract is deliberately distinct from the existing entry-indexed `make_steps.segment_ledger`.
[Passive producer observations](transport-capture.md) provide a separate diagnostic sidecar on current transport;
that sidecar does not construct a physical `SegmentHistory` or resolve the required stopping/time policy.

A support material column does not identify its ordered boundaries. Do not turn that column into a physical
segment by assigning an average material or spreading its energy loss between straw entries. A stopping decision
without an established endpoint is not a physical stop location. Production terminal/stop/common-clock failures
in `tests/test_rate_ledger_limits.py` remain explicit required follow-up gates.

## Representation

Rows are ordered by `(run, subrun, event, sp, segment)`, with contiguous segment ordinals starting at zero per
muon. Construction copies every array into immutable storage. All quantities are NumPy arrays; there is no
per-event Python loop. Position is in the manifest frame, path in mm, kinetic energy in MeV, time in ns.

- `kind`: physical path or condensed-scattering `displacement`. Physical path cannot be shorter than its chord.
- `medium`: manifest material/cut index; 0 is exact modeled vacuum, -1 is displacement, not an unknown medium.
- `entry_mm`, `exit_mm`, `path_mm`: established physical endpoints and actual traversed arc length.
- `ke_in_MeV`, `ke_out_MeV`, `applied_loss_MeV`: primary kinetic debit, checked for conservation.
- `sampled_loss_MeV`: supplied raw proposal, distinct from an applied loss capped by available energy.
- `deposited_MeV`, `escaping_MeV`: partition of that primary transfer, not all detector or shower energy.
- `dt_ns`: supplied flight time, not inferred automatically by the schema.
- `terminal`: `none`, `exhausted` (zero KE), or `model_stop` (residual KE allowed). Stops are absorbing.

Sample, deposit, escape and time each have a status: `known`, `unknown`, or `unmodeled`. Non-known quantities are
NaN, never fabricated zeroes. Debit closure does not certify partition closure. An incomplete partition has a
NaN residual in `accounting()`. Gas deposits pasted from secondary libraries are not a primary-transfer ledger;
no independent restricted gas sample is debited twice to force agreement.

Displacements have zero physical path, time and transfer, with explicit position continuity on either side.
The contract does not reconstruct scattering direction. Zero-length terminal records are supported, including
initially stopped lanes. The model-stop residual is retained, not silently deposited.

## Timing assumption

`flighttime.linear_kinetic_dt_ns(L, K0, K1)` computes

```text
dt = L (K0 + K1 + 2 m_mu) / [c (p(K0) + p(K1))]
```

with a numerically stable form. It is exact only for the declared **linear-in-path KE** evolution. Constant
momentum and a finite-time zero-energy endpoint are included; positive length starting at zero energy is an
error. Zero length contributes zero time. There is no momentum floor.

`history_dt_ns(history, energy_evolution="linear-in-path")` returns increments without changing the history or
its status flags. It neither proves the evolution law nor supplies a stopping policy or external time anchor.
Upstream world transport and the back-to-ring-top time convention still need their own consistent producer
history; do not disguise a negative backshift as positive physical flight.

## Shared state queries

`historystate.at_path(history, identities, path_mm, energy_evolution="linear-in-path")` is the explicit query
interface. It uses full four-part muon identity and supplies KE, momentum, position and relative flight time.
A partial segment uses the analytic integral to its interpolated KE, not a linear interpolation of momentum
or elapsed time. Histories must have known times consistent with the declared evolution and straight physical
segments. Unknown times or curved physical paths are rejected.

The state at a shared path coordinate is right-continuous: outgoing physical state after any zero-path
relocations. A terminal endpoint is available but not alive; outside the recorded domain, queries are unavailable
with NaN state, never extrapolated. The query does not insert generator/bunch/world offsets. Its use on supplied
histories is not a claim that the production entry, kink and surface clocks have been unified.

## Explicit surface projection

`historysurfaces.surface_steps(history, starts, energy_evolution="linear-in-path")` is a separate API; production
`surfaces.surface_steps` is unchanged. The manifest frame must be exactly `detector`. `starts` must contain
`run, subrun, event, sp, t_start_ns`, with one finite time anchor per recorded muon and no additional identities.
The SurfaceStep output schema has int32 `sp`, so unsupported keys are refused rather than truncated.

Only positive-length physical segments contribute crossings. Zero-path displacements and exact cylinder
tangencies are not physical crossings in this API. There is no generated/straw-hit population filter and no
upstream/downstream extension. Shared segment boundaries are half-open for crossing construction; a final
terminal endpoint is not a transmitted crossing. Standard finite VD-thickness endpoints are formatting around
the central crossing, not a transported step or an energy deposit in the detector surface.

## Provenance and persistence

`medium(...)` serializes the existing core `Material` and effective cut without introducing a new material/loss
model. The manifest requires full tracker/core hashes, geometry SHA256, ordered input hashes, frame, media and
settings. The embedded material/cut hash is verified. Supplied provenance is not checked against disk: a runner
must hash the actual loaded sources, including dirty files, and record effective policies and time origin.

`history.save(new_path)` uses NPZ and JSON without pickle and refuses overwrite. It is not an atomic campaign
publication protocol. `load` revalidates the schema. Batch complete muon histories; continuity across independent
files is not certified. `capture(..., enabled=False)` inspects nothing and consumes no RNG.

## Validation and remaining gates

Run relevant files in separate processes, using a pinned core containing the shared test hooks:

```bash
python3 -m mu2efastsim.testselect --run --isolate -- -q
```

Focused legacy controls remain `test_rate_ledger.py`, `test_rate_ledger_limits.py`,
`test_step_energy_ledger.py`, `test_surfsteps.py` and `test_surfsteps_vertices.py`. The strict expected failures
must not be removed because a supplied synthetic history satisfies its own contract.

Remaining producer work: ordered world/interior/exit material boundaries, physical terminal/stop nodes, common
energy evolution and clock, last-straw debit, evolving thin-material momentum, explicit deposition/escape
accounting and consistent secondary treatment. Then validate actual step/vertex/surface state, capture-on/off
outputs and subsequent RNG, low-energy range, current-code primary rate, and stock digitization/reconstruction
against existing baselines. No default promotion follows from contract tests alone.
