# Passive producer observations

## Explicit scope

`make_steps(..., capture_transport=True)` creates `make_steps.transport_capture`, a fresh
`TransportCapture` for one call. It supports NumPy stage C with interior scattering, primary-only, with the
legacy entry writer (the existing vacuum-writer and integrated/thin arms are allowed). It rejects unsupported
backends, secondary libraries, straight-only transport and entry-ledger diagnostics before transport/RNG use.
Default is `False`; the existing transport state, stopping thresholds, random call order, gas sampling, writers
and surface construction are not altered. The legacy `history=True` tables remain separate.

This is **algorithm-state evidence, not a physical `SegmentHistory`**. Aggregate support material columns do
not locate ordered material boundaries; a failed continuation does not determine a physical stopping point.
There is no material-time interpolation, terminal repair, downstream truncation or rate correction here.

## Observations and dispositions

Each phase has full `(run, subrun, event, sp)` identities and row-aligned copied batches. Input identity
columns are validated/cast independently so mixed signed/unsigned and nullable integer dtypes do not round
large event numbers through a floating-point array. Phase identity must be unique.

- **`initial`**: supplied algorithm state. In the exit phase this can be the stale interior restart.
- **`entry`**: a crossing actually committed to the output; position is its gas entry, not a terminal straw exit.
- **`kink_end`, `restart`**: incoming end and corrected outgoing restart at the same recorded path coordinate.
  The transverse displacement/gas-backout has no added path. No artificial vertex ordering epsilon is added.
- **`stop_attempt` / `piece_attempt`**: lower and attempted *local* path coordinates, attempted loss and raw
  model KE. Attempted position is not a physical stop; missing material ordering prevents that inference.
- **`legacy_restart`**: exact final computational arrays, separately retained even when behind emitted entries.

`summary()` returns the last observation per input lane and one disposition:

| Phase | Disposition | Meaning |
|---|---|---|
| Interior | `no_candidates` | No remaining accepted candidate in this iteration; no new endpoint established |
| Interior | `final_non_kink` | Final retained entries emitted, no further kink; last observation is the final entry |
| Both | `unresolved_stop` | Continuation rejected at the energy threshold; stop location/time unresolved |
| Both | `iteration_cap` | Iteration limit reached; not physical completion |
| Exit | `not_processed` | Outside the algorithm's momentum/extent admission |
| Exit | `no_support` | No support exceeding the algorithm's threshold; no vacuum endpoint advanced |
| Exit | `extent_complete` | Surviving kink reached the remaining-extent criterion |

All lanes are classified, including empty input, lanes disappearing while other lanes continue, and cap-zero
interior calls. The last observation is selected by producer order, not the maximum path or a time match.
The legacy floored momentum is retained separately from raw model KE; one is not evidence for the other.

`observe()` adds `phase_path_offset_mm` to its path; attempted paths and `legacy_restart.path` remain explicitly
local. A caller may add the stored offset when comparing them. Positions are in the producer's Mu2e frame.
No time clock is certified by this capture. Kink backoff can assign loss through `s_k` to a position at
`s_k - back`; the observation does not conceal or reinterpret that convention.

## Exit remains a separate phase

Production exit propagation receives `interior.last_final`, not the last emitted straw entry. It may receive
lanes already marked stopped by the interior. The capture keeps the interior disposition and legacy stopped
flag alongside that actual input, and adds the unchanged interior restart path as the exit phase offset.
Nothing masks those lanes or moves their origin. Joining these phases into one physical trajectory would
falsely imply continuity/complete terminal accounting, so this API does not do so.

## Storage and provenance

Only a completed capture can be saved with `capture.save(new_path)`. The NPZ uses numeric/string arrays and
an embedded JSON manifest, no pickle, and refuses overwrite. It is not a transactional campaign publisher.
Records and summary accessors return copies, so editing them cannot alter captured arrays.

The manifest records loaded primary-transport, identity/crossing and surface dependencies by SHA256, plus
Python/NumPy/Pandas versions and platform. Geometry/input hashes are the runner's responsibility. The bounded
runner hashes its source, baseline source, actual generator/GDML and all listed dependencies before/after;
no completion report is written if their contents change. A failed run may leave partial evidence and must
not be resumed as complete.

## Identity check and reproduction

```bash
python3 scripts/transport_capture_check.py \
  --gen GENERATOR_TABLE --gdml EXTRACTED_V04 \
  --base-source FROZEN_MAIN_STEPS_PY --max-events 50000 \
  --arm legacy --outdir NEW_DIRECTORY
```

The first N unique full event identities are selected without a passing-event filter. Runs are baseline,
current capture-off, current capture-on, using the same supplied core and seeds. Exact comparisons include
steps, particles, vertices, surfaces, legacy entry/muon ledgers, transport statistics, RNG state, child-seed
count and subsequent draws. `--arm AMT` checks passive capture in that existing opt-in arm; it does not test
AMT equality to the default or qualify its physical loss/time convention.

On one 50k-event prefix of an existing extracted-v04 generator table, both legacy and AMT pass this check:
36,875 primary muons, 2,626 interior lanes, 6,510 steps. In each arm 396 lanes have an observed entry ahead of
the legacy restart (maximum 1927.70 mm), one interior unresolved stop and four exit unresolved stops. These
are finite-sample bookkeeping counts, not rate estimates or a proof that stopping positions are known.
The three strict production terminal/stop/clock failures remain. No default promotion or stock digi/reco
closure follows. Run selected tests in isolated processes, including the capture and legacy-limit files.

## Saved-only audit

```bash
python3 scripts/transport_capture_audit.py CAPTURE_RUN_DIR --out NEW_JSON
```

This checks the completion/capture/source hashes and full-identity joins before comparing last observations
with computational restarts and joining phase dispositions. Input hashes are checked again before output.
On the 50k prefix the AMT arm has 11.615557 MeV of already-charged thin loss absent from restart bookkeeping
summed over lanes, maximum 0.096897 MeV per lane; the support-loss gap is zero. These are bookkeeping sums
through recorded entries, not total physical loss or missing deposited energy. Legacy has zero thin debit
by convention, so that same diagnostic is zero.

The one interior stop attempt lies between producer coordinates 147.938154 and 221.832856 mm; the stored
raw model KE at the attempted end is zero. That lane enters exit propagation from the legacy restart and
is rejected again; there is no surviving exit kink for it. One of the four exit failed attempts has negative
raw model KE. This does not locate a physical stop or show which surfaces the legacy extension crosses.
The passive capture keeps that uncertainty explicit and does not reinterpret attempted coordinates as a
terminal trajectory. Subsequent remediation requires an agreed separate transport model and its own gates.
