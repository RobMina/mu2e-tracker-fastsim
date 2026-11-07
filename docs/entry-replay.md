# Tracker-entry replay for a bounded benchmark

`TrackerEntryParticlesFromTable` restarts muons from saved tracker-entry states. It creates
`GenParticleCollection` in table-index order on the original RootInput event stream, including empty events.
It does not transport particles, generate CRY, or replace the StrawGasStep injector.

**C++ status: built; full-source replay and a configured Geant4 pilot verified.** Replay preserves all 52,037
source events, including 8,885 empty and four multi-entry events, and all 43,156 entry states exactly. The synthetic
inconsistent-count fixture is rejected at construction. The 20-source-event pilot validates integration, not physics
closure or throughput. The satellite SConscript supplies the dependencies; no new dictionary is required.

## Input format

Both objects are TTrees (not RNTuples); all branches are scalar. Empty trees still carry their full schema.

| Tree | Columns | ROOT / NumPy types |
|---|---|---|
| `entry/events` | `run, subrun, event, n_particles` | `UInt_t` / `uint32` |
| `entry/particles` | `run, subrun, event, index` | `UInt_t` / `uint32` |
| `entry/particles` | `source_sp` | `Long64_t` / `int64` |
| `entry/particles` | `pdg` | `Int_t` / `int32` |
| `entry/particles` | `x, y, z, px, py, pz, t` | `Double_t` / `float64` |

- Keys are original run/subrun/event, not row numbers; run/event must be positive and reserved uint32 maxima
  are rejected. Every source event has one census row, including zero-particle events.
- `source_sp` preserves the **signed key as persisted by the historical dumper**, which stored S2 keys in int32.
  It is an opaque identity, never converted through double or treated as a new native SimParticle key.
  The exact `(event, source_sp) -> index` mapping stays in the input table; no association product is invented.
- Each event's particle indices cover `0..n_particles-1` exactly. Duplicate events, indices or within-event
  source keys, undeclared particle events and count mismatches throw. Input row order is irrelevant.
- Position/momentum are in the Mu2e frame (mm, MeV/c); global time is ns. All are finite, momentum is positive,
  and PDG is +/-13. No shifts, smearing or momentum normalization are applied.
- The complete table is loaded at construction with typed TTreeReader values. Types, scalar shape, branch counts,
  normal end-of-input and row counts are checked. An empty particle tree is valid.

Output energy uses ParticleDataList's signed-PDG mass. Generator ID is `fromStepPointMCs`; proper time is zero.
The saved entries do not retain upstream proper-time/decay history. This is a fresh conditional entry experiment,
not a reproduction of each original Geant4 continuation. Muon decay is absent from the fast model.

## Prepare and verify

Use the same saved earliest-VD-or-surface selector without requiring a Geant4 straw hit. Supply the saved origin
explicitly to preserve its surface conversion; do not refit it or tune it to the benchmark output.

```bash
python3 -m trkfastsim.validation.entry_replay prepare /path/to/sgsteps.root \
  --gdml /path/to/run1a.gdml --origin -3904 0 10170.999666398546 \
  --source-art /path/to/source.dts --outdir /path/to/prepared
```

This writes `entries.root`, numerical `entries.json`, and `envelope.fcl` from the actual `Envelope(geometry)`.
`--source-art` checks the highest Events cycle against the generator-key census; full event identity coverage is
checked by the replay itself. The denominator is the **S2-selected source events**, not all generated CRY.

```fcl
physics.producers.entryReplay: {
  module_type: TrackerEntryParticlesFromTable
  tableFile: "/path/to/prepared/entries.root"
  writeReadback: true
}
physics.producers.g4run.inputs: { primaryType: GenParticles primaryTag: entryReplay }
```

`directory` defaults to `entry`, `writeReadback` defaults off. Schedule the producer before its consumer without
an upstream response filter. With readback enabled, TFileService writes `<label>/events` and `<label>/particles`,
including empty events and actual output states plus energy, proper time and generator ID. Only source_sp is
copied from the input mapping. No TFileService is requested when disabled.

```bash
python3 -m trkfastsim.validation.entry_replay check /path/to/prepared/entries.root \
  /path/to/readback.root --directory entryReplay --source-art /path/to/source.dts
```

Synthetic reader fixtures can be prepared with `entry_replay smoke --outdir DIR`. The supplied
`TrkFastSim/fcl/tracker_entry_smoke.fcl` uses EmptyEvent **only for this three-event test**, not the real benchmark.
Run it after the user build, then compare readback. `tracker_entry_smoke_bad_count.fcl` selects `bad-count.root`
and must fail at construction with `TRKENTRYTABLE: particle count mismatch`. Both checks passed in the built
runtime. Additional malformed-table cases are covered by Python tests, not yet by C++ runtime checks.

## Configured pilot

`tracker_entry_replay.fcl` runs the reader on all original RootInput event IDs with old products dropped.
`tracker_entry_g4_pilot.fcl` runs the first 20 source events through entry replay, single-stage Mu2eG4, and the
stock StrawGasStepMaker, with no response filter. Source products are dropped to prevent old tracker steps from
entering the new step maker. Run from a data directory next to `prepared/`, and add that prepared directory to
`FHICL_FILE_PATH` so the generated `envelope.fcl` is found. Pass the source art file with `-s`.
The pilot explicitly uses `geom_run1_a.txt` and `bfgeom_no_tsu_ps_v01.txt`, as recorded in the source configuration.

```bash
python3 -m trkfastsim.validation.entry_pilot --entries /path/to/prepared/entries.root \
  --coverage /path/to/replay/entry-readback.root --pilot /path/to/pilot/entry-g4-pilot.root \
  --pilot-art /path/to/pilot/entry-g4-pilot.art --metadata /path/to/prepared/entries.json \
  --reference-gdml /path/to/saved.gdml --observed-gdml /path/to/pilot/entry-g4-pilot.gdml \
  --events 20 --output /path/to/pilot/check.json
```

The first pilot retained 20 events (seven empty), transported 13 primary muons and wrote 209 gas steps in 11 events.
All 13 muons ended through the cylinder cut, with matching boundary hits and no step-limit stops. Endpoint
excess outside the cylinder was 14.965 mm median and 16.950 mm maximum: this is not surface-interpolated truth.
All parsed straw geometry fields matched the saved GDML exactly; other world materials and field-map values are
not validated by that straw-only comparison. The boundary dumper's `vdsteps` tree holds pre-step boundary hits,
not physical VD identifiers; SimParticle end states provide the post-step test. Momentum/time truth is compared
at the existing dumper's float precision. Small-pilot startup and GDML output must not be used as a throughput rate.

## Shared downstream chain

`tracker_entry_inject.fcl` applies the same existing StrawGasStepsFromTable adapter to both arms, running
entryReplay first so each per-event `genidx` refers to the same entry collection. It drops source payload and
keeps every original event, including no-entry and zero-step events. No straw-response filter is applied.
Both arms carry empty trajectory/surface truth and no CRV/calo response; the injector's stop codes are not
used for transport classification (the original transport tables are retained).

Stock FindMCPrimary throws on empty SimParticle collections. The shared injection instead supplies stock
NullMCPrimary event-level bookkeeping, while preserving actual SimParticles separately. For selected CH output
only, `tracker_entry_ntuple.fcl` reconstructs a nonempty event primary from `compressRecoMCs` before EventNtuple:
its genealogy helper requires one. This is not truth selection for track fitting. No common-surface truth-resolution
claim can be based on the absent SurfaceSteps or the event-primary-relative compression summaries.

`tracker_entry_digi.fcl` and `tracker_entry_reco.fcl` use stock OffSpill configurations, Run1a geometry, Sim_best
v1_1 conditions and the same seed. Digitized output keeps all events; stock CH output selection remains enabled,
so ntuple count must equal selected reco count, not the source count. Failed selections remain in the full
52,037-event efficiency/throughput denominator. The four-stage 20-event pilot completed, with no accepted CH
tracks; nonempty selected-event ntuple validation therefore remains a check of the full run.

Scripts `entry_g4_chain.sh` and `entry_fast_chain.sh` run already-built code and record external timing per stage.
`entry_follow_fast.sh` serializes the fast arm behind a specific G4 driver and refuses to launch it unless the
G4 chain finishes. `entry_benchmark` checks finished file counts and pairs reconstructed yields by event identity.

## Matched boundary and timing

The paired Geant4 job requires the opt-in `outwardCylinder` cut in Offline. It uses the same strict position and
outward-direction predicate as fast `Envelope.leaving`, with the generated full-precision radius and z bounds.
It is evaluated after a transport step, not an exact surface intersection; overshoots need checking in the pilot.
The fast model's path/chord cap is separately reported, not silently counted as an ordinary exit.

Only primary muons contribute in the scoped benchmark; Geant4 secondaries are killed by the existing `primary`
stacking cut. Full secondary showers, return passages, pileup and CRY-to-tracker world transport are excluded.
The Geant4 and fast physics models are not declared equivalent merely because their boundaries match.

Use the same built Offline, geometry, conditions, digi/reco configuration and event census for both new arms.
Charge each route for its real table/input adapters and output I/O; record payload and full process wall times
separately from environment setup and queue wait. Optional readback is a correctness test, not production output.
Historical completed downstream timings remain context, not a matched baseline for a different Offline build.
No stock-grid multiplier or end-to-end speedup exists until both finished timed arms and their yields are measured.
