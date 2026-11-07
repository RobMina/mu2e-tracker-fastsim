# Complete-census current-main entry fast arm

`python3 -m trkfastsim.validation.entry_fast` consumes the prepared **typed** `entry/events` and
`entry/particles` through `entry_replay.read`. It transports all entries once with the current NumPy
`fieldsteps.transport`, then exports TTrees for `StrawGasStepsFromTable`. It does not read historical
Geant4 gas steps, require a Geant4 response, select self-passing tracks, or impose a fast-exit time
window on a reference. The default census is **52,037 events and 43,156 entries**, including the
8,885 empty-entry events and four multi-entry events. `--expect-events` and `--expect-particles`
assert input counts; they do not truncate or select rows.

## Model and field

Supply the saved source GDML and the explicit origin `-3904 0 10170.999666398546`; a different
origin is rejected. `--metadata prepared/entries.json` checks its origin and outward-cylinder
parameters against the loaded geometry. Interior support comes from the same saved GDML.
Transport uses current-main defaults, explicitly **`complete_outer_wall=False`**: 20 mm maximum
chord, 0.02 mm sagitta, 30,000 mm path cap and 20,000 chord cap; default muon loss, wire, tail and
MSC remain enabled. Status 0 means outward-envelope exit, 1 stop, 2 cap. Caps are not exits.
`dump/final` retains each status and path/chord count, including zero-hit tracks.

The normal field argument is **`--map HEADER`**, as in `field-compare`, not `--field` or a geometry
configuration filename. The current `fieldcompare.py` has no `Field` class: it calls
`ehelix.read_g4bl` and `ehelix.grid_field`. This module reuses that route. Stock
`bfgeom_no_tsu_ps_v01.txt` inherits `bfgeom_v01.txt`'s G4BL/trilinear configuration, full DS field
and scale 1.0. Its tracker field is `BFieldMaps/Mau13/DSMap.header` with sibling `DSMap.bin`.
The other configured inner maps are TSd and DSExtension; the outer map is WorldMap. They are not
silently substituted here. Every field evaluation must lie within the DS interpolation grid or
the run fails, rather than inheriting the loader's zero-outside behavior. A basename check is
not proof of stock contents: use the unmodified map resolved for the matched Offline job and
record the input paths and repository revisions. Map-value agreement with Offline is still a
validation requirement, not a claim made by the adapter. Nominal field is evaluated at the
saved origin, matching the existing `fieldcompare` convention.

## Outputs and injector identity

- `fasttrk_field.root`: `dump/sgsteps`, `dump/particles`, `dump/events`, `dump/entry_map`, `dump/final`.
  All are TTrees with complete schemas even when empty. The shared core TTree exporter is reused;
  this wrapper adds empty trees that it normally skips.
- `dump/entry_map`: exact `(run, subrun, event, source_sp, index, sp)` mapping, with signed int64
  `source_sp` retained without double mediation. The source key is opaque, not a new SimParticle key.
- `dump/particles`: **every** transported primary, `key = index + 1`, `genidx = index`, `parent = -1`.
  Complete `sx,sy,sz,spx,spy,spz,st` and `ex,ey,ez,epx,epy,epz,et` come from entry and final states
  in the Mu2e frame, even with zero straw hits. This avoids inferring states from the first/last hit.
- Gas steps use the existing `fieldcompare` injector schema, detector-frame positions, and the same
  positive keys. `nsgs` is the whole event's step count (also correct for multi-entry events).
- `dump/events` preserves the complete source census and adds gas-step counts including zeros.
  The injector does not consume this tree: the original RootInput stream must supply every event.
- `entry-fast.json`: census, zero-hit counts, distinct exit/stop/cap counts, real transport wall time,
  internal transport timing, model settings, field bounds and input paths.
  There are no throughput extrapolations or benchmark ratios.

Run `entryReplay` before the injector on the original source stream, dropping old physics products,
without a response filter. Set these parameters on the matched downstream configuration:

```fcl
physics.producers.entryReplay: {
  module_type: TrackerEntryParticlesFromTable
  tableFile: "/path/to/prepared/entries.root"
  writeReadback: false
}
physics.producers.compressDetStepMCs: {
  module_type: StrawGasStepsFromTable
  trkTableFile: "/path/to/fast/fasttrk_field.root"
  genParticles: entryReplay
  crvTableFile: ""
  trkKeyOffset: 0
  primaryKey: 1
  makeTrajectories: false
}
```

`genParticles: entryReplay` is essential: `genidx` refers to the saved entry collection, **not**
original CRY generator ordering. The explicit particle states are used by the injector even when
there is no hit. This excerpt is not a complete digi/reco schedule. Trajectories and SurfaceSteps
are absent; do not claim equivalent `trksegsmc` truth. If downstream consumers require either,
settle a common policy for both arms before timing. The injector also uses float momentum/time
in its internal particle state and unknown end-process codes; exact double readback and stop/cap
classification live in the tables, not those injected fields.

## FNAL commands to run after review (not executed here)

Use the approved GPVM ana environment and the current-main core checkout, not a study branch.
Set the path variables to the already-local prepared data and saved geometry; no input transfer
or build is needed for this Python adapter.

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana
cd "$TRK2_BENCHMARK_CODE"
export PYTHONPATH="$PWD:$FASTSIM_CORE_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$FAST_OUT"
/usr/bin/time -v -o "$FAST_OUT/process-time.txt" \
  python3 -m trkfastsim.validation.entry_fast "$PREPARED/entries.root" \
  --metadata "$PREPARED/entries.json" --gdml "$SAVED_GDML" \
  --map "$STOCK_MAP_ROOT/BFieldMaps/Mau13/DSMap.header" \
  --origin -3904 0 10170.999666398546 --seed 1 --outdir "$FAST_OUT" \
  > "$FAST_OUT/process.log" 2>&1
```

All five thread variables must be 1 **before Python starts**. This is one NumPy process, with no
workers or device backend. `transport_s` brackets the real `fs.transport` call only. The external
wall includes imports, table loading, GDML/support and map setup, adapters and ROOT/JSON
output; setup and queue wait are outside it. Preserve both measurements. A future downstream run
must similarly be timed externally, on the same built Offline and conditions as the new G4 arm.
Do not add optional replay readback to only one production timing.

## Remaining matching checks

Only synthetic/mocked unit tests have run locally; no field transport, remote jobs or builds were
launched. Still required: actual full-census transport and injector readback; agreement of event IDs
and `genidx` including zero-hit/multi-entry cases; matched detector-frame origin in art; stock field
value comparison; common geometry/conditions/digi/reco and trajectory/truth policy; boundary
overshoot and non-exit (stop/cap) accounting. The finished matched G4 reference must keep all entries
and use its own outward-cylinder stopping rule, never the fast exit time. Muon decay and detailed
secondary/world physics are not modeled equivalently. Finished ntuple highest-cycle event counts
must equal the selected reco output count; injected/digitized counts must equal 52,037 and the throughput/yield
denominator remains the complete source census. No speedup claim is available yet.
