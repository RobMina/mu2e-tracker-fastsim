# Tracker with the field on (opt-in)

`trkfastsim.sim.fieldsteps` transports charged primaries through the tracker in a magnetic field and writes their
StrawGasSteps; `python3 -m trkfastsim field-compare` compares them with Geant4 on the same tracks. Nothing in the
default (field-free, extracted) chain changes. The world transport above the tracker (CRY to the DS) is not part of
this: tracks start from a given state at the tracker envelope.

## Model

- **Stepping:** chords of the local helix (`mu2efastsim.ehelix.helix`, field at the helix midpoint of the step), at
  most `chord_mm` long and with a sagitta below `sagitta_mm` (defaults 20 mm and 0.02 mm; 10 mm for electrons in the
  comparisons). The field is a G4BL map (`ehelix.read_g4bl` + `grid_field`, e.g. the DS map), a constant vector, or
  none. The tracker volume is vacuum.
- **Straws:** gas crossings per chord (`ChordIndex`: panels sorted by z, only those overlapping the chord are tested;
  same rows as `crossings.straw_crossings` clipped to the chord); a crossing cut by a chord end is merged with its
  continuation, so a row is one passage through a straw.
- **Material:** straw walls (outer-cylinder chord minus the gas chord), sense wire at its closest approach, gas, and the
  support structure from the GDML (`fieldsteps.interior`: TrackerInterior's coaxial shells, limited to pieces reaching
  inside the envelope cylinder; no calorimeter).
- **Loss:** muons, the slice sampler of `mu2efastsim.transport.slab_loss` (ionisation only); electrons,
  `fieldsteps.electron_loss` (`mu2efastsim.eioni` continuous loss below 0.1 MeV with Urban fluctuations, deltas and
  brem photons above it sampled one by one). Gas eIon per merged crossing: `eloss.sample_loss` (muons) or
  `electron_gas_eion` (restricted Moller loss, largest transfer T/2).
- **Scattering:** one angular kink per chord at its end, Highland with the log term on the cumulative x/X0, plus the
  single-scattering tail for muons (none for electrons); the kink's lateral displacement is dropped.
- **Step content** as `MakeStrawGasSteps` with the field on: path in the gas, mean of entry and exit momentum, width
  = min(sagitta rms, bend rms) with the nominal field at the tracker centre, shape from |p| against the straw-radius
  momentum, ionisation type, time along the arc.
- Tracks stop when they leave the envelope (straws + 150 mm radially and axially), range out, or reach the path or
  chord cap. Tracks that leave and come back (loopers in the field, electrons returning from the calorimeter) are not
  followed.

## Comparison

`field-compare` reads StrawGasStepTableDumper tables of a field-on Geant4 sample, takes the primaries of a species
(and, in a resampled cosmic stage, their same-species continuation), restarts each from its first tracker VD crossing
(`--entry vd`, `--entry-vds`), its first tracker SurfaceStep (`--entry surface`) or the earlier of the two
(`--entry first`), and compares the primary steps after that time, paired per track: straws per track, path, eIon,
width, shape and ionisation types, momentum and position at the last Geant4 straw, time at the first straw, and the
same block for the first passage (Geant4 steps up to the fast exit time). It also checks the geometry (Geant4 step
end points against the straw surfaces) and writes primary-only tables for the art injector
(`--write-tables`, `art/TrkFastSim` `StrawGasStepsFromTable`).

## Results (Run 1A Geant4 samples, DS map; first passage)

| | Muons (cosmic, ~13,600 tracks) | Conversion electrons (~56,100 tracks) |
|---|---|---|
| straws per track, fast - Geant4 | +0.02 +- 0.07 to +0.28 +- 0.07 (two seeds; Geant4 39.5) | +0.05 +- 0.02 (Geant4 39.5) |
| path in the gas, chi2/ndf | 44/48 - 68/48, means within 0.1 % | means within 0.1 %, chi2/ndf about 4 |
| eIon | mean within 0.3 %, MPV within 1.6 % | mean +2.3 %, MPV +2.8 % (open) |
| width | -1.9 to -2.3 %, chi2/ndf 1.3 | within 0.3 %, chi2/ndf 1.3-2.5 |
| time at the first straw | below 1 ps | below 0.1 ps |
| momentum at the last Geant4 straw | within 0.3 MeV/c below 500 MeV/c | +0.1 MeV/c (Geant4 drop 0.75) |

Control arm without field: -2.3 straws per track and zero width. Step end points of the Geant4 sample lie on the
straws of the GDML used (99 % within 5e-5 mm of the gas surface).

Throughput, NumPy, one CPU core, transport only: about 90 muons/s and 57 conversion electrons/s.

## Open

- Muons above 1 GeV/c lose less than in Geant4 at the last straw (+5 to +75 MeV/c): the muon loss has no radiative
  part (brem, pair production, photonuclear), also in the default chain.
- Electron gas eIon is 2-3 % high.
- Muons entering through the outer cylinder reach their first straw 0.5 mm (median, above 2 GeV/c) from Geant4 even
  without the support material; tracks entering through the end planes agree to 1 um. Not understood.
- No world transport to the tracker, no fast field-on digitisation (stock digitisation through the art injector
  instead), no field-on reconstruction.
