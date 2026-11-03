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
- **Loss:** muons, the slice sampler of `mu2efastsim.transport.slab_loss` (ionisation only; `radiative=True` /
  `--mu-radiative` adds bremsstrahlung, pair production and photonuclear losses in the solids); electrons,
  `fieldsteps.electron_loss` (`mu2efastsim.eioni` continuous loss below 0.1 MeV with Urban fluctuations, deltas and
  brem photons above it sampled one by one). Gas eIon per merged crossing: `eloss.sample_loss` (muons) or
  `electron_gas_eion` (restricted Moller loss, largest transfer T/2). `gas=fieldsteps.STRAW_GAS_G4STATE`
  (`--gas-state gas`) uses Geant4's gas branch of the density effect for the straw gas (the geometry's state).
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
| eIon | mean within 0.3 %, MPV within 1.6 % | mean +2.3 %, MPV +2.8 %; with `--gas-state gas` -0.02 +- 0.07 %, MPV within 0.1 % |
| width | -1.9 to -2.3 %, chi2/ndf 1.3 | within 0.3 %, chi2/ndf 1.3-2.5 |
| time at the first straw | below 1 ps | below 0.1 ps |
| momentum at the last Geant4 straw | within 0.3 MeV/c below 500 MeV/c | +0.1 MeV/c (Geant4 drop 0.75) |

Control arm without field: -2.3 straws per track and zero width. Step end points of the Geant4 sample lie on the
straws of the GDML used (99 % within 5e-5 mm of the gas surface).

Throughput, NumPy, one CPU core, transport only: about 90 muons/s and 57 conversion electrons/s.

Diagnostics: `--offset-bound` (first-straw offsets of side entries against the Highland spread of the support),
`--matched-straws` (time, wire DOCA and length on the straws both arms cross), `--delta-census` (Geant4 e+- daughters
above 10 MeV in the straws), `big_drops` (where tracks lose more than 10 MeV between straws), `--self-pass-only` and
`--unbiased-only` (the selection of the reference sample, below).

- **Side entries.** Muons entering through the outer cylinder reach their first straw 1.1, 0.75, 0.43, 0.25 mm
  (median, 2-5, 5-10, 10-20, 20-50 GeV/c) from Geant4 when the fast arm has no support material: this is Geant4's own
  scattering in 1.6-3.2 X0 of support over about 330 mm (inside the Highland band of every bin, offset x p roughly
  constant); with the fast support the offsets grow by 1.15-1.7 (independent draws: sqrt 2). Explained.
- **Electron gas eIon.** The straw gas is a gas in the Geant4 geometry, and Geant4 uses the gas branch of the
  Sternheimer parametrisation for it; the solid branch puts the restricted loss of 100 MeV electrons 2.4 % high.
  Explained; the opt-in gas state closes it (table above).
- **Reference-sample selection.** The cosmic reference keeps an event when some particle's straw steps have |p|
  between 30 and 250 MeV/c (15 steps, 16 planes), or above 1 GeV/c with 20 steps (prescaled 1/100). Almost every muon
  above 1 GeV/c in it was kept because another particle passed, typically a hard delta electron, and low-momentum
  muons are kept when their own loss upstream of the straws brought them into the window. The Geant4 muons are
  therefore conditioned on their energy loss: above 1 GeV/c, 13.6 % lose more than 10 MeV between two straws
  (fast 0.13 %, the Bethe rate), and e- daughters above 10 MeV appear in the straws about 100x more often than that
  rate. The momentum-loss differences above 1 GeV/c (+5 to +75 MeV/c at the last straw) are this selection, not
  missing physics: the radiative losses move the mean above 5 GeV/c by about +0.3 MeV/c. An unbiased high-momentum
  reference needs a sample without that filter.

## Stock reconstruction on field-on steps

Primary-only steps (fast; Geant4 after the entry state; Geant4 first passage) injected into the reference events
(`art/TrkFastSim` `StrawGasStepsFromTable`), then stock Run 1A OffSpill digitisation, stock reconstruction (CentralHelix
`KKCHmu`, `bfcorr` false) and EventNtuple, the same release and seeds for every arm; compared per event with
`python3 -m trkfastsim field-reco` (20 reference files, about 17,700 events with Geant4 primary steps, about 470 tracks
per arm). The EventNtuple configuration needs the CentralHelix fit under `trk.fits` (`EventNtuple.PathOff`, `trk.fits :
[ Off ]`, `FitType : CentralHelix`, `trk.fillTrkDtDt : false`) with the EventNtuple release of the backing Offline.

| Fast against Geant4 first passage (paired per event) | |
|---|---|
| track yield per event | +0.23 +- 0.06 % absolute (ratio 1.08); all of it from muons below 300 MeV/c at the Geant4 first straw (selection, above); 0.3-1, 1-2, 2-5, > 5 GeV/c within errors |
| active hits per track | +0.24 +- 0.16 (36.4) |
| chi2/ndof, fit consistency | -0.05 +- 0.03; +0.02 +- 0.01 |
| reco momentum - own first-straw momentum, median (sigma from the IQR) | 1-2 GeV/c +0.3 % (1.1 %) vs -0.1 % (1.1 %); 2-5 GeV/c +0.05 % (1.8 %) vs -0.2 % (1.8 %); > 5 GeV/c -0.3 % (6.2 %) vs -0.7 % (7.3 %) |

On the same straws, hit edep is +2.2 % higher in the fast arm (mean; TOT +0.7 %), as the solid-branch density
effect predicts for these mostly > 5 GeV/c tracks (the gas branch lowers the restricted loss by 1.5-2 % above 10
GeV/c); the steps differ by
35 um in wire DOCA (2-5 GeV/c, IQR) and below 1 ps in time. Geant4 all passages against first passage: yield -0.03 +- 0.03 %, hits
-0.13 +- 0.07.

## Open

- An unbiased reference for muons above 1 GeV/c and for side entries (the selection above).
- No world transport to the tracker, no fast field-on digitisation (stock digitisation through the art injector
  instead), no field-on reconstruction.
