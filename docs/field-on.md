# Tracker with the field on (opt-in)

`trkfastsim.sim.fieldsteps` transports charged primaries through the tracker in a magnetic field and writes their
StrawGasSteps; `python3 -m trkfastsim field-compare` compares them with Geant4 on the same tracks. Nothing in the
default (field-free, extracted) chain changes. The world transport above the tracker (CRY to the DS) is not part of
this: tracks start from a given state at the tracker envelope.

**Pause checkpoint (2026-11-05):** the CE hand-off and bounded cosmic stock-reconstruction studies below are
complete at their stated scope, not production certification. The experimental complete-wall development A/B is
complete and remains **disabled by default**: cosmic core/step discrepancies persist and reconstruction improvement
is not demonstrated. Historical CE distinct-straw output still needs a saved-only active-state re-audit. The
sections below retain the sequence of development results; the final two sections describe the current wall option
and completed pilot. General upstream world transport, device validation and matched end-to-end cost remain open.
See the [continuation checkpoint](field-on-checkpoint.md) for archive dependencies, check scope and resume gates.

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

## Initial-study limitations and later follow-up

- The initial CosmicSignal reference is selected; the later CosmicCRYAll parent studies below avoid that particular
  momentum-window/prescale, but retain their explicitly stated source and first-clock-window conditioning.
- General upstream world transport remains unvalidated. There is no validated fast field-on digitisation or fast
  field-on reconstruction; the completed studies below use stock digitisation and stock reconstruction through the
  art injector. The CE hand-off is a bounded development route, not general world transport.

## Conversion-electron hand-off (development)

`python3 -m trkfastsim.validation.fieldce` connects the umbrella's `beamfastsim.cechain` to tracker field steps,
with `--mode entry` as a control restarted from the Geant4 primary's first VD13 crossing and `--mode chain` for
transport from the electron's birth state. Both consume the tracker dumper tables and retain integer
`run:subrun:event` identities in the injector TTrees and `states.root`. If the GenParticle collection contains the
parent muon stops, birth states come from the primary electron SimParticles instead. Multiple primary electrons in
one event are rejected. The default extracted simulation and its CLI are unchanged.

```bash
# Include the umbrella's beam directory in PYTHONPATH for mode chain.
python3 -m trkfastsim.validation.fieldce "$TABLE" --mode chain --outdir "$OUTPUT" \
  --gdml "$GDML" --maps "$DS_MAP" "$TSD_MAP" --offline-root "$OFFLINE_SOURCE"
python3 -m trkfastsim.validation.cereco g4="$REFERENCE_NTUPLE" fast="$CANDIDATE_NTUPLE" \
  --states "$OUTPUT/states.root" --outdir "$COMPARISON"
```

`cereco` evaluates momentum on the first in-bounds TT_Front segment against that arm's own VD13 momentum, rather
than a truth momentum at a different straw or another particle. It retains no-track events in the denominator,
reads the highest ntuple cycle and reports paired errors on yield, active hits and tail fractions; the median and
IQR core use shared-index event bootstrap errors. **Reporting caveat:** the inherited `cereco` distinct-straw
reader still uses `state > 0`, which selects right-side drift states rather than all active KinKal hits. Historical
and current CE distinct-active-straw output remains uncertified pending a separate saved-only re-audit. Active-hit,
efficiency and momentum results are unaffected by that count definition; the corrected cosmic-pilot reader is
described below.

These are development tools, **not a validated end-to-end CE result**. In particular:

- A dts reference already passed Geant4's event filter. The paired same-birth sample is conditional on that
  selection; it cannot establish efficiency per generated electron.
- The stock OnSpill CE configuration reconstructs LoopHelices, not CentralHelices. The OffSpill CentralHelix uses
  cosmic LineFinder seeds, a muon mass, 10 GeV/c seed momentum and a filter requiring momentum above 300 MeV/c;
  that selection is unsuitable for 105 MeV/c CE. An electron mass/seed and unfiltered output are explicitly a
  different configuration, not stock CE production or a main/default promotion.
- The transport follows only the primary's first passage through the tracker. It has no generated secondaries or
  calorimeter return passages. The default gas-state setting remains unchanged; an explicit gas-state arm is a
  separate study.

### Pilot summary

On one selected CE reference file, the cosmic-seeded CentralHelix configuration has no paired good tracks between
the fast and Geant4 arms even with electron mass/seed settings. It is not a validated CE reconstruction path.
The stock LoopHelix electron fit does reconstruct the same inputs: the entry-state control's yield differs by
+0.64 +- 0.36 percentage points; the complete CE hand-off differs by -1.71 +- 0.57 percentage points. The robust
momentum core in the entry control is 0.020 +- 0.008 MeV/c narrower; tail fractions agree within the pilot errors.
The yield and core differences remain open. This is conditional on the reference's Geant4 selection, uses one
transport seed and compares first-passage fast primaries with Geant4 primaries including returns, not full production.

On a first cosmic parent file without the CosmicSignal momentum window/prescale, the fast first-passage straw count
is 2.32 % low (paired difference -0.461 +- 0.025 straws/track). The >5 GeV/c momentum-loss gap shrinks to
+1.75 +- 1.90 MeV/c at a common last straw. The loss-selection explanation is supported, but the straw multiplicity
is not closed. No new default is justified by these pilots.

The corresponding unchanged stock cosmic reconstruction on that first parent gives a paired track-yield difference
of +0.083 +- 0.053 percentage points and +0.077 +- 0.089 active hits per matched track. The total relative residual
IQR widths are 2.47 % fast and 2.46 % Geant4, using each arm's own first-straw momentum as a proxy; this is not the
same-surface CE truth metric. Momentum-bin yield/width excursions and the step deficit remain open; paired width errors are given below,
without multi-file confirmation. This pilot is not a full field-on validation.

### Saved-arm attribution diagnostic

`python3 -m trkfastsim.validation.ceaudit --entry-events "$ENTRY_COMPARISON/events.csv.gz" \
--chain-events "$CHAIN_COMPARISON/events.csv.gz" --entry-dir "$ENTRY_OUTPUT" \
--chain-dir "$CHAIN_OUTPUT" --outdir "$AUDIT_OUTPUT"` joins the saved entry/full-chain event tables only after
requiring identical event denominators and saved Geant4 baseline columns. Missing hand-offs and zero-step events
remain in an exhaustive signed-yield decomposition. Step passages and distinct physical straws are counted
separately; entry-state-conditioned summaries are explicitly truth diagnostics, not response or correction models.

The optional follow-up injector `g4cut_field.root` uses Geant4 steps up to the entry-fast exit clock plus 1 ns for
tracks that exited. It is a time-window sensitivity control, **not an independently determined Geant4 first passage**.
It must not be interpreted as an unbiased secondaries/returns removal or a validated production sample.

The saved pilot decomposition localizes the full-chain signed deficit to six reference tracks lost before VD13
and 60 in events reaching fast VD13 but producing no fast tracker steps, offset by 18 extra tracks where fast steps
exist. It does not identify a correction: upstream stochastic histories are conditioned on the reference selection.
The entry-control core remains narrower after excluding events flagged by the fast-exit clock; late-step removal
alone is not an explanation. These are development diagnostics on the original pilot, not held-out confirmation.

`python3 -m trkfastsim.validation.cegeometry --chain-root "$CHAIN_OFFLINE_SOURCE" \
--stock-root "$CAPTURED_RUNTIME_GEOMETRY" --outdir "$GEOMETRY_AUDIT"` compares represented CE geometry and selected
effective configuration parameters, hashing the include files. It does not certify unrepresented supports or
GDML material navigation. Compare the captured runtime source, not a moving local geometry alias.

For an independently audited unfiltered input, first run `python3 -m trkfastsim.validation.ceinput "$TABLE" \
--expected-events "$FINISHED_EVENT_COUNT" --outdir "$INPUT_AUDIT"`. It checks one primary electron per completed
event and counts primary tracker-step events without a VD13 crossing; zeros are explicit. A successful table audit
does not certify transport status or complete incident retention: those need external finished-art/status evidence.
Only then use `fieldce --reference-selection unfiltered --expected-starts "$FINISHED_EVENT_COUNT"` to label the
denominator. `cereco` propagates that receipt; absence of the receipt retains the selected-dts label. This population
may be bounded by its stop source and does not automatically measure a production-wide or POT-normalized efficiency.

`python3 -m trkfastsim.validation.fieldquantiles "$SAVED_COSMIC_EVENTS" --outdir "$CORE_AUDIT"` supplies paired
bootstrap errors on the cosmic relative median/IQR cores without changing the own-first-straw truth proxy.
On the pilot's 424 common-surface tracks in 2-5 GeV/c, the fast relative width exceeds Geant4 by
0.367 +- 0.131 percentage points. This is an open within-file discrepancy, before multiple-bin correction and
without independent-parent confirmation; the other width-bin differences are unresolved at these errors.

### Unfiltered CE development pilot

A separate, unfiltered 2,000-event bounded stop-pool sample has one audited primary electron per finished event,
all transport statuses successful, and no primary tracker-step event missing the VD13 entry boundary. Keeping all
2,000 generated events in the denominator, stock LoopHelix good-track efficiency is 51.70 % on Geant4 primary steps,
51.90 % on fast steps from Geant4 entry states and 51.20 % on the full fast chain. Paired differences are
+0.20 +- 0.29 and -0.50 +- 0.51 percentage points. This is compatible at the pilot's precision; it is consistent with
selection contributing to the larger conditional deficit, not a causal proof across different input populations.

The full-chain own-VD13 residual IQR width is 0.1782 vs 0.1980 MeV/c, difference -0.0198 +- 0.0106 MeV/c;
>1 MeV/c tail difference is +0.10 +- 1.06 percentage points. A second transport seed on the original selected sample
still gives a yield deficit (-1.32 +- 0.55 percentage points). Identical-Geant4-steps stock response reseeding gives
a width excursion -0.0076 +- 0.0057 MeV/c; a time-window control also changes unmodified-step events, demonstrating
stochastic-stream sensitivity. None of these controls establishes a precision resolution gate or erases an open
finding. No production OnSpill/pileup, secondaries, full-stop-pool, multi-file confirmation or default promotion is
claimed. All results use the unchanged stock electron fit and explicit development transport.

### Independent cosmic parents

`python3 -m trkfastsim.validation.fieldparents PILOT=events.csv.gz PARENT_A=events.csv.gz \
PARENT_B=events.csv.gz --pilot PILOT --outdir "$OUTPUT"` summarizes already-finished per-parent cosmic comparisons.
It rejects duplicate/missing/noninteger event identities and overlap across original parents. The explored pilot is
reported separately from the fixed confirmation parents; both confirmation-only and all-parent pooled estimates are
saved. Delete-one-parent jackknife errors accompany pooled yield, active-hit and fixed momentum-bin core differences,
while each parent retains its paired event-bootstrap core errors. Missing or underpopulated bins remain undefined.

The deletion unit is the original source parent, not the event; unequal parent sizes enter the event-weighted pooled
estimator. These are descriptive sampled-parent uncertainties, assuming independent parents, not a new acceptance
threshold or the histogram sign-flip family-wise gate. The denominator is still reference primary-step events after
the first-clock-window restriction, conditional on the original S2 selection, and the momentum truth is still each
arm's own first-straw proxy rather than the reco surface. No production-wide efficiency or closure is implied.

The optional `--steps LABEL=fieldcompare.json ...` requires exactly the same parent labels and also summarizes
frozen first-pass step multiplicities with parent-jackknife errors. Its exiting reference-track denominator is
reported separately from the reconstruction event denominator. Fixed momentum-bin yields are included with the
core metrics; no bins are selected or tuned after seeing confirmation data.

### Fixed independent CE input

With unchanged settings on a second selected source file (2,812 events), entry-only good-track yield differs by
+0.107 +- 0.354 percentage points; the full chain gives 86.202 % versus 88.265 % Geant4, difference
-2.063 +- 0.540 percentage points. The full-chain conditional deficit repeats independently. Its signed 58-track
loss decomposes into ten reference tracks before the fast entry, 47 in fast-entry events without tracker steps and
one net track where steps exist. That localizes the deficit without identifying a correction.

The full residual core is 0.1811 versus 0.1945 MeV/c, paired difference -0.01345 +- 0.00778 MeV/c; the entry core
difference is -0.00763 +- 0.00896 MeV/c. The narrowing has the same sign but is not independently resolved on this
file. Full median difference is +0.01094 +- 0.00548 MeV/c, and the >1 MeV/c tail difference is
-0.642 +- 0.648 percentage points. These conditional results do not erase the separate small unfiltered-pool
result, establish a production-wide efficiency, or close momentum resolution. No model was tuned between files.

### Efficiency root-cause diagnostics

`python3 -m trkfastsim.validation.cerootcause --entry-events "$ENTRY/events.csv.gz" \
--chain-events "$CHAIN/events.csv.gz" --entry-dir "$ENTRY_STEPS" --chain-dir "$CHAIN_STEPS" \
--gdml "$GDML" --outdir "$OUTPUT"` reports zero-inclusive migration for primary step delivery, primary tracker-term
acceptance and stock good tracks separately, together with saved zero-step endpoints. The primary tracker term uses
12 steps with ionization above 1 eV and momentum strictly above 50 MeV/c; it is **not** the full production dts filter,
which also accepts calorimeter and other-particle response. Conditioning on G4 good tracks is separately labelled
as tautological, not evidence for a physical cause.

Within the same unfiltered 2,000-event sample, good-track yield differs by -0.50 +- 0.51 percentage points overall,
but by -3.16 +- 0.81 points in the 1,106 events passing the G4 primary tracker term. The complement supplies 25 net
fast gains. This directly demonstrates a conditioning contribution within one population, without equating that
tracker term to the full dts selection or claiming unbiased production closure.

`python3 -m trkfastsim.validation.ceacceptance --events "$OUTPUT/events.csv.gz" --chain-dir "$CHAIN_STEPS" \
--gdml "$GDML" --map "$DS_MAP" --outdir "$VACUUM_OUTPUT"` follows zero-step fast entry states without material or
scattering until a gas crossing or exit. Full-panel and optimized first-hit searches are checked on identical
chords. Fixed position/direction/momentum hybrids are truth counterfactual diagnostics, not candidates or additive
mechanism fractions. On the two selected files, none of the 78 and 66 zero-step fast entry states hits gas in this
control; every corresponding G4 good-track entry does. Momentum and position/direction substitutions each restore
some crossings. This localizes acceptance sensitivity to the delivered entry states, without identifying a tuned
upstream correction or certifying the full stochastic material path.

`python3 -m trkfastsim.validation.ceupstream --events "$OUTPUT/events.csv.gz" --chain-dir "$CHAIN_STEPS" \
--map "$DS_MAP" --inner-radius "$GAS_INNER_RADIUS" --outdir "$UPSTREAM_OUTPUT"` reads saved fast material-encounter
counts and describes both lost and gained classes. It has no symmetric Geant4 material-path ledger. In the
unfiltered sample, G4-only step events have harder fast loss histories, while fast-only step events have harder
Geant4 loss histories. Such reciprocal migration prevents diagnosing excess fast loss from the selected lost tail
alone; it does not establish equality of the distributions. Local uniform-axial-field orbit reach is a truth-state
diagnostic, not a mapped-trajectory maximum or a response model.

Three fixed fast histories on the same unfiltered births give all-event good-track differences of
-0.50 +- 0.51, -0.05 +- 0.51 and -0.65 +- 0.54 percentage points. Conditioning on the G4 primary tracker term gives
-3.16 +- 0.81, -2.35 +- 0.80 and -3.80 +- 0.84 points; conditioning on the fast tracker term reverses the sign to
+1.62 +- 0.79, +2.23 +- 0.78 and +2.00 +- 0.82 points. This arm-dependent selection effect supports history migration,
not a correction that can be subtracted from the selected-file deficit. The three histories share the same births
and G4 baseline, so their paired errors must not be combined as independent confirmation or counted as more events.

`python3 -m trkfastsim.validation.ceseeds SEED_A=rootcause/events.csv.gz SEED_B=rootcause/events.csv.gz \
--outdir "$OUTPUT"` requires identical event keys and saved G4 baseline columns, reports each seed's migration and
descriptive seed spread, and retains the original event denominator. It does not estimate a stock multi-seed floor,
choose a preferred seed, or establish production closure. The full production dts filter and a symmetric upstream
material-path ledger remain outside these primary-tracker-only diagnostics.

The same-simulator control is also saved: pairs of fast histories differ by +0.45 +- 0.53, -0.15 +- 0.55 and
-0.60 +- 0.55 percentage points over all events, but by -1.71 +- 0.82, -3.24 +- 0.85 and -3.40 +- 0.87 points when
conditioned on the reference fast history's tracker term. Conditioning on the candidate history instead gives
positive differences. A selected-history deficit of this size therefore does not require a simulator-model mismatch;
it does not prove the fast model correct or attribute the full production-filter bias.

### Nine fixed cosmic confirmation parents

Keeping the explored pilot separate, nine additional source parents contain 368,559 reference-step events. Stock
CentralHelix good-track counts are 17,129 fast versus 17,131 Geant4, a paired difference of
-0.00054 +- 0.02353 percentage points using delete-one-parent errors (event-paired error 0.01778 points).
Active-hit difference is +0.08596 +- 0.03172 hits. These remain primary-only, reference-step-conditional quantities,
not generated-cosmic acceptance.

On the separate 364,052 exiting-track denominator, the first-clock-window step deficit is
-0.48994 +- 0.00626 steps per track, or -2.4786 +- 0.0311 % relative. This confirms the multiplicity finding despite
compatible total reconstructed yield. Fixed relative-core width differences, with parent-jackknife errors, are
+0.0916 +- 0.0474, -0.0102 +- 0.0555, +0.0901 +- 0.0328 and +0.0329 +- 0.1278 percentage points in
0.3-1, 1-2, 2-5 and above 5 GeV/c respectively. The 2-5 GeV/c width remains open before multiple-comparison
correction; the pilot's 1-2 GeV/c yield excess does not repeat. Truth remains each arm's own first-straw proxy,
not the reco location. Parent errors assume independent source parents conditional on shared source libraries;
these results do not supply a histogram family-wise gate or justify a default change.

### Stock detector-filter conditioning replay

The stock `DetectorStepFilter` was replayed on the existing unfiltered Geant4 art, without running Geant4 or
reconstruction again. Of 2,000 complete events, 1,151 pass its actual tracker-or-calorimeter selection: 1,106 pass
the tracker term, 1,028 the calorimeter term, with 983 passing both. Exact event identities, one diagnostic row per
completed event and the finished pass/fail counts were checked. On this sample the all-particle tracker flag equals
the primary-only flag; that equality must not be assumed for another sample.

For the same three fast histories, conditioning on the actual Geant4 detector-filter pass gives good-track
differences of **-2.87 +- 0.79, -2.00 +- 0.78 and -3.39 +- 0.82 percentage points**, compared with
-0.50 +- 0.51, -0.05 +- 0.51 and -0.65 +- 0.54 points over all 2,000 events. Net track differences inside/outside
the selection are -33/+23, -23/+22 and -39/+26. One Geant4 good track lies outside the detector-filter selection and
is retained; primary-injected reconstruction here does not require that filter to have passed.

This demonstrates detector-selection conditioning within a single complete population, beyond the tracker-only
proxy. Together with the same-simulator null, it makes history selection the leading explanation for the selected
comparison, not proof of fast-physics closure or a numerical correction to the separate production-selected sample.
The replay does not reproduce generator/status filters, and the fast model has no symmetric calorimeter or
secondary response. Independent complete populations and upstream path/material comparisons are still needed.

`python3 -m trkfastsim.validation.cestockfilter SEED_A=rootcause/events.csv.gz SEED_B=rootcause/events.csv.gz \
--replay-log "$STOCK_FILTER_LOG" --expected-events "$AUDITED_EVENTS" --outdir "$OUTPUT"` validates the saved
stock decisions and reports per-history full/selected/complement migration. It executes no art or transport itself.

### Explicit efficiency scope and denominator safeguards

`cereco` defaults to a labelled diagnostic. To request an efficiency estimate over an audited complete birth
population, add `--claim complete-birth`. This requires the sibling `handoff.json` to specify unfiltered input,
chain mode, a generated-start count and an equal positive externally audited event count. Selected input or an
absent/contradictory receipt cannot make this claim. An unfiltered entry-mode run remains a shared-Geant4-entry
tracker control, not validation of birth-to-entry transport. Scope eligibility is **not a physics acceptance verdict**.

Every report includes `efficiency_scope`: the input selection, transport boundary, denominator count, requested
claim and stable sorted integer birth-key digest. Future handoff receipts bind that digest; older count-audited
receipts remain readable but are explicitly marked as legacy count-only binding. Exact identities are normalized
to uint64 before joins; duplicate births and reconstruction/entry identities outside the denominator are rejected.
Missing handoffs, zero-step events and missing reconstruction remain in the original birth denominator.
`fieldce` refuses `--max-events` truncation of input labelled unfiltered; create and externally audit a bounded
complete input first instead of relabelling a selected subset.

Use separate comparisons for shared-entry tracker response, birth-to-entry delivery and complete birth-to-track
efficiency. A fresh fast history of a Geant4-filtered event is a conditional diagnostic: it need not retain the
favorable reference history. Neither an intersection selection nor agreement within limited errors establishes
unconditioned equivalence. A production filtered fast chain would need its own equivalent selection, including
calorimeter and secondary contributions; the present tracker-only chain does not certify that parity. No empirical
efficiency weight, momentum/radius correction or reconstruction-cut change follows from this scope fix.

`cereco --baseline-report "$SAVED_COMPARISON/cereco.json"` additionally requires exact agreement of all previous
physics metrics and event/ntuple counts before writing a new report. It permits new scope metadata, not changes to
finite values or bootstrap outputs. Saved selected and unfiltered comparisons pass this numerical no-change check;
selected inputs are rejected if explicitly requesting `--claim complete-birth`.

### Saved multiplicity definition audit

`python3 -m trkfastsim.validation.fieldmultiplicity PILOT="$PILOT_DIRECTORY" \
PARENT_A="$PARENT_DIRECTORY" --pilot PILOT --outdir "$AUDIT"` reads the three saved injector tables, requires
identical entry-particle identities, and separates step records, distinct per-particle physical straws and repeat
excess. Missing steps remain zero on the complete saved entry-particle denominator. The explored pilot stays separate.

Across nine confirmation parents, fast step records are 2.780 % low and distinct particle-straw pairs 2.784 % low
relative to the saved Geant4 clock-window arm. Only 267 fast records are repeat excess, versus none in the reference:
repeat records do not explain the deficit. This audit includes all entry particles, unlike the earlier exiting-track
subset, so its percentage does not replace that result. The Geant4 window remains paired fast exit plus 1 ns, not
an independently measured physical passage. No step-count scale or navigation correction is inferred.

### Partial upstream material diagnostic

`python3 -m trkfastsim.validation.cematerial --export "$RAW_STEP_EXPORT" \
--events "$SAVED_UPSTREAM/events.csv.gz" --expected-events "$AUDITED_EVENT_COUNT" --outdir "$AUDIT"` consumes
an already reconciled raw-step export. It identifies the generated primary through persistent StageParticle-parent
reseating into the raw transport ProductID, never by time or proximity. Target and absorber SD path lengths,
deposits and primary pre/post-step energy changes are summed before its first recorded VD13 crossing; no-entry
histories remain explicitly whole-history. `common_vd13_classes` separates histories with both arm entries.
Exact complete birth identities and saved reference VD13 coverage/clocks must agree. These are truth diagnostics,
not measured-variable corrections, and they neither supply all-material coverage nor a symmetric fast path ledger.

On the bounded unfiltered development pool, common-VD13 G4-only-step and fast-only-step classes have median G4
target path lengths 0.769 and 2.102 mm, respectively, and median summed primary energy decrements in target SD
records 0.581 and 38.622 MeV. These reciprocal hard-loss histories are not evidence for a uniform fast loss scale;
primary energy decrement is not local energy deposition. The partial ledger does not certify unbiased navigation.

The cosmic distinct-straw deficit also persists where both arms produce steps: confirmation parents contribute
-129,461 particle-straw pairs from shared step-producing particles, -74,369 from reference-only particles and +697
from fast-only particles. Their exact sum is -203,133. It is therefore not solely loss of entire step-producing
tracks. Trajectory/material/navigation and producer-grouping effects remain to be separated; no count correction
or physics closure is claimed.

### Disjoint upstream stop-record windows

Before generating a new inclusive control, `python3 -m trkfastsim.validation.cestopwindow \
--previous "$PREVIOUS_POOL" --new "$NEW_POOL" --expected-events "$AUDITED_EVENT_COUNT" \
--outdir "$WINDOW_AUDIT"` requires complete highest-cycle Events counts, an exact FileIndex physical-entry
permutation, unique integer upstream event identities, and zero overlap between the two bounded copies.
The optional `--fileindex-byteswap` explicitly handles byte-reversed custom-streamer metadata; it is never
inferred from plausible event numbers. Preserve the original copy command and record-window bounds separately.
This is an event-record disjointness check, not an audit of every generated particle, independent source-library
uncertainty, or POT normalization. Successful transport still needs an independent complete saved-event/status
census, stock selection decisions retained without pruning output, and a primary-birth audit before comparison.

### Closed-reference census before comparison

`python3 -m trkfastsim.validation.cecensus --art "$CLOSED_REFERENCE" \
--generation-log "$GENERATION_LOG" --saved-log "$INDEPENDENT_READ_LOG" \
--expected-events "$AUDITED_EVENT_COUNT" --run "$RUN" --subrun "$SUBRUN" \
--outdir "$CENSUS"` reconciles the original generated-event status log, an independent saved-art PrintModule
read and highest-cycle art FileIndex metadata against the declared complete integer event set. Add the explicit
`--fileindex-byteswap` option only for a verified custom-streamer byte-order workaround. Every status row must
belong to its own event and exact process/product; missing, duplicate or unknown trigger states are rejected.
Both inclusive and stock-shadow path decisions are retained. Nonzero statuses, overflows and step-limit kills
remain recorded as validity failures, not valid response zeros. Required raw product branches must be present,
but branch presence alone is not a per-particle completeness certificate. Require `physics_eligible` and a
separate complete primary/source audit before interpreting an all-birth transport comparison.

For exact table-to-reference binding, pass `ceinput --census-events "$CENSUS/events.csv.gz"` as well as the
external event count. Equal lengths with different integer birth identities are rejected; the report records both
the sorted birth-key digest and the census-file hash. This optional audit does not change transport settings.

When a ROOT custom streamer cannot decode persisted mixer identities, use installed `EventLister` and
`EventIDSequencePrinter` in a saved-input-only art job (`art/fcl/ce_reference_source.fcl`), then run
`python3 -m trkfastsim.validation.cesource --source-log "$SOURCE_READ_LOG" --window-art "$SOURCE_WINDOW" \
--expected-events "$AUDITED_EVENT_COUNT" --out "$SOURCE_AUDIT"`. The parser binds each generated ID to
exactly one persisted source ID in serial order and requires equality with the complete physical source-window
order; no inferred shift, phase-space matching or re-generation is allowed. This establishes source occurrence
only, not particle-link completeness or physics closure. Preserve the Python environment image hash alongside
the code snapshot, rather than silently following a replaced shared image.

`python3 -m trkfastsim.validation.ceadmit --census "$CENSUS/cecensus.json" --source "$SOURCE_AUDIT" \
--primary "$INPUT_AUDIT/ceinput.json" --out "$ADMISSION"` provides a fail-closed report-only admission receipt.
It requires matching birth-key digests, exact source occurrence, successful complete statuses and primary boundary
coverage. Admission is not a physics verdict or an independent re-read of the underlying products.

For the historical primary-only stock adapter, retain its injection, digitization, reconstruction and ntuple outputs.
`python3 -m trkfastsim.validation.cehistory --births "$STATES" --table "$INJECTOR_TABLE" \
--inject-art "$INJECTED" --digi-art "$DIGITIZED" --reco-art "$RECONSTRUCTED" --ntuple "$NTUPLE" \
--outdir "$HISTORY"` checks exact stage identities and writes a ledger on every supplied birth. Injection output
must equal step-bearing table events, digitized IDs must equal injected IDs, reconstruction output may select a
subset, and ntuple IDs must equal that reconstruction output. No-entry and zero-step births remain explicit in the
full comparison denominator, not silently removed. This is filtering/history evidence, not proof that skipped
births were physically processed or that zero response is valid. The stock `FindMCPrimary` requires a primary:
removing the adapter's step filter without changing empty-primary handling is not an inclusive-output solution.

Before final metrics, `cehistory --verify-report "$HISTORY/cehistory.json"` rehashes every closed input recorded
by the completed history. Recheck both transport manifests too: identity equality alone cannot detect a changed
momentum state, handoff receipt or step payload. A successful old stage audit does not validate a later-mutated file.

For an inclusive control that already persists actual stock-shadow decisions, `cestockfilter --census-events \
"$CENSUS/events.csv.gz" --expected-events "$AUDITED_EVENT_COUNT"` is an alternative to the saved replay-log
interface. Only all-birth, stock-OR-pass and stock-OR-fail conditioning are reported: tracker/calorimeter components
were not observed separately and must not be invented. Exact full identities, successful statuses and unchanged
shared reference columns are required before conditioning. The conditional numbers describe selection on a G4
history, not unbiased efficiency, symmetric fast filter parity, an efficiency correction or physics closure.

### Independent bounded CE confirmation

A second, disjoint 2,000-birth inclusive control was transported once with predeclared seeds and the frozen historical
runtime, geometry and physics. Independent closed-file status, source occurrence/order, primary coverage and exact
stage histories passed before the paired comparison. No generated event is removed from the denominator. The
primary-only adapter's explicit no-entry/zero-step/reconstruction-output selectors are accounted for separately;
a selected reconstruction output is not a processing receipt for omitted births.

With unchanged stock OffSpill digitisation and electron LoopHelix, good-track efficiency is **50.20% fast versus
49.95% Geant4**, paired difference **+0.25 +- 0.55 percentage points** (62 gains, 57 losses). On 942 paired tracks
with in-bounds TT_Front segments, using each arm's own VD13 momentum:

| Fast minus Geant4 | Paired estimate |
|---|---|
| Residual median | -0.00666 +- 0.01314 MeV/c |
| Residual IQR sigma | +0.01864 +- 0.01319 MeV/c |
| Absolute residual tail above 0.5 MeV/c | +1.486 +- 1.351 percentage points |
| Above 1 MeV/c | +0.637 +- 1.007 percentage points |
| Above 2 MeV/c | -0.106 +- 0.773 percentage points |
| Active hits per paired track | -0.019 +- 0.269 |
| Historical `state > 0` distinct-straw subset, not all active straws | +0.173 +- 0.188; active-straw re-audit required |

The shared-Geant4-entry control gives +0.55 +- 0.31 percentage points in good-track yield and
-0.00222 +- 0.01214 MeV/c in core width; it is not a birth-to-entry transport validation. The independent full-chain
core is wider, rather than reproducing the earlier narrowing. Neither this sign change nor an efficiency difference
smaller than its error establishes equivalence. The held-out sample is reported independently, not pooled with the
development pool or its shared-birth seed variants. Precision core/tail validation, production OnSpill/pileup,
secondaries/return passages, symmetric calorimeter selection and production-wide/POT normalization remain open.

Conditioning on the new sample's persisted stock detector selection changes the paired good-track difference from
+0.25 +- 0.55 points over all births to **-1.05 +- 0.89 points** over its 1,140 passing events. The complement has
17 fast-only good tracks and no Geant4 good tracks, giving the exact signed partition -12 + 17 = +5. The independent
sign reversal supports a history-selection contribution; the conditional difference alone is not resolved at this
precision and is not a correction or symmetric fast-filter comparison. Only the saved OR decision was observed;
tracker/calorimeter component decisions are not inferred.

### Cosmic search and truth-association diagnostics

`validation.fieldcrossing` compares the optimized chord search, full-panel search and a bounded independent
all-straw oracle on identical saved gas-step endpoint secants. It compares complete physical-straw sets and
finite intervals, not merely first-hit existence. Saved secants cannot reconstruct missing stochastic chords or
inter-straw material paths. `validation.fieldpathaudit` instead observes every search during a frozen fast replay,
returns the original search rows without RNG consumption, and requires exact ordered reproduction of all three
saved injector payloads before attributing its search evidence to the original run. Neither test establishes
physical Geant4 first passage or closes the cosmic step deficit by itself.

The frozen cosmic momentum-core numbers use an event-first-straw truth proxy. That construction does not guarantee
the truth straw belongs to the first reconstructed good track in multi-primary events, or that both arms reconstruct
the same primary. This is an association vulnerability to measure, not evidence that it caused the observed width
difference. `validation.fieldmcinventory` checks actual saved MC branches and record counts before attempting any
exact contributor-ID join. A corrected first-straw association would still not supply truth at the reco surface.
The original findings remain separately recorded; no proximity matching or response correction is implied.

On the explored cosmic pilot, the saved-secant check covers 792,917 fast, 816,152 Geant4-all and 814,223
Geant4-clock gas-step secants. The optimized and full-panel searches agree on all complete straw sets and finite
intervals at 1e-7 mm tolerance; the fixed first 256 secants per arm also agree with the uncullled all-straw oracle.
This rules out disagreement between those searches on those saved secants only. It neither reconstructs omitted
actual chords nor resolves merged/curved Geant4 passage definitions or the confirmed multiplicity discrepancy.

The actual pilot ntuples contain one valid best-contributor record per selected track, but every recorded particle
ID is zero, whereas the saved injector IDs are positive. Stock digitisation can rekey retained SimParticles from
zero; a local, nonpersisted remapping is not an exact link back to the injector. No additive ID offset is assumed.
An exact-identity first-straw comparison is therefore unavailable from the currently established saved links.
Nevertheless, all 424 paired pilot events in the 2–5 GeV/c core bin already have exactly one saved entry particle:
restricting the old proxy to such events leaves its +0.367 +- 0.131 percentage-point width excess exactly unchanged.
Multi-entry ambiguity does not explain that pilot bin. Surface truth remains empty, so no same-location resolution
claim follows; confirmation parents and the physical transport/grouping explanation remain separate questions.

The frozen stochastic-fast replay now reproduces the complete ordered particle/step payloads of all three saved
pilot injector tables exactly. On 3,656,249 actual transport chords, including zero-hit chords, optimized and
full-panel searches agree on all 1,012,272 gas-chord pieces and intervals. This excludes disagreement between
those two searches as the explored pilot's coverage-loss mechanism; shared box/geometry errors and material or
passage/grouping definitions remain outside that conclusion. The first 256 actual-chord oracle probes all had no
gas hit, so their independent coverage is vacuum-only; the separate saved-secant oracle is not a substitute for
independent all-straw coverage of omitted actual paths. Observer runtime is not transport throughput.

Across the nine fixed confirmation parents, only 53 of 368,559 reference-clock step events have multiple saved
entry particles. The singleton restriction removes three paired proxy events, none in 2–5 GeV/c. The relative
width difference there remains **+0.0901 +- 0.0328 percentage points**, with identical delete-one-parent replicates
and errors. Multi-entry association therefore does not explain this confirmation-bin finding either. It remains
an open first-straw-proxy discrepancy, not same-location resolution or a basis for tuning the response model.

### Material and saved-summary coverage diagnostics

`validation.fieldmaterial` observes the original `_straw_pieces` return values without changing inputs or consuming
RNG, comparing the wall actually supplied to the transport with finite outer-cylinder minus gas-cylinder lengths
on the same stochastic chords. It separates wall-only line grazes from pieces clipped away from gas by a chord
boundary. Expanded-radius candidate bounds and a bounded independent all-straw check on outer-hit chords provide
search checks, not an independent geometry certificate. Attribution requires exact ordered reproduction of all six
saved particle/step payloads and unchanged inputs, code and map bytes. Omitted wall length is not itself proof of
the cause of a reconstruction discrepancy, and observer timing is not transport throughput.

`validation.fieldgasledger` separately checks saved gas lengths, endpoint secants and each row's own assigned straw
geometry symmetrically in all three arms, preserving the full particle denominator, zero-step particles and repeat
rows. In the pilot, every endpoint lies within its own finite straw to 0.01 mm. Fast gas lengths never fall below
the endpoint distance by more than 0.001 mm, but 3,950 Geant4-all rows do, with a maximum shortfall of about 654 mm.
This does not establish an invalid trajectory: gas steps omit wire material, and a grouped summary can bridge
non-gas intervals. The table lacks constituent ordering, separate boundary momenta and full material paths, so
straight-secant gap diagnostics cannot certify physical first passage or provide a symmetric wall-energy ledger.

The actual-path wall diagnostic now reproduces all six ordered pilot payloads exactly with unchanged inputs and
code. On 3,656,249 stochastic chords, geometric straw-wall length is 49,289.3 mm, while production supplies
46,879.7 mm: **2,409.6 mm (4.89% of straw-wall path)** is omitted where that chord has no gas piece. Wall-only
finite-line grazes account for 2,345.8 mm and finite-chord boundary pieces for 63.8 mm. All matched gas-hit wall
lengths agree; an independent all-straw oracle agrees on the first 256 indexed outer-hit chords, covering 376 outer
intersections. That conditional sample does not certify lanes entirely missed by candidate enumeration or shared
geometry. This establishes an accounting omission on the explored paths, **not its contribution to momentum core,
step yield, total material or energy loss**. No correction is applied to these results or the default transport;
an experimental disabled-by-default implementation is described below.

`validation.fieldgasvoids` measures the own-straw finite wire overlap on saved gas endpoint secants without treating
it as the true trajectory. All 30 Geant4-all summaries with more than 1 mm of missing gas length remain above that
threshold after wire/inner-cylinder clipping; 26 remain in the clock-selected arm. The largest residual is about
653 mm, so simple straight-wire overlap does not explain the large summary gaps. Exact outlier identities and all
zero-step particles are preserved. These gaps make endpoint-secant substitution unsafe for a physical material
ledger; they do not identify whether returns, sensitive-step coverage or producer details caused the discrepancy.

The source/export audit reproduces all 816,152 ordered Geant4-all gas-step rows and the complete particle payload
exactly from the original dumper. Every large-gap row is bound to its explicit original particle key and source
row; the gaps were not introduced by injector renumbering/export. The original ART has compressed gas steps and
trajectories but no tracker raw-step collection or step associations. The dumper also discards ProductID, so this
mapping is not native persistent-pointer certification or recovery of the missing constituent history.

The pilot's large gaps occur in 12 Geant4-all events (11 clock-selected). None overlaps the 1,680 paired momentum
proxy events, including all 424 paired 2–5 GeV/c events, or the gained/lost reconstruction events. The pilot core
excess is unchanged. Thus these large-summary gaps do not explain that pilot core finding; no events are removed
from the headline comparison, and confirmation-parent gaps remain a separate question.

A passive frozen-code observer now attributes wall exposure to all 43,156 original entry particles, preserving zeros,
and again reproduces all six saved payloads and the prior global wall ledger. Omission affects 6,662 particles. In
the fixed 424-event paired 2–5 GeV/c pilot sample, all events have one entry particle and 116 have omitted wall:
40.23 mm total, averaging 0.0949 mm per paired track, or 4.82% of that sample's geometric straw-wall path. The
corresponding mean-model collision estimate is 0.0284 MeV per paired track including zeros, evaluated at current
fast-chord momenta. It is **not realized loss, Geant4 energy loss, a response correction or a causal prediction**.
No events are excluded and no revised core is substituted.

Original trajectories are available for all 12 large-gap particles, but their sampled polylines do not supply raw
sensitive-step constituents or an SGS exit clock. Even an available sampled trajectory can miss the gas volume
crossed by a stored gas summary. Physical passage/grouping attribution therefore remains unresolved; the large-gap
finding does not explain the pilot paired-core excess.

### Experimental complete-wall accounting (disabled by default)

`sim.fieldsteps.transport(..., complete_outer_wall=True)` now enables a separate material-only ledger in
`sim.fieldwall`. This Python API option is **off by default and is not exposed by the comparison CLI**. It uses
expanded outer-radius panel bounds with box-only candidates, subtracts finite inner-cylinder intervals, handles caps
and parallel/near-parallel rays without the legacy gas-search angular shortcut, and processes wall-only chords
without creating zero-gas steps. A stable root calculation preserves small nonzero transverse/axial displacements. Its wall columns **replace**, rather than duplicate,
the legacy wall contributions to sampled loss, tail material and cumulative scattering thickness. Wall scattering
uses the actual interval lever moments; gas/wire midpoint treatment and ownership stay unchanged. The homogeneous
Mylar/Al split, material constants, support, loss/scattering samplers, field, chord limits, stops and stock
reconstruction are not changed. Candidate-subtraction batching does not bound the outer search's peak allocations.

Local synthetic tests cover central/grazing/cap/parallel/split/reversed rays, a small all-straw oracle including
missed-gas and empty lanes, interval moments, actual consumer columns, mixed live/dead lanes, stop clipping, and
absence of gas-step inflation. Six pre-edit configurations provide exact ordered step/final-state and RNG receipts
for default and explicit `False`; those byte receipts are platform/dependency-specific. Separate default/False and
enabled repeatability tests run on the current platform. These checks are **not runtime-data or stock-reconstruction
validation**, a large-population memory/performance measurement, or an independent geometry certificate.

Enabled paths and random draws change, so validation needs paired births/seeds and predeclared efficiency, core,
tails and straw-count metrics, not enabled byte identity or guaranteed improvement. Existing CE and cosmic results
above continue to describe the legacy transport, not this candidate. The completed development pilot below measures
its cosmic impact; independent frozen-candidate confirmation and CE validation of the wall option remain undone.
No default promotion or response correction is justified.

### Complete-wall development pilot: accounting corrected, discrepancies not closed

The paired pilot is complete on the previously explored cosmic parent, with identical original entries/seeds,
unchanged stock cosmic digitisation/CentralHelix reconstruction, and retained Geant4/legacy baselines. All 43,156
entry particles, including zeros, remain. The headline efficiency denominator is the unchanged 40,991 reference-clock
step events. No selected ntuple event lies outside it or outside the entry population. This is **development-only,
not untouched confirmation**; no Geant4 generation, baseline reconstruction rerun or default promotion was used.

| Paired metric | Enabled - Geant4 | Legacy - Geant4 | Enabled - legacy |
|---|---:|---:|---:|
| Good-track events | 1,909 vs 1,914 | 1,948 vs 1,914 | 1,909 vs 1,948 |
| Efficiency difference, percentage points | -0.0122 +- 0.0517 | +0.0829 +- 0.0532 | -0.0951 +- 0.0506 |
| Gained / lost events | 222 / 227 | 255 / 221 | 196 / 235 |
| Active hits per paired good track | -0.0261 +- 0.0943 | +0.0768 +- 0.0885 | -0.1051 +- 0.0933 |
| Distinct active straws per paired good track | -0.0261 +- 0.0943 | +0.0768 +- 0.0885 | -0.1051 +- 0.0933 |

The active-straw reader now includes KinKal left, null and right states (`state > -2`), deduplicates by event and
physical straw, and verifies its active-hit row count against persisted `nactive` for every first good track in all
three ntuples. Here those active straws happen to be distinct, so hit/straw differences coincide. The earlier
`state > 0` count was a right-drift-only reporting bug, not an activity cut; that report is superseded for this
metric only. Track selection, momentum, efficiency and all event denominators are unchanged.

**The 2–5 GeV/c momentum-proxy core remains broader.** On 415 enabled/Geant4 paired events, relative IQR sigma is
1.516% versus 1.180%, a **+0.335 +- 0.133 percentage-point** difference. The retained legacy comparison is
+0.367 +- 0.131 points on 424 paired events. These pair cohorts differ: on the 393-event three-way common diagnostic,
the enabled-minus-legacy width change is only **-0.043 +- 0.172 points**. That diagnostic does not replace the
headline populations and provides no demonstrated narrowing. Truth remains each arm's own event-first-straw proxy,
not same-location resolution or certified reconstructed-primary matching.

For the full paired momentum population, enabled-minus-Geant4 absolute relative tail differences above 1%, 5% and
10% are respectively **+2.03 +- 1.37**, **+0.06 +- 1.12** and **+0.78 +- 0.82 percentage points**. Enabled-minus-legacy
at the 10% threshold is **+1.84 +- 0.82 points**; it is a predeclared descriptive finding, not a family-wise rejection
or proof of a mechanism. Momentum cores use 500 shared-index event-bootstrap replicates; mean/tail differences
use paired standard errors. Multiple bins and pair-specific cohorts preclude an equivalence or low-statistics closure
claim.

On the full 43,156-particle exposure, enabled-minus-retained-Geant4-clock differences are **-0.4679 +- 0.0266 step
rows** and **-0.4693 +- 0.0263 distinct straws per entry particle**. Enabled-minus-legacy distinct straws are only
+0.0257 +- 0.0222 per entry particle. These include zero-step entries and are not the historical exiting-track
first-passage denominator. The multiplicity deficit therefore remains open. All 2,161 entry events outside the
headline denominator remain separately accounted for.

Direct enabled consumer checks verify the complete wall columns in loss, tails and scattering. Enabled paths
consume 49,403 mm of wall in total; they need not reproduce the geometric wall length of the stochastic legacy
paths. Two path/chord-capped tracks remain in the ledger without retuning/removal. The exact saved stage history
also explains two step-bearing events rejected by the unchanged strict-positive-energy injection filter: each has
one zero-ionisation row. All accepted event keys agree with the retained injection/digitisation/reconstruction
products and highest-cycle ntuple counts. No event was repaired or removed to make the accounting pass.

**Verdict:** the geometry-accounting omission is addressed in the opt-in implementation, but this pilot does not
establish closure or a reliable reconstruction improvement. Keep it disabled by default. The confirmation-parent
core/step findings, general upstream world transport, device validation and matched end-to-end cost remain open.
