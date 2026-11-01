# Tracker: architecture

## The chain

The fast tracker chain replaces Geant4, the stock tracker digitisation and the stock no-field track reconstruction
of the extracted (KPP) cosmic configuration. It reads the same CRY primaries as the CRV stage, so a tracker-CRV study
sees one cosmic sample. Every stage writes flat tables keyed by `(run, subrun, event)`, so it can be swapped for its
stock counterpart and compared on the same events.

```
CRY primaries (gen tables, shared with the CRV stage)
     │
     ▼
world transport        GDML world above the tracker in height bands: sliced energy loss, correlated multiple
     │                 scattering to a plane above the straws
     ▼
tracker interior       loss in the support structure (rings, ribs, beams, manifolds, electronics), kinks every
     │                 support segment and every 1e-3 X0 of straw material, single-scattering tail, gas eIon
     ▼
StrawGasStep table  ◄── optional: secondary steps pasted from a Geant4 library
     │              ──► art: StrawGasStepsFromTable injects it into stock Offline (path A)
     ▼
digitisation           port of StrawDigisFromStrawGasSteps (NumPy; JAX opt-in) -> StrawHitTableDumper-layout digis
     │              ──► art: StrawDigisFromTable injects digis into stock reco
     ▼
reconstruction         straw hits, combo hits (ports of StrawHitReco, CombineStrawHits) -> time clusters, line seeds
     │                 (SimpleTimeCluster, LineFinder) -> line fit: F2 (fixed momentum) or F1 (KinKal port, opt-in)
     ▼
EventNtuple            stock branch names and types (RNTuple), docs/eventntuple.md
```

## Backends

NumPy is the reference and the default everywhere. JAX twins: `steps --backend jax` (world, interior, crossings,
gas eIon on the device; generator selection, library pasting and table assembly stay NumPy), `digi --backend jax`,
the hit/combo twins and the F1 fit. JAX output is reproducible per seed on one device and lane count; it is not a
paired draw with NumPy.

## Who owns what

| Area | Where | Notes |
|---|---|---|
| physics primitives | core `mu2efastsim` (energy loss, materials, scattering, world, crossings, tables) | shared with CRV and beam |
| straws, geometry, crossings | `trkfastsim` geometry and crossings modules | straws from the GDML (StrawSD ids, radii, wire) |
| step generator | `trkfastsim` steps (+ JAX twins), secondary library | stage A (straight) and stage C (default) |
| digitisation | `trkfastsim` digi and digi-conditions modules | conditions from the stock job's sources |
| reconstruction | `trkfastsim/reco/` | hit reco, combos, line finding, F1/F2, EventNtuple writer |
| comparisons | `trkfastsim/analyses/` (one module per CLI subcommand) | T-V0/T-V1/T-V2/T-V3, digi and reco checks |
| art modules | `art/TrkFastSim/` | dumpers and table-to-art injectors (built in a muse satellite area) |
| Geant4 references | `grid/`, `art/*.sh` | gen-only and T-V2 grid jobs, reference dumps |
| endpoints | the `tracker_point` kit task and its Slurm/EAF runners | one shard per job |

## Validation design

Each stage is checked against its stock counterpart on the same input: T-V0 geometry, T-V1 primary-muon steps vs
Geant4, digis vs stock digitisation on the same steps (with a reseeded stock floor), T-VR1/T-VR1b hits, clusters and
seeds on the same digis, F1/F2 fits vs stock KKLine, and T-V2/T-V3 the full chain vs Geant4 + stock. See
[validation.md](validation.md).
