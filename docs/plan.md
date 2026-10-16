# Tracker fastsim and fastreco: technical plan

Technical scope only; physics results live outside this repository.

## Fast simulation

- Straight-line transport of the CRY primary muon (shared with the CRV stage) through the KPP straws; one
  `StrawGasStep` row per particle per straw: strawId, step type, eIon, path length, width (0, no field), time,
  start/end position, mean momentum, SimParticle key.
- Geometry from an Offline `Tracker` dump with the KPP panel map; the sim alignment (`AlignedTrackerSim`) and the
  reco alignment (`AlignedTracker`) are separate, independently configurable inputs.
- Energy loss in the gas from `mu2efastsim.eloss` (Urban model, tracker region cut 0.001 mm); secondaries from a
  library built from one Geant4 run.
- Generator filter: the production `PrimaryFilter` OR (>= 12 good `StrawGasStep`s from one particle, step
  `ionizingEdep` > 1 eV and 50 MeV/c < |p| < 1 TeV/c; or >= 1 `CrvStep` with visible energy > 0.1 MeV).
- Truth tables for EventNtuple: SimParticle (shared tree with the CRV), GenParticle, MCTrajectory points of the
  primary muons, SurfaceSteps at the tracker and CRV surfaces; empty virtual-detector and calorimeter collections.
- Path A: an art module turns the tables into the stock products, so stock digitisation, reco and EventNtuple run
  unchanged.

## Fast reconstruction

- Digitisation, two prototypes compared: D-A, a JAX port of `StrawDigisFromStrawGasSteps`; D-B, a direct step -> hit
  model.
- Hit reco, combo hits, time clustering, line seed: JAX ports of `StrawHitReco`, `CombineStrawHits`,
  `SimpleTimeCluster`, `LineFinder`, starting on existing digis.
- Line fits, prototyped and compared on the same hits and seed: F1 faithful KinKal port; F2 batched Gauss-Newton with
  drift residuals; F3 deterministic annealing; F4 wire-only fit plus drift refit; F5 (stretch) learned ambiguity
  classifier plus F2.
- Outputs in EventNtuple `trk`/`trksegs`/`trkhits` names and units.

## Goals and validation

Agreed 2026-10-16: T-P1..T-P7, T-V0..T-V3 (simulation) and T-R1..T-R6, T-VR1..T-VR4 (reconstruction), with the
thresholds of the project plan.
