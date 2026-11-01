# Tracker: limitations

## Open discrepancies

- **Any-primary-muon rate** (the PrimaryFilter tracker term) is -0.89 +- 0.32 % against Geant4 on 11 M events (2.8
  sigma); the cause (support-structure loss, gas scattering, exit scatter, library placement) is not found.
- **T-V3**: the reco yield on fast steps through stock reco is -2.04 +- 0.37 % (mostly selection-only) and digis
  -1.09 +- 0.33 % against the full Geant4 chain; threshold 1 %.
- **Secondary eIon per straw** (T-V1b): binned ratio 12.4 % against a 5 % threshold.
- **F1** (KinKal-port fit, stock-conditioned) agrees with stock KKLine acceptance in 95.6 % of tracks against a 99.5 %
  target. The native-closure tooling is on branch `park/f1-native-closure`.

## Not implemented

- Only the primary muon is transported; other particles' steps come only from the optional Geant4 secondary library,
  and surface crossings (`surfsteps`) are written for the primary muon only (Geant4 has 52-72 % non-muon virtual
  detector crossings).
- No magnetic field: straight-line transport and a line fit, for the extracted configuration only.
- Digitisation: no on-spill folding, ghosts, cross talk, noise hits, TrackerStatus suppression, digi mixing or
  field-on drift (refused by `digi-cond`).
- The EventNtuple writer does not write the MC branches (`trkmcsim`, `trksegsmc`, `trkhitsmc`, `evtinfomc`), CRV and
  DS surfaces in `trksegs`, or `trk.nseg`; ROOT C++ reading of its RNTuple is not checked ([eventntuple.md](eventntuple.md)).
- F2 has a fixed momentum (2000 MeV/c) and no material: for statistical use only.
- The detector-only F1 path (`reco.autonomous`) is a prototype ([autonomous-f1.md](autonomous-f1.md)).
- `tracker_point` is a fixed extracted-v04, primary-only integration task (registered for EAF and Slurm, not a live
  rollout).

## Not validated

- Ordered stored-material navigation (`steps --navigation ordered-*`) passes its contract tests; the full-geometry
  gate refuses at cross-material overlaps ([ordered-stored-path.md](ordered-stored-path.md)).
- `steps --backend jax`: validate each geometry and configuration with `jaxcheck` ([device-world.md](device-world.md)).
- Performance goals T-P1 (3 ms per accepted event on one core; about 5 ms) and T-P2 (50x on one A100; 4.9-5.7x) are
  not met.
- The `panel cut rows` path test agrees to 1e-13 mm locally and on UVA but to 2.2e-3 mm on the CI runner (tolerance
  10 um); not explained yet.

## Inputs

Gen tables, the GDML export, the secondary library and the digitisation/reco conditions are Mu2e-internal and not
published as a collaborator release yet (umbrella `docs/running.md`).
