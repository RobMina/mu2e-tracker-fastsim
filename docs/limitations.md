# Tracker: limitations

## Open discrepancies

- **Any-primary-muon rate** (the PrimaryFilter tracker term) is -0.89 +- 0.32 % against Geant4 on 11 M events (2.8
  sigma); the cause (support-structure loss, gas scattering, exit scatter, library placement) is not found.
- **T-V3**: the reco yield on fast steps through stock reco is -2.04 +- 0.37 % (mostly selection-only) and digis
  -1.09 +- 0.33 % against the full Geant4 chain; threshold 1 %.
- **Secondary eIon per straw** (T-V1b): binned ratio 12.4 % against a 5 % threshold.

## Frozen

**F1 native closure** (frozen 2026-11-01). The F1 fit (KinKal port, stock-conditioned) stays opt-in and does not meet
its targets; no closure work is planned. Nothing depends on it: `tracker_point` uses F2 and T-V3 uses stock reco.

- **Acceptance** against stock KKLine on 2,315 eligible single-seed fits of 3,000 `kkts_extrap_v04` events (2,151 stock
  good), target 99.5 % agreement, that is at most 11 disagreements:
  - `--time-model legacy` (the default): 94.2 %, 113 gains and 21 losses, net yield +4.28 +- 0.54 %.
  - `--time-model stock` (used by `reco.autonomous`): 95.59 +- 0.43 %, 59 gains and 43 losses, net yield
    +0.74 +- 0.47 %. Most losses are extension ParamsDiverged.
- **Fixed stock hit states**, within 0.1 stock sigma on both-good tracks, target 99 %: d0 97.7, phi0 97.0, z0 96.5,
  theta 96.5, t0 99.9, momentum 42.6 % (89.8 % with the captured native seed, an opt-in diagnostic).
- **Yield** agrees better than single tracks do: the detector-only chain on the same MC digis gives +0.35 +- 0.44 %;
  500 uncalibrated KPP events give -16.0 +- 8.4 % ([autonomous-f1.md](autonomous-f1.md)).
- **Ruled out as the cause:** each of these changed no acceptance decision in a full-sample arm: piece-boundary and
  reference search, construction-fixed crossing offsets, the native constraint time, native seed momentum, covariance
  and range, and the per-material gas scattering fraction.
- **Not tried, the place to restart:** native conventions that were found by operand comparison but never put in the
  fit, together in one arm: the muon mass (105.658367 against 105.6583755 MeV), the native time and longitudinal
  residual derivatives (`kalman_jax.py` differentiates the full residual), the material derivative at the crossing's
  offset time, and the source signal-speed coefficients. Also unresolved: a strict replay of native effect algebra
  fails on 862 parameter vectors, and 31 of the 102 disagreements have no captured native history.
- **Possible limit:** some gains are stock ParamsDiverged failures on ill-conditioned covariances, which a looser
  momentum constraint nearly removes. Matching those may mean reproducing near-singular arithmetic, so 99.5 % may not
  be reachable by an independent fit. This is an inference, not a measurement.

The closure tooling is on branch `park/f1-native-closure`. The fit outputs behind these numbers were not kept: a
restart first regenerates the stock dump and the baseline arms from those 3,000 events.

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
