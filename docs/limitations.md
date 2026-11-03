# Tracker: limitations

## Open discrepancies

- **Any-primary-muon rate** (the PrimaryFilter tracker term), default: -0.82 +- 0.26 % against Geant4 (11 M events,
  mean of three generator seeds). Cause: the default interior transport scatters a muon in the support only in front
  of a straw it is about to cross and drops a line that crosses no further straw, so support scattering moves muons
  off the straws but never onto them. In a subset where Geant4 keeps every event (generated line at least 300 mm
  inside the CRV, 306,349 muons) 2.5 % fewer fast muons reach any straw, -9 % for directions nearly perpendicular to
  the tracker axis (|u_z| < 0.05, where the 10 mm straw bands of a plane make reach depend on the tilt), from 0.5 to
  5 GeV/c and not above. The opt-in `--scatter-in` (`make_steps(scatter_in=True)`) kinks such lines through the
  support ahead and searches their straws again: rate +0.14 +- 0.26 %, reach -0.1 +- 0.4 %. It costs about 35 % more
  wall time per 50,000-event file. The rate gain it brings differs between seeds by more than the paired error
  (+1.34, +0.50, +1.05 %, each +- 0.14 %); not understood. Below 750 MeV/c at the first straw -2.3 +- 1.1 % remains.
- **T-V3**: the reco yield on fast steps through stock reco is -2.04 +- 0.37 % (mostly selection-only) and digis
  -1.09 +- 0.33 % against the full Geant4 chain (default steps; reproduced on the saved products 2026-11-01);
  threshold 1 %. Not rerun with `--scatter-in`: that needs new fast steps through stock digitisation and reco.
- **Secondary content and placement** (T-V1b, paired per-shard statistic on 10 M events). The 12.4 % per-straw eIon
  band of 2026-10-18 was the legacy metric's noise: Geant4 against itself gives 5.6 % on that sample
  ([shard-statistics.md](shard-statistics.md)). Under the paired statistic eIon per straw is just resolved (1.83 +- 0.47
  against a limit of 1.81, rms excess 1.4 %). Its causes:
  - The off-axis placement: 4.4 +- 1.7 % fewer secondary-only straws, with same-plane steps +25 % and steps one or more
    planes away -27 to -29 %. This is the main cause. Opt-in `--offaxis-plane-sign` (plane offsets follow the muon's
    direction along the planes) leaves +18 / -18 / -21 %; opt-in `--offaxis-shift-from-muon` keeps the 2.7 % of
    off-axis steps that the default drops when they land in a fast muon straw (secondary-only straws -1.5 +- 1.7 %).
    About half of the placement excess remains; its cause is not identified.
  - Straws with only a muon step: their path-length population differs (chords above 20 mm +25 %, 8-11 mm -5 to
    -8 %), their eIon per mm agrees. The in-straw draw keys the path on bins whose last one starts at 8 mm, so long
    chords along the wire get too few in-straw secondaries. Opt-in `--instraw-fine-length` (keys up to 30 mm) removes
    the flag (path 2.21 -> 1.31, limit 1.81). Chords below 1.25 mm stay 3-4 % low (not a library effect).
  - The library builder keeps other primary muons of a donor event as secondaries (0.29 % of other steps), which the
    fast transport already carries. Opt-in `steps --library-drop-muons` leaves them out. With it the species flag goes
    and eIon per straw is no longer flagged (1.78 +- 0.47).
  - Other steps per event are -3.2 +- 1.1 % (-2.3 +- 1.1 % with the four options). Through stock digitisation the
    per-straw content shows in straw occupancy, TOT, PMP and hit energy (rms 1.2-1.9 %); panel occupancy passes. With
    the four options the fast-digitiser proxy no longer flags TOT, PMP, hit energy or dt; straw occupancy, wire distance
    and the digis-per-event shape stay flagged. Summary in [secondary-validation.md](secondary-validation.md).

## Frozen

**F1 native closure** (frozen 2026-11-01). The F1 fit (KinKal port, stock-conditioned) does not meet its targets; no
closure work is planned. Nothing depends on it: `tracker_point` uses F2 and T-V3 uses stock reco. Its code is on branch
`park/f1-native-closure-20261103` (see Parked on branches); the numbers below were measured with it.

- **Acceptance** against stock KKLine on 2,315 eligible single-seed fits of 3,000 `kkts_extrap_v04` events (2,151 stock
  good), target 99.5 % agreement, that is at most 11 disagreements:
  - `--time-model legacy` (its default): 94.2 %, 113 gains and 21 losses, net yield +4.28 +- 0.54 %.
  - `--time-model stock` (used by `reco.autonomous`): 95.59 +- 0.43 %, 59 gains and 43 losses, net yield
    +0.74 +- 0.47 %. Most losses are extension ParamsDiverged.
- **Fixed stock hit states**, within 0.1 stock sigma on both-good tracks, target 99 %: d0 97.7, phi0 97.0, z0 96.5,
  theta 96.5, t0 99.9, momentum 42.6 % (89.8 % with the captured native seed, an opt-in diagnostic).
- **Yield** agrees better than single tracks do: the detector-only chain on the same MC digis gives +0.35 +- 0.44 %;
  500 uncalibrated KPP events give -16.0 +- 8.4 %.
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
- No magnetic field in the default chain: straight-line transport and a line fit, for the extracted configuration
  only. The opt-in field-on transport starts at the tracker envelope (no world transport in the field), has no
  radiative muon loss, and is followed by stock digitisation and reco only ([field-on.md](field-on.md)).
- Digitisation: no on-spill folding, ghosts, cross talk, noise hits, TrackerStatus suppression, digi mixing or
  field-on drift (refused by `digi-cond`).
- The EventNtuple writer does not write the MC branches (`trkmcsim`, `trksegsmc`, `trkhitsmc`, `evtinfomc`), CRV and
  DS surfaces in `trksegs`, or `trk.nseg`; ROOT C++ reading of its RNTuple is not checked ([eventntuple.md](eventntuple.md)).
- F2 has a fixed momentum (2000 MeV/c) and no material: for statistical use only.
- `tracker_point` is a fixed extracted-v04, primary-only integration task (registered for EAF and Slurm, not a live
  rollout).

## Not validated

- `steps --backend jax`: validate each geometry and configuration with `jaxcheck` ([device-world.md](device-world.md)).
- **NumPy output across host CPU types.** The NumPy chain repeats byte for byte per seed on one host CPU type, not
  across types: for the same seed and shards the steps (and everything after them) from an AMD EPYC 7742 host differ
  from those of an Intel Xeon Gold 6248 host, while an AMD EPYC 9575F host agrees with the Xeon (4 of 4 shards each).
  The likely cause is NumPy's CPU-feature dispatch of its vectorised math (the two agreeing hosts have AVX-512);
  distributions agree. A saved NumPy reference is reproducible only on a host of the same kind.
- **Device chain identity** ([device-chain.md](device-chain.md)): JAX steps and digis are other random streams
  (compared as distributions); the hit and combo twins equal NumPy byte for byte on CPU but not on a GPU (contracted
  multiply-adds); line seeds and F2 differ between CPU and GPU in the last bits; the step-kernel settings of the
  device chain move step floats at the 1e-6 mm level against the `steps --backend jax` defaults.
- Performance goals T-P1 (3 ms per accepted event on one core; about 5 ms) and T-P2 (50x on one A100; 4.9-5.7x) are
  not met.
- **Accumulated path after a gas-exit move.** A kink restart point that lies inside a straw's gas is moved back along
  the new direction to just outside it (`_exit_gas_backwards`, NumPy and JAX), and that move is not added to the
  accumulated `path`. Straw entry positions are unaffected; `path` (and the step time from it) of later rows of that
  muon is short by the move. In `tests/test_steps_jax.py::test_panel_cut_rows_exact` (1e12 MeV/c, so no deflection)
  12 of 3,808 restarts are moved, by 1e-4 to 0.27 mm, and every one of them lies within 1e-12 mm of a gas surface, so
  whether it counts as inside is decided by rounding. One of them is moved by 2.26e-3 mm: a host whose XLA CPU code
  rounds it the other way gives the 2.2e-3 mm path difference seen on one GitHub runner, while the entry positions
  agree to 1e-9. The test tolerance on `path` is 1e-2 mm.

## Parked on branches

Options that were tried and not made the default are kept on branches, not on main:

- `park/offaxis-width-match`: the width-matched off-axis donor key (`steps --offaxis-width-match`), rejected because
  every placement histogram got worse.
- `park/ordered-navigation-20261103`: ordered stored-material navigation (`steps --navigation ordered-*`, the
  raised-plane forward handoff). It passed its contract tests, but the full-geometry gate refused at cross-material
  overlaps, so it never became the default.
- `park/f1-native-closure-20261103`: the F1 fit (`reco-vr2f1`, `--time-model legacy|stock`, `reco-ntuple --fit f1`),
  the detector-only `reco.autonomous` chain, the F1 validity and fit-trace diagnostics and `kit.kitf1`. Frozen
  below its targets (above). `reco-ntuple --fit stock|f2` and the EventNtuple writer stay on main.

## Inputs

Gen tables, the GDML export, the secondary library and the digitisation/reco conditions are Mu2e-internal and not
published as a collaborator release yet (umbrella `docs/running.md`).
