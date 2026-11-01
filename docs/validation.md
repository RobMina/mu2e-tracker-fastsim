# Tracker: validation

What the fast tracker simulation and reconstruction have been checked against. "Commit" is the
`mu2e-tracker-fastsim` commit the check ran on, or this repository's `main` on that date. Open items and what is not
validated at all: [`limitations.md`](limitations.md).

Samples: `g4ref_v1` = the Geant4 reference of the extracted (KPP) configuration (stock `ExtractedCRY`, run 1400,
20 shards of 50,000 generated events, StrawGasSteps dumped by `art/TrkFastSim`); `g4ref_v2` = the larger reference
(10 M generated events, 200 subruns); `kkts_extrap_v04` = extracted cosmic MC digis with stock reco.

## Simulation (StrawGasSteps)

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| T-V0 geometry | Geant4 step entry/exit points | 175,951 Geant4 steps | steps outside their own straw | 0 | 0 | pass | 2026-10-16 | `efba804` |
| T-V1 primary-muon steps, default stage C | Geant4 `g4ref_v1` | 20 shards, 1 M generated | rate per live time; straws per muon; path mean; eIon mean and MPV; deflection medians | -1.2 +- 1.9 %; -0.7 %; -0.44 %; +0.7 %, +1.6 +- 1.6 %; within 0.1 mrad from 0.3 to 5 GeV/c | every T-V1 threshold | pass | 2026-10-20 | `b1f77f2` |
| surface crossings (TT_Front..TT_Outer) | Geant4 virtual-detector steps | `g4ref_v1`, 20 shards | crossings per selected event per surface; momentum, angle, position, time bands | within 1 %; bands 0-3.5 % after 2 sigma | 1 %; 5 % | pass (muon only; see limitations) | 2026-10-21 | `1d6aa07` |
| T-V1b secondaries (library pasting) | Geant4 | shards 11-20 | eIon per event; eIon per step binned; eIon per straw binned | 0.0 %; 1.4 %; 12.4 % | -; 5 %; 5 % | per-straw **fail** | 2026-10-18 | `1f28a9e` |
| any-primary-muon rate (PrimaryFilter tracker term) | Geant4 | 11 M generated | rate | -0.89 +- 0.32 % | within 2 sigma of 0 | **open** (2.8 sigma) | 2026-10-27 | `1f6908e` |

## Digitisation

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| fast digis (NumPy) | stock `StrawDigisFromStrawGasSteps` on the same steps, floor = stock with another seed | 27,779 events | digis per event (paired); F2 good fits; panel, straw, TDC, TOT, ADC histograms | -0.052 +- 0.031 % (floor -0.007 +- 0.031 %); -0.045 +- 0.119 %; no histogram flagged | flag above max(2, 3 x floor) | pass | 2026-10-27 | `9aedb18` |
| JAX digitiser | NumPy digitiser | real tables, ten groups, A100 | output identity per backend; speed | repeated arrays identical; warm 4.2x NumPy | statistical equivalence | pass (opt-in) | 2026-10-30 | `62810cc` |
| digi-table producer (art) | stock digis | 2,866 MC + 500 KPP events | detector round trip; exported reco | exact; reco products identical except stock `CosmicTrackSeeds`, which are not reproducible stock vs stock | byte equality | pass with one stock exception | 2026-10-30 | `cd3d79d` |

## Reconstruction

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| T-VR1 straw hits, combo hits | stock `StrawHitReco`, `CombineStrawHits` on the same digis | 1.62 M `kkts_extrap_v04` digis | hit and combo lists | identical (NumPy; JAX twins bitwise equal to NumPy) | identical | pass | 2026-10-21 | `4186c1f` |
| T-VR1b time clusters, line seeds | stock `SimpleTimeCluster`, `LineFinder` | 86,981 events | clusters, seeds | 70,730 clusters and 67,239 seeds identical, seed parameters bitwise | identical | pass | 2026-10-22 | `54d4b35` |
| F1 line fit (KinKal-port, stock-conditioned) | stock KKLine | 2,151 fits | acceptance agreement; net yield; geometry within 0.1 stock sigma | 95.6 % (59 gains, 43 losses); +0.74 +- 0.47 %; 96.5-97.7 % per parameter | 99.5 %; -; 99 % | **open** | 2026-10-27 | `1f6908e` |
| F2 fit (fixed momentum, no material) | stock KKLine | digitisation sample above | good-fit yield | -0.045 +- 0.119 % vs stock fits on stock digis | statistical | statistical use only | 2026-10-27 | `9aedb18` |
| T-V2 occupancy on fast steps | full Geant4 chain, G4 even/odd floors | `g4ref_v2`, 10 M generated, paired | panel occupancy (event-clustered covariance) | within floors | floor | pass | 2026-10-27 | `da8df42` |
| T-V3 reco yield on fast steps | full Geant4 chain | same, paired | track yield; digis | -2.04 +- 0.37 % (mostly selection-only); -1.09 +- 0.33 % | 1 % | **fail** | 2026-10-27 | `6d67cf5` |

## Performance

| Goal | Target | Result |
|---|---|---|
| T-P1 step generation | <= 3 ms per accepted event, one core | about 5 ms (tracker interior): not met |
| T-P2 step stage on one A100 | >= 50x one core | 4.9-5.7x (world column on the GPU): not met |
| hits to seeds | | 1,244 clusters/s on one core; 47,000 on a 2g.20gb MIG slice |
| F2 line fit | | 2,676 tracks/s on one core; 57,000 on a MIG slice (stock KKLine about 26 tracks/s) |

## Reproduce the T-V1 result

```bash
tracker/scripts/stagec_msc_ab.sh default        # steps on the 20 g4ref_v1 shards, then v1compare, worldcheck, msccheck
python3 tracker/scripts/stagec_msc_ab_summary.py "$M2FS_DATA/trk_fastsim" c1SIsD
```

It reads the gen and StrawGasStep tables of `g4ref_v1` under `$M2FS_DATA/trk_fastsim` and writes
`plots/v1compare_c1SIsD/v1compare.json` with every threshold decision. An arm reproduces its numbers only with the
code it ran with: check out that tracker commit and the core commit pinned beside it in the umbrella history.
