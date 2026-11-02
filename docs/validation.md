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
| T-V1b per-straw band, legacy metric reproduced | Geant4 `g4ref_v1`, and Geant4 and fast against themselves | shards 11-20 (1-20) | eIon per straw binned ratio, minus 2 sigma | fast seed 1 / seed 2: 7.2 % / 2.0 % (0.1 % on 1-20); Geant4 1-10 vs 11-20: 5.6 %; fast seed 1 vs 2: 1.0 % | 5 % | metric not calibrated: Geant4 fails against itself ([shard-statistics.md](shard-statistics.md)) | 2026-11-02 | `6ee2a15` |
| T-V1 paired per-shard, default steps | Geant4 `g4ref_v2` | 10 M generated, 200 subruns, seeds 1 and 2 | 23 step histograms, shape mode, family-wise 1 % | primary-muon step histograms (eIon, path, \|p\|, time, plane, panel, straw) not flagged; rate -0.90 +- 0.33 % / -1.23 +- 0.32 %; flags: secondary placement and multiplicity (below) | per-histogram limit | muon steps pass; secondaries open | 2026-11-02 | `6ee2a15` |
| T-V1b paired per-shard, default steps | as above | as above | eIon per straw (shape) chi2/ndf (limit); rms excess; secondary-only straws; other steps per event | 1.83 +- 0.47 (1.81), 1.4 %; -4.4 +- 1.7 %; -3.2 +- 1.1 %; same plane +25 %, 1-2 planes -27 %, >= 3 planes -29 % (each +- 2-3 %) | per-histogram limit | per-straw just resolved; placement **open** | 2026-11-02 | `6ee2a15` |
| T-V1b, opt-in `--library-drop-muons` | as above | as above, seed 1 | as above; library muon steps | 0.29 % of other steps removed (all in muon straws); species flag 255 -> 2.7 (limit 7.3); eIon per straw 1.78 +- 0.47 (1.81) | per-histogram limit | per-straw not flagged (opt-in) | 2026-11-02 | `6ee2a15` |
| nulls of the paired statistic | fast against fast (seed 1 vs 2); split shards of Geant4 and fast | `g4ref_v2`, `g4ref_v1` | flags; pull rms over all bins | 0 of 23 (both samples); 0 split flags; pull rms 1.001 +- 0.005 (`g4ref_v2`), 1.005 +- 0.005 (`g4ref_v1`) | 0 flags | pass | 2026-11-02 | `6ee2a15` |
| any-primary-muon rate (PrimaryFilter tracker term) | Geant4 | 11 M generated | rate | -0.89 +- 0.32 % | within 2 sigma of 0 | **open** (2.8 sigma) | 2026-10-27 | `1f6908e` |
| same, three generator seeds | Geant4 | 11 M generated, seeds 1-3 | rate | default -0.91, -0.55, -1.00 % (each +- 0.32 %); mean -0.82 +- 0.26 % | within 2 sigma of 0 | **open** (default) | 2026-11-01 | `e648e66` |
| same, opt-in `--scatter-in` | Geant4 | 11 M generated, seeds 1-3 | rate; muons reaching >= 1 straw in a subset where Geant4 keeps every event | +0.42, -0.05, +0.04 % (each +- 0.32 %), mean +0.14 +- 0.26 %; reach -2.5 % -> -0.1 % (+- 0.4 %) | within 2 sigma of 0 | pass (opt-in) | 2026-11-01 | `e648e66` |

## Digitisation

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| fast digis (NumPy) | stock `StrawDigisFromStrawGasSteps` on the same steps, floor = stock with another seed | 27,779 events | digis per event (paired); F2 good fits; panel, straw, TDC, TOT, ADC histograms | -0.052 +- 0.031 % (floor -0.007 +- 0.031 %); -0.045 +- 0.119 %; no histogram flagged | flag above max(2, 3 x floor) | pass | 2026-10-27 | `9aedb18` |
| fast digis (NumPy), paired per-shard | stock digitisation of the same steps (saved default-L steps of `g4ref_v2`) | 55,432 events, 200 subruns, fast seeds 1 and 2 | 19 digi, hit and F2 histograms; multiplicities | 0 of 19 flagged (both seeds; seed 1 vs 2: 0); digis per event +0.03 +- 0.04 % / +0.14 +- 0.05 % (seed 1 vs 2: -0.11 +- 0.04 %); combo hits +0.00 +- 0.05 % / +0.08 +- 0.05 %; F2 fits per event +0.01 +- 0.09 % / -0.02 +- 0.09 % | per-histogram limit | pass | 2026-11-02 | `6ee2a15` |
| JAX digitiser | NumPy digitiser | real tables, ten groups, A100 | output identity per backend; speed | repeated arrays identical; warm 4.2x NumPy | statistical equivalence | pass (opt-in) | 2026-10-30 | `62810cc` |
| digi-table producer (art) | stock digis | 2,866 MC + 500 KPP events | detector round trip; exported reco | exact; reco products identical except stock `CosmicTrackSeeds`, which are not reproducible stock vs stock | byte equality | pass with one stock exception | 2026-10-30 | `cd3d79d` |

## Reconstruction

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| T-VR1 straw hits, combo hits | stock `StrawHitReco`, `CombineStrawHits` on the same digis | 1.62 M `kkts_extrap_v04` digis | hit and combo lists | identical (NumPy; JAX twins bitwise equal to NumPy) | identical | pass | 2026-10-21 | `4186c1f` |
| straw and combo hits, per shard | stored stock `StrawHitReco` / `CombineStrawHits` hits of the same digis | stock digis of the saved L steps and of `g4ref_v2`, 200 subruns each | wire distance, energy, hits and combo hits per event | identical in every shard | identical | pass | 2026-11-02 | `6ee2a15` |
| F2 against stock KKLine, same stock digis | stock KKLine (status > 0) | 55,432 events, 200 subruns | good fits; active hits; chi2/ndf | F2 +6.07 +- 0.11 % more fits (F2 on fast digis +6.08 +- 0.13 %); active hits per fit -3 %; chi2/ndf flagged | statistical | F2 is statistical-use only | 2026-11-02 | `6ee2a15` |
| T-VR1b time clusters, line seeds | stock `SimpleTimeCluster`, `LineFinder` | 86,981 events | clusters, seeds | 70,730 clusters and 67,239 seeds identical, seed parameters bitwise | identical | pass | 2026-10-22 | `54d4b35` |
| F1 line fit, `--time-model legacy` (default; KinKal-port, stock-conditioned) | stock KKLine | 2,315 fits, 2,151 stock good | acceptance agreement; net yield; geometry within 0.1 stock sigma | 94.2 % (113 gains, 21 losses); +4.28 +- 0.54 %; 96.6-97.8 % per parameter | 99.5 %; -; 99 % | **not met** (frozen, see limitations) | 2026-10-27 | `1f6908e` |
| F1 line fit, `--time-model stock` (opt-in) | stock KKLine | same | same | 95.6 % (59 gains, 43 losses); +0.74 +- 0.47 %; 96.5-97.7 % per parameter | 99.5 %; -; 99 % | **not met** (frozen, see limitations) | 2026-10-27 | `1f6908e` |
| F2 fit (fixed momentum, no material) | stock KKLine | digitisation sample above | good-fit yield | -0.045 +- 0.119 % vs stock fits on stock digis | statistical | statistical use only | 2026-10-27 | `9aedb18` |
| T-V2 occupancy on fast steps | full Geant4 chain, G4 even/odd floors | `g4ref_v2`, 10 M generated, paired | panel occupancy (event-clustered covariance) | within floors | floor | pass | 2026-10-27 | `da8df42` |
| T-V2 stock digis, paired per-shard | stock digitisation of Geant4 steps | saved default-L vs `g4ref_v2`, 200 subruns | 17 digi and hit histograms, shape mode | panel occupancy 1.17 (limit 1.39); digis per selected event -1.09 +- 0.44 %; flagged with rms 1.2-1.9 %: straw occupancy, TOT, PMP, dt, hit energy and wire distance | per-histogram limit | panel occupancy pass; per-straw content open | 2026-11-02 | `6ee2a15` |
| T-V3 reco yield on fast steps | full Geant4 chain | same, paired | track yield; digis | -2.04 +- 0.37 % (mostly selection-only); -1.09 +- 0.33 % | 1 % | **fail** | 2026-10-27 | `6d67cf5` |
| T-V3, saved products rescored | as above | same | track yield; digis | -2.04 +- 0.37 %; -1.09 +- 0.33 % (reproduced; default steps, without `--scatter-in`) | 1 % | **fail** | 2026-11-01 | `ff2204b` |
| T-V3, current steps, same 20 groups through the same stock digitisation and reco (UVA) | as above | same, paired | track yield; digis | default: -1.42 +- 0.37 %; -0.85 +- 0.33 %. `--scatter-in`: -0.85 +- 0.37 %; -0.85 +- 0.33 %. `--scatter-in` vs default (paired): yield +0.58 +- 0.21 %, digis +0.01 +- 0.29 % | 1 % | default **fail**; `--scatter-in` pass on central values (opt-in) | 2026-11-02 | `c36bc3f` |

The paired per-shard rows use one rule for every histogram: the variance of the per-subrun difference, a sign-flip
null and a family-wise 1 % limit per comparison ([shard-statistics.md](shard-statistics.md)).

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
