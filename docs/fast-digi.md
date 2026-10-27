# Fast tracker digitisation

`trkfastsim/digi.py` is a vectorised numpy port of Offline `StrawDigisFromStrawGasSteps` (TrackerMC, Offline
v13_39_00, the release behind SimJob MDC2025ay) for the extracted (KPP) configuration. It turns StrawGasStep tables
into StrawDigi tables in the `StrawHitTableDumper` layout, so the fast hit reco (`trkfastsim.reco.hitreco`, `combo`)
and the fast line finding and fit run on them unchanged. `trkfastsim/digicond.py` builds its conditions from the
sources the stock job reads.

## Conditions: same sources as the stock job

| Item | Source |
|---|---|
| StrawPhysics (ionisation table, cluster sizes, gain, drift spread, propagation velocity) | `fhicl-dump` of the digitisation job, `ProditionsService.strawPhysics`, derived as `StrawPhysicsMaker` |
| Drift distance to time | `strawDrift` table, integrated as `StrawDriftMaker` (field-free: no Lorentz angle) |
| Electronics response (current pulse, preamp/ADC/saturation filters, impulse response, reflection) | `strawElectronics` table, computed as `StrawElectronicsMaker` (bilinear IIR, 0.1 ns bins) |
| Thresholds per straw end, ADC gain (`dVdI`), ADC pedestals | DbService `TrkPreampStraw` of the run (`dbTool print-run ... --content`), as `StrawElectronicsMaker::fromDb` |
| Channel time offsets | `TrkDelayPanel`, `TrkDelayRStraw`, `TrkPreampStraw`; not applied when `overrideDbTimeOffsets` is true (the case here) |
| Straws and wires | nominal straws from the GDML export (`trkfastsim.geometry`) plus the `TrkAlignStrawSim` wire-end offsets in the panel UVW frame, as `alignedTrackerSim` with `onlyWireOffsets` |
| Event window, spill type | `eventTiming`, `EWMProducer.SpillType`, GlobalConstantsService `physicsParams` |
| makeSD settings | the job's `physics.producers.makeSD` (AllHitsPlanes, AllHitsStraw) and the module defaults |

`digi-cond` refuses configurations it does not implement (on spill, magnetic field, cross talk, noise,
DriftSingleElectrons, UseStatus, digi mixing). The response tables reproduce the stock job's own printout
(`strawElectronics.verbose`) to its printed precision (`tests/test_digi.py`).

## What is reproduced and what is approximated

| Stage | Stock | Fast |
|---|---|---|
| Step selection | eIon > minstepE, time window (digitisation start - buffers, event length + buffer) or read-all straws | same |
| Ionisation clusters | point steps and steps shorter than the mean free path: one cluster, Poisson electrons; else Poisson cluster count (capped at 20 for non-minion, at eIon / E1), uniform along the chord plus 2-D Gaussian width; minion: cluster sizes by `fillClusterMinion` (rejection loops), residual energy as extra ionisations on random clusters, every electron drifted on its own | same distributions; the rejection loops are sampled as the equivalent truncated distributions |
| Straw coordinates | truncation to the straw inner radius, radius and phi about the sim-aligned wire | same |
| Gain | Polya (gamma, shape A or n(A+1)-1) below nGainGauss electrons, Gaussian above | same |
| Drift | `D2T` table, Gaussian spread 1.3 + 0.5 r ns | same (phi-free, B = 0) |
| Propagation | half the charge to each end, wire distance / 299 mm/ns | same |
| Waveform | sum of cluster responses, 0.1 ns response bins, wire-distance interpolation, end reflection, 5 ns look-back | same; the tables are float32 and the bin index is computed in float32 (stock: double) |
| Threshold crossing | per end: rough crossing (sum of maximal responses), first cluster whose peak passes, regula falsi to 0.02 ns, return below threshold; coherent straw noise on the first threshold, incoherent per crossing; ends combined within 2 ADC clocks; 100 ns analog dead time | same algorithm, run in lock step over all straw ends |
| TDC | crossing + per-panel event-window-marker jitter (0.5 ns) + TDC jitter (0.1 ns), 24-bit, LSB 1/51.2 ns; window: TDC > start TDC and < event-window end (off spill); read-all straws: positive time | same |
| TOT | waveform below threshold (re-smeared) - hysteresis, 5 ns ticks, max 15 | same |
| ADC | 15 samples at 20 ns, 5 presamples, phase from the cal-end crossing; saturation model (clipped preamp response convolved with the ADC impulse response) when the summed maximal response exceeds 140 mV; noise 0.9 mV per sample; PMP | same |
| Digital dead time | 400 ns after a digi in the same straw | same |
| Truth | StrawDigiMC | the input step of the cluster referenced at each end's crossing (`d_step0/1`) only |
| Random numbers | CLHEP engines from SeedService | one numpy Generator: statistically equivalent, not the same numbers |
| Not implemented | on-spill folding and ghosts, cross talk, noise hits, TrackerStatus suppression, digi mixing, field-on drift | refused by `digi-cond` |

A regula-falsi step can divide by zero when two samples on the same side of the threshold are equal (the response
is piecewise constant in 0.1 ns bins) and the crossing time becomes NaN. The fast code keeps the stock consequences:
the pair loop tests `std::max(t_cal, t_hv)`, which is NaN (loop ends) for a NaN cal time but the cal time for a NaN
hv time; a NaN never combines, so a NaN hv end is the "earlier" end and searches on. It counts the ends left at NaN
(`nan_crossing_ends`).

## Validation against stock on the same StrawGasSteps

Sample: the `gen_as_v1` CRY extracted cosmics, subruns 1001-1010, fast step tables `fast_as_v18L`; the stock arm is
path A (`StrawGasStepsFromTable` on the same table, `digi_from_fasttable_alltrig.fcl`, `art/run_digi_ref.sh`), the
floor is the same stock job with `SeedService.baseSeed 9` (default 8). 27779 events pass the PrimaryFilter tracker
term in all three arms; the fast arm digitises exactly those events (`digi --events`). `digicmp` pairs the arms event by
event. Because the arms share their events and steps, the plain deck statistic sum (N1 - N2)^2 / (N1 + N2) and its
event-clustered form (variance sum_events n_eb^2) sit far below 1 for any correct digitiser; the paired statistic
(variance sum_events (n1_eb - n2_eb)^2) is about 1 and is the sensitive one. Every statistic is flagged above
max(2, 3 x floor).

| quantity | fast | floor (stock, other seed) |
|---|---|---|
| digis per event (stock 22.3727) | 22.3611, paired -0.052 +- 0.031 % | 22.3711, -0.007 +- 0.031 % |
| panel hits per event (stock 11.8360) | 11.8343, -0.015 +- 0.031 % | 11.8379, +0.016 +- 0.031 % |
| F2 good fits (stock 26693) | 26681, -0.045 +- 0.119 % | 26701, +0.030 +- 0.120 % |
| events with a good F2 fit (stock 26686) | 26672, -0.052 +- 0.118 % | 26695, +0.034 +- 0.120 % |

chi2/ndf against stock (deck / deck, event-clustered / paired), none flagged:

| histogram | fast | floor |
|---|---|---|
| panel occupancy (216 bins) | 0.030 / 0.005 / 1.055 | 0.030 / 0.005 / 1.041 |
| straw occupancy (20736 bins) | 0.028 / 0.028 / 1.012 | 0.028 / 0.028 / 1.013 |
| TDC cal, relative to the event's first (2 ns bins) | 0.940 / 0.528 / 1.162 | 0.863 / 0.553 / 1.022 |
| TDC hv, relative | 0.703 / 0.485 / 0.913 | 1.008 / 0.667 / 1.200 |
| TDC cal, absolute (500 ns bins) | 0.037 / 0.001 / 1.046 | 0.038 / 0.001 / 1.069 |
| TDC hv, absolute | 0.036 / 0.001 / 1.005 | 0.039 / 0.001 / 1.073 |
| TOT cal | 0.962 / 0.586 / 1.093 | 0.938 / 0.539 / 1.272 |
| TOT hv | 0.935 / 0.522 / 1.243 | 0.924 / 0.597 / 1.225 |
| PMP (ADC) | 1.056 / 0.809 / 1.189 | 1.038 / 0.810 / 1.157 |
| ADC samples (all 15) | 0.971 / 0.552 / 0.996 | 1.081 / 0.634 / 1.126 |
| t_hv - t_cal | 0.862 / 0.647 / 0.906 | 0.721 / 0.538 / 0.757 |
| wire distance (fast hit reco) | 0.950 / 0.736 / 0.993 | 0.895 / 0.726 / 0.930 |
| hit energy (fast hit reco) | 1.038 / 0.690 / 1.188 | 1.007 / 0.812 / 1.149 |
| digis per event | 0.634 / 0.634 / 1.013 | 0.596 / 0.596 / 0.963 |
| panel hits per event | 0.236 / 0.236 / 0.571 | 0.291 / 0.291 / 0.815 |

In the 603k straws with exactly one digi in both arms, the fast - stock differences have the same quantiles as
floor - stock (TDC cal 68 % interval -4.00 to +3.95 ns against -3.98 to +4.00 ns; TOT -1 to +1; PMP -38.6 to +38.6).

## Timing

One mu2egpvm05 core (AMD EPYC; `taskset`, one BLAS thread), the 27779 events above: fast digitisation 3.19-3.39 ms
per event (mean 3.30 ms, about 300 events/s; reading and writing the tables adds 0.1 ms). Stock `makeSD` in the same
jobs on the same host: 3.6-4.1 ms per event (TimeTracker mean; median 3.0-3.5 ms). The whole stock path A job (500k
generated events read, steps from the table, PrimaryFilter) runs at about 22 accepted events/s.

## Reproducing

```bash
D=$M2FS_DATA/trkfs_digi_20261026; TABLES=$M2FS_DATA/trk_fastsim/fast_as_v18L
# stock reference and floor (built TrkFastSim area), one shard per subrun and seed
SEED=8 bash art/run_digi_ref.sh $GEN_1001 $TABLES/fasttrkCL.001400_00001001.root $D/ref v18L_1001
SEED=9 bash art/run_digi_ref.sh $GEN_1001 $TABLES/fasttrkCL.001400_00001001.root $D/ref v18L_s9_1001
# conditions, fast digis, comparison (F2=1 needs a JAX python)
JAX_PLATFORMS=cpu F2=1 bash scripts/digi_validate.sh $D $D/ref $TABLES 1001,1002,...,1010 $D/v2 $VENV_PY
```

## JAX port

Clusters are flat arrays sorted by (straw end, time) with offsets per row (CSR); every waveform evaluation is a
segment sum, `jax.ops.segment_sum` with padded fixed-size buckets. The sequential parts are lock-step loops with
masks and small bounded iteration counts: return-below-threshold (bounded by clusters per row), rough crossing (a
cumulative sum), the peak search (K candidates per pass), regula falsi (10 fixed steps), the crossing-pair loop (a
few pairs per straw), the digital dead time (rank within the straw). Regular sample times (TOT ticks, ADC samples,
saturation steps) use integer index strides (`Waveforms.volt_lattice`). The residual-energy loop of minion steps
draws blocks with prefix acceptance and a short exact tail. Random numbers are drawn as arrays per stage, so the
port needs one key per stage.
