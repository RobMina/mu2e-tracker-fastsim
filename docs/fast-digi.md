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

## Opt-in JAX backend

`digi_clusters_jax.py` and `digi_jax.py` implement the same supported off-spill, field-free model. NumPy remains the
reference and default; importing/running the NumPy digitiser does not require JAX. Install the optional `jax` extra
in the project environment, then select explicitly:

```bash
python3 -m trkfastsim digi STEPS.root --cond digicond.npz --events STOCK.root --backend jax --seed 1 --outdir NEW_DIR
python3 -m trkfastsim.kitpoint --backend jax --gen GEN.root --gdml GEOMETRY.gdml \
  --digi-cond digicond.npz --reco-cond cond_mc.json --drift-source STOCK.root --outdir NEW_POINT
```

The Python API is `digi_jax.digitize(st, nev, conditions, jax.random.key(seed), stats=None)`; it returns host NumPy
arrays in the reference layout. `fastdigi.digitize_table(..., rng=key, backend="jax")` folds the key by event-chunk
offset. CLI file streams fold by input-file index; the point runner uses a separate digitiser key domain. Repeatability
requires the same inputs, seed, file order, chunk/bucket configuration and JAX/runtime version. Cross-device bitwise
identity is not promised. Conditions must not be mutated while cached kernels use them.

### Shape and resource contracts

- Ionisation uses fixed step batches (128 on CPU, 4,096 on GPU) and power-of-two initial-cluster buckets.
  Sampling lanes are at most `min(step_batch, 262144 // cluster_capacity)`; step-index-folded keys preserve each
  step's random stream across ionisation buckets. Minion residual energy is exhausted with a masked exact loop, no tail clip.
- Individual-electron expansion, compaction, straw grouping and bucket packing are host boundaries. Gain/drift use
  4096-cluster random domains, grouped 16 at a time on GPU with the exact original per-domain keys.
  Waveforms use dense padded `[2*batch, cluster_capacity]` rows, not a device-resident CSR pipeline.
  `batch = max(1, min(max_straws, lane_budget // cluster_capacity))` bounds memory independently of population:
  CPU `(max_straws,lane_budget)=(32,2048)`, GPU `(2048,262144)`.
- Event-marker jitter uses fixed `[128,216]` blocks, keyed by event-block index, generating only occupied blocks.
  Different straws in the same event/panel share one jitter. Adding empty events does not compile a new jitter shape.
  This block-keyed stream is a new opt-in JAX random sequence; the NumPy sequence is unchanged.
- Saturated waveforms evaluate 16 (CPU) or 64 (GPU) time ticks at a time, then convolve the complete response. Integer response-lattice
  indices retain the original time origin across tiles; no charge or time tail is clipped to obtain the memory bound.
- Electronics searches run as masked JAX loops, preserving both-end matching, analog/digital dead time and stock's
  asymmetric NaN crossing behavior. TOT, linear ADC and saturation use integer lattice strides.
- More than 4096 initial clusters per step, four million expanded drift clusters per call, or 4096 clusters in one
  straw instance raise `OverflowError`; no physics tail is silently dropped. These are resource limits, not cuts.
  Near-limit dense saturation allocations can still be large; smaller event chunks bound aggregate memory.
- Pair buffers begin with four entries per straw and retry with identical keys at doubled capacity. Nonprogress and
  saturation-time overflow fail explicitly. Nonfinite inputs/generated clusters are rejected before packing, distinct
  from legitimate NaN regula-falsi crossing results, whose stock semantics are retained.

### Validation status (2026-10-27)

A 2,866-event pilot on the existing matched step sample produced 63,330 JAX digis versus 63,361 saved NumPy digis:
paired relative difference **−0.049 ± 0.095 %**; panel hits **+0.077 ± 0.100 %**. None of 16 comparison histograms
flagged under the deck, event-clustered or paired rules. Paired chi2/ndf: panel/straw occupancy 1.059/0.996, relative
cal/hv TDC 0.804/0.736, TOT 0.786/0.216, PMP 0.970, ADC samples 0.936. The independent floor here is stock seed 9
versus NumPy, not a second NumPy draw. This one-shard pilot does not certify the full ten-shard sample.

F2 on exactly the same 2,866 persisted unique events gives **2,742 good JAX fits versus 2,728 NumPy**:
paired **+0.513 ± 0.398 %** (1.29 sigma; normal-approximate 95% interval −0.267% to +1.294%). Counts of events
with a fit equal these fit counts on this sample. Stock seed 9 gives 2,753 (+0.916 ± 0.396% versus NumPy).
This is pilot agreement, not demonstrated sub-percent equivalence. No stock fit products condition F2.

Tests cover repeatability, empty/zero-hit event retention, NumPy RNG and array identity, time windows/read-all,
truth indices, dead time and pair retry, nonfinite/capacity rejection, and identical-cluster noiseless linear/saturated
electronics against NumPy (crossing tolerance 0.04 ns, TDC two ticks, ADC one count, TOT exact).

Cold pilot timing on one pinned GPVM core was **539.97 s / 2,866 events = 188.405 ms/event**, including compilation
and host packing: no speedup is established. A subsequent local 100-event/2,200-step synthetic check measured
NumPy cold 1.212 s, warm 0.931/0.881 s; JAX cold 23.715 s, warm 0.827/0.861 s. Both repeated identically per seed.
These small, unpinned laptop timings are not production throughput or an A100 result. An unused-pair-loop optimization
and input/resource guards added after the remote pilot need full-sample validation; NumPy remains recommended.

A100-SXM4-80GB / JAX 0.11.2 validation: **19 tests pass**, explicit GPU selection checked, both backends
per-seed reproducible. The same synthetic 100-event benchmark measured JAX cold **52.884 s**, warm **0.862/0.851 s**;
NumPy on that node's CPU cold **0.624 s**, warm **0.592/0.578 s**. The allocated test completed in 2:47 with peak
host RSS 2.69 GiB and verified temporary-directory cleanup. This small host-bucketed workload shows **no GPU speedup**.

### Bounded real-table validation (small-batch baseline)

The initial bounded implementation is opt-in; the GPU throughput update below supersedes its resource policy and timings. A separate ten-group saved-step sample contains **27,786**
unique events; it is not the historical 27,779-event `fast_as_v18L` sample above. Regenerated current NumPy,
NumPy seed 9 and JAX digitisation use the same full event populations and conditions. Persisted counts are
619,710 / 619,541 / 619,603 digis respectively. JAX versus NumPy: digis/event **−0.017 ± 0.031%**,
panel hits/event **−0.051 ± 0.031%** (paired standard errors). All 16 digi/hit histograms are unflagged under
the plain, event-clustered and paired rules. The NumPy independent-seed floor is **−0.027 ± 0.031%** for
digis/event. F2 on all the same persisted events gives **26,740 good JAX fits versus 26,723 NumPy**, paired
**+0.064 ± 0.116%** (normal-approximate 95% interval **[−0.164%, +0.291%]**). Events with a fit:
26,732 versus 26,714, **+0.067 ± 0.116%**. Independent NumPy seed 9 gives 26,693 fits (**−0.112 ± 0.121%**).
F2 uses fixed 64-track blocks and only 64/128 hit capacities, with unchanged equations and acceptance cuts.
This checks distribution-level agreement, not equal RNG streams or proof of all possible tails.

For a real 2,707-event / 72,977-step group, each allocation ran JAX and NumPy sequentially, with three identical
seed/input passes per backend. Timings include table loading and synchronised host outputs; warm means the same
process already compiled its kernels, not a persistent on-disk cache. Different allocations are not a controlled
CPU-versus-GPU hardware comparison; compare JAX versus NumPy within each row:

| Allocation | JAX cold (s) | JAX warm (s) | NumPy cold (s) | NumPy warm (s) |
|---|---:|---:|---:|---:|
| UVA CPU, 2 allocated cores | 108.231 | 25.112, 24.800 | 17.023 | 16.469, 16.535 |
| UVA A100-SXM4-80GB, 2 host cores | 160.065 | 26.113, 25.604 | 11.172 | 10.986, 11.011 |

All repeated arrays match exactly within each backend/device. Jitter, gain and ionisation-count JIT caches each
hold one shape; this group uses 10 ionisation-sample and 10 electronics kernels, with no extra warm compilations.
The CPU timing job peaks at 1.94 GiB host RSS; the A100 allocation at 2.11 GiB host RSS (not GPU device memory).
These measurements do **not** establish GPU speedup; NumPy remains the recommended default.

`scripts/digi_real_bench.py` accepts explicit step/event/condition paths, refuses an existing output directory,
checks unique event identities, persists the first-pass digis, hashes every repeated output array and writes
`timing.json` including peak process RSS and kernel-cache counts. `--limit 0` uses the full saved event population;
`--limit N` is only a named timing probe, never a full closure claim. `scripts/digi_cert_report.py` verifies ten
complete file receipts against strict-admission comparison counts and rejects partial cold/warm timing receipts.

```bash
python3 scripts/digi_real_bench.py --steps MATCHED_STEPS.root --events MATCHED_EVENTS.root \
  --cond digicond.npz --backend jax --limit 0 --repeats 3 --outdir NEW_JAX
python3 scripts/digi_real_bench.py --steps MATCHED_STEPS.root --events MATCHED_EVENTS.root \
  --cond digicond.npz --backend numpy --limit 0 --repeats 3 --outdir NEW_NUMPY
```

The `uva_digi_{cert,timing,compare,selected}.sh` wrappers accept snapshot, image, input and output roots through
required environment variables. `uva_digi_cert.sh`'s named real-group layout requires the corresponding saved
step/event inputs; it is a reproducibility wrapper, not an input downloader or a generic quickstart. Submit to
an explicit Slurm account/partition with bounded memory; never compile this workload on a shared interactive host.

### GPU throughput update

Small CPU-sized dispatches, not GPU arithmetic throughput, dominated the first implementation. A synchronized
2,707-event A100 profile measured 11.27 s in cluster production, 4.11 s across 2,329 electronics calls, and substantial
array-transfer/dispatch cost (36,134 `jnp.asarray` calls and 31,918 device-to-host conversions). ROOT input was ~0.06 s.
The GPU resource policy above reduces electronics to 53 calls and groups charge/drift into 38 calls. A fresh
synchronized profile measures cluster production 1.431 s, electronics 0.211 s and charge/drift 0.041 s; total 2.683 s.
These are diagnostics with deliberate synchronization, not independent times to add to another timing run.

On the same A100-SXM4-80GB allocation (2 host cores), the production-policy 2,707-event / 72,977-step benchmark
measures JAX cold **139.353 s**, warm **2.662/2.692 s**, versus host NumPy cold **10.814 s**, warm
**10.736/10.666 s**: **4.00× warm speedup**, with synchronised host outputs and table loading included. Every
repeat within each backend is identical. The result is digitiser-only, not full detector transport/reco acceleration.
One real 50,000-generator-event point completes in both arms with exactly 50,000 persisted entries, 6,394 straw
steps and 267 versus 266 fitted events; its cold JAX digitisation (158.665 s) is much slower than NumPy (1.536 s).
Do not select JAX for a small one-off cold job on the basis of warm throughput.

The full 27,786-event ten-group GPU digi sample gives 619,575 digis against current NumPy 619,710:
**−0.022 ± 0.031%** paired; panelhits **−0.053 ± 0.031%**, with no flags in all 16 histograms under any rule.
Fresh F2 on these same persisted events gives **26,749 versus 26,723 fits**, **+0.097 ± 0.116%**
(95% normal interval **[−0.130%, +0.325%]**); events with a fit **26,741 versus 26,714**, **+0.101 ± 0.116%**.
The independent NumPy-seed9 floor gives 26,693 fits, **−0.112 ± 0.121%**. Each arm retains all 27,786 unique
full event identities; the fixed-shape fitter uses only `64x64` and `64x128` batches.
The larger GPU batch/reduction shapes are not promised bit-identical to the small-batch implementation.
Grouped charge preserves keys exactly; arithmetic agrees within two float32 ULP in the GPU regression (CPU exact).
No charge tail, ionisation residual or waveform window is truncated for speed. No precision downgrade is used.

Cold compilation remains substantial. An optional **private, trusted** `JAX_COMPILATION_CACHE_DIR` is forwarded
by `uva_digi_cert.sh`; use the same image/device/code. A measured fresh process with a populated 33 MB cache
reduces the first dense group from 142.218 to 20.784 s; subsequent groups are ~2.6–2.8 s. Never treat a persistent
cache hit as an in-process warm pass. Cache files are executable input: do not reuse an untrusted writable cache.
GPU allocator peak for the ten-group empty-cache run is ~1.01 GiB live, ~2.00 GiB pooled; host Slurm RSS is ~2.7 GiB.
A cache-loaded run may report smaller peaks. These bounds are sample measurements, not upper bounds for arbitrary
admitted conditions or tails. NumPy remains default; JAX is an explicit choice for adequately batched GPU workloads.

A same-allocation **ten-group** campaign (27,786 events / 755,276 steps per pass) confirms the speedup beyond one
group. With the populated private cache, the first process pass takes **46.094 s** in JAX versus **112.298 s** in
NumPy, including JAX's ~20 s first-group loading cost: **2.44×**. The next two complete passes take
**26.661 / 26.632 s** versus NumPy **114.066 / 112.109 s**, **4.28× / 4.21×**. Three-pass measured total:
**99.387 s JAX versus 338.472 s NumPy (3.41×)**. This test used one sequential CPU NumPy process;
it is not a GPU-versus-fully-utilised-multicore-node comparison. The totals include table loading and synchronised host outputs,
not output ROOT serialization. The benchmark verifies every repeated array and exact full event populations;
JAX uses 544 electronics dispatches per full pass. Empty-cache one-off performance remains a separate limitation.

### Paired comparison and timing

Fast files have no drift conditions tree. For direct JAX-versus-NumPy F2 comparisons, provide the same stock drift
source explicitly. `--f2-chunk N` bounds events per reconstruction batch, preserving empty events and pooling good-fit
values before calculating quantiles; `0` retains the single-batch default. Event chunks alone do not prevent
compilation churn: each distinct seed count otherwise produces another fitted-track JIT shape.

`--f2-fixed-shapes` uses the caller-only `reco/f2batch.py`: fixed 64-track blocks and power-of-two hit capacities
(at least 64), duplicate padding removed before any acceptance/counting, and smaller existing linefinder batches.
No hits or candidates are discarded; fit equations, schedule, iterations and good-fit cuts are unchanged. The default
path is retained for comparisons. Real legacy-versus-fixed checks agree on 32 events (30 seeds, 29 good fits), and
100-event checks with chunks 50 and 37 agree (96 seeds, 95 good fits). Ragged/reordered 63/64/65-hit and >64-track
unit tests compare real fitter parameters, covariance and hit states; finder tests cover the >256-hit NumPy fallback.
The completed three-arm pilot used only `64x64` and `64x128` fit shapes and summed batch wall times 122.24 s / 82.70 s /
75.21 s (JAX digi / NumPy digi / stock floor); warm-arm fit portions were 5.16 s / 4.65 s. Seeding dominates remaining
comparison runtime. `--f2-progress` emits per-batch timing and shape JSON lines.

`--require-same-events` rejects unequal or duplicate full `(run,subrun,event)` identities rather than silently
intersecting. Event remapping keeps digi and variable-length ADC rows in consistent event order. Full tuple identities
avoid packed-bit collisions for large event/subrun numbers.

```bash
python3 -m trkfastsim digicmp JAX.root --ref NUMPY.root --floor STOCK_OTHER_SEED.root \
  --hit-cond hitcond.json --drift-source STOCK.root --f2 --f2-chunk 50 \
  --f2-fixed-shapes --f2-progress --require-same-events --outdir NEW_COMPARE
python3 scripts/digi_bench.py 100 3 --backend jax --json timing.json
```

`digicmp` intersects event populations: independently verify persisted input counts and common-event coverage.
`scripts/digi_jax_validate.sh NEW_OUTPUT` reuses the existing stock/NumPy references, never reruns stock, and refuses
an existing output directory. Its default is one shard; widening it requires a separately sized/authorized run.
