# End-to-end device chain

`trkfastsim/kit/devchain.py` runs the whole fast tracker chain on one generator table in memory, with a backend
choice per stage, for identity and throughput measurements:

```
gen table -> steps (stage C, sliced world MSC, every primary muon) -> digis -> straw hits -> panel combo hits
          -> time clusters -> line seeds -> F2
```

With every backend `numpy` it is the `tracker_point` chain (`kitpoint.run`) without its file handling and writer:
same event keys, seeds and outputs. The combo hits are made and counted but not consumed, as in the stock extracted
reconstruction.

| Stage | `numpy` | `jax` | JAX against NumPy |
|---|---|---|---|
| steps | `make_steps` | `make_steps(backend="jax")` with the device-chain step settings below | other random stream: distributions |
| digis | `digitize_steps` (tracker_point chunking) | JAX digitiser, chunks of events with steps (`compact=True`) | other random stream: distributions |
| straw hits, combos | `hitreco`, `combo` | `hitreco_jax`, `combo_jax` | byte-identical on CPU; last bits differ on a GPU |
| time clusters | NumPy | NumPy | identical |
| line seeds, F2 | JAX | JAX | CPU and GPU differ in the last bits |

## Changes behind it

- **In-memory handoffs (default, identical).** `tracker_point` passed the steps and the digis between stages through
  ROOT files in a temporary directory. `fastdigi.digitize_steps` digitises a step DataFrame, `digis_as_read` and
  `events_as_read` give exactly what `hitreco.read_digis` and the track table read back from those files, and
  `tracktable.from_f2_digis` runs F2 on them. On a real 50,000-event shard every `tracker_point` metric and the
  EventNtuple content hash are unchanged.
- **F2 track buckets (default, identical).** `linefit_jax.fit_own_states` pads the seed count of a call to a power
  of two (at least 64) by repeating the last track, so a new table reuses the compiled fit instead of compiling one
  shape per seed count. All stage digests are unchanged on 20 shards.
- **Compact JAX digitisation (opt-in).** `digitize_steps(compact=True)` chunks over the events that have steps.
  About 0.8 % of generated cosmic events have tracker steps, so the default chunking by 1,000 generated events made
  about 50 calls per shard, each a full fixed-shape GPU call; compact makes one or two. Other chunking, so another
  random sequence; distributions agree (below). NumPy keeps the default chunking.
- **Step-kernel shape growth (opt-in).** `TRKFASTSIM_JAX_SHAPE_GROWTH` (default 2: powers of two, unchanged) sets
  the growth between padded sizes of the JAX step kernels; with a larger power of two (the device chain uses 8),
  and with the transport chunk state padded to row tiers, a stream of new tables stops compiling new kernel
  shapes after the first one or two tables (CPU census: 565 compilations on the first table, then 19, 12, 9, 1, 6,
  14, 5). With growth 2 the step kernels compile new shapes on every new table (200-240 s of compilation per table on
  an A100), and on CPU the accumulating executables exhaust the process memory maps after one or two tables.
  Padding lanes are inert, but the compiled code differs, so step floats move at the 1e-6 mm level; digis were
  identical on the table checked.
- **Unrolled loss sampler (opt-in, identical).** `steps_jax.SAMPLER_UNROLL` (default off; the device chain turns it
  on) passes `unroll=True` to the core energy-loss sampler inside the slab loop: its fixed-trip Poisson and Gamma
  loops are unrolled at trace time instead of running as device loops. On a GPU each trip of a loop nested in an
  iterated kernel is its own launch (about 28 per slab slice); unrolled, a slice is one kernel. Same keys and
  arithmetic: all seven stage digests are unchanged on an MIG slice. Warm steps time per table 1.8 -> 1.27 s; the
  first-table compilation grows from 223 to 667 s on the slice.
- **Device-resident slab loss (device chain default; `steps_jax.SLAB_KERNEL`, default `loop`).** With `device` the
  support slabs of each interior pass are one `mu2efastsim.slab_jax.slab_loss` call and the whole world column of a
  chunk (every band's materials in order, then its air) one `slab_jax.path_loss` call. On CUDA the call is a Triton
  kernel whose blocks run their own slice loops, so no host round trip per slice round and no host count per world
  material; elsewhere it is the XLA while loop (`MU2EFS_SLAB_KERNEL` overrides, reports record the choice). Same keys,
  draws and slice policy; values equal to rounding (a last-bit difference can move one slice boundary), counts equal
  to the loop path on every table checked, and downstream hits, combos, seeds and fit tables identical on them.
  `SLAB_BLOCK` (32) sets lanes per kernel block (shorter float64 latency of a slow block).
- **Radiative muon losses (default, as NumPy).** The support slabs (loop and kernel routes) and the world column
  (`world_jax`, loop and `path_loss`) draw the core's default bremsstrahlung, pair and photonuclear losses with
  `mu2efastsim.muradiative_jax` after the ionisation slices (key fold_in(pair key, `RAD_FOLD`)). Against the
  ionisation-only device chain on the same tables and keys (float64, A100): events with steps -0.05 +- 0.04 %, digis
  +0.00 +- 0.19 %, good fits -0.25 +- 0.37 %, 0 of 11 histograms flagged. Cost: MIG float32 warm 71,600 -> 69,900
  events/s (steps 8.4 -> 8.8 s per 20 tables); A100 float64 42,500 -> 38,500 (steps 16.1 -> 18.1 s).
- **Minion electron fill (device chain default, identical).** `digi_clusters_jax.MINION_UNROLL` (16 loop bodies per
  device iteration; the extra bodies past a lane's end are identity) and `SEARCH_METHOD="compare_all"` (searchsorted
  without a device loop): all seven stage digests unchanged on 20 tables.
- **Digitiser overflow.** A JAX chunk with a straw beyond the device digitiser's cluster capacity is digitised by
  NumPy, seeded from the chunk key (`stats["numpy_fallback_chunks"]`); before, the chain stopped (1 table in 80 at
  float64).
- **Other step options (opt-in, not used).** `steps_jax.SLICE_ROUNDS >= N_SLICES_MAX` (slab loss in one device call
  per material), `steps_jax.SLAB_UNROLL` (slices per loop iteration) and `world_jax.SYNC_FREE_PAIRS` (no host count
  per world material) remove host synchronisations but were not faster on an MIG slice, with or without the unrolled
  sampler, and triple the compile time.

## Measuring

```bash
python3 -m trkfastsim.validation.chain_bench --gen GEN_1.root GEN_2.root ... --gdml GEOMETRY.gdml \
  --digi-cond digicond.npz --reco-cond hitcond.json --drift-source STOCK_DUMP.root \
  --backends device --seed 1 --repeat 2 --log-compiles --outdir NEW_DIR
python3 -m trkfastsim.validation.chain_pair NUMPY_SEED1/report.json DEVICE/report.json --null NUMPY_SEED2/report.json
```

`chain_bench` builds the context once, runs `--repeat` passes per table and writes `report.json`: per pass the stage
wall times (and the step sub-stages), counts, an output digest per stage (repeatability and identity), peak host RSS
and device memory, the JAX compilations (`--log-compiles`), optionally a cProfile (`--profile-pass`) or a device trace
with the host/device copies (`--trace-pass`), and per table 11 histograms. "warm" sums the repeats of tables already
run in the process; "steady" every pass after the first (new tables, as in a production stream). `chain_pair`
compares two reports on their shared tables with paired per-shard errors (the same batch variance as
`shardstats`) and a null arm.

## Results

Inputs: 20 extracted-v04 cosmic generator tables of 50,000 events each (1 M generated events), the tracker_point
conditions, float64.

**Distributions** (device chain on an MIG slice against all-NumPy seed 1, null all-NumPy seed 2, 20 tables): events
with steps +0.15 +- 0.58 % (null -0.50 +- 0.69 %), digis -0.34 +- 1.01 % (-0.10 +- 1.03 %), good F2 fits
+0.02 +- 0.96 % (-0.18 +- 0.82 %); no count beyond 1 sigma; none of the 11 histograms (steps, digis, hits and
seeds per event, step eIon and length, digi panel, hit time and energy, F2 chi2/ndf and active hits) above
max(2, 3 x null), chi2/ndf 0.8-1.4 against an expectation of 1.12. The CPU-JAX and A100 arms agree the same way.
Repeat passes are byte-identical on every device.

**Throughput** (generated events per second, warm):

| Device | Events/s | Per table |
|---|---:|---:|
| one Xeon Gold 6248 core, all NumPy | 3,030 | 16.5 s (steps 12.4, digis 2.2, seeds 1.6, F2 0.3) |
| 40-core Xeon Gold 6248 node, 40 NumPy processes | 111,500 | |
| A100-SXM4-80GB, device chain before the unrolled sampler | 14,270 | 3.5 s (steps 2.9, digis 0.5) |
| RTX PRO 6000 MIG 1g.24gb slice, before the unrolled sampler | 22,160 | 2.3 s (steps 1.8, digis 0.5) |
| RTX PRO 6000 MIG 1g.24gb slice, before the device slab loss | 29,150 | 1.7 s (steps 1.2, digis 0.46) |
| RTX PRO 6000 MIG 1g.24gb slice, before the device slab loss, float32 (`TRKFASTSIM_DEVICE_FLOAT=core`) | 47,120 | 1.06 s (steps 0.68, digis 0.32) |
| RTX PRO 6000 MIG 1g.24gb slice, device chain | 46,340 | 1.08 s (steps 0.69, digis 0.33) |
| RTX PRO 6000 MIG 1g.24gb slice, device chain, float32 | 80,100 | 0.62 s (steps 0.37, digis 0.20) |
| RTX PRO 6000 MIG 1g.24gb slice, device chain, 2 processes (sum), float64 / float32 | 59,950 / 111,720 | |
| A100-SXM4-40GB, device chain, 1 process, float64 / float32 | 44,050 / 42,800 | |
| A100-SXM4-40GB, device chain, 4 processes (sum), float64 / float32 | 119,550 / 123,490 | |

At float32 (core float32, tracker following core; see [device-precision.md](device-precision.md)) the chain
against all-NumPy on the same 20 tables: no count beyond 1 sigma (largest pull 0.92, time clusters), 0 of 11
histograms flagged (chi2/ndf at most 1.66; null arm at most 1.58); device memory peak 1.4 GB. Float64 output with and
without the float32 changes is byte-identical at every stage.

On the MIG slice the stream of 20 new tables (every pass after the first, compiled kernels loaded from a persistent
cache) runs at 22,000 events/s: tables with new kernel shapes add 1-8 s of compilation. The first table costs 490 s
on the A100 and 223 s on the slice before the unrolled sampler, 667 s on the slice with it (compilation, paid once per
code and device kind with a persistent JAX compilation cache). Device memory peaks at 2.0 GB; host RSS 5-7 GB.

**Device work per table** (MIG slice, profiler trace of one warm pass): before the unrolled sampler 612 k device
events (540 k kernels), the device busy 1.6 s of a 3.0 s pass and 3.5 s of host time in XLA `while` thunks; with it
210 k device events (151 k kernels), busy 1.2 s of 2.0 s, 1.3 s in `while` thunks. Copies per table: about 6.8 k
device-to-host (nearly all of at most 64 bytes: loop predicates and work-list counts), 1.3 k host-to-device, 51 k
device-to-device.

With the device slab loss (80 tables, 4 M generated events, against all-NumPy seed 1 with seed 2 as the null): 0 of 11
histograms flagged at float64 and at float32; float64 events with fits +1.58 +- 0.52 % against seed 1 and
+0.58 +- 0.53 % against seed 2 (seed 2 against seed 1: +1.00 +- 0.46 %), and the device chain before the change shows
the same +1.31 +- 0.54 % / +0.54 +- 0.55 % on those tables: the offset is the NumPy seed-1 reference, not the change.
float32: steps +0.03 +- 0.43 %, fits +0.87 +- 0.49 %. Concurrent processes share a GPU: per device, two processes on
an MIG slice and four on an A100 match or exceed a 40-core node of NumPy processes (111,500 events/s); one process is
bound by its host (single process on the A100's EPYC 7742 host: 43-44 k).

## Limits

- Before the device slab loss the slab loop was a host-driven `lax.while_loop` (one predicate copy and a few small
  kernels per iteration; about 0.5 M device events and 12 k device-to-host copies per table). The interior pass loop,
  its support-column work lists and the digitiser's electronics loop are still host-driven `while` loops with small
  syncs; at float32 one process keeps an MIG slice busy about half the time, hence the gain from two processes.
- The slab kernel is NVIDIA CUDA only (Triton). Other GPUs and CPUs run the XLA path: same results to rounding,
  without the speed-up. At float64 a consumer Blackwell slice is bound by its float64 rate in the kernel.
- The unrolled sampler compiles about three times longer; a persistent compilation cache makes that a once-per-code
  cost per device kind.
- Host work that stays on the host: generator selection, DataFrame assembly, the digitiser's packing between its
  kernels.
- Identity: see [limitations.md](limitations.md) (device chain, NumPy across host CPU types).
