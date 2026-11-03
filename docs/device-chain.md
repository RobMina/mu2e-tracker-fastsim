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
| A100-SXM4-80GB, device chain | 14,270 | 3.5 s (steps 2.9, digis 0.5) |
| RTX PRO 6000 MIG 1g.24gb slice, device chain | 22,160 | 2.3 s (steps 1.8, digis 0.5) |

The first table costs 490 s on the A100 and 223 s on the MIG slice (compilation; a persistent JAX compilation cache
keeps it for the next process of the same code on the same device kind); later tables add 0-19 small compilations.
Device memory peaks at 1.9 GB; host RSS 4-6 GB.

## Limits

- The GPU stages are host-driven: the slab loop is a `lax.while_loop`, which XLA executes on a GPU as a host-driven
  loop (one predicate copy and a few small kernels per iteration). Before the unrolled sampler a table took about
  0.5 M device events and 12 k device-to-host copies, nearly all of at most 64 bytes, with the device busy about half
  of the pass. XLA's own loop unrolling and command buffers did not help. More processes on one MIG slice add 12 %
  at most: the slice is bound by the serial chain of small kernels.
- The unrolled sampler compiles about three times longer; a persistent compilation cache makes that a once-per-code
  cost per device kind.
- Host work that stays on the host: generator selection, DataFrame assembly, the digitiser's packing between its
  kernels.
- Identity: see [limitations.md](limitations.md) (device chain, NumPy across host CPU types).
