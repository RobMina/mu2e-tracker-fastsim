# Detector-only F1 prototype

`python3 -m trkfastsim.reco.autonomous` is an **opt-in, stock-independent prototype**, not native-exact KKLine
and not a replacement for the default F2 `tracker_point` task. It does not modify `kalman_jax.py` or existing
reconstruction defaults.

## Inputs and populations

The input is the digi-only `trkdump/events` layout produced by `digi` or `StrawHitTableDumper`. Reconstruction reads
only `(run,subrun,event,pbt)` and `d_*`, plus explicit detector conditions and the drift calibration. It never reads
stock `sd_*`, `kh_*`, `kst_*`, `kl_*`, stock hit columns, acceptance flags or MC truth. A calibration source may be a
stock dump: only its conditions tree is consumed. All input events remain in the output, including zero-hit events.
Every LineFinder seed is fitted; no single-seed or stock-good selection is imposed. The table retains failed fits
and per-event seed ordinals, while the EventNtuple writes the unmodified fitter's accepted tracks.

The front end is `straw_hits` (single-straw ComboHits), `time_clusters` and JAX `line_seeds`. Panel-combined hits
are not inputs to this LineFinder chain. Seed direction/momentum uses float-vector arithmetic before conversion
back to the six-parameter line; this is native-style, not a certified native seed/covariance/range reconstruction.

Every selected measured hit has its own crossing association, including repeated digis on one straw. The optional
track-table crossing field `hit` indexes the **original padded hit block**. The writer validates this association
and preserves it through its hit sorting; older tables without this field keep their existing straw-ID lookup.
No-hit crossings carry `hit=-1`, even if another hit on that straw exists elsewhere in the input event.

## Fixed-envelope limitation

The existing fitter receives a fixed candidate set and initializes from scratch. It cannot pause after seed fitting,
append detector candidates and continue the existing owned trajectory. This prototype therefore constructs:

- all measured LineFinder seed members;
- same-event extra hits within 50 ns and 10 mm of the initial seed line, within finite wire length and not dead;
- hit-associated material crossings plus unique hitless finite-straw candidates within 5 mm of the seed line.

Those cuts are an explicit **initial-seed envelope**, not native `addStrawHits`/`addStraws`: no fitted-seed search,
owned-piece construction-time POCA, native range gate or covariance consistency cut is claimed. Material axes
can be rebuilt from wire ends and a `TrkAlignStraw` CSV. Without that option the wire-axis approximation is labelled.
No-hit crossings cover both upstream and downstream line directions, not just a forward ray from y=0.

The fitter runs the existing complete stock-inspired time schedule (`time_model='stock'`, own states), including
owned-piece iterations. This does not resolve its native reference-search, boundaries or construction-fixed offset
limitations. A true continuation API must preserve owned pieces/covariances, immutable constraints and hit states
while extending identities and effects; starting an extension-only fit from a fitted line is not equivalent.

The writer still uses its t0 line at reference surfaces, lacks MC/CRV branches, and uses nominal material radiation
lengths. Final represented effect/t0 finite/covariance checks are diagnostic; they do not silently change acceptance
and do not certify intermediate histories. Failed tracks can fail these checks.

## Run and evaluate

From the compatible core/tracker environment:

```bash
python3 -m trkfastsim.reco.autonomous DIGIS.root --conditions reco.json \
  --drift-source CALIBRATION.root --straw-axis TrkAlignStraw.csv --settings mc \
  --outdir NEW_OUTPUT --chunk 32
python3 -m trkfastsim.reco.autonomous_compare NEW_OUTPUT/EventNtuple.root STOCK.root \
  --same-digis --out NEW_COMPARISON.json
```

Use `--settings data` with matching real-data conditions, geometry and drift calibration. The CPU wrapper
`scripts/autonomous_cpu.sh` pins one allowed core, fixes BLAS/XLA thread counts and selects CPU. Override `PY_JAX`
and `MU2E_SEARCH_PATH` for the site's environment. The module itself uses the JAX-selected device; GPU execution
requires a GPU-enabled runtime and separately measured validation. No GPU performance is implied by CPU results.

`--outdir` must not exist. Outputs are `tracks.npz`, `EventNtuple.root`, `report.json`. Declared input, actual resolved
ANN and reconstruction-source hashes are checked before/after consumption; runtime versions and diagnostic `F1_*`
environment overrides are recorded. This detects persistent changes, not transient replacement and restoration.
The report's writer coverage and missing-field conventions are unchanged. Keep these data products in a data area.

The separate evaluator requires identical unique full event populations and fails rather than intersecting away
missing events. It computes event-paired track-count and events-with-fit differences, with covariance-aware
delta-method relative-yield SEs. Zero reference yields or insufficient samples have null errors/ratios, not invented
precision. These counting errors assume independent events; they are not model/calibration uncertainties.
It counts every stored track, so comparing good-only and all-fit files is the caller's error. Counts-only is the
default. `--same-digis` asserts shared digi identity and enables existing shared-straw matched residual diagnostics;
never set it for independently digitised arms. `--bootstrap-replicates` (default 1000, minimum 2) and
`--bootstrap-seed` (default 0) control event-clustered matched-parameter uncertainties. Each replica draws the full
event population, including events without matches, and repeats every matched track of each drawn event together.
The bounded vectorized implementation reproduces NumPy linear/type-7 quantiles of the frequency-expanded sample.
It reports q16/q50/q84 and `(q84-q16)/2`, bootstrap SE and percentile CI95; per-variable absent/nonfinite values,
empty draws and insufficient/degenerate support are explicit. Tied/quantized zero medians do not imply zero error.
These are conditional shared-hit residual errors, not truth/model/calibration uncertainty or all-event residual
closure. Hashes bind both evaluation files to the report.

## Timing and scale

`batch_s` measures the device call plus synchronization, excluding array packing/transfers/output conversion. The
first call for each shape may compile; these calls are excluded from within-invocation warm timing, even when a
process cache might already hold the executable. Warm timing records real tracks and executed padded lanes separately. Fit-loop
wall includes conversions and final numerical checks. `preparation_s` includes read/hit/seed/candidate preparation;
`wall_s` covers reconstruction through product writing but excludes initial hashing/calibration loading.
`invocation_work_s` includes those initial operations, not process startup or final JSON serialization.
No timing is extrapolated to a larger sample.

`--buckets` enables optional power-of-two candidate-width groups on the device (minimum 16 slots, clipped at the
file maximum). Prefix widths use the last valid slot, not just hit counts, so masked holes cannot truncate a later
candidate. Original track order and crossing-to-hit indices are retained. Output slots are restored to the full
track-table widths. Full-width remains the default: shape changes can change floating-point histories even when
acceptance and discrete hit states agree. Multiple first-shape compilations can outweigh reduced warm work.

Compiled kernels are cached by immutable drift/schedule/ANN contents, callable identity and effective scalar fitter
configuration. Different calibrations do not reuse captured constants. This is in-process executable reuse, not a
persistent prepared detector context. Every new CLI invocation still reads/hash-checks inputs.

Host candidates and output tables remain globally padded to the file maximum. Chunking/bucketing bounds geometry
search and device work, **not** all host storage. The report records host candidate bytes, executed effect slots,
real/executed lanes and all per-batch shapes; busy-event host-memory optimization remains open.

Synthetic tests cover empty/multiple-seed events, exact hit identities, repeated-straw associations through the
writer, finite straw boundaries, event mismatch rejection, adapter padding, covariance-aware paired counts,
reordered/duplicate populations, clustered quantiles, immutable calibration-cache keys, masked-hole bucketing and
no-overwrite outputs. `scripts/autonomous_equivalence.py` audits saved table identities, acceptance exchanges,
per-parameter differences and explicit crossing associations without refitting the baseline. A cold CPU/GPU run
is not a throughput certification; preserve sample/device/runtime/source manifests and inspect decision exchanges,
not only total yields. `scripts/autonomous_gpu.sh` requires an already allocated GPU and fails if JAX is not on CUDA.

[KPP and compatible stock inputs](autonomous-reference-inputs.md) describes the all-event reference contract and
the unfiltered KPP dump FCL, runtime-tested on a compatible user-built area. In the uncalibrated 500-event test,
238 seeds retain identical membership, but autonomous F1 accepts 68 tracks versus 81 stock (paired relative yield
-16.0 +/- 8.4%). This is a validation failure to investigate, not closure or a default promotion. The fixed-envelope
and native-iteration limitations remain; native continuation and production physics closure are separate gates.
