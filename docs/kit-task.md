# Tracker point and stock-conditioned companion

## `tracker_point` contract

The optional `fastsimkit` task runs one complete cached-primary shard through:

1. All primary muons, extracted-v04 stage C, sliced world MSC, standard interior transport.
2. NumPy fast digitisation, preserving **every generator event**, without a PrimaryFilter selection.
3. Fast straw hits, time clusters, JAX line seeds and hybrid **F2**, fixed 2000 MeV/c and no fit material.
4. EventNtuple-compatible RNTuple writer and persisted event/track-count verification.

This first configuration is explicitly **primary-only**, with no secondary library. It does not supply a full
occupancy model, autonomous F1, stock KKLine closure, arbitrary geometry support, CE sensitivity or commissioned
BO objective. No numerical or calibration settings are exposed as optimisation parameters.

Typed parameters are `sample=extracted_v04_primary_v1`, `shard=1..20` (default 11) and `seed=0..2147483647`
(default 1). Unknown keys, nonintegral counts, booleans, numeric strings, unsupported files and geometry inputs fail
admission before dispatch. Routes are mounted EAF and per-job tmpfs streaming UVA; neither was deployed or tested
live as part of registration. The matching core contains parameter and capability descriptors.

### Counts, calibration and provenance

All 19 metrics are required, finite and nonnegative. Counts distinguish primary muons, events with steps, steps,
digis/hits, seeds/attempts, good F2 tracks, events with a good track, written events and active hits in written tracks.
`fit_event_fraction = events_with_fit / generated_events`; `fit_event_fraction_se` is the binomial counting SE only,
not model discrepancy, calibration uncertainty or independent-replica variance. One event can contain multiple
muons or tracks; zero-hit events remain in the denominator and output. The denominator is the distinct persisted
`(run,subrun,event)` population of the complete generator table, not a count inferred from logs or subrun exposure.

The drift-source file is read **only for its conditions tree**; stock event, hit, seed and fitted-track populations
are never F2 inputs. `tracktable.from_f2(..., drift=...)` accepts that calibration separately from digi-only files.
Existing callers retain the input-file conditions default. Negative extrapolated radii at negative drift times are
kept as stock; the guard requires finite, equal-length arrays, increasing time grid and nonnegative errors, not
nonnegative extrapolated radii. An empty-ADC guard permits genuine zero-digi batches.

`provenance.json` records input SHA256/size, source-file hashes, fixed settings, writer coverage/content hash,
product checksum and retention. Input hashes must agree before and after consumption, otherwise no metrics or
provenance are published. This detects persistent replacements, **not** adversarial transient change-and-restore
and is not a lifetime immutable snapshot. UVA also binds source expectations and a read-only code snapshot through
the existing core helpers. This core version hashes the SIF but does not retain a lifetime SIF copy. EAF's mounted
runtime/venv remains mutable. These limitations must be addressed before unattended commissioning.

### Products and timing

By default the writer runs in private scratch, verifies the highest-cycle product and removes it; only JSON
metadata remain. Endpoint adapters never pass `--retain-ntuple`. This does not grant permission to export internal
ROOT products. Local/FNAL use may pass `--retain-ntuple`, writing `EventNtuple.root` in the requested output directory.
Inputs and intermediate ROOT files do not enter fetched endpoint outputs. Cleanup receipts are required by both
routes; they are operational evidence, not secure-deletion proof or a SIGKILL cleanup guarantee.

The four stage timers include their I/O and cold compilation; `wall_s` also includes input reads/hashes and
verification but excludes endpoint staging, final provenance construction and process startup. No warm-throughput
metric or extrapolated runtime estimate is emitted. The existing writer's missing MC/CRV surfaces, sentinel fields,
fixed F2 momentum and RNTuple reader limitations still apply (README, EventNtuple output).

## Local/FNAL runner

No installation or C++ build is needed with a compatible Python/JAX environment and `core:tracker` on `PYTHONPATH`:

```bash
python3 -m trkfastsim.kitpoint --gen GEN.root --gdml extracted_v04.gdml \
  --digi-cond digicond.npz --reco-cond cond_mc.json --drift-source STOCK_DUMP.root \
  --seed 1 --outdir NEW_OUTPUT --workdir /tmp --retain-ntuple
```

The local CLI accepts `--backend jax` for the experimental explicit-key digitiser; the default remains NumPy.
Transport stays NumPy, event retention is unchanged, and provenance records the chosen `configuration.digitizer`.
This option is not an endpoint task parameter: the registered task and its typed schema still select NumPy.
See [fast-digi.md](fast-digi.md) for resource limits, repeatability scope, pilot F2 agreement, A100 tests/timings,
and the remaining full-sample certification gates.

The CLI's explicit paths are for local validation; callers must supply compatible extracted-v04 MC conditions.
The endpoint task instead fixes these selectors in `kit_task.py`. `scripts/kit_point_smoke.sh` reproduces the
one-shard check using `M2FS_DATA`, `M2FS_CORE`, `PY_JAX`, and an otherwise free single-core host. Outputs must be in
a data area. `scripts/kit_inputs_audit.py` reports persisted generator keys and drift-domain diagnostics.

## Local F1 companion: not an evaluator

`python3 -m trkfastsim.kitf1` is a deliberately separate, **stock-conditioned diagnostic**. It reuses a saved
`reco-vr2f1 --own-states --save-tracks` result where available, or runs the existing harness on a bounded stock
candidate population. It cannot take the new fast digis as F1 candidates and does not write kit `metrics.json`:

```bash
python3 -m trkfastsim.kitf1 --dump STOCK_DUMP.root --conditions cond_mc.json \
  --tracks EXISTING_F1/f1_tracks.npz --outdir NEW_F1_OUTPUT --workdir /tmp
```

Without `--tracks`, `--max-tracks` defaults to 1000 and bounds the diagnostic; with saved fits their coverage takes
precedence. The saved source path must match and be own-state; existing `from_f1` also checks the reconstructed
candidate event indices. Legacy saved fits have no historical input-content binding; current hashes must not be
presented as proof of the inputs originally fitted. No stock reference rerun is required. Output:
`EventNtuple_F1_stock_conditioned.root` and `f1_diagnostic.json`, labelled with stock seed/hit/crossing conditioning
and the covered event population. There is no generated-event efficiency or endpoint F1 task.

## Verification

Synthetic tests exercise real digi/empty-F2/writer paths, zero-step/non-muon/multi-muon denominators, persisted
multi-track counts, source-change failure, drift domain, retention, stage errors, separate F1 guards, typed endpoint
admission, shell construction, cached-input digests, durable retry bindings, cleanup and concurrent recovery.
Existing reconstruction/writer/digitiser tests remain applicable. These are integration tests, not physics closure.
