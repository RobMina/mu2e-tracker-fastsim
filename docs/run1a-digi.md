# Run 1A digitisation development

The shared NumPy/JAX field-on and on-spill implementation is already on tracker main:
[PR #22](https://github.com/RobMina/mu2e-tracker-fastsim/pull/22), merged as `78350a1`.
This validation-only follow-up adds independent drift checks and a same-step off-spill cosmic
comparison. It introduces no second digitiser implementation or response-parameter changes.
The PU1 worktree is untouched. Device reconstruction is a separate
[PR #24](https://github.com/RobMina/mu2e-tracker-fastsim/pull/24), not part of this follow-up.

## Response inputs

The four campaign-specific acquisition helpers named below are **not included in this
validation-only change**. Their historical source is preserved in the
[development archive at `265cf57`](https://github.com/RobMina/mu2e-tracker-fastsim/tree/265cf57cb341a3dbf72f7e5a5113afd844cfed78/docs/development/trk2-20261106).
They are not needed for focused tests or saved-only report refresh. They require absolute
paths and further checks of executed configuration, loaded build, run/IOV and reseed scheduling;
the export script's supplied configuration is not the separate reference FHiCL it executes.
The commands below describe historical acquisition, not portable validated entry points.

Use the resolved stock digitisation FHiCL and run/IOV-matched `TrkPreampStraw`, `TrkAlignStrawSim`,
`TrkDelayPanel` and `TrkDelayRStraw` CSVs, the sample's GDML and global constants. Field-on conditions
require the actual stock Bz at the detector origin, and an explicit Mu2e-frame detector origin.
The nominal field and the extracted-tracker origin are not substitutes.

`run1a_digi_reference.fcl` reads existing digis with `StrawDigiTableDumper` and samples the field with
`BFieldPlotter`. It runs no transport or digitisation. The central point of the 3-by-3 field grid is
at the Run 1A detector origin (-3904, 0, 10171) mm; grid edge points are shifted by BFieldPlotter.
The stock CSV prints five significant digits, so its field sample is not a full-precision field closure.

`scripts/run1a_digi_export.sh AREA DIG_ART DIG_FCL NEW_OUTPUT [EVENTS]` uses an already-built area.
The supplied FHiCL must be the completed benchmark's Run 1A off-spill configuration, with
`Sim_best/v1_1` at run 1430:0 and `bfgeom_no_tsu_ps_v01`. The script exports that conditions set;
it is not a general campaign exporter. No build, input staging or baseline rerun is performed.

## Independent checks

`tests/test_digi_run1a_drift.py` compares both backends to an independent scalar translation of
stock `StrawDrift::D2T`, including stock float rounding of folded phi, radial extrapolation and
phi-bin boundaries. The Python interpolation retains double folded phi in float64 mode, so
bitwise stock equality is not claimed. The checked synthetic grid has 57,677 radius/angle points;
maximum differences are 3.15e-7 ns (NumPy/JAX float64) and 1.29e-5 ns (JAX float32), below 1e-4 ns.
This checks interpolation against the stock formula, not independent generation of its drift table.

Tests also retain zero-field arrays, reverse Bz, and exercise field-on off-spill propagation,
late hits, default event timing and event-window endpoints. Stock `EventWindowMarkerProducer`
sets both truth and reconstructed bunch time to zero off spill; its event length here is 100000 ns.
The field-on and on-spill switches are independent.

## Same-step comparison

`python3 -m trkfastsim.validation.run1a_digi` uses the complete event census of a saved detector-digi
dump, including empty events, and the corresponding rows from a transport table. Both stock and
fast must receive the same steps. Start with a development prefix; keep the remaining events for
later checks rather than adjusting response settings to match the prefix.

Report digi-count differences with paired event errors, plane occupancy, TDC/TOT/PMP and ADC shapes.
NumPy and JAX have independent random streams: do not demand eventwise equality to stock.
Same-seed repeats within a backend must reproduce output. Separate condition setup, first call,
repeat processing and writing; JAX on a CPU is not a GPU throughput measurement. A subsequent
stock reseed control is needed to establish the fluctuation floor.

The first 1,000 source events were read from the completed primary-only G4 benchmark without
rerunning stock response. They contain 14,881 steps, 13,492 stock digis and 235 zero-step events.
Using the sampled Bz = 1.0117 T, both fast backends retain the full census and the same empty events.

| Development result | NumPy | JAX on CPU, float64 |
|---|---:|---:|
| Digis | 13,514 | 13,463 |
| Fast/stock count ratio, paired SE | 1.001631 +/- 0.002239 | 0.997851 +/- 0.002335 |
| Plane occupancy diagonal chi2/ndf | 34.93/36 | 34.69/36 |
| TDC cal/hv diagonal chi2/ndf | 94.46/100, 84.35/100 | 109.13/99, 107.54/99 |
| TOT cal/hv diagonal chi2/ndf | 6.67/12, 6.40/12 | 7.82/12, 9.84/12 |
| PMP / ADC diagonal chi2/ndf | 10.71/13, 7.79/9 | 8.68/13, 8.75/9 |
| Identical same-seed repeats | yes | yes; no NumPy fallback |

Histograms include flows and paired event variance; these are unnormalised diagonal diagnostics,
not a global covariance-based test or held-out closure. The completed stock-reseed control is
reported below; the GPU run on this sample remains pending.
`--refresh-report REPORT.json --output NEW.json` recomputes comparisons from saved raw-digi
outputs without rerunning the response or rewriting the original report. Processing timings
remain those of the saved run. New reports hash input files and fast/conditions outputs;
refresh verifies hashes when present and records every consumed file hash. Historical reports
without producing hashes remain explicitly unbound: a current hash cannot authenticate their
original job, code or loaded binary. Full event and per-event waveform counts are checked before
flattening. Existing outputs (including input aliases) are refused by exclusive file creation.
Relative legacy reference paths are rejected; new runs record resolved absolute paths.
`--require-gpu` requires the selected JAX default device/backend to be GPU, but reported NumPy
fallback chunks remain possible; this flag is not a GPU-only/resident digitisation guarantee.

## Disjoint block and stock reseed control

The next 10,000 events (skipping the development 1,000) were exported from the saved stock digis;
full event-key checks confirm disjoint populations within the same explored parent. Both backends
used the fixed development settings, seed 20261105 and chunk size 32, without tuning.
The block contains 152,630 steps and 2,184 events with no steps.

| Result | Stock | NumPy | JAX on CPU |
|---|---:|---:|---:|
| Digis | 137,649 | 137,659 | 137,759 |
| Fast/stock ratio, paired SE | -- | 1.000073 +/- 0.000708 | 1.000799 +/- 0.000716 |
| Zero-digi events | 2,207 | 2,207 | 2,205 |
| Plane diagonal chi2/ndf | -- | 39.45/36 | 36.68/36 |
| TDC cal/hv | -- | 102.41/100, 105.51/100 | 85.28/100, 86.52/100 |
| TOT cal/hv | -- | 7.14/13, 12.49/13 | 11.33/13, 14.62/13 |
| PMP / ADC | -- | 20.58/13, 3.19/9 | 18.75/14, 12.57/9 |

JAX reports two NaN threshold-crossing ends under the documented stock-compatible search semantics;
no fallback or nonfinite saved detector payload is reported. Zero-step and zero-digi counts are different
quantities: the latter can change with the independent response random stream.

An independent stock seed (20261106) on the first 1,000 saved injected events produced 13,504 digis
against 13,492 in the saved seed: ratio 1.000889 +/- 0.002152. Its plane/TDCcal/TDChv diagonal
chi2/ndf are 31.37/36, 99.80/99 and 90.45/99, a development fluctuation control, not a held-out floor.
No Geant4 or baseline rerun was used. `run1a_digi_controls.sh` runs these two read/response jobs;
`--stock-control` compares the saved stock tables and checks disjoint IDs.
GPU execution for this cosmic sample remains pending; CPU processing times are not GPU projections.

No cross talk, noise-hit generation, status suppression, single-electron mode or digi-level mixing
is added. Native detector digis and ADCs are supported; full `StrawDigiMC` is not exported. Do not
connect the detector-only table to truth-dependent reconstruction and call it equivalent MC.

## PU1 profiling handoff

PU1 handed shared tracker-kernel optimisation to this thread; mixer/export optimisation remains with
PU1. The functional digitiser matches merged tracker PR #22 (one local comment differs).
Preserve that single shared implementation; do not republish local copies of the dependency.
Main includes a newer off-spill regression test, which this validation follow-up must retain.

The saved A100 development profile uses 2,741 on-spill events, 3,474,473 mixed steps, 48,798,407
clusters, seed 20261105 and chunk32. Uninstrumented warm calls took 91.657 and 90.456 s (29.91 and
30.30 events/s). The first process pass took 113.503 s with a populated compilation cache; it is
not cold compilation. Input/setup 4.556 s and ROOT writing 2.297 s are separate. All five repeats
matched every output array and the saved detector reference; no NumPy fallback occurred.

| Synchronized diagnostic component | Seconds | Fraction of its 90.329 s total |
|---|---:|---:|
| Ionisation/cluster generation | 45.643 | 50.5% |
| Work outside the top-level profiled stages | 33.015 | 36.5% |
| Electronics | 10.999 | 12.2% |
| Charge/drift | 0.671 | 0.74% |

These synchronized wrappers include dispatch/waiting; they are not a GPU timeline or a decomposition
of the separate uninstrumented pass. Cluster `_sample` accounts for 38.505 s, but random-position,
Poisson and minion-loop costs were not separately resolved. Host sorting, event association and
repeated array conversion are next candidates; overlapping profiler totals must not be summed.
No utilisation or memory-bandwidth bottleneck was established. Field interpolation and physical
ghost handling are not sensible first optimisation targets given their measured small contribution.

Start bounded probes with the saved dense development input, fixed precision/seed/chunk and exact
output comparison. Preserve full event IDs, explicit timing, per-end ghost boundaries, truth indices
and visible overflow/fallback. A changed chunk/RNG domain needs a separate physics comparison, not
an exact-repeat claim. Keep dense on-spill validation distinct from this thread's sparse off-spill
cosmics: neither the measured rate nor our disjoint-event result transfers between workloads. PU1's
dense held-out fast-digitiser validation and this thread's cosmic GPU measurement remain pending.

## Validation-only publication checks (2026-11-06)

The hardened driver and independent drift tests pass **89 focused tests**, including
masked/missing payloads, event-index bounds, waveform boundaries, producing-hash tampering,
legacy provenance limits, no-overwrite guards and empty controls. The same 89 tests pass
with the five shared digitiser modules fetched from main `7af3daa`, verified by their file
hashes and runtime import paths; no response module is changed by this follow-up. These are
selected local tests, not a separate full CI or cosmic GPU validation. The final five-file
publication also passes all 89 tests in 6.91 s in a clean tracker-main `7af3daa` clone
with core `2088ab2`; main's newer off-spill golden test and every response module are retained.

Saved-only refresh of the four development/disjoint-block NumPy/JAX outputs reproduces every
`paired` comparison entry exactly. A separate saved stock-control recount also reproduces
its full comparison exactly. Before/after hashes confirm all four original reports,
`fast.root` files and condition archives remain untouched. The reported historical timings
are preserved rather than relabeled as new measurements. No new detector simulation,
real-sample digitisation, held-out tuning, C++ build or exhaustive association was run.
