# Run 1A digitisation development

The shared NumPy/JAX field-on and on-spill implementation is already on tracker main:
[PR #22](https://github.com/RobMina/mu2e-tracker-fastsim/pull/22), merged as `78350a1`.
This validation-only follow-up adds independent drift checks and a same-step off-spill cosmic
comparison. It introduces no second digitiser implementation or response-parameter changes.
Device reconstruction is a separate [PR #24](https://github.com/RobMina/mu2e-tracker-fastsim/pull/24),
not part of this follow-up.

## Response inputs

Use the resolved stock digitisation FHiCL and run/IOV-matched `TrkPreampStraw`, `TrkAlignStrawSim`,
`TrkDelayPanel` and `TrkDelayRStraw` CSVs, the sample's GDML and global constants. Field-on conditions
require the actual stock Bz at the detector origin, and an explicit Mu2e-frame detector origin.
The nominal field and the extracted-tracker origin are not substitutes.

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

Source of the table below: saved outputs of the stock export (not in this repository), stock
conditions `Sim_best/v1_1` at run 1430:0 with field map `bfgeom_no_tsu_ps_v01`, and the saved
reports of this driver run on them.

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

The driver refuses an existing output directory or report file. `--require-gpu` requires the selected JAX default device/backend to be GPU, but reported NumPy
fallback chunks remain possible; this flag is not a GPU-only/resident digitisation guarantee.

## Disjoint block and stock reseed control

The next 10,000 events (skipping the development 1,000) were exported from the saved stock digis;
full event-key checks confirm disjoint populations within the same explored parent. Both backends
used the fixed development settings, seed 20261105 and chunk size 32, without tuning.
The block contains 152,630 steps and 2,184 events with no steps.

Source of the table below: saved outputs of the stock export (not in this repository), same
conditions as above, and the saved reports of this driver run on them.

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

The stock-reseed numbers below come from saved outputs of the stock export (not in this
repository), same conditions as above. An independent stock seed (20261106) on the first 1,000
saved injected events produced 13,504 digis against 13,492 in the saved seed: ratio 1.000889 +/- 0.002152. Its plane/TDCcal/TDChv diagonal
chi2/ndf are 31.37/36, 99.80/99 and 90.45/99, a development fluctuation control, not a held-out floor.
No Geant4 or baseline rerun was used.
`--stock-control` compares the saved stock tables and checks disjoint IDs.
GPU execution for this cosmic sample remains pending; CPU processing times are not GPU projections.

No cross talk, noise-hit generation, status suppression, single-electron mode or digi-level mixing
is added. Native detector digis and ADCs are supported; full `StrawDigiMC` is not exported. Do not
connect the detector-only table to truth-dependent reconstruction and call it equivalent MC.
