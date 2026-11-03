# Tracker: device precision (float32 opt-in)

The device float comes from core, as for the CRV: `MU2EFS_DEVICE_FLOAT` (`mu2efastsim.precision`, float32 by default).
The tracker adds one override, `TRKFASTSIM_DEVICE_FLOAT`, which keeps it at float64 until its float32 validation is
accepted:

| `TRKFASTSIM_DEVICE_FLOAT` | Tracker kernels |
|---|---|
| unset or `float64` (default) | float64, byte-identical per seed to the code before the setting |
| `core` (opt-in) | follow `MU2EFS_DEVICE_FLOAT`: float32 by default, float64 if that says float64 |

Following core in float32, the step and digitiser modules register with core's precision module (`dprec.register`) and
their kernels are `dprec.jit` kernels. Output provenance (`devfloat.provenance()`) records core's device float,
determinism setting and device model, and the tracker's effective float and setting. Code: `trkfastsim/devfloat.py`.

## What follows the setting

| Stage | Module | float32 opt-in | Why |
|---|---|---|---|
| world column, ordered loss, scattering to the tracker plane | `sim/world_jax.py` | yes | |
| interior transport, exit scatter, stage-A crossings, gas eIon | `sim/steps_jax.py` | yes | |
| ionisation clusters | `digi/digi_clusters_jax.py` | yes (the standalone gain and wire helpers stay float64) | |
| charge, drift, electronics (thresholds, TDC, TOT, ADC) | `digi/digi_jax.py` | yes | |
| straw hits, combo hits | `reco/hitreco_jax.py`, `reco/combo_jax.py` | no | ports of stock in stock's mixed float/double, bitwise equal to stock |
| line seeds | `reco/linefind_jax.py` | no | bitwise equal to stock `LineFinder` |
| F2 line fit | `reco/linefit_jax.py` | no | normal equations on Mu2e-frame positions and absolute t0 would need centring; small share of the chain |

## Precision-aware forms (float32 only)

Each form is chosen at trace time, so the float64 programs are unchanged.

- **Positions.** The step kernels work in a frame centred on the straw centroid, rounded to whole mm. The world and
  interior tables (boxes, planes, shells, calorimeter pieces, straws) are shifted on the host in float64 and the outputs
  are shifted back. In the Mu2e frame the tracker sits at z of about 24 m, where the float32 spacing is 2e-3 mm.
- **Straw chords.** The entry and exit of a line in a straw are solved about the point of closest approach. The usual
  roots of A s^2 + B s + C cancel when the line origin is about a metre from the straw. Before this change a
  float32 run flipped 0.27 % of near-tangent crossings against NumPy in the test geometry; after it, the crossing
  lists are identical and the largest entry/exit difference is 2.4e-4 mm.
- **Point in gas.** The distance from the wire axis comes from the perpendicular vector, not from |w|^2 - (w.a)^2.
- **Digitiser positions.** Cluster positions are relative to the straw middle on the device.
- **Digitiser times.** Absolute times reach 1e5 ns, where the float32 spacing (0.008 ns) is 0.4 TDC counts. The charge
  kernel works relative to whole 20 ns periods. The electronics work relative to a per-straw reference that is a whole
  number of ADC periods (one ADC period = 1024 TDC counts = 4 TOT counts = 200 response bins). TDC, ADC phase and the
  window bounds are integer offsets of the relative values, so they are exact.
- **Integer arithmetic.** Integers are cast to the device float before they meet a Python float. Otherwise
  `int * float` gives float64 inside a float32 kernel.

`TRKFASTSIM_DRAW64=1` is a float32-mode diagnostic. It draws the digitiser's random numbers in float64 and rounds them,
so a float32 and a float64 run of the same seed differ only by kernel arithmetic.

## Validation

See [validation.md](validation.md#device-precision). Tests: `tests/test_devfloat.py` (float32 in subprocesses).
