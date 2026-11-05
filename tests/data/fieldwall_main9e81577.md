# Frozen default-wall outputs

`fieldwall_main9e81577.npz` contains both output tables and the complete final PCG64 state for the six synthetic
`run_case` cases in `tests/test_fieldwall_parity.py`: muon, tail off, radiative on/off, electron and slab support.
Each starts with 16 tracks (geometry seed 7100, transport seeds 7101–7106). There are 21 gas-step rows per case
(22 for electrons), plus 16 final-state rows. No detector files or real-data transport were used.

Source: tracker **main 9e81577**, with core **d041bb3**; generated with NumPy **2.4.6** and pandas **3.0.3**.
The independently loaded main source, not the implementation under test, supplies the transport function.

The compressed archive stores ordered, typed structured arrays for the tables and Unicode JSON scalars for
RangeIndex metadata, axis names and RNG state (including full 128-bit integers). No object arrays or pickles;
tests use `allow_pickle=False`. Normal pytest only reads the fixture and fails if it is missing or differs.

Schema, integer values and complete RNG state are exact. Float values use `rtol=1e-10`, `atol=1e-11` in each
column's native units, with NaNs rejected. This allows small BLAS/libm/NumPy rounding differences, not
physics-scale changes; cross-platform float bytes are not promised. CI uses NumPy **2.5.4**/pandas **3.0.6**.
Omitted versus explicit False remains byte-exact; the optional in-process main comparison is also byte-exact.
Negative controls perturb both routes identically: same-route parity passes, but the frozen fixture rejects
float, integer, dtype, column, index and RNG drift.

## One-time generation

From a tracker checkout, in the analysis venv with core pinned to d041bb3, extract the independent source:

```bash
git show 9e81577:trkfastsim/sim/fieldsteps.py > /path/to/main/fieldsteps.py
```

Load that file with `importlib.util.spec_from_file_location`, run the six `run_case(baseline, name)` cases, and
save `np.savez_compressed`: `{case}_{0|1}` holds `frame.to_records(index=False)`; `{case}_{0|1}_schema` holds
JSON `index_range` (start, stop, step), `index_name` and `columns_name`; `{case}_rng` holds the sorted-keys RNG JSON.
All arrays must have `dtype.hasobject == False`. This is provenance, not an instruction to refresh a failing test.

```bash
python3 -m pytest -q -p no:cacheprovider tests/test_fieldwall_parity.py
python3 tests/test_fieldwall_parity.py --baseline /path/to/main/fieldsteps.py
```
