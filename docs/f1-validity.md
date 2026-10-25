# F1 numerical validity diagnostics

`reco-vr2f1` records `numerical_validity` in `vr2f1.json` and per-track Boolean checks in `validity.npz`.
This is **diagnostic only**: the fitter, `good` flag, stock FitOK selection and comparison masks are unchanged.
It is not a new production acceptance definition or proof that an extra F1 fit is physical.

## Contract

`trkfastsim.reco.fitvalidity.fit_checks` checks final returned effect states (masked by the input hit/crossing
validity arrays) and the fitted t0 state. It checks finite parameters, finite covariance entries, nonnegative
covariance diagonals, symmetry and PSD, finite chi2/ndof, nonnegative chi2 and positive ndof. A track must have at
least one represented effect. Padding is excluded. This does not cover intermediate iterations, unbiased hit
states, or trajectory pieces not represented by the returned states.

Covariances are scaled by the square roots of their positive diagonal entries; zero/negative diagonal entries use
unit scale. Negative diagonals independently fail. The eigenvalue test uses the symmetric part of this scaled
matrix, with an independent symmetry check and a numerical tolerance of `1e-10`. Nonfinite/overflowed scaling
fails closed. Small positive eigenvalues do **not** fail a condition-number cut. Eigenvalues alone are not a
covariance validity decision: use the combined checks.

Population counts retain the original flags: all, F1-good, both-good, F1-only, stock-only and neither. Failure
counts overlap. No fractions with fabricated denominators are produced. The helper supports empty batches and
zero effect axes; this does **not** establish empty-input support for the entire `reco-vr2f1` harness.

`validity.npz` contains the checks, original `good`/`stockok`, and the existing per-file `event_index`, `subrun`,
`event` keys. Use the JSON input path as part of identity; these are not globally unique multi-run track keys.
Use a fresh output directory when running a candidate. Historical `yield.npz` files contain only the transformed
t0 covariance (`Ck_front`, despite its name), not all returned effect covariances, so they cannot retrospectively
certify this full check. Current six-state fit throughput must be measured separately from this CPU diagnostic.

## Regression checks

From the umbrella on the laptop:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=core:tracker ~/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  tracker/tests/test_fitvalidity.py tracker/tests/test_reco_empty.py tracker/tests/test_reco_combo.py \
  tracker/tests/test_kalman_jax.py tracker/tests/test_reco_jax.py tracker/tests/test_kkmaterial.py
```

October 25 local result: **42 passed, 1 skipped** (missing KinKal ANN weights). Covariance tests cover positive
and indefinite matrices with positive diagonals, off-diagonal NaN, infinity, asymmetry, negative/zero diagonals,
weak positive directions, padding and empty arrays. Empty panel/time-cluster regression fixtures first reproduced
six IndexErrors; empty/all-rejected panel hits and empty/all-background time clusters now return typed empty
products. Nonempty behavior is checked by the existing NumPy/JAX comparisons. No real-input F1 rerun or GPU
validation was performed for this change.

## Matched-history trace blocker

Source inspection of `art/TrkFastSim/src/KinematicLineFitTrace_module.cc` and local KinKal `Fit/Track.hh` shows:

- Per-iteration reruns set `Config.maxniter_` for every retained extension meta-iteration, not only the last.
- `Track::iterate` calls `setStatus(candidate)` before swapping the candidate into `fittraj_`; unusable candidates
  are discarded. A failed trace row's `fitTraj()` parameters are the last accepted trajectory, not that candidate.
- Parameter-change chi2 uses reference pieces selected by the candidate endpoint ranges' midpoints. The stored
  previous front/back covariance is not guaranteed to be the exact reference covariance used in the test.

The next controlled diagnostic is an opt-in, passive rejection snapshot inside KinKal's actual
`paramsdiverged` decision, retaining double-precision candidate and exact reference endpoint states/covariances,
reference piece identities/ranges, parameter steps, chi2 thresholds, and actual iteration/stage identifiers.
Run the unchanged full extension schedule and compare capture-enabled/disabled histories and results first.
Splitting the schedule into new `extendTrack` calls is not an established equivalent alternative.

Inspected local KinKal was `2011f342bb3f233ebc11c4c57dea358e4af8d02f`; Offline was
`acc62589e19d122ba5880bc49a9f425f1e4f1887`. The tracker art setup names a CVMFS MDC2025ay backing; confirm its
actual headers/build before implementing this interface. No C++ source was changed and no build was started.
