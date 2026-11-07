# Run1A digitisation validation-only follow-up (prepared, not published)

## Purpose

Publish the independent field-on drift checks and saved same-step off-spill cosmic
comparison tooling without republishing the shared implementation already merged
in tracker PR #22. Keep resident reconstruction PR #24 unchanged during review.

## Candidate scope

- `tests/test_digi_run1a_drift.py`: independent scalar mixed-precision D2T oracle,
  zero-field identity and field-on/off-spill endpoint contracts.
- `trkfastsim/validation/run1a_digi.py`: complete-census same-step comparison,
  paired event uncertainties, saved-only refresh and stock reseed/disjoint checks.
- `trkfastsim/validation/run1a_digi_io.py`: strict per-event schemas, missing-value
  rejection, file hashing, hash verification and exclusive report creation.
- `tests/test_run1a_digi.py`: independent synthetic tables, mocked response, failure
  contracts and provenance/immutability regressions.
- `docs/run1a-digi.md`: supported scope, saved results and reproduction.

Potential additional scope after separate script/FHiCL review:
`art/TrkFastSim/fcl/run1a_digi_reference.fcl`, `run1a_digi_reseed.fcl`,
`scripts/run1a_digi_export.sh`, `run1a_digi_controls.sh`.
Shell syntax checks pass, but these were not rerun in this continuation. The export
configuration is intentionally fixed to the historical run/IOV/geometry and must
not be advertised as a generic conditions exporter.

Do not include local copies of digi kernels or PU1 tests: their functional changes
are on main; main has a newer golden regression. Also exclude all CPU reconstruction
studies, native drift oracle work and PR #24 files. A new main-based publication
branch/area is needed; do not append these changes to the reviewed PR #24 branch.
No branch transition, staging, commit, push or new PR occurred in this continuation.

## Validation and evidence

- 89 selected tests pass locally (78 driver/11 drift), warnings-as-errors.
- Same 89 tests pass against pinned current-main `7af3daa` digitiser modules, verified
  by runtime import paths and file hashes (9.30 s). Not full CI or real-sample re-digitisation.
- Strict saved refresh reproduces all four existing NumPy/JAX development/disjoint
  `paired` reports exactly. Original report/fast ROOT/condition archive hashes stay
  identical. Saved stock-control comparison also matches exactly.
- Reports cannot overwrite an existing input/output. Refresh writes NEW output,
  checks available producing hashes and per-event waveform lengths, and records the
  source report/consumed bytes. Legacy reports remain explicitly unbound; hashing
  them now does not prove original loaded code or historical job/config linkage.
- Selected GPU default backend/device is checked; NumPy overflow fallback remains
  visible and permitted, so this driver is not a GPU-only/resident API.

## Remaining publication checks

Review optional exporter scope and resolve the new main-based branch without
moving or resetting the dirty TRK2 worktree. The shared code itself needs no new
implementation PR. No physics acceptance, new cosmic GPU validation, complete
StrawDigiMC or matched end-to-end speedup is claimed.
