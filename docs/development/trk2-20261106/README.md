# TRK2 development preservation — 2026-11-06

This branch is a source/evidence archive, **not a merge-ready reconstruction release**.
Its parent is resident-reconstruction PR #24 (`370f9bd7ccc1648600b0a26bbb2b2097424c7835`).
PR #24 itself is unchanged. The archive retains the exact saved local development
files from `TRK2-run1a-digi` and `TRK2-benchmark`; `source-manifest.json` records their
SHA256 hashes. It also preserves the three publication handoffs and the four saved
cosmic/CE deck inputs, without changing the shared deck application or facts.

## Contents and relationships

- Measured-hit CPU stereo seeding, association, wire/timing/drift fits, analytic and
  left/right continuation, selected-stock comparison and attrition/pair diagnostics;
  corresponding tests, contracts, validation drivers and native drift-oracle source.
- Same-step Run1A digitisation validation, strict saved I/O, tests and provenance.
  The five-file validation-only change is published separately from current main.
- Tracker-entry replay, primary-only matched transport/downstream benchmark,
  adapter C++ source, FHiCL, scripts and Python verification.
- Historical local copies of the PU1 digitiser and tests. Their implementation is
  already merged in PR #22. **Do not merge these copies over current main:** main
  has a stronger off-spill golden test. Archive history starts at the older TRK2 base.
- Compact saved reports and test summaries. Large ROOT/NPZ/art inputs and outputs
  remain at their documented data/cache locations; they are not embedded in git.

The corresponding six-file Mu2eG4 cylinder-cut source is preserved in the
`RobMina/Offline` fork on `TRK2-outward-cylinder-archive-20261106`, from
`b24d4163e5e8f5ed2b5f72fccb25e4980fec64e2`. No upstream Offline PR is opened.
No umbrella submodule pointer is advanced to this development archive.

## Evidence and limits

Historical selected CPU diagnostic suite: 1,730 passes across 28 files, not full CI.
Digitisation validation: 89 focused passes, also with five pinned-main digitiser
modules. Saved refresh reproduces all four paired reports and the stock-control
comparison exactly; original report/ROOT/conditions files remain unchanged.
Resident tests/results belong to PR #24: 201 local passes; A100 snapshot 198 passes
and three optional CPU-parity skips. All 1,000 events/13,492 hits/773 groups survive;
39 groups numerically converge. This is not tracking efficiency or stock equivalence.

The richer CPU chain's 336/510 combined continuation certificates are not resident
results, tracking efficiency, calibrated chi-square or momentum resolution. Selected
stock output is not an efficiency denominator. Current provenance hashes do not
retroactively bind historical inputs to loaded configurations/binaries.

Known open issue: the forced eigensolve-failure synthetic case can overflow the
`run1a_stabledrift.py` summary division; its test locally suppresses that warning.
Historical source and saved numerical baselines are preserved, not silently repaired.
The single capped fast benchmark trajectory and field-value closure still need work.

## Acquisition helpers are historical, not portable tools

The four `run1a_digi_{reference,reseed}.fcl` / `run1a_digi_{export,controls}.sh`
helpers are deliberately excluded from the validation-only PR. Export expands a
caller-supplied configuration but executes another FHiCL, independently fixes the
run/IOV, and does not bind loaded builds/conditions to its input. Relative arguments
can change meaning after `cd`. Actual executed-config and reseed scheduling checks
remain required. Prior `bash -n` establishes syntax only, not new runtime validation.
Use saved tables for the validation-only PR's tests and refresh operations.

## Publication mechanics

FNAL mirrors were disconnected at session close. Publication therefore uses
isolated non-synced laptop clones and saved local source bytes; it is not a fresh
remote inventory or build verification. Original synced worktrees, unrelated work,
PR #24, saved baselines and PU1/TRK1 ownership remain untouched. No C++ build,
simulation, response rerun, association search or held-out tuning was performed.
Restore FNAL authentication/sync before reconciling the original dirty worktrees;
do not overwrite them or interpret this archive as a clean-main integration.
