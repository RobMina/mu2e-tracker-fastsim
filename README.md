# mu2e-tracker-fastsim

Geant4-free tracker simulation (`StrawGasStep`s) and fast tracker reconstruction for the Mu2e extracted (KPP, no
field) cosmic configuration: stage 2 of the Mu2e fast simulation, beside
[`mu2e-crv-fastsim`](https://github.com/RobMina/mu2e-crv-fastsim). Physics primitives (energy loss, materials,
crossings, tables) come from [`mu2e-fastsim-core`](https://github.com/RobMina/mu2e-fastsim-core). All three are
submodules of the umbrella repository [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim).

Status (2026-10-16): skeleton; technical plan in `docs/plan.md`; reference-table dumper `art/TrkFastSim`
(`StrawGasStepTableDumper`, see `art/README.md`); table readers and the PrimaryFilter tracker term in
`trkfastsim/tables.py`; CLI `python3 -m trkfastsim refcheck <tables>` (content check, stats JSON).

This repository holds technical software only: no physics results, no Mu2e-internal conditions, tables or data.

License: Apache-2.0 (`LICENSE`, `NOTICE`).
