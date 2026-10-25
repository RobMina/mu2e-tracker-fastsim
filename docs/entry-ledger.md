# Straw-entry ledger diagnostics

`steps.make_steps` accepts `entry_ledger="legacy"` (the default), `"energy"`, or `"vacuum"`.
These are **writer-only diagnostic arms on fixed trajectories**, not physics-tuning knobs or a corrected
production transport. The normal `steps` CLI does not expose them.

| Arm | Air term | Support term | What changes |
|---|---|---|---|
| legacy | Frozen entrance-momentum air dE subtracted from momentum | Existing kinetic-energy subtraction | Nothing; diagnostic capture is optional |
| energy | Same air estimate subtracted from kinetic energy | Same support loss samples | Written entry momentum, gas eIon sampling and stop filtering |
| vacuum | Air estimate recorded but not debited | Same support loss samples | Written entry momentum, gas eIon sampling and stop filtering |

The vacuum arm requires stage C and an interior model. Nonlegacy arms require `exit_scatter=False` and do
not supply path vertices: their momenta would otherwise disagree with the surface path. Secondary libraries
are rejected for these arms. No normal
surface/digi/reconstruction validation is implied. The diagnostic runner writes summaries, not ROOT
products for downstream reconstruction.

## Cumulative accounting

With `capture_entry_ledger=True`, `make_steps.entry_ledger` contains one row for each candidate straw entry
available to the writer (before the nonlegacy stop filter). Identity is `(run, subrun, event, sp, sid)`.
Fields include path from the interior start, start momentum/kinetic energy, cumulative support loss,
frozen-momentum air estimate, actual air debit, requested/written kinetic energy, written/legacy momentum,
energy closure defect and the emitted-row mask. Units are in field names (mm, MeV, MeV/c).

This is **not a per-material segment ledger**. Gas deposition is sampled later; wall, gas and wire loss are
not added to transport by this change. The legacy straight arm may already have removed stopped rows
before capture. The interior transporter never returns rows after its own support-loss range-out, so
those absent trajectories cannot be recovered by changing writer arithmetic.

Energy-space arms use the core momentum-from-energy routine and require remaining kinetic energy >0.5 MeV.
A stop is absorbing: a later entry of the same muon cannot reappear even if numerical cumulative losses
would make its energy rise. No new floor manufactures a transmitted momentum after energy exhaustion.

The generated-beta time calculation is deliberately untouched. Correct flight times need a path-segment
speed history, including the world-to-interior boundary, support loss, displacement conventions and exit
segments; replacing beta with each entry's final momentum is not an integrated-time fix.

## Comparisons and interpretation

Use `python3 -m trkfastsim.analyses.ledgercheck` on a bounded set of existing gen files and the matching GDML.
The three arms use the same input and initial random seed per file. Support trajectories are identical;
energy/eIon changes can alter subsequent gas random-number consumption. Report muon/event identity and
crossing conservation separately from selection changes. These comparisons bound the writer contribution,
not the remaining scattering or geometric loss of slow muons.

Existing G4, production and slab baselines must not be regenerated for this check. Record the exact tracker
and core revisions, dirty source identity, input order, geometry identity and source-event count before
interpreting a result. Full kinetic/deposited/escaping-energy closure and integrated timing remain open.
