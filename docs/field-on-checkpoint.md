# Field-on development pause checkpoint — 2026-11-05

This is a continuation checkpoint, **not a release, merge-ready change or production validation**. The default
extracted chain and the experimental wall option's default remain unchanged.

## Current result

The complete-wall cosmic development A/B is finished. Its material ledger reaches the actual loss/scattering/tail
consumers, but the cosmic step deficit and momentum-proxy core discrepancy remain. Reconstruction improvement is
not demonstrated. Keep `complete_outer_wall=False`; this is a Python API option, not a comparison CLI switch.

The [field-on record](field-on.md) contains the summary results, denominators, paired errors and limitations. The
cosmic reader now counts all active KinKal states (`state > -2`) and independently checks against persisted
`nactive` in each arm. The inherited CE reader **still uses the right-only `state > 0` subset**: historical and
current CE distinct-active-straw output remains uncertified until a saved-only re-audit. Persisted active-hit,
efficiency and momentum results are unaffected by that count definition.

## Archive and dependencies

Use branch `checkpoint/r2-pause-20261105` in both the tracker and core repositories. This tracker checkpoint
preserves the accumulated CE/cosmic validation tools and the inherited world prototype as well as the new wall
option. It is not accurate to attribute every preserved change to the wall patch.

- Tracker inherited base: `e13efe4d8b7b563d86da50320096474ae90dc54e`.
- Core inherited base: `c366e0c28a9c8c889731dfa3f5e89d6345a4515d`.
- Core checkpoint: `21c5bbadaa1a9970b762fed2867fb81d39a2a77d`, preserving pre-existing opt-in GDML material,
  cone/polyhedra and subtree-skip work. No core physics was changed for the wall pilot.
- Frozen wall-pilot `fieldsteps.py` SHA256: `6bc76fbcc44a2ea1d070d34bac9e58294ede77173ad607421c4bd81e73e6ccb0`.
- Frozen `fieldwall.py` SHA256: `2c446b8c855cd375eab19516300e8f61f480b320f7b815444130c0753096ae1a`.

These archive branches intentionally retain their historical bases; they are not rebased onto current main.
Main-only files absent at those bases are not new checkpoint deletions. The world prototype is unvalidated and
its delivery pilot failed; the CE target-to-entry hand-off does not complete general world transport.

Historical launcher files are portable **templates**, not byte-identical executed wrappers. They require
`FASTSIMKIT_SCRATCH` and an externally supplied scheduler output destination. Site-path cleanup changed only these
repository templates; immutable executed snapshots and physics products were not edited. The older dependent
comparison template is not the successful recovery route: the finish template performs saved-only reconciliation
of the already completed stock stages. A failed post-run wrapper must not be relabeled as failed physics or given
a fabricated completion marker.

## Checks at pause

- Local CE/field and adjacent selection plus site-path checker: **1,158 passed, 3 skipped, 34 warnings**.
  Warnings are the known synthetic empty-parent replicas. This is not full repository/main/device integration.
- Separate inherited core world/material/site-path selection: **17 passed**.
- Targeted publication-hygiene/history/wrapper selection: **19 passed** (overlaps the broader selection).
- Portable historical shell templates passed syntax checks; they have not been re-executed.
- The inherited optional `hependpoints` dependency failure remains a full-integration blocker.

## Resume gates

1. Re-audit CE distinct-active-straw reporting from saved products only; retain old right-only diagnostics and
   verify counted active rows against persisted `nactive` before publishing replacement straw claims.
2. Decide explicitly on valid frozen-candidate confirmation. All ten original cosmic parents have been examined;
   the wall development pilot is not untouched confirmation, and old CE results do not validate the wall option.
3. Keep general upstream world transport, field-on device validation, matched CPU/grid full-chain throughput and
   charged cost per event open. Diagnostic observer timings do not establish production throughput.

Do not regenerate the retained inclusive CE reference or rerun completed transport/reconstruction. No additional
Geant4, build, grid job, extra-parent transport/reconstruction or default promotion is authorized by this pause.
