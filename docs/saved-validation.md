# Relocated saved-product validation

`secondary_tv3_site_rescore.py` reruns current comparison code on an immutable saved campaign after relocation:

```bash
python3 scripts/secondary_tv3_site_rescore.py SAVED_COMPARISON.json \
  --original-root ORIGINAL_DATA_ROOT --staging-root STAGED_DATA_ROOT --outdir NEW_DIRECTORY
```

The comparison imports `crvextrap.io` for identity-aligned surface arrays. Put a pinned checkout of
`mu2e_track_crv_mcresponse` on `PYTHONPATH` alongside core, CRV and tracker. The analysis uses numpy, awkward,
uproot and scipy; the rate-ledger Parquet replay additionally requires pyarrow or fastparquet.
Install `trkfastsim[parquet]` for these analyses, or `trkfastsim[test]` for the full test dependency set.

### Additive Parquet environment for an existing image

For an immutable project image without a Parquet engine, `scripts/install_shared_parquet_uva.sh`
installs a binary-only PyArrow wheel into a new shared directory without replacing the image or its
NumPy/pandas/JAX packages. It requires a Slurm allocation, an owned evidence directory, an existing
image and the verification script. The current deployment additionally checks the completed tracker
snapshot's provenance tests before publication. It refuses an existing target, retains the wheel,
SHA 256 manifest and package/image receipts, and publishes a read-only version only after validation.
Use a new directory for any upgrade; do not mutate an environment used by other jobs.

```bash
source "$SHARED_PARQUET_ENV/env.sh"
apptainer exec --cleanenv \
  --env "PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$FASTSIM_PARQUET_SITE_PACKAGES:$CODE_PATHS" \
  "$SIF" python3 -m trkfastsim.analyses.ratearms compare ...
```

The environment is shared across jobs and repositories, not a test-only fallback. It is specific to
the image's Python ABI/platform. `parquet_environment_check.py` tests nullable and empty DataFrame
roundtrips and records actual package versions and the imported engine's location. A future image
rebuild should add PyArrow to both container recipes; this deployment does not rebuild an image or
claim GPU/physics validation. Scratch purge policies still apply to the deployed directory.

The staging root mirrors the relative original paths. Every required input must exist inside it; missing or
escaping/symlinked paths are rejected. Input bytes are hashed before and after analysis. The saved comparison
is never rewritten. Differences ignore **only** the necessary path relocation; all other changed leaves are
retained. Standard `tv2compare` retains highest-cycle event coverage checks, full event identities, paired
selection/content covariance, event-clustered shape errors and matched file-group width jackknives.
This is current **analysis** on historical simulation/digi/reco products, not current-transport production.

## What the saved campaign can and cannot certify

The saved default-library campaign has 10 million generated events, 55,432/56,254 fast/G4 tracker-selected
events and 50,271/51,317 tracks. Historical yield is -2.0383 +/- 0.3661%, selection-only -1.7187 percentage
points and both-selected content -0.3196 points with covariance retained. These are conditional on one fixed
secondary bank. Rescoring cannot make a historical sample represent a newly chosen material navigator.

Event-cluster errors describe recipient-event variation **conditional on the bank**. Donor reuse across
recipient events and finite-bank uncertainty are not an extra Poisson term or a count-based inflation factor.
The saved marginal occupancy/group moments do not retain a donor-index contribution matrix, retrained banks,
or digitisation/reconstruction for alternative banks. They therefore cannot recover an unconditional error
for track yield, TDC or occupancy. A defensible next calculation needs event-clustered donor resampling within
the sampler's matching/fallback strata, replay of zero-secondary donors as well as nonempty donors, and the
same response/reco chain; nested recipient and bank replicates separate Monte Carlo and finite-bank effects.
Draw concentration is useful evidence but is not that variance estimator.

The exterior-only topology arm is a **lattice-only, truth-conditioned diagnostic** selected after examining
the same 20 evaluation groups. Its 1–2-plane and nearby-panel deficits remain approximately -13.87 +/- 2.15%
and -14.46 +/- 1.56%; quoted errors are fixed-bank group jackknives. The covariance-score difference relative
to default is -0.0292 +/- 0.1833, not significant evidence of improvement. Splitting or relabeling those already
examined groups is not an independent holdout. No saved exterior-only endpoint/digi/reco campaign exists.
Do not promote topology or request production without a separately approved validation plan.

`secondary_occupancy_jackknife.py` can rescore the retained moment archive without rereading physics payloads.
It recomputes covariance and pseudoinverse for each deletion and requires stable support/rank. This remains
conditional-bank sensitivity, not a replacement for the missing bank replicas or independent holdout.

## Conditional G4 interior checks

`ordered_g4_entry_replay.py` starts at a saved reference first-straw position/time and its **mean
StrawGasStep momentum**, so it is a truth-conditioned diagnostic rather than a fully specified G4
boundary state. Full particle/straw identities, ambiguous recrossings, all endpoints and refusal reports
are retained. `ordered_g4_endpoint_audit.py` compares the saved terminal states by full identity and
reports finite-path censoring separately from positive-cutoff stops. The G4 particle end can occur beyond
the requested fast path; a survivor at that artificial limit is not a physical escape or a false stop.
Endpoint distances compare world-frame positions; G4 straight displacement is never called an arc range.

`ordered_geometry_exposure.py` inventories unsupported/approximate placements along actual traversed
slices using the explicit preserved `navigationcensus.py` source. Conservative bounding boxes generate
candidates only. `ordered_omitted_solid_audit.py` consumes the preserved `navigationcensus.py`,
`omittedpath.py` and `torusprobe.py` as explicit hash-recorded inputs, evaluates supported torus/nonconvex
prism predicates and retains unresolved hierarchy/caps. It never silently inserts material or calls
an AABB chord a missing-material loss. A no-hit result on a bounded trajectory set is not global
GDML completeness; retained approximate polycones and Boolean ownership require separate work.

## Source-event bank replicas and bounded response propagation

`bankreplica.replica` is an opt-in helper: resample eligible full source-event identities within their
original momentum bins, retaining empty secondary donors. One donor multiplicity applies jointly to its
entire in-straw crossing collection and off-axis content. Separate ancestry tables retain the original
run/subrun/event and crossing IDs; synthetic copy identities keep repeated donors distinct. It does not
change the production samplers' measured matching variables/fallbacks or invent correlations absent from
the default component draws. The bank's eligibility conditions are retained; this is not raw-source-event
or training-file resampling and does not include geometry/physics model uncertainty.

`secondary_bank_replicas.py` gates exact saved-default replay and exact all-once donor identity cloning,
then nests independent draw repeats inside each bank replica. Negative method-of-moments donor variance
estimates remain visible: clipping to zero is not evidence of no bank uncertainty. The recipient events
are fixed; errors are not full unconditional T-V3 yield errors. Every primary and generator/subrun/surface
table is protected, and source-event ancestry is written even for empty donors.

If a platform changes only secondary placement coordinate arithmetic, the historical exact gate still
fails. Explicit `--current-reference` records that failure, requires exact noncoordinate, primary and
particle payloads, records every coordinate difference, and compares against a separately recomputed
current reference. It never overwrites or declares exact historical output. This is not a tolerance-based
bypass: changed eIon/time/keys/counts or primary coordinates still refuse.

`secondary_bank_response.py` propagates a bounded first baseline-selected recipient set through the existing
NumPy digitizer and fast F2 reco, retaining all chosen event identities, including empty digi/fit events.
It reuses real saved conditions/drift tables; response/reco code is not edited. Stored bank ancestry is
reused across nested draw trials. Conditioning on baseline selection and using fast F2 rather than stock
KKLine means this is an implementation pilot, **not** the requested full saved T-V3 stock-reco uncertainty.
Its selection, seeds and bounded RNG realignment are explicit; no result is an independent exterior holdout.

## Actual ordered output consumers and domain location

`ordered_domain_audit.py` uses all saved full-chain vertices, muon identities and emitted gas entries, plus the
hash-bound actual omission witness. It reports full forward intervals and overlaps of actual recorded spans
with the loader-selection AABB and the named VD cylinder. Outside approaches, tangencies, misses, terminal
hypothetical rays and unresolved omission bands remain distinct. Neither domain is a GDML completeness claim,
a transport clipping policy or a physical world boundary. No-gas lanes remain in the coverage table.

`ordered_response_summary.py` consumes saved response trials and domain lane coverage without rerunning any
physics. Its draw-to-draw errors are conditional on the fixed recipient sample and describe response MC only.
Independent-arm mean contrasts are not paired event errors, finite-bank uncertainties or equivalence tests.

The stock-consumer route must first establish that the existing trusted runtime contains the tracker table
producer plugin. `ordered_stock_inventory_uva.sh` records the tarball hash and relevant paths on a Slurm CPU
node without extracting, rebuilding or running physics. File presence alone is not ABI/loadability validation.
A CRV-only tarball is not a tracker runtime. Before actual stock injection, independently prove complete
source-prefix generator coverage and retain the configured PrimaryFilter selection; never process an entire
shard with a partial 1000-event table. The all-generated denominator in the NumPy probe does not override that
stock filter or certify stock reco compatibility.

The exact prefix gate also needs an independently requested event count, a successful source-dumper log and
the trusted source-art SHA 256. Two identically truncated generator tables are not sufficient evidence of a
complete prefix. The saved-source gate can be rerun without another art invocation. Saved slice readers reject
negative/out-of-range lane indices, duplicate primary keys, nonpositive gas arcs, nonfinite or negative deposits,
nonunit directions and inconsistent individual endpoints before constructing diagnostic response rows.

The existing historical T-V3 tracker tarball can be reused on UVA; the separately staged CRV-only tarball
lacks the tracker injection plugin. Stock output auditing uses the actual highest-cycle selected event keys and
retains trackless rows. It deliberately does not call `livetime()` on a partial-subrun smoke or fabricate a
subrun exposure table. A small successful stock run establishes bounded consumer compatibility, not rate,
secondary-model, stopping or Geant 4 agreement.

The bounded smoke accepts up to 10000 whole generated events for a larger actual-navigation check, retaining
all slice arrays and model-stop identities. Its paired saved-G4 filter census verifies the retained reference
generator payload first and explicitly reports how many reference events are present. A filtered G4 table's
retained generation rows are not a full generated denominator or independent holdout. No runtime extrapolation
or precision certification follows from the larger smoke.

### Forward-handoff guarded producer evidence

`scripts.ordered_guarded_producer` retains all requested generated events and every primary identity before
transport. `scripts.ordered_guarded_audit` requires completed or explicitly refused receipts, validates complete
handoff identity order/dispositions and world-plane/start-clock continuity, and binds every fresh query's local
lane map to the delivered full-primary keys. Successful physics products require full ledger conservation/time
checks; refused batches must have no published ROOT physics sample. Missing handoff evidence is labelled
unavailable, never counted as zero delivered events. The old lower-plane baseline is not a same-policy control:
a raised-plane stored control is separately named, and no statistical precision follows from this bounded run.
