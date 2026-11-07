# Saved Run 1A stock-track census and fixed-support overlap

This is a **read-only validation stage**, not another reconstruction candidate. It reads the saved
[measured LR continuation](run1a-lrdrift.md) and an existing EventNtuple; no simulation, digitization,
association, fit, derivative check or LR update is rerun. Reconstruction consumes neither stock
membership nor truth. All prior outputs and algorithms remain unchanged.

## Why coverage comes first

The saved entry benchmark's `tracker_entry_reco.fcl` includes stock `recoMC/OffSpill.fcl` and writes
`CentralHelixOutput`. The included configuration selects `CHPath`; the benchmark does not override
that selection. `tracker_entry_ntuple.fcl` reads this selected `mcs-CH.art` with `FitType: CentralHelix`.
The earlier full benchmark recorded1888 G4 selected reco/ntuple events out of52037 source events.
The local read-only comparison below now confirms1888 events/tracks in the saved ntuple; fetched
historical job logs independently report52037 processed reco events and1888 CHPath passes.

Consequently, the saved ntuple does not contain every development event. A missing key is not an
observed zero-track event, and event-key agreement alone does not authenticate a common digitization.
This stage keeps missing stock rows as `-1` and `stock_present=false`. Only keys actually recorded
in both inputs enter its event contingency table. It does not infer efficiency from missing rows,
normalize to a truth denominator or claim that the selected stock sample is unbiased. An eventual
selected-path yield comparison needs authenticated producing configuration, successful full-source
processing and exact source coverage; that is a separate contract.

## Data contract

`validation.stockcensus.read_stock` reads the **highest existing ROOT tree cycle**, never sums cycles
or falls back to an older valid cycle. It supports the stock TTree and synthetic RNTuple layouts.
Required fields are the full `run/subrun/event` key, track `status/goodfit/fitalg/nhits/nactive/ndof/
chisq/fitcon`, and recorded hit `plane/panel/straw/state`. All track rows, including failed and non-CH
tracks, all repeated/inactive hits and original per-event/per-track indices are retained. There is
no first-good-track selection. Raw floating diagnostics retain NaNs and sentinel values.

The stock helper maps `fitalg=2` to CentralHelix. `status>0` includes both `kalmanConverged` (1) and
`kalmanOK` (2); `goodfit` is a separate geometric flag. None is equivalent to the fast combined
continuous/LR numerical certificate. This stage reports these predicates separately. It never uses
`hit.state>0` as an active-hit definition, since negative and zero ambiguities can be active.

Full event keys must be unique integer triples. Malformed jagged alignment, missing required fields,
invalid straw IDs and discrepancies between `trk.nhits` and recorded hit counts raise rather than
silently dropping rows. Both stock count/fill helpers traverse the same KalSeed hit collection;
an ntuple without hit output is explicitly unsupported. These checks establish the census of the
**recorded** track rows, not completeness of filtered-away reconstruction objects.

Reads use bounded entry chunks. Default caps are1000000 events,1000000 tracks and10000000 recorded
hits; exceeding a cap raises, never truncates. ROOT decoding of one chunk precedes its flattened
track/hit census checks: these are not a byte-level decoder-memory ceiling.

## Overlap is not track matching

`validation.run1a_stockcompare` joins exact full event keys and retains **every same-event fast
group / recorded stock track pair**, including zero-support groups, failed fast fits, failed/non-CH
stock tracks and pairs sharing no straws. Both fast flight hypotheses remain present without a
winner. No score ranking, assignment threshold or one-to-one matching is performed.

For each pair, the diagnostic intersection is the set of distinct packed straw IDs in the fixed
fast support and **all recorded stock hits**. It records intersection size, both set sizes, both
fractions and Jaccard overlap. Repeated stock straw hits are deduplicated only for these set counts;
the original full hit census is preserved. The pair table is capped at2000000 and the expanded
hit/group work at4000000 cells, with no truncation. All operations are vectorized over events,
tracks, groups and hits; Python loops are over fixed fields or bounded input chunks only.

A shared straw is not an authenticated shared pulse. The counts are neither active-hit purity nor
a physical track match, and high overlap does not prove a common true particle. The displayed
at-least1/4/8 shared-straw counts are descriptive inventories, not tuned selections or success gates.
Momentum and covariance comparisons require separately justified track/pulse identity and common
reference surfaces; none is inferred here. Truth matching, when introduced, must use particle
identity, never nearest time or distance.

## Preservation and provenance

Every fast input array is copied unchanged, including all event/group/hit identities, empty lanes,
failed fits, both directions and optimizer traces. Every flat stock input array is also saved.
The LR output hash must match its producing report; its recorded event/hit census is checked against
the hash-bound measured-hit input. The comparator checks support uniqueness/counts, exact equality
of selected support to both saved LR hit masks, and consistency of saved combined-success flags
including analytic stationarity without certifying the fit again.

Reports record input/source hashes and distinguish current binding from historical provenance.
Current hashes do not repair missing original association-input or earlier output authentication,
or prove the stock loaded binary/config. No full historical optimizer replay is claimed.

```bash
python3 -m trkfastsim.validation.run1a_stockcompare \
  --lr SAVED_LR_DIRECTORY --stock SAVED_STOCK_NTUPLE.root --outdir NEW_COMPARISON_DIRECTORY
# Repeat with --chunk-entries 17 --compare-saved NEW_COMPARISON_DIRECTORY/diagnostics.npz
```

Outputs refuse overwrite. This Python-only stage requires no C++ build or new exporter.

## Validation status

Focused validation: **1407 passes in26 selected files**, warnings-as-errors; **236 new tests**
comprise76 reader and160 comparison tests, with the actual saved CH file used for the TTree test. This selection differs from the earlier1176-test set;
no CI/full-suite claim. Review found omitted analytic-stationarity and saved-LR-mask consistency
checks; both are now enforced with rehashed-payload regressions. Masked missing API values and
saved-comparison dtype changes also reject. The reader has synthetic malformed/empty/multi-track/
repeated-hit, highest-cycle and exact chunk regressions plus a real local stock-TTree schema test.
An independent tiny loop/set oracle checks every pair and repeated/reordered support. A separate
unrelated KinematicLine RNTuple
has exercised the full CLI against the saved LR file: its event keys have zero overlap, so all1000
fast events remain unknown rather than becoming1000 stock failures. That is an I/O/coverage smoke
check only, **not a Run 1A physics comparison**. Smoke output arrays are exact with4096/17-entry
chunks, including every historical fast array and all2156 stock KinematicLine tracks/42248 hits.
Smoke NPZ SHA256: `7729eabb3e191b8821d965c6484774c737d2f46cb9e4a10c52a0e6f8ed6c3b1d`.
The occupied GPVM was not a blocker for this analysis: the saved25 MiB ntuple and its existing
logs/timing/completion marker were copied into a nonsynced laptop cache, then processed locally.
No GPVM analysis, C++ build or new physics job was needed.

The subsequent [saved-stage and stereo-pair attrition audit](run1a-stockattrition.md) identifies
the three missing-seed stages and seven supported nonpasses without changing reconstruction.

## Saved CentralHelix development1000 comparison

The actual stock TTree `EventNtuple/ntuple;1` contains1888 events,1888 CentralHelix tracks and62204
recorded hits, with no repeated straw within a track. All recorded tracks have status1 and
`goodfit=1`. Exactly38 full event keys occur in the preserved1000-event fast development census;
the other1850 stock events lie outside it. Each of those38 events has one fast group and one stock
track, so the all-pairs table has38 rows. This one-to-one census is an observation, not a matching
rule imposed by the code. The summaries are pair-weighted; with multiple groups/tracks they are
not generally event-weighted or unique-track-weighted.

| Fast availability within the38 recorded CH events | Events |
|---|---:|
| At least one combined-passing flight hypothesis |28|
| Fixed seed support, valid objective, neither direction passes |7|
| No selected seed support |3|
| No group, or seeded group without a valid objective |0|

The28 passing groups contain40 passing direction hypotheses. Across the10 nonpassing groups,
the20 direction-lane labels are9 derivative-unstable,1 nondifferentiable,2 pending-LR,
2 sticky refit-numerical-failure and6 empty. Labels have precedence and are not disjoint physical
failure mechanisms. These counts expose both an association gap and numerical-fit gaps without
changing either reconstruction stage or selecting only its successful rows.

**Every one of the35 nonempty fast supports is a subset of the corresponding stock track's
recorded straw set**, including the7 numerically nonpassing groups. Five sets are identical.
For the28 passing groups, the shared support ranges14–46 straws (median27.5); the fraction of
stock straws covered has median0.9310 and minimum0.5833. Three passing sets are identical.
This is encouraging measured straw-support agreement on the selected sample, not pulse identity,
truth purity, a resolved flight direction or an efficiency measurement.

The962 development events absent from this selected ntuple remain unknown in this comparison;
194 have a passing fast group. They are **not**194 demonstrated additional tracks or fakes. The
fetched historical reco log records52037 processed events,1888 CHPath passes,50149 failures and
zero path errors; reco and ntuple logs/time files end successfully and `chain.done` says finished.
Those sidecars corroborate the selected-output history but are not a cryptographic ROOT/log/
loaded-binary/config binding and do not show whether rejected events contained usable tracks.

Accepted reports `stockcompare-ch-1000-20261106-2` and `stockcompare-ch-1000-batch17-20261106` have
exactly identical arrays, including every historical fast array and all stock rows. The earlier
`-1` payload is also exact; its report predates the added stage/overlap summaries. NPZ SHA256:
`a2cf25a609f4a04d74e8d32cfa114f0c404d18ac311cda74e1c748833f852b46`.
Stock ROOT SHA256: `3c3a07ac164808d6d76bd60722ae121e5eaf369781a49517063ff64c8380877d`.
Default-chunk local load/hash time0.664 s and comparison0.0226 s exclude output and transfer;
17-entry chunks take1.437 s and0.0239 s. These are reader/comparison costs, not reconstruction
throughput or a speedup against stock.

No stock-relative efficiency, fake rate, momentum bias/resolution, covariance coverage or matched
reconstruction throughput has yet been measured by this stage. The fast tracker still lacks
multiple-track separation/outlier treatment, field-map/material propagation and physical uncertainty
validation. The numerical LR baseline remains336 combined hypotheses in222 groups, not336 tracks.
