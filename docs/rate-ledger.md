# Transport ledger arms and the PrimaryFilter tracker-term rate

The stage C transport keeps the muon's energy, its straw deposits and its step times in several places. Before this
change the conventions were:

- the step writer subtracted `dedx_restricted(AIR) x path` from the **momentum** of every straw entry, although the
  tracker envelopes are `DSVacuum` in the GDML and the interior transport has no gap material;
- the support loss (sampled, `slab_loss`) was charged to the kinetic energy;
- the straw walls, gas and sense wire scattered the muon but took **no energy**;
- the step time used the **generated** speed for the whole path, `t = t_gen + path / (beta(p_gen) c)`;
- the world column above the plane (core `transport.world_loss`) passed `p - loss` (momentum minus an energy) to the
  next loss call before core `2bcefde`; current core main converts the remaining kinetic energy to momentum.

`trkfastsim.ledger` names each convention as a separately switchable diagnostic arm. Thin loss changes transport;
writer loss and the time convention have their own switches. **These arms do not yet constitute a unified
energy/deposition/time ledger. T3 remains open.** Terminal bookkeeping and step/surface clock consistency fail
explicit tests below. Production defaults remain unchanged; no arm is recommended for promotion.

## Arms

| Arm | Setting | Convention |
|---|---|---|
| writer | `momentum` (legacy) | air estimate subtracted from \|p\| (writer only) |
| | `energy` | the same air estimate debited from the kinetic energy (writer only; `entry_ledger="energy"`, exit scattering off) |
| | `none` | no writer debit: written \|p\| = transport \|p\| (vacuum convention of the GDML) |
| thin | `False` (legacy) | walls, gas, wire scatter only |
| | `True` | unrestricted mean loss (`eloss.dedx_restricted(mat, p, 1e9) x length`, no random numbers) of the previous straw's walls, gas and wire (and of a gap medium, none by default) charged to the kinetic energy from the next straw on, before the stopping, scattering and kink momenta |
| time | `generated` (legacy) | `t_gen + path / (beta(p_gen) c)` |
| | `integrated` | world: per height band; interior: per segment between straw entries; exact flight time for kinetic energy linear in path, `dt = L (E_in + E_out) / (c (p_in + p_out))`; displacement jumps of condensed scattering take no time; path vertices use a separate kink-node approximation, not a common clock |
| world (core) | core before `2bcefde` | `slab_loss(p - loss)`, air at `p - loss` |
| | core `2bcefde` and later | `slab_loss(p(T - loss))` |

Presets (`ledger.PRESETS`): `legacy`, `E`, `A` (writer none), `M` (thin), `T` (integrated time), `AM`, `AMT`.
`steps.LEDGER_ARMS_DEFAULT` is the preset used when `make_steps` gets `arms=None` in stage C with the interior
transport (entry-ledger diagnostics always run on the legacy transport). The `steps` CLI takes `--ledger-arms`.
The world arm is the imported core; `ledger.core_world_convention()` records which one and the hash of its
`world_loss` source in every `ratearms` output.

Defaults are bit-identical to the transport before this change when the preset is `legacy`: no arm draws random
numbers, and the extra bookkeeping only adds zeros (checked on a full g4ref shard with `scripts/tables_identical.py`
against the previous commit; `tests/test_rate_ledger.py` checks it on synthetic straws).

## Per-segment ledger

`make_steps(history=True)` sets two tables (units in the column names):

- `segment_ledger`, one row per emitted primary-muon step = the segment from the previous entry of the same muon (or
  the interior start) to this straw entry: `path_mm`, `seg_mm`, `ke_prev_MeV`, `ke_entry_MeV` (transport),
  `ke_written_MeV` and `writer_debit_MeV` (transport minus writer), the cumulative `lost_support_MeV`,
  `lost_thin_MeV`, `l_support_mm`, `l_thin_mm` (walls + gas + wire of the straws crossed before), `l_gap_mm`, the gas
  chord `gas_mm`, the deposit `deposited_gas_MeV` (eIon, restricted at the tracker delta cut, sampled), the escaping
  estimate `escaping_gas_MeV` (mean unrestricted gas loss minus the deposit; an expectation, not an event-by-event
  closure), and the time in both conventions;
- `muon_ledger`, one row per generated muon: \|p\| generated and at the plane, alive at the plane, interior stop flag,
  time at the interior start, and **last-restart** energy/path/loss fields. The historical column names
  `ke_final_MeV`, `path_interior_mm`, `lost_support_MeV`, `lost_thin_MeV` are misleading: the final non-kink/stop group
  is not committed. Do not use these fields as terminal energy or total material loss.

The world column has no per-material deposit (the deposited and escaping terms there are unmodeled); the passive
interior material deposits all its loss locally in this bookkeeping (delta-ray escape from it is not modelled).

Analytic limits (`tests/test_rate_ledger.py`): flight time at constant momentum exact; linear kinetic energy against
quadrature (1e-8); additivity under subdivision; finite time to a stop; world-band entry/exit momenta reconstructed
from the band means; vacuum world time; vacuum writer (no debit); energy writer debits kinetic energy; one straw's
walls and gas charged exactly to the next straw (single material per piece); legacy preset equal to the default;
step and path-vertex times equal at the straw entries at constant momentum; G4-to-fast muon identity.
The strict xfails of `tests/test_step_energy_ledger.py` document the legacy defects (they test the default, which is
unchanged); the arm tests above check the corresponding expectations under A and T.

### Open T3 requirements (audited 2026-10-27)

`tests/test_rate_ledger_limits.py` adds three strict expected failures and three passing controls:

1. **Terminal accounting:** three 80 MeV/c synthetic straw entries reach 1020 mm and 0.036911 MeV thin loss,
   but the muon ledger reports zero path/loss. These are restart fields, not a final accounting domain.
2. **Absorbing stop:** a 12 MeV/c muon emits only the first straw, suppressing the entry at 1010 mm, but its path
   vertices continue to 4000 mm. Downstream surfaces and exit transport must terminate at an explicit stop node.
3. **Common clock:** a deterministic 5 MeV support loss produces about 203 ps disagreement between step and vertex
   times. The defect occurs without thin loss; the earlier sub-ps thin-material test does not bound it.

The thin-loss arm evaluates straw material at the momentum at the start of a kink group, not each evolving entry
momentum. At 20 MeV/c in a three-straw fixture, the second straw is charged 0.138067 MeV using group-start momentum,
versus 0.146894 MeV at its 19.2438 MeV/c entry momentum. Group subdivision and stopping convergence remain unvalidated.
The last straw's loss is not charged if no later entry exists. Exit-scatter losses are outside the reported totals.

`escaping_gas_MeV` is mean unrestricted loss minus a separately sampled deposit; it can be negative and is not a
physical escaping-energy measurement. World/passive-material deposited and escaping terms are not explicitly closed.
There are no complete material-identified segment/end/stop records. The entry-to-entry linear-KE timing helper is
analytically correct, but its nodes differ from the kink-node surface clock. Completion requires a shared physical
segment history, explicit terminal/stop nodes, energy/deposition accounting and a single timing interpolation;
changing material widths or adopting an arm because its rate is unchanged would not resolve these requirements.

## Rate measurement

`python3 -m trkfastsim.analyses.ratearms` (module docstring): `g4` (per primary muon of the G4 reference), `fast`
(one arm on gen tables; file i of the 220 g4ref_v1 + g4ref_v2 gen tables uses the generator seed `(1, i)` in every
arm), `compare` (paired tables), and `check` (validated restart against exact input hashes and physics settings).
`scripts/rate_ledger_arms.sh` defaults to one worker and enforces one worker/library thread on GPVMs; failures
propagate. Existing chunks are skipped only after validation; partial files and logs are preserved and rejected.
`compare` checks chunk settings, input hashes/indices/counts, duplicate identities, retained-event/pass counts and
G4 identity coverage. New runs also record platform/library versions. Generated event counts come from unique
ROOT gen-table event identities per completed input, not job-log counters.

The tracker term here is evaluated over every **primary muon**, taking the maximum good-step count per particle,
then one pass per event. This is not the full PrimaryFilter, which accepts other particles and the CRV/calo OR.
The historical `fatecheck` convention (fast first muon against G4 any muon) is reported separately for reproduction.
`fatecheck` now uses any primary muon on both sides and exposes `fast_first_pass`; `seedcheck` compares only the same
seeded SimParticle and keeps `g4_any_pass` separately. Generated-momentum/world-column bins are truth diagnostics,
not response/correction models. They use the first generated muon, not necessarily the passing muon.

**Muon identity.** A G4 primary's SimParticle key is its index among all the event's generated particles + 1 (CRY
also makes p, n, e, gamma primaries); the `genidx` column of the G4 particles table is 0 throughout. The fast key is
the muon's rank among the generated muons. `compare` maps every G4 muon to the fast key through the generator index
(`ratearms.g4_fast_keys`); first muon = fast key 1. `scripts/rate_conventions.py` gives the pass counts per
convention, the offset of an earlier `fatecheck` run, and the seedcheck CRV subset recounted with the seeded muon.

`scripts/rate_ledger_campaign.sh` runs the G4 table and the default arm set on a GPVM (site.env sourced). Arms compared
with each other must run on one platform: the same code and seeds on Linux and macOS differ in about 1.3 % of the
passing events (floating-point divergence of the stochastic transport), -0.02 % +- 0.05 % in the rate.

## Laptop results on the paired 11 M sample (2026-10-26)

Reference: G4 `g4ref_v1` + `g4ref_v2` tracker tables (220 files, 11 M generated, 61,864 events passing through any
primary muon). Fast: tracker `56e66bf` + this working tree, core `4f57013` (world convention `energy`), extracted v04
GDML, generator seed `(1, i)` for file i, stage C defaults, primary muons only, no secondary library.

| Comparison | Events | fast/G4 - 1 or arm/legacy - 1 (paired error) |
|---|---|---|
| legacy vs G4, any muon | 61,328 / 61,864 | -0.87 % +- 0.32 % |
| legacy vs G4, first muon vs first muon | 61,039 / 61,572 | -0.87 % +- 0.32 % |
| legacy vs G4, fatecheck convention (fast first, G4 any) | 61,039 / 61,864 | -1.33 % +- 0.32 % |
| 2026-10-23 fatecheck run (v16) vs G4, same convention | 60,792 / 61,864 | -1.73 % +- 0.32 % (reproduced) |
| current legacy first muon vs that run | 61,039 / 60,792 | +0.41 % +- 0.32 % |
| A (vacuum writer) vs legacy | 61,327 / 61,328 | -0.00 % +- 0.01 % |
| E (energy writer) vs legacy | identical | 0 |
| M (thin-material loss) vs legacy | 61,311 / 61,328 | -0.03 % +- 0.04 % |
| T (integrated time) vs legacy | identical | 0 (no time cut in the filter) |
| AM, AMT vs legacy | 61,310 / 61,328 | -0.03 % +- 0.04 % |
| world convention `momentum` (core `a554571`) vs legacy | 61,263 / 61,328 | -0.11 % +- 0.32 % (not draw-paired) |

G4 has 292 events passing only through a later CRY muon (fast: 289). The asymmetric convention loses the fast
289 passes: a 0.47 percentage-point shift; alternatively restricting G4 to its first muon removes 292 passes.
These two convention corrections must not be added together. The historical v16 rate is not reinterpreted as a
rerun with current transport: its change to current first-muon counts is a separate, statistically unresolved effect.
Ledger magnitudes (medians of per-file quantiles): writer air debit 0.30 MeV (max 4.2 MeV); thin loss **through the
last recorded straw entry** 0.29 MeV (q99 0.89 MeV), not total interior loss; gas deposit / mean unrestricted gas loss
0.57; integrated minus generated-beta step time median 3 ps, q99 1.9 ns. These do not establish energy closure.

Same-straw time against G4 (`v1compare` on the 20 `g4ref_v1` shards, 5,153 matched steps): median bias 0.8-0.9 ps
above 1 GeV/c in every arm; at G4 |p| 300-500 MeV/c (11 steps) -274 ps legacy, -126 ps T; 500-1000 MeV/c (79 steps)
-57 ps legacy, -16 ps T.

## Completed Linux campaign and recommendation (2026-10-27)

Reused the completed 2026-10-26 campaign, rather than regenerating baseline transport. Same tracker base `56e66bf`
plus this worktree's transport changes, core `4f57013`, v04 geometry and 220 input files; **11,000,000 generated
ROOT-table events in each arm**, zero unmapped G4 muons. Cross-arm input hashes and indices, all chunk settings,
retained-event/pass counts and G4 manifest counts pass the completion checks. An independent `check` against all
220 current generator-file contents, geometry hash and transport-source hashes passes. Linux source snapshots have
an empty git-head field because worktree pointers target laptop paths; the per-source SHA256s are authoritative.
This validates the campaign snapshot, not the subsequently merged GPU/digitiser/writer main.

| Comparison | Fast / reference passes | Relative change, paired error |
|---|---:|---:|
| Legacy, any primary muon vs G4 any primary muon | 61,315 / 61,864 | -0.8874% +- 0.3173% |
| Legacy, first vs first | 61,028 / 61,572 | -0.8835% +- 0.3181% |
| Legacy, historical asymmetric convention | 61,028 / 61,864 | -1.3514% +- 0.3178% |
| A vs legacy | 61,315 / 61,315 | 0.0000% +- 0.0065% |
| E vs legacy | 61,315 / 61,315 | 0.0000% +- 0.0023% (one gain and one loss) |
| M vs legacy | 61,314 / 61,315 | -0.0016% +- 0.0442% |
| AM vs legacy | 61,311 / 61,315 | -0.0065% +- 0.0441% |
| AMT vs legacy | 61,311 / 61,315 | -0.0065% +- 0.0441% |
| Old world-momentum core vs legacy | 61,247 / 61,315 | -0.1109% +- 0.3194% (not draw-paired) |

AM and AMT have identical passing identities: adding integrated time does not change the filter, which has no time
cut. T alone was measured on the laptop, not rerun on Linux. A/E are not identity-equal on Linux despite equal totals.
The independent laptop/Linux replicas must not be pooled or treated as new G4 statistics.

The corrected convention removes about **0.46 percentage points** in the Linux sample. The remaining deficit is
549 events (2.8 paired standard errors from zero): margin <1.25 contributes -178, margin 1.25-2 contributes -34,
margin 2-5 contributes -377, margin >5 contributes +40. These are first-primary truth-diagnostic bins. Of 19,541
G4-only events, the identity-matched fast muon misses all straws in 9,627 and has too few in-volume steps in 9,608;
216 stop above, 83 are outside the transport footprint, and 7 have low-momentum steps. The corresponding 18,992
fast-only events include 11,953 not saved by the G4 filter, so loss-side fates alone are not a net causal attribution.

**Timing replay:** `v1compare` now matches event + generated-particle identity + straw and excludes ambiguous
recrossings; it never matches by time/position proximity. Re-reading the existing 20-shard outputs reproduces all
5,153 legacy/T timing matches and the -274 -> -126 ps (300-500 MeV/c, 11 steps), -57 -> -16 ps (500-1000 MeV/c,
79 steps) medians. This is sparse same-straw evidence, not a surface-time or stopping validation. The added entry
position diagnostic fails its 10 um threshold (legacy medians 176, -356, 36 um), unchanged by T; no all-T-V1 pass
is claimed.

**Decision:** fix the counting/reporting tools; retain legacy production defaults. No ledger arm explains the
remaining rate difference. The corrected point estimate is inside 1%, but its uncertainty does not establish a
confidence bound within 1%; the physical residual and full any-particle/OR acceptance remain open. Do not promote
A/M/T or claim T3 closed. Merge/rebase onto updated main requires separate integration, particularly its JAX
backend; no JAX ledger equivalence was tested here.

### Reproduction and test evidence

Set `O` to the retained campaign output root (`trk_rate_ledger_20261026`), and use the saved core `4f57013` for the
Linux campaign. From this worktree, with the analysis Python environment and one library thread:

```bash
python3 -m trkfastsim.analyses.ratearms compare --g4 "$O/arms_v1/g4.parquet" \
  --arm legacy="$O/arms_v1/legacy" --arm A="$O/arms_v1/A" --arm E="$O/arms_v1/E" \
  --arm M="$O/arms_v1/M" --arm AM="$O/arms_v1/AM" --arm AMT="$O/arms_v1/AMT" \
  --arm W0="$O/arms_v1/W0" --base legacy --out "$O/reports_takeover/linux_validated.json"
```

The command refuses an existing report; choose a new report name for a new invocation. Retained Linux evidence:
`reports_takeover/{linux_validated,linux_conventions}.json`, `resume_check.log`, plus the original per-arm manifests
and first replay `reports_takeover/replay/`. Laptop evidence: `reports_takeover/time_identity_{legacy,T,AMT}/`,
`fatecheck_one/` (real one-shard smoke test), `tests_full.log`. The smoke test has 36,850 muon-containing events,
259 fast passes and 267 G4 passes; `generated` historically counts muon-containing events, while new
`generated_events` explicitly counts all generator ROOT-table events.

Full local suite: **169 passed, 1 skipped, 9 strict xfailed** (five legacy energy/time defects, one thin-clock
mismatch, three newly exposed T3 limits). New regression suites cover muon conventions, particle/straw timing
identity, incomplete/mixed/duplicate campaign rejection, terminal accounting and stopping. Runner syntax and 23
mock restart/failure/thread-limit assertions pass. No C++ build, new G4 production, commit or default change.
