# PrimaryFilter tracker-term rate and the transport ledger

The stage C transport keeps the muon's energy, its straw deposits and its step times in several places. Its
conventions are:

- the step writer subtracts `dedx_restricted(AIR) x path` from the **momentum** of every straw entry, although the
  tracker envelopes are `DSVacuum` in the GDML and the interior transport has no gap material;
- the support loss (sampled, `slab_loss`) is charged to the kinetic energy;
- the straw walls, gas and sense wire scatter the muon but take **no energy**;
- the step time uses the **generated** speed for the whole path, `t = t_gen + path / (beta(p_gen) c)`;
- the world column above the plane (core `transport.world_loss`) converts the remaining kinetic energy to momentum
  before each loss call (core `2bcefde` and later; earlier cores passed `p - loss`).

`trkfastsim.sim.ledger` records these conventions (`ledger.Arms`) and holds the flight-time helpers of the
per-segment ledger below. Alternative conventions (writer debit from the kinetic energy or none, thin-material loss,
integrated step time: the arms `E`, `A`, `M`, `T`, `AM`, `AMT`) were measured on 11 M events in October 2026 and none
changed the rate (tables below). They, the entry-ledger and transport-capture diagnostics, the physical-segment
contract (`transportledger`, `flighttime`) and the open T3 limit tests are on branch `park/rate-ledger-20261103`.
**T3 remains open:** the transport has no unified energy/deposition/time ledger.

`ledger.core_world_convention()` records which world convention the imported core implements and the hash of its
`world_loss` source in every `ratearms` output.

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
from the band means; vacuum world time; the history draws no random numbers; step and path-vertex times equal at the
straw entries at constant momentum; G4-to-fast muon identity. The strict xfails of `tests/test_step_energy_ledger.py`
document the legacy defects of the default transport.

## Rate measurement

`python3 -m trkfastsim.sim.ratearms` (module docstring): `g4` (per primary muon of the G4 reference), `fast`
(the default transport, optionally with `--scatter-in`, on gen tables; file i of the 220 g4ref_v1 + g4ref_v2 gen
tables uses the generator seed `(1, i)` in every arm), `compare` (paired tables of arms given as NAME=DIR), and `check`
(validated restart against exact input hashes and physics settings). Partial files are rejected.
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

Arms compared with each other must run on one platform: the same code and seeds on Linux and macOS differ in about 1.3 % of the
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

The arm runs above, their campaign drivers (`rate_ledger_arms.sh`, `rate_ledger_campaign.sh`) and the T3 limit tests
are on branch `park/rate-ledger-20261103`.

## Support scattering onto the straws (`--scatter-in`, 2026-11-01)

The residual above has one cause: the default interior transport kinks a muon in the support only in front of a straw
it is about to cross, and drops a line that crosses no further straw. Support scattering therefore moves muons off
the straws but never onto them. Muons whose direction is nearly perpendicular to the tracker axis are the most
exposed: a plane's straws fill 10 mm bands in z, so whether such a muon reaches a straw depends on its tilt.
Geant4 keeps an event that fails the tracker term only when it passes the CRV term, so the reach test uses the muons
whose generated line passes at least 300 mm inside the CRV footprint (306,349 muons in the 220 files); Geant4 keeps
99.7 % of those events. Same 220 files, three generator seeds, paired errors:

| Quantity | Default | `--scatter-in` |
|---|---:|---:|
| tracker-term rate, seed 1 / 2 / 3 | -0.91 / -0.55 / -1.00 % | +0.42 / -0.05 / +0.04 % |
| rate, mean of the seeds | -0.82 +- 0.26 % | +0.14 +- 0.26 % |
| muons reaching >= 1 straw (subset), mean | -2.5 +- 0.4 % | -0.1 +- 0.4 % |
| same, \|u_z\| < 0.05 (1,797 Geant4 muons) | -8.6 % | +1.3 % |
| muons below 750 MeV/c at the first straw, mean | -6.4 +- 1.1 % | -2.3 +- 1.1 % |

Each per-seed rate has a paired error of 0.32 %; the mean's error counts the Geant4 fluctuation once. The default is
unchanged (muon tables and interior counters identical on 10 files against the earlier campaign). The interior-width
arms (support width x0.85, entry/exit mean momentum, finer thin and support kinks, no single-scattering tail, slab
calibration) each moved the rate by at most +0.4 % and left the reach deficit as it was. The gain from
`--scatter-in` differs between seeds more than its paired error (+1.34, +0.50, +1.05 %, each +- 0.14 %); not
understood.

```bash
python3 -m trkfastsim.sim.ratearms fast GEN... --gdml GDML --seed 1 --index0 0 --g4 G4.parquet \
    --outdir DIR --scatter-in
python3 -m trkfastsim.sim.ratearms compare --g4 G4.parquet --arm legacy=BASE --arm scatter_in=DIR --base legacy \
    --out REPORT.json
```
