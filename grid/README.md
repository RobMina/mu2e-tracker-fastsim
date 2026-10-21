# Grid campaigns of the tracker stage

## CRY primaries without Geant4 (gen tables for fast runs)

The fast simulation reruns CRY primaries. The `g4ref_v1` gen stream gives the primaries of the 1 M-event G4 reference;
for a fast run of larger live time (for example against the MDC2025as production) the primaries come from this
campaign instead: the stock `ExtractedCRY` generator and cosmic time offset with no Geant4 transport.

| file | role |
|---|---|
| `gen_only_extracted.fcl` | stock ExtractedCRY with `PrimaryPath` = generator + cosmic time offset; keeps GenParticles, CosmicLivetime and GenEventCount of every generated event. The CRY generator reads the `WorldG4` geometry, which only the Geant4 module registers, so `g4run` stays in the schedule behind a filter that never passes: Geant4 initialises once (about 30 s) and transports no event |
| `grid_gen_only.sh` | worker: shard k = subrun FIRST_SUBRUN+k, baseSeed SEED_OFFSET+subrun; skips when its output exists |
| `submit_gen_only.sh [--test\|--cluster] [-N n] [--nevts N] [--first-subrun S] [--tag TAG]` | stages the fcl to `$TRKFS_PNFS/<tag>/fcl` and submits (no tarball: Musing `SimJob MDC2025ay` from CVMFS); lifetime 2 h, 3 GB memory, 2 GB disk |
| `dump_gen_only.sh TAG FIRST LAST` | on one GPVM, after the jobs: art -> `$M2FS_DATA/trk_fastsim/<tag>/gen.001400_<subrun>.root` with `art/`'s `dump_sgsteps_gen.fcl` (needs the TrkFastSim muse area); skips subruns whose table exists |

Measured on the grid (`gen_as_v1`, 2026-10-20): 500 k events in 272-1049 s of CPU per job depending on the node
(4.5-17 min; one job took 51 min of wall time on a slow node), peak RSS 1.9 GB, 73 MB of art per job. The dump takes
about 30 s per file read from pnfs and gives 28 MB of tables. Live time 7.79 s per 500 k events, as `g4ref_v1`
(0.0156 s per 1000 events).

Subruns from 1001 up keep clear of the `g4ref_v1` shards (1-20). A subrun's seed depends only on its number, so a
campaign extended later (`--first-subrun 1021 -N ...`) never repeats primaries.

### gen_as_v1 (2026-10-20)

| item | value |
|---|---|
| purpose | ~156 s of live time, the same as the 20 `prod_as20` production files, for `scripts/prodas_compare.sh` |
| test job | `30451359.0@jobsub04`: subrun 1001, exit 0, 8.4 min, 1.9 GB; its table identical to an interactive run of the same seed |
| cluster | `87182252@jobsub01`, `-N 20`: subruns 1001-1020, seeds 20261020-20261039; job 0 skipped (output from the test job); all 20 complete |
| output | art `$TRKFS_PNFS/gen_as_v1/` (20 x 73 MB, logs in `logs/`); tables `$M2FS_DATA/trk_fastsim/gen_as_v1/` (20 x 28 MB) |
| total | **10,000,000 generated events, 155.770 s live time** (prod_as20: 155.811 s, ratio 0.9997) |
