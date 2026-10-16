# art helpers (muse satellite area)

`TrkFastSim/` is a muse package (a `.muse` file plus `src/SConscript`) built in its own work area, backed by the CVMFS
Musing `SimJob MDC2025ay`, so no Offline checkout is touched. Same layout as `crv/art/CrvFastSim` of the umbrella.

| module | role |
|---|---|
| `StrawGasStepTableDumper` | the tracker truth of a dts file -> flat TTrees in `dump/`: `sgsteps`, `surfsteps`, `vdsteps`, `trajpts`, `particles`, `gen`, `subruns`. uproot cannot read these art products directly (memberwise `SimParticle`/`GenParticle`, the `MCTrajectory` map) |

## Trees

All trees carry `run`, `subrun`, `event`. `sp` is the `SimParticle` key (`particles.key`).

| Tree | Product (default tag) | Columns | Frame |
|---|---|---|---|
| `sgsteps` | `StrawGasStep` (`compressDetStepMCs`) | `nsgs` (steps in the event), `sid` (`StrawId::asUint16`), `plane`, `panel`, `straw`, `upanel`, `shape` (line/arc/curl/point), `ion` (minion/highion/neutral), `sp`, `eion` (MeV), `len` (path in gas, mm), `width` (mm), `t` (ns, gas entry), `x0..z1` (entry/exit), `px..pz` (mean momentum, MeV/c) | Mu2e |
| `surfsteps` | `SurfaceStep` (`compressDetStepMCs`) | `surf` (SurfaceId enum, as EventNtuple `sid`), `sindex`, `sp`, `edep`, `t`, `x0..z1`, `px..pz` | detector |
| `vdsteps` | `StepPointMC` (`compressDetStepMCs:virtualdetector`) | `vdid` (VirtualDetectorId), `sp`, `x..z`, `px..pz`, `t` | Mu2e |
| `trajpts` | `MCTrajectory` (`compressDetStepMCs`) | `sp`, `ipt` (point index), `x..z`, `t`, `ke` (kinetic energy, MeV) | Mu2e |
| `particles` | `SimParticle` (`compressDetStepMCs`), `PrimaryParticle` (`FindMCPrimary`) | the `CrvStepTableDumper` columns (`key`, `parent`, `pdg`, `ccode`, `scode`, `isprim`, `genidx`, start/end position, 4-momentum, time) + `ismcprim` | Mu2e |
| `gen` | `GenParticle` (`cosmicTimeOffset`) | `idx`, `pdg`, `genid`, `x..z`, `px..pz`, `e`, `t` | Mu2e |
| `subruns` | `CosmicLivetime` (`generate`) | `livetime` (s), `primaries` | |

Missing optional products (`SurfaceStep`, VD steps, `MCTrajectory`, `PrimaryParticle`) leave their tree empty rather
than failing; `StrawGasStep`, `SimParticle` and `GenParticle` are required.

## Build and run

```bash
source site.env                                        # umbrella root: sets TRKFS_CODE=$M2FS_CODE/tracker
$TRKFS_CODE/art/setup_area.sh                          # backing link only
# user, on mu2ebuild02:
cd $TRKFS_CODE/art && source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && setup muse && muse setup && muse build -j 32
# run (GPVM):
mu2e -c TrkFastSim/fcl/dump_sgsteps.fcl -s dts....art -T sgsteps.root
$TRKFS_CODE/art/dump_ref.sh files.txt /exp/mu2e/data/users/$USER/trk_fastsim/tables
```
