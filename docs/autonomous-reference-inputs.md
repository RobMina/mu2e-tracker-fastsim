# Reference inputs for autonomous F1

## Required population contract

The MC/KPP comparison starts from **unfiltered detector digis** with full run/subrun/event identity. The reference
job must write one dumper row per input event, even if it has no hits/seeds/accepted KKLine fit. Save failed fits
with `SaveAllFits`; acceptance is evaluated separately from input selection. Never make the detector input from
`SelectReco` digis: those are skimmed by reconstruction. Never repair a mismatched population by intersecting keys.

`StrawHitTableDumper` exports the digis, condition snapshot, seed and stock fit products together. Autonomous
reconstruction reads only the digi branches and conditions; the separate writer/evaluator may read stock products.
The same file therefore binds input event coverage to the comparator without conditioning reconstruction.

## MC digis from fast steps

Existing digitiser-validation dumps need not contain `kl_*`. If absent, use the **same saved stock digi art file**
from that digitisation job with `TrkFastSim/fcl/dump_tracks_mc_saveall.fcl`, on a verified compatible user-built area.
Do not substitute another similarly named transport arm's EventNtuple. Check full event identities against the fast
digi table before fitting/comparing; identical subrun names or similar counts are insufficient.

This is a stock baseline for *independently digitised versions of the same step population*, so it supports
identity-paired counts but **not `--same-digis` shared-hit residual matching**. To isolate reconstruction on identical
fast digis in stock, a validated StrawDigi/ADC producer from the fast digi table is needed; there is no such input
adapter in this prototype. That is a separate C++ build/user-approval task, not achieved by the step-table producer.

Example, after the build and runtime area are verified, with the input paths explicitly supplied:

```bash
cd USER_BUILT_WORKDIR
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
muse setup
mu2e -c TrkFastSim/fcl/dump_tracks_mc_saveall.fcl -s EXACT_STOCK_DIGI.art \
  -n 100 -T DATA_AREA/mc_smoke.root --debug-config DATA_AREA/mc_smoke.expanded.fcl
mu2e -c TrkFastSim/fcl/dump_tracks_mc_saveall.fcl -s EXACT_STOCK_DIGI.art \
  -n 100 -T DATA_AREA/mc_smoke.root
```

`--debug-config` expands the configuration and exits without processing events. Inspect that file, then run the
second command without the flag. An exit-zero configuration expansion is not a successful reconstruction job.

Run in a fresh output directory, inspect the actual highest-cycle census, then size a full sample from measured
resource usage; no runtime extrapolation is implied by this example. A new stock reconstruction uses an updated,
verified build, not a historical native-capture area as if it were current head.

## User build gate

The satellite is the existing `art/TrkFastSim` package, built in a clean detached worktree of tracker main
(`trkfs_wt/dumper_build`); no source is copied into a shared main checkout. The backing is pinned to the validated CI
build by a symlink, not `muse backing HEAD`, which resolves to whatever CI build is newest. On **mu2ebuild02**, fresh
shell:

```bash
cd /path/to/dumper_build/art
ln -sfn /cvmfs/mu2e-development.opensciencegrid.org/museCIBuild/main/bac4c588_4479f3fa_a664b74b backing
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
muse setup
muse build -j 32
```

These commands are run by the user (`backing` is git- and sync-ignored, so the link is made
on the cluster). The validated satellite uses Offline `bac4c588`, Production `4479f3fa` and envset `p109`; a rebuild of
tracker `ff4ba22` in this area reproduced the earlier 500-event KPP dump bit for bit on its first 100 events. CI builds
are eventually removed from CVMFS: if that one is gone, a newer backing is a new configuration and needs its own
smoke comparison. The first run on a GPVM whose CVMFS cache lacks the CI build can spend minutes loading libraries.
For each new area, record the resolved backing revision/envset, expanded
FCL and successful full build before a smoke run; verify module timestamps/linkage. The backing must contain the intended
updated Offline/Production configuration. Do not reuse a historical p107 or instrumented p109 capture area as if it
were this current-HEAD test. Any incompatibility in the helper package stops at the build gate.

## KPP run 124155 (trk.vst stream)

`TrkFastSim/fcl/dump_tracks_data_autonomous.fcl` is an unfiltered variant of the existing data reco configuration,
runtime-tested on 500 preserved KPP events with Offline `bac4c588` / Production `4479f3fa`, envset `p109`.
It inherits the uncalibrated offspill `reco_data_vst.fcl`, explicitly pins extracted
v04 geometry, drops CRV/calo work and KLFilter/SelectReco from its path, runs hit/seed/KKLine reconstruction, and dumps
every event with `DumpTracks` and `SaveAllFits`. It keeps the base data conditions; it is not the calibrated daqana3
configuration, so a daqana3 ntuple is not silently used as its stock comparator.

Supply `crv_efficiency/fcl` and this worktree's `art` on `FHICL_FILE_PATH` after setup. The input must be a preserved
unfiltered digi file unpacked with the **complete 216-row TrkPanelMap** and waveform retention. If unavailable,
unpack a known raw file into a per-job scratch area first using the head-compatible full-map configuration. The map
is required at unpack time; enabling it during reconstruction cannot repair collapsed StrawIds. The online raw
stream is named `trk.vst`; the detector is KPP, not the Wideband CRV VST.

```bash
export FHICL_FILE_PATH="TRACKER_WORKTREE/art:CRV_EFFICIENCY/fcl:$FHICL_FILE_PATH"
mu2e -c TrkFastSim/fcl/dump_tracks_data_autonomous.fcl -s FULLMAP_UNFILTERED_DIGI.art \
  -n 100 -T DATA_AREA/kpp_smoke.root --debug-config DATA_AREA/kpp_smoke.expanded.fcl
mu2e -c TrkFastSim/fcl/dump_tracks_data_autonomous.fcl -s FULLMAP_UNFILTERED_DIGI.art \
  -n 100 -T DATA_AREA/kpp_smoke.root
bash scripts/autonomous_reference.sh DATA_AREA/kpp_smoke.root DATA_AREA/NEW_REFERENCE data
```

The reference wrapper requires a new directory and writes `input_inventory.json`, matching `conditions.json` and
all-event good-only `stock.root`. `autonomous_inventory.py` records readout planes/panels, ADC coverage and
zero/multiple seed/fit populations. `autonomous_frontend.sh DUMP CONDITIONS data NEW_DIR` runs existing hit and
seed diagnostics separately; neither diagnostic supplies stock values to autonomous reconstruction.
`autonomous_compare.sh` forwards arguments to the strict saved-product evaluator.
No script in this change builds software.
Do not generate new outputs in an app tree or overwrite a preserved baseline.

Before a full KPP comparison:

1. Count the actual input/dumped unique event rows and digi readout planes, including empty events.
2. Inspect expanded geometry, offspill windows, PBT tags, no calo dependency and **absence of selection filters**.
3. Extract hit/drift calibration from this same dump/IOV, not the MC JSON. Nominal data geometry must not receive an
   unrelated MC alignment table; explicitly label wire-axis approximation until matching straw axes are available.
4. Run autonomous with `--settings data`; separately convert the dump's stock fits with `reco-ntuple --fit stock`.
5. Require identical full event keys in `autonomous_compare --same-digis`; compare all event multiplicities first,
   then conditional shared-hit residuals. Do not claim truth residuals for real data.
6. Any field/configuration or dumper source change requires the user build protocol before art execution.
