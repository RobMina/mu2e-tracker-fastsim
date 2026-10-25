# Thread F p109 diagnostic build area

Prepared October 25, 2026. Source-only; **no C++ build has been started**.

This is one coherent full-source area, with no p107 or frozen Offline backing. The sources are git worktrees
sharing existing object stores; `TrkFastSim` is a real directory copied on the cluster from tracker `79e4bec`.
Do not remove/prune the owning repositories or these worktrees while this area is in use.

| Repository | Revision |
|---|---|
| Offline | `9b3d463371f34731f8515d1a63914ccf070536a8` (canonical main when prepared; ENVSET p109) |
| KinKal | `bb23c26348fc6e5fcbeefb5c7138919ef21f7abc` (v3.9.0 + passive rejection snapshot) |
| Production | `4479f3fad37a85ecd9485394198658237961dcc3` |
| mu2e-trig-config | `d3ec315c7e8a626e8cf9c417ab6522a8e9a8f669` |
| EventNtuple | `76ecb8af80c60e98d80a62a29f0082f36c20add5` |
| TrkFastSim source | tracker `79e4becd1a67bcc9f26725cab2019648b570795a` |

KinKal fork branch `thread-f-rejection-trace-p109` is pushed. Base `v3.9.0` is `8fa3f62`; its `Fit/Track.hh`
SHA256 `3868a3af55b10e976066e4173ceaeb4d91a5fb7e01893a5c03e1d6a5f41a6d49` exactly matches p109's installed
header. The snapshot change cherry-picked cleanly (18 added lines). p109 has different fitting code from p107,
including the upstream first-piece-prior correction: historical p107 failure/yield populations are not a p109
baseline. Capture off/on will use this one build and identical p109 inputs/settings.

Read-only `muse setup -q prof:p109` succeeded on gpvm07 and reported `al9-prof-e29-p109`, KinKal `v03_09_00`.
Muse v4_14_01 emitted its known hyphenated MU2E-TRIG-CONFIG export warning, and could not install a pre-commit
hook under the Offline worktree's `.git` file; setup still returned 0 and selected all five local muse repos.
KinKal has no muse package: its headers are found first via the workDir include path; the separate local KinKal
build/install below is necessary before any runtime use. No source API compatibility or compilation is certified.

## Quota gate

The app directory has a **25 GiB per-user quota**, regardless of `df` reporting hundreds of GB free globally.
Observed after source preparation: 25,031,833,511 bytes used (23.31 GiB), only 1.69 GiB free. The source area
itself was about 36.9 MB at initial accounting. Existing full build trees occupy about 3.92-3.94 GiB; their
Offline `tmp/` objects alone occupy about 3.06-3.08 GiB.

Require **at least 6 GiB free before starting** as a conservative operational reserve, not a measured upper bound
on this new build. Do not launch another app-area build concurrently. Build one prof/p109 stub only, not a
separate control tree or debug build. No tarball is needed. Outputs and long logs belong in the data area.

No files have been deleted. Reclaim space only from explicitly approved, idle older areas. Removing only a chosen
`build/<stub>/Offline/tmp` preserves its libraries and sources but makes its next build recompile those objects.
Two approved old tmp directories should free roughly 6.1 GiB; the user must choose them after checking that no
build or job needs their reconstruction. Do not touch another thread's area by default. Recheck Ceph accounting
after cleanup (it can lag), and during the build from a second shell.

## User build on mu2ebuild02

Use a fresh shell. Do not use the existing p107 trace area. These commands are for the user, not the agent:

```bash
cd /exp/mu2e/app/users/ram2aq/analysis/fastreco_p109
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
setup muse
muse setup -q prof:p109
```

Then, in that shell (the guard exits only its subshell):

```bash
(
  set -e
  APP=/exp/mu2e/app/users/ram2aq
  AREA=$APP/analysis/fastreco_p109
  test "${HOSTNAME%%.*}" = mu2ebuild02
  test "$MUSE_WORK_DIR" = "$AREA"
  test "$MUSE_ENVSET" = p109
  test "$MUSE_STUB" = al9-prof-e29-p109
  test ! -e "$AREA/backing"
  used=$(getfattr --only-values -n ceph.dir.rbytes "$APP")
  limit=$(getfattr --only-values -n ceph.quota.max_bytes "$APP")
  echo "App quota: $used / $limit bytes; free $((limit-used))"
  if (( limit-used < 6*1024*1024*1024 )); then
    echo 'STOP: free at least 6 GiB in the app quota before building.'
    exit 1
  fi
  muse build -j 32
  KINKAL_BLD="$AREA/kinkal_build_p109" \
  KINKAL_DST="$AREA/build/al9-prof-e29-p109/Offline/lib" \
    bash "$APP/analysis/crv_track_extrapolation/pr_stability/build_kinkal_into_area.sh" "$AREA"
)
```

The existing local-KinKal helper does its own CMake/make build and installs the matching soname libraries. Its
build pipelines use `tail` without `pipefail`, so **do not treat its exit code alone as validation**. Require the
full muse success line and `INSTALL_KINKAL_DONE`, then have the agent verify libraries, timestamps, NEEDED/RPATH
and local KinKal resolution before any art run. The helper only cleans its explicitly named KinKal build directory;
no old area is selected automatically. Stop on build errors rather than running jobs with a partial install.

Do not delete the new build's tmp objects until compilation and linkage have been verified and the first
capture-off/on tests pass; they are valuable for the first incremental corrections. Any later cleanup remains
user-approved. No build or runtime testing was performed during area preparation.
