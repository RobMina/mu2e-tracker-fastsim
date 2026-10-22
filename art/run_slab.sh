#!/bin/bash
# Slab test, G4 arm: mu- at normal incidence through one NIST box (default G4_CONCRETE, 2.3 g/cm3) in vacuum, ShieldingM, one job
# per (thickness, |p|) point, serially on one GPVM; dumps the muon's SimParticle end state (StrawGasStepTableDumper).
# usage: run_slab.sh <outdir> [nevents per point, default 100000] ["g/cm2 list"] ["p list MeV/c"] [NIST material, default G4_CONCRETE]
# The g4study geometry builds NIST materials only (ConstructMaterials skips Mu2e materials), so MAT must be a G4NistManager name
# with its density listed below; thicknesses may be fractional (tag t<g/cm2>_p<MeV>).
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1 || true
cd "${TRKFS_CODE:?source site.env}/art" && muse setup >/dev/null 2>&1 || true
set -u
[ "$MUSE_WORK_DIR" = "$TRKFS_CODE/art" ] || { echo "wrong MUSE_WORK_DIR=$MUSE_WORK_DIR"; exit 2; }
O=$(readlink -f "${1:?outdir}"); N=${2:-100000}; TS=${3:-"50 150 300"}; PS=${4:-"500 750 1000 2000 5000"}; MAT=${5:-G4_CONCRETE}
case $MAT in                                                                     # G4NistMaterialBuilder densities [g/cm3]
  G4_CONCRETE) RHO=2.30 ;; G4_Al) RHO=2.699 ;; G4_Cu) RHO=8.96 ;; G4_Pb) RHO=11.35 ;; G4_Sn) RHO=7.31 ;; G4_BRONZE) RHO=8.82 ;;
  *) echo "unknown material $MAT (add its NIST density)"; exit 2 ;;
esac
F=$TRKFS_CODE/art/TrkFastSim/fcl
mkdir -p "$O/cfg" "$O/logs"
for T in $TS; do
  HZ=$(awk -v t="$T" -v r="$RHO" 'BEGIN{printf "%.4f", 0.5 * t / r * 10.0}')     # half thickness [mm]
  for P in $PS; do
    TAG="t${T}_p${P}"; OUT="$O/slab.$TAG.root"
    [ -s "$OUT" ] && { echo "$TAG exists"; continue; }
    sed -e "s|^vector<double> box.halfLengths = .*|vector<double> box.halfLengths = {20000., 20000., $HZ};|" \
        -e "s|^string         box.wallMaterialName = .*|string         box.wallMaterialName = \"$MAT\";|" \
        "$F/slab_geom.txt" > "$O/cfg/geom.$TAG.txt"
    sed -e "s|^vector<double> particleGun.point = .*|vector<double> particleGun.point = {0., 0., -$(awk -v h="$HZ" 'BEGIN{printf "%.4f", h + 0.001}')};|" \
        -e "s|^double particleGun.pmin = .*|double particleGun.pmin = $P.;|" -e "s|^double particleGun.pmax = .*|double particleGun.pmax = $P.;|" \
        "$F/slab_gun.txt" > "$O/cfg/gun.$TAG.txt"
    # seed as before for integer thicknesses of the default material; other materials get their own offset
    SEED=$(awk -v t="$T" -v p="$P" -v m="$MAT" 'BEGIN{o = (m == "G4_CONCRETE") ? 0 : 7919 * length(m) + 104729 * index("AlCuPbSnBR", substr(m, 4, 2)); printf "%d", 20261022 + int(t * 10 + 0.5) + int(p / 250) + o}')
    printf '#include "TrkFastSim/fcl/slab_g4.fcl"\nservices.GeometryService.inputFile : "cfg/geom.%s.txt"\nphysics.producers.generate.inputfile : "cfg/gun.%s.txt"\nservices.TFileService.fileName : "%s"\nservices.SeedService.baseSeed : %d\n' \
      "$TAG" "$TAG" "$OUT" "$SEED" > "$O/cfg/slab.$TAG.fcl"
    T0=$(date +%s)
    # SimpleConfig files are looked up on MU2E_SEARCH_PATH (relative names only)
    MU2E_SEARCH_PATH="$O:$MU2E_SEARCH_PATH" mu2e -c "$O/cfg/slab.$TAG.fcl" -n "$N" > "$O/logs/$TAG.log" 2>&1
    echo "$TAG rc=$? $(( $(date +%s) - T0 )) s"
  done
done
echo "done $(date -Is)"
