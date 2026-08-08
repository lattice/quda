#!/bin/bash
# Deflation-space size study: sweep the eigentracker pool size and
# measure, per trajectory, the cost of maintaining the space (TRLM
# seeds/re-anchors, RR evolution, probe) against the benefit of using
# it (force-solve CG iterations, wall time).  All points start from the
# same thermalized configuration, so differences are attributable to
# the pool size alone.
#
# Env: START_CFG (required), DIMS BETA KAPPA CSW N_STEPS TAU N_TRAJ
#      NEV_LIST OUT BUILD_TESTS
set -u
BUILD_TESTS=${BUILD_TESTS:-$HOME/applications/build_quda/tests}
DIMS=${DIMS:-"24 24 24 48"}
BETA=${BETA:-5.3}
KAPPA=${KAPPA:-0.1340}
CSW=${CSW:-1.0}
N_STEPS=${N_STEPS:-10}
TAU=${TAU:-1.0}
N_TRAJ=${N_TRAJ:-6}
NEV_LIST=${NEV_LIST:-"4 8 16 32"}
START_CFG=${START_CFG:?set START_CFG to a thermalized checkpoint}
OUT=${OUT:?set OUT to an output directory}

mkdir -p "$OUT"
echo "# nEv  pool  wall_s/traj  mean_cg_iters  trlm_count  trlm_opx  mean_subspace_res" | tee "$OUT"/sweep.dat

run_point() {
  local nev=$1 log=$2 et_flags=$3
  local t0 t1
  t0=$(date +%s.%N)
  timeout ${POINT_TIMEOUT:-1800} "$BUILD_TESTS"/hmc_test --dim $DIMS --dslash-type clover --clover-csw "$CSW" --kappa "$KAPPA" \
    --prec double --prec-sloppy double --mg-levels 1 --hmc-beta "$BETA" \
    --hmc-integrator 2 --hmc-n-steps "$N_STEPS" --hmc-tau "$TAU" \
    --hmc-thermalization 0 --hmc-n-trajectories "$N_TRAJ" \
    --hmc-gauge-infile "$START_CFG" $et_flags \
    --gtest_filter=HMC.Production > "$log" 2>&1
  t1=$(date +%s.%N)
  python3 - "$log" "$t0" "$t1" "$N_TRAJ" <<'PYEOF'
import re, sys, statistics
log, t0, t1, ntraj = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
cg, trlm_opx, res = [], [], []
trlm_count = 0
for line in open(log):
    m = re.search(r"CG: Convergence at (\d+) iterations", line)
    if m: cg.append(int(m.group(1)))
    m = re.search(r"TRLM computed the requested \d+ vectors in \d+ restart steps and (\d+) OP\*x", line)
    if m: trlm_count += 1; trlm_opx.append(int(m.group(1)))
    m = re.search(r"subspace residual .* = ([0-9.e+-]+)", line)
    if m: res.append(float(m.group(1)))
wall = (t1 - t0) / max(1, ntraj)
print(f"{wall:.2f} {statistics.mean(cg) if cg else -1:.1f} {trlm_count} {sum(trlm_opx)} "
      f"{statistics.mean(res) if res else -1:.3f}")
PYEOF
}

# Baseline: eigentracking off
read -r wall cgi tc topx sres <<< "$(run_point 0 "$OUT"/nev0.log "")"
echo "   0     0  $wall  $cgi  $tc  $topx  $sres" | tee -a "$OUT"/sweep.dat

for nev in $NEV_LIST; do
  pool=$((2 * nev))
  flags="--eigentracking true --eigentracking-n-ev $nev --eigentracking-pool-capacity $pool"
  read -r wall cgi tc topx sres <<< "$(run_point $nev "$OUT"/nev$nev.log "$flags")"
  echo "  $nev    $pool  $wall  $cgi  $tc  $topx  $sres" | tee -a "$OUT"/sweep.dat
done
echo "SWEEP COMPLETE $(date)"
