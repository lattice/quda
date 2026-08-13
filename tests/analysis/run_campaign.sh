#!/bin/bash
# Unified campaign dispatcher: every operation is a subcommand driven by
# a single configuration file (see campaign.conf.example).  No script
# editing per run — change the conf.
#
#   run_campaign.sh therm    <conf> [PREFIX=...]        chunked therm-tune
#   run_campaign.sh ladder   <conf>                     kappa ladder to targets
#   run_campaign.sh w0       <conf> <cfg> [cfg...]      scale setting
#   run_campaign.sh residual <conf> <cfg> <kappa>       convergence-class probe
#   run_campaign.sh sweep    <conf> <start_cfg> <out>   deflation-size sweep
#   run_campaign.sh measure  <conf> <prefix> <lo> <hi> <kappa> <out.dat>
set -u
sub=${1:?subcommand}; conf=${2:?conf file}
source "$conf"
shift 2

export QUDA_RESOURCE_PATH=$TUNECACHE
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps
mkdir -p "$TUNECACHE"

# solver flag groups assembled once, from config
if [ "${USE_MG:-0}" -eq 1 ]; then
  SOLVER_FLAGS="--prec double --prec-sloppy single --prec-precondition single --prec-null single \
--mg-levels ${MG_LEVELS:-2} --inv-multigrid true --niter $NITER ${MG_EXTRA_FLAGS:-}"
  THERM_MG=""; PROD_MG=""
  [ "${MG_SETUP_INTERVAL_THERM:-0}" -gt 0 ] && THERM_MG="--hmc-mg-setup-interval $MG_SETUP_INTERVAL_THERM"
  [ "${MG_SETUP_INTERVAL_PROD:-0}" -gt 0 ] && PROD_MG="--hmc-mg-setup-interval $MG_SETUP_INTERVAL_PROD"
else
  SOLVER_FLAGS="--prec double --prec-sloppy double --mg-levels 1 --niter $NITER"
  THERM_MG=""; PROD_MG=""
fi
# Thermalization always runs plain CG unless USE_MG_THERM=1: MG setup cannot
# hot-start on unthermalized gauge (coarse-op verification fails), and rung-1
# of a ladder at a new beta has no seed configuration.
if [ "${USE_MG_THERM:-0}" -eq 1 ]; then
  THERM_SOLVER_FLAGS="$SOLVER_FLAGS"
else
  THERM_SOLVER_FLAGS="--prec double --prec-sloppy double --mg-levels 1 --niter $NITER"
fi
export THERM_SOLVER_FLAGS
# MG+ET flags for thermalization chunks >= 2 (and seeded chunk 1); consumed
# by therm_tune.sh per Dean's chunk-switching scheme (2026-08-12)
if [ "${USE_MG:-0}" -eq 1 ]; then
  MG_THERM_FLAGS="$SOLVER_FLAGS $ET_FLAGS"
  export MG_THERM_FLAGS
fi
MEAS_FLAGS="--tol $MEAS_TOL --niter $MEAS_NITER --prec double --prec-sloppy double \
--prec-refine double --prec-precondition double"
ET_FLAGS=""
[ "${ET_ENABLED:-0}" -eq 1 ] && ET_FLAGS="--eigentracking true --eigentracking-n-ev $ET_NEV \
--eigentracking-fresh-interval $ET_FRESH_INTERVAL --eigentracking-refresh-residual $ET_REFRESH_RESIDUAL"
export SOLVER_FLAGS THERM_MG PROD_MG MEAS_FLAGS ET_FLAGS

case "$sub" in
therm)
  DIMS="$DIMS" BETA=$BETA KAPPA=${KAPPA:?set KAPPA=... in env} CSW=$CSW \
  INTEGRATOR=$INTEGRATOR N_STEPS=$N_STEPS TAU=$TAU CHUNK=$CHUNK \
  PLATEAU_CHUNKS=$PLATEAU_CHUNKS MAX_CHUNKS=$MAX_CHUNKS CHUNK_TIMEOUT=$CHUNK_TIMEOUT \
  ACC_LO=$ACC_LO ACC_HI=$ACC_HI BUILD_TESTS=$BUILD_TESTS \
  PREFIX=${PREFIX:?set PREFIX=... in env} \
    bash "$ANALYSIS_DIR"/therm_tune.sh
  ;;
ladder)
  LADDER_DIMS="$DIMS" TARGET_AMS="$TARGET_AMS" LADDER_START_KAPPA=$KAPPA_START \
  PROBE_CONFIGS=$PROBE_CONFIGS PROD_CONFIGS=$PROD_CONFIGS WINDOW_FRAC=$WINDOW_FRAC \
  CLAMP_MAX=$CLAMP_MAX CLAMP_FRAC=$CLAMP_FRAC MAX_RUNGS=$MAX_RUNGS \
  CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL DATA_DIR=$DATA_DIR LADDER_SEED_CFG=${LADDER_SEED_CFG:-} \
  BUILD_TESTS=$BUILD_TESTS ANALYSIS_DIR=$ANALYSIS_DIR BETA=$BETA CSW=$CSW \
  INTEGRATOR=$INTEGRATOR TAU=$TAU ANALYSIS_BLOCK=$ANALYSIS_BLOCK \
    bash "$ANALYSIS_DIR"/kappa_ladder.sh
  ;;
scout)
  # Scale-setting scout: chunked therm at KAPPA on DIMS, short production,
  # w0 measurement, and a suggested beta correction toward TARGET_A_FM.
  ED=$DATA_DIR/scout_b$(echo $BETA | tr -d .)
  mkdir -p "$ED"
  DIMS="$DIMS" BETA=$BETA KAPPA=${SCOUT_KAPPA:?set SCOUT_KAPPA in conf} CSW=$CSW \
  INTEGRATOR=$INTEGRATOR N_STEPS=$N_STEPS TAU=$TAU CHUNK=$CHUNK \
  PLATEAU_CHUNKS=$PLATEAU_CHUNKS MAX_CHUNKS=$MAX_CHUNKS CHUNK_TIMEOUT=$CHUNK_TIMEOUT \
  ACC_LO=$ACC_LO ACC_HI=$ACC_HI BUILD_TESTS=$BUILD_TESTS \
  PREFIX="$ED"/tt \
    bash "$ANALYSIS_DIR"/therm_tune.sh > "$ED"/thermtune.log 2>&1
  tuned=$(grep "TUNED:" "$ED"/thermtune.log | tail -1)
  n_steps=$(echo "$tuned" | sed -E 's/.*n_steps=([0-9]+).*/\1/')
  start_cfg=$(echo "$tuned" | sed -E 's/.*start_config=(\S+).*/\1/')
  [ -z "$start_cfg" ] && { echo "SCOUT HALT: therm failed (see $ED/thermtune.log)"; exit 1; }
  ntraj=$(( ${SCOUT_CONFIGS:-20} * 5 * 12 / 10 ))
  timeout $((ntraj * ${SCOUT_TRAJ_TIMEOUT:-300} + 3600)) "$BUILD_TESTS"/hmc_test --dim $DIMS $SOLVER_FLAGS \
    --dslash-type clover --clover-csw $CSW --kappa $SCOUT_KAPPA --hmc-beta $BETA \
    --hmc-integrator $INTEGRATOR --hmc-n-steps ${n_steps:-$N_STEPS} --hmc-tau $TAU \
    --hmc-thermalization 0 --hmc-n-trajectories $ntraj \
    --hmc-gauge-infile "$start_cfg" \
    --hmc-checkpoint 5 --hmc-checkpoint-prefix "$ED"/cfg_ \
    --gtest_filter=HMC.Production > "$ED"/production.log 2>&1
  cfgs=$(ls "$ED"/cfg_* 2>/dev/null | grep -vE "pool|evals" | tail -${SCOUT_CONFIGS:-20})
  [ -z "$cfgs" ] && { echo "SCOUT HALT: no configs produced"; exit 1; }
  bash "$0" w0 "$conf" $cfgs | tee "$ED"/w0.txt
  a_meas=$(grep -E "^a = " "$ED"/w0.txt | grep -oE "[0-9.]+" | head -1)
  if [ -n "$a_meas" ] && [ -n "${TARGET_A_FM:-}" ]; then
    python3 -c "
import math
a, tgt = $a_meas, $TARGET_A_FM
db = math.log(a/tgt)/1.29  # d(ln a)/d(beta) = -1.29: a too fine -> lower beta
print(f'SCOUT RESULT: a={a} fm (target {tgt}); suggested beta correction {db:+.3f} -> beta={$BETA+db:.3f}')"
  fi
  ;;
w0)
  out=$DATA_DIR/w0_$(date +%s); mkdir -p "$out"; i=0
  for cfg in "$@"; do
    i=$((i+1))
    timeout 1800 "$BUILD_TESTS"/su3_test --dim $DIMS --load-gauge "$cfg" \
      --su3-smear-type wilson --su3-smear-steps $FLOW_STEPS --su3-smear-epsilon $FLOW_EPS \
      --su3-measurement-interval $FLOW_MEAS_INTERVAL > "$out"/flow_$i.log 2>&1
  done
  W0_FM=$W0_FM A_LO=$A_SANITY_LO A_HI=$A_SANITY_HI FLOW_DIR=$out \
    python3 "$ANALYSIS_DIR"/w0_extract.py
  ;;
residual)
  CFG=${1:?cfg} KAPPA=${2:?kappa} DIMS="$DIMS" BUILD_TESTS=$BUILD_TESTS \
  OUT=$DATA_DIR/residual_probe_$(date +%s) \
    bash "$ANALYSIS_DIR"/residual_probe.sh
  ;;
sweep)
  START_CFG=${1:?start cfg} OUT=${2:?out dir} DIMS="$DIMS" BETA=$BETA KAPPA=${KAPPA:?} \
  CSW=$CSW N_STEPS=$N_STEPS TAU=$TAU BUILD_TESTS=$BUILD_TESTS \
    bash "$ANALYSIS_DIR"/deflation_sweep.sh
  ;;
measure)
  prefix=${1:?}; lo=${2:?}; hi=${3:?}; kappa=${4:?}; out=${5:?}
  timeout $(( (hi - lo + 1) * 150 * ${MEAS_SOURCES:-1} + 1800 )) "$BUILD_TESTS"/correlator_distribution_test \
    --dim $DIMS --dslash-type $DSLASH --clover-csw $CSW --kappa $kappa $MEAS_FLAGS \
    --corrdist-config-prefix "$prefix" --corrdist-config-start $lo --corrdist-config-end $hi \
    --corrdist-num-sources ${MEAS_SOURCES:-1} \
    --corrdist-out "$out"
  python3 "$ANALYSIS_DIR"/correlator_distribution_analysis.py "$out" --channel G5 \
    --block $ANALYSIS_BLOCK
  ;;
*)
  echo "unknown subcommand: $sub"; exit 1;;
esac
