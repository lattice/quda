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
--mg-levels 2 --inv-multigrid true --niter $NITER"
  THERM_MG="--hmc-mg-setup-interval $MG_SETUP_INTERVAL_THERM"
  PROD_MG="--hmc-mg-setup-interval $MG_SETUP_INTERVAL_PROD"
else
  SOLVER_FLAGS="--prec double --prec-sloppy double --mg-levels 1 --niter $NITER"
  THERM_MG=""; PROD_MG=""
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
  CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL DATA_DIR=$DATA_DIR \
  BUILD_TESTS=$BUILD_TESTS ANALYSIS_DIR=$ANALYSIS_DIR BETA=$BETA CSW=$CSW \
  INTEGRATOR=$INTEGRATOR TAU=$TAU ANALYSIS_BLOCK=$ANALYSIS_BLOCK \
    bash "$ANALYSIS_DIR"/kappa_ladder.sh
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
  timeout $(( (hi - lo + 1) * 150 + 1800 )) "$BUILD_TESTS"/correlator_distribution_test \
    --dim $DIMS --dslash-type $DSLASH --clover-csw $CSW --kappa $kappa $MEAS_FLAGS \
    --corrdist-config-prefix "$prefix" --corrdist-config-start $lo --corrdist-config-end $hi \
    --corrdist-out "$out"
  python3 "$ANALYSIS_DIR"/correlator_distribution_analysis.py "$out" --channel G5 \
    --block $ANALYSIS_BLOCK
  ;;
*)
  echo "unknown subcommand: $sub"; exit 1;;
esac
