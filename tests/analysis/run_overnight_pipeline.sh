#!/bin/bash
# Overnight correlator-distribution pipeline: extend a checkpointed
# quenched heatbath stream, measure per-configuration correlators on
# every saved configuration, then run the log-normality analysis.
#
# The stream is continued bit-exactly from the last saved configuration
# and RNG checkpoint, so repeated invocations with a growing CFG_END
# keep extending one Markov chain.  All parameters can be overridden
# from the environment, e.g.:
#
#   DATA_DIR=/data/lognormal CFG_END=100000 ./run_overnight_pipeline.sh
set -u

DATA_DIR=${DATA_DIR:-$HOME/lognormal_data}
BUILD_TESTS=${BUILD_TESTS:-$HOME/applications/build_quda/tests}
ANALYSIS_DIR=${ANALYSIS_DIR:-$(dirname "$0")}

# lattice and stream parameters
DIMS=${DIMS:-"8 8 8 32"}
BETA=${BETA:-5.9}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
CFG_START=${CFG_START:-2000}   # last configuration of the existing stream
CFG_END=${CFG_END:-100000}     # final configuration number to generate
PREFIX=$DATA_DIR/ensembleA/b59
RNG=$DATA_DIR/ensembleA/rng

# valence parameters
DSLASH=${DSLASH:-wilson}
KAPPA=${KAPPA:-0.15}
TOL=${TOL:-1e-10}

DATASET=$DATA_DIR/pion_dist_10k.dat

export QUDA_RESOURCE_PATH=$DATA_DIR/tunecache
mkdir -p "$QUDA_RESOURCE_PATH"

nsteps=$((CFG_END - CFG_START))

echo "=== PHASE 1: generation (cfg $CFG_START -> $CFG_END) start $(date) ==="
"$BUILD_TESTS"/heatbath_test --dim $DIMS --heatbath-beta $BETA \
  --heatbath-warmup-steps 0 --heatbath-num-steps $nsteps \
  --load-gauge ${PREFIX}_cfg_${CFG_START}.lime \
  --heatbath-rng-load "$RNG" \
  --heatbath-config-start $CFG_START \
  --heatbath-save-config-prefix "$PREFIX" \
  --heatbath-save-config-interval $SAVE_INTERVAL \
  --heatbath-rng-save "$RNG" \
  > "$DATA_DIR"/generation_10k.log 2>&1
gen_rc=$?
echo "=== PHASE 1 done rc=$gen_rc n_cfg=$(ls "$DATA_DIR"/ensembleA/*.lime | wc -l) $(date) ==="
if [ $gen_rc -ne 0 ]; then echo "GENERATION FAILED"; exit 1; fi

echo "=== PHASE 2: measurement (cfg $((CFG_START + SAVE_INTERVAL)) -> $CFG_END) start $(date) ==="
"$BUILD_TESTS"/correlator_distribution_test --dim $DIMS --dslash-type $DSLASH \
  --kappa $KAPPA --tol $TOL --niter 10000 \
  --corrdist-config-prefix "$PREFIX" \
  --corrdist-config-start $((CFG_START + SAVE_INTERVAL)) \
  --corrdist-config-end $CFG_END --corrdist-config-step $SAVE_INTERVAL \
  --corrdist-out "$DATASET" \
  > "$DATA_DIR"/measure_10k.log 2>&1
meas_rc=$?
echo "=== PHASE 2 done rc=$meas_rc n_measured=$(grep -c Measured "$DATA_DIR"/measure_10k.log) $(date) ==="
if [ $meas_rc -ne 0 ]; then echo "MEASUREMENT FAILED"; exit 1; fi

echo "=== PHASE 3: analysis $(date) ==="
python3 "$ANALYSIS_DIR"/correlator_distribution_analysis.py "$DATASET" \
  --channel G5 --block 5 --histogram 2,5,8,12 --out "$DATA_DIR"/pion_10k \
  > "$DATA_DIR"/analysis_10k.txt 2>&1
echo "=== PIPELINE COMPLETE $(date) ==="
