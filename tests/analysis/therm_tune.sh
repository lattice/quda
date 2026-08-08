#!/bin/bash
# Chunked thermalization with integrator auto-tuning.
#
# Runs HMC thermalization 10 trajectories at a time, monitoring between
# chunks: var(dH), <dH> vs var/2, Creutz <exp(-dH)>, the predicted
# Metropolis acceptance erfc(sqrt(var/8)), and the per-trajectory solver
# iteration count (the IR-sensitive thermalization criterion).  The
# integration step count is retuned between chunks toward a target
# acceptance window, using var(dH) ~ n_steps^-4 for a 2nd-order
# integrator.  Thermalization is declared complete when the solver
# iteration count has plateaued across consecutive chunks, so no
# resources are spent equilibrating beyond need.
#
# Environment overrides: DIMS BETA KAPPA CSW INTEGRATOR N_STEPS TAU
# CHUNK MAX_CHUNKS PLATEAU_CHUNKS ACC_LO ACC_HI PREFIX BUILD_TESTS
set -u

BUILD_TESTS=${BUILD_TESTS:-$HOME/applications/build_quda/tests}
DIMS=${DIMS:-"16 16 16 32"}
BETA=${BETA:-5.3}
KAPPA=${KAPPA:-0.1330}
CSW=${CSW:-1.0}
INTEGRATOR=${INTEGRATOR:-1}
N_STEPS=${N_STEPS:-60}
TAU=${TAU:-1.0}
CHUNK=${CHUNK:-10}
MAX_CHUNKS=${MAX_CHUNKS:-30}
PLATEAU_CHUNKS=${PLATEAU_CHUNKS:-3}
ACC_LO=${ACC_LO:-0.80}
ACC_HI=${ACC_HI:-0.92}
ET_ARGS=${ET_ARGS:-}
ET_ARGS_FILE=${ET_ARGS_FILE:-}
PREFIX=${PREFIX:?set PREFIX to the ensemble directory/prefix, e.g. ~/data/ensF/thermtune}

mkdir -p "$(dirname "$PREFIX")"
STATE_GAUGE=""
plateau=0
prev_iters=-1

for chunk in $(seq 1 "$MAX_CHUNKS"); do
  log="${PREFIX}_chunk${chunk}.log"
  infile_args=()
  [ -n "$STATE_GAUGE" ] && infile_args=(--hmc-gauge-infile "$STATE_GAUGE")
  # eigentracking per chunk: ET_ARGS fixed, ET_ARGS_FILE re-read each chunk
  # for live adjustment; a pool saved at the previous chunk checkpoint is
  # chained in so the deflation space evolves across chunk boundaries.
  et_args=($ET_ARGS)
  [ -n "$ET_ARGS_FILE" ] && [ -f "$ET_ARGS_FILE" ] && et_args+=($(cat "$ET_ARGS_FILE"))
  if [ ${#et_args[@]} -gt 0 ] && [ -n "$STATE_GAUGE" ] && [ -f "${STATE_GAUGE}.pool.evals" ]; then
    et_args+=(--eigentracking-pool-infile "${STATE_GAUGE}.pool")
  fi

  timeout ${CHUNK_TIMEOUT:-2400} "$BUILD_TESTS"/hmc_test --dim $DIMS --dslash-type clover --clover-csw "$CSW" --kappa "$KAPPA" \
    --prec double --prec-sloppy double --mg-levels 1 --hmc-beta "$BETA" \
    --hmc-integrator "$INTEGRATOR" --hmc-n-steps "$N_STEPS" --hmc-tau "$TAU" \
    --hmc-thermalization "$CHUNK" --hmc-n-trajectories "$CHUNK" \
    --hmc-checkpoint "$CHUNK" --hmc-checkpoint-prefix "${PREFIX}_c${chunk}_" \
    "${infile_args[@]}" ${et_args[@]+"${et_args[@]}"} \
    --gtest_filter=HMC.Production > "$log" 2>&1 || { echo "chunk $chunk FAILED, see $log"; exit 1; }

  STATE_GAUGE=$(ls -t "${PREFIX}_c${chunk}_"* 2>/dev/null | grep -vE "\.pool|\.evals" | head -1)
  [ -z "$STATE_GAUGE" ] && { echo "chunk $chunk produced no checkpoint"; exit 1; }

  # ---- monitor -------------------------------------------------------
  read -r var mean creutz acc iters <<< "$(python3 - "$log" <<'PYEOF'
import math, re, statistics, sys
dh, it, cur = [], [], []
for line in open(sys.argv[1]):
    m = re.search(r"Convergence at (\d+) iterations", line)
    if m: cur.append(int(m.group(1)))
    m = re.search(r"hmcTrajectoryQuda: H_final.*dH = ([+-e0-9.]+)", line)
    if m:
        dh.append(float(m.group(1)))
        if cur: it.append(max(cur)); cur = []
v = statistics.variance(dh) if len(dh) > 1 else 0.0
print(f"{v:.6g} {statistics.mean(dh):+.6g} "
      f"{statistics.mean(math.exp(-x) for x in dh):.4f} "
      f"{math.erfc(math.sqrt(v/8)):.4f} "
      f"{max(it) if it else -1}")
PYEOF
)"
  echo "chunk $chunk: n_steps=$N_STEPS var(dH)=$var <dH>=$mean <e^-dH>=$creutz pred_acc=$acc max_cg_iters=$iters"

  # ---- thermalization criterion: solver-iteration plateau ------------
  if [ "$iters" -eq "$prev_iters" ]; then plateau=$((plateau + 1)); else plateau=0; fi
  prev_iters=$iters
  if [ "$plateau" -ge "$PLATEAU_CHUNKS" ]; then
    echo "THERMALIZED after $((chunk * CHUNK)) trajectories (iteration count stable at $iters for $PLATEAU_CHUNKS chunks)"
    echo "TUNED: n_steps=$N_STEPS  start_config=$STATE_GAUGE"
    exit 0
  fi

  # ---- integrator retune: var(dH) ~ n_steps^-4 -----------------------
  in_band=$(python3 -c "print(1 if $ACC_LO <= $acc <= $ACC_HI else 0)")
  if [ "$in_band" -eq 0 ] && [ "$(python3 -c "print(1 if $var > 0 else 0)")" -eq 1 ]; then
    # target the middle of the acceptance band
    N_STEPS=$(python3 -c "
import math
acc_t = 0.5*($ACC_LO + $ACC_HI)
var_t = 8 * (math.erfc if False else (lambda p: p))(0)  # placeholder
# invert erfc(sqrt(v/8)) = acc_t numerically
lo, hi = 1e-8, 50.0
for _ in range(80):
    mid = 0.5*(lo+hi)
    if math.erfc(math.sqrt(mid/8)) > acc_t: lo = mid
    else: hi = mid
var_t = 0.5*(lo+hi)
expo = 0.125 if $INTEGRATOR >= 2 else 0.25  # 4th-order FGI: var ~ n^-8; 2nd-order: n^-4
n = $N_STEPS * ($var/var_t)**expo
print(max(4, int(round(n))))
")
    echo "  retuned n_steps -> $N_STEPS (targeting acceptance $(python3 -c "print(0.5*($ACC_LO+$ACC_HI))"))"
  fi
done

echo "MAX_CHUNKS reached without iteration-count plateau; inspect ${PREFIX}_chunk*.log"
exit 1
