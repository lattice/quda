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
STATE_GAUGE="${START_GAUGE:-}"   # seed chunk 1 from a nearby-kappa thermalized config
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

  # Solver selection per chunk: chunk 1 from a COLD start runs plain CG
  # (ordered gauge, no setup churn while the UV disorders); every later
  # chunk — and chunk 1 of a SEEDED start — runs the full MG+eigentracking
  # stack (MG_THERM_FLAGS), whose re-setup/re-anchor machinery tracks the
  # fast early-therm drift. The therm period doubles as the MG-parameter
  # timing bed (secs/traj per chunk logged below).
  CHUNK_FLAGS="${SOLVER_FLAGS:?}"
  if [ -n "${MG_THERM_FLAGS:-}" ]; then
    if [ "$chunk" -gt 1 ] || [ -n "${START_GAUGE:-}" ]; then CHUNK_FLAGS="$MG_THERM_FLAGS"; fi
  fi
  t_chunk0=$(date +%s)
  timeout ${CHUNK_TIMEOUT:-2400} "$BUILD_TESTS"/hmc_test --dim $DIMS $CHUNK_FLAGS ${THERM_MG:-} --dslash-type clover --clover-csw "$CSW" --kappa "$KAPPA" --hmc-beta "$BETA" \
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
  t_chunk1=$(date +%s)
  spt=$(( (t_chunk1 - t_chunk0) / CHUNK ))
  echo "chunk $chunk: n_steps=$N_STEPS var(dH)=$var <dH>=$mean <e^-dH>=$creutz pred_acc=$acc max_cg_iters=$iters secs_per_traj=$spt" | tee -a "$PREFIX"_tune.log

  # ---- thermalization criterion: solver-iteration plateau ------------
  # Tolerance-based: at light masses lambda_min fluctuations jitter the
  # iteration count by a few per chunk; require stability within
  # max(2, 3%) rather than exact equality.
  tol=$(python3 -c "print(max(2, int(0.03 * $iters)))")
  if [ "$prev_iters" -ge 0 ] && [ $((iters - prev_iters)) -le "$tol" ] && [ $((prev_iters - iters)) -le "$tol" ]; then
    plateau=$((plateau + 1))
  else
    plateau=0
  fi
  prev_iters=$iters
  if [ "$plateau" -ge "$PLATEAU_CHUNKS" ]; then
    echo "THERMALIZED after $((chunk * CHUNK)) trajectories (iteration count stable at $iters for $PLATEAU_CHUNKS chunks)"
    # Hand off the step count of the best in-band chunk, not the final
    # retune: the last retune is an extrapolation that was never validated
    # by a chunk of its own (observed: handoff n_steps=20 -> 45% production
    # acceptance while chunk-measured 26 -> 95%).
    BEST_N=$(grep -E "^chunk" "$PREFIX"_tune.log 2>/dev/null | python3 -c "
import sys, re
best, bn = 1e9, 0
for l in sys.stdin:
    m = re.search(r'n_steps=(\d+).*pred_acc=([0-9.]+)', l)
    if m:
        d = abs(float(m.group(2)) - 0.86)
        if d < best: best, bn = d, int(m.group(1))
print(bn if bn else 0)")
    [ "${BEST_N:-0}" -gt 0 ] && N_STEPS=$BEST_N
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
# Cap the per-chunk increase at 3x: runaway trajectories (integrator
# instability) inflate var(dH) far outside the smooth-error scaling regime,
# and the power law then extrapolates absurdly (e.g. n=432). A 3x step-count
# increase re-enters the stability region; further growth happens next chunk
# if genuinely needed.
n = min(n, 3*$N_STEPS)
# Runaway floor: never relax below 1.15x the largest step count that ever
# produced a runaway (<dH> > 5) this rung — downward probes across the
# stability boundary waste a chunk each and inject violent configs.
import re
floor = 0
try:
    for l in open('$PREFIX' + '_tune.log'):
        m = re.search(r'n_steps=(\d+).*<dH>=\+?(-?[0-9.e+]+)', l)
        if m and abs(float(m.group(2))) > 5: floor = max(floor, int(1.15*int(m.group(1))))
except FileNotFoundError:
    pass
n = max(n, floor)
# absolute ceiling (conf N_STEPS_MAX): transition-spike trajectories are
# force-accepted during therm and cease once equilibrated — escalating the
# step count without bound only produces timeout-length chunks
n = min(n, ${N_STEPS_MAX:-64})
print(max(4, int(round(n))))
")
    echo "  retuned n_steps -> $N_STEPS (targeting acceptance $(python3 -c "print(0.5*($ACC_LO+$ACC_HI))"))"
  fi
done

echo "MAX_CHUNKS reached without iteration-count plateau; inspect ${PREFIX}_chunk*.log"
exit 1
