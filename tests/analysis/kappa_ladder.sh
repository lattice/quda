#!/bin/bash
# Kappa ladder at 24^3x48: lower the quark mass rung by rung until the
# measured pion mass reaches am ~ 0.2.  Each rung: chunked therm-tune
# (integrator auto-retuned), production with eigentracking (adaptive
# re-anchor, pool persisted at checkpoints), unitary-point measurement,
# am extraction, then extrapolate the next kappa from m_pi^2 linear in
# 1/kappa using the last two rungs (step clamped, safety halts).
set -u
D=${DATA_DIR:?}
T=${BUILD_TESTS:?}
A=${ANALYSIS_DIR:?}
# resource path comes from the dispatcher (conf TUNECACHE); fallback for standalone use
export QUDA_RESOURCE_PATH=${QUDA_RESOURCE_PATH:-$D/tunecache3}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps

MAX_RUNGS=${MAX_RUNGS:-16}
RUNG_CONFIGS=${PROBE_CONFIGS:-20}   # probe-rung statistics
PROD_CONFIGS=${PROD_CONFIGS:-200}   # full production at each physical target
TARGETS=(${TARGET_AMS:?set TARGET_AMS, e.g. "0.43 0.39 0.34"})
target_idx=0
WINDOW_FRAC=${WINDOW_FRAC:-0.04}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-5}
LDIMS=${LADDER_DIMS:-"24 24 24 48"}
STATE=$D/ladder_state_b$(echo ${BETA:-5.3} | tr -d .).dat   # lines: kappa am (one file per beta)
LOG=$D/ladder.log

log() { echo "[ladder $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# seed prior from ensemble E if state file absent
[ -f "$STATE" ] || echo "0.13300 1.861" > "$STATE"

kappa=${LADDER_START_KAPPA:?set LADDER_START_KAPPA (first rung kappa)}
skip_therm_prefix=${LADDER_SKIP_THERM_PREFIX:-}  # reuse an existing therm-tune result for rung 1
last_cfg=${LADDER_SEED_CFG:-}   # thermalized config seeding rung-1 therm; chained thereafter

for rung in $(seq 1 $MAX_RUNGS); do
  tag=$(echo "$kappa" | sed 's/0\.//;s/^/k0/')
  ED=$D/ladder_$tag
  mkdir -p "$ED"
  log "=== RUNG $rung: kappa=$kappa ($ED) ==="

  # ---- thermalize (chunked, auto-tuned) ----
  if [ "$rung" -eq 1 ] && [ -n "$skip_therm_prefix" ] && grep -q "^TUNED:" "$skip_therm_prefix"/thermtune.log 2>/dev/null; then
    tuned=$(grep "^TUNED:" "$skip_therm_prefix"/thermtune.log | tail -1)
    log "reusing existing therm-tune: $tuned"
  else
    # COLD START at every kappa change (Dean, 2026-08-13): seeding a new
    # kappa from the previous rung's sea puts the chain far from the new
    # equilibrium in the low-mode structure, and the re-ordering transient
    # drives the integrator through near-exceptional force spikes. Cold
    # starts thermalize longer but smoothly, at stable step counts.
    SOLVER_FLAGS="${THERM_SOLVER_FLAGS:-$SOLVER_FLAGS}" \
      DIMS="$LDIMS" BETA=${BETA:-5.3} KAPPA=$kappa CSW=${CSW:-1.0} INTEGRATOR=${INTEGRATOR:-2} N_STEPS=12 \
      PREFIX=$ED/tt BUILD_TESTS=$T bash $A/therm_tune.sh > "$ED"/thermtune.log 2>&1
    if ! grep -q "^THERMALIZED" "$ED"/thermtune.log; then
      log "HALT: therm-tune failed at kappa=$kappa (see $ED/thermtune.log)"; exit 1
    fi
    tuned=$(grep "^TUNED:" "$ED"/thermtune.log | tail -1)
  fi
  n_steps=$(echo "$tuned" | sed -E 's/.*n_steps=([0-9]+).*/\1/')
  start_cfg=$(echo "$tuned" | sed -E 's/.*start_config=(\S+).*/\1/')
  log "rung $rung tuned: n_steps=$n_steps start=$start_cfg"

  # ---- production: RUNG_CONFIGS configs at checkpoint interval 5 ----
  ntraj=$((RUNG_CONFIGS * 5 * 12 / 10))  # ~20% margin for rejects
  timeout $((ntraj * 150 + 3600)) $T/hmc_test --dim $LDIMS ${SOLVER_FLAGS:?} ${PROD_MG:-} --dslash-type clover --clover-csw ${CSW:-1.0} --kappa $kappa --hmc-beta ${BETA:-5.3} \
    --hmc-integrator ${INTEGRATOR:-2} --hmc-n-steps $n_steps --hmc-tau ${TAU:-1.0} \
    --hmc-thermalization 0 --hmc-n-trajectories $ntraj \
    --hmc-gauge-infile "$start_cfg" \
    ${ET_FLAGS:-} \
    --hmc-checkpoint $CHECKPOINT_INTERVAL --hmc-checkpoint-prefix "$ED"/cfg_ \
    --gtest_filter=HMC.Production > "$ED"/production.log 2>&1
  acc=$(grep -oE "acceptance = [0-9]+/[0-9]+" "$ED"/production.log | tail -1)
  log "rung $rung production done: $acc"

  # safety: acceptance collapse or solver blowup
  accfrac=$(grep -oE "rate = [0-9.]+%" "$ED"/production.log | tail -1 | grep -oE "[0-9.]+")
  if python3 -c "exit(0 if float('$accfrac' or 0) < 50 else 1)"; then
    log "HALT: acceptance collapsed (${accfrac}%) at kappa=$kappa"; exit 1
  fi
  # only genuine outer-solver failures halt: MG-internal setup/polish solvers
  # (prefixed "MG level") run to their iteration caps by design
  if grep -v "^MG level" "$ED"/production.log | grep -q "Exceeded maximum iterations"; then
    log "HALT: solver iteration blowup at kappa=$kappa"; exit 1
  fi

  last_cfg=$(ls "$ED"/cfg_* | grep -vE "\.pool|\.evals" | sort | tail -1)

  # lambda_min record from the fresh-TRLM anchors (exceptionality watch)
  grep -oE "([Ii]nitialized. Smallest eval = |seeded from MG. Pool=[0-9]+, smallest eval=)[0-9.e+-]+" "$ED"/production.log | grep -oE "[0-9.e+-]+$" > "$ED"/lambda_min.dat
  lmin=$(sort -g "$ED"/lambda_min.dat | head -1)
  log "rung $rung lambda_min anchors: min=$lmin n=$(wc -l < "$ED"/lambda_min.dat)"

  # ---- measure: unitary point, all precisions double ----
  mkdir -p "$ED"/links
  for f in "$ED"/cfg_*; do
    case "$f" in *.pool|*.evals|*links*) continue;; esac
    n=$((10#${f##*_})); ln -sf "$f" "$ED"/links/l_cfg_${n}.lime
  done
  maxn=$(ls "$ED"/cfg_* | grep -vE "\.pool|\.evals" | sed 's/.*_//' | sort -n | tail -1)
  timeout $((10#$maxn * 120 + 1800)) $T/correlator_distribution_test --dim $LDIMS --dslash-type clover --clover-csw 1.0 --kappa $kappa \
    --tol 1e-10 --niter 20000 --prec double --prec-sloppy double --prec-refine double --prec-precondition double \
    --corrdist-config-prefix "$ED"/links/l \
    --corrdist-config-start 1 --corrdist-config-end $((10#$maxn)) --corrdist-config-step 1 \
    --corrdist-out "$ED"/pion.dat > "$ED"/measure.log 2>&1
  python3 $A/correlator_distribution_analysis.py "$ED"/pion.dat --channel G5 --block ${ANALYSIS_BLOCK:-3} > "$ED"/analysis.txt 2>&1

  # am from the mean-based effective mass averaged over the plateau window
  am=$(python3 - "$ED"/analysis.txt <<'PYEOF'
import sys, re
rows = []
in_meff = False
for line in open(sys.argv[1]):
    if "m_eff(mean C)" in line: in_meff = True; continue
    if in_meff:
        tok = line.split()
        if len(tok) >= 2 and tok[0].isdigit():
            rows.append((int(tok[0]), float(tok[1])))
T = max(t for t, m in rows) + 2
plateau = [m for t, m in rows if T//4 <= t <= T//2 - 3]
print(f"{sum(plateau)/len(plateau):.4f}" if plateau else "FAIL")
PYEOF
)
  [ "$am" = "FAIL" ] && { log "HALT: could not extract am at kappa=$kappa"; exit 1; }
  echo "$kappa $am" >> "$STATE"
  log "rung $rung RESULT: kappa=$kappa am=$am (lambda_min=$lmin)"

  # ---- within the current physical-target window? ----
  tgt=${TARGETS[$target_idx]}
  # widen the window 1.5x for the two lightest targets: rung cost dominates
  # there and a ~6% mass miss is inside the accepted volume-matching tolerance
  wfrac=$WINDOW_FRAC
  [ $target_idx -ge $(( ${#TARGETS[@]} - 2 )) ] && wfrac=$(python3 -c "print(1.5*$WINDOW_FRAC)")
  if python3 -c "exit(0 if abs(float('$am') - $tgt)/$tgt < $wfrac else 1)"; then
    log "TARGET $((target_idx+1)) WINDOW HIT: am=$am ~ $tgt at kappa=$kappa — starting $PROD_CONFIGS-config production"
    last_probe=$(ls "$ED"/cfg_* | grep -vE "\.pool|\.evals" | sort | tail -1)
    ptraj=$((PROD_CONFIGS * 5 * 12 / 10))
    timeout $((ptraj * 150 + 3600)) $T/hmc_test --dim $LDIMS ${SOLVER_FLAGS:?} ${PROD_MG:-} --dslash-type clover --clover-csw ${CSW:-1.0} --kappa $kappa --hmc-beta ${BETA:-5.3} \
      --hmc-integrator ${INTEGRATOR:-2} --hmc-n-steps $n_steps --hmc-tau ${TAU:-1.0} \
      --hmc-thermalization 0 --hmc-n-trajectories $ptraj \
      --hmc-gauge-infile "$last_probe" \
      ${ET_FLAGS:-} \
      --hmc-checkpoint $CHECKPOINT_INTERVAL --hmc-checkpoint-prefix "$ED"/prod_ \
      --gtest_filter=HMC.Production > "$ED"/production_full.log 2>&1
    for f in "$ED"/prod_*; do
      case "$f" in *.pool|*.evals) continue;; esac
      n=$((10#${f##*_})); ln -sf "$f" "$ED"/links/p_cfg_${n}.lime
    done
    pmax=$(ls "$ED"/prod_* | grep -vE "\.pool|\.evals" | sed 's/.*_//' | sort -n | tail -1)
    timeout $((10#$pmax * 150 + 3600)) $T/correlator_distribution_test --dim $LDIMS --dslash-type clover --clover-csw 1.0 --kappa $kappa \
      --tol 1e-10 --niter 20000 --prec double --prec-sloppy double --prec-refine double --prec-precondition double \
      --corrdist-config-prefix "$ED"/links/p \
      --corrdist-config-start 1 --corrdist-config-end $((10#$pmax)) --corrdist-config-step 1 \
      --corrdist-out "$ED"/pion_production.dat > "$ED"/measure_production.log 2>&1
    python3 $A/correlator_distribution_analysis.py "$ED"/pion_production.dat --channel G5 --block ${ANALYSIS_BLOCK:-4} \
      > "$ED"/analysis_production.txt 2>&1
    log "TARGET $((target_idx+1)) COMPLETE: kappa=$kappa am=$am, $(grep -c Measured "$ED"/measure_production.log) configs measured"
    target_idx=$((target_idx+1))
    if [ $target_idx -ge ${#TARGETS[@]} ]; then log "ALL TARGETS COMPLETE. Ladder done."; exit 0; fi
  fi

  # ---- next kappa: m^2 linear in 1/kappa from last two rungs, clamped ----
  export CURRENT_TARGET=${TARGETS[$target_idx]}
  kappa=$(python3 - "$STATE" "$kappa" <<'PYEOF'
import sys
pts = [tuple(map(float, l.split())) for l in open(sys.argv[1]) if l.strip()]
(k1, m1), (k2, m2) = pts[-2], pts[-1]
x1, x2 = 1/k1, 1/k2
s = (m1*m1 - m2*m2) / (x1 - x2) if x1 != x2 else None
import os
tgt = float(os.environ['CURRENT_TARGET'])
m2_target = max(tgt*tgt, 0.45 * m2*m2)
if s and s > 0:
    x_next = x2 - (m2*m2 - m2_target) / s
    k_next = 1/x_next
    # adaptive clamp: step at most a quarter of the estimated distance to
    # kappa_c (where m^2 extrapolates to zero), never more than 0.005
    x_c = x2 - m2*m2 / s
    k_c = 1/x_c
    # floor raised 0.001 -> 0.0015 and default fraction 0.25 -> 0.5: the
    # two-point linear fit is convex-biased at heavy mass and estimates
    # kappa_c far too close, throttling steps to a creep (observed at
    # beta=5.85: four ~0.001 rungs from am 0.49 to 0.40). The m2_target
    # floor above still limits each hop to a ~33% mass reduction.
    cap = min(float(os.environ.get('CLAMP_MAX', '0.005')),
              max(0.0015, float(os.environ.get('CLAMP_FRAC', '0.5')) * (k_c - k2)))
else:
    k_next = k2 + 0.0010
    cap = 0.0010
k_next = min(k_next, k2 + cap)
print(f"{k_next:.5f}")
PYEOF
)
  [ -z "$kappa" ] && { log "HALT: empty next kappa (extrapolation failed)"; exit 1; }
  log "next rung kappa=$kappa"
done
log "MAX_RUNGS reached without hitting am target; last state: $(tail -1 $STATE)"
exit 1
