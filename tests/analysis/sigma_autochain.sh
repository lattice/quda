#!/bin/bash
# sigma_autochain.sh — bridges the beta scout to the pi-pi ladder launch.
#
# Waits for the running scout (SCOUT RESULT/HALT in $SCOUT_LOG), then:
#   |suggested dbeta| >  0.05 -> update BETA in the scout conf, relaunch the
#                                scout (max $MAX_RESCOUTS attempts);
#   |suggested dbeta| <= 0.05 -> freeze beta, derive TARGET_AMS for the six
#                                pion masses from the measured 1/a, write the
#                                pi-pi ladder conf (16^3x64), launch the ladder.
#
# Usage: sigma_autochain.sh <scout_conf> <scout_log>
set -u
conf=${1:?scout conf}; slog=${2:?scout log}
source "$conf"
A=$(cd "$(dirname "$0")" && pwd)
MAX_RESCOUTS=${MAX_RESCOUTS:-2}
attempt=0

log() { echo "[autochain $(date +%H:%M:%S)] $*"; }

while :; do
  # wait for a verdict line beyond the current attempt marker
  until grep -qE "SCOUT RESULT|SCOUT HALT" <(tail -n 40 "$slog" 2>/dev/null); do sleep 300; done
  verdict=$(grep -E "SCOUT RESULT|SCOUT HALT" "$slog" | tail -1)
  log "verdict: $verdict"
  case "$verdict" in
  *"SCOUT HALT"*)
    log "scout halted — manual intervention required"; exit 1 ;;
  esac

  a_meas=$(echo "$verdict" | grep -oE "a=[0-9.]+" | grep -oE "[0-9.]+")
  beta_new=$(echo "$verdict" | grep -oE "beta=[0-9.]+" | tail -1 | grep -oE "[0-9.]+")
  db=$(python3 -c "print(abs($beta_new - $BETA))")
  if python3 -c "exit(0 if $db > 0.05 else 1)"; then
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_RESCOUTS" ]; then log "rescout limit reached"; exit 1; fi
    log "correction $db too large: re-scouting at beta=$beta_new (attempt $attempt)"
    sed -i "s/^BETA=.*/BETA=$beta_new/" "$conf"
    source "$conf"
    bash "$A"/run_campaign.sh scout "$conf" >> "$slog" 2>&1
    continue
  fi

  # ---- freeze beta, write the pi-pi ladder conf, launch ----
  ainv=$(python3 -c "print(197.327/$a_meas)")
  ams=$(python3 -c "
ainv=$ainv
print(' '.join(f'{m/ainv:.4f}' for m in (400,350,300,250,200,150)))")
  pconf=$(dirname "$conf")/b$(echo $BETA | tr -d .)_pipi.conf
  sed -e "s/^DIMS=.*/DIMS=\"16 16 16 64\"/" \
      -e "s/^BETA=.*/BETA=$BETA/" \
      -e "s/^USE_MG=.*/USE_MG=1                    # MG for seeded rungs\/production; therm is CG (USE_MG_THERM=0)/" \
      -e "s/^KAPPA_START=.*/KAPPA_START=${SCOUT_KAPPA:-0.140}/" \
      -e "s/^TARGET_AMS=.*/TARGET_AMS=\"$ams\"   # 400..150 MeV at measured ainv=${ainv%%.*} MeV/" \
      -e "s/^MAX_RUNGS=.*/MAX_RUNGS=24/" \
      -e "s/^FLOW_STEPS=.*/FLOW_STEPS=250          # fine lattice: w0 crossing needs t up to ~5/" \
      -e "s/^MEAS_SOURCES=.*/MEAS_SOURCES=6              # mixture analysis wants dense histograms/" \
      "$conf" > "$pconf"
  grep -q "^WINDOW_FRAC" "$pconf" && sed -i "s/^WINDOW_FRAC=.*/WINDOW_FRAC=0.04            # opened to 0.06 for the two lightest by ladder logic/" "$pconf"
  log "beta frozen at $BETA, a=$a_meas fm, ainv=$ainv MeV"
  log "TARGET_AMS=$ams"
  log "launching pi-pi ladder: $pconf"
  setsid nohup bash "$A"/run_campaign.sh ladder "$pconf" >> $(dirname "$conf")/pipi_ladder.log 2>&1 < /dev/null &
  log "ladder launched (pid $!)"
  exit 0
done
