#!/bin/bash
# Residual-trajectory probe: run one verbose pion solve on a given
# configuration and classify the CG convergence as healthy geometric,
# critical slowing down (clean geometric, rate near 1), or stagnation
# (residual plateau / true-vs-iterated divergence).
#
# Usage: CFG=<config> KAPPA=<k> DIMS="12 12 12 32" bash residual_probe.sh
set -u
BUILD_TESTS=${BUILD_TESTS:-$HOME/applications/build_quda/tests}
DIMS=${DIMS:-"12 12 12 32"}
KAPPA=${KAPPA:?}
CFG=${CFG:?}
OUT=${OUT:-/tmp/residual_probe}
mkdir -p "$OUT"

ln -sf "$CFG" "$OUT/rp_cfg_1.lime"
timeout 1800 "$BUILD_TESTS"/correlator_distribution_test --dim $DIMS --dslash-type clover --clover-csw 1.0 \
  --kappa "$KAPPA" --tol 1e-10 --niter 20000 \
  --prec double --prec-sloppy double --prec-refine double --prec-precondition double \
  --verbosity verbose \
  --corrdist-config-prefix "$OUT/rp" --corrdist-config-start 1 --corrdist-config-end 1 \
  --corrdist-out "$OUT/rp.dat" > "$OUT/probe.log" 2>&1

python3 - "$OUT/probe.log" <<'PYEOF'
import re, sys, math
# collect the first full solve's per-iteration residuals
res = []
for line in open(sys.argv[1]):
    m = re.search(r"CG: (\d+) iterations?, <r,r> = ([0-9.e+-]+)", line)
    if not m:
        m = re.search(r"CG: (\d+) iterations, r2 = ([0-9.e+-]+)", line)
    if m:
        it, r2 = int(m.group(1)), float(m.group(2))
        if res and it < res[-1][0]:  # next solve started
            break
        res.append((it, r2))
if len(res) < 10:
    print("PROBE FAIL: too few residual samples; check verbose output format in probe.log")
    sys.exit(0)
its = [i for i, _ in res]
lgr = [0.5 * math.log10(r) for _, r in res]  # log10 |r|
n = len(res)
# geometric rate over early vs late thirds (decades per iteration)
def slope(seg):
    (i0, l0), (i1, l1) = seg[0], seg[-1]
    return (l1 - l0) / max(1, i1 - i0)
early = slope(list(zip(its, lgr))[: n // 3])
late = slope(list(zip(its, lgr))[-n // 3 :])
total = slope(list(zip(its, lgr)))
# stagnation index: late progress relative to early progress
stag = late / early if early < 0 else float("nan")
print(f"solve: {its[-1]} iterations, log10|r| {lgr[0]:.2f} -> {lgr[-1]:.2f}")
print(f"convergence rate: {total:.4f} decades/iter (early {early:.4f}, late {late:.4f})")
print(f"stagnation index (late/early rate): {stag:.2f}")
if stag < 0.3:
    print("VERDICT: STAGNATION — residual progress collapsed late; check precision floors/reliable updates")
elif -total < 0.01:
    print("VERDICT: CRITICAL SLOWING DOWN — clean but slow geometric convergence; MG territory")
else:
    print("VERDICT: healthy geometric convergence")
PYEOF
