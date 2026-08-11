#!/usr/bin/env python3
"""Extract w0/a from performWFlowQuda logs and convert targets.

Env: FLOW_DIR (dir of flow_*.log), W0_FM (physical w0, default 0.176),
A_LO/A_HI (sanity band on a in fm).  Prints w0/a, a, 1/a, and am values
for 500/450/400 MeV; exits nonzero on extraction failure or insane a.
"""
import glob
import os
import re
import statistics
import sys

flow_dir = os.environ["FLOW_DIR"]
w0_fm = float(os.environ.get("W0_FM", 0.176))
a_lo = float(os.environ.get("A_LO", 0.08))
a_hi = float(os.environ.get("A_HI", 0.35))

w0s = []
for fn in sorted(glob.glob(os.path.join(flow_dir, "flow_*.log"))):
    ts, Es = [], []
    for line in open(fn):
        m = re.match(r"performWFlowQuda: ([0-9.e+-]+) ([0-9.e+-]+) ([0-9.e+-]+) ", line)
        if m:
            ts.append(float(m.group(1)))
            Es.append(float(m.group(2)) + float(m.group(3)))
    if len(ts) < 6:
        continue
    W = [(ts[i], ts[i] * (ts[i+1]**2*Es[i+1] - ts[i-1]**2*Es[i-1]) / (ts[i+1] - ts[i-1]))
         for i in range(1, len(ts) - 1)]
    for (t1, w1), (t2, w2) in zip(W, W[1:]):
        if w1 < 0.3 <= w2:
            w0s.append((t1 + (0.3 - w1) * (t2 - t1) / (w2 - w1)) ** 0.5)
            break

if not w0s:
    print("FAIL: no w0 extracted from", flow_dir)
    sys.exit(1)

w0m = statistics.mean(w0s)
a_fm = w0_fm / w0m
ainv = 197.327 / a_fm
print(f"w0/a = {w0m:.4f} ({len(w0s)} configs, spread {max(w0s)-min(w0s):.4f})")
print(f"a = {a_fm:.4f} fm   ainv = {ainv:.1f} MeV")
if not (a_lo <= a_fm <= a_hi):
    print(f"FAIL: a outside sanity band [{a_lo}, {a_hi}] fm")
    sys.exit(1)
for mev in (500, 450, 400):
    print(f"target {mev} MeV -> am = {mev/ainv:.4f}")
print(f"AMS {500/ainv:.4f} {450/ainv:.4f} {400/ainv:.4f}")
