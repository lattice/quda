#!/usr/bin/env python3
"""Two-component Gaussian-mixture analysis of ln C(t) parent distributions.

For each timeslice: fit 1- and 2-component Gaussian mixtures to ln C
samples (EM algorithm, numpy only), compare via BIC, and report the
mixture weight, component separation (in units of the pooled sigma), and
the BIC preference. A single hadron in equilibrium should prefer one
component; a genuine two-branch parent (the pi-pi double Gaussian
hypothesis) prefers two with a stable, t-growing separation.

Usage: mixture_analysis.py DATA.dat [--channel G5] [--tmin 2] [--tmax T/2]
"""
import argparse
import sys

import numpy as np


def read_channel(fname, channel):
    key, tt, re = [], [], []
    for line in open(fname):
        if line.startswith("#"):
            continue
        tok = line.split()
        if len(tok) != 9 or tok[5] != channel:
            continue
        key.append((int(tok[0]), tok[1], tok[2], tok[3], tok[4]))
        tt.append(int(tok[6]))
        re.append(float(tok[7]))
    if not key:
        sys.exit(f"no rows for {channel}")
    T = max(tt) + 1
    keys = sorted(set(key))
    idx = {k: i for i, k in enumerate(keys)}
    data = np.full((len(keys), T), np.nan)
    for k, t, r in zip(key, tt, re):
        data[idx[k], t] = r
    return data, T


def em_fit(x, k, iters=500, seed=0):
    """EM for a k-component 1D Gaussian mixture; returns (loglike, params)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    mu = np.quantile(x, np.linspace(0.25, 0.75, k)) + rng.normal(0, x.std() * 1e-3, k)
    sg = np.full(k, x.std())
    w = np.full(k, 1.0 / k)
    ll_old = -np.inf
    for _ in range(iters):
        # E step
        comp = w * np.exp(-0.5 * ((x[:, None] - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
        tot = comp.sum(axis=1)
        tot[tot == 0] = 1e-300
        r = comp / tot[:, None]
        # M step
        nk = r.sum(axis=0)
        w = nk / n
        mu = (r * x[:, None]).sum(axis=0) / nk
        sg = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk)
        sg = np.maximum(sg, 1e-8)
        ll = np.log(tot).sum()
        if abs(ll - ll_old) < 1e-9 * abs(ll):
            break
        ll_old = ll
    order = np.argsort(mu)
    return ll, (w[order], mu[order], sg[order])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("datafile")
    p.add_argument("--channel", default="G5")
    p.add_argument("--tmin", type=int, default=2)
    p.add_argument("--tmax", type=int, default=0)
    args = p.parse_args()

    C, T = read_channel(args.datafile, args.channel)
    tmax = args.tmax or T // 2
    print(f"# {args.datafile} channel={args.channel}: {C.shape[0]} samples/t")
    print("#  t     N    dBIC(2-1)  verdict    w2     sep/sigma   mu1        mu2")
    for t in range(args.tmin, tmax):
        x = C[:, t]
        x = np.log(x[x > 0])
        n = len(x)
        if n < 100:
            continue
        ll1, _ = em_fit(x, 1)
        best = max((em_fit(x, 2, seed=s) for s in range(3)), key=lambda r: r[0])
        ll2, (w, mu, sg) = best
        # BIC = -2 ln L + p ln n ; p = 2 vs 5
        dbic = (-2 * ll2 + 5 * np.log(n)) - (-2 * ll1 + 2 * np.log(n))
        sep = abs(mu[1] - mu[0]) / np.sqrt((sg**2 * w).sum())
        verdict = "TWO-COMP" if dbic < -10 else ("weak-2" if dbic < 0 else "single")
        print(f"{t:4d} {n:6d}  {dbic:+9.1f}  {verdict:8s}  {min(w):.3f}   {sep:8.2f}  {mu[0]:+.4f}  {mu[1]:+.4f}")


if __name__ == "__main__":
    main()
