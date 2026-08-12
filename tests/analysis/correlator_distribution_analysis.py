#!/usr/bin/env python3
"""Log-normality analysis of per-configuration correlators.

Reads the dataset written by correlator_distribution_test and, for a
chosen channel, computes per-timeslice cumulants kappa_1..kappa_4 of
ln C(t) with blocked-jackknife errors, tests whether kappa_3 and
kappa_4 are consistent with zero (claim C1: log-normality), and
compares the mean-based effective mass with the log-normal estimator
Delta_t(kappa_1 + kappa_2/2) (input to claim C2: bias closure).

Usage:
  correlator_distribution_analysis.py DATAFILE [--channel G5] [--block 1]
                                      [--histogram t1,t2,...] [--out PREFIX]

Only numpy is required; matplotlib is optional (histogram plots).
"""

import argparse
import sys

import numpy as np


def read_dataset(fname, channel):
    """Return (cfgs, T, data, n_src) where each row of data is one
    (configuration, source) sample of Re C(t), rows ordered so that all
    sources of a configuration are contiguous (required for correct
    jackknife blocking by configuration), and n_src is the (uniform)
    number of sources per configuration."""
    key_col, t_col, re_col = [], [], []
    with open(fname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            tok = line.split()
            if len(tok) != 9 or tok[5] != channel:
                continue
            # sample key: (cfg, sx, sy, sz, st) — one row per source
            key_col.append((int(tok[0]), int(tok[1]), int(tok[2]), int(tok[3]), int(tok[4])))
            t_col.append(int(tok[6]))
            re_col.append(float(tok[7]))
    if not key_col:
        sys.exit(f"no rows for channel {channel} in {fname}")
    keys = sorted(set(key_col))  # sorts by cfg first: sources stay contiguous
    T = max(t_col) + 1
    index = {k: i for i, k in enumerate(keys)}
    data = np.full((len(keys), T), np.nan)
    for k, t, re in zip(key_col, t_col, re_col):
        data[index[k], t] = re
    if np.isnan(data).any():
        sys.exit("dataset has missing (sample, t) entries")
    cfgs = sorted(set(k[0] for k in keys))
    n_src = len(keys) // len(cfgs)
    if n_src * len(cfgs) != len(keys):
        sys.exit("non-uniform number of sources per configuration")
    return np.array(cfgs), T, data, n_src


def cumulants(x):
    """kappa_1..kappa_4 of samples x (kappa_4 = excess, unstandardized)."""
    m = x.mean()
    d = x - m
    m2 = (d**2).mean()
    m3 = (d**3).mean()
    m4 = (d**4).mean()
    return np.array([m, m2, m3, m4 - 3 * m2**2])


def blocked_jackknife(samples, func, block):
    """Jackknife estimate and error of func over blocked samples."""
    n = len(samples) // block
    if n < 2:
        sys.exit("not enough blocks for jackknife; reduce --block")
    trimmed = samples[: n * block]
    estimates = []
    for i in range(n):
        mask = np.ones(n * block, dtype=bool)
        mask[i * block : (i + 1) * block] = False
        estimates.append(func(trimmed[mask]))
    estimates = np.array(estimates)
    center = func(trimmed)
    err = np.sqrt((n - 1) / n * ((estimates - center) ** 2).sum(axis=0))
    return center, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("datafile")
    p.add_argument("--channel", default="G5", help="channel name (default G5, the pion)")
    p.add_argument("--block", type=int, default=1, help="jackknife block length in configs")
    p.add_argument("--histogram", default="", help="comma-separated timeslices to histogram")
    p.add_argument("--out", default="", help="output prefix for histogram plots")
    args = p.parse_args()

    cfgs, T, C, n_src = read_dataset(args.datafile, args.channel)
    N = len(cfgs)
    # --block is in units of configurations; sources within a configuration
    # are correlated, so they always share a jackknife block
    args.block *= n_src
    print(f"# {args.datafile}: channel {args.channel}, {N} configurations x {n_src} sources, T = {T}")

    n_nonpos = (C <= 0).sum()
    if n_nonpos:
        print(f"# WARNING: {n_nonpos} non-positive correlator values; "
              "log-normal analysis restricted to C > 0 per timeslice")

    print("#  t  kappa1 (err)        kappa2 (err)        kappa3 (err)  [sig]   kappa4 (err)  [sig]")
    k1 = np.full(T, np.nan)
    k2 = np.full(T, np.nan)
    for t in range(T):
        c_t = C[:, t]
        c_t = c_t[c_t > 0]
        if len(c_t) < 2 * args.block:
            continue
        lnc = np.log(c_t)
        k, err = blocked_jackknife(lnc, cumulants, args.block)
        k1[t], k2[t] = k[0], k[1]
        sig3 = abs(k[2]) / err[2] if err[2] > 0 else 0.0
        sig4 = abs(k[3]) / err[3] if err[3] > 0 else 0.0
        print(f"{t:4d}  {k[0]:+.6e} ({err[0]:.1e})  {k[1]:+.6e} ({err[1]:.1e})  "
              f"{k[2]:+.2e} ({err[2]:.1e}) [{sig3:4.1f}]  {k[3]:+.2e} ({err[3]:.1e}) [{sig4:4.1f}]")

    # Effective-mass estimator battery with blocked-jackknife errors:
    #   mean:  m(t) = ln(<C(t)>/<C(t+1)>)
    #   n=2:   log-normal, from g2 = k1 + k2/2
    #   n=3:   cumulant expansion truncated at n_max=3, g3 = k1 + k2/2 + k3/6
    # ln<C> = sum_n kappa_n/n!, so n=3 corrects the log-normal estimator's
    # bias with the measured skewness while keeping its variance behaviour.
    def estimator_battery(sample):
        mean_c = sample.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lnc = np.log(np.where(sample > 0, sample, np.nan))
        m1 = np.nanmean(lnc, axis=0)
        d = lnc - m1
        m2 = np.nanmean(d**2, axis=0)
        m3 = np.nanmean(d**3, axis=0)
        g2 = m1 + 0.5 * m2
        g3 = g2 + m3 / 6.0
        with np.errstate(invalid="ignore", divide="ignore"):
            m_mean = np.log(mean_c[:-1] / mean_c[1:])
        return np.array([m_mean, g2[:-1] - g2[1:], g3[:-1] - g3[1:]])

    n = (len(C) // args.block) * args.block
    trimmed = C[:n]
    nb = n // args.block
    center = estimator_battery(trimmed)
    jk = np.empty((nb,) + center.shape)
    for i in range(nb):
        mask = np.ones(n, dtype=bool)
        mask[i * args.block : (i + 1) * args.block] = False
        jk[i] = estimator_battery(trimmed[mask])
    err = np.sqrt((nb - 1) / nb * ((jk - center) ** 2).sum(axis=0))

    print("#")
    print("#  t  m_eff(mean C) (err)     m_eff(n=2 lognormal) (err)  m_eff(n=3 cumulant) (err)")
    for t in range(T - 1):
        print(f"{t:4d}  {center[0][t]:+.6f} ({err[0][t]:.6f})   {center[1][t]:+.6f} ({err[1][t]:.6f})     "
              f"{center[2][t]:+.6f} ({err[2][t]:.6f})")

    if args.histogram:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("matplotlib not available for --histogram")
        ts = [int(s) for s in args.histogram.split(",")]
        fig, axes = plt.subplots(1, len(ts), figsize=(5 * len(ts), 4), squeeze=False)
        for ax, t in zip(axes[0], ts):
            c_t = C[:, t]
            lnc = np.log(c_t[c_t > 0])
            ax.hist(lnc, bins=max(10, N // 50), density=True, alpha=0.6, label=f"ln C(t={t})")
            x = np.linspace(lnc.min(), lnc.max(), 200)
            mu, var = lnc.mean(), lnc.var()
            ax.plot(x, np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var),
                    "k--", label="Gaussian")
            ax.set_xlabel("ln C")
            ax.legend()
        prefix = args.out if args.out else "lognormal"
        fig.tight_layout()
        fig.savefig(f"{prefix}_hist.png", dpi=150)
        print(f"# histogram written to {prefix}_hist.png")


if __name__ == "__main__":
    main()
