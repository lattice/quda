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
    """Return (cfgs, T, data) where data[i, t] = Re C(t) on config i."""
    cfg_col, t_col, re_col = [], [], []
    with open(fname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            tok = line.split()
            if len(tok) != 9 or tok[5] != channel:
                continue
            cfg_col.append(int(tok[0]))
            t_col.append(int(tok[6]))
            re_col.append(float(tok[7]))
    if not cfg_col:
        sys.exit(f"no rows for channel {channel} in {fname}")
    cfgs = sorted(set(cfg_col))
    T = max(t_col) + 1
    index = {c: i for i, c in enumerate(cfgs)}
    data = np.full((len(cfgs), T), np.nan)
    for c, t, re in zip(cfg_col, t_col, re_col):
        data[index[c], t] = re
    if np.isnan(data).any():
        sys.exit("dataset has missing (cfg, t) entries")
    return np.array(cfgs), T, data


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

    cfgs, T, C = read_dataset(args.datafile, args.channel)
    N = len(cfgs)
    print(f"# {args.datafile}: channel {args.channel}, {N} configurations, T = {T}")

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

    print("#")
    print("#  t  m_eff(mean C)     m_eff(log-normal)   difference")
    mean_c = C.mean(axis=0)
    for t in range(T - 1):
        m_mean = np.nan
        if mean_c[t] > 0 and mean_c[t + 1] > 0:
            m_mean = np.log(mean_c[t] / mean_c[t + 1])
        m_ln = np.nan
        if np.isfinite(k1[t]) and np.isfinite(k1[t + 1]):
            mu_sig_t = k1[t] + 0.5 * k2[t]
            mu_sig_t1 = k1[t + 1] + 0.5 * k2[t + 1]
            m_ln = mu_sig_t - mu_sig_t1
        diff = m_mean - m_ln if np.isfinite(m_mean) and np.isfinite(m_ln) else np.nan
        print(f"{t:4d}  {m_mean:+.8e}  {m_ln:+.8e}  {diff:+.2e}")

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
