#!/usr/bin/env python3
"""Wave-packet view of the correlator distribution.

For each timeslice, histogram ln C(t) and estimate the parent Gaussian
(mu, sigma).  Then display the evolution of that Gaussian along t:
mu(t) moving ballistically (constant velocity = -m_eff) and sigma(t)
broadening, in direct analogy with a dispersing quantum-mechanical
Gaussian wave packet.

Usage:
  wave_packet_analysis.py DATAFILE [--channel G5] [--tmin 1] [--tmax 15]
                          [--fit-lo 4] [--fit-hi 14] [--out PREFIX]
"""

import argparse
import sys

import numpy as np


def read_dataset(fname, channel):
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
    return np.array(cfgs), T, data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("datafile")
    p.add_argument("--channel", default="G5")
    p.add_argument("--tmin", type=int, default=1)
    p.add_argument("--tmax", type=int, default=15)
    p.add_argument("--fit-lo", type=int, default=4, help="first t of the ballistic-fit window")
    p.add_argument("--fit-hi", type=int, default=14, help="last t of the ballistic-fit window")
    p.add_argument("--out", default="wave_packet")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cfgs, T, C = read_dataset(args.datafile, args.channel)
    N = len(cfgs)
    ts = list(range(args.tmin, args.tmax + 1))

    # parent-Gaussian estimates per timeslice (ML for a Gaussian)
    mu = np.full(T, np.nan)
    sig = np.full(T, np.nan)
    mu_err = np.full(T, np.nan)
    for t in range(T):
        c_t = C[:, t]
        c_t = c_t[c_t > 0]
        lnc = np.log(c_t)
        mu[t] = lnc.mean()
        sig[t] = lnc.std(ddof=1)
        mu_err[t] = sig[t] / np.sqrt(len(lnc))

    # sequential single-hue ramp (light -> dark teal) along t
    ramp = LinearSegmentedColormap.from_list("teal", ["#BFE3E0", "#0E4744"])
    ink, muted, grid = "#23262B", "#6D7178", "#E3E2DC"

    # ---- Figure 1: the wave packet -------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    offset_scale = 1.0  # vertical offset per timeslice, in density units
    for i, t in enumerate(ts):
        lnc = np.log(C[:, t][C[:, t] > 0])
        color = ramp(i / max(1, len(ts) - 1))
        hist, edges = np.histogram(lnc, bins=max(24, N // 250), density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        base = i * offset_scale
        ax.fill_between(centers, base, base + hist * sig[t], color=color, alpha=0.55, lw=0)
        x = np.linspace(mu[t] - 4 * sig[t], mu[t] + 4 * sig[t], 300)
        gauss = np.exp(-((x - mu[t]) ** 2) / (2 * sig[t] ** 2)) / np.sqrt(2 * np.pi * sig[t] ** 2)
        ax.plot(x, base + gauss * sig[t], color=ink, lw=1.0, ls="--")
        ax.plot(mu[t], base, marker="o", ms=5, color=color, mec=ink, mew=0.5, clip_on=False)
        ax.text(mu[t] + 4.5 * sig[t], base + 0.12, f"t={t}", fontsize=8, color=muted, va="bottom")

    # ballistic line through the crest positions
    tt = np.array(ts, dtype=float)
    sel = (tt >= args.fit_lo) & (tt <= args.fit_hi)
    v, b = np.polyfit(tt[sel], mu[args.tmin : args.tmax + 1][sel], 1)
    ax.plot(v * tt + b, (tt - args.tmin) * offset_scale, color="#A66A1F", lw=1.4,
            label=f"ballistic crest: v = {v:+.4f} / slice")

    ax.set_xlabel("ln C", color=ink)
    ax.set_ylabel("timeslice (offset)", color=ink)
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("P(ln C, t): a dispersing wave packet", color=ink)
    for s in ax.spines.values():
        s.set_color(grid)
    fig.tight_layout()
    fig.savefig(f"{args.out}_ridge.png", dpi=150)

    # ---- Figure 2: ballistic motion of the mean ------------------------
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    t_all = np.arange(args.tmin, args.tmax + 1)
    ax1.errorbar(t_all, mu[t_all], yerr=mu_err[t_all], fmt="o", ms=4, color="#2A7F7A",
                 ecolor=muted, capsize=2)
    ax1.plot(tt, v * tt + b, color="#A66A1F", lw=1.2,
             label=f"linear fit t∈[{args.fit_lo},{args.fit_hi}]: v = {v:+.5f}")
    ax1.set_ylabel("μ(t) = ⟨ln C⟩", color=ink)
    ax1.legend(frameon=False)
    ax1.set_title("Ballistic motion of the packet centre", color=ink)
    resid = mu[t_all] - (v * t_all + b)
    ax2.axhline(0, color=grid, lw=1)
    ax2.errorbar(t_all, resid, yerr=mu_err[t_all], fmt="o", ms=4, color="#2A7F7A", ecolor=muted, capsize=2)
    ax2.set_xlabel("t", color=ink)
    ax2.set_ylabel("residual", color=ink)
    for a in (ax1, ax2):
        for s in a.spines.values():
            s.set_color(grid)
        a.grid(color=grid, lw=0.5, alpha=0.6)
    fig2.tight_layout()
    fig2.savefig(f"{args.out}_ballistic.png", dpi=150)

    # ---- Figure 3: dispersion ------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    var = sig**2
    # variance error from Gaussian sampling theory: var * sqrt(2/(N-1))
    var_err = var * np.sqrt(2.0 / (N - 1))
    ax3.errorbar(t_all, var[t_all], yerr=var_err[t_all], fmt="o", ms=4, color="#2A7F7A",
                 ecolor=muted, capsize=2)
    d, c0 = np.polyfit(tt[sel], var[args.tmin : args.tmax + 1][sel], 1)
    ax3.plot(tt, d * tt + c0, color="#A66A1F", lw=1.2,
             label=f"diffusive fit: σ² = {c0:+.3f} + {d:.4f}·t")
    ax3.set_xlabel("t", color=ink)
    ax3.set_ylabel("σ²(t) = Var[ln C]", color=ink)
    ax3.legend(frameon=False)
    ax3.set_title("Dispersion of the packet", color=ink)
    for s in ax3.spines.values():
        s.set_color(grid)
    ax3.grid(color=grid, lw=0.5, alpha=0.6)
    fig3.tight_layout()
    fig3.savefig(f"{args.out}_dispersion.png", dpi=150)

    # ---- numbers --------------------------------------------------------
    print(f"# {args.datafile}: {N} configs, channel {args.channel}")
    print(f"# ballistic velocity (t in [{args.fit_lo},{args.fit_hi}]): v = {v:+.6f} per slice")
    print(f"# diffusion constant: d(sigma^2)/dt = {d:.6f} per slice")
    print("#  t     mu           sigma       sigma^2")
    for t in t_all:
        print(f"{t:4d}  {mu[t]:+.6e}  {sig[t]:.6e}  {var[t]:.6e}")


if __name__ == "__main__":
    main()
