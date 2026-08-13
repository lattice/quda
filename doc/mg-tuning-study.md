# MG Tuning Study — protocol for the dedicated pause

Directive (Dean, 2026-08-13): when the time comes, pause production and
run a separate, systematic MG tuning study rather than tuning
opportunistically inside the campaign. This document is the protocol so
the study runs properly on the day.

## Objective

Minimize wall-clock per HMC trajectory (force + action solves) and per
measurement solve at the campaign's working points, with the MG-GCR
preconditioned solver on 16^3x64 (and later 32^3x64) Nf=2 clover
lattices on a single GB10.

## When to run

Trigger any of:
- entering a production at a mass point where outer iterations exceed
  ~30 with a fresh setup, or
- before the first 32^3x64 production, or
- before committing to the 250/200/150 MeV productions.

Cost of the study: ~2-4 h GPU. It pays for itself if it saves >10% of a
single production.

## Test set and metric

- Configs: 4 thermalized configs of the target ensemble (spread across
  the stream; include the config with the smallest lambda_min anchor —
  the exceptional tail matters more than the average).
- Solves: the production mix — NORMOP_PC force-solve tolerance (1e-8
  path) and one all-double measurement-style solve — 12 RHS each via
  invert_test / a small driver, plus one full HMC trajectory per
  candidate for end-to-end confirmation.
- Metric: wall-clock per solve INCLUDING amortized setup
  (setup_time / solves_per_setup_interval) — a candidate that halves
  iterations but doubles setup only wins if the refresh cadence allows.
  Record: outer iterations, setup time, solve time, total per-traj time.

## Parameter axes (in priority order)

1. **Blocking** (strongest lever at our aspect ratio):
   - 2-level: `--mg-block-size 0 2 2 2 8` (aspect-respecting 2228,
     coarse 8^3x8 — Dean's lead candidate) vs default 4^4 / 2^4.
   - 3-level: 2228 then 2222 (-> 4^3x4); also 2224+2222.
   - At 32^3x64 later: 4448 and 2228+2222 variants.
2. **Levels**: 2 vs 3. The coarse-grid solve dominates when the coarse
   volume is still large (8^3x32 at 2^4 blocking is oversized); 3-level
   pays when level-1 GCR iterations exceed ~20.
3. **Smoother iterations**: `--mg-nu-pre {0,2,4} x --mg-nu-post {2,4,8}`
   around the current (2,2). More post-smoothing typically buys outer
   iterations at fixed setup; measure the tradeoff at the light masses
   where the smoother struggles against near-null modes.
4. **Precision**: null-space and halo precision at QUARTER
   (`--prec-null quarter`, `--mg-smoother-halo-prec quarter`) vs half vs
   single. GB10 FP64 is weak and bandwidth is the binding resource:
   quarter-precision null vectors halve the dominant memory traffic of
   the prolongator/restrictor and coarse links. Validate against the
   exactness policy: outer solver stays double-CG/GCR with true
   residuals, so precision here affects ITERATIONS not correctness —
   but confirm per-config correlator agreement on 2 configs anyway (V7
   gate pattern).
5. **Low-mode deflation of the coarsest grid**: `--mg-eig 1/2 true` with
   n_ev ~ 16-32 on the coarsest level (deflated coarse solve). Interacts
   with eigentracking: the ET pool tracks the FINE-level low modes; the
   coarse-level eigensolver is separate and cheap (coarse volume is
   tiny). Expected to matter most at 250 MeV and below where the coarse
   operator inherits the criticality.
6. **Setup cadence** (already partially deployed): setup interval during
   therm (currently 3) and production (currently 25), and the ET
   pool-driven refresh (`--eigentracking-mg-refresh-iters {0,8,16}`)
   vs standard CG-based refresh. Use the per-chunk secs_per_traj
   telemetry already in the therm logs.

## Design

Not a full grid — staged sweeps with the best of each stage carried
forward: blocking (5 candidates) -> levels (2) -> smoother (4 combos) ->
precision (3) -> deflation (2) -> cadence (3): ~19 runs x ~5 min plus
2 HMC trajectories per finalist. Automate as a dispatcher subcommand
(`run_campaign.sh mgtune <conf> <cfg...>`) writing one CSV row per
candidate; keep the config list and seeds fixed across candidates.

## Baselines measured so far (16^3x64, beta=5.85)

| context | outer iters | notes |
|---|---|---|
| production, fresh setup, am=0.33 | 6-16 | 2-level default blocking |
| therm, stale setup (no refresh), am~0.22 | up to 180 | drove the setup-interval fix |
| CG (no MG) therm, am~0.49 | ~150-235 | reference |

## Deliverable

Winning parameter set written into the campaign confs as MG_LEVELS /
MG_EXTRA_FLAGS (+ any new conf variables), a committed CSV + short
summary appended to this file, and the same study repeated once at
32^3x64 before the volume-anchor production.
