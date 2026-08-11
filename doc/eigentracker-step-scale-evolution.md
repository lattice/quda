# Eigentracker next step: chronological prediction at integrator-step scale

*Working note, experimental/log-normal branch, 2026-08-07.  Context: the
adaptive re-anchoring work in commits `8573907b7` (residual-triggered
TRLM re-anchor + single-matvec subspace probe) and `6124dd325` (pool
persistence across restarts/chunks).*

## What we measured

The Rayleigh-Ritz pool evolution is exact in isolation
(`EigenTracking.StaticFixedPoint`: bit-exact eigenvalues, pool
orthonormality at 2e-15 on a static field) but its tracking radius is
far smaller than a trajectory:

| evolution step scale        | behaviour                                     |
|-----------------------------|-----------------------------------------------|
| tau = 0.01 per RR step      | tracks true lambda_min to ~1e-3 per step      |
| tau = 0.125 (one FGI-10 integrator step) | untested — plausibly inside the tracking radius |
| tau = 1.0 per RR step (current per-trajectory driving) | complete subspace decoherence, every trajectory: pool reports RQ ~1.1 of a lost subspace while true lambda_min stays ~0.071 |

Conclusion: as currently driven, the tracker is a **low-mode subspace
evolver** — adequate for rapid MG setup refresh (null vectors only need
to span the low modes approximately) — not an eigenvector tracker.

## The architecture to implement

Predictor–corrector layering at integrator-step granularity:

1. **Predictor (every integrator step, ~free):** apply the
   chronologically forecast rotation (the existing `EigenForecast`
   rotation-history machinery *is* the chronological predictor; today it
   is starved at one rotation per trajectory — at step scale it gets
   n_steps times the history at n_steps times finer spacing).
2. **Corrector (sparse, gated):** full RR evolution only when the
   single-matvec subspace probe (`EigenTracker::subspaceResidual`)
   exceeds a correction threshold.  Full RR at every step would cost
   ~pool-size matvecs ≈ +40% trajectory cost at nEv=8/pool 15; gating
   keeps the corrector rare when the forecast is good.
3. **Ground truth (checkpoint anchors):** fresh TRLM at the checkpoint
   interval (already implemented via `--eigentracking-fresh-interval`,
   aligned with `--hmc-checkpoint`), so every saved configuration
   carries a converged lambda_min for the exceptional-configuration
   diagnostics regardless of tracking quality between anchors.

## Implementation checklist

- [ ] Move the evolve/forecast hooks from the trajectory boundary
      (`EigenTrackingState::beforeTrajectory` / `betweenTrajectories`)
      to the integrator step (the per-step `forceUpdate(matHalf)` site
      in `hmcTrajectoryQuda`, interface_quda.cpp ~line 5590).
- [ ] Record forecast rotations per step; extend `EigenForecast` order
      handling to the longer, finer-spaced history.
- [ ] Gate RR correction on `subspaceResidual` (new threshold param,
      distinct from the re-anchor threshold; suggest correcting at
      ~0.02 and re-anchoring at ~0.2).
- [ ] Promote the subspace-probe log line from VERBOSE to SUMMARIZE so
      production logs always carry the deviation series.
- [ ] Validate: per-step tracking at tau = 1.0 trajectories should hold
      lambda_min near truth between TRLM anchors (compare anchor values
      against tracked values at anchor time); StaticFixedPoint must stay
      bit-exact.

## Decision gate

Await the deflation sweep on ensemble F (`~/lognormal_data/ensembleF/
sweep/sweep.dat`): if force-solve CG iterations are flat in pool size,
the solves do not consume the pool and the *physics program* needs only
(a) MG-refresh-grade subspace evolution and (b) TRLM anchors at
checkpoints — both already in place.  Step-scale fidelity then becomes
relevant when MG-assisted light-mass running arrives at the bottom of
the kappa ladder, where deflation/setup-refresh quality directly sets
solve cost.

## Decision gate: RESOLVED (2026-08-08 sweep, ensemble F thermalized state)

Force-solve CG iterations are flat in pool size (38.9 / 38.9 / 38.0 /
38.0 for ET-off / nEv 4 / 8 / 16): **the force solves do not consume
the pool** — the deflation benefit axis is currently zero.  TRLM anchor
cost is ~250 operator applications roughly independent of nEv (the
n_kr floor dominates for small pools).  Consequences:

- The cost/benefit crossover is at the smallest pool giving reliable
  lambda_min anchors: **nEv = 4-8**.
- Step-scale predictor-corrector fidelity work is deferred (as gated).
- The higher-value upstream item is wiring the pool into force-solve
  deflation (deflated CG / init-guess projection) — without it the
  eigentracker is a diagnostics instrument, which is all this physics
  program requires, but not what the machinery was built for.

Sweep caveats: per-point wall-clock numbers were truncated by the
30-minute orchestration timeout (2-4 of 6 trajectories completed per
point); CG-iteration flatness and TRLM cost are unaffected.  The
subspace-residual column was empty because the probe logs at VERBOSE —
promote to SUMMARIZE (checklist item above) before the next sweep.

## MG bring-up: RESOLVED (2026-08-10/11)

The MG crashes (calculateY precision mismatch, MGPreconditionedRun
segfault) were precision-flag incoherence, not library defects: MG
internals are single-precision and collide with --prec-sloppy double.
The working set is

    --prec double --prec-sloppy single --prec-precondition single
    --prec-null single --mg-levels 2 --inv-multigrid true

Measured at kappa=0.1456, 12^3x32: 12 MG-GCR outer iterations to 1e-10
(true residual 5e-11) vs 105 plain-CG iterations; HMC+MG 4/4 accepted
trajectories with eigentracking seeding its 24-vector pool from the MG
null space (no TRLM cost; fresh-TRLM anchors remain the exact
lambda_min diagnostic).  MG-era eigentracking is live; the step-scale
predictor-corrector item remains open for when pool fidelity between
anchors matters.

## Known issue: HMC MG setup-refresh path (2026-08-11)

--hmc-mg-setup-interval N triggers an MG re-setup during the run that
fails coarse-operator verification (L2 deviation ~0.4 vs 2e-3) on the
evolved gauge, aborting the run; the initial setup on the same fields
verifies fine.  Mitigation: refresh disabled; seeded rung starts keep
the setup field close to the running field so stale-setup drift is
small.  Upstream: debug the re-setup path (stale vector reuse vs full
regeneration) before long single-setup productions at light mass.
