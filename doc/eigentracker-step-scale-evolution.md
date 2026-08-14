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

## RESOLVED: HMC MG setup-refresh failure = stale sloppy-gauge ghost zones (2026-08-11)

--hmc-mg-setup-interval N triggered an MG re-setup that failed
coarse-operator verification (L2 deviation ~0.4 vs 2e-3) on the evolved
gauge while the initial setup on the same fields verified fine.

Root cause (established by controlled bisection, all inputs
hash-verified): the Wilson-type dslash reads gauge ghost pads even on a
single GPU (no_comms_fill — "dslash kernels presently require this"),
and hmcRefreshResidentGaugeState's in-place gaugeSloppy->copy(precise)
does not reliably refresh the destination's ghost zone.  After a
trajectory of in-place evolution the sloppy family's ghost links are
stale by O(1) relative to the bulk, so the boundary-corrupted sloppy
operator gaps out the near-null modes; the 20-iteration null-vector
polish then converges to machine zero (solutions ~1e-11 vs healthy
~1e-5), post-orthonormalization normalises that noise into garbage null
vectors, and D_c = P^dag D P fails at 0.38.  Evidence chain: deviation
scales with tau (passes at tau=1e-3); sloppy BULK bit-identical to a
fresh copy of precise (element diff 0.0) yet in-place recopy does not
fix while family reallocation does; B[0]/cloverSloppy/gaugePrecondition
hashes identical between failing and passing runs; a bare
gaugeSloppy->exchangeGhost() with no reallocation fixes verification
completely.

Fix: hmcRefreshResidentGaugeState now calls exchangeGhost() on each
distinct sloppy-family gauge after its in-place copy (and now also
covers gaugeRefinement/gaugeEigensolver, which were previously never
refreshed at all).  Side benefit: the stale ghosts also made every
mixed-precision inner solve inconsistent with the outer operator at the
boundary throughout MD — the likely cause of the reliable-update
stagnation (~1e-7 CG stalls) observed earlier in the campaign.
--hmc-mg-setup-interval is safe to enable again.

## Nested-FGI deployment findings (2026-08-13)

Validated for production at 16^3x64 beta=5.85: 36 outer x 8 inner with
per-trajectory MG re-setup gives dH ~ 0.1 at 100% acceptance, ~2x
cheaper per accepted trajectory than 80-step single FGI, gap widening
toward light mass. Two characterized limitations:

1. Frozen-basis decoherence ACROSS trajectories: with thin updates only,
   dH grew 0.2 -> 4.6 -> 19 over three trajectories (coarse operator Y
   and deflation basis frozen while the gauge moves). Cure deployed:
   full MG re-setup + coarse re-solve every accepted trajectory
   (MG_SETUP_INTERVAL_PROD=1). The intra-trajectory decoherence that
   remains is the step-scale predictor-corrector's target — measured
   fine-pool residual ~1.5 per tau=1 trajectory sets the error floor.

2. Unequilibrated-state startup: the first trajectory from a mid-therm
   configuration explodes (dH ~ +1e7) while equilibrated starts are
   clean from trajectory 1. Interpretation: the force split is exact for
   any basis, but its stiffness allocation is not — on unequilibrated
   states the coarse deflation basis does not capture the true stiff
   low-mode directions, leaving the stiff force on the outer timescale.
   Not a code bug; nested FGI requires an equilibrated operator. Policy:
   THERM_INTEGRATOR=2 (single FGI) for thermalization, nested for
   probes/productions.

## Nested-FGI inner/outer balance (2026-08-14, kappa=0.14342)

RETRACTED (2026-08-14, same day): the inner-step scan, outer descent,
grid, and 15-trajectory confirmations described in earlier drafts of
this section all omitted --hmc-gauge-infile — hmc_test then starts from
a RANDOM gauge (plaquette 0.12 vs the thermalized 0.598), so the entire
tuning surface was measured in the wrong stiffness regime and is
invalid. The symptom that exposed it: ladder productions (which DO pass
the infile) blew up (dH +62..+172) at settings the fake-state scans
had "confirmed" at var(dH) ~ 5e-3. Probe drivers now pass the infile,
chain the pool, and carry a start-state plaquette guard that aborts the
scan on mismatch. The corrected grid (real state) is the authority for
production settings; do not cite numbers from the retracted scans.

Probe hygiene: the binary's default --hmc-momentum-seed 12345 made
every invocation replay one momentum stream — probe validations were
correlated replicas (44x8 "5/5 clean" was the same noise draw as its
36x8 failure, with more steps). Scripts now seed per invocation from
the launch epoch. Within a run the RNG family seeds once and streams
advance (verified in lib/hmc.cpp): fresh state-independent momenta and
Metropolis coins per trajectory — detailed balance intact; distinct
per-segment seeds across restarts are the standard valid practice.
Latent wart: momentum_seed=0 re-seeds from time() every trajectory
(second resolution) — never pass 0.
