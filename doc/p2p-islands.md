# Diagnostic P2P islands

`QUDA_TEST_P2P_ISLANDS=0,0,1,1` restricts QUDA P2P communication to
neighbors with equal island IDs. The comma-separated list contains one
nonnegative integer per rank in the original QUDA communicator (including
when the application supplies a communicator). All ranks must supply the same
list. Empty values, malformed IDs, and incorrect lengths are errors.

Membership is inherited across split-grid communicator changes, independent
of the new local rank. Unset means normal P2P selection. The mask never enables
hardware access that would otherwise be unavailable. MPI and QMP are supported.

With islands enabled, the cached communication configuration string contains
`,p2p_islands=<64-bit fingerprint>`. The fingerprint includes the complete
original assignment, current communicator membership, dimensions, and
coordinate-to-rank mapping. Dslash policy tuning already includes the
communication configuration in its auxiliary key, so different island
layouts/orderings do not share tuning entries. Each communicator computes
this string only once; returning to a cached communicator restores its key.

For a single-GPU emulation, enable `QUDA_ENABLE_MPS=1`, use an MPS server,
and set `QUDA_ENABLE_GDR=0` for host-buffer MPI across islands. MPS alone
does not define island membership. `QUDA_ENABLE_P2P` accepts only `0`
(off), `1` (copy engine), `2` (remote write), or `3` (both). These permit
features; the tuner may still choose the non-P2P policy.
`QUDA_ENABLE_P2P_FORCE=1` excludes that candidate when eligible links exist;
it defaults to `0` and accepts only `0` or `1`. All other values, including
negative and legacy 5/6/7 settings, are errors. `QUDA_ENABLE_P2P=0` with
force enabled is also an error.

For controlled copy-engine testing use `QUDA_ENABLE_P2P=1` with
`QUDA_ENABLE_P2P_FORCE=1`; for remote writes use `2` with force `1`.
Islands and hardware eligibility still restrict which links use P2P. With
no eligible peers, ordinary MPI remains available even with force set.
GDR no longer implicitly forces P2P. The cached configuration includes
`,p2p_force=0|1`, separating tuning entries for the two policy sets.

Build `p2p_islands_test` in an MPI build. On one GPU:

```sh
export QUDA_ENABLE_MPS=1 QUDA_ENABLE_GDR=0 QUDA_ENABLE_P2P=3
export QUDA_TEST_P2P_ISLANDS=0,0,1,1
mpirun -np 4 ./tests/p2p_islands_test --rank-order col
mpirun -np 4 ./tests/p2p_islands_test --rank-order row
```

This test checks every neighbor's P2P eligibility on the original grid and
after three split/restore cycles. It accepts `--gridsize X Y Z T`,
`--grid-partition X Y Z T`, and the existing QUDA ordering convention:
`--rank-order col` means T-fastest; `--rank-order row` means X-fastest.
The default grid is `2 1 1 N/2`, split `1 1 1 2`, ordering `col`.
It assumes all processes share one GPU and P2P is enabled. Use the existing
dslash and inversion tests to validate actual halo exchanges and residuals.

For the eight-rank regression matrix, build `p2p_islands_test` and
`dslash_test`, start MPS, and run:

```sh
python3 tests/p2p_islands_matrix.py /path/to/build /path/to/results
```

The runner emulates eight ranks on one GPU, with contiguous islands of
sizes 1, 2, 4, and 8. It covers both rank orderings, pure X/Y/Z/T process
grids, and six mixed grids, including splitting multiple dimensions at
once and splitting into single-rank subgrids. Each case saves its full log.
Two independent MPI worlds run at a time (16 total MPS clients).
The runner defaults to `--p2p 1 --p2p-force 1` to force copy-engine P2P
on eligible faces; use `--p2p 2` to force remote writes. Use
`--p2p-force 0` to permit the non-P2P policy. With tuning disabled this
can choose the non-P2P default even when P2P methods are permitted.

## Checkpoint validation

The eight-rank matrix passed all 448 cases on the L4 under MPS:
256 topology checks (each with three split/restore cycles) and 192
CPU-reference Wilson dslash checks. Each island size (1, 2, 4, 8) passed
112 cases, covering both orderings. The largest dslash relative L2 deviation
was `3.540053e-8`.

Process grids were `8 1 1 1`, `1 8 1 1`, `1 1 8 1`, `1 1 1 8`,
`4 1 1 2`, `2 1 1 4`, `1 4 2 1`, `1 2 1 4`, `2 2 2 1`, and
`2 1 2 2`. Pure-axis grids used split factors 2, 4, and 8. Mixed grids
used factor 2 along each partitioned axis separately, plus a full split
along all partitioned axes together. The pure-axis numerical checks used
factor 2; all mixed-grid splits had numerical checks.

Those original numerical runs used P2P flag 3 with tuning disabled, so they
could select the no-P2P default. They validate results with the island setup
but do not prove forced-P2P halo correctness. The subsequent
[performance investigation](p2p-islands-performance.md) adds runtime path
traces and 36 passing forced-copy/remote CPU-reference checks through `32^4`.

Validated on one NVIDIA L4 with MPS, Open MPI 4.1.6, single precision,
uncompressed gauge fields, and GPU-direct RDMA disabled:

- All eight topology combinations passed: `0,0,1,1`, `0,1,0,1`,
  `0,0,0,0`, and `0,1,2,3`, each with both rank orderings and three
  split/restore cycles.
- Wilson dslash matched the CPU reference on a `2 1 1 2` process grid,
  local lattice `4 4 4 4`, for both `--rank-order col` and `row`,
  with and without `--grid-partition 1 1 1 2`. Relative L2 deviations
  were below `3.5e-8` with paired islands.
- A mixed-island CG inversion converged in 13 iterations with true residual
  `7.071e-6` against a requested `1e-5` tolerance.
- Incorrect list lengths, trailing commas, negative/overflowing IDs, and
  inconsistent lists across ranks were rejected without timeouts.

QMP support is implemented but was not built or exercised in this environment.
These checks validate the island machinery; they do not reproduce the
Laplace-to-MG failure, whose pipeline test is a separate next step.

## Dslash performance sample

This initial one-volume sample is superseded by the
[multi-volume cache audit and controlled-policy investigation](p2p-islands-performance.md).

Eight sequentially benchmarked MPS ranks on the L4, local volume `16^4`,
process grid `2 1 1 4`, Wilson dslash, single precision, reconstruction 18,
`QUDA_ENABLE_P2P=3`, and GDR disabled. Each configuration had one initial
cache-population run and three measured runs of 1,000 calls, each with
the test's own warm-up. Configuration order was shuffled per round.
All configurations shared a tuning cache, with eight distinct island keys.

Median reported microseconds per dslash call (min–max in parentheses):

| Island size | Column / T-fastest | Row / X-fastest |
| --- | --- | --- |
| 1 | 984.208 (982.516–997.219) | 972.105 (971.443–987.397) |
| 2 | 969.844 (968.790–976.857) | 971.333 (970.216–985.317) |
| 4 | 993.195 (986.366–1017.496) | 983.684 (983.667–1008.643) |
| 8 | 958.544 (957.795–965.449) | 972.109 (971.198–990.689) |

The largest median improvement over size 1 was 2.7% (size 8, column
ordering). There was no consistent monotonic trend across both orderings.
Autotuning selected the non-P2P policy for some configurations, including
size 4 in both orderings and size 8 in row ordering. These numbers describe
automatically selected policies on a shared physical GPU, not inter-GPU or
inter-node bandwidth.

After adding the tuning fingerprint, all eight size/ordering topology
regressions passed, including rank agreement on the config string and
restoration of the original string after each split/restore cycle.

## P2P cleanup validation boundary

The mask/force cleanup is covered by `tests/p2p_force_test.py`: 65 checks
per run, with four or eight MPS ranks, and 16 distinct policy cache keys.
GDR-enabled checks cover startup configuration, not device-buffer MPI traffic.
These are processes sharing one physical L4, not eight physical GPUs.

The original cleanup checkpoint did not fix communicator-transition state.
The subsequent split/global correction resets halo-buffer phase on switching
communicators and refreshes fine-dslash candidates when effective P2P
capabilities change. See [the investigation and regression tests](p2p-split-grid.md).
The earlier unvalidated candidate remains preserved on branch
`investigate/p2p-policy-refresh`. Colin's Laplace/MG sequence has not yet
been tested; these dslash regressions do not establish MG correctness.
