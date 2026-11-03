# Diagnostic P2P islands

`QUDA_TEST_P2P_ISLANDS=0,0,1,1` restricts QUDA P2P communication to
neighbors with equal island IDs. The comma-separated list contains one
nonnegative integer per rank in the original QUDA communicator (including
when the application supplies a communicator). All ranks must supply the same
list. Empty values, malformed IDs, and incorrect lengths are errors.

Membership is inherited across split-grid communicator changes, independent
of the new local rank. Unset means normal P2P selection. The mask never enables
hardware access that would otherwise be unavailable. MPI and QMP are supported.

For a single-GPU emulation, enable `QUDA_ENABLE_MPS=1`, use an MPS server,
and set `QUDA_ENABLE_GDR=0` for host-buffer MPI across islands. MPS alone
does not define island membership. Do not use P2P-only policy settings
(`QUDA_ENABLE_P2P=5,6,7`) for this mixed-path experiment.

Build `p2p_islands_test` in an MPI build. On one GPU:

```sh
export QUDA_ENABLE_MPS=1 QUDA_ENABLE_GDR=0 QUDA_ENABLE_P2P=3
export QUDA_TEST_P2P_ISLANDS=0,0,1,1
mpirun -np 4 ./tests/p2p_islands_test
mpirun -np 4 ./tests/p2p_islands_test t-fast
```

This test checks every neighbor's P2P eligibility on the original grid and
after three T-split/restore cycles, with two coordinate-to-rank orderings.
It assumes all processes share one GPU and P2P is enabled. Use the existing
dslash and inversion tests to validate actual halo exchanges and residuals.

## Checkpoint validation

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
