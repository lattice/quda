# Stream-gated P2P regression

These tests guard the **stream-gated P2P flag-buffer lifecycle invariant**: the
stream-gated transport writes/waits on per-(FieldKind,dim,dir) signal slots; if the
flag buffers are ever not created while stream-gating is the resolved transport, the
invariant in `lib/targets/cuda/comm_target.cpp` aborts (`errorQuda`) rather than
silently skipping the halo signal. The test therefore fails **deterministically** if
flag-buffer creation is suppressed — it does not depend on a numerical race surfacing.

CTest registration (in `tests/CMakeLists.txt`) is labelled:

    multi_gpu;stream_gated;regression

## Required configuration

Build:
- `QUDA_NVSHMEM=ON`, `QUDA_MPI=ON`, `QUDA_DIRAC_STAGGERED=ON`,
  `QUDA_CTEST_SEP_DSLASH_POLICIES=ON`.

Runtime:
- `QUDA_ENABLE_NVSHMEM=0` (NVSHMEM compiled in but transport disabled — the exact
  config the lifecycle bug lived in), `QUDA_P2P_TRANSPORT=stream_gated`,
  `QUDA_ENABLE_P2P=5`, `QUDA_ENABLE_DSLASH_POLICY=0`.

Non-vacuous guarantee:
- The executable takes `--require-p2p true`; it aborts if no P2P neighbour is
  enabled (uses the global, allreduce-backed `comm_peer2peer_enabled_global()` so all
  ranks decide identically — no split-brain). This needs **>1 rank on P2P-connected
  GPUs**; a single rank has no peer and the stream-gated primitives never run.

## IMPORTANT: set the grid before CMake configure

`QUDA_TEST_GRID_SIZE` and `QUDA_CTEST_LAUNCH` are consumed at **configure** time.
The CTests above are only registered when the configure-time rank count
(`QUDA_TEST_NUM_PROCS`, derived from `QUDA_TEST_GRID_SIZE`) is `> 1`, and the
split-grid case derives its explicit `--grid-partition` from it. So you must export
the grid **before** running `cmake`:

    export QUDA_TEST_GRID_SIZE="1 1 2 2"   # 4 ranks; product > 1
    cmake -S <src> -B <build> \
      -DQUDA_TARGET_TYPE=CUDA -DQUDA_GPU_ARCH=sm_90 \
      -DQUDA_MPI=ON -DQUDA_NVSHMEM=ON \
      -DQUDA_DIRAC_STAGGERED=ON -DQUDA_CTEST_SEP_DSLASH_POLICIES=ON
    cmake --build <build> -j

If `QUDA_TEST_GRID_SIZE` is unset at configure, the regression is skipped (with a
CMake STATUS message) rather than registered as a vacuous single-rank test.

## Future CI (GitLab) — currently future work

`ci/pipeline.yml` builds on the public `docker.io/nvidia/cuda` base with MPI only
(no NVSHMEM). Running this regression in CI needs an NVSHMEM-capable image (and, for
the genuine MNNVL teardown path, an MNNVL fabric runner). We deliberately do **not**
add such a build to `pipeline.yml` — it would require a non-public image. When an
NVSHMEM-capable image/runner is available, enforce it with:

    ctest --test-dir <build> -L stream_gated --output-on-failure

GitLab enforcement therefore remains future work; its absence is not a blocker as
long as the equivalent direct runs below pass.

## Current validation: direct `srun` (no mpirun / no CTest launcher)

On a queued machine where jobs launch with Slurm `srun` (not `mpirun`), the parallel
CTest launcher does not apply. Run the executable directly under `srun` with the same
runtime environment. Use the site's required account/partition/container options.

Normal (single-grid, P2P halo exchange):

    srun <site opts> --mpi=pmix -N1 -n4 env \
      QUDA_ENABLE_NVSHMEM=0 QUDA_P2P_TRANSPORT=stream_gated \
      QUDA_ENABLE_P2P=5 QUDA_ENABLE_DSLASH_POLICY=0 \
      <build>/tests/staggered_dslash_test \
        --dslash-type staggered --test MatPC --require-p2p true \
        --dim 24 24 24 24 --gridsize 1 1 2 2 --prec single --recon 9 --verify true

Split-grid (eventually drives `push_communicator()` -> flag-buffer destroy/recreate);
pass an explicit `--grid-partition` whose product is > 1 and divides the rank count:

    srun <site opts> --mpi=pmix -N1 -n4 env \
      QUDA_ENABLE_NVSHMEM=0 QUDA_P2P_TRANSPORT=stream_gated \
      QUDA_ENABLE_P2P=5 QUDA_ENABLE_DSLASH_POLICY=0 \
      <build>/tests/staggered_dslash_test \
        --dslash-type staggered --test MatPC --require-p2p true \
        --grid-partition 1 1 2 2 \
        --dim 24 24 24 24 --gridsize 1 1 2 2 --prec single --recon 9 --verify true

(Adjust topology/args to the tested machine; do not copy blindly.) Confirm from the
log that P2P is enabled (the fabric/IPC "peer-to-peer ... enabled" lines), that
verification passes, and — for split-grid — that communicator push/pop occurs and
teardown leaves no imported VMM mappings (the registry-empty assertion does not fire).

For a global `1 1 2 2` grid split by `1 1 2 2`, each resulting communicator is
`1 1 1 1`: operations inside it are local and do not create a stream-gated flag
buffer. Communication is still used while redistributing fields on the original
communicator and after returning to it. Initial fat/long gauge loading also occurs
on the original communicator, before the first communicator push.

## Hang isolation (WIP only)

`QUDA_P2P_DIAGNOSTIC_SYNC` adds a synchronization point to the gauge P2P send path.
It is unset by default and has no production effect. Run the reproducer in three
modes, preferably interleaved over many fresh processes:

    # Existing asynchronous path
    env -u QUDA_P2P_DIAGNOSTIC_SYNC <command above>

    # Synchronize after the peer halo copy, before writing the remote signal
    QUDA_P2P_DIAGNOSTIC_SYNC=copy <command above>

    # Synchronize after both the peer halo copy and remote signal write
    QUDA_P2P_DIAGNOSTIC_SYNC=signal <command above>

Interpretation:
- `copy` hangs at `qudaStreamSynchronize`: the data mapping/copy is stalled.
- `copy` passes but `signal` hangs: the remote flag mapping/write is stalled.
- both diagnostic modes pass while the normal mode hangs: investigate the receive
  wait and CUDA scheduling/dependency chain rather than VMM import alone.

Because either synchronization can perturb scheduling, a passing diagnostic mode is
not proof by itself. A hang at the inserted synchronization is the decisive result.
