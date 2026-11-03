# Split/global P2P communication state

## Cause of the dslash numerical failures

Fine-grid halo exchange double-buffers its data and IPC events using the
process-local static `LatticeField::bufferIndex`. Each dslash policy invocation
toggles that index, including invocations made by autotuning. Independent
subgrids can perform different numbers of invocations and finish on different
indices. `push_communicator` destroyed the old buffers and IPC state but left
the index unchanged.

After restoring the global communicator, a P2P sender addresses the remote
buffer with its own index. Its peer reads the buffer with the peer's index.
If the indices differ, the data transfer and event synchronization refer to
the wrong double-buffer slot. Host-staged MPI does not have that addressing
assumption: receive requests target each receiver's local selected buffer.
This explains why the matched P2P-off control passed.

Instrumentation observed mismatched indices immediately after returning from
split-grid calls. Repeated runs with capability refresh alone both passed and
failed, depending on the phases. Manually aligning the indices before the
global dslash passed six repeated runs of the three previously failing
configurations. These included unforced copy-engine, forced remote-write,
and eight-rank heterogeneous-island cases.

The fix resets the index to zero in `push_communicator`, after the old ghost
buffers, IPC state and cached temporaries have been torn down. This happens
on every switch, including switches that start with no allocated buffers.
It adds no synchronization to the steady-state dslash path.

## A separate policy-candidate problem

The fine-dslash policy candidate vectors were initialized once per process.
After independent subgrids with different P2P availability rejoin, different
ranks can retain different sets of candidates. This can violate forcing,
retain unavailable methods, or make ranks tune incompatible communication
paths. A heterogeneous pre-fix repeat produced an MPI message-truncation
error in addition to the numerical failures.

The policy initializer now rebuilds its candidates when the effective P2P
mask or force setting changes, clearing both vectors and reconstructing the
policy string. The buffer-phase correction and candidate refresh address
different state, and both are needed for the broader transition coverage.

## Regression tests

`DslashTest.communicator_halo_phase_restore` deliberately assigns different
phases to subgrids, restores the global communicator, checks alignment, and
compares the first global Wilson dslash to the CPU reference, after warming
the tuning cache before switching. Repeated identical calls could otherwise
hide the wrong-slot read once both buffers contain identical halo data.
The test does not swap or
reload gauge fields, avoiding the resident-gauge lifetime ambiguity of an
application-level split pipeline.

Removing only the phase reset reproduced corruption in both forced copy-engine
and forced remote-write tests even with autotuning disabled: the printed norm
deviation was `1.980009e-2`, with maximum component deviation `3.279129`.
The matched P2P-off calculation remained numerically correct (`1.699759e-8`
printed norm deviation); its deliberately stricter phase-alignment assertion
still failed. Thus an overall test failure in that negative-control log is
expected and does not mean the MPI calculation failed.

`DslashTest.split_grid_policy_restore` runs two actual split/global dslash
cycles, reconstructing global spinors and the operator after each split.
It checks CPU agreement, finite results and the effective tuning policies.

`p2p_islands_test` also deliberately diverges the indices and checks alignment
on both entering and leaving split communicators, including empty-buffer
switches.

With CUDA MPS already running, use the following commands.

For the minimal deterministic test (no actual split gauge operation):

```sh
QUDA_ENABLE_MPS=1 QUDA_ENABLE_GDR=0 QUDA_ENABLE_P2P=1 QUDA_ENABLE_P2P_FORCE=1 \
QUDA_ENABLE_TUNING=0 QUDA_ENABLE_DSLASH_POLICY=0 QUDA_TEST_P2P_ISLANDS=0,0,1,1 \
mpirun --oversubscribe -np 4 /path/to/build/tests/dslash_test \
  --dim 4 4 4 4 --gridsize 2 1 1 2 --rank-order col --prec single --recon 18 \
  --nsrc 1 --gtest_filter=DslashTest.communicator_halo_phase_restore
```

For the actual split/global transition matrix, use fresh results directories:

```sh
python3 tests/p2p_split_grid_test.py /path/to/build /path/to/results4 --ranks 4
python3 tests/p2p_split_grid_test.py /path/to/build /path/to/results8 --ranks 8
```

Each matrix covers row/column ordering, T/X splits, all valid mask/force
combinations, uniform/heterogeneous islands, explicit all-peer/no-peer
controls, and cold/warm disk tuning caches. `--local-dim` and `--sources`
vary the local lattice size and restored-global RHS count. Numerical tests
use GDR disabled and explicitly select outer dslash policy 0.

The available hardware is one L4 shared by MPS ranks, not four/eight physical
GPUs. These tests do not establish HIP/QMP, CUDA-aware MPI, or MG correctness.
The mechanism matches the transition/P2P/mapping sensitivity reported for
Colin's application, but his Laplace/MG sequence remains to be tested.

## Completed transition matrix

Both the four-rank and eight-rank matrices passed all 128 invocations each
(256 total), each invocation containing two split/global cycles. Each of
64 configurations per rank count was run once with a cold disk cache and
again with the resulting cache. Local volume was `4^4`, single precision,
reconstruction 18, with one restored-global RHS.

Six additional invocations of the previously failing configurations passed
at local volume `8^4` with two restored-global RHS. These larger-volume
checks used in-memory tuning without a persistent cache.

The deterministic first-call phase regression passed all 32 combinations
of four/eight ranks, row/column order, P2P off/copy/remote/both, and tuning
off/on after restoring the reset. Ten actual split/global tests also passed
with the default outer dslash-policy search (the policy-0 restriction unset).
The 65-check mask/force suite passed again at both rank counts, including
16 distinct tuning-policy cache identities per run. These make 434 positive
test invocations/checks in total; invalid-input checks are included in that
count and are not numerical tests. The three intentional without-reset
controls are recorded separately, not counted as passing invocations.
