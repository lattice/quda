# P2P island performance investigation

The island filter works, and the examined tuning entries do not alias. There is
no monotonic relationship between island size and dslash performance on this
single-L4 MPS emulation. Larger-volume differences are small even when P2P is
forced. Small-volume timings and autotuner choices are substantially noisier.
The initial one-volume result was therefore insufficient to characterize this.

## Setup and scope

- One NVIDIA L4, eight MPI ranks registered with an isolated CUDA MPS server.
  This is not an eight-physical-GPU or multi-node performance measurement.
- Device query: 58 SMs, 48 MiB L2, 192-bit memory bus. Driver 580.126.09,
  CUDA toolkit 13.3.33, Open MPI 4.1.6. A sampled loaded state was P0,
  SM clock 2040 MHz, memory clock 6251 MHz, 47 C. Clocks were not locked.
- QUDA library checkpoint `5f087c784`, plus diagnostic timing in the test driver.
- Wilson parity dslash, single precision, reconstruction 18; local volumes
  `8^4`, `12^4`, `16^4`, `24^4`, `32^4`; process grid `2 1 1 4`.
- Contiguous original-rank islands of sizes 1, 2, 4, 8; both column/T-fastest
  and row/X-fastest mappings. GDR disabled: non-P2P payloads use host MPI.
- Autotuned matrix: all five volumes, four island sizes, two orderings.
  Controlled matrix: volumes 8, 16, 32 with outer dslash policy fixed to 0;
  host-only baseline, and forced copy-engine / remote-write P2P at sizes 2/4/8.
  Kernel autotuning remained enabled in the controlled performance tests.
- 82 configurations, three sequential measured runs each, shuffled per round:
  246 runs. No competing test jobs were launched during timed runs. Each run
  warmed/tuned before its timed batch. Iteration counts were respectively
  1000, 700, 500, 200, 80 as local side length increased.
- Primary comparison below is the generated `tunecache.tsv` policy `time`
  field, converted from seconds to microseconds. Repeated batch wall times
  are a separate cross-check, not a replacement for that field.

## Tuning cache audit

The final cache contains all **82 expected, distinct Wilson policy entries**:
40 autotuned configurations and 42 controlled configurations. There are no
missing expected configurations or duplicate serialized policy keys.

The audited identity is the complete `(volume, name, aux)` key. Island size and
ordering map to eight distinct `p2p_islands` fingerprints. Existing volume
fields distinguish the five volumes; `p2p` and `pol` distinguish allowed
communication/policy sets. Selected `aux.x` and `aux.y` were also checked.
Forced copy entries all selected P2P policy 1; forced remote-write entries all
selected P2P policy 2. Maximum observed policy `aux` length is 226 bytes,
below the 384-byte capacity: there is no observed truncation.

Local kernel entries may legitimately be shared. The count above refers to
the communication-sensitive **policy** entries, not every row in the cache.
This rules out key degeneracy for the examined configurations; it is not a
claim about all possible jobs or a mathematical collision proof for the hash.

## Autotuned cache timings

Units: microseconds per dslash call. H = no-P2P policy, C = copy-engine,
R = remote-write. The outer policy may differ between cells; these tables
compare the tuner's chosen configuration, not a fixed communication algorithm.

Column ordering (T fastest):

| Local side | Island 1 | Island 2 | Island 4 | Island 8 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 115.5 H | 107.6 R | 110.1 H | 121.7 H |
| 12 | 246.8 H | 239.1 R | 239.6 R | 238.0 R |
| 16 | 965.0 H | 938.0 R | 879.6 R | 920.2 C |
| 24 | 4744.9 H | 4749.3 H | 4717.2 C | 4671.8 R |
| 32 | 14442.2 H | 14301.9 R | 14259.5 R | 14254.4 R |

Row ordering (X fastest):

| Local side | Island 1 | Island 2 | Island 4 | Island 8 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 119.2 H | 113.7 R | 114.8 R | 120.5 H |
| 12 | 251.4 H | 209.1 H | 236.3 R | 231.8 C |
| 16 | 981.0 H | 944.8 R | 895.0 R | 864.3 R |
| 24 | 4744.2 H | 4729.9 R | 4688.6 R | 4732.9 H |
| 32 | 14491.3 H | 14257.2 R | 14172.2 R | 14247.6 C |

At `16^4`, the best cache entry reduces time by 8.8% in column ordering and
11.9% in row ordering relative to island 1. At `24^4` and `32^4`, the best
reductions are only 1.2–2.2%. The `12^4` row/island-2 result must not be
interpreted as a P2P benefit: it selected H, as did island 1, with the same
outer policy. Fresh retunes below show substantial selection/timing variability.

## Controlled policies and repeated execution

Settings: `QUDA_ENABLE_DSLASH_POLICY=0`, `QUDA_ENABLE_GDR=0`, P2P flag 0
for host, 5 for copy-engine, 6 for remote-write. Flags 5/6 force P2P only on
eligible links: cross-island faces still use host MPI. The earlier documentation
warning against these flags for mixed islands was incorrect and has been fixed.

Raw controlled **cache times**, in microseconds:

| Side / order | Host | C2 | R2 | C4 | R4 | C8 | R8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 / col | 135.2 | 125.5 | 128.0 | 129.9 | 277.0* | 147.6 | 137.2 |
| 8 / row | 127.5 | 248.2* | 122.8 | 134.9 | 365.2* | 154.2 | 136.2 |
| 16 / col | 982.7 | 981.0 | 965.0 | 993.6 | 953.0 | 994.3 | 961.5 |
| 16 / row | 978.9 | 985.4 | 972.1 | 1001.8 | 975.9 | 976.9 | 932.9 |
| 32 / col | 14498.8 | 14488.6 | 14553.1 | 14488.6 | 14413.5 | 14549.7 | 14411.1 |
| 32 / row | 14565.7 | 14488.9 | 14407.7 | 14532.3 | 14302.9 | 14525.1 | 14549.7 |

C2 means copy-engine with island size 2; R4 means remote-write with size 4.
Starred entries were investigated with three independent fresh-cache retunes:

| Case | Original cache | Fresh-cache range | Fresh-run wall range |
| --- | ---: | ---: | ---: |
| 8 / col / R4 | 277.0 | 119.7–129.2 | 122.1–123.0 |
| 8 / row / R4 | 365.2 | 123.8–130.7 | 123.4–124.3 |
| 8 / row / C2 | 248.2 | 139.3–143.0 | 130.4–131.6 |

These retained the same key and forced policy. Their original large times
were not reproducible performance characteristics of those islands. Across
21 fresh-cache follow-up runs, outer policy and/or P2P choices also changed
for some unrestricted configurations. For example, `12^4`, row, island 2
originally chose H, but fresh runs chose R/C/R. This is an autotuning
repeatability concern under MPS, separate from cache-key degeneracy.

The tuner uses two candidate iterations, then a minimum 1 ms / three
iterations for finalist timing (`Tunable` defaults). Its cached time is a
short tuning measurement, not the mean of a long production run.

Repeated batch **median wall times** for selected fixed-policy comparisons:

| Side / order | Host | Copy, island 8 | Remote, island 8 |
| --- | ---: | ---: | ---: |
| 8 / col | 136.5 | 165.9 | 132.2 |
| 8 / row | 133.1 | 155.6 | 145.2 |
| 16 / col | 989.0 | 1006.9 | 963.9 |
| 16 / row | 1029.9 | 999.5 | 964.5 |
| 32 / col | 14505.3 | 14663.7 | 14419.2 |
| 32 / row | 14507.0 | 14685.9 | 14446.2 |

The full CSV contains all repeats and configurations. At small volumes,
ranges can be wide; three samples are not enough to establish statistical
significance for differences of only a few percent. The small-volume
copy-engine penalty is nevertheless not explained by a silent host fallback:
the path traces below verify the forced path.

## Timing, execution-path, and correctness checks

- Whole-batch host wall time ends after synchronization of the CUDA stop
  event, then is reduced to the slowest rank. This is outside the timed loop.
  Across 246 runs, it differed from slowest-rank CUDA-event time by at most
  0.178%. Rank-0 event time differed from slowest-rank event time by at most
  0.101%. No evidence of omitted asynchronous work or a misleading rank 0.
- Separate interposition traces counted spinor send-path invocations and the
  actual QUDA IPC-copy wrapper. No tracing was used during performance runs.
  Forced modes produced exactly the expected P2P send fractions:

| Island size | Column | Row |
| --- | ---: | ---: |
| 1 | 0% | 0% |
| 2 | 25% | 50% |
| 4 | 50% | 75% |
| 8 | 100% | 100% |

For 10 calls plus one warm-up, all eight ranks made 352 total face sends.
Column/size-2/copy produced 264 host sends and 88 copy sends, with 88 actual
IPC-copy calls transferring 1,081,344 bytes. Size-8/copy produced 352 copy
sends/copies and zero host payload sends. Remote mode produced the expected
remote-send counts and no copy calls. Two additional traces loading the
shared autotuning cache confirmed H for `8^4`/col/island-8 and R for
`16^4`/row/island-8, as recorded in the cache.
- All 36 CPU-reference checks passed with **forced** copy and remote paths,
  island sizes 2/4/8, both orderings, volumes `8^4`, `16^4`, `32^4`.
  Maximum L2 relative deviation: `1.715217e-8`.
- Correction to the earlier 448-case checkpoint: its topology checks were
  valid, but numerical runs allowed P2P flag 3 with tuning disabled. That can
  choose the no-P2P default and did not establish forced-P2P numerical
  coverage. The 36 checks above supply that coverage for this process grid.
  The matrix runner now defaults to flag 5 and accepts `--p2p 6` explicitly.

## Why P2P does not produce a large monotonic speedup here

Measured one-rank baselines used the same aggregate lattice dimensions
`2L x L x L x 4L`, with no partitioned faces. They are a combined
process/decomposition/halo overhead comparison, not an isolated MPI timer:

| Eight-rank local side | One-rank median | Eight-rank autotuned medians |
| --- | ---: | ---: |
| 8 | 9.974 us | 113.3–139.6 us |
| 16 | 782.898 us | 956.1–1003.0 us |
| 32 | 12467.688 us | 14282.0–14615.9 us |

The small case is dominated by work that disappears or changes when eight
processes and partitioned halos become one larger local problem. At larger
volumes, bulk work consumes much more time. The surface/volume ratio also
falls as L grows. The source schedules communication alongside interior
dslash work, so payload savings can be hidden by interior execution.
These observations support an overhead/overlap explanation, but do not
separately measure DRAM saturation or assign all baseline differences to MPI.

Importantly, P2P does **not** eliminate MPI/control overhead. In
`ColorSpinorField::sendStart`, both copy and remote-write paths still record
an IPC event and start `mh_send_p2p`, an MPI synchronization message. They
avoid the host payload route, not the per-face signaling. All eight MPS
clients also share one GPU's resources. The copy path here reaches a
same-device `cudaMemcpyAsync(DeviceToDevice)` through imported IPC memory;
there is no physical NVLink/PCIe peer transfer between GPUs, nor a network
between the emulated islands. These results cannot predict an eight-GPU
multi-node speedup.

## Conclusions and next steps

1. The island mechanism and tuning-key separation pass the examined checks.
2. There is no general “larger islands are faster” result for this emulation.
   At large volumes, fixed-policy payload-path differences are small; at
   small volumes, signaling, process scheduling, and policy choice matter.
3. Autotuning repeatability under MPS deserves care. Use fresh-cache repeats
   and longer policy trials before optimizing based on a single cached time.
   CPU affinity and controlled clocks would be useful follow-up controls;
   they were not imposed in this study.
4. Keep the islands for the Laplace-to-MG correctness reproducer. Physical
   multi-GPU/multi-node measurements are needed for transport conclusions.
   No Laplace-to-MG failure was exercised or explained by these dslash tests.

## Preserved evidence

The workspace directory
`/home/mclark/codex/agent-sandbox/quda-islands-study-results` contains:

- `cache/tunecache.tsv`: original 82-policy-entry comparison cache.
- `cache-policy-times.csv`, `cache-audit.json`: decoded times, selections,
  full auxiliary strings, and coverage audit.
- `measurements.csv` / `.json`: 246 repeated batch measurements.
- `validation.json`: 14 forced path traces and 36 forced numerical checks;
  `logs/auto-trace-*.log` contains the two cached-auto path checks.
- `retune.json`, `retune-caches/`: all 21 fresh-cache checks and original TSVs.
- `baseline.json`, `one-rank-cache/`: nine one-rank runs and tuning cache.
- `logs/` and `scripts/`: full run output and the experimental harnesses.

The copied harnesses preserve the original temporary-directory/build paths;
set their `root` and build/binary paths before reproducing elsewhere.
