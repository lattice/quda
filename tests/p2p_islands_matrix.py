"""Single-GPU, eight-MPS-rank island regression matrix. Start MPS beforehand."""
import argparse
import itertools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("build", type=Path)
parser.add_argument("logs", type=Path)
args = parser.parse_args()
args.logs.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, QUDA_ENABLE_MPS="1", QUDA_ENABLE_GDR="0",
           QUDA_ENABLE_P2P="3", QUDA_ENABLE_TUNING="0")
env.pop("DISPLAY", None)
cases = []
# Each physical lattice dimension is tested as the only partitioned axis.
for axis in range(4):
    grid = [1] * 4
    grid[axis] = 8
    for factor in (2, 4, 8):
        split = [1] * 4
        split[axis] = factor
        cases.append((grid.copy(), split))
# Mixed grids expose ordering-dependent neighbors and mixed P2P/MPI faces.
for grid in ([4, 1, 1, 2], [2, 1, 1, 4], [1, 4, 2, 1],
             [1, 2, 1, 4], [2, 2, 2, 1], [2, 1, 2, 2]):
    for axis, extent in enumerate(grid):
        if extent > 1:
            split = [1] * 4
            split[axis] = 2
            cases.append((grid, split))
    cases.append((grid, grid))

jobs = []
for island_size, order, (grid, split) in itertools.product((1, 2, 4, 8), ("col", "row"), cases):
    env["QUDA_TEST_P2P_ISLANDS"] = ",".join(str(rank // island_size) for rank in range(8))
    common = ["--rank-order", order, "--gridsize", *map(str, grid),
              "--grid-partition", *map(str, split)]
    programs = ["p2p_islands_test"]
    # Numerically exercise each mixed-grid split and each pure-axis half split.
    if sum(x > 1 for x in grid) > 1 or max(split) == 2:
        programs.append("dslash_test")
    for program in programs:
        command = ["mpirun", "--oversubscribe", "-np", "8",
                   str(args.build.resolve() / "tests" / program), *common]
        if program == "dslash_test":
            command += ["--dim", "4", "4", "4", "4", "--nsrc", str(math.prod(split)),
                        "--prec", "single", "--recon", "18", "--niter", "2",
                        "--gtest_filter=DslashTest.verify"]
        name = f"{program}-island{island_size}-{order}-grid{''.join(map(str, grid))}-split{''.join(map(str, split))}"
        marker = "PASS (0 mismatches)" if program == "p2p_islands_test" else "[  PASSED  ] 1 test."
        jobs.append((command, env.copy(), name, marker))

def run(job):
    command, environment, name, marker = job
    result = subprocess.run(command, env=environment, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=90)
    (args.logs / (name + ".log")).write_text(result.stdout)
    if result.returncode or marker not in result.stdout:
        raise RuntimeError(f"FAIL {name}: exit={result.returncode}; see logs")
    return name

# Two independent eight-rank MPI worlds keep total MPS clients bounded at 16.
with ThreadPoolExecutor(max_workers=2) as executor:
    for count, name in enumerate(executor.map(run, jobs), 1):
        print(f"PASS {count}: {name}", flush=True)
print(f"All {len(jobs)} cases passed", flush=True)
