"""CPU-reference split/global dslash regressions; requires a running CUDA MPS server."""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("logs", type=Path)
    parser.add_argument("--ranks", type=int, choices=(4, 8), default=4)
    parser.add_argument("--local-dim", type=int, default=4)
    parser.add_argument("--sources", type=int, default=1, help="RHS count in the restored global dslash")
    parser.add_argument("--repeats", type=int, default=2, help="First run cold, subsequent runs reuse its cache")
    args = parser.parse_args()
    if args.local_dim < 4 or args.local_dim % 2 or args.sources < 1 or args.repeats < 1:
        parser.error("require even local dimension >=4, positive sources and repeats")
    args.logs.mkdir(parents=True, exist_ok=True)
    base = dict(os.environ, QUDA_ENABLE_MPS="1", QUDA_ENABLE_GDR="0", QUDA_ENABLE_TUNING="1",
                QUDA_ENABLE_DSLASH_POLICY="0")
    for name in ("QUDA_RESOURCE_PATH", "QUDA_TEST_P2P_ISLANDS", "QUDA_ENABLE_P2P", "QUDA_ENABLE_P2P_FORCE"):
        base.pop(name, None)
    results = []
    configurations = [(mask, force, layout) for mask, force in
                      ((0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1))
                      for layout in ("uniform", "heterogeneous")]
    # Explicit all-P2P and no-peer fallback controls in both orderings/splits.
    configurations += [(3, 1, layout) for layout in ("all", "isolated")]
    for order, split_dim, (mask, force, layout) in itertools.product(("col", "row"), ("T", "X"), configurations):
        if layout == "uniform":
            islands = [r // (args.ranks // 2) if order == "col" else r % 2 for r in range(args.ranks)]
        elif layout == "heterogeneous":
            islands = [0] * (args.ranks - 1) + [1]
        elif layout == "all":
            islands = [0] * args.ranks
        else:
            islands = list(range(args.ranks))
        split = [1, 1, 1, args.ranks // 2] if split_dim == "T" else [2, 1, 1, 1]
        name = f"{order}-split{split_dim}-mask{mask}-force{force}-{layout}"
        cache = args.logs / name / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        if (cache / "tunecache.tsv").exists():
            raise RuntimeError(f"Use a fresh results directory: {cache} already has a cache")
        env = dict(base, QUDA_ENABLE_P2P=str(mask), QUDA_ENABLE_P2P_FORCE=str(force),
                   QUDA_TEST_P2P_ISLANDS=",".join(map(str, islands)), QUDA_RESOURCE_PATH=str(cache.resolve()))
        command = ["mpirun", "--oversubscribe", "-np", str(args.ranks),
                   str(args.build.resolve() / "tests" / "dslash_test"),
                   "--dim", *([str(args.local_dim)] * 4), "--gridsize", "2", "1", "1", str(args.ranks // 2),
                   "--grid-partition", *map(str, split), "--rank-order", order,
                   "--prec", "single", "--recon", "18", "--nsrc", str(args.sources),
                   "--gtest_filter=DslashTest.split_grid_policy_restore"]
        for repeat in range(args.repeats):
            result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=90)
            log = cache.parent / f"run{repeat}.log"
            log.write_text(result.stdout)
            passed = (result.returncode == 0 and "[  PASSED  ] 1 test." in result.stdout
                      and "[  SKIPPED ]" not in result.stdout)
            results.append(dict(case=name, repeat=repeat, passed=passed, returncode=result.returncode,
                                command=command, settings={key: env[key] for key in base.keys() | env.keys()
                                                          if key.startswith("QUDA_")}, log=str(log)))
            print(f"{'PASS' if passed else 'FAIL'} {len(results)}: {name} run{repeat}", flush=True)
    (args.logs / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    failures = sum(not result["passed"] for result in results)
    print(f"COMPLETE: {len(results)} split/global checks, {failures} failures", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
