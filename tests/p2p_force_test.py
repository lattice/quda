"""P2P mask/force regression checks. Start a CUDA MPS server before running."""
import argparse
import csv
import itertools
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("build", type=Path)
parser.add_argument("logs", type=Path)
parser.add_argument("--ranks", type=int, choices=(4, 8), default=4)
args = parser.parse_args()
args.logs.mkdir(parents=True, exist_ok=True)
base = dict(os.environ, QUDA_ENABLE_MPS="1", QUDA_ENABLE_GDR="0")
for key in ("QUDA_ENABLE_P2P", "QUDA_ENABLE_P2P_FORCE", "QUDA_ENABLE_TUNING", "QUDA_RESOURCE_PATH"):
    base.pop(key, None)
base["QUDA_TEST_P2P_ISLANDS"] = ",".join(str(rank // 2) for rank in range(args.ranks))
count = 0

def run(name, program, settings, arguments=(), error=None):
    global count
    env = dict(base, **settings)
    command = ["mpirun", "--oversubscribe", "-np", str(args.ranks),
               str(args.build.resolve() / "tests" / program), *arguments]
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=90)
    (args.logs / (name + ".log")).write_text(result.stdout)
    if error:
        if not result.returncode or error not in result.stdout:
            raise RuntimeError(name + ": expected validation error")
    else:
        marker = "PASS (0 mismatches)" if program == "p2p_islands_test" else "[  PASSED  ] 1 test."
        if result.returncode or marker not in result.stdout or "[  SKIPPED ]" in result.stdout:
            raise RuntimeError(name + ": see log")
    count += 1
    print(f"PASS {count}: {name}", flush=True)
    return result.stdout

for index, value in enumerate(("", "-1", "-2", "-3", "-2147483648", "4", "5", "6", "7", "8",
                               "2147483648", "text", "1x", "01", " 1", "1 ")):
    run(f"bad-mask-{index}", "p2p_islands_test", {"QUDA_ENABLE_P2P": value}, error="Invalid QUDA_ENABLE_P2P=")
for index, value in enumerate(("", "-1", "2", "3", "5", "text", "1x", "01")):
    run(f"bad-force-{index}", "p2p_islands_test", {"QUDA_ENABLE_P2P_FORCE": value}, error="Invalid QUDA_ENABLE_P2P_FORCE=")
run("contradictory", "p2p_islands_test", {"QUDA_ENABLE_P2P": "0", "QUDA_ENABLE_P2P_FORCE": "1"},
    error="QUDA_ENABLE_P2P_FORCE=1 requires")
for gdr in (0, 1):
    output = run(f"defaults-gdr{gdr}", "p2p_islands_test", {"QUDA_ENABLE_GDR": str(gdr)})
    assert ",p2p=3,p2p_force=0," in output
    for mask, force in ((0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)):
        output = run(f"topology-gdr{gdr}-mask{mask}-force{force}", "p2p_islands_test",
                     {"QUDA_ENABLE_P2P": str(mask), "QUDA_ENABLE_P2P_FORCE": str(force), "QUDA_ENABLE_GDR": str(gdr)})
        assert f",p2p={mask},p2p_force={force}," in output

cache = args.logs / "cache"
cache.mkdir(exist_ok=True)
for order, mask, force, isolated in itertools.product(("col", "row"), (1, 2, 3), (0, 1), (False, True)):
    settings = {"QUDA_ENABLE_P2P": str(mask), "QUDA_ENABLE_P2P_FORCE": str(force),
                "QUDA_ENABLE_TUNING": "1", "QUDA_RESOURCE_PATH": str(cache.resolve()), "QUDA_ENABLE_DSLASH_POLICY": "0"}
    if isolated:
        settings["QUDA_TEST_P2P_ISLANDS"] = ",".join(map(str, range(args.ranks)))
    run(f"verify-{order}-mask{mask}-force{force}-isolated{int(isolated)}", "dslash_test", settings,
        ["--dim", "4", "4", "4", "4", "--gridsize", "2", "1", "1", str(args.ranks // 2),
         "--rank-order", order, "--prec", "single", "--recon", "18", "--niter", "2",
         "--gtest_filter=DslashTest.verify"])

entries = list(csv.DictReader((cache / "tunecache.tsv").read_text().splitlines()[2:], delimiter="\t", skipinitialspace=True))
policies = [e for e in entries if e.get("aux", "").startswith("policy,")]
seen = set()
for entry in policies:
    key = (entry["volume"], entry["name"], entry["aux"])
    assert key not in seen
    seen.add(key)
    mask = int(re.search(r",p2p=(\d+)", entry["aux"])[1])
    force = int(re.search(r",p2p_force=(\d+)", entry["aux"])[1])
    selected = int(entry["aux.y"])
    allowed = ([0] if not force or not mask else []) + ([1] if mask & 1 else []) + ([2] if mask & 2 else [])
    assert selected in allowed, (entry["aux"], selected)
assert {int(re.search(r",p2p_force=(\d+)", e["aux"])[1]) for e in policies} == {0, 1}
assert len(policies) == 16, "Expected distinct mask/force/order keys, sharing only equivalent no-peer cases"
print(f"COMPLETE: {count} checks; {len(policies)} distinct policy cache entries audited", flush=True)
