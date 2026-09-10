#!/usr/bin/env python3
"""Stream Linux build resource snapshots to the job log and a JSONL file."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def counters(path):
    return {
        key.rstrip(":"): int(value)
        for key, value, *_ in (line.split() for line in (read(path) or "").splitlines())
    }


def snapshot():
    memory = counters("/proc/meminfo")
    cgroups = {}
    root = Path("/sys/fs/cgroup")
    for entry in (read("/proc/self/cgroup") or "").splitlines():
        if not entry.startswith("0::"):
            continue
        group = (root / entry[3:].lstrip("/")).resolve()
        while group.is_relative_to(root):
            cgroups[str(group)] = {
                name: read(group / name)
                for name in (
                    "memory.current", "memory.peak", "memory.max", "memory.high",
                    "memory.swap.current", "memory.swap.max", "memory.events", "memory.pressure",
                )
            }
            if group == root:
                break
            group = group.parent

    try:
        processes = subprocess.run(
            ["ps", "-eo", "pid,ppid,rss,pcpu,comm", "--sort=-rss"],
            capture_output=True, text=True, timeout=5,
        )
        largest = (processes.stdout if processes.returncode == 0 else processes.stderr).splitlines()[:16]
    except (OSError, subprocess.TimeoutExpired) as error:
        largest = [str(error)]

    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "monitor_pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "memory_kib": {
            key: memory.get(key)
            for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Dirty", "Writeback")
        },
        "oom_kills": counters("/proc/vmstat").get("oom_kill"),
        "pressure": {kind: read("/proc/pressure/" + kind) for kind in ("cpu", "memory", "io")},
        "loadavg": read("/proc/loadavg"),
        "disk_bytes": {
            path: shutil.disk_usage(path)._asdict()
            for path in sorted({"/", os.environ["GITHUB_WORKSPACE"], os.environ["RUNNER_TEMP"]})
        },
        "cgroups": cgroups,
        "largest_processes_rss_kib": largest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            data = snapshot()
        except Exception as error:
            data = {"time_utc": datetime.now(timezone.utc).isoformat(), "monitor_error": str(error)}
        line = json.dumps(data, separators=(",", ":"))
        print("RESOURCE " + line, flush=True)
        try:
            with args.output.open("a") as output:
                output.write(line + "\n")
        except OSError as error:
            print("Resource log write failed: " + str(error), file=sys.stderr, flush=True)
        if args.once:
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
