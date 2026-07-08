#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def run_command(command: list[str], *, cwd: Path = REPO_ROOT) -> str:
    print("$ " + " ".join(command), flush=True)
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_parts.append(line)
        print(line, end="", flush=True)
    returncode = proc.wait()
    output = "".join(output_parts)
    if returncode != 0:
        raise RuntimeError(f"command failed with exit code {returncode}: {' '.join(command)}")
    return output


def xmake_command(*args: str) -> list[str]:
    if not args:
        return ["xmake", "-y"]
    return ["xmake", args[0], "-y", *args[1:]]


def bind_command(command: list[str], taskset_cpus: str | None = None, numa_node: int | None = None) -> list[str]:
    prefix: list[str] = []
    if taskset_cpus:
        prefix.extend(["taskset", "-c", taskset_cpus])
    if numa_node is not None:
        prefix.extend(["numactl", f"--membind={numa_node}"])
    return prefix + command


def append_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_k_time_ms(output: str) -> list[tuple[int, float]]:
    rows = []
    for match in re.finditer(r"k\s*=\s*(\d+).*?time\s*=\s*([0-9.]+)\s*ms", output):
        rows.append((int(match.group(1)), float(match.group(2))))
    return rows


def parse_total_cost_ms(output: str) -> list[float]:
    return [float(x) for x in re.findall(r"Total cost time:\s*([0-9.]+)", output)]


def parse_msm_config(output: str) -> dict[str, str]:
    match = re.search(r"Config:\s*(.*)", output)
    if not match:
        return {}
    config = {}
    for token in match.group(1).split():
        if "=" in token:
            key, value = token.split("=", 1)
            config[key] = value
    return config


def parse_named_ms(output: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)\s*ms", output)
    return float(match.group(1)) if match else None


def split_ints(value: str) -> list[int]:
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid integer list: {value}") from exc


def main_failed(exc: Exception) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 1
