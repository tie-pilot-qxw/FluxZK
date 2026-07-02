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
    proc = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(command)}")
    return proc.stdout


def xmake_command(*args: str) -> list[str]:
    if not args:
        return ["xmake", "-y"]
    return ["xmake", args[0], "-y", *args[1:]]


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
