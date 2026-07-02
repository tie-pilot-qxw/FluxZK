#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from ae_common import RESULTS_DIR, main_failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize AE timing CSV files.")
    parser.add_argument("csv_files", nargs="*", type=Path, default=list(RESULTS_DIR.glob("*.csv")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in args.csv_files:
        if not path.exists():
            print(f"skip missing file: {path}")
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        groups: dict[tuple[str, str, str], list[float]] = {}
        for row in rows:
            target = row.get("target", "")
            size = row.get("log_len") or row.get("k") or ""
            metric = row.get("metric", "time")
            try:
                time_ms = float(row["time_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            groups.setdefault((target, size, metric), []).append(time_ms)
        print(path)
        for (target, size, metric), values in sorted(groups.items()):
            print(f"  {target:20s} size={size:>4s} metric={metric:12s} n={len(values):3d} mean_ms={mean(values):.3f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
