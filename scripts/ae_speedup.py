#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from ae_common import RESULTS_DIR, main_failed, split_ints


@dataclass(frozen=True)
class Series:
    target: str
    metric: str
    filters: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Comparison:
    claim: str
    flux: Series
    baseline: Series


COMPARISONS = (
    Comparison(
        "FluxMSM vs Sppark (BN254, batch=4, execution)",
        Series("bench-msm-bn254-ae", "time", (("curve", "bn254"), ("batch_size", "4"))),
        Series("sppark-msm", "execution", (("curve", "bn254"), ("batch_size", "4"))),
    ),
    Comparison(
        "FluxNTT Core vs Sppark (BN254, kernel)",
        Series("bench-ntt", "kernel"),
        Series("sppark-ntt", "kernel", (("curve", "bn254"),)),
    ),
    Comparison(
        "FluxNTT Core vs Sppark (MNT4753, kernel)",
        Series("bench-ntt-mnt4753", "kernel"),
        Series("sppark-ntt", "kernel", (("curve", "mnt4753"),)),
    ),
    Comparison(
        "FluxNTT Extended vs CUDA managed (BN254, execution)",
        Series("bench-ntt-4step", "time"),
        Series("bench-ntt-managed-bn254", "time"),
    ),
    Comparison(
        "FluxNTT Extended vs CUDA managed (MNT4753, execution)",
        Series("bench-ntt-4step-mnt4753", "time"),
        Series("bench-ntt-managed", "time"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare observed FluxZK and baseline timings on the same machine.")
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--ks", default="20,22,24,26,28")
    parser.add_argument("--stat", choices=("median", "mean", "min"), default="median")
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=0.25,
        help="relative slowdown still labeled PARITY for hardware-sensitive comparisons",
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ae_speedup.csv")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="include comparisons for which neither side has observations",
    )
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as file:
            rows.extend(csv.DictReader(file))
    return rows


def matches(row: dict[str, str], series: Series, k: int) -> bool:
    try:
        row_k = int(row.get("k") or row.get("log_len") or "")
    except ValueError:
        return False
    return (
        row.get("target") == series.target
        and row.get("metric", "time") == series.metric
        and row_k == k
        and all(row.get(key, "") == value for key, value in series.filters)
    )


def aggregate(rows: list[dict[str, str]], series: Series, k: int, stat: str) -> tuple[float | None, int]:
    values: list[float] = []
    for row in rows:
        if matches(row, series, k):
            try:
                values.append(float(row["time_ms"]))
            except (KeyError, ValueError):
                pass
    if not values:
        return None, 0
    if stat == "mean":
        return mean(values), len(values)
    if stat == "min":
        return min(values), len(values)
    return median(values), len(values)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv_files)
    report: list[dict[str, object]] = []
    for comparison in COMPARISONS:
        for k in split_ints(args.ks):
            flux_ms, flux_n = aggregate(rows, comparison.flux, k, args.stat)
            baseline_ms, baseline_n = aggregate(rows, comparison.baseline, k, args.stat)
            if flux_n == 0 and baseline_n == 0 and not args.include_empty:
                continue
            speedup = None if flux_ms is None or baseline_ms is None else baseline_ms / flux_ms
            if speedup is None:
                status = "MISSING"
            elif speedup > 1.0:
                status = "PASS"
            elif speedup >= 1.0 / (1.0 + args.parity_tolerance):
                status = "PARITY"
            else:
                status = "REGRESSION"
            report.append(
                {
                    "claim": comparison.claim,
                    "k": k,
                    "flux_ms": "" if flux_ms is None else f"{flux_ms:.6f}",
                    "baseline_ms": "" if baseline_ms is None else f"{baseline_ms:.6f}",
                    "speedup": "" if speedup is None else f"{speedup:.3f}",
                    "flux_n": flux_n,
                    "baseline_n": baseline_n,
                    "stat": args.stat,
                    "status": status,
                }
            )

    fieldnames = ["claim", "k", "flux_ms", "baseline_ms", "speedup", "flux_n", "baseline_n", "stat", "status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report)
    print(f"Wrote {len(report)} rows to {args.output}")
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
