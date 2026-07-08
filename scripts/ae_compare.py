#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from ae_common import RESULTS_DIR, main_failed


@dataclass(frozen=True)
class Expected:
    figure: str
    target: str
    metric: str
    k: int
    expected_ms: float
    label: str
    filters: tuple[tuple[str, str], ...] = ()


# Expected values reported by the paper for artifact-facing experiments.
EXPECTED: tuple[Expected, ...] = (
    Expected(
        "MSM",
        "bench-msm-bn254-ae",
        "time",
        20,
        43.980,
        "FluxMSM, BN254 batch=4",
        (("curve", "bn254"), ("batch_size", "4")),
    ),
    Expected(
        "MSM",
        "bench-msm-bn254-ae",
        "time",
        22,
        130.08,
        "FluxMSM, BN254 batch=4",
        (("curve", "bn254"), ("batch_size", "4")),
    ),
    Expected(
        "MSM",
        "bench-msm-bn254-ae",
        "time",
        24,
        430.95,
        "FluxMSM, BN254 batch=4",
        (("curve", "bn254"), ("batch_size", "4")),
    ),
    Expected(
        "MSM",
        "bench-msm-bn254-ae",
        "time",
        26,
        1597.60,
        "FluxMSM, BN254 batch=4",
        (("curve", "bn254"), ("batch_size", "4")),
    ),
    Expected(
        "MSM",
        "bench-msm-bn254-ae",
        "time",
        28,
        6339.10,
        "FluxMSM, BN254 batch=4",
        (("curve", "bn254"), ("batch_size", "4")),
    ),
    Expected(
        "MSM",
        "bench-msm-bn254-ae",
        "time",
        30,
        25458.00,
        "FluxMSM, BN254 batch=4",
        (("curve", "bn254"), ("batch_size", "4")),
    ),
    Expected("Fig.12(a)", "bench-ntt", "kernel", 20, 0.65, "FluxNTT Core, BN254/256-bit"),
    Expected("Fig.12(a)", "bench-ntt", "kernel", 22, 2.15, "FluxNTT Core, BN254/256-bit"),
    Expected("Fig.12(a)", "bench-ntt", "kernel", 24, 8.80, "FluxNTT Core, BN254/256-bit"),
    Expected("Fig.12(a)", "bench-ntt", "kernel", 26, 37.30, "FluxNTT Core, BN254/256-bit"),
    Expected("Fig.12(a)", "bench-ntt", "kernel", 28, 174.50, "FluxNTT Core, BN254/256-bit"),
    Expected("Fig.12(b)", "bench-ntt-4step", "time", 20, 13.60, "FluxNTT Extended, BN254/256-bit"),
    Expected("Fig.12(b)", "bench-ntt-4step", "time", 22, 33.00, "FluxNTT Extended, BN254/256-bit"),
    Expected("Fig.12(b)", "bench-ntt-4step", "time", 24, 140.20, "FluxNTT Extended, BN254/256-bit"),
    Expected("Fig.12(b)", "bench-ntt-4step", "time", 26, 437.60, "FluxNTT Extended, BN254/256-bit"),
    Expected("Fig.12(b)", "bench-ntt-4step", "time", 28, 1547.50, "FluxNTT Extended, BN254/256-bit"),
    Expected("Fig.12(b)", "bench-ntt-4step", "time", 30, 6106.70, "FluxNTT Extended, BN254/256-bit"),
    Expected("Fig.12(b)", "bench-ntt-managed", "time", 20, 73.00, "CUDA managed-memory baseline, MNT4753/768-bit"),
    Expected("Fig.12(b)", "bench-ntt-managed", "time", 22, 245.00, "CUDA managed-memory baseline, MNT4753/768-bit"),
    Expected("Fig.12(b)", "bench-ntt-managed", "time", 24, 903.00, "CUDA managed-memory baseline, MNT4753/768-bit"),
    Expected("Fig.12(b)", "bench-ntt-managed", "time", 26, 3580.00, "CUDA managed-memory baseline, MNT4753/768-bit"),
    Expected("Fig.12(b)", "bench-ntt-managed", "time", 28, 46934.00, "CUDA managed-memory baseline, MNT4753/768-bit"),
    Expected("Fig.12(b)", "bench-ntt-managed", "time", 30, 121671.00, "CUDA managed-memory baseline, MNT4753/768-bit"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AE benchmark CSV files with paper expected values.")
    parser.add_argument("csv_files", nargs="*", type=Path, default=list(RESULTS_DIR.glob("*.csv")))
    parser.add_argument("--stat", choices=("median", "mean", "min"), default="median")
    parser.add_argument("--tolerance", type=float, default=0.25, help="relative tolerance for PASS/DRIFT labels")
    parser.add_argument("--output", type=Path, help="optional CSV report path")
    return parser.parse_args()


def choose_stat(values: list[float], stat: str) -> float:
    if stat == "mean":
        return mean(values)
    if stat == "min":
        return min(values)
    return median(values)


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def row_k(row: dict[str, str]) -> int | None:
    try:
        return int(row.get("k") or row.get("log_len") or "")
    except ValueError:
        return None


def row_matches(row: dict[str, str], expected: Expected) -> bool:
    if row.get("target", "") != expected.target:
        return False
    if row.get("metric", "time") != expected.metric:
        return False
    if row_k(row) != expected.k:
        return False
    return all(row.get(key, "") == value for key, value in expected.filters)


def observed_values(rows: list[dict[str, str]], expected: Expected) -> list[float]:
    values = []
    for row in rows:
        if not row_matches(row, expected):
            continue
        try:
            values.append(float(row["time_ms"]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def format_filters(expected: Expected) -> str:
    return ",".join(f"{key}={value}" for key, value in expected.filters)


def main() -> int:
    args = parse_args()
    observed_rows = load_rows(args.csv_files)
    report_rows: list[dict[str, object]] = []

    for expected in EXPECTED:
        values = observed_values(observed_rows, expected)
        if values:
            observed_ms: float | None = choose_stat(values, args.stat)
            ratio: float | None = observed_ms / expected.expected_ms
            rel_error = abs(observed_ms - expected.expected_ms) / expected.expected_ms
            status = "PASS" if rel_error <= args.tolerance else "DRIFT"
        else:
            observed_ms = None
            ratio = None
            status = "MISSING"
        row = {
            "figure": expected.figure,
            "label": expected.label,
            "target": expected.target,
            "metric": expected.metric,
            "k": expected.k,
            "filters": format_filters(expected),
            "expected_ms": expected.expected_ms,
            "observed_ms": "" if observed_ms is None else f"{observed_ms:.3f}",
            "ratio": "" if ratio is None else f"{ratio:.3f}",
            "n": len(values),
            "stat": args.stat,
            "status": status,
        }
        report_rows.append(row)

    fieldnames = [
        "figure",
        "label",
        "target",
        "metric",
        "k",
        "filters",
        "expected_ms",
        "observed_ms",
        "ratio",
        "n",
        "stat",
        "status",
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"Wrote {len(report_rows)} rows to {args.output}")

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(report_rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
