#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from ae_common import REPO_ROOT, RESULTS_DIR, append_csv, main_failed, run_command, split_ints


SPPARK_ROOT = REPO_ROOT / "baselines" / "sppark"
MSM_DIR = SPPARK_ROOT / "poc" / "msm-cuda"
NTT_DIR = SPPARK_ROOT / "poc" / "ntt-cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run the vendored Sppark MSM/NTT baselines."
    )
    parser.add_argument("--components", default="msm,ntt", help="comma-separated: msm,ntt")
    parser.add_argument("--msm-curve", default="bn254", choices=("bn254", "bls12_377", "bls12_381"))
    parser.add_argument("--msm-ks", default="20,22,24")
    parser.add_argument("--msm-batch-size", type=int, default=4)
    parser.add_argument("--ntt-fields", default="bn254", help="comma-separated Sppark field features")
    parser.add_argument("--ntt-ks", default="20,22,24,26,28")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1, help="independent input generations")
    parser.add_argument("--target-dir", type=Path, default=RESULTS_DIR / "sppark-target")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "sppark.csv")
    parser.add_argument(
        "--rust-toolchain",
        default="1.81.0",
        help="Rustup toolchain for Sppark; pinned separately from FluxZK's legacy 1.67 toolchain",
    )
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def cargo_command(
    package_dir: Path,
    target_dir: Path,
    bench: str,
    feature: str,
    *,
    no_run: bool = False,
) -> list[str]:
    command = [
        "cargo",
        "bench",
        "--locked",
        "--bench",
        bench,
        "--features",
        feature,
        "--target-dir",
        str(target_dir.resolve()),
    ]
    if no_run:
        command.append("--no-run")
    return command


def parse_msm_times(output: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(r"repeated time\s*:\s*([0-9.]+)\s*ms", output)
    ]


def parse_ntt_times(output: str, warmups: int) -> list[tuple[int, int, float, float]]:
    rows: list[tuple[int, int, float, float]] = []
    current_k: int | None = None
    seen_for_k = 0
    compute_ms: float | None = None
    for line in output.splitlines():
        size = re.search(r"Testing NTT on domain size 2\^(\d+)", line)
        if size:
            current_k = int(size.group(1))
            seen_for_k = 0
            compute_ms = None
            continue
        compute = re.search(r"Compute time:\s*([0-9.]+)\s*ms", line)
        if compute:
            compute_ms = float(compute.group(1))
            continue
        total = re.search(r"Total time\s*:\s*([0-9.]+)\s*ms", line)
        if total and current_k is not None and compute_ms is not None:
            if seen_for_k >= warmups:
                rows.append((current_k, seen_for_k - warmups, compute_ms, float(total.group(1))))
            seen_for_k += 1
            compute_ms = None
    return rows


def run_msm(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.skip_build:
        run_command(
            cargo_command(MSM_DIR, args.target_dir, "msm_single", args.msm_curve, no_run=True),
            cwd=MSM_DIR,
            env={"RUSTUP_TOOLCHAIN": args.rust_toolchain},
        )

    rows: list[dict[str, object]] = []
    for k in split_ints(args.msm_ks):
        for run_id in range(args.runs):
            output = run_command(
                cargo_command(MSM_DIR, args.target_dir, "msm_single", args.msm_curve),
                cwd=MSM_DIR,
                env={
                    "RUSTUP_TOOLCHAIN": args.rust_toolchain,
                    "BENCH_NPOW": str(k),
                    "BENCH_NBATCH": str(args.msm_batch_size),
                    "BENCH_NRUNS": str(args.warmups + args.samples),
                },
            )
            times = parse_msm_times(output)
            expected = args.warmups + args.samples
            if len(times) != expected:
                raise RuntimeError(f"expected {expected} Sppark MSM timings for k={k}, found {len(times)}")
            for sample_id, time_ms in enumerate(times[args.warmups :]):
                rows.append(
                    {
                        "target": "sppark-msm",
                        "curve": args.msm_curve,
                        "k": k,
                        "batch_size": args.msm_batch_size,
                        "run": run_id,
                        "sample": sample_id,
                        "metric": "execution",
                        "time_ms": time_ms,
                    }
                )
    return rows


def run_ntt(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ks = split_ints(args.ntt_ks)
    for field in (value.strip() for value in args.ntt_fields.split(",") if value.strip()):
        if not args.skip_build:
            run_command(
                cargo_command(NTT_DIR, args.target_dir, "ntt", field, no_run=True),
                cwd=NTT_DIR,
                env={"RUSTUP_TOOLCHAIN": args.rust_toolchain},
            )
        for run_id in range(args.runs):
            output = run_command(
                cargo_command(NTT_DIR, args.target_dir, "ntt", field),
                cwd=NTT_DIR,
                env={
                    "RUSTUP_TOOLCHAIN": args.rust_toolchain,
                    "SPPARK_KS": ",".join(str(k) for k in ks),
                    "SPPARK_WARMUPS": str(args.warmups),
                    "SPPARK_SAMPLES": str(args.samples),
                },
            )
            samples = parse_ntt_times(output, args.warmups)
            expected = len(ks) * args.samples
            counts = Counter(k for k, _, _, _ in samples)
            expected_counts = {k: args.samples for k in ks}
            if len(samples) != expected or counts != expected_counts:
                raise RuntimeError(
                    f"expected Sppark NTT counts {expected_counts} for {field}, found {dict(counts)}"
                )
            for k, sample_id, compute_ms, total_ms in samples:
                common = {
                    "target": "sppark-ntt",
                    "curve": field,
                    "k": k,
                    "batch_size": "",
                    "run": run_id,
                    "sample": sample_id,
                }
                rows.append({**common, "metric": "kernel", "time_ms": compute_ms})
                rows.append({**common, "metric": "execution", "time_ms": total_ms})
    return rows


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.samples <= 0 or args.runs <= 0:
        raise ValueError("warmups must be non-negative; samples and runs must be positive")
    components = {value.strip() for value in args.components.split(",") if value.strip()}
    unknown = components - {"msm", "ntt"}
    if unknown:
        raise ValueError(f"unknown components: {','.join(sorted(unknown))}")

    rows: list[dict[str, object]] = []
    if "msm" in components:
        rows.extend(run_msm(args))
    if "ntt" in components:
        rows.extend(run_ntt(args))

    append_csv(
        args.output,
        ["target", "curve", "k", "batch_size", "run", "sample", "metric", "time_ms"],
        rows,
    )
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
