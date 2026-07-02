#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ae_common import RESULTS_DIR, append_csv, main_failed, parse_k_time_ms, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run out-of-core NTT and managed-memory baseline benchmarks.")
    parser.add_argument("--targets", default="bench-ntt-4step,bench-ntt-managed")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ntt_ooc.csv")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    rows: list[dict[str, object]] = []

    for target in targets:
        if not args.skip_build:
            run_command(["xmake", "build", target])
        for run_id in range(args.runs):
            output = run_command(["xmake", "run", target])
            samples = parse_k_time_ms(output)
            if not samples:
                raise RuntimeError(f"could not parse timing lines from {target}")
            for sample_id, (k, time_ms) in enumerate(samples):
                rows.append(
                    {
                        "target": target,
                        "k": k,
                        "run": run_id,
                        "sample": sample_id,
                        "time_ms": time_ms,
                    }
                )

    append_csv(args.output, ["target", "k", "run", "sample", "time_ms"], rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
