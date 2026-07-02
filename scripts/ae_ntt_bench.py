#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ae_common import RESULTS_DIR, append_csv, main_failed, parse_k_time_ms, parse_named_ms, run_command, split_ints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FluxZK on-GPU NTT benchmarks and collect timing CSV.")
    parser.add_argument("--target", default="bench-ntt", help="xmake target, usually bench-ntt or bench-ntt-end2end")
    parser.add_argument("--ks", default="24", help="comma-separated k values for bench-ntt-end2end")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ntt.csv")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []

    if not args.skip_build:
        run_command(["xmake", "build", args.target])

    if args.target == "bench-ntt-end2end":
        for k in split_ints(args.ks):
            for run_id in range(args.runs):
                output = run_command(["xmake", "run", args.target, str(k)])
                rows.append(
                    {
                        "target": args.target,
                        "k": k,
                        "run": run_id,
                        "sample": 0,
                        "metric": "end_to_end",
                        "time_ms": parse_named_ms(output, "End-to-end time"),
                    }
                )
                rows.append(
                    {
                        "target": args.target,
                        "k": k,
                        "run": run_id,
                        "sample": 0,
                        "metric": "computation",
                        "time_ms": parse_named_ms(output, "Computation time"),
                    }
                )
    else:
        for run_id in range(args.runs):
            output = run_command(["xmake", "run", args.target])
            samples = parse_k_time_ms(output)
            if not samples:
                raise RuntimeError(f"could not find 'k = ..., time = ... ms' lines in {args.target} output")
            for sample_id, (k, time_ms) in enumerate(samples):
                rows.append(
                    {
                        "target": args.target,
                        "k": k,
                        "run": run_id,
                        "sample": sample_id,
                        "metric": "kernel",
                        "time_ms": time_ms,
                    }
                )

    append_csv(args.output, ["target", "k", "run", "sample", "metric", "time_ms"], rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
