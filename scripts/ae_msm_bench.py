#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ae_common import REPO_ROOT, RESULTS_DIR, append_csv, main_failed, parse_total_cost_ms, run_command, split_ints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FluxZK MSM benchmark target and collect timing CSV.")
    parser.add_argument("--log-lens", default="12", help="comma-separated MSM sizes as log2(n), e.g. 20,22")
    parser.add_argument("--runs", type=int, default=1, help="number of benchmark repetitions per input")
    parser.add_argument("--input-dir", type=Path, default=RESULTS_DIR / "inputs")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "msm.csv")
    parser.add_argument("--target", default="test-msm")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--regenerate-inputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_lens = split_ints(args.log_lens)
    args.input_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        run_command(["xmake", "build", args.target])

    rows: list[dict[str, object]] = []
    for log_len in log_lens:
        input_path = args.input_dir / f"msm_{log_len}.input"
        if args.regenerate_inputs or not input_path.exists():
            run_command(["python3", "utils/generate_msm_problem.py", str(log_len), str(input_path)])

        for run_id in range(args.runs):
            output = run_command(["xmake", "run", args.target, str(input_path)])
            times = parse_total_cost_ms(output)
            if not times:
                raise RuntimeError(f"could not find 'Total cost time:' in output for log_len={log_len}")
            for sample_id, time_ms in enumerate(times):
                rows.append(
                    {
                        "target": args.target,
                        "log_len": log_len,
                        "run": run_id,
                        "sample": sample_id,
                        "time_ms": time_ms,
                        "input": str(input_path),
                    }
                )

    append_csv(args.output, ["target", "log_len", "run", "sample", "time_ms", "input"], rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
