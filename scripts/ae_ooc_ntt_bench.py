#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ae_common import RESULTS_DIR, append_csv, bind_command, main_failed, parse_k_time_ms, run_command, split_ints, xmake_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run out-of-core NTT and managed-memory baseline benchmarks.")
    parser.add_argument("--targets", default="bench-ntt-4step,bench-ntt-managed")
    parser.add_argument("--ks", default="20,22,24,26,28", help="comma-separated k values; add 30 for the full paper-scale run")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ntt_ooc.csv")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--taskset-cpus", help="CPU affinity list for benchmark runs, e.g. the GPU-local socket from nvidia-smi topo -m")
    parser.add_argument("--numa-node", type=int, help="NUMA node for benchmark host allocations, e.g. the GPU-local NUMA node")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    ks = split_ints(args.ks)

    for target in targets:
        if not args.skip_build:
            run_command(xmake_command("build", target))
        for k in ks:
            for run_id in range(args.runs):
                output = run_command(
                    bind_command(
                        xmake_command("run", target, "--ks", str(k), "--samples", str(args.samples)),
                        args.taskset_cpus,
                        args.numa_node,
                    )
                )
                samples = parse_k_time_ms(output)
                if not samples:
                    raise RuntimeError(f"could not parse timing lines from {target}")
                rows = []
                for sample_id, (observed_k, time_ms) in enumerate(samples):
                    rows.append(
                        {
                            "target": target,
                            "k": observed_k,
                            "run": run_id,
                            "sample": sample_id,
                            "metric": "time",
                            "time_ms": time_ms,
                            "taskset_cpus": args.taskset_cpus or "",
                            "numa_node": "" if args.numa_node is None else args.numa_node,
                        }
                    )
                append_csv(
                    args.output,
                    ["target", "k", "run", "sample", "metric", "time_ms", "taskset_cpus", "numa_node"],
                    rows,
                )

    print(f"Wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
