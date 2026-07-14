#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from ae_common import (
    REPO_ROOT,
    RESULTS_DIR,
    append_csv,
    main_failed,
    parse_msm_config,
    parse_total_cost_ms,
    run_command,
    split_ints,
    xmake_command,
)


BN254_CONFIG_TEMPLATE = """#include "{msm_impl}"
#include "{bn254}"

namespace msm {{
    using Config = MsmConfig<255, {window_size}, {precompute}, false>;
    template class MSM<Config, bn254::Number, bn254::Point, bn254::PointAffine>;
    template class MSMPrecompute<Config, bn254::Point, bn254::PointAffine>;
    template class MultiGPUMSM<Config, bn254::Number, bn254::Point, bn254::PointAffine>;
}}
"""


# Configurations used for the paper's A100 PCIe 40 GB, BN254, batch-4
# measurements.  The k=20 cost-model candidates are nearly tied in the
# analytical model, but s=16/alpha=4 is measurably faster than the nominal
# minimum on the target GPU.
PAPER_BN254_BATCH4_CONFIGS = {
    20: {"s": 16, "alpha": 4, "divide": 4, "h": 4},
    22: {"s": 17, "alpha": 5, "divide": 4, "h": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FluxZK MSM benchmark target and collect timing CSV.")
    parser.add_argument("--log-lens", default="12", help="comma-separated MSM sizes as log2(n), e.g. 20,22")
    parser.add_argument("--runs", type=int, default=1, help="number of benchmark repetitions per input")
    parser.add_argument("--warmups", type=int, default=1, help="untimed MSM runs before each measured run")
    parser.add_argument(
        "--max-generated-log-len",
        type=int,
        help="debug-only cap for generated input size; values smaller than --log-lens repeat scalars and are rejected",
    )
    parser.add_argument(
        "--point-pool-log-len",
        type=int,
        default=12,
        help="number of random points to generate as log2(pool size); scalars are still random for every row",
    )
    parser.add_argument("--seed", type=int, help="optional deterministic seed for generated MSM inputs")
    parser.add_argument("--input-dir", type=Path, default=RESULTS_DIR / "inputs")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "msm.csv")
    parser.add_argument("--target", default="test-msm")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--regenerate-inputs", action="store_true")
    parser.add_argument(
        "--use-cost-model",
        action="store_true",
        help="select BN254 MSM configuration with utils/cost_model.py",
    )
    parser.add_argument(
        "--paper-configs",
        action="store_true",
        help="use verified paper configurations when available, falling back to the cost model",
    )
    parser.add_argument("--batch-sizes", default="4", help="comma-separated batch sizes for --use-cost-model")
    parser.add_argument("--gpu-mem-gb", type=int, default=40)
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=RESULTS_DIR / "generated_msm_configs",
    )
    parser.add_argument("--xmake-cuda", help="optional CUDA SDK path to preserve during cost-model reconfiguration")
    parser.add_argument("--xmake-cu-ccbin", help="optional nvcc host compiler to preserve during cost-model reconfiguration")
    return parser.parse_args()


def parse_cost_model(output: str) -> dict[str, int]:
    values = {}
    for key in ("alpha", "s", "divide", "h"):
        match = re.search(rf"{key}:\s*(\d+)", output)
        if not match:
            raise RuntimeError(f"could not parse {key} from cost model output")
        values[key] = int(match.group(1))
    return values


def cost_model_bn254(log_len: int, batch_size: int, gpu_mem_gb: int) -> dict[str, int]:
    output = run_command(
        [
            "python3",
            "utils/cost_model.py",
            "--k",
            str(log_len),
            "--n",
            str(batch_size),
            "--l",
            "255",
            "--p",
            "255",
            "--mem",
            str(gpu_mem_gb * (2**30)),
        ]
    )
    return parse_cost_model(output)


def ensure_bn254_config(config_dir: Path, window_size: int, precompute: int) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = (config_dir / f"msm_bn254_{window_size}_{precompute}_f.cu").resolve()
    msm_impl = os.path.relpath(REPO_ROOT / "msm" / "src" / "msm_impl.cuh", path.parent)
    bn254 = os.path.relpath(REPO_ROOT / "msm" / "src" / "bn254.cuh", path.parent)
    path.write_text(
        BN254_CONFIG_TEMPLATE.format(
            msm_impl=msm_impl,
            bn254=bn254,
            window_size=window_size,
            precompute=precompute,
        ),
        encoding="utf-8",
    )
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def configure_cost_model_target(
    config_file: Path,
    cfg: dict[str, int],
    batch_size: int,
    args: argparse.Namespace,
) -> str:
    target = "bench-msm-bn254-ae"
    configure_args = [
        f"--msm_window_size={cfg['s']}",
        f"--msm_precompute={cfg['alpha']}",
        f"--msm_batch_size={batch_size}",
        f"--msm_batch_per_run={cfg['h']}",
        f"--msm_parts={cfg['divide']}",
        f"--msm_warmups={args.warmups}",
        f"--msm_config_file={config_file}",
    ]
    if args.xmake_cuda:
        configure_args.append(f"--cuda={args.xmake_cuda}")
    if args.xmake_cu_ccbin:
        configure_args.append(f"--cu-ccbin={args.xmake_cu_ccbin}")
    run_command(xmake_command("f", *configure_args))
    return target


def main() -> int:
    args = parse_args()
    log_lens = split_ints(args.log_lens)
    batch_sizes = split_ints(args.batch_sizes) if args.use_cost_model else [None]
    args.input_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build and not args.use_cost_model:
        run_command(xmake_command("build", args.target))

    rows: list[dict[str, object]] = []
    generated_bases: set[int] = set()
    for batch_size in batch_sizes:
        for log_len in log_lens:
            base_log_len = log_len
            if args.max_generated_log_len is not None:
                if args.max_generated_log_len < log_len:
                    raise RuntimeError(
                        "--max-generated-log-len would repeat scalar values; generate the full input for AE runs"
                    )
                base_log_len = min(log_len, args.max_generated_log_len)
            input_path = (args.input_dir / f"msm_base_{base_log_len}.input").resolve()
            if (args.regenerate_inputs and base_log_len not in generated_bases) or not input_path.exists():
                generate_command = [
                    "python3",
                    "utils/generate_msm_problem.py",
                    str(base_log_len),
                    str(input_path),
                    "--point-pool-log-len",
                    str(args.point_pool_log_len),
                ]
                if args.seed is not None:
                    generate_command.extend(["--seed", str(args.seed)])
                run_command(generate_command)
                generated_bases.add(base_log_len)

            target = args.target
            cost_model_cfg: dict[str, int] = {}
            config_source = "target_default"
            config_file: Path | None = None
            if args.use_cost_model:
                assert batch_size is not None
                paper_cfg = PAPER_BN254_BATCH4_CONFIGS.get(log_len) if batch_size == 4 else None
                if args.paper_configs and paper_cfg is not None:
                    cost_model_cfg = dict(paper_cfg)
                    config_source = "paper_pinned"
                else:
                    cost_model_cfg = cost_model_bn254(log_len, batch_size, args.gpu_mem_gb)
                    config_source = "cost_model"
                config_file = ensure_bn254_config(
                    args.generated_config_dir,
                    cost_model_cfg["s"],
                    cost_model_cfg["alpha"],
                )
                target = configure_cost_model_target(config_file, cost_model_cfg, batch_size, args)
                if not args.skip_build:
                    run_command(xmake_command("build", target))

            for run_id in range(args.runs):
                output = run_command(xmake_command("run", target, str(input_path), str(log_len)))
                times = parse_total_cost_ms(output)
                if not times:
                    raise RuntimeError(f"could not find 'Total cost time:' in output for log_len={log_len}")
                config = parse_msm_config(output)
                for sample_id, time_ms in enumerate(times):
                    rows.append(
                        {
                            "target": target,
                            "curve": config.get("curve", "bn254"),
                            "log_len": log_len,
                            "k": log_len,
                            "batch_size": config.get("batch_size", ""),
                            "window_size": config.get("window_size", ""),
                            "precompute": config.get("precompute", ""),
                            "batch_per_run": config.get("batch_per_run", ""),
                            "parts": config.get("parts", ""),
                            "run": run_id,
                            "sample": sample_id,
                            "metric": "time",
                            "time_ms": time_ms,
                            "input": display_path(input_path),
                            "config_source": config_source,
                            "config_file": "" if config_file is None else display_path(config_file),
                        }
                    )

    append_csv(
        args.output,
        [
            "target",
            "curve",
            "log_len",
            "k",
            "batch_size",
            "window_size",
            "precompute",
            "batch_per_run",
            "parts",
            "run",
            "sample",
            "metric",
            "time_ms",
            "input",
            "config_source",
            "config_file",
        ],
        rows,
    )
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(main_failed(exc))
