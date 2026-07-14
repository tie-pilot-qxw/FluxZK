#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m py_compile scripts/ae_common.py scripts/ae_msm_bench.py scripts/ae_ntt_bench.py scripts/ae_ooc_ntt_bench.py scripts/ae_sppark_bench.py scripts/ae_collect.py scripts/ae_compare.py scripts/ae_speedup.py
python3 -m py_compile utils/cost_model.py utils/generate_msm_problem.py utils/generate_ec_point.py utils/common.py

python3 scripts/ae_msm_bench.py --log-lens 8 --runs 1 --regenerate-inputs --input-dir results/quick_check_inputs --output results/quick_check_msm.csv
python3 scripts/ae_ntt_bench.py --target bench-ntt-end2end --ks 10 --runs 1 --output results/quick_check_ntt.csv
python3 scripts/ae_collect.py results/quick_check_msm.csv results/quick_check_ntt.csv
