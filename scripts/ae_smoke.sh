#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m py_compile scripts/ae_common.py scripts/ae_msm_bench.py scripts/ae_ntt_bench.py scripts/ae_ooc_ntt_bench.py scripts/ae_collect.py
python3 -m py_compile utils/cost_model.py utils/generate_msm_problem.py utils/generate_ec_point.py utils/common.py

python3 scripts/ae_msm_bench.py --log-lens 8 --runs 1 --regenerate-inputs --input-dir results/smoke_inputs --output results/smoke_msm.csv
python3 scripts/ae_ntt_bench.py --target bench-ntt-end2end --ks 10 --runs 1 --output results/smoke_ntt.csv
python3 scripts/ae_collect.py results/smoke_msm.csv results/smoke_ntt.csv
