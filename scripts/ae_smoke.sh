#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m py_compile scripts/ae_common.py scripts/ae_msm_bench.py scripts/ae_ntt_bench.py scripts/ae_ooc_ntt_bench.py scripts/ae_collect.py
python3 -m py_compile utils/cost_model.py utils/bench_numpy.py utils/generate_msm_problem.py

xmake build test-bn254
xmake run test-bn254

python3 utils/bench_numpy.py --backend numpy --n 4 --m 4 --runs 1
