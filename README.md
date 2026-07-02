# Accelerating zk-SNARK with GPU

A high-performance CUDA-based library for accelerating zero-knowledge proof systems on GPUs.

## Key Features

- Efficient GPU Montgomery arithmetic implementation
- Optimized Multi-scalar Multiplication (MSM) computation
- High-performance Number Theoretic Transform (NTT)
- C/Rust language bindings

## Components

- `mont` - GPU implementation of Montgomery arithmetic
- `msm` - Efficient Multi-scalar Multiplication implementation
- `ntt` - Various optimized implementations of Number Theoretic Transform
- `wrapper` - C/Rust language binding interfaces

## Build

Build the entire project using xmake:
```sh
xmake build
```

## Test

CUDA tests using xmake:

```sh
# Montgomery arithmetic tests
xmake run test-mont

# MSM tests
xmake run test-bn254
python3 utils/generate_msm_problem.py 20 /tmp/fluxzk-msm.input
xmake run test-msm /tmp/fluxzk-msm.input

# NTT tests
xmake run test-ntt-4step
xmake run test-ntt-big
xmake run test-ntt-int
xmake run test-ntt-parallel
xmake run test-ntt-recompute
xmake run test-ntt-transpose
xmake run transpose

```

Benchmarks:
```sh
xmake run bench-mont
xmake run bench-mont0
xmake run bench-ntt
xmake run bench-ntt-4step
xmake run bench-ntt-end2end 24
```

## Artifact helper scripts

The `scripts/` directory contains lightweight wrappers for artifact evaluation.
They build the relevant xmake targets, run benchmarks, parse timing output, and
append CSV files under `results/`.

```sh
# Optional on clusters where HOME is on NFS:
mkdir -p /tmp/fluxzk-ae-home
export HOME=/tmp/fluxzk-ae-home

# Optional when GitHub archive downloads need a local proxy:
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# Configure CUDA and the nvcc host compiler on CUDA 12.x systems:
xmake f --cuda=/usr/local/cuda --cu-ccbin=gcc-11

# Quick sanity check for the MSM and NTT artifact paths
bash scripts/ae_smoke.sh

# MSM benchmark wrapper. Use larger log sizes for full experiments.
python3 scripts/ae_msm_bench.py --log-lens 12 --runs 1

# On-GPU NTT benchmark wrapper.
python3 scripts/ae_ntt_bench.py --target bench-ntt --runs 1

# Out-of-core NTT and CUDA managed-memory baseline wrapper.
python3 scripts/ae_ooc_ntt_bench.py --runs 1

# Summarize generated CSV files.
python3 scripts/ae_collect.py results/*.csv
```

For Rust binding tests:
```sh
cargo test
cargo test --package zk0d99c_msm --release --test msm -- --nocapture # test msm
```
# Cost Model

You can run `utils/cost_model.py` to get the best configuration for MSM.
use `python ./utils/cost_model.py -h` to see the parameters.

The output will looks like this:
```
alpha: 8
s: 16
c: 262144
divide: 4
h: 1
```
To apply this to the code, you need change `wrapper/msm/c_api/msm_c_api.cu`
where
```
using Config = msm::MsmConfig<field-bits, s, alpha, false>;
u32 batch_size = you batch size;
u32 batch_per_run = h;
u32 parts = divide;
u32 stage_scalers = 2;
u32 stage_points = 2;
```

## Acknowledgments

The Montgomery arithmetic implementation in the `mont/field` directory incorporates code from [sppark](https://github.com/supranational/sppark) (Apache-2.0 licensed). We gratefully acknowledge their contribution.
