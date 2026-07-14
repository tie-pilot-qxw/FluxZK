# Accelerating zk-SNARK with GPU

A high-performance CUDA-based library for accelerating zero-knowledge proof systems on GPUs.

## Paper

This repository contains the implementation and artifact-evaluation scripts for
the forthcoming SC26 paper:

**FluxZK: Scalable and Efficient Zero-Knowledge Proof Computation via GPU
Acceleration**

Authors: Xinwei Qiang, Liukun Yu, Xiyu Wang, Zhengyi Li, Shixuan Sun,
Jingwen Leng, Chen Chen, Jiaping Gui, Zhenzhe Zheng, Jin Dong, and Minyi Guo.
Xinwei Qiang and Liukun Yu contributed equally to this work.

Formal citation metadata will be added after the paper is officially published.

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
# Optional on clusters where HOME is on a shared filesystem:
mkdir -p /tmp/fluxzk-ae-home
export HOME=/tmp/fluxzk-ae-home

# Configure CUDA and the nvcc host compiler on CUDA 12.x systems:
xmake f --cuda=/usr/local/cuda --cu-ccbin=gcc-11

# Quick check for the MSM and NTT artifact paths
bash scripts/ae_quick_check.sh

# MSM paper-timing wrapper. Verified paper configurations are pinned for k=20
# and k=22; other sizes fall back to the BN254 cost model.
# The input generator writes random scalars for the full log size and reuses
# a bounded random point pool to keep generation time manageable.
python3 scripts/ae_msm_bench.py --use-cost-model --paper-configs --warmups 3 \
  --log-lens 20,22 --batch-sizes 4 --runs 3 \
  --xmake-cuda /usr/local/cuda --xmake-cu-ccbin gcc-11

# On-GPU NTT benchmark wrapper.
python3 scripts/ae_ntt_bench.py --target bench-ntt --ks 20,22,24,26,28 --runs 1

# Out-of-core NTT and CUDA managed-memory baseline wrapper.
# Set CPU_LIST and NUMA_NODE to the GPU-local socket reported by nvidia-smi topo -m.
python3 scripts/ae_ooc_ntt_bench.py --ks 20,22,24,26,28 --runs 1 \
  --taskset-cpus CPU_LIST --numa-node NUMA_NODE

# Summarize and compare generated CSV files.
python3 scripts/ae_collect.py results/*.csv
python3 scripts/ae_compare.py results/*.csv --stat median --output results/ae_compare.csv
```

### Results Reproduced workflow

The repository vendors the Sppark baseline under `baselines/sppark` at upstream
commit `cb1bc09bcb69134f13ac1f59145dc659bf15bf34`. The fork includes the paper's
MNT4753 field extension. The following workflow runs FluxMSM against Sppark,
FluxNTT Core against Sppark, and FluxNTT Extended against CUDA managed memory.
Every reported speedup compares the same GPU, scale, field, batch size, and
timing scope.

Sppark's lock files require a newer Cargo than FluxZK's bindings. The wrapper
pins Rust 1.81.0 for Sppark while retaining FluxZK's legacy Rust 1.67.0 pin for
its own bindings. For a bare-metal run, install it once with
`rustup toolchain install 1.81.0 --profile minimal`.

```sh
# Use the GPU-local CPU socket reported by `nvidia-smi topo -m`. If NUMA_NODE
# is unset and numactl is installed, the driver detects the closest node.
export CPU_LIST=16-31,48-63
export NUMA_NODE=1

# A small end-to-end check of every comparison path.
bash scripts/ae_reproduce.sh smoke

# Recommended AE run: representative points for every central kernel claim.
bash scripts/ae_reproduce.sh core

# Optional broader scale sweep (still omits resource-heavy k=30 for 768-bit fields).
# At least 128 GB of host memory and 30 GB of writable result space are recommended.
bash scripts/ae_reproduce.sh full
```

Each invocation creates a timestamped directory under `results/`. The main
output is `ae-speedup.csv`: `PASS` means FluxZK is faster, `PARITY` means it is
within the default 25% hardware-variation band, and `REGRESSION` means it is
more than 25% slower. BN254 NTT at k=20 is retained as `DIAGNOSTIC`, rather
than treated as a performance pass/fail point, because both kernels complete
in under 1 ms on A100; the evaluated BN254 NTT range is k=22 through k=28.
The paper describes this 256-bit comparison as marginal, so either `PASS` or
`PARITY` is consistent with that claim; the MSM, 768-bit NTT, and out-of-core
comparisons are expected to report `PASS`. Exact timings and speedups are
hardware dependent.

### Docker environment

The supplied container fixes the paper-facing build environment to CUDA 12.6
(including the NGC image digest),
GCC 11, xmake 2.9.9, Rust 1.67.0 for FluxZK, and Rust 1.81.0 for the bundled
Sppark baseline. It uses NVIDIA's NGC CUDA development image and downloads all
xmake/Sppark Cargo dependencies while building, so the AE benchmark execution
can be performed without network access.

```sh
# Log in first if your NGC configuration requires authentication.
docker login nvcr.io

docker build \
  --build-arg USER_ID="$(id -u)" --build-arg GROUP_ID="$(id -g)" \
  -t fluxzk-ae:cuda12.6 .

# Functional smoke test.
docker run --gpus all \
  -v "$PWD/results:/workspace/FluxZK/results" \
  fluxzk-ae:cuda12.6 bash scripts/ae_quick_check.sh

# Recommended Results Reproduced run. Replace the CPU and memory-node values
# with those reported for GPU 0 by `nvidia-smi topo -m` on the host.
docker run --gpus all \
  --cpuset-cpus=16-31,48-63 --cpuset-mems=1 \
  -e CPU_LIST=16-31,48-63 -e NUMA_NODE=1 \
  -v "$PWD/results:/workspace/FluxZK/results" \
  fluxzk-ae:cuda12.6 bash scripts/ae_reproduce.sh core
```

The host must have Docker with the NVIDIA Container Toolkit installed. The
container supplies the CUDA toolkit, but it uses the host NVIDIA driver and
GPU. Override the `CUDA_IMAGE` build argument if the NGC registry exposes a
site-specific CUDA 12.6 tag. Inside Docker, `--cpuset-mems` provides the NUMA
memory binding; the reproduction script therefore skips the redundant inner
`numactl --membind` call that an unprivileged container may reject.

For MSM and out-of-core benchmarks, pin the process to CPUs in the GPU-local
socket and bind host allocations to the matching NUMA node. These measurements
are sensitive to host DRAM bandwidth, PCIe topology, and cold GPU clocks. Each
comparison pair uses identical warmup and sample counts: 3/3 for MSM, 200/3
for on-GPU NTT, and 1/3 for out-of-core NTT (warmups/measured samples).
The driver records the protocol and chosen NUMA node in `environment.txt`.
The `full` workflow was validated end to end on an A100 PCIe 40 GB: with the
image already built, it took about 1 hour 16 minutes and produced about 19 GB
of results and generated inputs. Allow at least 30 GB of writable space.
The paper used CUDA 12.6; CUDA 12.8 is functionally compatible but may shift
exact timings slightly.

For Rust binding tests:
```sh
cargo test
cargo test --package zk0d99c_msm --release --test msm -- --nocapture # test msm
```
## Cost Model

You can run `utils/cost_model.py` to get the best configuration for MSM.
Use `python ./utils/cost_model.py -h` to see the parameters.

The output looks like this:
```
alpha: 8
s: 16
c: 262144
divide: 4
h: 1
```
To apply this to the code, change `wrapper/msm/c_api/msm_c_api.cu`
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
