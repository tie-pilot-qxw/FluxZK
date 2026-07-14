#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-core}"
case "$PROFILE" in
  smoke)
    MSM_KS="20"
    NTT_BN_KS="20"
    NTT_MNT_KS="20"
    OOC_KS="20"
    ;;
  core)
    MSM_KS="20,22"
    NTT_BN_KS="20,22,24,26"
    NTT_MNT_KS="20,22,24"
    OOC_KS="22,24,26"
    ;;
  full)
    MSM_KS="20,22,24,26"
    NTT_BN_KS="20,22,24,26,28"
    NTT_MNT_KS="20,22,24,26"
    OOC_KS="20,22,24,26,28"
    ;;
  *)
    echo "usage: $0 [smoke|core|full]" >&2
    exit 2
    ;;
esac

command -v nvidia-smi >/dev/null
command -v xmake >/dev/null
command -v cargo >/dev/null

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_DIR:-results/reproduce-${PROFILE}-${STAMP}}"
mkdir -p "$RESULT_DIR"

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVCC_HOST_COMPILER="${NVCC_HOST_COMPILER:-gcc-11}"
SAMPLES="${SAMPLES:-3}"
FLUX_MSM_WARMUPS="${FLUX_MSM_WARMUPS:-3}"
SPPARK_WARMUPS="${SPPARK_WARMUPS:-3}"
SPPARK_RUST_TOOLCHAIN="${SPPARK_RUST_TOOLCHAIN:-1.81.0}"
# Short NTTs need enough aggregate work to bring an idle data-center GPU to
# stable clocks; 30 warmups can be less than 100 ms at the smallest scales.
FLUX_NTT_WARMUPS="${FLUX_NTT_WARMUPS:-200}"
CPU_LIST="${CPU_LIST:-}"
NUMA_NODE="${NUMA_NODE:-}"

if [[ -z "$NUMA_NODE" ]] && command -v numactl >/dev/null; then
  if DETECTED_NUMA="$(nvidia-smi topo -M -i 0 2>/dev/null | awk '/closest memory/ {print $NF}')" \
      && [[ "$DETECTED_NUMA" =~ ^[0-9]+$ ]]; then
    NUMA_NODE="$DETECTED_NUMA"
  fi
fi

BIND_ARGS=()
if [[ -n "$CPU_LIST" ]]; then
  BIND_ARGS+=(--taskset-cpus "$CPU_LIST")
fi
if [[ -n "$NUMA_NODE" ]]; then
  BIND_ARGS+=(--numa-node "$NUMA_NODE")
fi

MSM_RUN_PREFIX=()
if [[ -n "$NUMA_NODE" ]]; then
  MSM_RUN_PREFIX+=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
fi
if [[ -n "$CPU_LIST" ]]; then
  MSM_RUN_PREFIX+=(taskset -c "$CPU_LIST")
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >"$RESULT_DIR/environment.txt"
nvcc --version >>"$RESULT_DIR/environment.txt"
cargo --version >>"$RESULT_DIR/environment.txt"
rustup run "$SPPARK_RUST_TOOLCHAIN" cargo --version >>"$RESULT_DIR/environment.txt"
xmake --version >>"$RESULT_DIR/environment.txt"
echo "MSM_NUMA_NODE=${NUMA_NODE:-unbound}" >>"$RESULT_DIR/environment.txt"
echo "MSM_CPU_LIST=${CPU_LIST:-unbound}" >>"$RESULT_DIR/environment.txt"

xmake f --cuda="$CUDA_PATH" --cu-ccbin="$NVCC_HOST_COMPILER"

"${MSM_RUN_PREFIX[@]}" python3 scripts/ae_msm_bench.py \
  --use-cost-model --paper-configs --log-lens "$MSM_KS" --batch-sizes 4 --runs "$SAMPLES" \
  --warmups "$FLUX_MSM_WARMUPS" \
  --seed 20260713 --regenerate-inputs --output "$RESULT_DIR/flux-msm.csv" \
  --input-dir "$RESULT_DIR/inputs" --generated-config-dir "$RESULT_DIR/generated-msm-configs" \
  --xmake-cuda "$CUDA_PATH" --xmake-cu-ccbin "$NVCC_HOST_COMPILER"

python3 scripts/ae_sppark_bench.py \
  --components msm --msm-curve bn254 --msm-ks "$MSM_KS" --msm-batch-size 4 \
  --warmups "$SPPARK_WARMUPS" --samples "$SAMPLES" \
  --rust-toolchain "$SPPARK_RUST_TOOLCHAIN" \
  --target-dir "$RESULT_DIR/sppark-target" --output "$RESULT_DIR/sppark.csv"

python3 scripts/ae_ntt_bench.py \
  --target bench-ntt --ks "$NTT_BN_KS" --samples "$SAMPLES" \
  --warmups "$FLUX_NTT_WARMUPS" --output "$RESULT_DIR/flux-ntt.csv"
python3 scripts/ae_ntt_bench.py \
  --target bench-ntt-mnt4753 --ks "$NTT_MNT_KS" --samples "$SAMPLES" \
  --warmups "$FLUX_NTT_WARMUPS" --output "$RESULT_DIR/flux-ntt.csv"

python3 scripts/ae_sppark_bench.py \
  --components ntt --ntt-fields bn254 --ntt-ks "$NTT_BN_KS" \
  --warmups "$SPPARK_WARMUPS" --samples "$SAMPLES" \
  --rust-toolchain "$SPPARK_RUST_TOOLCHAIN" \
  --target-dir "$RESULT_DIR/sppark-target" --output "$RESULT_DIR/sppark.csv"
python3 scripts/ae_sppark_bench.py \
  --components ntt --ntt-fields mnt4753 --ntt-ks "$NTT_MNT_KS" \
  --warmups "$SPPARK_WARMUPS" --samples "$SAMPLES" \
  --rust-toolchain "$SPPARK_RUST_TOOLCHAIN" \
  --target-dir "$RESULT_DIR/sppark-target" --output "$RESULT_DIR/sppark.csv"

python3 scripts/ae_ooc_ntt_bench.py \
  --targets bench-ntt-4step,bench-ntt-managed-bn254,bench-ntt-4step-mnt4753,bench-ntt-managed \
  --ks "$OOC_KS" --samples "$SAMPLES" --output "$RESULT_DIR/ntt-ooc.csv" \
  "${BIND_ARGS[@]}"

python3 scripts/ae_speedup.py \
  "$RESULT_DIR/flux-msm.csv" "$RESULT_DIR/sppark.csv" \
  "$RESULT_DIR/flux-ntt.csv" "$RESULT_DIR/ntt-ooc.csv" \
  --ks "20,22,24,26,28" --output "$RESULT_DIR/ae-speedup.csv"

echo "Results Reproduced report: $RESULT_DIR/ae-speedup.csv"
