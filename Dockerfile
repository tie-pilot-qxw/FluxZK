# syntax=docker/dockerfile:1

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu22.04@sha256:d49bb8a4ff97fb5fe477947a3f02aa8c0a53eae77e11f00ec28618a0bcaa2ad1
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG XMAKE_VERSION=v2.9.9
ARG XMAKE_SHA256=fc4f7618d5343e6d1b4c352e7e6df86f650b8021a69dda9f8b256717728556bc
ARG FLUX_RUST_TOOLCHAIN=1.67.0
ARG SPPARK_RUST_TOOLCHAIN=1.81.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        g++-11 \
        gcc-11 \
        git \
        numactl \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${GROUP_ID}" ae \
    && useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash ae

ENV HOME=/home/ae \
    CARGO_HOME=/home/ae/.cargo \
    RUSTUP_HOME=/home/ae/.rustup \
    PATH=/home/ae/.cargo/bin:/home/ae/.local/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    NVCC=/usr/local/cuda/bin/nvcc \
    CC=gcc-11 \
    CXX=g++-11 \
    CUDAHOSTCXX=g++-11 \
    NVCC_HOST_COMPILER=gcc-11 \
    SPPARK_RUST_TOOLCHAIN=${SPPARK_RUST_TOOLCHAIN} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER ae

RUN curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "${FLUX_RUST_TOOLCHAIN}" \
    && rustup toolchain install "${SPPARK_RUST_TOOLCHAIN}" --profile minimal \
    && rm /tmp/rustup-init.sh

# Use the self-contained official release binary. The bootstrap installer may
# select a regional git mirror, which makes container builds network-dependent.
RUN mkdir -p "${HOME}/.local/bin" \
    && curl -fL --retry 3 \
        "https://github.com/xmake-io/xmake/releases/download/${XMAKE_VERSION}/xmake-bundle-${XMAKE_VERSION}.linux.x86_64" \
        -o "${HOME}/.local/bin/xmake" \
    && echo "${XMAKE_SHA256}  ${HOME}/.local/bin/xmake" | sha256sum --check --strict \
    && chmod 0755 "${HOME}/.local/bin/xmake" \
    && xmake --version

WORKDIR /workspace/FluxZK
COPY --chown=ae:ae . .

# Cache the dependencies used by the AE comparison while image construction
# has network access. The root Rust workspace is not part of this workflow.
RUN RUSTUP_TOOLCHAIN="${SPPARK_RUST_TOOLCHAIN}" cargo fetch --locked \
        --manifest-path baselines/sppark/poc/msm-cuda/Cargo.toml \
    && RUSTUP_TOOLCHAIN="${SPPARK_RUST_TOOLCHAIN}" cargo fetch --locked \
        --manifest-path baselines/sppark/poc/ntt-cuda/Cargo.toml \
    && xmake require --yes doctest

ENV CARGO_NET_OFFLINE=true

CMD ["bash"]
