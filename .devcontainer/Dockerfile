# CPU-only by default. --build-arg VARIANT=gpu selects a CUDA 12.8 + cuDNN 9.x base image
# and GPU-enabled ONNX Runtime together, so the two can't drift out of sync.
ARG VARIANT=cpu

FROM ubuntu:24.04 AS base-cpu
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base-gpu

FROM base-${VARIANT}
ARG VARIANT

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# htslib, the TensorFlow C API, ONNX Runtime and bats-core are pinned to explicit release
# versions and installed from upstream archives (not apt) to avoid version drift.
ARG HTSLIB_VERSION=1.24
# TF stopped publishing prebuilt libtensorflow archives after 2.18.1. Kept installed
# alongside ONNX Runtime for the `tensorflow` backend used in parity testing.
ARG TENSORFLOW_VERSION=2.18.1
# ONNX Runtime is the default inference backend; the release asset (CPU vs GPU-CUDA-12.8)
# is picked from VARIANT below.
ARG ONNXRUNTIME_VERSION=1.28.0
ARG BATS_CORE_VERSION=1.14.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    git \
    curl \
    ca-certificates \
    vim \
    tmux \
    sudo \
    bcftools \
    build-essential \
    cmake \
    pkg-config \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# htslib (VCF/BAM/FASTA access), providing htslib.pc for CMakeLists.txt.
RUN curl -fsSL "https://github.com/samtools/htslib/releases/download/${HTSLIB_VERSION}/htslib-${HTSLIB_VERSION}.tar.bz2" \
        -o /tmp/htslib.tar.bz2 \
    && tar -xjf /tmp/htslib.tar.bz2 -C /tmp \
    && cd "/tmp/htslib-${HTSLIB_VERSION}" \
    && ./configure \
    && make -j"$(nproc)" \
    && make install \
    && cd / \
    && rm -rf /tmp/htslib.tar.bz2 "/tmp/htslib-${HTSLIB_VERSION}"

# TensorFlow C API (CPU-only). Ships no .pc file, so one is written by hand.
RUN curl -fsSL "https://storage.googleapis.com/tensorflow/versions/${TENSORFLOW_VERSION}/libtensorflow-cpu-linux-x86_64.tar.gz" \
        -o /tmp/libtensorflow.tar.gz \
    && tar -C /usr/local -xzf /tmp/libtensorflow.tar.gz \
    && rm /tmp/libtensorflow.tar.gz \
    && mkdir -p /usr/local/lib/pkgconfig \
    && printf 'prefix=/usr/local\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n\nName: TensorFlow\nDescription: Library for computation using data flow graphs for scalable machine learning\nVersion: %s\nLibs: -L${libdir} -ltensorflow -ltensorflow_framework\nCflags: -I${includedir}\n' "${TENSORFLOW_VERSION}" > /usr/local/lib/pkgconfig/tensorflow.pc

# ONNX Runtime, extracted to /opt/onnxruntime. Also ships no .pc matching this layout, so
# one is written by hand.
RUN if [ "$VARIANT" = "gpu" ]; then \
        ONNXRUNTIME_ASSET="onnxruntime-linux-x64-gpu_cuda12-${ONNXRUNTIME_VERSION}.tgz"; \
    else \
        ONNXRUNTIME_ASSET="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"; \
    fi \
    && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ASSET}" \
        -o /tmp/onnxruntime.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -C /opt/onnxruntime --strip-components=1 -xzf /tmp/onnxruntime.tgz \
    && rm /tmp/onnxruntime.tgz \
    && mkdir -p /usr/local/lib/pkgconfig \
    && printf 'prefix=/opt/onnxruntime\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n\nName: onnxruntime\nDescription: ONNX Runtime\nVersion: %s\nLibs: -L${libdir} -lonnxruntime\nCflags: -I${includedir}\n' "${ONNXRUNTIME_VERSION}" > /usr/local/lib/pkgconfig/onnxruntime.pc \
    && echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig

# bats-core, for `make check` / ctest.
RUN curl -fsSL "https://github.com/bats-core/bats-core/archive/refs/tags/v${BATS_CORE_VERSION}.tar.gz" \
        -o /tmp/bats-core.tar.gz \
    && tar -xzf /tmp/bats-core.tar.gz -C /tmp \
    && bash "/tmp/bats-core-${BATS_CORE_VERSION}/install.sh" /usr/local \
    && rm -rf /tmp/bats-core.tar.gz "/tmp/bats-core-${BATS_CORE_VERSION}"

# pkg-config on some distros doesn't search /usr/local by default.
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig"

# Non-root user matching the host UID/GID. The base image's built-in "ubuntu" user (1000:1000)
# is removed first to avoid colliding with the common default.
RUN if getent passwd ubuntu > /dev/null; then userdel -r ubuntu; fi \
    && if getent group ubuntu > /dev/null; then groupdel ubuntu; fi \
    && if getent passwd $USER_UID > /dev/null; then userdel -r $(getent passwd $USER_UID | cut -d: -f1); fi \
    && if getent group $USER_GID > /dev/null; then groupdel $(getent group $USER_GID | cut -d: -f1); fi \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /workspaces

ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash
