# syntax=docker/dockerfile:1
#
# Runtime image: just the three CLI binaries and the ONNX models they need by default. Not to
# be confused with .devcontainer/Dockerfile, which is a development image (editor tooling, test
# frameworks, a Python analysis venv) and carries none of the minimalism this one does.
#
# Build both the CPU and GPU variants (the default): docker buildx bake
# Build just one:                                     docker buildx bake cpu   (or gpu)
#
# Compiling against ONNX Runtime only needs its C API header and shared library - both come
# from the same prebuilt release tarball either way, and CUDA itself is never touched until the
# GPU tarball's .so actually runs. So a single builder stage suffices for both variants; only
# which ONNX Runtime asset it fetches differs. Only the final runtime stage's base image needs
# to carry the CUDA/cuDNN runtime for the GPU variant.

# Global: available to every FROM below. Each stage that needs it after its own FROM
# re-declares it with a bare `ARG` line, per Docker's build-arg scoping rules.
ARG VARIANT=cpu
ARG HTSLIB_VERSION=1.24
ARG ONNXRUNTIME_VERSION=1.28.0

FROM ubuntu:24.04 AS builder
ARG VARIANT
ARG HTSLIB_VERSION
ARG ONNXRUNTIME_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# htslib (VCF/BAM/FASTA access), same as .devcontainer/Dockerfile.
RUN curl -fsSL "https://github.com/samtools/htslib/releases/download/${HTSLIB_VERSION}/htslib-${HTSLIB_VERSION}.tar.bz2" \
        -o /tmp/htslib.tar.bz2 \
    && tar -xjf /tmp/htslib.tar.bz2 -C /tmp \
    && cd "/tmp/htslib-${HTSLIB_VERSION}" \
    && ./configure \
    && make -j"$(nproc)" \
    && make install \
    && cd / \
    && rm -rf /tmp/htslib.tar.bz2 "/tmp/htslib-${HTSLIB_VERSION}"

# ONNX Runtime, extracted to /opt/onnxruntime. CPU or GPU-CUDA12 asset per VARIANT - same
# selection logic as .devcontainer/Dockerfile and scripts/install_onnxruntime.sh.
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
    && printf 'prefix=/opt/onnxruntime\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n\nName: onnxruntime\nDescription: ONNX Runtime\nVersion: %s\nLibs: -L${libdir} -lonnxruntime\nCflags: -I${includedir}\n' "${ONNXRUNTIME_VERSION}" > /usr/local/lib/pkgconfig/onnxruntime.pc

ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

WORKDIR /src
COPY . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPLICEAI_INFERENCE_BACKEND=onnxruntime \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build --prefix /out

# --- runtime bases: same images .devcontainer/Dockerfile uses for VARIANT=cpu|gpu ---
FROM ubuntu:24.04 AS runtime-cpu
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime-gpu

FROM runtime-${VARIANT} AS runtime
ARG VARIANT

# Both base images ship a default "ubuntu" user at 1000:1000 (as .devcontainer/Dockerfile also
# has to account for) - remove it first so it doesn't collide with the one created here.
RUN if getent passwd ubuntu > /dev/null; then userdel -r ubuntu; fi \
    && if getent group ubuntu > /dev/null; then groupdel ubuntu; fi \
    && groupadd --gid 1000 cpliceai \
    && useradd --uid 1000 --gid 1000 --shell /usr/sbin/nologin --create-home cpliceai

# htslib and ONNX Runtime shared libraries. Everything else the binaries link against
# (liblzma, libbz2, libz, libstdc++, libc/libm/libpthread) already ships in these base images.
COPY --from=builder /usr/local/lib/libhts.so* /usr/local/lib/
COPY --from=builder /opt/onnxruntime /opt/onnxruntime
COPY --from=builder /out/bin/ /usr/local/bin/

RUN echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
    && ldconfig

# The only bundled data: the ONNX models. Everything else a run needs - the reference
# FASTA, the regions file, the variants file - is supplied by the caller via a bind mount, so it
# lives under /opt/cpliceai rather than at /workspace below, where it would be shadowed by the
# caller's own mount.
COPY models/onnx /opt/cpliceai/models/onnx
COPY models/onnx_fp16 /opt/cpliceai/models/onnx_fp16

RUN mkdir -p /workspace && chown cpliceai:cpliceai /workspace

ENV CPLICEAI_ORT_EP=auto

# Add uv for python (standalone static binary)
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /usr/local/bin/
COPY scripts/parse_gene_regions_from_gff.py /utils/parse_gene_regions_from_gff.py

USER cpliceai
WORKDIR /home/cpliceai
RUN uv venv && uv pip install polars pandas pyarrow tqdm gff3_parser 

ENV VIRTUAL_ENV=/home/cpliceai/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY README.docker.md /README.md

WORKDIR /workspace


