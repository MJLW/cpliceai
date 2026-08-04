FROM ubuntu:24.04

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# htslib and the TensorFlow C API are pinned to explicit release versions
# and installed from upstream release archives (not apt) so the toolchain
# doesn't drift as distro/package versions change underneath the project.
ARG HTSLIB_VERSION=1.24
# TensorFlow stopped publishing prebuilt libtensorflow archives after
# 2.18.1 (2.19+ only ships inside the much larger Python pip wheel), so
# this is the newest version available as a standalone C API archive.
ARG TENSORFLOW_VERSION=2.18.1
# bats-core runs the end-to-end test suite under tests/ (see CMakeLists.txt's
# `check` target). Installed from its release tarball rather than apt, for
# the same reproducibility reason as htslib/TensorFlow above.
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

# htslib (VCF/BAM/FASTA access), providing htslib.pc for the
# pkg_check_modules(HTSLIB ...) call in CMakeLists.txt.
RUN curl -fsSL "https://github.com/samtools/htslib/releases/download/${HTSLIB_VERSION}/htslib-${HTSLIB_VERSION}.tar.bz2" \
        -o /tmp/htslib.tar.bz2 \
    && tar -xjf /tmp/htslib.tar.bz2 -C /tmp \
    && cd "/tmp/htslib-${HTSLIB_VERSION}" \
    && ./configure \
    && make -j"$(nproc)" \
    && make install \
    && cd / \
    && rm -rf /tmp/htslib.tar.bz2 "/tmp/htslib-${HTSLIB_VERSION}"

# TensorFlow C API (CPU-only), extracted to /usr/local as documented at
# https://www.tensorflow.org/install/lang_c. The archive ships no .pc file,
# so one is written by hand to satisfy CMakeLists.txt's
# pkg_check_modules(TENSORFLOW REQUIRED tensorflow).
RUN curl -fsSL "https://storage.googleapis.com/tensorflow/versions/${TENSORFLOW_VERSION}/libtensorflow-cpu-linux-x86_64.tar.gz" \
        -o /tmp/libtensorflow.tar.gz \
    && tar -C /usr/local -xzf /tmp/libtensorflow.tar.gz \
    && rm /tmp/libtensorflow.tar.gz \
    && mkdir -p /usr/local/lib/pkgconfig \
    && printf 'prefix=/usr/local\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n\nName: TensorFlow\nDescription: Library for computation using data flow graphs for scalable machine learning\nVersion: %s\nLibs: -L${libdir} -ltensorflow -ltensorflow_framework\nCflags: -I${includedir}\n' "${TENSORFLOW_VERSION}" > /usr/local/lib/pkgconfig/tensorflow.pc

# Register /usr/local/lib with the dynamic linker so libtensorflow.so and
# libhts.so are found at runtime without exporting LD_LIBRARY_PATH by hand.
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig

# bats-core (pure bash + a small C helper), for `make check` / ctest.
RUN curl -fsSL "https://github.com/bats-core/bats-core/archive/refs/tags/v${BATS_CORE_VERSION}.tar.gz" \
        -o /tmp/bats-core.tar.gz \
    && tar -xzf /tmp/bats-core.tar.gz -C /tmp \
    && bash "/tmp/bats-core-${BATS_CORE_VERSION}/install.sh" /usr/local \
    && rm -rf /tmp/bats-core.tar.gz "/tmp/bats-core-${BATS_CORE_VERSION}"

# pkg-config on some distros doesn't search /usr/local by default.
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig"

# Non-root user with a UID/GID matching your host user, so files touched in
# the container aren't left owned by root when you edit them outside it.
# devcontainer.json's "updateRemoteUserUID" will nudge this UID to match
# your actual host user automatically when the container is created on Linux.
#
# ubuntu:24.04 ships with a built-in "ubuntu" user/group at 1000:1000, which
# collides with the common default (and with most single-user Linux hosts),
# so it's removed first rather than picking a different id and risking the
# same collision later.
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
