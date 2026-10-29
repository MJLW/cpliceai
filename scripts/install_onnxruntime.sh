#!/usr/bin/env bash
# Installs an ONNX Runtime release tarball to /opt/onnxruntime, for machines not using Docker
# (see .devcontainer/Dockerfile's --build-arg VARIANT=cpu|gpu for that path instead).
# The cuda12/cuda13 variants require CUDA + cuDNN already installed, matching the variant chosen.
#
# Usage:
#   ./scripts/install_onnxruntime.sh              # CPU (default)
#   ./scripts/install_onnxruntime.sh cpu           # CPU, explicit
#   ./scripts/install_onnxruntime.sh cuda12        # CUDA 12.8 + cuDNN 9.x
#   ./scripts/install_onnxruntime.sh cuda13        # CUDA 13.0 + cuDNN 9.x

set -euo pipefail

VERSION=1.28.0
VARIANT="${1:-cpu}"
case "$VARIANT" in
    cpu)    ASSET_SUFFIX="" ;;
    cuda12) ASSET_SUFFIX="-gpu_cuda12" ;;
    cuda13) ASSET_SUFFIX="-gpu_cuda13" ;;
    *) echo "Usage: $0 [cpu|cuda12|cuda13]" >&2; exit 1 ;;
esac

PREFIX=/opt/onnxruntime
TARBALL="onnxruntime-linux-x64${ASSET_SUFFIX}-${VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${TARBALL}"

echo "Installing ONNX Runtime ${VERSION} (${VARIANT}) to ${PREFIX}..."

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

curl -fsSL "$URL" -o "$TMPDIR/onnxruntime.tgz"
sudo mkdir -p "$PREFIX"
sudo tar -C "$PREFIX" --strip-components=1 -xzf "$TMPDIR/onnxruntime.tgz"

sudo mkdir -p /usr/local/lib/pkgconfig
sudo tee /usr/local/lib/pkgconfig/onnxruntime.pc > /dev/null <<EOF
prefix=/opt/onnxruntime
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: onnxruntime
Description: ONNX Runtime
Version: ${VERSION}
Libs: -L\${libdir} -lonnxruntime
Cflags: -I\${includedir}
EOF

echo "/opt/onnxruntime/lib" | sudo tee /etc/ld.so.conf.d/onnxruntime.conf > /dev/null
sudo ldconfig

echo "Done. Verify with: pkg-config --libs --cflags onnxruntime"
echo "Then build with: cmake -S . -B build -DCPLICEAI_INFERENCE_BACKEND=onnxruntime"

if [ "$VARIANT" != "cpu" ]; then
    echo
    echo "Requires CUDA 12.8 + cuDNN 9.x already installed (for cuda12) or CUDA 13.0 + cuDNN 9.x (for cuda13)."
    echo "Check with: nvcc --version"
fi
