# CpliceAI

## Inference backend

The default inference backend is ONNX Runtime (`CPLICEAI_INFERENCE_BACKEND=onnxruntime` in
`CMakeLists.txt`), which is what makes GPU execution possible: TensorFlow no longer publishes a
GPU-enabled C API tarball for the version this project is pinned to. A `tensorflow` backend is
kept buildable alongside it (`-DCPLICEAI_INFERENCE_BACKEND=tensorflow`) for parity testing -- see
`tests/backend_parity.bats`.

TensorFlow C API requires:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

### Runtime environment variables (ONNX Runtime backend)

| Variable | Default | Purpose |
|---|---|---|
| `CPLICEAI_ORT_EP` | `auto` | `auto` (try CUDA, fall back to CPU), `cuda` (hard-fail if unavailable), `cpu` |
| `CPLICEAI_ORT_DEVICE_ID` | `0` | Which GPU to use (multi-GPU machines) |
| `CPLICEAI_ORT_CUDNN_CONV_ALGO_SEARCH` | `HEURISTIC` | cuDNN conv algorithm search strategy. Deliberately not ORT's own default `EXHAUSTIVE`: every inference call here uses a different sequence length (no batching), so exhaustive per-shape autotuning would re-benchmark every convolution layer on every single call |
| `CPLICEAI_ORT_INTRA_OP_THREADS` | ORT default | CPU-EP thread count. Setting this to exactly `1` also pins inter-op threads to 1 and forces sequential execution mode, for fully reproducible output (used by the test suite) |
| `CPLICEAI_ORT_LOG_SEVERITY` | `2` (warning) | `0`=verbose, prints per-node execution-provider placement -- useful for confirming a node didn't silently fall back to CPU |
| `CPLICEAI_ORT_MAX_CHUNK_LEN` | `250000` | Max sequence length (bases) fed to `Run()` in one call. Longer inputs (large genes) are split into overlapping windows and stitched back together -- safe because the model's receptive field is bounded by `CONTEXT_SIZE`/`BOUNDARY_SIZE`. The default is the largest length confirmed to run reliably on the CUDA execution provider without exhausting GPU memory; lower it on GPUs with less VRAM |

### Building for GPU

The default image is CPU-only. To build a GPU-capable image (CUDA 12.8 + cuDNN 9.x, on a machine
with an NVIDIA GPU, driver, and `nvidia-container-toolkit`):

```
docker build --build-arg VARIANT=gpu -f .devcontainer/Dockerfile .
```

This is ~3.2GB heavier to build/pull than the CPU image (mostly the CUDA base image), so it's
opt-in rather than the default. The CUDA/cuDNN *runtime* has to come from the base image --
`nvidia-container-toolkit` only injects the host's driver (`libcuda.so`), not `libcudart`/cuDNN.

Not using Docker at all? Install CUDA 12.8 + cuDNN 9 directly, then run
`scripts/install_onnxruntime_gpu.sh` (or unpack the GPU release tarball yourself) and set
`ONNXRUNTIME_ROOT=/opt/onnxruntime` when configuring CMake.

Once you have a GPU build, see **[`docs/gpu-validation.md`](docs/gpu-validation.md)** for the
validation checklist (confirming the CUDA EP actually engages, fp32/fp16 accuracy parity,
benchmarking against the CPU baselines already recorded there, and a known perf caveat worth
reading before drawing conclusions from the numbers).

### Model formats

`models/spliceai1..5/` are the original TensorFlow SavedModels (source of truth). `models/onnx/`
and `models/onnx_fp16/` are converted from them via `scripts/onnx/convert_models.py` (see that
script for the pinned conversion toolchain and its numeric-parity checks against the TF originals).
FP16 is GPU-only in practice -- CPU EP has partial fp16 kernel coverage and no throughput benefit
from it; point `MODEL_DIR` at `models/onnx_fp16` only when running with `CPLICEAI_ORT_EP=cuda`.

## Testing

End-to-end tests for `cpliceai_reference`, `cpliceai_predict_variant`, and `cpliceai_predict_gene`
live under `tests/` as bats-core scripts, wired into CTest and a `check` target:

```
cmake -S . -B build
cmake --build build -j
cmake --build build --target check   # equivalent to: cd build && make check
```

`ctest --test-dir build -L cli` runs just the fast argument-parsing tests; `-L e2e` runs the full
pipeline test, which loads the real SpliceAI models against a small synthetic fixture
(`tests/fixtures/`) and takes tens of seconds.

Requires `bats-core` on `PATH`; installed in `.devcontainer/Dockerfile`, or install it yourself
from https://github.com/bats-core/bats-core. The test helpers only use `setup`/`teardown` and
`run` (no `setup_file`/`BATS_FILE_TMPDIR`), so any reasonably recent bats-core works.
