# CpliceAI

## Usage

CpliceAI scores how a variant affects splicing. There are three binaries. `cpliceai_reference`
scores the unaltered genome once and saves the result; the two predict binaries then compare
your variants against it. Use `cpliceai_predict_variant` for a score per variant, and
`cpliceai_predict_gene` for a score at every position of the gene a variant falls in.

```
cpliceai_reference        <model_dir> <fasta> <regions> <output.bin>

cpliceai_predict_variant  <variants> <reference_scores> <model_dir> <fasta> <regions> <output> \
                          [--window-radius N] [--input-format vcf|tsv|auto]

cpliceai_predict_gene     <variants> <reference_scores> <model_dir> <fasta> <regions> <output> \
                          [--input-format vcf|tsv|auto]
```

### Example

```sh
# 1. Score the reference. Slow, but done once per genome + annotation.
#    Rebuild this whenever you change the regions file.
cpliceai_reference models/onnx GRCh37.fa data/grch37.tsv reference.bin

# 2. Annotate a VCF.
cpliceai_predict_variant variants.vcf reference.bin models/onnx GRCh37.fa \
                         data/grch37.tsv annotated.vcf

# 3. Or a TSV, which comes back as a TSV.
cpliceai_predict_variant variants.tsv reference.bin models/onnx GRCh37.fa \
                         data/grch37.tsv annotated.tsv
```

Pass the same `<fasta>` and `<regions>` to all three. The reference scores are only meaningful
for the assembly and gene set they were computed from, so `reference.bin` records a fingerprint
of both and the predict binaries refuse to run against a mismatch:

```
ERROR The gene regions file does not match the one reference.bin was built from
      (gene names, coordinates or strands differ). Rebuild with cpliceai_reference,
      or pass the original regions file.
```

The fingerprints are taken over content, not file bytes — the gene set's names, coordinates and
strands, and the fasta's contig names and lengths. Recompressing an annotation, adding or
removing its header row, or reformatting it will not trigger a rebuild. Note the fasta side
identifies the *assembly*; it will not distinguish two builds that differ only in soft-masking.

### Input

Both predict binaries read either a VCF/BCF or a TSV, plain or compressed:

```
CHROM	POS	REF	ALT
chrTest	1000	G	A
chrTest	1500	C	G,GT
```

`POS` is 1-based, `ALT` may list comma-separated alleles, and extra columns are ignored. The
header row is optional. Genes are taken from the regions file, so variants need no gene column.

The format is detected from the file. If a VCF is not recognised, pass `--input-format vcf`.

### Output

`cpliceai_predict_variant` returns your input with the scores added, in the format you supplied:
a VCF gains an `INFO/SpliceAI` field, a TSV gains a `SpliceAI` column. Everything else in the
file is left alone, so the output can be fed into another run.

```
CHROM	POS	REF	ALT	SpliceAI
chrTest	1000	G	A	A|GENE1|0.12|0.00|0.03|0.41|-2|-8|21|-5
```

Each entry reads `ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL`: four delta
scores (acceptor gain/loss, donor gain/loss) and the position of each, relative to the variant.
There is one entry per alternate allele per gene, comma-separated, and `.` where a variant was
skipped. `--window-radius` sets how far either side of the variant is scored (default 500).

A variant is scored against a gene only if it falls **entirely** inside it. A deletion anchored
near the end of a gene but reaching past it is reported as `.`, since there is no reference
sequence beyond the boundary to compare the alternate against. Variants longer than
`--window-radius` are reported the same way.

`cpliceai_predict_gene` writes a score table instead, one block per variant, listing every
position in the gene where the reference or the alternate crosses a low threshold:

```
#GENE1_+_0_2000:chrTest_1000_G_A
112	0.000000	0.630000	0.000000	0.620000
679	0.340000	0.000000	0.380000	0.000000
```

The block header names the gene, strand and span, then the variant. Each row is a position
followed by its reference acceptor and donor scores, then the same two for the alternate.

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
| `CPLICEAI_ORT_PROFILE` | unset | Path prefix; enables ORT's per-node profiler and writes one JSON per ensemble member (`<prefix>_model<N>_<timestamp>.json`). Summarise with `scripts/profile_summary.sh` |

#### GPU performance A/B knobs (CUDA EP)

Forwarded to ORT only when set, so leaving them unset keeps ONNX Runtime's own defaults. Added
for the investigation in `docs/gpu-validation.md`; none has a proven-best value for this model yet.

| Variable | ORT provider option | Purpose |
|---|---|---|
| `CPLICEAI_ORT_PREFER_NHWC` | `prefer_nhwc` | `1` makes the CUDA EP prefer NHWC kernels, applying layout transforms automatically. NVIDIA tensor cores favour NHWC, but ORT warns this can *add* transposes where NHWC operator coverage is incomplete -- measure, don't assume. Requires ORT >= 1.20 |
| `CPLICEAI_ORT_USE_TF32` | `use_tf32` | `0` disables TF32, dropping convolutions to true fp32 FMA math. Useful as a probe for whether tensor cores are engaged at all at this model's 32-channel width |
| `CPLICEAI_ORT_CONV1D_PAD_NC1D` | `cudnn_conv1d_pad_to_nc1d` | Controls how 1D convolutions are mapped onto cuDNN. Every conv in this model is 1D, so it is plausibly relevant |

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

`models/tf/spliceai1..5/` are the original TensorFlow SavedModels (source of truth).
`models/onnx/` and `models/onnx_fp16/` are converted from them via
`scripts/onnx/convert_models.py`. FP16 is GPU-only in practice -- CPU EP has partial fp16 kernel
coverage and no throughput benefit from it; point `MODEL_DIR` at `models/onnx_fp16` only when
running with `CPLICEAI_ORT_EP=cuda`.

### The ONNX models are not a plain tf2onnx conversion

`convert_models.py` rewrites the graph after converting it, and **the rewrite is hardcoded to
this architecture**. Anyone regenerating the models needs to know why, and what that constrains.

**The problem.** Keras `Conv1D` has no native form in the traced SavedModel graph. TensorFlow
lowers each of the 39 conv layers to
`[SpaceToBatchND ->] ExpandDims(x2) -> Conv2D(NHWC) -> Squeeze [-> BatchToSpaceND] -> BiasAdd`,
with 24 of the 39 wrapped in `SpaceToBatchND`/`BatchToSpaceND` -- TF's way of emulating a
dilated convolution with a dilation-1 `Conv2D`. ONNX's `Conv` has no channels-last mode, so
`tf2onnx` has to transpose around every one of them: **~200 `Transpose` nodes per model, where
~2 would be expected.**

`onnx-simplifier` does not help (measured: zero reduction). Its passes need concrete shapes to
prove a fusion safe, and the sequence-length axis must stay dynamic because every call site
feeds a different length.

**The fix.** `simplify_dilated_convs` replaces each of the 39 layers with a single native ONNX
`Conv`, which supports dilation and `same` padding directly. Transposes drop to ~53: one global
`NWC->NCW` on the input, one pre-existing skip-connection transpose, and 51 from a layout quirk
-- each dilated layer's `SpaceToBatchND`/`BatchToSpaceND` round-trip flips its own output to
NWC while plain layers stay NCW, so each of the 24 needs a matching `NCW->NWC` on its output to
leave the downstream BatchNorm/Relu/skip-Add untouched. The graph deliberately keeps TensorFlow's
original layout rather than ONNX Runtime's preferred one.

**The constraint.** The pass carries a 39-entry table of kernel width, dilation rate and true
input tensor name, extracted from the traced `GraphDef` rather than assumed. It is
architecture-specific by construction and will not adapt: **changing the model architecture
means updating that table before re-running conversion.** It is valid for all five committed
models only because they share one architecture and differ solely in weights.

**What is checked.** Opset 17; TF-vs-ONNX(fp32) agreement asserted at sequence lengths 15000 and
40000 with max absolute difference below `1e-5`; toolchain pinned in
`scripts/onnx/requirements.txt` (`tf2onnx==1.17.0`, `onnx==1.17.0`, `onnxruntime==1.28.0`). The
fp32-vs-fp16 comparison is reported but not asserted.

## Testing

End-to-end tests for `cpliceai_reference`, `cpliceai_predict_variant`, and `cpliceai_predict_gene`
live under `tests/` as bats-core scripts, wired into CTest and a `check` target:

```
cmake -S . -B build
cmake --build build -j
cmake --build build --target check   # equivalent to: cd build && make check
```

`ctest --test-dir build -L cli` runs just the fast argument-parsing tests; `-L e2e` runs the full
pipeline test plus `tests/cross_format.bats` (which asserts the same variant scores identically
whether it arrives as VCF or TSV). Both load the real SpliceAI models against a small synthetic
fixture (`tests/fixtures/`) and take tens of seconds.

Requires `bats-core` on `PATH`; installed in `.devcontainer/Dockerfile`, or install it yourself
from https://github.com/bats-core/bats-core. The test helpers only use `setup`/`teardown` and
`run` (no `setup_file`/`BATS_FILE_TMPDIR`), so any reasonably recent bats-core works.
