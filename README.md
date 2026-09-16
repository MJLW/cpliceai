# CpliceAI

CpliceAI is a C implementation of SpliceAI (splice-effect prediction from sequence), running on
ONNX Runtime with CPU and GPU inference, plus haplotype-aware scoring for phased variants.

## Installation

### Docker / devcontainer (recommended)

CPU by default:

```sh
docker build -f .devcontainer/Dockerfile .
```

GPU (CUDA 12.8 + cuDNN 9.x; needs an NVIDIA GPU, driver, and `nvidia-container-toolkit` on the
host — see [Building for GPU](#building-for-gpu) below for details):

```sh
docker build --build-arg VARIANT=gpu -f .devcontainer/Dockerfile .
```

Opening `.devcontainer/` in VS Code (or another devcontainer-aware editor) does the same CPU build
by default; it works without a GPU.

### Runtime image

A separate `Dockerfile` at the repository root (distinct from the devcontainer above, which is
for developing CpliceAI itself) builds a minimal image: just the three CLI binaries and the
default ONNX models. Nothing else is bundled — a reference FASTA, a regions file, and a variants
file all need to be supplied at run time (see [Usage](#usage) below).

Build both a CPU and a GPU image (the default):

```sh
docker buildx bake
```

Build just one:

```sh
docker buildx bake cpu
docker buildx bake gpu
```

Run it against files in the current directory:

```sh
docker run --rm -v "$PWD":/workspace cpliceai:cpu \
    cpliceai_reference /opt/cpliceai/models/onnx GRCh37.fa data/grch37.tsv reference.bin
```

The GPU image needs the same NVIDIA GPU, driver, and `nvidia-container-toolkit` setup as the GPU
devcontainer (see [Building for GPU](#building-for-gpu)), plus `--gpus all` on `docker run`:

```sh
docker run --rm --gpus all -v "$PWD":/workspace cpliceai:gpu \
    cpliceai_predict_variant variants.vcf reference.bin /opt/cpliceai/models/onnx GRCh37.fa \
    data/grch37.tsv annotated.vcf
```

The bundled models live at `/opt/cpliceai/models/onnx` rather than under `/workspace`, since
anything bind-mounted there would otherwise hide them.

### Manual build

Prerequisites: `build-essential`, `cmake`, `pkg-config`, and:

- **htslib** 1.24 (VCF/BAM/FASTA access), built from source — there's no package-manager version
  new enough on most distros:
  ```sh
  curl -fsSL https://github.com/samtools/htslib/releases/download/1.24/htslib-1.24.tar.bz2 -o htslib.tar.bz2
  tar -xjf htslib.tar.bz2 && cd htslib-1.24
  ./configure && make -j && sudo make install && cd ..
  ```
- **ONNX Runtime** 1.28.0 — `scripts/install_onnxruntime.sh` (CPU by default; pass `cuda12` or
  `cuda13` for a GPU build, see [Building for GPU](#building-for-gpu)):
  ```sh
  ./scripts/install_onnxruntime.sh
  ```
- **bats-core**, optional, only needed for `cmake --build build --target check` (see
  [Testing](#testing)) — install from https://github.com/bats-core/bats-core, or skip it and CMake
  will configure without it.

Then build:

```sh
cmake -S . -B build
cmake --build build -j
```

This produces `build/cpliceai_reference`, `build/cpliceai_predict_variant`, and
`build/cpliceai_predict_gene`. Both htslib and ONNX Runtime install to `/usr/local`/`/opt` by
default; if `pkg-config` can't find ONNX Runtime, pass `-DONNXRUNTIME_ROOT=/opt/onnxruntime` when
configuring.

## Usage

There are three binaries. `cpliceai_reference` scores the unaltered genome once and saves the
result; the two predict binaries then compare your variants against it. Use
`cpliceai_predict_variant` for a score per variant, and `cpliceai_predict_gene` for a score at
every position of the gene a variant falls in.

```
cpliceai_reference        <model_dir> <fasta> <regions> <output.bin>

cpliceai_predict_variant  <variants> <reference_scores> <model_dir> <fasta> <regions> <output> \
                          [--window-radius N] [--input-format vcf|tsv|auto] \
                          [--include-unphased] [--local]

cpliceai_predict_gene     <variants> <reference_scores> <model_dir> <fasta> <regions> <output> \
                          [--input-format vcf|tsv|auto] [--include-unphased] \
                          [--ref-hapalt-only] [--local]
```

`cpliceai_predict_variant` also accepts `--splice-output <file>`, reserved for a not-yet-implemented
sparse per-position output; passing it currently exits with an error rather than doing anything.

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
CHROM	POS	REF	ALT	GT
chrTest	1000	G	A	0|1
chrTest	1500	C	G,GT	1|2
```

`POS` is 1-based, `ALT` may list comma-separated alleles, and extra columns are ignored. The
header row is optional. Genes are taken from the regions file, so variants need no gene column.

The format is detected from the file. If a VCF is not recognised, pass `--input-format vcf`.

`GT` is optional and says which of the sample's two copies of the chromosome each allele sits
on, which is what makes the haplotype scores below possible. A VCF supplies it as `FORMAT/GT`
instead, and must carry exactly one sample — extract the one you want with
`bcftools view -s <SAMPLE>` first. A file with no genotypes scores exactly as it always has.

**Input must be sorted by position**, with each contig's variants contiguous. Haplotypes are
assembled from a sliding window, so an out-of-order record would be silently left out of its
neighbours' backgrounds; the run fails instead. `bcftools sort` if in doubt.

### Output

`cpliceai_predict_variant` returns your input with the scores added, in the format you supplied:
a VCF gains `INFO/SpliceAI`, `INFO/SpliceAI_HAP` and `INFO/SpliceAI_TOT`; a TSV gains the same
three as columns. Everything else in the file is left alone, so the output can be fed into
another run.

```
CHROM	POS	REF	ALT	GT	SpliceAI	SpliceAI_HAP	SpliceAI_TOT
chrTest	1000	G	A	0|1	A|GENE1|0.12|0.00|0.03|0.41|-2|-8|21|-5	A|GENE1|2|0.02|0.00|0.01|0.09|-2|-8|21|-5	A|GENE1|2|0.14|0.00|0.03|0.63|-2|-8|21|-5
```

Each entry reads `ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL`: four delta
scores (acceptor gain/loss, donor gain/loss) and the position of each, relative to the variant.
The two haplotype fields carry a `HAP` after `SYMBOL` naming the copy they were computed on.
There is one entry per alternate allele per gene — and for the haplotype fields, per copy the
allele sits on — comma-separated, with `.` where a variant was skipped. `--window-radius` sets
how far either side of the variant is scored (default 500).

The three differ only in what is compared against what:

| Field | Comparison | Reads as |
|---|---|---|
| `SpliceAI` | `REF` → `ALT` | the variant on its own, against the reference genome |
| `SpliceAI_HAP` | `HAP_REF` → `HAP_ALT` | the variant against the rest of its own copy |
| `SpliceAI_TOT` | `REF` → `HAP_ALT` | that whole copy, against the reference genome |

where `HAP_REF` is the copy of the chromosome the variant sits on carrying every *other*
co-phased variant but not this one, and `HAP_ALT` is that same copy complete. So `SpliceAI` says
what the variant would do by itself, `SpliceAI_HAP` what it adds to the molecule it is really
on, and `SpliceAI_TOT` what that molecule does altogether. A variant whose neighbour has already
abolished a splice site scores high on `SpliceAI_TOT` and low on `SpliceAI_HAP`.

With no genotype in the input there is no copy to speak of: `HAP_REF` is the reference genome
and `HAP_ALT` is the variant alone, so all three fields carry the same numbers with `.` for the
`HAP`, and the two extra predictions are reused rather than recomputed — an unphased run costs
what it always did. A phased variant costs three predictions where it used to cost one.

#### Genotypes and phasing

| Genotype | Treated as |
|---|---|
| `0\|1`, `1\|0`, `1\|1` | phased; scored on the copy or copies named |
| `1/1` | on both copies — a homozygous call needs no phasing to be placed |
| `0/1`, `1/0` | phase unknown; **dropped entirely** unless `--include-unphased` |
| `0/0`, `0\|0` | sample carries no alternate allele; written through unscored |
| `./.`, `.`, absent | no genotype; scored as a lone variant, as above |
| `1` (haploid) | one copy, which is all the sample has |

`--include-unphased` scores an unphased heterozygote greedily on **both** copies, and puts it
into the background of its neighbours on both. That is a guess, which is why it is off by
default: without it, every haplotype number in the output is backed by real phasing.

`--local` ignores genotype and phasing altogether: every variant is scored on its own against
the reference genome, exactly as if the input carried no `GT` at all. Nothing is ever dropped,
no variant is ever split across copies, and no haplotype background is ever assembled — so
`SpliceAI_HAP` and `SpliceAI_TOT` would always just repeat `SpliceAI`, and are left out of the
output entirely instead: a VCF gets `INFO/SpliceAI` only, a TSV only the one extra column. The
`GT` column itself is unaffected and still round-trips in the output; only its effect on scoring
is turned off. Combining it with `--include-unphased` has no additional effect, since nothing is
dropped either way.

**Phase sets are not consulted.** Every phased variant in a gene is treated as belonging to one
pair of haplotypes. Where a gene spans more than one phase block, those blocks' orientations
were assigned independently by the phasing tool and nothing here can pair them up, so variants
from different blocks may be combined onto a copy they were never observed on together. This
matters most for short-read read-backed phasing, where blocks are a few kb and a gene routinely
spans several; chromosome-wide statistical or trio phasing is unaffected.

A variant is scored against a gene only if it falls **entirely** inside it. A deletion anchored
near the end of a gene but reaching past it is reported as `.`, since there is no reference
sequence beyond the boundary to compare the alternate against. Variants longer than
`--window-radius` are reported the same way.

`cpliceai_predict_gene` writes a score table instead, one block per variant per copy, listing
every position in the gene where any of the four sequences crosses a low threshold:

```
#GENE1_+_0_2000:chrTest_1000_G_A:HAP2
112	0.000000	0.630000	0.000000	0.620000	0.000000	0.610000	0.000000	0.590000
679	0.340000	0.000000	0.380000	0.000000	0.350000	0.000000	0.390000	0.000000
```

The block header names the gene, strand and span, then the variant, then the copy (`HAP.` when
the variant has no genotype). Each row is a position followed by an acceptor and donor score for
each of `REF`, `ALT`, `HAP_REF` and `HAP_ALT`, in that order — the same four sequences the three
`predict_variant` fields are computed from.

`--ref-hapalt-only` keeps just the `REF` and `HAP_ALT` pairs, which is what the gene actually
looks like on the reference and on this sample's copy, without the isolated-variant working:

```
#GENE1_+_0_2000:chrTest_1000_G_A:HAP2
112	0.000000	0.630000	0.000000	0.590000
```

`--local` also keeps two pairs, `REF` and `ALT` — genotype and phasing are ignored entirely (as
described above), so there is no haplotype background and `HAP_REF`/`HAP_ALT` would always just
repeat them. It takes priority over `--ref-hapalt-only` if both are passed, since under `--local`
the two would produce the same columns anyway. The copy is always `HAP.`, since nothing is ever
split across copies:

```
#GENE1_+_0_2000:chrTest_1000_G_A:HAP.
112	0.000000	0.630000	0.000000	0.020000
```

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

Forwarded to ORT only when set, so leaving them unset keeps ONNX Runtime's own defaults; none has
a proven-best value for this model yet.

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
`scripts/install_onnxruntime.sh cuda12` (`cuda13` for CUDA 13.0 + cuDNN 9.x instead; or unpack the
GPU release tarball yourself) and set `ONNXRUNTIME_ROOT=/opt/onnxruntime` when configuring CMake.

To confirm the CUDA execution provider is actually engaged, look for `active: CUDAExecutionProvider`
in the startup log printed by any of the three binaries; `CPLICEAI_ORT_LOG_SEVERITY=0` (see the
environment variable table above) goes further and prints per-node execution-provider placement.

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
live under `tests/` as bats-core scripts, wired into CTest and a `check` target. After the
`cmake -S . -B build && cmake --build build -j` from [Installation](#manual-build):

```
cmake --build build --target check   # equivalent to: cd build && make check
```

`ctest --test-dir build -L cli` runs just the fast argument-parsing tests; `-L e2e` runs the full
pipeline test plus `tests/input_formats.bats` (which asserts the same variant scores identically
whether it arrives as VCF or TSV) and `tests/haplotype.bats` (phased genotypes, across both
binaries and both formats). Both labels load the real SpliceAI models against a small synthetic
fixture (`tests/fixtures/`) and take tens of seconds.

Requires `bats-core` on `PATH` (see [Installation](#manual-build)). The test helpers only use
`setup`/`teardown` and `run` (no `setup_file`/`BATS_FILE_TMPDIR`), so any reasonably recent
bats-core works.
