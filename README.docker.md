# CpliceAI

CpliceAI is a rewrite of SpliceAI that allows for faster predictions (more than 10x), the output of raw scores for entire genes, and background haplotype support. It consists of three binary executables:

- **`cpliceai_reference`** — Creates a condensed file of sparse reference scores that should be used with `cpliceai_predict_variant` and `cpliceai_predict_gene`. Precomputing reference scores reduces repetition when making the actual predictions.
- **`cpliceai_predict_variant`** — Executes the rewritten SpliceAI functionality; it predicts N bp around each variant and gives the largest donor gain, acceptor gain, donor loss, and acceptor loss scores within the window. Defaults to a 500 bp window and attempts to build haplotypes (see available parameters to override).
- **`cpliceai_predict_gene`** — Provides sparsely encoded scores for the entire gene a variant is located on, and attempts to build haplotypes by default (see available parameters to override).

Predictions can be made on VCFs/TSVs and gene region files, the latter of which can be created from GFFs using `parse_gene_regions.py`. When using a TSV as input, the expected format is:

```
CHROM	POS	REF	ALT	GT
```

- `CHROM` — the contig field
- `POS` — the variant position
- `REF` — the variant reference sequence
- `ALT` — the variant alternate sequence
- `GT` — the phased GT tag (not necessary when running with `--local`)

## Running CpliceAI

The easiest way to run CpliceAI is with Docker or Singularity. If neither can work for you, please get in touch directly, as building from source is not currently streamlined.

### Docker

**`cpliceai_reference`**

```bash
# cpliceai_reference <models_dir> <fa> <gene_regions> <reference binary output>
docker run --rm -v "$PWD":/workspace jhampstead/cpliceai_cpu cpliceai_reference /opt/cpliceai/models/onnx grch37.fa grch37.tsv grch37.bin
```

For more information on other optional arguments, run without arguments or with `-h`:

```bash
docker run --rm jhampstead/cpliceai_cpu cpliceai_reference -h
```

**`cpliceai_predict_variant`**

```bash
# cpliceai_predict_variant <variants> <reference_scores> <models_dir> <fa> <gene_regions> <annotated variants output>

# VCF
docker run --rm -v "$PWD":/workspace jhampstead/cpliceai_cpu cpliceai_predict_variant input.vcf grch37.bin /opt/cpliceai/models/onnx grch37.fa grch37.tsv output.vcf

# TSV
docker run --rm -v "$PWD":/workspace jhampstead/cpliceai_cpu cpliceai_predict_variant input.tsv grch37.bin /opt/cpliceai/models/onnx grch37.fa grch37.tsv output.tsv
```

For more information on other optional arguments, run without arguments or with `-h`:

```bash
docker run --rm jhampstead/cpliceai_cpu cpliceai_predict_variant -h
```

**`cpliceai_predict_gene`**

```bash
# cpliceai_predict_variant <variants> <reference_scores> <models_dir> <fa> <gene_regions> <annotated variants output>

# VCF
docker run --rm -v "$PWD":/workspace jhampstead/cpliceai_cpu cpliceai_predict_gene input.vcf grch37.bin /opt/cpliceai/models/onnx grch37.fa grch37.tsv output.tsv

# TSV
docker run --rm -v "$PWD":/workspace jhampstead/cpliceai_cpu cpliceai_predict_gene input.tsv grch37.bin /opt/cpliceai/models/onnx grch37.fa grch37.tsv output.tsv
```

For more information on other optional arguments, run without arguments or with `-h`:

```bash
docker run --rm jhampstead/cpliceai_cpu cpliceai_predict_gene -h
```

### Singularity

Before running with Singularity, containers must first be converted to a Singularity container file:

```bash
# If executed on an HPC, Singularity will write to tmp or cache. It shouldn't be
# necessary, but some systems are more fragile than others — best to be safe
# and define your own.
export SINGULARITY_TMPDIR=/path/to/scratch/tmp
export SINGULARITY_CACHEDIR=/path/to/scratch/cache

# Creates the cpliceai_cpu.sif container file
singularity pull cpliceai_cpu.sif docker://jhampstead/cpliceai_cpu
```

**`cpliceai_reference`**

```bash
# cpliceai_reference <models_dir> <fa> <gene_regions> <reference binary output>
singularity exec cpliceai_cpu.sif cpliceai_reference /opt/cpliceai/models/onnx grch37.fa grch37.tsv grch37.bin
```

For more information on other optional arguments, run without arguments or with `-h`:

```bash
singularity exec cpliceai_cpu.sif cpliceai_reference -h
```

**`cpliceai_predict_variant`**

```bash
# cpliceai_predict_variant <variants> <reference_scores> <models_dir> <fa> <gene_regions> <annotated variants output>

# VCF
singularity exec cpliceai_cpu.sif cpliceai_predict_variant input.vcf grch37.bin /opt/cpliceai/models/onnx grch37.fa grch37.tsv output.vcf

# TSV
singularity exec cpliceai_cpu.sif cpliceai_predict_variant input.tsv grch37.bin /opt/cpliceai/models/onnx grch37.fa grch37.tsv output.tsv
```

For more information on other optional arguments, run without arguments or with `-h`:

```bash
singularity exec cpliceai_cpu.sif cpliceai_predict_variant -h
```

**`cpliceai_predict_gene`**

```bash
# cpliceai_predict_variant <variants> <reference_scores> <models_dir> <fa> <gene_regions> <annotated variants output>

# VCF
singularity exec cpliceai_cpu.sif cpliceai_predict_gene input.vcf grch37.bin /opt/cpliceai/models/onnx grch37.fa grch37.tsv output.tsv

# TSV
singularity exec cpliceai_cpu.sif cpliceai_predict_gene input.tsv grch37.bin /opt/cpliceai/models/onnx grch37.fa grch37.tsv output.tsv
```

For more information on other optional arguments, run without arguments or with `-h`:

```bash
singularity exec cpliceai_cpu.sif cpliceai_predict_gene -h
```

> **Note:** the TSV example lines in this section were added for parity with the Docker section above, and the `# cpliceai_predict_variant` comment heading the `cpliceai_predict_gene` code block (both here and in the Docker section) appears to be a copy-paste slip carried over from the `cpliceai_predict_variant` block.

### Running on GPU

Assuming a properly configured GPU (CUDA), the same commands can be run with a slight modification to the Docker/Singularity wrapper:

```bash
# Docker
docker run --rm -v "$PWD":/workspace --gpus all jhampstead/cpliceai_gpu <command>

# Singularity, using the CUDA toolkit
singularity pull cpliceai_gpu.sif docker://jhampstead/cpliceai_gpu
singularity exec --nv cpliceai_gpu.sif <command>
```

## Required files

Running CpliceAI requires:

- A FASTA file
- A GFF file (see [Converting from GFF file to a gene regions TSV](#converting-from-gff-file-to-a-gene-regions-tsv))
- Variants (VCF or TSV)

### Converting from GFF file to a gene regions TSV

Use the provided script to convert a GFF file into a gene regions TSV:

```bash
# Docker
docker run --rm -v "$PWD":/workspace jhampstead/cpliceai_cpu uv run /utils/parse_gene_regions_from_gff.py -g grch37.gff -o grch37.tsv

# Singularity
singularity exec cpliceai_cpu.sif uv run /utils/parse_gene_regions_from_gff.py -g grch37.gff -o grch37.tsv
```
