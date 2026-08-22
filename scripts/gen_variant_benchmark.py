#!/usr/bin/env python3
"""Generate synthetic variant files for benchmarking cpliceai_predict_gene / _predict_variant.

Motivation: on the GPU, a large share of runtime is per-*shape* setup cost -- ONNX Runtime pays a
substantial penalty (~58ms on an RTX 3060) every time the input sequence length differs from the
previous call, and effectively caches only the most recent shape. See docs/gpu-validation.md.

That makes indels interesting. In predict_gene.c the alt sequence is rebuilt per variant by
create_alt_seq() (src/utils.c:60), whose length is `ref_seq->l + alt_len - ref_len`, and the model
input is `alt.l + CONTEXT_SIZE` (predict_gene.c:135). So:

  * SNVs   (alt_len == ref_len) -> input width constant for a given gene -> shapes repeat
  * indels (alt_len != ref_len) -> input width shifts by the indel size -> new shape, and the
                                   following SNV shifts it back, costing two shape changes

Generating one SNV-only file and one with a fixed indel rate isolates exactly that effect.

Constraint honoured here: predict_variant.c:112 skips any allele longer than --window-size
(default 500) with an "Oversized indel" warning, which would silently remove indels from the
benchmark. Indel lengths are kept well under that.

Requires pysam, which reads the reference through its .fai/.gzi index -- plain or bgzip-compressed
FASTA both work, and nothing is held in memory:

    pip install pysam
    samtools faidx REF.fa.gz        # if the index is missing

Usage:
    scripts/gen_variant_benchmark.py REF.fa.gz REGIONS.tsv OUT --indel-period 0
    scripts/gen_variant_benchmark.py REF.fa.gz REGIONS.tsv OUT --indel-period 10

    --format vcf   (default) VCF
    --format tsv             plain CHROM/POS/REF/ALT TSV

Both predict binaries read either format, so the choice only affects what is being benchmarked
(VCF parsing vs TSV parsing), not which binary the file can be fed to.

--gt {none,phased} controls whether a genotype is attached (none, the default, is the original
behaviour: no GT at all, which is what makes a run "naive" -- no HAP_REF/HAP_ALT recomputation,
see README.md). `phased` assigns each variant a phased genotype (mostly 0|1/1|0 heterozygous, some
1|1 homozygous), which is what makes a run "haplotype": every gene with more than one variant gets
a real per-copy background to assemble.

The genotype is drawn from a *second*, independent RNG stream seeded off `--seed`, so it never
consumes from the stream that picks positions/SNV-vs-indel/alleles. That means the same invocation
with `--gt none` vs `--gt phased` (all other args equal) produces byte-identical CHROM/POS/REF/ALT,
differing only in the GT column -- a matched naive/haplotype pair for apples-to-apples benchmarking.
"""

import argparse
import random
import sys

import pysam

BASES = ("A", "C", "G", "T")


def load_regions(path, fa, margin):
    """Parse regions.tsv (NAME CHROM STRAND TX_START TX_END ...) into usable gene spans.

    `margin` keeps variants clear of gene edges, where predict_variant clamps its window and the
    context would be partly N-padded -- not wrong, but not what we want to be timing.
    """
    order = {name: i for i, name in enumerate(fa.references)}
    genes, skipped = [], set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            name, chrom, start, end = f[0], f[1], int(f[3]), int(f[4])
            if chrom not in order:
                skipped.add(chrom)
                continue
            lo, hi = start + margin, end - margin
            if hi > lo:
                genes.append((name, chrom, lo, hi))
    if skipped:
        print(f"warning: {len(skipped)} contig(s) in {path} absent from the FASTA, e.g. "
              f"{sorted(skipped)[:3]} -- check naming (chr1 vs 1)", file=sys.stderr)
    if not genes:
        sys.exit(f"no usable gene regions in {path}")
    # Sort by the FASTA's own contig order so the output matches the VCF header ordering.
    return sorted(genes, key=lambda g: (order[g[1]], g[2]))


def make_indel(fa, chrom, pos, max_len, rng):
    """Return (ref, alt) for a small indel at 1-based `pos`, or None if the reference is unusable.

    VCF/TSV convention: both alleles carry the anchor base at `pos`.
    """
    length = rng.randint(1, max_len)
    if rng.random() < 0.5:
        span = fa.fetch(chrom, pos - 1, pos + length).upper()          # deletion
        if len(span) != length + 1 or any(b not in "ACGT" for b in span):
            return None
        return span, span[0]
    anchor = fa.fetch(chrom, pos - 1, pos).upper()                     # insertion
    if anchor not in BASES:
        return None
    return anchor, anchor + "".join(rng.choice(BASES) for _ in range(length))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fasta", help="reference FASTA (plain or bgzip-compressed, must be indexed)")
    p.add_argument("regions", help="regions.tsv with gene spans")
    p.add_argument("output")
    p.add_argument("--count", type=int, default=10_000, help="number of variants (default: 10000)")
    p.add_argument("--indel-period", type=int, default=0,
                   help="every Nth variant is an indel; 0 means SNVs only (default: 0)")
    p.add_argument("--indel-max-len", type=int, default=10,
                   help="max indel length, must stay under --window-size (default: 10)")
    p.add_argument("--format", choices=("vcf", "tsv"), default="vcf",
                   help="output format; both predict binaries read either (default: vcf)")
    p.add_argument("--margin", type=int, default=100, help="keep variants this far inside gene edges")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gt", choices=("none", "phased"), default="none",
                   help="none (default): no GT column, a 'naive' benchmark input. phased: attach a "
                        "phased genotype to every variant, a 'haplotype' benchmark input")
    p.add_argument("--phased-hom-rate", type=float, default=0.3,
                   help="with --gt phased, fraction of variants made homozygous-alt (1|1) rather "
                        "than heterozygous (0|1/1|0) (default: 0.3)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    # Independent stream: genotype assignment must never consume from `rng`, or turning --gt on
    # would shift every subsequent position/allele draw and break the naive/phased matched pair.
    gt_rng = random.Random(args.seed + 1_000_003) if args.gt == "phased" else None
    try:
        fa = pysam.FastaFile(args.fasta)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not open {args.fasta}: {exc}\n"
                 f"An indexed FASTA is required -- run: samtools faidx {args.fasta}")

    genes = load_regions(args.regions, fa, args.margin)

    total_span = sum(hi - lo + 1 for _, _, lo, hi in genes)
    counts, assigned = {}, 0
    for i, (_name, _chrom, lo, hi) in enumerate(genes):
        n = args.count - assigned if i == len(genes) - 1 else args.count * (hi - lo + 1) // total_span
        counts[i] = n
        assigned += n

    n_snv = n_ins = n_del = 0
    index = 0

    with open(args.output, "w") as out:
        if args.format == "vcf":
            out.write("##fileformat=VCFv4.2\n")
            for name, length in zip(fa.references, fa.lengths):
                out.write(f"##contig=<ID={name},length={length}>\n")
            if gt_rng is not None:
                out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
                out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
            else:
                out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        else:
            out.write("#CHROM\tPOS\tREF\tALT\tGT\n" if gt_rng is not None else "#CHROM\tPOS\tREF\tALT\n")

        # Genes are emitted in contig order with positions sorted within each, so the file comes
        # out globally sorted without holding every record in memory.
        for i, (_gene_name, chrom, lo, hi) in enumerate(genes):
            for pos in sorted(rng.randint(lo, hi) for _ in range(counts[i])):
                index += 1
                ref = alt = None
                if args.indel_period and index % args.indel_period == 0:
                    made = make_indel(fa, chrom, pos, args.indel_max_len, rng)
                    if made:
                        ref, alt = made
                        if len(alt) > len(ref):
                            n_ins += 1
                        else:
                            n_del += 1
                if ref is None:
                    ref = fa.fetch(chrom, pos - 1, pos).upper()
                    if ref not in BASES:
                        continue          # N or masked base -- skip rather than fabricate a REF
                    alt = rng.choice([b for b in BASES if b != ref])
                    n_snv += 1

                gt = None
                if gt_rng is not None:
                    gt = "1|1" if gt_rng.random() < args.phased_hom_rate else gt_rng.choice(("0|1", "1|0"))

                if args.format == "vcf":
                    if gt is not None:
                        out.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\tGT\t{gt}\n")
                    else:
                        out.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")
                else:
                    if gt is not None:
                        out.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{gt}\n")
                    else:
                        out.write(f"{chrom}\t{pos}\t{ref}\t{alt}\n")

    print(f"{args.output}: {n_snv + n_ins + n_del} variants "
          f"({n_snv} SNV, {n_ins} ins, {n_del} del) across {len(genes)} genes", file=sys.stderr)


if __name__ == "__main__":
    main()
