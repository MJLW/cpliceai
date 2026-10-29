#!/usr/bin/env python3
"""Generate synthetic genome/regions/variant fixtures for scripts/run_benchmark_suite.py.

Self-contained alternative to scripts/gen_variant_benchmark.py: that script needs pysam and a real,
multi-GB reference genome. Neither is reliably available in every environment this suite runs in (in
particular pysam isn't installed in this sandbox's venvs), and the benchmark suite doesn't need real
biological sequence -- only wall-clock timing -- so this works entirely off an in-memory synthetic
contig instead. Mirrors scripts/benchmark.sh's existing "no dependency on the real reference genome"
approach, generalised to arbitrarily many genes and reused by scripts/gen_variant_benchmark.py's
naive/haplotype matched-pair trick (see `variants` below).

Two subcommands:

    gen_synthetic_fixture.py genome  --out-prefix P --num-genes N [--min-size --max-size --seed]
        Writes P.fasta (+ .fai) and P.regions.tsv: one contig ("chrBench"), N genes back to back,
        each a single exon of uniform-random size in [--min-size, --max-size].

    gen_synthetic_fixture.py variants --fasta P.fasta --regions P.regions.tsv --out OUT
        --count N --gt {none,phased} [--seed --margin --phased-hom-rate]
        Writes an OUT TSV (#CHROM\\tPOS\\tREF\\tALT[\\tGT]) of SNVs spread proportionally across the
        genes in --regions, format matching src/variant_input.c's TSV reader exactly (CHROM POS REF
        ALT [GT], 1-based POS). SNV-only -- shape-churn from indels is a separate concern already
        covered by gen_variant_benchmark.py and isn't part of what this suite contrasts.

        --gt none (default) omits the GT column: no HAP_REF/HAP_ALT recomputation, a "naive" run.
        --gt phased attaches a phased genotype (mostly 0|1/1|0, some 1|1) drawn from an independent
        RNG stream seeded off --seed, so `--gt none` and `--gt phased` at the same --seed/--count
        produce byte-identical CHROM/POS/REF/ALT -- a matched naive/haplotype pair.
"""

import argparse
import random
import sys

BASES = ("A", "C", "G", "T")
FASTA_WIDTH = 70
CONTIG = "chrBench"


def cmd_genome(args):
    rng = random.Random(args.seed)
    sizes = [rng.randint(args.min_size, args.max_size) for _ in range(args.num_genes)]
    total = sum(sizes)

    seq_parts = []
    remaining = total
    while remaining > 0:
        chunk = min(remaining, 1_000_000)
        seq_parts.append("".join(rng.choice(BASES) for _ in range(chunk)))
        remaining -= chunk
    seq = "".join(seq_parts)

    fasta_path = f"{args.out_prefix}.fasta"
    with open(fasta_path, "w") as fh:
        fh.write(f">{CONTIG}\n")
        for i in range(0, len(seq), FASTA_WIDTH):
            fh.write(seq[i:i + FASTA_WIDTH])
            fh.write("\n")

    offset = len(f">{CONTIG}\n")
    with open(f"{fasta_path}.fai", "w") as fh:
        fh.write(f"{CONTIG}\t{total}\t{offset}\t{FASTA_WIDTH}\t{FASTA_WIDTH + 1}\n")

    regions_path = f"{args.out_prefix}.regions.tsv"
    with open(regions_path, "w") as fh:
        fh.write("NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\n")
        start = 1
        for i, size in enumerate(sizes):
            end = start + size - 1
            strand = "+" if i % 2 == 0 else "-"
            fh.write(f"GENE{i}\t{CONTIG}\t{strand}\t{start}\t{end}\t{start},\t{end},\n")
            start = end + 1

    print(f"{fasta_path}: {total:,} bp across {args.num_genes} genes "
          f"(min={min(sizes):,} max={max(sizes):,} mean={total // args.num_genes:,})", file=sys.stderr)


def load_sequence(fasta_path):
    """Single-contig only -- this generator never writes more than one."""
    with open(fasta_path) as fh:
        lines = fh.read().splitlines()
    if not lines or not lines[0].startswith(">"):
        sys.exit(f"{fasta_path}: not a FASTA (missing '>' header)")
    return "".join(lines[1:])


def load_regions(path, margin):
    genes = []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            name, chrom, start, end = f[0], f[1], int(f[3]), int(f[4])
            lo, hi = start + margin, end - margin
            if hi > lo:
                genes.append((name, chrom, lo, hi))
    if not genes:
        sys.exit(f"no usable gene regions in {path} (genes shorter than 2*margin={2*margin}?)")
    return genes


def cmd_variants(args):
    rng = random.Random(args.seed)
    # Independent stream, same rationale as gen_variant_benchmark.py: genotype assignment must
    # never consume from `rng`, or turning --gt on would shift every position/allele draw and
    # break the naive/phased matched pair.
    gt_rng = random.Random(args.seed + 1_000_003) if args.gt == "phased" else None

    seq = load_sequence(args.fasta)
    genes = load_regions(args.regions, args.margin)

    total_span = sum(hi - lo + 1 for _, _, lo, hi in genes)
    counts, assigned = {}, 0
    for i, (_name, _chrom, lo, hi) in enumerate(genes):
        n = args.count - assigned if i == len(genes) - 1 else args.count * (hi - lo + 1) // total_span
        counts[i] = n
        assigned += n

    n_written = 0
    with open(args.out, "w") as out:
        out.write("#CHROM\tPOS\tREF\tALT\tGT\n" if gt_rng is not None else "#CHROM\tPOS\tREF\tALT\n")
        for i, (_gene_name, chrom, lo, hi) in enumerate(genes):
            for pos in sorted(rng.randint(lo, hi) for _ in range(counts[i])):
                ref = seq[pos - 1]  # 1-based POS -> 0-based string index
                if ref not in BASES:
                    continue
                alt = rng.choice([b for b in BASES if b != ref])

                gt = None
                if gt_rng is not None:
                    gt = "1|1" if gt_rng.random() < args.phased_hom_rate else gt_rng.choice(("0|1", "1|0"))

                if gt is not None:
                    out.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{gt}\n")
                else:
                    out.write(f"{chrom}\t{pos}\t{ref}\t{alt}\n")
                n_written += 1

    print(f"{args.out}: {n_written} SNVs across {len(genes)} genes (--gt {args.gt})", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("genome", help="write a synthetic FASTA + regions.tsv")
    g.add_argument("--out-prefix", required=True, help="writes <prefix>.fasta(.fai) and <prefix>.regions.tsv")
    g.add_argument("--num-genes", type=int, required=True)
    g.add_argument("--min-size", type=int, default=2_000)
    g.add_argument("--max-size", type=int, default=100_000)
    g.add_argument("--seed", type=int, default=1234)
    g.set_defaults(func=cmd_genome)

    v = sub.add_parser("variants", help="write a matched naive/haplotype-ready SNV TSV")
    v.add_argument("--fasta", required=True)
    v.add_argument("--regions", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--count", type=int, required=True)
    v.add_argument("--gt", choices=("none", "phased"), default="none")
    v.add_argument("--margin", type=int, default=100, help="keep variants this far inside gene edges")
    v.add_argument("--phased-hom-rate", type=float, default=0.3)
    v.add_argument("--seed", type=int, default=1234)
    v.set_defaults(func=cmd_variants)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
