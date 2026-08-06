load 'lib/common'

# Gene-boundary behaviour, especially for indels.
#
# tests/fixtures/regions.tsv's only gene spans the whole 2000 bp contig, which conflates the
# gene boundary with the contig edge. regions.interior.tsv puts GENE2 at 0-based [500, 1500)
# instead, so a variant can run past the gene while still being inside the reference.
#
# Coordinates: the regions file is 0-based half-open, VCF/TSV POS is 1-based. So GENE2's first
# base is POS 501 and its last is POS 1500.

REGIONS="regions.interior.tsv"

build_reference() {
    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/$REGIONS" \
        "$TEST_TMPDIR/reference.bin"
    [ "$status" -eq 0 ]
}

# base_at <1-based pos> [length] - the reference base(s) there, so fixtures never hardcode a
# REF that disagrees with the fasta.
base_at() {
    awk -v p="$1" -v n="${2:-1}" 'NR>1{s=s$0} END{print substr(s,p,n)}' "$FIXTURES_DIR/chrTest.fasta"
}

# score <pos> <ref> <alt> - annotate one variant, echoing its SpliceAI field ('.' if none).
score() {
    printf 'CHROM\tPOS\tREF\tALT\n%s\t%s\t%s\t%s\n' chrTest "$1" "$2" "$3" > "$TEST_TMPDIR/one.tsv"
    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$TEST_TMPDIR/one.tsv" \
        "$TEST_TMPDIR/reference.bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/$REGIONS" \
        "$TEST_TMPDIR/one.out"
    [ "$status" -eq 0 ]
    tail -n +2 "$TEST_TMPDIR/one.out" | cut -f5
}

@test "a SNV on the first or last base of a gene is scored" {
    build_reference

    [[ "$(score 501  "$(base_at 501)"  T)" == *"GENE2"* ]]
    [[ "$(score 1500 "$(base_at 1500)" G)" == *"GENE2"* ]]
}

@test "a SNV one base outside the gene is not scored" {
    build_reference

    # POS 500 is 0-based 499, one before tx_start; POS 1501 is 0-based 1500, i.e. tx_end.
    [ "$(score 500  "$(base_at 500)"  T)" = "." ]
    [ "$(score 1501 "$(base_at 1501)" T)" = "." ]
}

@test "a deletion is scored iff it ends exactly at the gene boundary, not past it" {
    build_reference

    # REF spanning POS 1498-1500 ends on the gene's last base: 0-based 1497 + 3 == tx_end.
    local exact="$(base_at 1498 3)"
    [[ "$(score 1498 "$exact" "${exact:0:1}")" == *"GENE2"* ]]

    # One base longer runs past tx_end, so there is no reference left to compare against.
    local over="$(base_at 1498 4)"
    [ "$(score 1498 "$over" "${over:0:1}")" = "." ]
}

@test "a deletion crossing the 5' boundary is not scored" {
    build_reference

    # Anchored before tx_start, reaching into the gene.
    local across="$(base_at 499 4)"
    [ "$(score 499 "$across" "${across:0:1}")" = "." ]

    # Anchored exactly on the first gene base, so fully contained.
    local inside="$(base_at 501 4)"
    [[ "$(score 501 "$inside" "${inside:0:1}")" == *"GENE2"* ]]
}

@test "an insertion at either gene boundary is scored" {
    build_reference

    # An insertion spans a single reference base (the anchor), so one at the last base of the
    # gene is contained even though the inserted bases notionally land beyond it.
    local first="$(base_at 501)" last="$(base_at 1500)"
    [[ "$(score 501  "$first" "${first}TTT")" == *"GENE2"* ]]
    [[ "$(score 1500 "$last"  "${last}TTT")"  == *"GENE2"* ]]
}

@test "an indel longer than the window radius is reported as unscored" {
    build_reference

    # predict_variant refuses alleles it cannot fit in the window, emitting '.' rather than
    # silently truncating. 600 > the default --window-radius of 500.
    local big="$(base_at 700 600)"
    [ "$(score 700 "$big" "${big:0:1}")" = "." ]
}

@test "predict_gene applies the same containment rule" {
    build_reference

    # Contained: one score block. Crossing tx_end: no block, and a clean exit.
    local exact="$(base_at 1498 3)" over="$(base_at 1498 4)"

    printf 'CHROM\tPOS\tREF\tALT\nchrTest\t1498\t%s\t%s\n' "$exact" "${exact:0:1}" > "$TEST_TMPDIR/in.tsv"
    run "$CPLICEAI_PREDICT_GENE_BIN" \
        "$TEST_TMPDIR/in.tsv" "$TEST_TMPDIR/reference.bin" "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/$REGIONS" "$TEST_TMPDIR/in.out"
    [ "$status" -eq 0 ]
    [ "$(grep -c '^#GENE2_' "$TEST_TMPDIR/in.out")" -eq 1 ]

    printf 'CHROM\tPOS\tREF\tALT\nchrTest\t1498\t%s\t%s\n' "$over" "${over:0:1}" > "$TEST_TMPDIR/out.tsv"
    run "$CPLICEAI_PREDICT_GENE_BIN" \
        "$TEST_TMPDIR/out.tsv" "$TEST_TMPDIR/reference.bin" "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/$REGIONS" "$TEST_TMPDIR/out.out"
    [ "$status" -eq 0 ]
    [ "$(grep -c '^#GENE2_' "$TEST_TMPDIR/out.out")" -eq 0 ]
}
