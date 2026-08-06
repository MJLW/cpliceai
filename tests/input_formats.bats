load 'lib/common'

# Both predict binaries accept either a VCF or a plain CHROM/POS/REF/ALT TSV. The point of
# this file is that the *analysis* does not depend on the input format: the same variant
# expressed either way must score identically. Output always mirrors the input format, so
# there is no conversion to test - only that nothing is lost passing through.
#
# Like pipeline.bats, each test builds its own reference.bin rather than sharing one, to
# avoid depending on bats-core >= 1.3.0's setup_file()/BATS_FILE_TMPDIR.

build_reference() {
    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$TEST_TMPDIR/reference.bin"
    [ "$status" -eq 0 ]
}

predict_variant() {
    local variants="$1" output="$2"; shift 2
    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$variants" \
        "$TEST_TMPDIR/reference.bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$output" \
        "$@"
}

predict_gene() {
    local variants="$1" output="$2"; shift 2
    run "$CPLICEAI_PREDICT_GENE_BIN" \
        "$variants" \
        "$TEST_TMPDIR/reference.bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$output" \
        "$@"
}

# The pipe-delimited SpliceAI value(s) of the single record in an annotated VCF.
vcf_annotation() {
    bcftools view -H "$1" | cut -f8 | sed 's/^SpliceAI=//'
}

# The SpliceAI column of the single data row in an annotated TSV (skipping the header).
tsv_annotation() {
    tail -n +2 "$1" | cut -f5
}

@test "predict_variant scores a TSV variant identically to the same variant in a VCF" {
    build_reference

    # No --input-format: detection is the default.
    predict_variant "$FIXTURES_DIR/variants.vcf" "$TEST_TMPDIR/from_vcf.vcf"
    [ "$status" -eq 0 ]
    predict_variant "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/from_tsv.tsv"
    [ "$status" -eq 0 ]

    # The output format mirrors the input, so the two runs produced different formats...
    [[ "$(head -1 "$TEST_TMPDIR/from_vcf.vcf")" == "##fileformat=VCF"* ]]
    [ "$(head -1 "$TEST_TMPDIR/from_tsv.tsv")" = "CHROM	POS	REF	ALT	SpliceAI" ]

    # ...carrying the same annotation.
    [ "$(vcf_annotation "$TEST_TMPDIR/from_vcf.vcf")" = "$(tsv_annotation "$TEST_TMPDIR/from_tsv.tsv")" ]
}

@test "predict_gene scores a VCF variant identically to the same variant in a TSV" {
    build_reference

    predict_gene "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/from_tsv.tsv"
    [ "$status" -eq 0 ]
    predict_gene "$FIXTURES_DIR/variants.vcf" "$TEST_TMPDIR/from_vcf.tsv"
    [ "$status" -eq 0 ]

    # Byte-identical, block headers included - and those name the gene, which the TSV no
    # longer carries. Matching output is what proves the regions lookup recovers it.
    run diff "$TEST_TMPDIR/from_tsv.tsv" "$TEST_TMPDIR/from_vcf.tsv"
    [ "$status" -eq 0 ]

    run grep -c '^#GENE1_+_' "$TEST_TMPDIR/from_vcf.tsv"
    [ "$output" -eq 1 ]
}

@test "annotating a VCF preserves every other field of it" {
    build_reference

    # A VCF must come back as itself plus one INFO tag. variants.rich.vcf carries a non-empty
    # ID, QUAL, FILTER and INFO field, so the columns being compared are not all empty.
    predict_variant "$FIXTURES_DIR/variants.rich.vcf" "$TEST_TMPDIR/annotated.vcf"
    [ "$status" -eq 0 ]

    # The header gains exactly the SpliceAI INFO line and nothing else. bcftools stamps its
    # own ##bcftools_viewCommand provenance line onto whatever it reads, so exclude that -
    # it comes from the reader, not from us.
    run bash -c "diff <(bcftools view -h '$FIXTURES_DIR/variants.rich.vcf') \
                      <(bcftools view -h '$TEST_TMPDIR/annotated.vcf') \
                 | grep '^>' | grep -v '##bcftools_' | grep -vc '##INFO=<ID=SpliceAI,'"
    [ "$output" -eq 0 ]

    # CHROM POS ID REF ALT QUAL FILTER are byte-identical; only INFO grew.
    run bash -c "diff <(bcftools view -H '$FIXTURES_DIR/variants.rich.vcf' | cut -f1-7) \
                      <(bcftools view -H '$TEST_TMPDIR/annotated.vcf'      | cut -f1-7)"
    [ "$status" -eq 0 ]

    # The pre-existing INFO field survives alongside the new one.
    run bcftools view -H "$TEST_TMPDIR/annotated.vcf"
    [[ "$output" == *"DP=42"* ]]
    [[ "$output" == *"SpliceAI="* ]]
}

@test "gzipped variants and regions files are read transparently" {
    build_reference

    gzip -c "$FIXTURES_DIR/variants.tsv" > "$TEST_TMPDIR/variants.tsv.gz"
    gzip -c "$FIXTURES_DIR/regions.tsv"  > "$TEST_TMPDIR/regions.tsv.gz"

    predict_variant "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/plain.tsv"
    [ "$status" -eq 0 ]

    # The regions file is compressed here too, so this covers both readers at once.
    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$TEST_TMPDIR/variants.tsv.gz" \
        "$TEST_TMPDIR/reference.bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$TEST_TMPDIR/regions.tsv.gz" \
        "$TEST_TMPDIR/gz.tsv"
    [ "$status" -eq 0 ]

    run diff "$TEST_TMPDIR/plain.tsv" "$TEST_TMPDIR/gz.tsv"
    [ "$status" -eq 0 ]

    # cpliceai_reference reads the regions file through the same reader. Compare the scores
    # it yields rather than the .bin bytes: reference.bin is not byte-reproducible even across
    # two runs on identical input (struct padding is written uninitialised), so cmp on the
    # file would fail for reasons that have nothing to do with compression.
    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$TEST_TMPDIR/regions.tsv.gz" \
        "$TEST_TMPDIR/reference_from_gz.bin"
    [ "$status" -eq 0 ]

    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$FIXTURES_DIR/variants.tsv" \
        "$TEST_TMPDIR/reference_from_gz.bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$TEST_TMPDIR/via_gz_reference.tsv"
    [ "$status" -eq 0 ]

    run diff "$TEST_TMPDIR/plain.tsv" "$TEST_TMPDIR/via_gz_reference.tsv"
    [ "$status" -eq 0 ]
}

@test "a header is detected by content, and only on the first line" {
    build_reference

    predict_variant "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/headed.tsv"
    [ "$status" -eq 0 ]

    # A '#'-prefixed header still works, for files that carry one.
    sed '1s/^/#/' "$FIXTURES_DIR/variants.tsv" > "$TEST_TMPDIR/hashed.tsv"
    predict_variant "$TEST_TMPDIR/hashed.tsv" "$TEST_TMPDIR/from_hashed.tsv"
    [ "$status" -eq 0 ]
    run diff "$TEST_TMPDIR/headed.tsv" "$TEST_TMPDIR/from_hashed.tsv"
    [ "$status" -eq 0 ]

    # Detection is confined to one line: a bad POS on the *second* line is a malformed row,
    # not a second header, and must fail rather than be silently skipped.
    printf 'CHROM\tPOS\tREF\tALT\nchrTest\tnot-a-number\tG\tA\n' > "$TEST_TMPDIR/bad_second.tsv"
    predict_variant "$TEST_TMPDIR/bad_second.tsv" "$TEST_TMPDIR/unused.tsv"
    [ "$status" -ne 0 ]
}

@test "a TSV without a header row keeps its first variant" {
    build_reference

    predict_variant "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/with_header.tsv"
    [ "$status" -eq 0 ]
    predict_variant "$FIXTURES_DIR/variants.noheader.tsv" "$TEST_TMPDIR/no_header.tsv"
    [ "$status" -eq 0 ]

    run diff "$TEST_TMPDIR/with_header.tsv" "$TEST_TMPDIR/no_header.tsv"
    [ "$status" -eq 0 ]
}

@test "--input-format agrees with detection, including on compressed VCFs" {
    build_reference

    predict_variant "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/detected.tsv"
    [ "$status" -eq 0 ]
    predict_variant "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/forced.tsv" --input-format tsv
    [ "$status" -eq 0 ]
    run diff "$TEST_TMPDIR/detected.tsv" "$TEST_TMPDIR/forced.tsv"
    [ "$status" -eq 0 ]

    predict_variant "$FIXTURES_DIR/variants.vcf" "$TEST_TMPDIR/detected.vcf"
    [ "$status" -eq 0 ]
    predict_variant "$FIXTURES_DIR/variants.vcf" "$TEST_TMPDIR/forced.vcf" --input-format vcf
    [ "$status" -eq 0 ]
    run diff "$TEST_TMPDIR/detected.vcf" "$TEST_TMPDIR/forced.vcf"
    [ "$status" -eq 0 ]

    # BGZF and BCF exercise the magic-number branches of htslib's detection, not the
    # "##fileformat=" text branch the plain .vcf above goes through.
    bcftools view -Oz -o "$TEST_TMPDIR/variants.vcf.gz" "$FIXTURES_DIR/variants.vcf"
    bcftools view -Ob -o "$TEST_TMPDIR/variants.bcf" "$FIXTURES_DIR/variants.vcf"

    predict_variant "$TEST_TMPDIR/variants.vcf.gz" "$TEST_TMPDIR/from_gz.vcf"
    [ "$status" -eq 0 ]
    predict_variant "$TEST_TMPDIR/variants.bcf" "$TEST_TMPDIR/from_bcf.vcf"
    [ "$status" -eq 0 ]

    local expected
    expected="$(vcf_annotation "$TEST_TMPDIR/detected.vcf")"
    [ "$(vcf_annotation "$TEST_TMPDIR/from_gz.vcf")" = "$expected" ]
    [ "$(vcf_annotation "$TEST_TMPDIR/from_bcf.vcf")" = "$expected" ]
}

@test "predict_variant TSV output can be fed straight back in" {
    build_reference

    predict_variant "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/once.tsv"
    [ "$status" -eq 0 ]

    # The first four columns are exactly the input schema, so the annotated output is itself
    # valid input; the trailing SPLICEAI column is read past and ignored.
    predict_variant "$TEST_TMPDIR/once.tsv" "$TEST_TMPDIR/twice.tsv"
    [ "$status" -eq 0 ]

    run diff "$TEST_TMPDIR/once.tsv" "$TEST_TMPDIR/twice.tsv"
    [ "$status" -eq 0 ]
}

@test "every alternate allele of a multiallelic record is annotated" {
    build_reference

    predict_variant "$FIXTURES_DIR/variants.multiallelic.vcf" "$TEST_TMPDIR/ma.vcf"
    [ "$status" -eq 0 ]
    predict_variant "$FIXTURES_DIR/variants.multiallelic.tsv" "$TEST_TMPDIR/ma.tsv"
    [ "$status" -eq 0 ]

    # One annotation per ALT, in allele order.
    local from_vcf from_tsv
    from_vcf="$(vcf_annotation "$TEST_TMPDIR/ma.vcf")"
    from_tsv="$(tsv_annotation "$TEST_TMPDIR/ma.tsv")"
    [ "$(awk -F, '{print NF}' <<< "$from_vcf")" -eq 2 ]
    [ "$from_vcf" = "$from_tsv" ]
    [[ "$from_vcf" == "A|GENE1|"*",T|GENE1|"* ]]

    # One input record stays one output row, so TSV output round-trips.
    [ "$(tail -n +2 "$TEST_TMPDIR/ma.tsv" | wc -l)" -eq 1 ]
    [ "$(tail -n +2 "$TEST_TMPDIR/ma.tsv" | cut -f4)" = "A,T" ]

    # The single-allele annotation must be unchanged by the presence of a second allele.
    predict_variant "$FIXTURES_DIR/variants.vcf" "$TEST_TMPDIR/single.vcf"
    [ "$status" -eq 0 ]
    [ "$(cut -d, -f1 <<< "$from_vcf")" = "$(vcf_annotation "$TEST_TMPDIR/single.vcf")" ]
}

@test "predict_gene emits one score block per alternate allele" {
    build_reference

    predict_gene "$FIXTURES_DIR/variants.multiallelic.tsv" "$TEST_TMPDIR/ma_tsv.tsv"
    [ "$status" -eq 0 ]
    predict_gene "$FIXTURES_DIR/variants.multiallelic.vcf" "$TEST_TMPDIR/ma_vcf.tsv"
    [ "$status" -eq 0 ]

    run diff "$TEST_TMPDIR/ma_tsv.tsv" "$TEST_TMPDIR/ma_vcf.tsv"
    [ "$status" -eq 0 ]

    # Block headers name a single allele each, so a 2-allele record yields 2 blocks.
    run grep -c '^#GENE1_' "$TEST_TMPDIR/ma_vcf.tsv"
    [ "$output" -eq 2 ]
    run grep -c '^#GENE1_+_0_2000:chrTest_1000_G_A$' "$TEST_TMPDIR/ma_vcf.tsv"
    [ "$output" -eq 1 ]
    run grep -c '^#GENE1_+_0_2000:chrTest_1000_G_T$' "$TEST_TMPDIR/ma_vcf.tsv"
    [ "$output" -eq 1 ]
}

@test "a reference built from different inputs is rejected" {
    build_reference   # from regions.tsv

    # Wrong gene set: reference.bin knows which regions it was computed from.
    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/reference.bin" "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/regions.interior.tsv" \
        "$TEST_TMPDIR/wrong_regions.tsv"
    [ "$status" -ne 0 ]
    [[ "$output" == *"gene regions file does not match"* ]]

    # Wrong assembly: a fasta with different contigs or lengths.
    printf '>chrOther\nACGTACGTAC\n' > "$TEST_TMPDIR/other.fasta"
    printf 'chrOther\t10\t10\t10\t11\n' > "$TEST_TMPDIR/other.fasta.fai"
    run "$CPLICEAI_PREDICT_GENE_BIN" \
        "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/reference.bin" "$MODEL_DIR" \
        "$TEST_TMPDIR/other.fasta" "$FIXTURES_DIR/regions.tsv" \
        "$TEST_TMPDIR/wrong_fasta.tsv"
    [ "$status" -ne 0 ]
    [[ "$output" == *"reference fasta does not match"* ]]
}

@test "the reference check compares gene content, not file bytes" {
    build_reference

    # Same genes, different encoding: gzipped and with a '#'-prefixed header. The digest is
    # accumulated over parsed genes, so reformatting an annotation must not force a rebuild.
    sed '1s/^/#/' "$FIXTURES_DIR/regions.tsv" | gzip -c > "$TEST_TMPDIR/reformatted.tsv.gz"

    predict_variant "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/plain.tsv"
    [ "$status" -eq 0 ]

    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$FIXTURES_DIR/variants.tsv" "$TEST_TMPDIR/reference.bin" "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" "$TEST_TMPDIR/reformatted.tsv.gz" \
        "$TEST_TMPDIR/reformatted.out.tsv"
    [ "$status" -eq 0 ]

    run diff "$TEST_TMPDIR/plain.tsv" "$TEST_TMPDIR/reformatted.out.tsv"
    [ "$status" -eq 0 ]
}

@test "a malformed TSV row fails the run instead of truncating it" {
    build_reference

    predict_variant "$FIXTURES_DIR/variants.badpos.tsv" "$TEST_TMPDIR/bad.tsv"
    [ "$status" -ne 0 ]

    predict_gene "$FIXTURES_DIR/variants.badpos.tsv" "$TEST_TMPDIR/bad_gene.tsv"
    [ "$status" -ne 0 ]
}
