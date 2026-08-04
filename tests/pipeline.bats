load 'lib/common'

# Each test below builds its own reference.bin in $TEST_TMPDIR (from
# tests/lib/common.bash's setup()) rather than sharing one built once for the
# whole file. That costs a couple of extra `cpliceai_reference` runs, but
# avoids depending on bats-core's setup_file()/BATS_FILE_TMPDIR, which are
# only available on bats-core >= 1.3.0.

@test "cpliceai_reference builds a non-empty reference scores binary" {
    local ref_bin="$TEST_TMPDIR/reference.bin"

    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$ref_bin"
    [ "$status" -eq 0 ]
    [ -s "$ref_bin" ]
}

@test "cpliceai_reference fails cleanly on a missing regions file" {
    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/does-not-exist.tsv" \
        "$TEST_TMPDIR/unused.bin"
    [ "$status" -ne 0 ]
}

@test "cpliceai_predict_variant annotates a VCF with SpliceAI scores" {
    local ref_bin="$TEST_TMPDIR/reference.bin"
    local output_vcf="$TEST_TMPDIR/annotated.vcf"

    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$ref_bin"
    [ "$status" -eq 0 ]

    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$FIXTURES_DIR/variants.vcf" \
        "$ref_bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$output_vcf"
    [ "$status" -eq 0 ]
    [ -s "$output_vcf" ]

    run bcftools view "$output_vcf"
    [ "$status" -eq 0 ]
    [[ "$output" == *"##INFO=<ID=SpliceAI,"* ]]

    # Check the delta scores (DS_AG|DS_AL|DS_DG|DS_DL) to 2dp; already %.2f-formatted
    # by cpliceai_predict_variant itself. Skip the DP_* delta-position integers -
    # they're argmax indices over near-flat, near-zero deltas here, so they're more
    # sensitive to float noise than the scores actually being checked.
    run bash -c "bcftools view -H '$output_vcf' | cut -f8 | sed 's/^SpliceAI=//' | awk -F'|' '{printf \"%s|%s|%s|%s\", \$3, \$4, \$5, \$6}'"
    [ "$status" -eq 0 ]
    [ "$output" = "0.00|0.00|0.00|0.00" ]
}

@test "cpliceai_predict_gene reports per-position splice scores for the variant" {
    local ref_bin="$TEST_TMPDIR/reference.bin"
    local output_tsv="$TEST_TMPDIR/scores.tsv"

    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$ref_bin"
    [ "$status" -eq 0 ]

    run "$CPLICEAI_PREDICT_GENE_BIN" \
        "$FIXTURES_DIR/variants.tsv" \
        "$ref_bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$output_tsv"
    [ "$status" -eq 0 ]
    [ -s "$output_tsv" ]

    run grep -c '^#GENE1_' "$output_tsv"
    [ "$status" -eq 0 ]
    [ "$output" -ge 1 ]

    # At least one data row: <pos>\t<ref_acceptor>\t<ref_donor>\t<alt_acceptor>\t<alt_donor>
    run bash -c "grep -v '^#' '$output_tsv' | awk -F'\t' 'NF==5' | wc -l"
    [ "$status" -eq 0 ]
    [ "$output" -ge 1 ]

    # Spot-check predicted scores (ref_acceptor, ref_donor, alt_acceptor, alt_donor),
    # rounded to 2dp, at a few representative positions - pins the SpliceAI/TensorFlow
    # inference pipeline against regressions while tolerating float noise below 0.01
    # across CPUs/TF builds.
    assert_gene_score "$output_tsv" 112  0.00 0.63 0.00 0.62
    assert_gene_score "$output_tsv" 679  0.34 0.00 0.38 0.00
    assert_gene_score "$output_tsv" 1260 0.40 0.00 0.37 0.00
    assert_gene_score "$output_tsv" 1290 0.25 0.00 0.23 0.00
}
