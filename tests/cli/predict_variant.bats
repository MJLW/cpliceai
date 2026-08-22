load '../lib/common'

@test "cpliceai_predict_variant -h prints usage and exits non-zero" {
    run "$CPLICEAI_PREDICT_VARIANT_BIN" -h
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}

@test "cpliceai_predict_variant with no arguments prints usage and exits non-zero" {
    run "$CPLICEAI_PREDICT_VARIANT_BIN"
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}

@test "cpliceai_predict_variant with insufficient arguments prints usage and exits non-zero" {
    run "$CPLICEAI_PREDICT_VARIANT_BIN" "$FIXTURES_DIR/variants.vcf" "does-not-exist.bin"
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}

# The model_dir here is deliberately bogus: reaching the "input format" message proves the
# value is validated before load_models, which otherwise costs seconds.
@test "cpliceai_predict_variant rejects an unknown --input-format value before loading models" {
    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$FIXTURES_DIR/variants.vcf" \
        "does-not-exist.bin" \
        "/nonexistent-model-dir" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$TEST_TMPDIR/unused.vcf" \
        --input-format xml
    [ "$status" -ne 0 ]
    [[ "$output" == *"input format"* ]]
}

@test "cpliceai_predict_variant documents its haplotype flag" {
    run "$CPLICEAI_PREDICT_VARIANT_BIN" -h
    [[ "$output" == *"--include-unphased"* ]]
}

# Also before load_models: genotypes are read from one sample, and choosing between several
# silently would be a wrong answer with no outward sign.
@test "cpliceai_predict_variant rejects a multi-sample VCF before loading models" {
    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$FIXTURES_DIR/variants.multisample.vcf" \
        "does-not-exist.bin" \
        "/nonexistent-model-dir" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$TEST_TMPDIR/unused.vcf"
    [ "$status" -ne 0 ]
    [[ "$output" == *"2 samples"* ]]
    [[ "$output" == *"bcftools view -s"* ]]
}

# Checked before even opening the variants file: the flag hard-errors rather than silently
# doing nothing, which is what it used to do.
@test "cpliceai_predict_variant rejects --splice-output" {
    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "does-not-exist.vcf" \
        "does-not-exist.bin" \
        "/nonexistent-model-dir" \
        "does-not-exist.fasta" \
        "does-not-exist.tsv" \
        "$TEST_TMPDIR/unused.vcf" \
        --splice-output "$TEST_TMPDIR/unused-splice.tsv"
    [ "$status" -ne 0 ]
    [[ "$output" == *"not yet implemented"* ]]
}
