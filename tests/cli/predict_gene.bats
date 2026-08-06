load '../lib/common'

@test "cpliceai_predict_gene -h prints usage and exits non-zero" {
    run "$CPLICEAI_PREDICT_GENE_BIN" -h
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}

@test "cpliceai_predict_gene with no arguments prints usage and exits non-zero" {
    run "$CPLICEAI_PREDICT_GENE_BIN"
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}

@test "cpliceai_predict_gene with insufficient arguments prints usage and exits non-zero" {
    run "$CPLICEAI_PREDICT_GENE_BIN" "$FIXTURES_DIR/variants.tsv" "does-not-exist.bin"
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}

# The model_dir here is deliberately bogus: reaching the "input format" message proves the
# value is validated before load_models, which otherwise costs seconds.
@test "cpliceai_predict_gene rejects an unknown --input-format value before loading models" {
    run "$CPLICEAI_PREDICT_GENE_BIN" \
        "$FIXTURES_DIR/variants.tsv" \
        "does-not-exist.bin" \
        "/nonexistent-model-dir" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$TEST_TMPDIR/unused.tsv" \
        --input-format xml
    [ "$status" -ne 0 ]
    [[ "$output" == *"input format"* ]]
}
