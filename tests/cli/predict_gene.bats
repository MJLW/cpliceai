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
