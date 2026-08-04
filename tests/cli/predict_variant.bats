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
