load '../lib/common'

@test "cpliceai_reference -h prints usage and exits non-zero" {
    run "$CPLICEAI_REFERENCE_BIN" -h
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}

@test "cpliceai_reference with no arguments prints usage and exits non-zero" {
    run "$CPLICEAI_REFERENCE_BIN"
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}

@test "cpliceai_reference with insufficient arguments prints usage and exits non-zero" {
    run "$CPLICEAI_REFERENCE_BIN" "$MODEL_DIR"
    [ "$status" -eq 1 ]
    [[ "$output" == *"USAGE:"* ]]
}
