load 'lib/common'

# Compares the ONNX Runtime backend against the TensorFlow backend at full precision, on
# the same fixture. Not run by default -- CMakeLists.txt's `check` target only builds and
# tests one backend at a time (see CPLICEAI_INFERENCE_BACKEND). To run this manually, build
# both trees and point these env vars at each:
#
#   cmake -S . -B build-ort -DCPLICEAI_INFERENCE_BACKEND=onnxruntime && cmake --build build-ort
#   cmake -S . -B build-tf  -DCPLICEAI_INFERENCE_BACKEND=tensorflow  && cmake --build build-tf
#   CPLICEAI_ORT_REFERENCE_BIN=build-ort/cpliceai_reference \
#   CPLICEAI_ORT_PREDICT_GENE_BIN=build-ort/cpliceai_predict_gene \
#   CPLICEAI_ORT_MODEL_DIR=models/onnx \
#   CPLICEAI_TF_REFERENCE_BIN=build-tf/cpliceai_reference \
#   CPLICEAI_TF_PREDICT_GENE_BIN=build-tf/cpliceai_predict_gene \
#   CPLICEAI_TF_MODEL_DIR=models \
#   CPLICEAI_ORT_EP=cpu CPLICEAI_ORT_INTRA_OP_THREADS=1 \
#   bats tests/backend_parity.bats

MAX_ABS_DIFF=1e-3

setup() {
    for var in CPLICEAI_ORT_REFERENCE_BIN CPLICEAI_ORT_PREDICT_GENE_BIN CPLICEAI_ORT_MODEL_DIR \
               CPLICEAI_TF_REFERENCE_BIN CPLICEAI_TF_PREDICT_GENE_BIN CPLICEAI_TF_MODEL_DIR; do
        [ -n "${!var:-}" ] || skip "backend_parity.bats needs both backends' paths set (see file header) -- $var is unset"
    done
    for bin in "$CPLICEAI_ORT_REFERENCE_BIN" "$CPLICEAI_ORT_PREDICT_GENE_BIN" \
               "$CPLICEAI_TF_REFERENCE_BIN" "$CPLICEAI_TF_PREDICT_GENE_BIN"; do
        [ -x "$bin" ] || skip "binary not built: $bin"
    done
    common_tmpdir_setup
}

teardown() {
    common_tmpdir_teardown
}

@test "ONNX Runtime and TensorFlow backends produce matching predict_gene output" {
    local ort_ref="$TEST_TMPDIR/reference.ort.bin"
    local tf_ref="$TEST_TMPDIR/reference.tf.bin"
    local ort_tsv="$TEST_TMPDIR/scores.ort.tsv"
    local tf_tsv="$TEST_TMPDIR/scores.tf.tsv"

    run "$CPLICEAI_ORT_REFERENCE_BIN" "$CPLICEAI_ORT_MODEL_DIR" "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/regions.tsv" "$ort_ref"
    [ "$status" -eq 0 ]
    run "$CPLICEAI_TF_REFERENCE_BIN" "$CPLICEAI_TF_MODEL_DIR" "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/regions.tsv" "$tf_ref"
    [ "$status" -eq 0 ]

    run "$CPLICEAI_ORT_PREDICT_GENE_BIN" "$FIXTURES_DIR/variants.tsv" "$ort_ref" "$CPLICEAI_ORT_MODEL_DIR" "$FIXTURES_DIR/chrTest.fasta" "$ort_tsv"
    [ "$status" -eq 0 ]
    run "$CPLICEAI_TF_PREDICT_GENE_BIN" "$FIXTURES_DIR/variants.tsv" "$tf_ref" "$CPLICEAI_TF_MODEL_DIR" "$FIXTURES_DIR/chrTest.fasta" "$tf_tsv"
    [ "$status" -eq 0 ]

    # reference.bin is a sensitive canary for numerical drift: build_reference_scores.c only
    # writes a PositionScore when it clears ZERO_EPSILON, so a differing byte count means at
    # least one position crossed that threshold differently between backends.
    local ort_size tf_size
    ort_size=$(wc -c < "$ort_ref")
    tf_size=$(wc -c < "$tf_ref")
    echo "reference.bin size: ort=$ort_size tf=$tf_size" >&3
    [ "$ort_size" -eq "$tf_size" ]

    # Full 6-decimal-place diff (predict_gene.c prints %f), not the 2dp bats fixture checks --
    # this is the highest-precision signal available without touching Python/numpy.
    run bash -c "
        join -t \$'\t' <(grep -v '^#' '$ort_tsv' | sort -n) <(grep -v '^#' '$tf_tsv' | sort -n) |
        awk -F'\t' '
            function ad(a,b) { return (a>b) ? a-b : b-a }
            { for (i=2; i<=5; i++) { d = ad(\$i, \$(i+4)); if (d > max) max = d } }
            END { printf \"%.9f\", max+0 }
        '
    "
    [ "$status" -eq 0 ]
    echo "max abs diff (ONNX Runtime vs TensorFlow, full precision): $output" >&3

    run awk -v d="$output" -v max="$MAX_ABS_DIFF" 'BEGIN { exit !(d < max) }'
    [ "$status" -eq 0 ]
}
