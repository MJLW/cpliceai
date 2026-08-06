load 'lib/common'

# Runs the ONNX Runtime backend twice over the same fixtures -- once with the default
# CPLICEAI_ORT_MAX_CHUNK_LEN (unset; larger than the tiny fixture gene, so unchunked) and once
# with a small forced value that splits the ~2000bp fixture gene into several chunks -- and
# checks the outputs match. This exercises the chunk-boundary stitching logic in predict_ort.c's
# predict_chunked(): each output base should come out identical whether or not the sequence was
# chunked, since the model's receptive field is fully contained within CONTEXT_SIZE/BOUNDARY_SIZE.
#
# CMakeLists.txt only registers this test when built with -DCPLICEAI_INFERENCE_BACKEND=onnxruntime
# (chunking is ORT-only; the TensorFlow backend has no chunk size limit or chunking logic).

MAX_ABS_DIFF=1e-5

# Fixture gene (tests/fixtures/regions.tsv) is ~2000bp; CONTEXT_SIZE is 10000, so any
# CPLICEAI_ORT_MAX_CHUNK_LEN above ~12000 leaves it unchunked. 10500 forces the ~2000bp gene into
# 4 chunks of 500bp each.
FORCED_MAX_CHUNK_LEN=10500

@test "chunked and unchunked ONNX Runtime predictions match" {
    local ref_default="$TEST_TMPDIR/reference.default.bin"
    local ref_chunked="$TEST_TMPDIR/reference.chunked.bin"
    local tsv_default="$TEST_TMPDIR/scores.default.tsv"
    local tsv_chunked="$TEST_TMPDIR/scores.chunked.tsv"

    run "$CPLICEAI_REFERENCE_BIN" "$MODEL_DIR" "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/regions.tsv" "$ref_default"
    [ "$status" -eq 0 ]
    run env "CPLICEAI_ORT_MAX_CHUNK_LEN=$FORCED_MAX_CHUNK_LEN" \
        "$CPLICEAI_REFERENCE_BIN" "$MODEL_DIR" "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/regions.tsv" "$ref_chunked"
    [ "$status" -eq 0 ]

    run "$CPLICEAI_PREDICT_GENE_BIN" "$FIXTURES_DIR/variants.tsv" "$ref_default" "$MODEL_DIR" "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/regions.tsv" "$tsv_default"
    [ "$status" -eq 0 ]
    run env "CPLICEAI_ORT_MAX_CHUNK_LEN=$FORCED_MAX_CHUNK_LEN" \
        "$CPLICEAI_PREDICT_GENE_BIN" "$FIXTURES_DIR/variants.tsv" "$ref_chunked" "$MODEL_DIR" "$FIXTURES_DIR/chrTest.fasta" "$FIXTURES_DIR/regions.tsv" "$tsv_chunked"
    [ "$status" -eq 0 ]

    # reference.bin is a sensitive canary for numerical drift, same rationale as
    # backend_parity.bats: build_reference_scores.c only writes a PositionScore when it clears
    # ZERO_EPSILON, so a differing byte count means at least one position crossed that threshold
    # differently between the chunked and unchunked runs.
    local default_size chunked_size
    default_size=$(wc -c < "$ref_default")
    chunked_size=$(wc -c < "$ref_chunked")
    echo "reference.bin size: default=$default_size chunked=$chunked_size" >&3
    [ "$default_size" -eq "$chunked_size" ]

    # Full 6-decimal-place diff (predict_gene.c prints %f), not the 2dp fixture checks used
    # elsewhere -- this is the highest-precision signal available without touching Python/numpy.
    run bash -c "
        join -t \$'\t' <(grep -v '^#' '$tsv_default' | sort -n) <(grep -v '^#' '$tsv_chunked' | sort -n) |
        awk -F'\t' '
            function ad(a,b) { return (a>b) ? a-b : b-a }
            { for (i=2; i<=5; i++) { d = ad(\$i, \$(i+4)); if (d > max) max = d } }
            END { printf \"%.9f\", max+0 }
        '
    "
    [ "$status" -eq 0 ]
    echo "max abs diff (chunked vs unchunked ONNX Runtime): $output" >&3

    run awk -v d="$output" -v max="$MAX_ABS_DIFF" 'BEGIN { exit !(d < max) }'
    [ "$status" -eq 0 ]
}
