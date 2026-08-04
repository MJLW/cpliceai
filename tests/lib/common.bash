# Shared setup for all .bats files in this suite.
#
# CTest injects CPLICEAI_*_BIN, MODEL_DIR and FIXTURES_DIR via the ENVIRONMENT
# test property (see CMakeLists.txt). When a .bats file is run directly with
# `bats` (outside of ctest), fall back to the build/ directory next to the
# source tree and require the caller to have built it first.

repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

: "${FIXTURES_DIR:="$(repo_root)/tests/fixtures"}"
: "${MODEL_DIR:="$(repo_root)/models"}"
: "${CPLICEAI_REFERENCE_BIN:="$(repo_root)/build/cpliceai_reference"}"
: "${CPLICEAI_PREDICT_VARIANT_BIN:="$(repo_root)/build/cpliceai_predict_variant"}"
: "${CPLICEAI_PREDICT_GENE_BIN:="$(repo_root)/build/cpliceai_predict_gene"}"

# TensorFlow/htslib are installed to /usr/local/lib; ldconfig normally
# registers this, but set it defensively as the README instructs.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib"

setup() {
    for bin in "$CPLICEAI_REFERENCE_BIN" "$CPLICEAI_PREDICT_VARIANT_BIN" "$CPLICEAI_PREDICT_GENE_BIN"; do
        [ -x "$bin" ] || skip "binary not built: $bin (build the project before running tests)"
    done
}

# assert_gene_score <tsv> <pos> <expected ref_acceptor> <ref_donor> <alt_acceptor> <alt_donor>
#
# Rounds a predict_gene output row to 2 decimal places before comparing, since
# raw TensorFlow inference isn't guaranteed bit-identical across CPUs/TF
# builds but is stable well within 0.01 for a pinned model + fixed input.
assert_gene_score() {
    local tsv="$1" pos="$2" expected="$3	$4	$5	$6"
    local actual
    actual=$(awk -F'\t' -v pos="$pos" '$1==pos { printf "%.2f\t%.2f\t%.2f\t%.2f", $2, $3, $4, $5 }' "$tsv")
    if [ "$actual" != "$expected" ]; then
        echo "gene score mismatch at pos $pos: expected [$expected], got [$actual]" >&2
        return 1
    fi
}
