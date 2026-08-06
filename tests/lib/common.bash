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
: "${MODEL_DIR:="$(repo_root)/models/onnx"}"
: "${CPLICEAI_REFERENCE_BIN:="$(repo_root)/build/cpliceai_reference"}"
: "${CPLICEAI_PREDICT_VARIANT_BIN:="$(repo_root)/build/cpliceai_predict_variant"}"
: "${CPLICEAI_PREDICT_GENE_BIN:="$(repo_root)/build/cpliceai_predict_gene"}"

# TensorFlow/htslib are installed to /usr/local/lib and ONNX Runtime to
# /opt/onnxruntime/lib; ldconfig normally registers both, but set this
# defensively as the README instructs.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib:/opt/onnxruntime/lib"

# A per-test scratch dir. Not using bats-core's BATS_TEST_TMPDIR/BATS_FILE_TMPDIR here:
# those are only populated by bats-core >= 1.3.0, and setup()/the test body/teardown()
# all run in the same process regardless of bats version, so a plain mktemp works
# everywhere. Exposed separately so files with their own setup()/teardown() (e.g.
# backend_parity.bats, which checks a different set of binaries) can still reuse it.
common_tmpdir_setup() {
    TEST_TMPDIR="$(mktemp -d)"
}

common_tmpdir_teardown() {
    # Must always return 0: bats runs teardown() even when setup() skipped before ever
    # calling common_tmpdir_setup(), and a non-zero teardown exit code overrides a "skip"
    # outcome, reporting the whole test as failed instead of skipped.
    if [ -n "${TEST_TMPDIR:-}" ]; then
        rm -rf "$TEST_TMPDIR"
    fi
    return 0
}

setup() {
    for bin in "$CPLICEAI_REFERENCE_BIN" "$CPLICEAI_PREDICT_VARIANT_BIN" "$CPLICEAI_PREDICT_GENE_BIN"; do
        [ -x "$bin" ] || skip "binary not built: $bin (build the project before running tests)"
    done
    common_tmpdir_setup
}

teardown() {
    common_tmpdir_teardown
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

# assert_gene_score_tol <tsv> <pos> <tol> <expected ref_acceptor> <ref_donor> <alt_acceptor> <alt_donor>
#
# Like assert_gene_score, but compares each of the 4 raw (unrounded) score columns against
# an expected value within +/- <tol>, instead of an exact 2dp string match. Needed for FP16
# comparisons: a true value like 0.6251 rounds to 0.63 in fp32 and 0.62 in fp16, which
# assert_gene_score's exact-string comparison would flag as a failure even though the
# underlying values only differ by 0.0001.
assert_gene_score_tol() {
    local tsv="$1" pos="$2" tol="$3" e1="$4" e2="$5" e3="$6" e4="$7"
    local row
    row=$(awk -F'\t' -v pos="$pos" '$1==pos { printf "%s\t%s\t%s\t%s", $2, $3, $4, $5 }' "$tsv")
    if [ -z "$row" ]; then
        echo "gene score tolerance check: no row found at pos $pos in $tsv" >&2
        return 1
    fi

    local ok
    ok=$(awk -F'\t' -v tol="$tol" -v e1="$e1" -v e2="$e2" -v e3="$e3" -v e4="$e4" '
        function ad(a,b) { return (a>b) ? a-b : b-a }
        { if (ad($1,e1)<=tol && ad($2,e2)<=tol && ad($3,e3)<=tol && ad($4,e4)<=tol) print "1"; else print "0" }
    ' <<< "$row")

    if [ "$ok" != "1" ]; then
        echo "gene score mismatch (tol $tol) at pos $pos: expected [$e1 $e2 $e3 $e4], got [$row]" >&2
        return 1
    fi
}
