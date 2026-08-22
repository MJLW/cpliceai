#!/usr/bin/env bash
# Benchmarks cpliceai_predict_gene on an SNV-only variant set against one containing indels, to
# measure the per-shape-change cost described in docs/gpu-validation.md section 8.
#
# Why these two files differ: predict_gene rebuilds the alt sequence per variant via
# create_alt_seq() (src/utils.c:60), whose length is `ref_seq->l + alt_len - ref_len`, and feeds
# `alt.l + CONTEXT_SIZE` to the model (predict_gene.c:135). SNVs leave that width unchanged, so
# consecutive variants within a gene reuse one shape; every indel shifts it and the next SNV
# shifts it back, costing two shape changes. ONNX Runtime pays a substantial penalty per change.
#
# Generate the inputs with scripts/gen_variant_benchmark.py (--format tsv), e.g.:
#   scripts/gen_variant_benchmark.py REF.fa.gz REGIONS.tsv snv_2k.tsv    --format tsv --count 2000 --indel-period 0
#   scripts/gen_variant_benchmark.py REF.fa.gz REGIONS.tsv indel_2k.tsv  --format tsv --count 2000 --indel-period 10
#
# Usage:
#   ./scripts/benchmark_predict_gene.sh <fasta> <regions.tsv> <variants1.tsv> [variants2.tsv ...]
#
# Env vars:
#   BIN_DIR         directory holding cpliceai_reference / cpliceai_predict_gene (default: build)
#   MODEL_DIR       model directory (default: models/onnx)
#   REFERENCE_BIN   reuse an existing reference.bin instead of building one (optional)
#   RUNS            repetitions per variant file, best time reported (default: 1)
#   CPLICEAI_ORT_*  passed through unchanged (EP, PREFER_NHWC, MAX_CHUNK_LEN, ...)

set -euo pipefail

if [ "$#" -lt 3 ]; then
    sed -n '2,24p' "$0" >&2
    exit 1
fi

FASTA="$1"; REGIONS="$2"; shift 2
VARIANT_FILES=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$REPO_ROOT/build}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/onnx}"
RUNS="${RUNS:-1}"
REFERENCE_EXE="$BIN_DIR/cpliceai_reference"
PREDICT_GENE_EXE="$BIN_DIR/cpliceai_predict_gene"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib:/opt/onnxruntime/lib"
export CPLICEAI_ORT_TIMING=1        # makes predict_ort.c report time spent inside Run()

for exe in "$REFERENCE_EXE" "$PREDICT_GENE_EXE"; do
    [ -x "$exe" ] || { echo "error: $exe not found or not executable (set BIN_DIR)" >&2; exit 1; }
done
for f in "$FASTA" "$REGIONS" "${VARIANT_FILES[@]}"; do
    [ -r "$f" ] || { echo "error: cannot read $f" >&2; exit 1; }
done

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# --- reference scores (shared by every variant file, so build once) ---------------------------
if [ -n "${REFERENCE_BIN:-}" ]; then
    [ -r "$REFERENCE_BIN" ] || { echo "error: cannot read REFERENCE_BIN=$REFERENCE_BIN" >&2; exit 1; }
    REF_BIN="$REFERENCE_BIN"
    echo "Using existing reference: $REF_BIN"
else
    REF_BIN="$WORKDIR/reference.bin"
    echo "Building reference scores (once; this is the slow part on a large annotation)..."
    "$REFERENCE_EXE" "$MODEL_DIR" "$FASTA" "$REGIONS" "$REF_BIN" > "$WORKDIR/reference.log" 2>&1 || {
        echo "error: cpliceai_reference failed, see below" >&2; tail -20 "$WORKDIR/reference.log" >&2; exit 1; }
    echo "  -> $(du -h "$REF_BIN" | cut -f1)"
fi
echo

# --- per-variant-file timing -------------------------------------------------------------------
printf "%-28s %10s %10s %12s %10s\n" "variant file" "wall(s)" "Run()(s)" "calls" "ms/call"

BASELINE=""
declare -a NAMES=() TOTALS=()

for vf in "${VARIANT_FILES[@]}"; do
    label="$(basename "$vf")"
    best_wall=""; best_run=""; best_calls=""; best_per=""

    for _ in $(seq 1 "$RUNS"); do
        log="$WORKDIR/$label.log"
        start=$(date +%s.%N)
        "$PREDICT_GENE_EXE" "$vf" "$REF_BIN" "$MODEL_DIR" "$FASTA" "$REGIONS" "$WORKDIR/$label.out.tsv" \
            > "$log" 2>&1 || {
            echo "error: cpliceai_predict_gene failed on $vf" >&2; tail -20 "$log" >&2; exit 1; }
        end=$(date +%s.%N)

        wall=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e-s}')
        # "Timing: 8.449s in Run() across 25 calls (337.96 ms/call) | ..." -- absent if BIN_DIR
        # was built before the instrumentation landed.
        timing_line=$(grep -o 'Timing:.*' "$log" | tail -1 || true)
        if [ -n "$timing_line" ]; then
            run_s=$(echo "$timing_line"  | grep -o 'Timing: [0-9.]*'    | awk '{print $2}')
            calls=$(echo "$timing_line"  | grep -o 'across [0-9]* calls' | awk '{print $2}')
            per=$(echo "$timing_line"    | grep -o '([0-9.]* ms/call'    | tr -d '(' | awk '{print $1}')
        else
            run_s="n/a"; calls="n/a"; per="n/a"
        fi

        if [ -z "$best_wall" ] || awk -v a="$wall" -v b="$best_wall" 'BEGIN{exit !(a<b)}'; then
            best_wall="$wall"; best_run="$run_s"; best_calls="$calls"; best_per="$per"
        fi
    done

    printf "%-28s %10s %10s %12s %10s\n" "$label" "$best_wall" "$best_run" "$best_calls" "$best_per"
    NAMES+=("$label"); TOTALS+=("$best_wall")
    [ -z "$BASELINE" ] && BASELINE="$best_wall"
done

if [ -z "$(grep -o 'Timing:' "$WORKDIR"/*.log 2>/dev/null | head -1)" ]; then
    echo
    echo "note: no Timing line found -- $BIN_DIR predates CPLICEAI_ORT_TIMING. Rebuild to get"
    echo "      Run() time and call count (wall-clock columns above are still valid)."
fi

# --- relative comparison -----------------------------------------------------------------------
if [ "${#NAMES[@]}" -gt 1 ]; then
    echo
    echo "Relative to ${NAMES[0]}:"
    for i in "${!NAMES[@]}"; do
        [ "$i" -eq 0 ] && continue
        awk -v n="${NAMES[$i]}" -v t="${TOTALS[$i]}" -v b="$BASELINE" \
            'BEGIN{ printf "  %-26s %+.1f%%  (%.2fx)\n", n, 100*(t-b)/b, t/b }'
    done
fi
