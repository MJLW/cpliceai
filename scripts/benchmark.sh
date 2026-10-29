#!/usr/bin/env bash
# Benchmarks cpliceai_reference's throughput on a synthetic, realistically-sized workload.
#
# The committed bats fixture (tests/fixtures/chrTest.fasta, ~2kb) is dominated by model-load
# time and SpliceAI's fixed 10,000bp context padding, not representative of real gene sizes --
# this generates its own larger synthetic contig (three genes: 20kb/100kb/300kb) instead, with
# no dependency on the real reference genome.
#
# Usage (env-var driven, so the same script covers every backend/precision/EP combination):
#   MODEL_DIR=models/onnx                                  BIN_DIR=build-ort ./scripts/benchmark.sh
#   CPLICEAI_ORT_EP=cuda MODEL_DIR=models/onnx              BIN_DIR=build-ort ./scripts/benchmark.sh
#   CPLICEAI_ORT_EP=cuda MODEL_DIR=models/onnx_fp16         BIN_DIR=build-ort ./scripts/benchmark.sh
#   MODEL_DIR=models/tf                                      BIN_DIR=build-tf  ./scripts/benchmark.sh
#
# Env vars:
#   BIN_DIR    - directory containing cpliceai_reference (default: build)
#   MODEL_DIR  - model directory to pass to cpliceai_reference (default: models/onnx)
#   RUNS       - number of timed repetitions per fixture size (default: 5)
#   CPLICEAI_ORT_EP, CPLICEAI_ORT_INTRA_OP_THREADS, etc. - passed through to the binary unchanged

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$REPO_ROOT/build}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/onnx}"
RUNS="${RUNS:-5}"
REFERENCE_BIN="$BIN_DIR/cpliceai_reference"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib:/opt/onnxruntime/lib"

if [ ! -x "$REFERENCE_BIN" ]; then
    echo "error: $REFERENCE_BIN not found or not executable (set BIN_DIR)" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
echo "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

# --- Fixture generation -----------------------------------------------------------------
# "small" isolates load time (a single ~2kb gene, inference cost is negligible relative to
# model loading). "large" adds three realistically-sized genes (20kb/100kb/300kb) so
# (large - small) approximates inference-only time. Fixed seed for reproducibility.
gen_fixture() {
    local name="$1" seed="$2"
    shift 2
    local sizes=("$@")

    perl -e '
        srand($ARGV[0]);
        my @sizes = @ARGV[1..$#ARGV];
        my @bases = ("A","C","G","T");
        my $total = 0;
        $total += $_ for @sizes;

        open(my $fh, ">", "'"$WORKDIR"'/'"$name"'.fasta") or die;
        print $fh ">chrBench\n";
        my $width = 70;
        my $seq = "";
        for (my $i = 0; $i < $total; $i++) { $seq .= $bases[int(rand(4))]; }
        for (my $i = 0; $i < length($seq); $i += $width) {
            print $fh substr($seq, $i, $width), "\n";
        }
        close($fh);

        my $offset = length(">chrBench\n");
        open(my $fai, ">", "'"$WORKDIR"'/'"$name"'.fasta.fai") or die;
        print $fai "chrBench\t$total\t$offset\t$width\t" . ($width+1) . "\n";
        close($fai);

        open(my $reg, ">", "'"$WORKDIR"'/'"$name"'.regions.tsv") or die;
        my $start = 1;
        my $i = 0;
        for my $size (@sizes) {
            my $end = $start + $size - 1;
            print $reg "GENE$i\tchrBench\t+\t$start\t$end\t$start,\t$end,\n";
            $start = $end + 1;
            $i++;
        }
        close($reg);
    ' "$seed" "${sizes[@]}"
}

gen_fixture small 1 2000
gen_fixture large 2 20000 100000 300000

# --- Timing ------------------------------------------------------------------------------
time_runs() {
    local fixture="$1"
    local times=()
    for i in $(seq 1 "$RUNS"); do
        local start end
        start=$(date +%s.%N)
        echo "$WORKDIR/$fixture.ref.bin"
        "$REFERENCE_BIN" "$MODEL_DIR" "$WORKDIR/$fixture.fasta" "$WORKDIR/$fixture.regions.tsv" "$WORKDIR/$fixture.ref.bin" \
            >"$WORKDIR/$fixture.$i.log" 2>&1 || {
                echo "error: run $i on $fixture fixture failed, see $WORKDIR/$fixture.$i.log" >&2
                cat "$WORKDIR/$fixture.$i.log" >&2
                exit 1
            }
        end=$(date +%s.%N)
        times+=("$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e-s}')")
    done
    # Discard the first run (cold file-cache) unless RUNS==1.
    if [ "$RUNS" -gt 1 ]; then
        times=("${times[@]:1}")
    fi
    printf '%s\n' "${times[@]}" | sort -n
}

# cp "$WORKDIR/large.ref.bin" "."

echo "Configuration: BIN_DIR=$BIN_DIR MODEL_DIR=$MODEL_DIR CPLICEAI_ORT_EP=${CPLICEAI_ORT_EP:-<unset>} RUNS=$RUNS"
echo

echo "small fixture (1 gene, 2kb -- load-time-dominated proxy):"
small_times=$(time_runs small)
echo "$small_times" | awk '{a[NR]=$1; s+=$1} END{n=NR; printf "  min=%.3fs  median=%.3fs  n=%d\n", a[1], a[int((n+1)/2)], n}'

echo
echo "large fixture (3 genes, 20kb/100kb/300kb):"
large_times=$(time_runs large)
echo "$large_times" | awk '{a[NR]=$1; s+=$1} END{n=NR; printf "  min=%.3fs  median=%.3fs  n=%d\n", a[1], a[int((n+1)/2)], n}'

small_median=$(echo "$small_times" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
large_median=$(echo "$large_times" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
echo
awk -v s="$small_median" -v l="$large_median" 'BEGIN {
    printf "Approx. load time (~small fixture median): %.3fs\n", s
    printf "Approx. inference time for 420kb across 3 genes (large - small): %.3fs\n", l-s
}'
