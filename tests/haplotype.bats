load 'lib/common'

# Haplotype-aware scoring: a variant read alongside the others sharing its copy of the
# chromosome, rather than on its own against the reference.
#
# Three fields come out of every run. SpliceAI is the variant against the reference genome and
# does not depend on phasing at all. SpliceAI_HAP is the variant against the rest of its own
# copy, and SpliceAI_TOT is that whole copy against the reference; both name the copy they were
# computed on.
#
# The fixtures all sit around chrTest:113, whose G>A destroys the canonical GT donor
# dinucleotide at 113-114 belonging to the site scored 0.63 at position 112 (see pipeline.bats).
# That 0.63 donor loss is the signal every assertion here follows between the fields.
#
# Like input_formats.bats, each test builds its own reference.bin rather than sharing one, to
# avoid depending on bats-core >= 1.3.0's setup_file()/BATS_FILE_TMPDIR.

build_reference() {
    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$TEST_TMPDIR/reference.bin"
    [ "$status" -eq 0 ]
}

# The output path is deliberately not held in a local named `output`: `run` reports what the
# binary printed in $output, and a local of that name inside the helper would swallow it.
predict_variant() {
    local variants="$1" out_path="$2"; shift 2
    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$variants" \
        "$TEST_TMPDIR/reference.bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$out_path" \
        "$@"
}

predict_gene() {
    local variants="$1" out_path="$2"; shift 2
    run "$CPLICEAI_PREDICT_GENE_BIN" \
        "$variants" \
        "$TEST_TMPDIR/reference.bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.tsv" \
        "$out_path" \
        "$@"
}

# field <tsv> <pos> <SpliceAI|SpliceAI_HAP|SpliceAI_TOT> - one annotation field of the row at POS.
field() {
    local column
    case "$3" in
        SpliceAI)     column=6 ;;
        SpliceAI_HAP) column=7 ;;
        SpliceAI_TOT) column=8 ;;
    esac
    awk -F'\t' -v pos="$2" -v c="$column" '$2==pos { print $c }' "$1"
}

# part <annotation> <1-based index> - one pipe-delimited part of an annotation entry.
part() {
    awk -F'|' -v i="$2" '{ print $i }' <<< "$1"
}

# base_at <1-based pos> [length] - the reference base(s) there, so a hand-built fixture never
# disagrees with the fasta.
base_at() {
    awk -v p="$1" -v n="${2:-1}" 'NR>1{s=s$0} END{print substr(s,p,n)}' "$FIXTURES_DIR/chrTest.fasta"
}

@test "a variant with no genotype scores the same on all three fields" {
    build_reference

    # Nothing says which copy this variant is on, so its haplotype is the reference genome and
    # its complete form is the variant alone. The two haplotype fields then have nothing to add
    # and must agree with SpliceAI exactly, rather than drifting by a re-prediction.
    predict_variant "$FIXTURES_DIR/variants.donor.tsv" "$TEST_TMPDIR/out.tsv"
    [ "$status" -eq 0 ]

    local isolated marginal total
    isolated="$(field "$TEST_TMPDIR/out.tsv" 113 SpliceAI)"
    marginal="$(field "$TEST_TMPDIR/out.tsv" 113 SpliceAI_HAP)"
    total="$(field "$TEST_TMPDIR/out.tsv" 113 SpliceAI_TOT)"

    # The donor loss this whole file is built around.
    [ "$(part "$isolated" 6)" = "0.63" ]

    # The haplotype fields carry a copy label the isolated one does not, and '.' where there is
    # no copy to name. Past that they are the same numbers.
    [ "$(part "$marginal" 3)" = "." ]
    [ "$(part "$total" 3)" = "." ]
    [ "$(cut -d'|' -f4- <<< "$marginal")" = "$(cut -d'|' -f3- <<< "$isolated")" ]
    [ "$(cut -d'|' -f4- <<< "$total")" = "$(cut -d'|' -f3- <<< "$isolated")" ]

    # GT round-trips as '.', which reads back as no genotype, so the output is valid input.
    [ "$(awk -F'\t' '$2==113 { print $5 }' "$TEST_TMPDIR/out.tsv")" = "." ]
    predict_variant "$TEST_TMPDIR/out.tsv" "$TEST_TMPDIR/again.tsv"
    [ "$status" -eq 0 ]
    run diff "$TEST_TMPDIR/out.tsv" "$TEST_TMPDIR/again.tsv"
    [ "$status" -eq 0 ]
}

@test "a co-phased neighbour changes the haplotype fields but not SpliceAI" {
    build_reference

    predict_variant "$FIXTURES_DIR/variants.phased.tsv" "$TEST_TMPDIR/out.tsv"
    [ "$status" -eq 0 ]

    # chrTest:130 G>C does almost nothing by itself...
    local isolated total
    isolated="$(field "$TEST_TMPDIR/out.tsv" 130 SpliceAI)"
    [ "$(part "$isolated" 6)" != "0.63" ]

    # ...but it shares a copy with the variant that abolishes the donor site, so the molecule
    # it is really on has lost that site. That is what SpliceAI_TOT reports.
    total="$(field "$TEST_TMPDIR/out.tsv" 130 SpliceAI_TOT)"
    [ "$(part "$total" 3)" = "2" ]
    [ "$(part "$total" 7)" = "0.63" ]

    # Its own contribution to that loss is still nothing: the site was already gone.
    local marginal
    marginal="$(field "$TEST_TMPDIR/out.tsv" 130 SpliceAI_HAP)"
    [ "$(part "$marginal" 7)" != "0.63" ]
}

@test "variants on opposite copies are not in each other's background" {
    build_reference

    # The same two variants as above, phased 0|1 and 1|0. Neither can be on the molecule the
    # other is on, so both see an empty background and all three fields must agree. This is the
    # test that catches a phase direction read backwards.
    predict_variant "$FIXTURES_DIR/variants.antiphased.tsv" "$TEST_TMPDIR/out.tsv"
    [ "$status" -eq 0 ]

    local pos
    for pos in 113 130; do
        local isolated marginal total
        isolated="$(field "$TEST_TMPDIR/out.tsv" "$pos" SpliceAI)"
        marginal="$(field "$TEST_TMPDIR/out.tsv" "$pos" SpliceAI_HAP)"
        total="$(field "$TEST_TMPDIR/out.tsv" "$pos" SpliceAI_TOT)"

        [ "$(cut -d'|' -f4- <<< "$marginal")" = "$(cut -d'|' -f3- <<< "$isolated")" ]
        [ "$(cut -d'|' -f4- <<< "$total")" = "$(cut -d'|' -f3- <<< "$isolated")" ]
    done

    # And they are reported on the copies their genotypes name: 0|1 is the second, 1|0 the first.
    [ "$(part "$(field "$TEST_TMPDIR/out.tsv" 113 SpliceAI_HAP)" 3)" = "2" ]
    [ "$(part "$(field "$TEST_TMPDIR/out.tsv" 130 SpliceAI_HAP)" 3)" = "1" ]
}

@test "a variant on both copies is reported once per copy" {
    build_reference

    # chrTest:113 is 1|1 and chrTest:130 is 0|1, so copy 1 carries only 113 while copy 2 carries
    # both. The two backgrounds differ, so 113 gets two entries that are not the same numbers.
    predict_variant "$FIXTURES_DIR/variants.homalt.tsv" "$TEST_TMPDIR/out.tsv"
    [ "$status" -eq 0 ]

    local marginal
    marginal="$(field "$TEST_TMPDIR/out.tsv" 113 SpliceAI_HAP)"
    [ "$(awk -F, '{print NF}' <<< "$marginal")" -eq 2 ]
    [ "$(part "$(cut -d, -f1 <<< "$marginal")" 3)" = "1" ]
    [ "$(part "$(cut -d, -f2 <<< "$marginal")" 3)" = "2" ]
    [ "$(cut -d, -f1 <<< "$marginal")" != "$(cut -d, -f2 <<< "$marginal")" ]

    # SpliceAI does not depend on the copy, so it stays a single entry.
    [ "$(awk -F, '{print NF}' <<< "$(field "$TEST_TMPDIR/out.tsv" 113 SpliceAI)")" -eq 1 ]
}

@test "an unphased heterozygous variant is dropped unless asked for" {
    build_reference

    # chrTest:113 is 0/1: both alleles are known, which copy each is on is not. There is no
    # haplotype to put it on, so it is left out of the output entirely.
    predict_variant "$FIXTURES_DIR/variants.unphased.tsv" "$TEST_TMPDIR/dropped.tsv"
    [ "$status" -eq 0 ]
    [ -z "$(field "$TEST_TMPDIR/dropped.tsv" 113 SpliceAI)" ]
    [ -n "$(field "$TEST_TMPDIR/dropped.tsv" 130 SpliceAI)" ]

    # Being dropped, it is not in its neighbour's background either, so 130 sees nothing.
    local isolated total
    isolated="$(field "$TEST_TMPDIR/dropped.tsv" 130 SpliceAI)"
    total="$(field "$TEST_TMPDIR/dropped.tsv" 130 SpliceAI_TOT)"
    [ "$(cut -d'|' -f4- <<< "$total")" = "$(cut -d'|' -f3- <<< "$isolated")" ]

    # --include-unphased puts it back, greedily on both copies...
    predict_variant "$FIXTURES_DIR/variants.unphased.tsv" "$TEST_TMPDIR/kept.tsv" --include-unphased
    [ "$status" -eq 0 ]
    [ "$(awk -F, '{print NF}' <<< "$(field "$TEST_TMPDIR/kept.tsv" 113 SpliceAI_HAP)")" -eq 2 ]

    # ...including into the background of the neighbour, which now inherits the donor loss.
    [ "$(part "$(field "$TEST_TMPDIR/kept.tsv" 130 SpliceAI_TOT)" 7)" = "0.63" ]
}

@test "--local ignores genotype and phasing entirely, and drops the haplotype columns" {
    build_reference

    # The same co-phased pair the neighbour-background test above uses: without --local,
    # chrTest:130 inherits chrTest:113's donor loss via SpliceAI_TOT.
    predict_variant "$FIXTURES_DIR/variants.phased.tsv" "$TEST_TMPDIR/local.tsv" --local
    [ "$status" -eq 0 ]

    # SpliceAI_HAP and SpliceAI_TOT are not just redundant, they are absent - the header carries
    # six columns, not eight, and every row matches it.
    [ "$(head -1 "$TEST_TMPDIR/local.tsv")" = "$(printf 'CHROM\tPOS\tREF\tALT\tGT\tSpliceAI')" ]
    [ "$(awk -F'\t' 'NR>1 {print NF}' "$TEST_TMPDIR/local.tsv" | sort -u)" = "6" ]

    # Both variants still score, on their own, at the position the isolated donor-loss test
    # expects (chrTest:113 destroys the site scored 0.63 at 112).
    [ "$(part "$(field "$TEST_TMPDIR/local.tsv" 113 SpliceAI)" 6)" = "0.63" ]

    # The genotype itself still round-trips to the GT column even though it drove none of the
    # scoring above.
    [ "$(awk -F'\t' '$2==113 { print $5 }' "$TEST_TMPDIR/local.tsv")" = "0|1" ]

    # An unphased heterozygote, normally dropped, is scored instead of skipped.
    predict_variant "$FIXTURES_DIR/variants.unphased.tsv" "$TEST_TMPDIR/local_unphased.tsv" --local
    [ "$status" -eq 0 ]
    [ -n "$(field "$TEST_TMPDIR/local_unphased.tsv" 113 SpliceAI)" ]

    # A VCF gets only INFO/SpliceAI in its header, not the other two.
    predict_variant "$FIXTURES_DIR/variants.phased.vcf" "$TEST_TMPDIR/local.vcf" --local
    [ "$status" -eq 0 ]
    run bash -c "grep -c '^##INFO=<ID=SpliceAI' '$TEST_TMPDIR/local.vcf'"
    [ "$output" -eq 1 ]
    run bash -c "grep -c '^##INFO=<ID=SpliceAI_HAP\|^##INFO=<ID=SpliceAI_TOT' '$TEST_TMPDIR/local.vcf'"
    [ "$output" -eq 0 ]
}

@test "a co-phased indel leaves reported positions relative to the reference" {
    build_reference

    # A 2bp deletion at chrTest:101 sits on the same copy as chrTest:113, so every base after it
    # is 2 further along the haplotype than it is in the reference. Delta positions are reported
    # against the reference, so they must not move with it - if the alignment back from
    # haplotype coordinates is wrong, this is where it shows.
    predict_variant "$FIXTURES_DIR/variants.phased_indel.tsv" "$TEST_TMPDIR/out.tsv"
    [ "$status" -eq 0 ]

    # The donor loss is one base upstream of chrTest:113, at the site it destroyed, on all
    # three fields - the background deletion does not move it to +1.
    local f
    for f in SpliceAI SpliceAI_HAP SpliceAI_TOT; do
        [ "$(part "$(field "$TEST_TMPDIR/out.tsv" 113 "$f")" "$([ "$f" = SpliceAI ] && echo 10 || echo 11)")" = "-1" ]
    done

    # SpliceAI never sees the background at all, so it is exactly the score the same variant
    # gets on its own.
    printf 'CHROM\tPOS\tREF\tALT\tGT\nchrTest\t113\tG\tA\t0|1\n' > "$TEST_TMPDIR/solo.tsv"
    predict_variant "$TEST_TMPDIR/solo.tsv" "$TEST_TMPDIR/solo.out.tsv"
    [ "$status" -eq 0 ]
    [ "$(field "$TEST_TMPDIR/out.tsv" 113 SpliceAI)" = "$(field "$TEST_TMPDIR/solo.out.tsv" 113 SpliceAI)" ]
}

@test "a phased VCF scores identically to the same genotypes as a TSV" {
    build_reference

    predict_variant "$FIXTURES_DIR/variants.phased.vcf" "$TEST_TMPDIR/from_vcf.vcf"
    [ "$status" -eq 0 ]
    predict_variant "$FIXTURES_DIR/variants.phased.tsv" "$TEST_TMPDIR/from_tsv.tsv"
    [ "$status" -eq 0 ]

    local tag column
    for tag in SpliceAI SpliceAI_HAP SpliceAI_TOT; do
        case "$tag" in
            SpliceAI)     column=6 ;;
            SpliceAI_HAP) column=7 ;;
            SpliceAI_TOT) column=8 ;;
        esac
        run bash -c "diff <(bcftools query -f '%INFO/$tag\n' '$TEST_TMPDIR/from_vcf.vcf') \
                          <(tail -n +2 '$TEST_TMPDIR/from_tsv.tsv' | cut -f$column)"
        [ "$status" -eq 0 ]
    done
}

@test "predict_gene writes a block per copy, with four tracks or two" {
    build_reference

    predict_gene "$FIXTURES_DIR/variants.phased.tsv" "$TEST_TMPDIR/full.tsv"
    [ "$status" -eq 0 ]

    # One block per (allele, gene, copy); both variants are on copy 2 only.
    run grep -c '^#GENE1_+_0_2000:.*:HAP2$' "$TEST_TMPDIR/full.tsv"
    [ "$output" -eq 2 ]

    # POS then an acceptor/donor pair for each of REF, ALT, HAP_REF, HAP_ALT.
    run bash -c "grep -vc '^#' '$TEST_TMPDIR/full.tsv'"
    local rows="$output"
    run bash -c "grep -v '^#' '$TEST_TMPDIR/full.tsv' | awk -F'\t' 'NF==9' | wc -l"
    [ "$output" -eq "$rows" ]

    # The two variants share a copy, so its complete sequence - the HAP_ALT pair - is the same
    # for both blocks, while each one's HAP_REF is the other one's ALT.
    local first second
    first="$(awk '/:HAP2$/{n++} n==1 && $1==112 { print }' "$TEST_TMPDIR/full.tsv")"
    second="$(awk '/:HAP2$/{n++} n==2 && $1==112 { print }' "$TEST_TMPDIR/full.tsv")"
    [ -n "$first" ]
    [ "$(cut -f8,9 <<< "$first")" = "$(cut -f8,9 <<< "$second")" ]
    [ "$(cut -f4,5 <<< "$first")" = "$(cut -f6,7 <<< "$second")" ]
    [ "$(cut -f6,7 <<< "$first")" = "$(cut -f4,5 <<< "$second")" ]

    # --ref-hapalt-only keeps POS and the REF and HAP_ALT pairs, dropping the working.
    predict_gene "$FIXTURES_DIR/variants.phased.tsv" "$TEST_TMPDIR/slim.tsv" --ref-hapalt-only
    [ "$status" -eq 0 ]
    run bash -c "grep -v '^#' '$TEST_TMPDIR/slim.tsv' | awk -F'\t' 'NF==5' | wc -l"
    [ "$output" -eq "$rows" ]

    local slim
    slim="$(awk '/:HAP2$/{n++} n==1 && $1==112 { print }' "$TEST_TMPDIR/slim.tsv")"
    [ "$(cut -f2,3 <<< "$slim")" = "$(cut -f2,3 <<< "$first")" ]
    [ "$(cut -f4,5 <<< "$slim")" = "$(cut -f8,9 <<< "$first")" ]
}

@test "predict_gene --local keeps only REF and ALT" {
    build_reference

    # Same variant, run normally (four tracks) and with --local (genotype/phasing ignored).
    predict_gene "$FIXTURES_DIR/variants.donor.tsv" "$TEST_TMPDIR/full.tsv"
    [ "$status" -eq 0 ]
    predict_gene "$FIXTURES_DIR/variants.donor.tsv" "$TEST_TMPDIR/local.tsv" --local
    [ "$status" -eq 0 ]

    # No copy to name, same as an ordinary genotype-less run.
    run grep -c ':HAP\.$' "$TEST_TMPDIR/local.tsv"
    [ "$output" -eq 1 ]

    # Five columns (POS + REF + ALT), not nine.
    run bash -c "grep -v '^#' '$TEST_TMPDIR/local.tsv' | awk -F'\t' 'NF==5' | wc -l"
    local rows="$output"
    [ "$rows" -gt 0 ]

    # REF and ALT themselves are unchanged from the full run - --local only drops HAP_REF/HAP_ALT,
    # it does not change how REF or ALT are scored.
    local full_row local_row
    full_row="$(awk '$1==112 { print }' "$TEST_TMPDIR/full.tsv")"
    local_row="$(awk '$1==112 { print }' "$TEST_TMPDIR/local.tsv")"
    [ "$(cut -f1-5 <<< "$full_row")" = "$local_row" ]
}

@test "unsorted input fails instead of assembling the wrong haplotype" {
    build_reference

    # Haplotypes are assembled from a sliding window, so a variant arriving after the window has
    # moved past it would silently be left out of its neighbours' backgrounds.
    printf 'CHROM\tPOS\tREF\tALT\tGT\nchrTest\t130\tG\tC\t0|1\nchrTest\t113\tG\tA\t0|1\n' > "$TEST_TMPDIR/unsorted.tsv"
    predict_variant "$TEST_TMPDIR/unsorted.tsv" "$TEST_TMPDIR/unsorted.out.tsv"
    [ "$status" -ne 0 ]
    [[ "$output" == *"sorted"* ]]
}

@test "co-phased variants that overlap each other score as if isolated, with a warning" {
    build_reference

    # chrTest:101 GCT>A spans [101,103], chrTest:102 C>T sits inside that span: two edits can
    # never both replace the same base. Whichever one is being scored, the other overlaps the
    # space its own REF span needs, so each ends up with an empty background - not a corrupted
    # one - and a warning is logged for both.
    predict_variant "$FIXTURES_DIR/variants.overlap.tsv" "$TEST_TMPDIR/out.tsv"
    [ "$status" -eq 0 ]
    [[ "$output" == *"overlaps"* ]]

    local pos
    for pos in 101 102; do
        local isolated marginal total
        isolated="$(field "$TEST_TMPDIR/out.tsv" "$pos" SpliceAI)"
        marginal="$(field "$TEST_TMPDIR/out.tsv" "$pos" SpliceAI_HAP)"
        total="$(field "$TEST_TMPDIR/out.tsv" "$pos" SpliceAI_TOT)"

        [ "$(cut -d'|' -f4- <<< "$marginal")" = "$(cut -d'|' -f3- <<< "$isolated")" ]
        [ "$(cut -d'|' -f4- <<< "$total")" = "$(cut -d'|' -f3- <<< "$isolated")" ]
    done
}

@test "a genotype naming more than two copies uses only the first two" {
    build_reference

    # chrTest:113 is 0|1|1 - three copies. Only the first two are read, so this must score
    # exactly like the ordinary 0|1 case (variants.phased.vcf's first record), not like a
    # variant present on both copies. The fixture is a VCF, so its output is too - read it back
    # with bcftools rather than the TSV-column field()/part() helpers.
    predict_variant "$FIXTURES_DIR/variants.triploid.vcf" "$TEST_TMPDIR/out.vcf"
    [ "$status" -eq 0 ]
    [[ "$output" == *"3-ploid"* ]]

    run bash -c "bcftools query -f '%INFO/SpliceAI_HAP\n' '$TEST_TMPDIR/out.vcf'"
    [ "$status" -eq 0 ]
    [ "$(awk -F, '{print NF}' <<< "$output")" -eq 1 ]
    [ "$(part "$output" 3)" = "2" ]
    [ "$(part "$output" 7)" = "0.63" ]
}

@test "a multiallelic record's genotype scores each ALT on its named copy" {
    build_reference

    # chrTest:1000 G>A,T is 1|2: A sits on copy 1 alone, T on copy 2 alone. chrTest:1010 C>G is
    # 0|1, co-phased with T's copy only. So A's background is empty (its HAP field matches its
    # isolated score exactly) while T's is not.
    predict_variant "$FIXTURES_DIR/variants.multiallelic.phased.tsv" "$TEST_TMPDIR/out.tsv"
    [ "$status" -eq 0 ]

    local isolated marginal
    isolated="$(field "$TEST_TMPDIR/out.tsv" 1000 SpliceAI)"
    marginal="$(field "$TEST_TMPDIR/out.tsv" 1000 SpliceAI_HAP)"

    local isolated_a isolated_t marginal_a marginal_t
    isolated_a="$(cut -d, -f1 <<< "$isolated")"
    isolated_t="$(cut -d, -f2 <<< "$isolated")"
    marginal_a="$(cut -d, -f1 <<< "$marginal")"
    marginal_t="$(cut -d, -f2 <<< "$marginal")"

    [ "$(part "$marginal_a" 3)" = "1" ]
    [ "$(part "$marginal_t" 3)" = "2" ]
    [ "$(cut -d'|' -f4- <<< "$marginal_a")" = "$(cut -d'|' -f3- <<< "$isolated_a")" ]
    [ "$(cut -d'|' -f4- <<< "$marginal_t")" != "$(cut -d'|' -f3- <<< "$isolated_t")" ]
}

@test "an oversized indel is skipped for its own report but still enters a phased neighbour's background" {
    build_reference

    # 600 bases, one more than the default --window-radius of 500: predict_variant refuses to
    # score it directly (see boundaries.bats), but hap_edits_collect has no window_radius limit
    # of its own, so the buffered variant can still be pulled into a nearby phased neighbour's
    # background even though its own row comes back unscored.
    local big
    big="$(base_at 700 600)"
    printf 'CHROM\tPOS\tREF\tALT\tGT\nchrTest\t113\tG\tA\t0|1\nchrTest\t700\t%s\t%s\t0|1\n' \
        "$big" "${big:0:1}" > "$TEST_TMPDIR/oversized.tsv"

    predict_variant "$TEST_TMPDIR/oversized.tsv" "$TEST_TMPDIR/oversized.out.tsv"
    [ "$status" -eq 0 ]
    [[ "$output" == *"Oversized indel"* ]]

    [ "$(field "$TEST_TMPDIR/oversized.out.tsv" 700 SpliceAI)" = "." ]

    local isolated marginal
    isolated="$(field "$TEST_TMPDIR/oversized.out.tsv" 113 SpliceAI)"
    marginal="$(field "$TEST_TMPDIR/oversized.out.tsv" 113 SpliceAI_HAP)"
    [ "$(cut -d'|' -f4- <<< "$marginal")" != "$(cut -d'|' -f3- <<< "$isolated")" ]
}

@test "a variant outside the gene does not enter a neighbour's haplotype background" {
    run "$CPLICEAI_REFERENCE_BIN" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.interior.tsv" \
        "$TEST_TMPDIR/reference.interior.bin"
    [ "$status" -eq 0 ]

    # GENE2 is 0-based [500,1500), i.e. 1-based POS 501-1500. POS 500 sits one base outside it,
    # co-phased with the first base inside.
    local outside inside
    outside="$(base_at 500)"
    inside="$(base_at 501)"
    printf 'CHROM\tPOS\tREF\tALT\tGT\nchrTest\t500\t%s\tT\t0|1\nchrTest\t501\t%s\tT\t0|1\n' \
        "$outside" "$inside" > "$TEST_TMPDIR/boundary.tsv"

    run "$CPLICEAI_PREDICT_VARIANT_BIN" \
        "$TEST_TMPDIR/boundary.tsv" \
        "$TEST_TMPDIR/reference.interior.bin" \
        "$MODEL_DIR" \
        "$FIXTURES_DIR/chrTest.fasta" \
        "$FIXTURES_DIR/regions.interior.tsv" \
        "$TEST_TMPDIR/boundary.out.tsv"
    [ "$status" -eq 0 ]

    # The out-of-gene variant is written through unscored...
    [ "$(field "$TEST_TMPDIR/boundary.out.tsv" 500 SpliceAI)" = "." ]

    # ...and, despite being co-phased, never reaches the in-gene neighbour's background: the
    # gene bounds clamp hap_edits_collect's search range before phasing even comes into it.
    local isolated marginal total
    isolated="$(field "$TEST_TMPDIR/boundary.out.tsv" 501 SpliceAI)"
    marginal="$(field "$TEST_TMPDIR/boundary.out.tsv" 501 SpliceAI_HAP)"
    total="$(field "$TEST_TMPDIR/boundary.out.tsv" 501 SpliceAI_TOT)"
    [ "$(cut -d'|' -f4- <<< "$marginal")" = "$(cut -d'|' -f3- <<< "$isolated")" ]
    [ "$(cut -d'|' -f4- <<< "$total")" = "$(cut -d'|' -f3- <<< "$isolated")" ]
}

@test "phase sets are not consulted" {
    build_reference

    # Two variants tagged with different FORMAT/PS values, phased 0|1 relative to each other.
    # Phase-set-aware tooling would treat them as independently phased and refuse to combine
    # them; this tool never reads PS, so they are still paired as one haplotype. This test exists
    # so that starting to honour PS is a deliberate, visible change here, not a silent one (see
    # the README's phase-set caveat).
    cat > "$TEST_TMPDIR/ps.vcf" <<'VCF'
##fileformat=VCFv4.2
##contig=<ID=chrTest,length=2000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1
chrTest	113	.	G	A	.	.	.	GT:PS	0|1:100
chrTest	130	.	G	C	.	.	.	GT:PS	0|1:200
VCF

    predict_variant "$TEST_TMPDIR/ps.vcf" "$TEST_TMPDIR/ps.out.vcf"
    [ "$status" -eq 0 ]

    run bash -c "bcftools query -f '%POS\t%INFO/SpliceAI_TOT\n' '$TEST_TMPDIR/ps.out.vcf' | awk -F'\t' '\$1==130'"
    [ "$status" -eq 0 ]
    [[ "$output" == *"0.63"* ]]
}
