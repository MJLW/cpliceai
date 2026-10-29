/*
 * Unit tests for the haplotype assembly module (src/haplotype.c): genotype-to-copy mask
 * assignment, the sliding-window variant buffer, and edit collection. Each test opens a
 * VariantReader on a small TSV/VCF fixture written to a temp file - no FASTA, regions file, or
 * model - so these run without ever touching prediction.
 */
#include <check.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "haplotype.h"
#include "variant_input.h"

static char tmpdir[] = "/tmp/cpliceai_unit_hapbuf_XXXXXX";

static void suite_setup(void) {
    /* Runs in main(), outside any forked test, so ck_assert (which needs a running test's IPC
       state) is not usable here. */
    if (mkdtemp(tmpdir) == NULL) {
        perror("mkdtemp");
        exit(EXIT_FAILURE);
    }
}

static void suite_teardown(void) {
    DIR *d = opendir(tmpdir);
    if (d != NULL) {
        struct dirent *ent;
        while ((ent = readdir(d)) != NULL) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
            char p[600];
            snprintf(p, sizeof p, "%s/%s", tmpdir, ent->d_name);
            unlink(p);
        }
        closedir(d);
    }
    rmdir(tmpdir);
}

static const char *write_file(const char *name, const char *content) {
    static char path[600];
    snprintf(path, sizeof path, "%s/%s", tmpdir, name);
    FILE *fp = fopen(path, "w");
    ck_assert_ptr_nonnull(fp);
    fputs(content, fp);
    fclose(fp);
    return path;
}

static void open_buffer_ex(const char *path, bool include_unphased, bool local_only, hts_pos_t span,
                           VariantReader **reader_out, HapBuffer **buffer_out) {
    ck_assert_int_eq(variant_reader_open(path, VARIANT_FORMAT_AUTO, reader_out), EXIT_SUCCESS);
    ck_assert_int_eq(hap_buffer_open(*reader_out, include_unphased, local_only, span, buffer_out), EXIT_SUCCESS);
}

static void open_buffer(const char *path, bool include_unphased, hts_pos_t span,
                        VariantReader **reader_out, HapBuffer **buffer_out) {
    open_buffer_ex(path, include_unphased, false, span, reader_out, buffer_out);
}

/* --- genotype -> copy mask assignment, driven through hap_buffer_next -------------------- */

START_TEST(test_no_genotype_scores_alone) {
    const char *path = write_file("v.tsv", "CHROM\tPOS\tREF\tALT\nchr1\t100\tG\tA\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer(path, false, 1000, &reader, &buffer);

    const HapRecord *record;
    ck_assert_int_eq(hap_buffer_next(buffer, &record), EXIT_SUCCESS);

    ck_assert_int_eq(record->ploidy, 0);
    ck_assert(!record->has_gt);
    ck_assert(!record->drop);
    ck_assert_int_eq(record->hap_mask[0], 0);

    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

typedef struct {
    const char *gt;
    bool        include_unphased;
    int         ploidy;
    bool        phased;
    bool        has_gt;
    bool        drop;
    int         mask0;
} MaskCase;

static const MaskCase mask_cases[] = {
    /* gt      incl_unphased  ploidy  phased  has_gt  drop   mask0 */
    { "0/0",   false,         2,      false,  true,   false, 0              },
    { "./.",   false,         0,      false,  false,  false, 0              },
    { "1/1",   false,         2,      false,  true,   false, HAP_1 | HAP_2  },
    { "1|1",   false,         2,      true,   true,   false, HAP_1 | HAP_2  },
    { "0|1",   false,         2,      true,   true,   false, HAP_2          },
    { "1|0",   false,         2,      true,   true,   false, HAP_1          },
    { "0/1",   false,         2,      false,  true,   true,  0              },
    { "0/1",   true,          2,      false,  true,   false, HAP_1 | HAP_2  },
};
#define N_MASK_CASES (sizeof(mask_cases) / sizeof(mask_cases[0]))

START_TEST(test_mask_assignment_table) {
    const MaskCase *c = &mask_cases[_i];

    char content[256];
    snprintf(content, sizeof content, "CHROM\tPOS\tREF\tALT\tGT\nchr1\t100\tG\tA\t%s\n", c->gt);
    const char *path = write_file("v.tsv", content);

    VariantReader *reader; HapBuffer *buffer;
    open_buffer(path, c->include_unphased, 1000, &reader, &buffer);

    const HapRecord *record;
    ck_assert_int_eq(hap_buffer_next(buffer, &record), EXIT_SUCCESS);

    ck_assert_int_eq(record->ploidy, c->ploidy);
    ck_assert_int_eq(record->phased, c->phased);
    ck_assert_int_eq(record->has_gt, c->has_gt);
    ck_assert_int_eq(record->drop, c->drop);
    ck_assert_int_eq(record->hap_mask[0], c->mask0);

    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

START_TEST(test_triploid_vcf_uses_only_first_two_copies) {
    const char *path = write_file("v.vcf",
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        "chr1\t100\t.\tG\tA\t.\t.\t.\tGT\t0|1|1\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer(path, false, 1000, &reader, &buffer);

    const HapRecord *record;
    ck_assert_int_eq(hap_buffer_next(buffer, &record), EXIT_SUCCESS);

    /* Only the first two copies (0, 1) are read; the third '1' is discarded, so this reads as
       an ordinary phased het rather than something homozygous-like. */
    ck_assert_int_eq(record->ploidy, 2);
    ck_assert_int_eq(record->gt[0], 0);
    ck_assert_int_eq(record->gt[1], 1);
    ck_assert_int_eq(record->hap_mask[0], HAP_2);

    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

/* --- local_only: scored as if genotype-less, whatever GT is actually present -------------- */

START_TEST(test_local_only_scores_a_phased_variant_as_if_genotype_less) {
    const char *path = write_file("v.tsv", "CHROM\tPOS\tREF\tALT\tGT\nchr1\t100\tG\tA\t0|1\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer_ex(path, false, true, 1000, &reader, &buffer);

    const HapRecord *record;
    ck_assert_int_eq(hap_buffer_next(buffer, &record), EXIT_SUCCESS);

    /* The genotype itself still round-trips (gt/ploidy/phased untouched)... */
    ck_assert_int_eq(record->ploidy, 2);
    ck_assert_int_eq(record->gt[0], 0);
    ck_assert_int_eq(record->gt[1], 1);
    ck_assert(record->phased);

    /* ...but it has no effect on scoring: same shape as a genuinely genotype-less record. */
    ck_assert(!record->has_gt);
    ck_assert(!record->drop);
    ck_assert_int_eq(record->hap_mask[0], 0);

    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

START_TEST(test_local_only_never_drops_an_unphased_heterozygote) {
    const char *path = write_file("v.tsv", "CHROM\tPOS\tREF\tALT\tGT\nchr1\t100\tG\tA\t0/1\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer_ex(path, false, true, 1000, &reader, &buffer);

    const HapRecord *record;
    ck_assert_int_eq(hap_buffer_next(buffer, &record), EXIT_SUCCESS);

    /* Without --local this would be dropped (see the mask_assignment table above); with it,
       phase is irrelevant and it is scored like any other genotype-less allele. */
    ck_assert(!record->drop);
    ck_assert(!record->has_gt);

    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

START_TEST(test_local_only_never_builds_a_haplotype_background) {
    /* Two variants that would ordinarily be co-phased on the same copy. */
    const char *path = write_file("v.tsv",
        "CHROM\tPOS\tREF\tALT\tGT\n"
        "chr1\t100\tG\tA\t0|1\n"
        "chr1\t120\tG\tC\t0|1\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer_ex(path, false, true, 1000, &reader, &buffer);

    const HapRecord *first;
    ck_assert_int_eq(hap_buffer_next(buffer, &first), EXIT_SUCCESS);

    SeqEditList edits;
    seq_edit_list_init(&edits);

    /* Neither HAP_1 nor HAP_2 ever matches a mask of 0, so nothing is ever collected - not even
       the neighbour that would otherwise be co-phased with it. */
    hap_edits_collect(buffer, HAP_1, "chr1", 0, 0, 1000, first, 0, &edits);
    ck_assert_uint_eq(edits.n, 0);
    hap_edits_collect(buffer, HAP_2, "chr1", 0, 0, 1000, first, 0, &edits);
    ck_assert_uint_eq(edits.n, 0);

    seq_edit_list_destroy(&edits);
    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

/* --- hap_edits_collect: exclude logic ----------------------------------------------------- */

START_TEST(test_edits_collect_excludes_the_variant_under_consideration) {
    const char *path = write_file("v.tsv",
        "CHROM\tPOS\tREF\tALT\tGT\n"
        "chr1\t100\tG\tA\t0|1\n"
        "chr1\t120\tG\tC\t0|1\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer(path, false, 1000, &reader, &buffer);

    const HapRecord *first;
    ck_assert_int_eq(hap_buffer_next(buffer, &first), EXIT_SUCCESS); /* pos 99 (0-based) */

    SeqEditList edits;
    seq_edit_list_init(&edits);

    /* Excluding the first variant should leave only the second (position 119, 0-based). */
    hap_edits_collect(buffer, HAP_2, "chr1", 0, 0, 1000, first, 0, &edits);
    ck_assert_uint_eq(edits.n, 1);
    ck_assert_int_eq(edits.edits[0].gene_pos, 119);

    const HapRecord *second;
    ck_assert_int_eq(hap_buffer_next(buffer, &second), EXIT_SUCCESS);

    /* Excluding the second leaves only the first (position 99). */
    hap_edits_collect(buffer, HAP_2, "chr1", 0, 0, 1000, second, 0, &edits);
    ck_assert_uint_eq(edits.n, 1);
    ck_assert_int_eq(edits.edits[0].gene_pos, 99);

    /* Collecting everything (no exclusion) sees both, in position order. */
    hap_edits_collect(buffer, HAP_2, "chr1", 0, 0, 1000, NULL, 0, &edits);
    ck_assert_uint_eq(edits.n, 2);
    ck_assert_int_eq(edits.edits[0].gene_pos, 99);
    ck_assert_int_eq(edits.edits[1].gene_pos, 119);

    seq_edit_list_destroy(&edits);
    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

/* --- hap_edits_collect: overlap handling -------------------------------------------------- */

START_TEST(test_edits_collect_drops_an_edit_overlapping_an_already_collected_one) {
    /* pos 100 (1-based 100 -> 0-based 99), REF "AAA" spans [99,102); pos 101 (0-based 100)
       falls inside that span, so the two cannot both be applied. */
    const char *path = write_file("v.tsv",
        "CHROM\tPOS\tREF\tALT\tGT\n"
        "chr1\t100\tAAA\tT\t0|1\n"
        "chr1\t101\tA\tG\t0|1\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer(path, false, 1000, &reader, &buffer);

    const HapRecord *first;
    ck_assert_int_eq(hap_buffer_next(buffer, &first), EXIT_SUCCESS);

    SeqEditList edits;
    seq_edit_list_init(&edits);

    hap_edits_collect(buffer, HAP_2, "chr1", 0, 0, 1000, NULL, 0, &edits);

    /* First wins; the overlapping second is left out entirely. */
    ck_assert_uint_eq(edits.n, 1);
    ck_assert_int_eq(edits.edits[0].gene_pos, 99);
    ck_assert_int_eq(edits.edits[0].ref_len, 3);

    seq_edit_list_destroy(&edits);
    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

START_TEST(test_edits_collect_drops_an_edit_overlapping_the_excluded_variant) {
    /* Same overlap, but this time the *first* record is the one being scored (excluded), and
       its REF span must stay free for it - so the overlapping second is left out of its
       background even though it isn't overlapping any other collected edit. */
    const char *path = write_file("v.tsv",
        "CHROM\tPOS\tREF\tALT\tGT\n"
        "chr1\t100\tAAA\tT\t0|1\n"
        "chr1\t101\tA\tG\t0|1\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer(path, false, 1000, &reader, &buffer);

    const HapRecord *first;
    ck_assert_int_eq(hap_buffer_next(buffer, &first), EXIT_SUCCESS);

    SeqEditList edits;
    seq_edit_list_init(&edits);

    hap_edits_collect(buffer, HAP_2, "chr1", 0, 0, 1000, first, 0, &edits);

    ck_assert_uint_eq(edits.n, 0);

    seq_edit_list_destroy(&edits);
    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

/* --- multiallelic + phasing ---------------------------------------------------------------- */

START_TEST(test_multiallelic_genotype_scores_each_allele_on_its_named_copy) {
    const char *path = write_file("v.tsv",
        "CHROM\tPOS\tREF\tALT\tGT\nchr1\t100\tG\tA,T\t1|2\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer(path, false, 1000, &reader, &buffer);

    const HapRecord *record;
    ck_assert_int_eq(hap_buffer_next(buffer, &record), EXIT_SUCCESS);

    ck_assert_int_eq(record->n_alt, 2);
    ck_assert_str_eq(record->alt[0], "A");
    ck_assert_str_eq(record->alt[1], "T");
    ck_assert_int_eq(record->hap_mask[0], HAP_1);
    ck_assert_int_eq(record->hap_mask[1], HAP_2);

    SeqEditList edits;
    seq_edit_list_init(&edits);

    hap_edits_collect(buffer, HAP_1, "chr1", 0, 0, 1000, NULL, 0, &edits);
    ck_assert_uint_eq(edits.n, 1);
    ck_assert_str_eq(edits.edits[0].alt, "A");

    hap_edits_collect(buffer, HAP_2, "chr1", 0, 0, 1000, NULL, 0, &edits);
    ck_assert_uint_eq(edits.n, 1);
    ck_assert_str_eq(edits.edits[0].alt, "T");

    seq_edit_list_destroy(&edits);
    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

/* --- sliding window eviction ---------------------------------------------------------------- */

START_TEST(test_evicted_variant_is_not_collected) {
    /* Three co-phased variants, gaps (20) wider than span (5): each hap_buffer_next call only
       has to look one record ahead to know it's out of range, so eviction happens promptly.
       By the time record 2 (0-based pos 20) is current, record 1 (pos 0) is more than span
       behind it and gets evicted - even though it would otherwise match the hap mask and fall
       inside the [lo,hi) range passed to hap_edits_collect. */
    const char *path = write_file("v.tsv",
        "CHROM\tPOS\tREF\tALT\tGT\n"
        "chr1\t1\tG\tA\t0|1\n"
        "chr1\t21\tG\tA\t0|1\n"
        "chr1\t41\tG\tA\t0|1\n");

    VariantReader *reader; HapBuffer *buffer;
    open_buffer(path, false, 5, &reader, &buffer);

    const HapRecord *record;
    ck_assert_int_eq(hap_buffer_next(buffer, &record), EXIT_SUCCESS);
    ck_assert_int_eq(record->pos, 0);

    ck_assert_int_eq(hap_buffer_next(buffer, &record), EXIT_SUCCESS);
    ck_assert_int_eq(record->pos, 20);

    SeqEditList edits;
    seq_edit_list_init(&edits);

    /* A wide range that would include position 0 if it were still buffered. */
    hap_edits_collect(buffer, HAP_2, "chr1", 0, 0, 1000, record, 0, &edits);

    ck_assert_uint_eq(edits.n, 1);
    ck_assert_int_eq(edits.edits[0].gene_pos, 40);

    seq_edit_list_destroy(&edits);
    hap_buffer_close(buffer);
    variant_reader_close(reader);
}
END_TEST

/* --- seq_edit_list_push_sorted -------------------------------------------------------------- */

START_TEST(test_push_sorted_orders_by_position) {
    SeqEditList list;
    seq_edit_list_init(&list);

    seq_edit_list_push_sorted(&list, (SeqEdit) { .gene_pos = 30, .ref_len = 1, .alt = "A", .alt_len = 1 });
    seq_edit_list_push_sorted(&list, (SeqEdit) { .gene_pos = 10, .ref_len = 1, .alt = "A", .alt_len = 1 });
    seq_edit_list_push_sorted(&list, (SeqEdit) { .gene_pos = 20, .ref_len = 1, .alt = "A", .alt_len = 1 });

    ck_assert_uint_eq(list.n, 3);
    ck_assert_int_eq(list.edits[0].gene_pos, 10);
    ck_assert_int_eq(list.edits[1].gene_pos, 20);
    ck_assert_int_eq(list.edits[2].gene_pos, 30);

    seq_edit_list_destroy(&list);
}
END_TEST

START_TEST(test_push_sorted_is_stable_on_ties) {
    SeqEditList list;
    seq_edit_list_init(&list);

    seq_edit_list_push_sorted(&list, (SeqEdit) { .gene_pos = 5, .ref_len = 1, .alt = "A", .alt_len = 1 });
    seq_edit_list_push_sorted(&list, (SeqEdit) { .gene_pos = 5, .ref_len = 1, .alt = "B", .alt_len = 1 });

    /* Equal positions never trigger a shift, so insertion order is preserved. */
    ck_assert_uint_eq(list.n, 2);
    ck_assert_str_eq(list.edits[0].alt, "A");
    ck_assert_str_eq(list.edits[1].alt, "B");

    seq_edit_list_destroy(&list);
}
END_TEST

static Suite *haplotype_buffer_suite(void) {
    Suite *s = suite_create("haplotype_buffer");

    TCase *tc_mask = tcase_create("mask_assignment");
    tcase_add_test(tc_mask, test_no_genotype_scores_alone);
    tcase_add_loop_test(tc_mask, test_mask_assignment_table, 0, N_MASK_CASES);
    tcase_add_test(tc_mask, test_triploid_vcf_uses_only_first_two_copies);
    suite_add_tcase(s, tc_mask);

    TCase *tc_local = tcase_create("local_only");
    tcase_add_test(tc_local, test_local_only_scores_a_phased_variant_as_if_genotype_less);
    tcase_add_test(tc_local, test_local_only_never_drops_an_unphased_heterozygote);
    tcase_add_test(tc_local, test_local_only_never_builds_a_haplotype_background);
    suite_add_tcase(s, tc_local);

    TCase *tc_collect = tcase_create("edits_collect");
    tcase_add_test(tc_collect, test_edits_collect_excludes_the_variant_under_consideration);
    tcase_add_test(tc_collect, test_edits_collect_drops_an_edit_overlapping_an_already_collected_one);
    tcase_add_test(tc_collect, test_edits_collect_drops_an_edit_overlapping_the_excluded_variant);
    tcase_add_test(tc_collect, test_multiallelic_genotype_scores_each_allele_on_its_named_copy);
    suite_add_tcase(s, tc_collect);

    TCase *tc_window = tcase_create("sliding_window");
    tcase_add_test(tc_window, test_evicted_variant_is_not_collected);
    suite_add_tcase(s, tc_window);

    TCase *tc_push = tcase_create("push_sorted");
    tcase_add_test(tc_push, test_push_sorted_orders_by_position);
    tcase_add_test(tc_push, test_push_sorted_is_stable_on_ties);
    suite_add_tcase(s, tc_push);

    return s;
}

int main(void) {
    suite_setup();

    SRunner *sr = srunner_create(haplotype_buffer_suite());
    srunner_run_all(sr, CK_NORMAL);
    const int failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    suite_teardown();

    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
