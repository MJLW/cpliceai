/*
 * Unit tests for the SeqEdit coordinate/sequence machinery in src/utils.c: the pure functions
 * that map positions and sequences between a gene's reference coordinates and a haplotype
 * edited by one or more variants. No model, no CLI binary, no fixture files - these are called
 * directly.
 */
#include <check.h>
#include <stdlib.h>
#include <string.h>

#include <htslib/kstring.h>

#include "utils.h"

/* --- seq_edits_ref_to_hap ------------------------------------------------------------- */

START_TEST(test_ref_to_hap_no_edits_is_identity) {
    ck_assert_int_eq(seq_edits_ref_to_hap(NULL, 0, 0), 0);
    ck_assert_int_eq(seq_edits_ref_to_hap(NULL, 0, 42), 42);
}
END_TEST

START_TEST(test_ref_to_hap_substitution_no_shift) {
    /* Net-zero length change: everything maps to itself, including inside the REF span. */
    const SeqEdit edits[] = { { .gene_pos = 10, .ref_len = 1, .alt = "A", .alt_len = 1 } };

    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 5), 5);
    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 10), 10);
    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 15), 15);
}
END_TEST

START_TEST(test_ref_to_hap_insertion_shifts_downstream) {
    /* gene_pos=10, REF "A" (len 1) -> ALT "CCC" (len 3): net +2. */
    const SeqEdit edits[] = { { .gene_pos = 10, .ref_len = 1, .alt = "CCC", .alt_len = 3 } };

    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 5), 5);
    /* Inside the one-base REF span: maps to where the ALT begins. */
    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 10), 10);
    /* Past it: shifted by the +2 the insertion added. */
    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 11), 13);
}
END_TEST

START_TEST(test_ref_to_hap_deletion_shifts_downstream) {
    /* gene_pos=10, REF "AAA" (len 3) -> ALT "A" (len 1): net -2, REF span [10, 13). */
    const SeqEdit edits[] = { { .gene_pos = 10, .ref_len = 3, .alt = "A", .alt_len = 1 } };

    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 9), 9);
    /* Anywhere inside the deleted span maps to the ALT's single base. */
    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 12), 10);
    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 1, 13), 11);
}
END_TEST

START_TEST(test_ref_to_hap_cumulative_shift_across_edits) {
    const SeqEdit edits[] = {
        { .gene_pos = 5,  .ref_len = 1, .alt = "A",     .alt_len = 1 }, /* net  0 */
        { .gene_pos = 20, .ref_len = 2, .alt = "AAAAA", .alt_len = 5 }, /* net +3 */
    };

    ck_assert_int_eq(seq_edits_ref_to_hap(edits, 2, 25), 28);
}
END_TEST

/* --- seq_edits_hap_len ----------------------------------------------------------------- */

START_TEST(test_hap_len_no_edits) {
    ck_assert_int_eq(seq_edits_hap_len(NULL, 0, 100), 100);
}
END_TEST

START_TEST(test_hap_len_mixed_edits) {
    const SeqEdit edits[] = {
        { .gene_pos = 5,  .ref_len = 1, .alt = "AAA", .alt_len = 3 }, /* net +2 */
        { .gene_pos = 20, .ref_len = 4, .alt = "A",   .alt_len = 1 }, /* net -3 */
    };

    ck_assert_int_eq(seq_edits_hap_len(edits, 2, 100), 99);
}
END_TEST

/* --- seq_edits_snap ---------------------------------------------------------------------- */

START_TEST(test_snap_noop_when_already_outside_every_edit) {
    const SeqEdit edits[] = { { .gene_pos = 10, .ref_len = 2, .alt = "A", .alt_len = 1 } };
    int64_t lo = 0, hi = 5;

    seq_edits_snap(edits, 1, &lo, &hi);

    ck_assert_int_eq(lo, 0);
    ck_assert_int_eq(hi, 5);
}
END_TEST

START_TEST(test_snap_widens_edge_inside_one_edit) {
    /* REF span [10, 13). ref_lo=11 falls inside it and must widen out to 10. */
    const SeqEdit edits[] = { { .gene_pos = 10, .ref_len = 3, .alt = "A", .alt_len = 1 } };
    int64_t lo = 11, hi = 20;

    seq_edits_snap(edits, 1, &lo, &hi);

    ck_assert_int_eq(lo, 10);
    ck_assert_int_eq(hi, 20);
}
END_TEST

START_TEST(test_snap_widens_both_edges_across_different_edits) {
    /* REF spans [10,13) and [20,23); lo=11 and hi=21 each fall inside a different one. */
    const SeqEdit edits[] = {
        { .gene_pos = 10, .ref_len = 3, .alt = "A", .alt_len = 1 },
        { .gene_pos = 20, .ref_len = 3, .alt = "A", .alt_len = 1 },
    };
    int64_t lo = 11, hi = 21;

    seq_edits_snap(edits, 2, &lo, &hi);

    ck_assert_int_eq(lo, 10);
    ck_assert_int_eq(hi, 23);
}
END_TEST

/* --- build_hap_window -------------------------------------------------------------------- */

static kstring_t make_seq(const char *s) {
    kstring_t ks = { 0, 0, NULL };
    kputs(s, &ks);
    return ks;
}

START_TEST(test_build_hap_window_entirely_inside_an_insertion) {
    /* "AAAAAAAAAA" (len 10), edit at gene_pos=3 replaces 1 base with "CCCCC" (net +4). The
       edited sequence is "AAA" + "CCCCC" + "AAAAAA". A window fully inside the insertion reads
       straight out of the ALT allele. */
    kstring_t gene_seq = make_seq("AAAAAAAAAA");
    const SeqEdit edits[] = { { .gene_pos = 3, .ref_len = 1, .alt = "CCCCC", .alt_len = 5 } };
    char out[3];

    build_hap_window(&gene_seq, edits, 1, 4, 7, out);

    ck_assert_int_eq(memcmp(out, "CCC", 3), 0);
    free(gene_seq.s);
}
END_TEST

START_TEST(test_build_hap_window_spans_an_edit_boundary) {
    kstring_t gene_seq = make_seq("AAAAAAAAAA");
    const SeqEdit edits[] = { { .gene_pos = 3, .ref_len = 1, .alt = "CCCCC", .alt_len = 5 } };
    char out[4];

    /* hap offsets [2,6): one reference base before the edit, then the first three of the ALT. */
    build_hap_window(&gene_seq, edits, 1, 2, 6, out);

    ck_assert_int_eq(memcmp(out, "ACCC", 4), 0);
    free(gene_seq.s);
}
END_TEST

START_TEST(test_build_hap_window_pads_past_gene_start_with_n) {
    kstring_t gene_seq = make_seq("AAAAAAAAAA");
    const SeqEdit edits[] = { { .gene_pos = 3, .ref_len = 1, .alt = "CCCCC", .alt_len = 5 } };
    char out[5];

    /* hap offsets [-2,3): two bases before the gene start (padded 'N'), then three reference
       bases; the edit at hap offset 3 lies entirely past this window. */
    build_hap_window(&gene_seq, edits, 1, -2, 3, out);

    ck_assert_int_eq(memcmp(out, "NNAAA", 5), 0);
    free(gene_seq.s);
}
END_TEST

/* --- create_alt_seq_multi ------------------------------------------------------------------ */

START_TEST(test_create_alt_seq_multi_single_edit_matches_create_alt_seq) {
    kstring_t gene_seq = make_seq("AAAAAAAAAA");
    const SeqEdit edit = { .gene_pos = 3, .ref_len = 1, .alt = "CCCCC", .alt_len = 5 };

    char *multi_seq; size_t multi_len;
    create_alt_seq_multi(&gene_seq, &edit, 1, &multi_seq, &multi_len);

    char *single_seq; size_t single_len;
    create_alt_seq(&gene_seq, 3, 1, 5, "CCCCC", &single_seq, &single_len);

    ck_assert_uint_eq(multi_len, single_len);
    ck_assert_int_eq(memcmp(multi_seq, single_seq, multi_len), 0);

    free(gene_seq.s);
    free(multi_seq);
    free(single_seq);
}
END_TEST

START_TEST(test_create_alt_seq_multi_two_nonoverlapping_edits) {
    kstring_t gene_seq = make_seq("AAAAAAAAAA"); /* len 10 */
    const SeqEdit edits[] = {
        { .gene_pos = 2, .ref_len = 1, .alt = "TT", .alt_len = 2 }, /* insertion, net +1 */
        { .gene_pos = 6, .ref_len = 2, .alt = "G",  .alt_len = 1 }, /* deletion,  net -1 */
    };

    char *seq; size_t len;
    create_alt_seq_multi(&gene_seq, edits, 2, &seq, &len);

    /* "AA" + "TT" + "AAA" (gene[3..6)) + "G" + "AA" (gene[8..10)) */
    ck_assert_uint_eq(len, 10);
    ck_assert_int_eq(memcmp(seq, "AATTAAAGAA", 10), 0);

    free(gene_seq.s);
    free(seq);
}
END_TEST

/* --- align_predictions_multi ---------------------------------------------------------------
 *
 * Score layout per position is [ref_prob, acceptor, donor] (NUM_SCORES == 3).
 */

START_TEST(test_align_predictions_multi_insertion_collapses_to_strongest_site) {
    const SeqEdit edits[] = { { .gene_pos = 5, .ref_len = 1, .alt = "CCC", .alt_len = 3 } };

    /* hap has 12 positions: [0,5) unedited, [5,8) the 3-base ALT, [8,12) the trailing reference
       (mapping back to ref positions [6,10)). */
    float hap[12 * NUM_SCORES] = {
        1,0,0,  1,0,0,  1,0,0,  1,0,0,  1,0,0,       /* hap 0..4  -> ref 0..4 */
        0.5f,0.2f,0.3f,                               /* hap 5    -> ref 5, shared base */
        0.1f,0.7f,0.1f,                               /* hap 6    -> inserted, max acceptor */
        0.2f,0.1f,0.6f,                               /* hap 7    -> inserted, max donor */
        1,0,0,  1,0,0,  1,0,0,  1,0,0,                /* hap 8..11 -> ref 6..9 */
    };
    float out[10 * NUM_SCORES];

    align_predictions_multi(edits, 1, 0, 10, hap, out);

    for (int i = 0; i < 5; i++) {
        ck_assert_float_eq(out[i * NUM_SCORES + 0], 1);
        ck_assert_float_eq(out[i * NUM_SCORES + 1], 0);
        ck_assert_float_eq(out[i * NUM_SCORES + 2], 0);
    }

    /* Position 5: strongest acceptor (0.7, from hap 6) and strongest donor (0.6, from hap 7),
       reported together at the position the insertion is anchored to. */
    ck_assert_float_eq_tol(out[5 * NUM_SCORES + 1], 0.7f, 1e-6);
    ck_assert_float_eq_tol(out[5 * NUM_SCORES + 2], 0.6f, 1e-6);
    ck_assert_float_eq(out[5 * NUM_SCORES + 0], 0); /* 0.7+0.6 > 1.0, clipped to 0 */

    for (int i = 6; i < 10; i++) {
        ck_assert_float_eq(out[i * NUM_SCORES + 0], 1);
        ck_assert_float_eq(out[i * NUM_SCORES + 1], 0);
        ck_assert_float_eq(out[i * NUM_SCORES + 2], 0);
    }
}
END_TEST

START_TEST(test_align_predictions_multi_deletion_fills_no_site) {
    const SeqEdit edits[] = { { .gene_pos = 5, .ref_len = 3, .alt = "C", .alt_len = 1 } };

    /* hap has 8 positions: [0,5) unedited, [5,6) the shared base, [6,8) the trailing reference
       (mapping back to ref positions [8,10) - the two deleted ref bases 6,7 have no hap
       counterpart at all). */
    float hap[8 * NUM_SCORES] = {
        1,0,0,  1,0,0,  1,0,0,  1,0,0,  1,0,0,   /* hap 0..4 -> ref 0..4 */
        0.4f,0.15f,0.25f,                         /* hap 5   -> ref 5, shared base */
        0.6f,0.05f,0.35f,                         /* hap 6   -> ref 8 */
        0.7f,0.02f,0.11f,                         /* hap 7   -> ref 9 */
    };
    float out[10 * NUM_SCORES];

    align_predictions_multi(edits, 1, 0, 10, hap, out);

    for (int i = 0; i < 5; i++) {
        ck_assert_float_eq(out[i * NUM_SCORES + 0], 1);
    }

    ck_assert_float_eq_tol(out[5 * NUM_SCORES + 0], 0.4f, 1e-6);
    ck_assert_float_eq_tol(out[5 * NUM_SCORES + 1], 0.15f, 1e-6);
    ck_assert_float_eq_tol(out[5 * NUM_SCORES + 2], 0.25f, 1e-6);

    /* Ref positions 6 and 7 were deleted: "no site here", not a guess at a score. */
    for (int i = 6; i < 8; i++) {
        ck_assert_float_eq(out[i * NUM_SCORES + 0], 1);
        ck_assert_float_eq(out[i * NUM_SCORES + 1], 0);
        ck_assert_float_eq(out[i * NUM_SCORES + 2], 0);
    }

    ck_assert_float_eq_tol(out[8 * NUM_SCORES + 0], 0.6f, 1e-6);
    ck_assert_float_eq_tol(out[8 * NUM_SCORES + 1], 0.05f, 1e-6);
    ck_assert_float_eq_tol(out[8 * NUM_SCORES + 2], 0.35f, 1e-6);
    ck_assert_float_eq_tol(out[9 * NUM_SCORES + 0], 0.7f, 1e-6);
    ck_assert_float_eq_tol(out[9 * NUM_SCORES + 1], 0.02f, 1e-6);
    ck_assert_float_eq_tol(out[9 * NUM_SCORES + 2], 0.11f, 1e-6);
}
END_TEST

static Suite *seqedit_suite(void) {
    Suite *s = suite_create("seqedit");

    TCase *tc_ref_to_hap = tcase_create("ref_to_hap");
    tcase_add_test(tc_ref_to_hap, test_ref_to_hap_no_edits_is_identity);
    tcase_add_test(tc_ref_to_hap, test_ref_to_hap_substitution_no_shift);
    tcase_add_test(tc_ref_to_hap, test_ref_to_hap_insertion_shifts_downstream);
    tcase_add_test(tc_ref_to_hap, test_ref_to_hap_deletion_shifts_downstream);
    tcase_add_test(tc_ref_to_hap, test_ref_to_hap_cumulative_shift_across_edits);
    suite_add_tcase(s, tc_ref_to_hap);

    TCase *tc_hap_len = tcase_create("hap_len");
    tcase_add_test(tc_hap_len, test_hap_len_no_edits);
    tcase_add_test(tc_hap_len, test_hap_len_mixed_edits);
    suite_add_tcase(s, tc_hap_len);

    TCase *tc_snap = tcase_create("snap");
    tcase_add_test(tc_snap, test_snap_noop_when_already_outside_every_edit);
    tcase_add_test(tc_snap, test_snap_widens_edge_inside_one_edit);
    tcase_add_test(tc_snap, test_snap_widens_both_edges_across_different_edits);
    suite_add_tcase(s, tc_snap);

    TCase *tc_window = tcase_create("build_hap_window");
    tcase_add_test(tc_window, test_build_hap_window_entirely_inside_an_insertion);
    tcase_add_test(tc_window, test_build_hap_window_spans_an_edit_boundary);
    tcase_add_test(tc_window, test_build_hap_window_pads_past_gene_start_with_n);
    suite_add_tcase(s, tc_window);

    TCase *tc_create = tcase_create("create_alt_seq_multi");
    tcase_add_test(tc_create, test_create_alt_seq_multi_single_edit_matches_create_alt_seq);
    tcase_add_test(tc_create, test_create_alt_seq_multi_two_nonoverlapping_edits);
    suite_add_tcase(s, tc_create);

    TCase *tc_align = tcase_create("align_predictions_multi");
    tcase_add_test(tc_align, test_align_predictions_multi_insertion_collapses_to_strongest_site);
    tcase_add_test(tc_align, test_align_predictions_multi_deletion_fills_no_site);
    suite_add_tcase(s, tc_align);

    return s;
}

int main(void) {
    SRunner *sr = srunner_create(seqedit_suite());
    srunner_run_all(sr, CK_NORMAL);
    const int failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
