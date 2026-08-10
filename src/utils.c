#include "utils.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logging/log.h"

FILE *open_file_or_log(const char *path, const char *mode) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        log_error("Could not open file: %s", path);
    }
    return fp;
}

uint64_t digest_update(uint64_t digest, const void *data, size_t len) {
    const unsigned char *bytes = data;
    for (size_t i = 0; i < len; i++) {
        digest ^= bytes[i];
        digest *= 1099511628211ULL; // FNV prime
    }
    return digest;
}

uint64_t digest_update_str(uint64_t digest, const char *s) {
    // Includes the terminator, so ("AB","C") and ("A","BC") do not collide.
    return digest_update(digest, s, strlen(s) + 1);
}

uint64_t digest_update_u64(uint64_t digest, uint64_t value) {
    unsigned char buf[8];
    for (int i = 0; i < 8; i++) buf[i] = (unsigned char) (value >> (i * 8));
    return digest_update(digest, buf, sizeof(buf));
}

uint64_t fasta_digest(const faidx_t *fa) {
    uint64_t digest = DIGEST_SEED;
    const int n = faidx_nseq(fa);

    digest = digest_update_u64(digest, (uint64_t) n);
    for (int i = 0; i < n; i++) {
        const char *name = faidx_iseq(fa, i);
        digest = digest_update_str(digest, name);
        digest = digest_update_u64(digest, (uint64_t) faidx_seq_len64(fa, name));
    }

    return digest;
}

void reverse_encoding(float enc[], int len) {
    float tmp;
    for (int i = 0, j = len - 1; i < j; i++, j--) {
        tmp = enc[i];
        enc[i] = enc[j];
        enc[j] = tmp;
    }
}

void reverse_prediction(float preds[], int len, int size) {
    int num_preds = len / size;
    float tmp;
    for (int i = 0; i < num_preds / 2; i++) {
        for (int j = 0; j < size; j++) {
            tmp = preds[i * size + j];
            preds[i * size + j] = preds[(num_preds - 1 - i) * size + j];
            preds[(num_preds - 1 - i) * size + j] = tmp;
        }
    }
}

Range find_transcript_boundary(const int position, const int start, const int end, const int width) {
    int distance_from_start = width/2 + (start - position);
    int distance_from_end = width/2 - (end - (position+1)); // End is open, so +1
    return (Range) { distance_from_start > 0 ? distance_from_start : 0, distance_from_end > 0 ? distance_from_end : 0 };
}

char *pad_sequence(const char *seq, const Range boundary, const int width) {
    char *padded_seq = malloc(width + 1);
    if (padded_seq == NULL) {
        log_fatal("Failed to allocate %d bytes for padded sequence", width + 1);
        exit(EXIT_FAILURE);
    }

    int c = 0;
    for (; c < boundary.start; c++) padded_seq[c] = 'N';
    for (; c < width - boundary.end; c++) padded_seq[c] = seq[c];
    for (; c < width; c++) padded_seq[c] = 'N';
    padded_seq[width] = '\0';

    return padded_seq;
}

int64_t seq_edits_ref_to_hap(const SeqEdit *edits, const int n_edits, const int64_t ref_pos) {
    int64_t shift = 0;
    for (int i = 0; i < n_edits && edits[i].gene_pos <= ref_pos; i++) {
        // A position inside this edit's REF span lands where the alternate allele starts.
        if (edits[i].gene_pos + edits[i].ref_len > ref_pos) return edits[i].gene_pos + shift;
        shift += edits[i].alt_len - edits[i].ref_len;
    }
    return ref_pos + shift;
}

int64_t seq_edits_hap_len(const SeqEdit *edits, const int n_edits, const int64_t gene_len) {
    int64_t len = gene_len;
    for (int i = 0; i < n_edits; i++) len += edits[i].alt_len - edits[i].ref_len;
    return len;
}

/*
 * Write the edited sequence for haplotype offsets [hap_lo, hap_hi) into out, which must hold
 * hap_hi - hap_lo bytes. Offsets falling outside the edited gene are left untouched, so a
 * caller wanting them padded pre-fills out.
 *
 * The edited sequence is a run of alternating segments - reference stretch, alternate allele,
 * reference stretch, ... - so this walks them in order and copies whatever overlaps the
 * requested range. Nothing outside that range is ever materialised, which is what lets
 * predict_variant build an 11kb window over a multi-megabase gene.
 */
static void hap_write_range(const kstring_t *gene_seq, const SeqEdit *edits, const int n_edits,
                            const int64_t hap_lo, const int64_t hap_hi, char *out) {
    const int64_t gene_len = (int64_t) gene_seq->l;
    int64_t ref_cursor = 0; // next unconsumed reference offset
    int64_t hap_cursor = 0; // where that offset lands in the edited sequence

    // copy_segment clips [seg_lo, seg_lo + seg_len) against the requested range before copying.
    #define copy_segment(src, seg_lo, seg_len)                                        \
        do {                                                                          \
            const int64_t lo = (seg_lo) > hap_lo ? (seg_lo) : hap_lo;                 \
            const int64_t hi = (seg_lo) + (seg_len) < hap_hi ? (seg_lo) + (seg_len) : hap_hi; \
            if (hi > lo) memcpy(out + (lo - hap_lo), (src) + (lo - (seg_lo)), (size_t) (hi - lo)); \
        } while (0)

    for (int i = 0; i < n_edits; i++) {
        const int64_t stretch = edits[i].gene_pos - ref_cursor;
        if (stretch > 0) {
            copy_segment(gene_seq->s + ref_cursor, hap_cursor, stretch);
            hap_cursor += stretch;
        }

        copy_segment(edits[i].alt, hap_cursor, edits[i].alt_len);
        hap_cursor += edits[i].alt_len;

        ref_cursor = edits[i].gene_pos + edits[i].ref_len;

        // Past the end of the last segment there is nothing left to write.
        if (hap_cursor >= hap_hi) return;
    }

    // A REF reaching past the end of the gene leaves no trailing reference stretch.
    if (ref_cursor < gene_len) copy_segment(gene_seq->s + ref_cursor, hap_cursor, gene_len - ref_cursor);

    #undef copy_segment
}

void create_alt_seq_multi(const kstring_t *ref_seq, const SeqEdit *edits, const int n_edits,
                          char *alt_seq[], size_t *alt_seq_len) {
    const int64_t gene_len = (int64_t) ref_seq->l;

    // A REF running past the end of the sequence can only delete as far as the sequence goes.
    int64_t hap_len = gene_len;
    for (int i = 0; i < n_edits; i++) {
        int effective_ref_len = edits[i].ref_len;
        if (edits[i].gene_pos + effective_ref_len > gene_len) {
            effective_ref_len = (int) (gene_len - edits[i].gene_pos);
            log_warn("Reference allele at offset %"PRId64" spans %d bases but only %d remain in the gene; truncating.",
                     edits[i].gene_pos, edits[i].ref_len, effective_ref_len);
        }
        hap_len += edits[i].alt_len - effective_ref_len;
    }

    char *new_seq = malloc(hap_len + 1);
    if (new_seq == NULL) {
        log_fatal("Failed to allocate %"PRId64" bytes for alt sequence", hap_len + 1);
        exit(EXIT_FAILURE);
    }
    new_seq[hap_len] = '\0';

    hap_write_range(ref_seq, edits, n_edits, 0, hap_len, new_seq);

    *alt_seq = new_seq;
    *alt_seq_len = (size_t) hap_len;
}

void create_alt_seq(const kstring_t *ref_seq, const uint64_t pos, const int ref_len, const int alt_len, const char *alt, char *alt_seq[], size_t *alt_seq_len) {
    const SeqEdit edit = { (int64_t) pos, ref_len, alt, alt_len };
    create_alt_seq_multi(ref_seq, &edit, 1, alt_seq, alt_seq_len);
}

void build_hap_window(const kstring_t *gene_seq, const SeqEdit *edits, const int n_edits,
                      const int64_t hap_lo, const int64_t hap_hi, char *out) {
    memset(out, 'N', (size_t) (hap_hi - hap_lo));
    hap_write_range(gene_seq, edits, n_edits, hap_lo, hap_hi, out);
}

void seq_edits_snap(const SeqEdit *edits, const int n_edits, int64_t *ref_lo, int64_t *ref_hi) {
    for (int i = 0; i < n_edits; i++) {
        const int64_t edit_lo = edits[i].gene_pos;
        const int64_t edit_hi = edits[i].gene_pos + edits[i].ref_len;

        if (edit_lo < *ref_lo && *ref_lo < edit_hi) *ref_lo = edit_lo;
        if (edit_lo < *ref_hi && *ref_hi < edit_hi) *ref_hi = edit_hi;
    }
}

void build_alt_window(const kstring_t *gene_seq, const int64_t gene_pos, const int ref_len,
                      const char *alt, const int alt_len, char *out, const int width,
                      const int radius) {
    const SeqEdit edit = { gene_pos, ref_len, alt, alt_len };
    // A lone variant sits at its own haplotype offset, so the window still starts radius bases
    // ahead of it and the alternate allele still lands at out[radius].
    build_hap_window(gene_seq, &edit, 1, gene_pos - radius, gene_pos - radius + width, out);
}

void align_predictions_multi(const SeqEdit *edits, const int n_edits,
                             const int64_t ref_start, const int64_t n_ref,
                             const float *hap, float *out) {
    const int64_t hap_start = seq_edits_ref_to_hap(edits, n_edits, ref_start);
    const int64_t ref_end = ref_start + n_ref;

    int64_t ref_cursor = ref_start;
    int64_t hap_cursor = hap_start;

    for (int i = 0; i < n_edits && ref_cursor < ref_end; i++) {
        if (edits[i].gene_pos + edits[i].ref_len <= ref_cursor) continue; // wholly behind us
        if (edits[i].gene_pos >= ref_end) break;                          // wholly past the output

        // Unedited stretch before this edit: reference and haplotype run in step.
        const int64_t stretch = edits[i].gene_pos - ref_cursor;
        if (stretch > 0) {
            memcpy(out + (ref_cursor - ref_start) * NUM_SCORES, hap + (hap_cursor - hap_start) * NUM_SCORES,
                   (size_t) stretch * NUM_SCORES * sizeof(float));
            ref_cursor += stretch;
            hap_cursor += stretch;
        }

        const int ref_len = edits[i].ref_len;
        const int alt_len = edits[i].alt_len;
        const int shared = ref_len < alt_len ? ref_len : alt_len;

        // As far as the two alleles overlap, each reference base has a counterpart.
        for (int k = 0; k < shared && ref_cursor + k < ref_end; k++) {
            memcpy(out + (ref_cursor + k - ref_start) * NUM_SCORES, hap + (hap_cursor + k - hap_start) * NUM_SCORES,
                   NUM_SCORES * sizeof(float));
        }

        if (alt_len > ref_len && ref_cursor < ref_end) {
            // Insertion: the inserted bases have no reference position of their own, so the
            // strongest site anywhere across the allele is reported at the first one.
            float *dst = out + (ref_cursor - ref_start) * NUM_SCORES;
            float acceptor = dst[ACCEPTOR_POS];
            float donor = dst[DONOR_POS];
            for (int k = shared; k < alt_len; k++) {
                const float *src = hap + (hap_cursor + k - hap_start) * NUM_SCORES;
                if (src[ACCEPTOR_POS] > acceptor) acceptor = src[ACCEPTOR_POS];
                if (src[DONOR_POS] > donor) donor = src[DONOR_POS];
            }
            dst[ACCEPTOR_POS] = acceptor;
            dst[DONOR_POS] = donor;
            dst[0] = acceptor + donor > 1.0 ? 0 : 1 - acceptor - donor;
        } else {
            // Deletion: the bases it removed are not in the haplotype to be scored at all.
            for (int k = shared; k < ref_len && ref_cursor + k < ref_end; k++) {
                float *dst = out + (ref_cursor + k - ref_start) * NUM_SCORES;
                dst[0] = 1.0;
                dst[ACCEPTOR_POS] = 0.0;
                dst[DONOR_POS] = 0.0;
            }
        }

        ref_cursor += ref_len;
        hap_cursor += alt_len;
    }

    if (ref_cursor < ref_end) {
        memcpy(out + (ref_cursor - ref_start) * NUM_SCORES, hap + (hap_cursor - hap_start) * NUM_SCORES,
               (size_t) (ref_end - ref_cursor) * NUM_SCORES * sizeof(float));
    }
}

void align_predictions_alt_to_ref(const uint64_t gene_pos, const uint64_t gene_len, const int ref_len, const int alt_len, float *alt[]) {
    const SeqEdit edit = { (int64_t) gene_pos, ref_len, NULL, alt_len };

    float *aligned = malloc(gene_len * NUM_SCORES * sizeof(float));
    if (aligned == NULL) {
        log_fatal("Failed to allocate %zu bytes for aligned predictions", (size_t) gene_len * NUM_SCORES * sizeof(float));
        exit(EXIT_FAILURE);
    }

    align_predictions_multi(&edit, 1, 0, (int64_t) gene_len, *alt, aligned);

    free(*alt);
    *alt = aligned;
}

int one_hot_encode(const char *sequence, const int len, float *encoding) {
    int enc_len = len * ENCODING_SIZE;
    for (int i = 0; i < enc_len; i+=ENCODING_SIZE, sequence++) {
        switch (*sequence) {
            // Anything unmatched - N, IUPAC ambiguity codes - stays all-zeros, the model's
            // encoding for an unknown base.
            case BASE_A_UPPER:
            case BASE_A_LOWER:
                encoding[i + BASE_A_ENC] = 1.0f;
                break;
            case BASE_C_UPPER:
            case BASE_C_LOWER:
                encoding[i + BASE_C_ENC] = 1.0f;
                break;
            case BASE_G_UPPER:
            case BASE_G_LOWER:
                encoding[i + BASE_G_ENC] = 1.0f;
                break;
            case BASE_T_UPPER:
            case BASE_T_LOWER:
                encoding[i + BASE_T_ENC] = 1.0f;
                break;
        }
    }

    return enc_len;
}

Score calculate_delta_scores(char *allele, char *gene_symbol, float *predictions_ref, float *predictions_alt, int len, int offset) {
    float ag_best = 0.0, al_best = 0.0, dg_best = 0.0, dl_best = 0.0;
    int ag_idx = 0, al_idx = 0, dg_idx = 0, dl_idx = 0;

    for (int p = 0; p < len; p += NUM_SCORES) {
        float ag = predictions_alt[p + ACCEPTOR_POS] - predictions_ref[p + ACCEPTOR_POS];
        float al = predictions_ref[p + ACCEPTOR_POS] - predictions_alt[p + ACCEPTOR_POS];
        float dg = predictions_alt[p + DONOR_POS] - predictions_ref[p + DONOR_POS];
        float dl = predictions_ref[p + DONOR_POS] - predictions_alt[p + DONOR_POS];

        if (ag > ag_best) { ag_best = ag; ag_idx = (p / NUM_SCORES); }
        if (al > al_best) { al_best = al; al_idx = (p / NUM_SCORES); }
        if (dg > dg_best) { dg_best = dg; dg_idx = (p / NUM_SCORES); }
        if (dl > dl_best) { dl_best = dl; dl_idx = (p / NUM_SCORES); }
    }

    ag_idx = ag_idx-offset;
    al_idx = al_idx-offset;
    dg_idx = dg_idx-offset;
    dl_idx = dl_idx-offset;

    return (Score) { allele, gene_symbol, ag_best, al_best, dg_best, dl_best, ag_idx, al_idx, dg_idx, dl_idx };
}


