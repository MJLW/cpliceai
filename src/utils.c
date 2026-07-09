#include "utils.h"

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

    int c = 0;
    for (; c < boundary.start; c++) padded_seq[c] = 'N';
    for (; c < width - boundary.end; c++) padded_seq[c] = seq[c];
    for (; c < width; c++) padded_seq[c] = 'N';
    padded_seq[width] = '\0';

    return padded_seq;
}

void create_alt_seq(const kstring_t *ref_seq, const uint64_t pos, const int ref_len, const int alt_len, const char *alt, char *alt_seq[], size_t *alt_seq_len) {
    const int new_seq_len = ref_seq->l + alt_len - ref_len;
    char *new_seq = malloc(new_seq_len + 1);
    new_seq[new_seq_len] = '\0';

    // Copy everything before variant
    memcpy(
        new_seq,
        ref_seq->s,
        pos
    );

    // Copy over alt
    memcpy(
        new_seq + pos,
        alt,
        alt_len
    );

    // Copy everything after variant
    memcpy(
        new_seq + pos + alt_len,
        ref_seq->s + (pos + ref_len),
        ref_seq->l - (pos + ref_len)
    );

    *alt_seq = new_seq;
    *alt_seq_len = new_seq_len;
}

void align_predictions_alt_to_ref(const uint64_t gene_pos, const uint64_t gene_len, const int ref_len, const int alt_len, float *alt[]) {
    float *tmp = *alt;
    if (alt_len > ref_len) { // Insertion
        float max_donor = 0, max_acceptor = 0;
        for (int i = 0; i < alt_len; i++) {
            int index = (gene_pos + i) * NUM_SCORES;

            if (tmp[index + DONOR_POS] > max_donor) {
                max_donor = tmp[index + DONOR_POS];
            }
            if (tmp[index + ACCEPTOR_POS] > max_acceptor) {
                max_acceptor = tmp[index + ACCEPTOR_POS];
            }
        }

        memmove(
            tmp + ((gene_pos + ref_len) * NUM_SCORES),
            tmp + ((gene_pos + alt_len) * NUM_SCORES),
            (gene_len - (gene_pos + ref_len)) * NUM_SCORES * sizeof(float)
        );

        tmp[gene_pos * NUM_SCORES] = max_donor + max_acceptor > 1.0 ? 0 : 1 - max_donor - max_acceptor;
        tmp[gene_pos * NUM_SCORES + ACCEPTOR_POS] = max_acceptor;
        tmp[gene_pos * NUM_SCORES + DONOR_POS] = max_donor;
    } else if (alt_len < ref_len) { // Deletion
        tmp = realloc(tmp, gene_len * NUM_SCORES * sizeof(float));

        memmove(
            tmp + ((gene_pos + ref_len) * NUM_SCORES),
            tmp + ((gene_pos + alt_len) * NUM_SCORES),
            (gene_len - (gene_pos + ref_len)) * NUM_SCORES * sizeof(float)
        );

        for (int i = (gene_pos + alt_len) * NUM_SCORES; i < (gene_pos + ref_len) * NUM_SCORES; i += NUM_SCORES) {
            tmp[i] = 1.0;
            tmp[i + ACCEPTOR_POS] = 0.0;
            tmp[i + DONOR_POS] = 0.0;
        }
    }

    *alt = tmp;
}

int one_hot_encode(const char *sequence, const int len, float *encoding) {
    int enc_len = len * ENCODING_SIZE;
    for (int i = 0; i < enc_len; i+=ENCODING_SIZE, sequence++) {
        switch (*sequence) {
            case BASE_A:
                encoding[i + BASE_A_ENC] = 1.0f;
                break;
            case BASE_C:
                encoding[i + BASE_C_ENC] = 1.0f;
                break;
            case BASE_G:
                encoding[i + BASE_G_ENC] = 1.0f;
                break;
            case BASE_T:
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


