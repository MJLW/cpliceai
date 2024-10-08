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

char *replace_variant(const char *seq, const int len, const int rlen, const char *alt, const int alen) {
    int alt_seq_len = len - rlen + alen;
    char *alt_seq = malloc(alt_seq_len + 1);

    int c = 0;
    for (; c < len/2; c++) alt_seq[c] = seq[c];
    for (int ai = 0; ai < alen; ai++, c++) alt_seq[c] = alt[ai];
    for (int ri = len/2 + rlen; ri < len; c++, ri++) alt_seq[c] = seq[ri];
    alt_seq[alt_seq_len] = '\0';

    return alt_seq;
}

int one_hot_encode(const char *sequence, const int len, float *encoding_out[]) {
    int enc_len = len * ENCODING_SIZE;
    float *encoding = calloc(enc_len, sizeof(float));
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

    *encoding_out = encoding;
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


