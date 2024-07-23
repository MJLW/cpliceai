#include <htslib/hts.h>
#include <htslib/vcf.h>
#include <htslib/tbx.h>
#include <htslib/regidx.h>


#define ENCODING_SIZE 4
#define BASE_A 'A'
#define BASE_A_ENC 0
#define BASE_C 'C'
#define BASE_C_ENC 1
#define BASE_G 'G'
#define BASE_G_ENC 2
#define BASE_T 'T'
#define BASE_T_ENC 3

#define NUM_SCORES 3
#define ACCEPTOR_POS 1
#define DONOR_POS 2

#define SPLICEAI_TAG "SpliceAI"
#define SPLICEAI_DESC "##INFO=<ID=SpliceAI,Number=.,Type=String,Description=\"SpliceAIv1.3.1 variant annotation. These include delta scores (DS) and delta positions (DP) for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL\">"


typedef struct {
    char *alt;
    char *gene;
    float ag;
    float al;
    float dg;
    float dl;
    int ag_idx;
    int al_idx;
    int dg_idx;
    int dl_idx;
} Score;


void reverse_encoding(float enc[], int len);

void reverse_prediction(float preds[], int len, int size);

reg_t find_transcript_boundary(const int position, const int start, const int end, const int width);

char *pad_sequence(const char *seq, const reg_t boundary, const int width);

char *replace_variant(const char *seq, const int len, const int rlen, const char *alt, const int alen);

int one_hot_encode(const char *sequence, const int len, float *encoding_out[]);

Score calculate_delta_scores(char *allele, char *gene_symbol, float *predictions_ref, float *predictions_alt, int len, int offset);

