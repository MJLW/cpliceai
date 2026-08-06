#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <htslib/faidx.h>
#include <htslib/kstring.h>

#include "range.h"

#define ENCODING_SIZE 4
/*
 * Both cases are recognised: soft-masked references (Ensembl dna_sm, UCSC hg19/hg38) lowercase
 * repeat regions, and faidx_fetch_seq preserves case.
 */
#define BASE_A_UPPER 'A'
#define BASE_A_LOWER 'a'
#define BASE_A_ENC 0
#define BASE_C_UPPER 'C'
#define BASE_C_LOWER 'c'
#define BASE_C_ENC 1
#define BASE_G_UPPER 'G'
#define BASE_G_LOWER 'g'
#define BASE_G_ENC 2
#define BASE_T_UPPER 'T'
#define BASE_T_LOWER 't'
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

FILE *open_file_or_log(const char *path, const char *mode);

/*
 * FNV-1a, 64-bit. Used to fingerprint the inputs a reference scores file was built from, so a
 * later run can tell it is being pointed at a mismatched fasta or regions file. Not a security
 * primitive - it only has to catch mistakes.
 */
#define DIGEST_SEED 1469598103934665603ULL

uint64_t digest_update(uint64_t digest, const void *data, size_t len);

/* Convenience wrappers, so callers need not spell out sizeof or strlen. */
uint64_t digest_update_str(uint64_t digest, const char *s);
uint64_t digest_update_u64(uint64_t digest, uint64_t value);

/*
 * fasta_digest - Fingerprint a reference fasta by its contig names and lengths, taken from the
 * already-loaded .fai index.
 *
 * Not a digest of the bases, which would cost more than the prediction itself. Identifies the
 * assembly - contig set, naming and lengths - but not two assemblies differing only in masking.
 */
uint64_t fasta_digest(const faidx_t *fa);

void reverse_encoding(float enc[], int len);

void reverse_prediction(float preds[], int len, int size);

Range find_transcript_boundary(const int position, const int start, const int end, const int width);

char *pad_sequence(const char *seq, const Range boundary, const int width);

void create_alt_seq(const kstring_t *ref_seq, const uint64_t pos, const int ref_len, const int alt_len, const char *alt, char *alt_seq[], size_t *alt_seq_len);

/*
 * build_alt_window - Fill a fixed-width window centred on a variant with the gene sequence,
 * with the alternate allele substituted in.
 *
 * Positions the window so the variant sits at its centre, padding with 'N' wherever the window
 * runs past either end of the gene. Unlike create_alt_seq this never copies the whole gene,
 * which is the point: predict_variant only needs the neighbourhood of the variant.
 *
 * Parameters:
 *   gene_seq - the gene's reference sequence.
 *   gene_pos - 0-based offset of the variant within gene_seq.
 *   ref_len  - length of the REF allele.
 *   alt      - the alternate allele.
 *   alt_len  - length of the alternate allele.
 *   out      - receives width bytes; not NUL-terminated.
 *   width    - window width, CONTEXT_SIZE + window_size.
 *   radius   - bases of gene sequence to keep either side of the variant, i.e.
 *              BOUNDARY_SIZE + window_radius.
 */
void build_alt_window(const kstring_t *gene_seq, const int64_t gene_pos, const int ref_len,
                      const char *alt, const int alt_len, char *out, const int width,
                      const int radius);

void align_predictions_alt_to_ref(const uint64_t gene_pos, const uint64_t gene_len, const int ref_len, const int alt_len, float *alt[]);

int one_hot_encode(const char *sequence, const int len, float *encoding_out);

Score calculate_delta_scores(char *allele, char *gene_symbol, float *predictions_ref, float *predictions_alt, int len, int offset);

#endif

