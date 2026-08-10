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

/*
 * Three comparisons are reported per variant, differing only in what the alternate sequence is
 * compared against and what it contains:
 *
 *   SpliceAI      REF     -> ALT       the variant alone, against the reference genome
 *   SpliceAI_HAP  HAP_REF -> HAP_ALT   the variant alone, against the rest of its haplotype
 *   SpliceAI_TOT  REF     -> HAP_ALT   the whole haplotype, against the reference genome
 *
 * The first is haplotype-independent and keeps the exact format it has always had. The other
 * two name the copy they were computed on, since a variant on both copies is reported once per
 * copy and their backgrounds differ.
 */
#define SPLICEAI_TAG "SpliceAI"
#define SPLICEAI_HAP_TAG "SpliceAI_HAP"
#define SPLICEAI_TOT_TAG "SpliceAI_TOT"
#define SPLICEAI_DESC "##INFO=<ID=SpliceAI,Number=.,Type=String,Description=\"SpliceAIv1.3.1 variant annotation. These include delta scores (DS) and delta positions (DP) for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL\">"
#define SPLICEAI_HAP_DESC "##INFO=<ID=SpliceAI_HAP,Number=.,Type=String,Description=\"SpliceAI delta scores for the variant against the rest of its haplotype, i.e. the sequence carrying the co-phased variants around it but not the variant itself. HAP is the copy scored (1 or 2). Format: ALLELE|SYMBOL|HAP|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL\">"
#define SPLICEAI_TOT_DESC "##INFO=<ID=SpliceAI_TOT,Number=.,Type=String,Description=\"SpliceAI delta scores for the variant's whole haplotype against the reference genome, i.e. the combined effect of every co-phased variant on that copy. HAP is the copy scored (1 or 2). Format: ALLELE|SYMBOL|HAP|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL\">"


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

/*
 * One substitution to apply to a gene's reference sequence: alt replaces the ref_len bases at
 * gene_pos. A haplotype is a list of these, sorted by gene_pos and non-overlapping, which is
 * what every function below taking (edits, n_edits) assumes.
 *
 * A single variant is the one-edit case, and the single-variant functions are written in terms
 * of the multi-edit ones so the coordinate arithmetic exists in exactly one place.
 */
typedef struct {
    int64_t     gene_pos; /* 0-based offset into the gene's reference sequence */
    int         ref_len;
    const char *alt;      /* not read by the alignment functions, which only need the lengths */
    int         alt_len;
} SeqEdit;

/*
 * seq_edits_ref_to_hap - Map a gene-reference offset to its offset in the edited sequence.
 *
 * Defined for positions outside every edit's REF span. A position inside one has no single
 * counterpart - that is exactly what align_predictions_multi is for - and maps to where the
 * edit's alternate allele begins.
 */
int64_t seq_edits_ref_to_hap(const SeqEdit *edits, const int n_edits, const int64_t ref_pos);

/* Length of the gene sequence once every edit is applied. */
int64_t seq_edits_hap_len(const SeqEdit *edits, const int n_edits, const int64_t gene_len);

void create_alt_seq(const kstring_t *ref_seq, const uint64_t pos, const int ref_len, const int alt_len, const char *alt, char *alt_seq[], size_t *alt_seq_len);

/*
 * create_alt_seq_multi - create_alt_seq over a whole haplotype: the gene sequence with every
 * edit applied. Caller frees *alt_seq.
 */
void create_alt_seq_multi(const kstring_t *ref_seq, const SeqEdit *edits, const int n_edits,
                          char *alt_seq[], size_t *alt_seq_len);

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

/*
 * build_hap_window - Fill a window with the edited sequence for haplotype offsets
 * [hap_lo, hap_hi), padding with 'N' wherever that runs past either end of the gene.
 *
 * This is build_alt_window's general form, and the reason it is expressed in haplotype rather
 * than reference offsets: with several edits in play there is no fixed width a reference range
 * maps to. A caller wanting to cover reference range [ref_lo, ref_hi) takes its image under
 * seq_edits_ref_to_hap and adds whatever context the model needs either side.
 *
 * Only the requested range is ever materialised, which is what lets a caller build an 11kb
 * window over a multi-megabase gene.
 *
 * Parameters:
 *   gene_seq - the gene's reference sequence.
 *   edits    - sorted, non-overlapping edits; those outside the range cost nothing.
 *   n_edits  - number of edits.
 *   hap_lo   - first haplotype offset to write; may be negative.
 *   hap_hi   - one past the last.
 *   out      - receives hap_hi - hap_lo bytes; not NUL-terminated.
 */
void build_hap_window(const kstring_t *gene_seq, const SeqEdit *edits, const int n_edits,
                      const int64_t hap_lo, const int64_t hap_hi, char *out);

/*
 * seq_edits_snap - Widen [*ref_lo, *ref_hi) until neither edge falls inside an edit's REF span.
 *
 * A position an edit deleted or replaced has no single counterpart in the edited sequence, so
 * a range cannot begin or end there and still be aligned back. Widening to the edit's own
 * boundaries costs a few extra positions and keeps the mapping well defined.
 */
void seq_edits_snap(const SeqEdit *edits, const int n_edits, int64_t *ref_lo, int64_t *ref_hi);

void align_predictions_alt_to_ref(const uint64_t gene_pos, const uint64_t gene_len, const int ref_len, const int alt_len, float *alt[]);

/*
 * align_predictions_multi - Project predictions made over an edited sequence back onto
 * reference coordinates, so they can be compared position-for-position with reference scores.
 *
 * Indels make the two coordinate systems disagree, and every edit shifts everything after it.
 * Each edit's REF span is resolved the way the single-variant path has always resolved it: an
 * insertion collapses to the strongest acceptor and donor over the inserted bases, and the
 * bases a deletion removed are filled in as "no splice site here" (neither = 1).
 *
 * Parameters:
 *   edits    - sorted, non-overlapping edits, in gene coordinates.
 *   n_edits  - number of edits.
 *   ref_start - gene offset the output begins at; must lie outside every edit's REF span.
 *   n_ref     - number of reference positions to produce.
 *   hap       - predictions over the edited sequence, starting at the haplotype offset
 *               corresponding to ref_start, and long enough to cover n_ref reference
 *               positions once the edits in range are accounted for.
 *   out       - receives n_ref * NUM_SCORES floats.
 */
void align_predictions_multi(const SeqEdit *edits, const int n_edits,
                             const int64_t ref_start, const int64_t n_ref,
                             const float *hap, float *out);

int one_hot_encode(const char *sequence, const int len, float *encoding_out);

Score calculate_delta_scores(char *allele, char *gene_symbol, float *predictions_ref, float *predictions_alt, int len, int offset);

#endif

