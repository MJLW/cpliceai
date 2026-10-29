#ifndef GENE_REFERENCE_H
#define GENE_REFERENCE_H

#include <htslib/kstring.h>
#include <htslib/faidx.h>

#include "gene_regions.h"
#include "predict.h"
#include "reference.h"
#include "utils.h"

#define INITIAL_REF_SIZE_MALLOC 100000

typedef struct {
    char name[FIELD_MAX_LEN];
    kstring_t seq;
    uint64_t start, end; // 0-based, open-ended
    char strand;

    float *scores;
    size_t n_scores, m_scores;
} GeneReference;

int gene_reference_init(GeneReference *gene);

int gene_reference_update(const char *chr, const char *name, const faidx_t *fa, const Reference *reference, GeneReference *gene);

int gene_reference_get_score_window(const hts_pos_t variant_pos, const int window_radius, const GeneReference *gene, float *score_window[]);

/*
 * gene_reference_predict - Predict over the gene with a set of edits applied, handing the
 * scores back in reference coordinates so they line up with the gene's reference scores.
 *
 * The sequence built is the image of [ref_lo, ref_hi) under the edits, plus BOUNDARY_SIZE of
 * context either side for the model to trim. Taking the image rather than a fixed width is
 * what keeps the scored region the same set of reference positions however much the edits have
 * inserted or deleted; the predictions are then mapped back with align_predictions_multi.
 *
 * ref_lo and ref_hi must lie outside every edit's REF span - see seq_edits_snap - and may run
 * past either end of the gene, which is padded with 'N' as an unedited window would be.
 *
 * *scores receives (ref_hi - ref_lo) * NUM_SCORES floats, which the caller frees.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE if prediction fails.
 */
int gene_reference_predict(Model *models, const GeneReference *gene, const SeqEdit *edits,
                           const int n_edits, const int64_t ref_lo, const int64_t ref_hi,
                           float *scores[]);

void gene_reference_destroy(GeneReference *gene);

#endif
