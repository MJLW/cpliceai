#ifndef GENE_REFERENCE_H
#define GENE_REFERENCE_H

#include <htslib/kstring.h>
#include <htslib/faidx.h>

#include "gene_regions.h"
#include "reference.h"

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

void gene_reference_destroy(GeneReference *gene);

#endif
