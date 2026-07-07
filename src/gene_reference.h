#ifndef GENE_REFERENCE_H
#define GENE_REFERENCE_H

#include <htslib/kstring.h>
#include <htslib/faidx.h>

#include "reference.h"

#define MAX_FIELD 256
#define INITIAL_REF_SIZE_MALLOC 100000

typedef struct {
    char name[MAX_FIELD];
    kstring_t seq;
    uint64_t start, end; // 0-based, open-ended
    char strand;

    float *scores;
    size_t n_scores, m_scores;
} GeneReference;

extern int GeneReference_init(GeneReference *gene);

extern int GeneReference_update(const char *chr, const char *name, const faidx_t *fa, const Reference *reference, GeneReference *gene);

extern int GeneReference_get_score_window(const hts_pos_t variant_pos, const int window_radius, const GeneReference gene, float *score_window[]);

extern void GeneReference_destroy(GeneReference *gene);

#endif
