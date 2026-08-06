#include "gene_reference.h"

#include "logging/log.h"
#include "utils.h"
#include <stdlib.h>

int gene_reference_init(GeneReference *gene) {
    gene->name[0] = '\0';
    gene->seq = (kstring_t) { 0 };

    gene->n_scores = 0;
    gene->m_scores = INITIAL_REF_SIZE_MALLOC * NUM_SCORES;
    gene->scores = malloc(gene->m_scores * sizeof(float));
    if (gene->scores == NULL) {
        log_fatal("Failed to allocate %zu bytes for gene scores", gene->m_scores * sizeof(float));
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}

int gene_reference_update(const char *chr, const char *name, const faidx_t *fa, const Reference *reference, GeneReference *gene) {
    if (gene->seq.s != NULL) {
        free(gene->seq.s);
        gene->seq.s = NULL;
        gene->seq.l = 0;
        gene->seq.m = 0;
    }

    // Callers cache on the name to skip reloading the same gene, so the name must not outlive
    // the data it describes: on the failure paths below, seq is already freed.
    gene->name[0] = '\0';

    // Find contig
    int contig_index = -1;
    for (int i = 0; i < reference->n_contigs; i++) {
        const Contig contig = reference->contigs[i];
        const char *contig_name = reference->contig_names + contig.name_start;

        if (strncmp(contig_name, chr, FIELD_MAX_LEN) == 0) {
            contig_index = i;
            break;
        }
    }

    if (contig_index == -1) {
        log_warn("Could not find matching contig %s in reference scores", chr);
        return EXIT_FAILURE;
    }

    // Find gene
    const Contig contig = reference->contigs[contig_index];
    const int region_end = contig.region_start + contig.n_regions;
    int gene_index = -1;
    for (int i = contig.region_start; i < region_end; i++) {
        const Region gene = reference->genes[i];
        const char *region_name = reference->region_names + gene.name_start;

        if (strncmp(region_name, name, FIELD_MAX_LEN) == 0) {
            gene_index = i;
            break;
        }
    }

    if (gene_index == -1) {
        log_warn("Could not find matching gene %s on contig %s in reference scores", name, chr);
        return EXIT_FAILURE;
    }

    // Get current gene sequence
    const Region region = reference->genes[gene_index];
    uint64_t start = reference->gene_starts[gene_index];
    uint64_t end = reference->gene_ends[gene_index];

    // end is exclusive; faidx_fetch_seq's range is inclusive, so it stops one base short.
    int seq_len;
    char *seq = faidx_fetch_seq(fa, chr, (int) start, (int) end - 1, &seq_len);
    if (seq == NULL) {
        log_warn("Could not fetch %s:%lu-%lu from the reference fasta", chr, start, end);
        return EXIT_FAILURE;
    }
    seq[seq_len] = '\0';

    strncpy(gene->name, name, FIELD_MAX_LEN);
    kputsn(seq, seq_len, &(gene->seq));
    gene->start = start;
    gene->end = end;
    gene->strand = region.strand;


    // Allocate extra space if current alloc'ed array for ref scores isn't big enough
    gene->n_scores = (gene->end - gene->start) * NUM_SCORES;
    if (gene->m_scores < gene->n_scores) {
        float *new_scores = realloc(gene->scores, gene->n_scores * sizeof(float));
        if (new_scores == NULL) {
            log_fatal("Failed to reallocate %zu bytes for gene scores", gene->n_scores * sizeof(float));
            exit(EXIT_FAILURE);
        }
        gene->scores = new_scores;
        gene->m_scores = gene->n_scores;
    }

    // Initialize blanks for the reference scores, NEITHER=1, ACCEPTOR=0, DONOR=0
    memset(gene->scores, 0, gene->n_scores * sizeof(float));
    for (int i = 0; i < gene->n_scores; i+=NUM_SCORES) {
        gene->scores[i] = 1;
    }

    // Override blanks with scores when present
    int chunk_end = region.chunk_start + region.n_chunks;
    for (int i = region.chunk_start; i < chunk_end; i++) {
        const Chunk chunk = reference->chunks[i];

        int scores_end = chunk.scores_start + chunk.n_scores;
        for (int j = chunk.scores_start; j < scores_end; j++) {
            const PositionScore score = reference->scores[j];
            const int score_index = score.pos * NUM_SCORES;

            gene->scores[score_index] = 1 - score.acceptor - score.donor;
            gene->scores[score_index + ACCEPTOR_POS] = score.acceptor;
            gene->scores[score_index + DONOR_POS] = score.donor;
        }
    }

    return EXIT_SUCCESS;
}

int gene_reference_get_score_window(const hts_pos_t variant_pos, const int window_radius, const GeneReference *gene, float *score_window[]) {
    int window_size = window_radius * 2 + 1;

    // Initialize blank predictions
    float *ref_predictions = calloc(window_size * NUM_SCORES, sizeof(float));
    if (ref_predictions == NULL) {
        log_error("Failed to allocate memory.");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < window_size; i++) {
        ref_predictions[i * NUM_SCORES] = 1.0; // Neither to 1, acceptor and donor to 0
    }

    // Does window radius overlap start?
    hts_pos_t gene_pos = variant_pos - gene->start;
    hts_pos_t start = gene_pos - window_radius;
    hts_pos_t start_offset = 0;
    if (start < 0) {
        start_offset = -start;
        start = 0;
    }

    // Does window radius overlap end?
    hts_pos_t gene_length = gene->end - gene->start;
    hts_pos_t end = gene_pos + window_radius + 1;
    if (end > gene_length) {
        end = gene_length;
    }

    if (end - start <= 0) {
        log_error("Position %li is not located in gene %s, %li-%li. This is a bug.", variant_pos, gene->name, gene->start, gene->end);
        return EXIT_FAILURE;
    }

    // Override blank predictions with actual scores within bounds
    memcpy(ref_predictions + start_offset * NUM_SCORES, gene->scores + start * NUM_SCORES, (end - start) * NUM_SCORES * sizeof(float));

    *score_window = ref_predictions;

    return EXIT_SUCCESS;
}


void gene_reference_destroy(GeneReference *gene) {
    if (gene->seq.s != NULL) free(gene->seq.s);
    free(gene->scores);
}
