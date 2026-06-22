#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <htslib/faidx.h>
#include <string.h>

#include "gene_regions.h"
#include "logging/log.h"
#include "predict.h"
#include "reference.h"
#include "utils.h"

#define MAX_LINE 8192
#define MAX_FIELD 256

#define INITIAL_REF_SIZE_MALLOC 100000
#define SCORE_THRESHOLD ZERO_EPSILON

#define REQUIRED_ARGS \
    REQUIRED_STRING_ARG(variants, "variants", "TSV file containing SNV variants to predict for using SpliceAI") \
    REQUIRED_STRING_ARG(reference_bin, "reference_scores", "Binary file containing reference scores") \
    REQUIRED_STRING_ARG(model_dir, "model_dir", "Directory containing SpliceAI models") \
    REQUIRED_STRING_ARG(fasta, "fasta", "Human reference fasta") \
    REQUIRED_STRING_ARG(regions, "regions", "Gene region structure parsed from GFF with gff_to_bed.py") \
    REQUIRED_STRING_ARG(output, "output", "TSV of splice sites found, where REF>0.2|ALT>0.2.")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>


typedef struct {
    char chr[MAX_FIELD];
    uint64_t pos;
    char ref[MAX_FIELD];
    char alt[MAX_FIELD];
    char gene[MAX_FIELD];
} Variant;

static int parse_line(const char *line, Variant *snv) {
    char buf[MAX_LINE];
    strncpy(buf, line, MAX_LINE-1);
    buf[MAX_LINE-1] = '\0';
    buf[strcspn(buf, "\r\n")] = '\0';

    const char *delim = "\t";
    char *tok;
    tok = strtok(buf, delim);
    if (!tok) {
        log_error("Failed to parse SNV contig from line: %s", line);
        return EXIT_FAILURE;
    }
    strncpy(snv->chr, tok, MAX_FIELD-1);
    snv->chr[MAX_FIELD-1] = '\0';

    tok = strtok(NULL, delim);
    if (!tok) {
        log_error("Failed to parse SNV position from line: %s", line);
        return EXIT_FAILURE;
    }
    char *end_pos;
    snv->pos = strtoull(tok, &end_pos, 10) - 1;
    if (end_pos[0] != '\0') {
        log_error("Failed to parse %s into a positive integer.", tok);
        return EXIT_FAILURE;
    }

    tok = strtok(NULL, delim);
    if (!tok) {
        log_error("Failed to parse reference base(s) from line: %s", line);
        return EXIT_FAILURE;
    }
    strncpy(snv->ref, tok, MAX_FIELD-1);
    snv->ref[MAX_FIELD-1] = '\0';

    tok = strtok(NULL, delim);
    if (!tok) {
        log_error("Failed to parse alternative base(s) from line: %s", line);
        return EXIT_FAILURE;
    }
    strncpy(snv->alt, tok, MAX_FIELD-1);
    snv->alt[MAX_FIELD-1] = '\0';

    tok = strtok(NULL, delim);
    if (!tok) {
        log_error("Failed to parse SNV gene from line: %s", line);
        return EXIT_FAILURE;
    }
    strncpy(snv->gene, tok, MAX_FIELD-1);
    snv->gene[MAX_FIELD-1] = '\0';

    return EXIT_SUCCESS;
}

int read_next_snv_from_tsv(FILE *fp, Variant *snv) {
    char line[MAX_LINE];

    // WARN: Assumes skipped/no header
    while(fgets(line, sizeof(line), fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\r') continue;

        if (parse_line(line, snv) != EXIT_SUCCESS) {
            log_error("Failed parsing of %s");
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;

    }

    return EXIT_FAILURE;
}

int main(int argc, char *argv[]) {
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 1);

    // Parse arguments
    args_t args = make_default_args();
    if (!parse_args(argc, argv, &args) || args.help) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    // Load SpliceAI tensorflow models
    Model *models = load_models(args.model_dir);

    // Load annotations for transcript regions
    FILE *gene_regions_in;
    gene_regions_in = fopen(args.regions, "r");
    // TODO: Parse gene regions into a HashMap for quick access
    if (gene_regions_in == NULL) {
        log_error("Could not open file: %s", args.regions);
        return EXIT_FAILURE;
    }

    Reference ref;
    reference_read(args.reference_bin, &ref);

    // Load reference fasta for sequence lookup
    faidx_t *fa_in;
    if ((fa_in = fai_load(args.fasta)) < 0) return EXIT_FAILURE; // Load reference fasta for sequence lookup

    FILE *variants_fp = fopen(args.variants, "r");
    if (variants_fp == NULL) {
        log_error("Could not open file: %s", args.variants);
        return EXIT_FAILURE;
    }

    FILE *output = fopen(args.output, "w");
    if (output == NULL) {
        log_error("Could not open file: %s", args.output);
        return EXIT_FAILURE;
    }

    // Skip buffer
    char buffer[MAX_LINE];
    fgets(buffer, sizeof(buffer), variants_fp);

    // Loop over VCF variants
    Variant variant;
    char current_gene[MAX_FIELD] = { 0 };
    char *current_gene_seq = NULL;
    uint64_t current_gene_start = 0, current_gene_end = 0;
    int current_gene_len = 0;
    char current_gene_strand = '+';

    size_t n_ref_scores = 0, m_ref_scores = INITIAL_REF_SIZE_MALLOC * NUM_SCORES;
    float *ref_scores = malloc(m_ref_scores * sizeof(float));
    while (read_next_snv_from_tsv(variants_fp, &variant) == EXIT_SUCCESS) {
        fprintf(output, "#%s-%li-%s-%s\n", variant.chr, variant.pos+1, variant.ref, variant.alt);

        // If gene is different from previous variant, we need to load the reference scores for the current gene
        // TODO: Refactor this into a function
        if (strcmp(variant.gene, current_gene) != 0) {
            strncpy(current_gene, variant.gene, MAX_FIELD);

            // Find contig
            int contig_index = -1;
            for (int i = 0; i < ref.n_contigs; i++) {
                const Contig contig = ref.contigs[i];
                const char *contig_name = ref.contig_names + contig.name_start;

                if (strncmp(contig_name, variant.chr, MAX_FIELD) == 0) {
                    contig_index = i;
                    break;
                }
            }

            if (contig_index == -1) {
                log_error("Could not find matching contig %s in reference scores", variant.chr);
                return EXIT_FAILURE;
            }

            // Find gene
            const Contig contig = ref.contigs[contig_index];
            const int region_end = contig.region_start + contig.n_regions;
            int gene_index = -1;
            for (int i = contig.region_start; i < region_end; i++) {
                const Region gene = ref.genes[i];
                const char *region_name = ref.region_names + gene.name_start;

                if (strncmp(region_name, variant.gene, MAX_FIELD) == 0) {
                    gene_index = i;
                    break;
                }
            }

            if (gene_index == -1) {
                log_error("Could not find matching gene %s on contig %s in reference scores", variant.gene, variant.chr);
                return EXIT_FAILURE;
            }

            // Get current gene sequence
            // TODO: Put all the current_gene stuff into a struct
            const Region gene = ref.genes[gene_index];
            current_gene_start = ref.gene_starts[gene_index];
            current_gene_end = ref.gene_ends[gene_index];
            current_gene_strand = gene.strand;
            // -1 for 0-based
            current_gene_seq = faidx_fetch_seq(fa_in, variant.chr, (int) current_gene_start, (int) current_gene_end, &current_gene_len); 
            current_gene_seq[current_gene_len] = '\0';

            // Allocate extra space current alloc'ed array for ref scores isn't big enough
            n_ref_scores = gene.size * NUM_SCORES;
            if (m_ref_scores < n_ref_scores) {
                ref_scores = realloc(ref_scores, n_ref_scores * sizeof(float));
                m_ref_scores = n_ref_scores;
            }

            // Initialize blanks for the reference scores, NEITHER=1, ACCEPTOR=0, DONOR=0
            memset(ref_scores, 0, n_ref_scores * sizeof(float));
            for (int i = 0; i < n_ref_scores; i+=NUM_SCORES) {
                ref_scores[i] = 1;
            }

            // Override blanks with scores when present
            int chunk_end = gene.chunk_start + gene.n_chunks;
            for (int i = gene.chunk_start; i < chunk_end; i++) {
                const Chunk chunk = ref.chunks[i];

                int scores_end = chunk.scores_start + chunk.n_scores;
                for (int j = chunk.scores_start; j < scores_end; j++) {
                    const PositionScore score = ref.scores[j];
                    const int score_index = score.pos * NUM_SCORES;

                    ref_scores[score_index] = 1 - score.acceptor - score.acceptor;
                    ref_scores[score_index + ACCEPTOR_POS] = score.acceptor;
                    ref_scores[score_index + DONOR_POS] = score.donor;
                }
            }
        }

        // Replace ref by alt in gene sequence
        const int ref_len = strnlen(variant.ref, MAX_FIELD);
        const int alt_len = strnlen(variant.alt, MAX_FIELD);
        // const int allele_dif = alt_len - ref_len;

        const uint64_t pos_in_gene = variant.pos - current_gene_start;
        const int alt_seq_len = current_gene_len + alt_len - ref_len;
        char *alt_seq = malloc(alt_seq_len + 1);
        alt_seq[alt_seq_len] = '\0';

        // Copy everything before variant
        memcpy(
            alt_seq,
            current_gene_seq,
            pos_in_gene
        );

        // Copy over alt
        memcpy(
            alt_seq + pos_in_gene,
            variant.alt,
            alt_len
        );

        // Copy everything after variant
        memcpy(
            alt_seq + pos_in_gene + alt_len,
            current_gene_seq + (pos_in_gene + ref_len),
            current_gene_len - (pos_in_gene + ref_len)
        );

        // Add BOUNDAR_SIZE'd padding to the gene sequence, so that each position of the gene gets a prediction
        int padded_slen = alt_seq_len + CONTEXT_SIZE;
        char *padded_seq = malloc(padded_slen);
        memset(padded_seq, 'N', BOUNDARY_SIZE); // Prepend with 5000 Ns
        memcpy(padded_seq + BOUNDARY_SIZE, alt_seq, alt_seq_len);
        memset(padded_seq + (padded_slen - BOUNDARY_SIZE), 'N', BOUNDARY_SIZE); // Append with 5000 Ns
        free(alt_seq);

        float *encoding = malloc(padded_slen * ENCODING_SIZE * sizeof(float));
        memset(encoding, 0, padded_slen * ENCODING_SIZE * sizeof(float));
        int encoding_len = one_hot_encode(padded_seq, padded_slen, (float *) encoding);
        free(padded_seq);

        if (current_gene_strand == NEGATIVE_STRAND) reverse_encoding(encoding, encoding_len);

        // log_info("Predicting.");
        int num_alt_predictions;
        float *alt_predictions;
        predict(models, encoding_len, 1, (float *) encoding, &num_alt_predictions, &alt_predictions);
        free(encoding);

        if (current_gene_strand == NEGATIVE_STRAND) reverse_prediction(alt_predictions, num_alt_predictions, NUM_SCORES);

        // Fix predictions order
        if (alt_len != ref_len) {
            if (alt_len > ref_len) { // Insertion
                float max_donor = 0, max_acceptor = 0;
                for (int i = 0; i < alt_len; i++) {
                    int index = (pos_in_gene + i) * NUM_SCORES;

                    if (alt_predictions[index + DONOR_POS] > max_donor) {
                        max_donor = alt_predictions[index + DONOR_POS];
                    }
                    if (alt_predictions[index + ACCEPTOR_POS] > max_acceptor) {
                        max_acceptor = alt_predictions[index + ACCEPTOR_POS];
                    }
                }

                memmove(
                    alt_predictions + ((pos_in_gene + ref_len) * NUM_SCORES),
                    alt_predictions + ((pos_in_gene + alt_len) * NUM_SCORES),
                    (current_gene_len - (pos_in_gene + ref_len)) * NUM_SCORES * sizeof(float)
                );

                alt_predictions[pos_in_gene * NUM_SCORES] = max_donor + max_donor > 1.0 ? 0 : 1 - max_donor - max_acceptor;
                alt_predictions[pos_in_gene * NUM_SCORES + ACCEPTOR_POS] = max_acceptor;
                alt_predictions[pos_in_gene * NUM_SCORES + DONOR_POS] = max_donor;
            } else { // Deletion
                alt_predictions = realloc(alt_predictions, current_gene_len * NUM_SCORES * sizeof(float));

                memmove(
                    alt_predictions + ((pos_in_gene + ref_len) * NUM_SCORES),
                    alt_predictions + ((pos_in_gene + alt_len) * NUM_SCORES),
                    (current_gene_len - (pos_in_gene + ref_len)) * NUM_SCORES * sizeof(float)
                );

                for (int i = (pos_in_gene + alt_len) * NUM_SCORES; i < (pos_in_gene + ref_len) * NUM_SCORES; i += NUM_SCORES) {
                    alt_predictions[i] = 1.0;
                    alt_predictions[i + ACCEPTOR_POS] = 0.0;
                    alt_predictions[i + DONOR_POS] = 0.0;
                }
            }
        }

        for (int i = 0; i < current_gene_len; i++) {
            const float ref_acceptor = ref_scores[i * NUM_SCORES + ACCEPTOR_POS];
            const float ref_donor = ref_scores[i * NUM_SCORES + DONOR_POS];

            const float alt_acceptor = alt_predictions[i * NUM_SCORES + ACCEPTOR_POS];
            const float alt_donor = alt_predictions[i * NUM_SCORES + DONOR_POS];

            if (ref_acceptor < SCORE_THRESHOLD && ref_donor < SCORE_THRESHOLD && alt_acceptor < SCORE_THRESHOLD && alt_donor < SCORE_THRESHOLD) continue;

            fprintf(output, "%i,%f,%f,%f,%f\n", i, ref_acceptor, ref_donor, alt_acceptor, alt_donor);
        }

        free(alt_predictions);
    }

    if (current_gene_seq != NULL) free(current_gene_seq);
    free(ref_scores);

    destroy_models(models);
}

