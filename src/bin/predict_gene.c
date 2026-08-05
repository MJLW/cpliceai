#include <htslib/hts.h>
#include <htslib/kstring.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <htslib/faidx.h>
#include <string.h>

#include "../logging/log.h"
#include "../predict.h"
#include "../gene_regions.h"
#include "../gene_reference.h"
#include "../reference.h"
#include "../utils.h"

#define MAX_LINE 8192

#define SCORE_THRESHOLD ZERO_EPSILON

#define REQUIRED_ARGS \
    REQUIRED_STRING_ARG(variants, "variants", "TSV file containing variants to predict for using SpliceAI") \
    REQUIRED_STRING_ARG(reference_bin, "reference_scores", "Binary file containing reference scores") \
    REQUIRED_STRING_ARG(model_dir, "model_dir", "Directory containing SpliceAI models") \
    REQUIRED_STRING_ARG(fasta, "fasta", "Human reference fasta") \
    REQUIRED_STRING_ARG(output, "output", "TSV of splice sites found, where REF>0.2|ALT>0.2.")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>


typedef struct {
    char chr[FIELD_MAX_LEN];
    uint64_t pos;
    char ref[FIELD_MAX_LEN];
    char alt[FIELD_MAX_LEN];
    char gene[FIELD_MAX_LEN];
} Variant;

static int parse_line(const char *line, Variant *snv) {
    char buf[MAX_LINE];
    strncpy(buf, line, MAX_LINE-1);
    buf[MAX_LINE-1] = '\0';
    buf[strcspn(buf, "\r\n")] = '\0';

    char *cursor = buf;
    char *tok;
    tok = next_tsv_field(&cursor);
    if (!tok) {
        log_error("Failed to parse SNV contig from line: %s", line);
        return EXIT_FAILURE;
    }
    strncpy(snv->chr, tok, FIELD_MAX_LEN-1);
    snv->chr[FIELD_MAX_LEN-1] = '\0';

    tok = next_tsv_field(&cursor);
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

    tok = next_tsv_field(&cursor);
    if (!tok) {
        log_error("Failed to parse reference base(s) from line: %s", line);
        return EXIT_FAILURE;
    }
    strncpy(snv->ref, tok, FIELD_MAX_LEN-1);
    snv->ref[FIELD_MAX_LEN-1] = '\0';

    tok = next_tsv_field(&cursor);
    if (!tok) {
        log_error("Failed to parse alternative base(s) from line: %s", line);
        return EXIT_FAILURE;
    }
    strncpy(snv->alt, tok, FIELD_MAX_LEN-1);
    snv->alt[FIELD_MAX_LEN-1] = '\0';

    tok = next_tsv_field(&cursor);
    if (!tok) {
        log_error("Failed to parse SNV gene from line: %s", line);
        return EXIT_FAILURE;
    }
    strncpy(snv->gene, tok, FIELD_MAX_LEN-1);
    snv->gene[FIELD_MAX_LEN-1] = '\0';

    return EXIT_SUCCESS;
}

int variant_tsv_read_next(FILE *fp, Variant *variant) {
    char line[MAX_LINE];

    // WARN: Assumes skipped/no header
    while(fgets(line, sizeof(line), fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\r') continue;

        if (parse_line(line, variant) != EXIT_SUCCESS) {
            log_error("Failed parsing of %s");
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;

    }

    return EXIT_FAILURE;
}

int process_variant_row(Model *models, faidx_t *fa, const Reference *ref, GeneReference *current_gene, const Variant *variant, float **alt_predictions, int *num_alt_predictions) {
    // If gene is different from previous variant, we need to load the reference scores for the current gene
    if (strncmp(variant->gene, current_gene->name, FIELD_MAX_LEN) != 0) {
        if (gene_reference_update(variant->chr, variant->gene, fa, ref, current_gene) != EXIT_SUCCESS) {
            log_warn("Failed to find reference for gene %s. Skipping variant %s:%li.", variant->gene, variant->chr, variant->pos + 1);
            return EXIT_FAILURE;
        }
    }

    // Replace ref by alt in gene sequence
    const int ref_len = strnlen(variant->ref, FIELD_MAX_LEN);
    const int alt_len = strnlen(variant->alt, FIELD_MAX_LEN);
    const uint64_t pos_in_gene = variant->pos - current_gene->start;

    kstring_t alt = { 0 };
    create_alt_seq(&current_gene->seq, pos_in_gene, ref_len, alt_len, variant->alt, &(alt.s), &(alt.l));

    // Add BOUNDAR_SIZE'd padding to the gene sequence, so that each position of the gene gets a prediction
    int padded_slen = alt.l + CONTEXT_SIZE;
    char *padded_seq = malloc(padded_slen);
    if (padded_seq == NULL) {
        log_fatal("Failed to allocate %d bytes for padded sequence", padded_slen);
        exit(EXIT_FAILURE);
    }
    memset(padded_seq, 'N', BOUNDARY_SIZE); // Prepend with 5000 Ns
    memcpy(padded_seq + BOUNDARY_SIZE, alt.s, alt.l);
    memset(padded_seq + (padded_slen - BOUNDARY_SIZE), 'N', BOUNDARY_SIZE); // Append with 5000 Ns
    free(alt.s);

    if (predict_padded_sequence(models, padded_seq, padded_slen, current_gene->strand, alt_predictions, num_alt_predictions) != EXIT_SUCCESS) {
        free(padded_seq);
        return EXIT_FAILURE;
    }
    free(padded_seq);

    // Fix predictions order
    if (alt_len != ref_len) {
        align_predictions_alt_to_ref(pos_in_gene, current_gene->seq.l, ref_len, alt_len, alt_predictions);
    }

    return EXIT_SUCCESS;
}

void write_gene_scores(FILE *output, const GeneReference *gene, const float *alt_predictions) {
    for (int i = 0; i < gene->seq.l; i++) {
        const float ref_acceptor = gene->scores[i * NUM_SCORES + ACCEPTOR_POS];
        const float ref_donor = gene->scores[i * NUM_SCORES + DONOR_POS];

        const float alt_acceptor = alt_predictions[i * NUM_SCORES + ACCEPTOR_POS];
        const float alt_donor = alt_predictions[i * NUM_SCORES + DONOR_POS];

        if (ref_acceptor < SCORE_THRESHOLD && ref_donor < SCORE_THRESHOLD && alt_acceptor < SCORE_THRESHOLD && alt_donor < SCORE_THRESHOLD) continue;

        fprintf(output, "%li\t%f\t%f\t%f\t%f\n", i + gene->start + 1, ref_acceptor, ref_donor, alt_acceptor, alt_donor);
    }
}

int main(int argc, char *argv[]) {
    setenv("TF_CPP_MIN_LOG_LEVEL", "2", 1);

    // Parse arguments
    args_t args = make_default_args();
    if (!parse_args(argc, argv, &args) || args.help) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    // Load SpliceAI tensorflow models
    Model *models = load_models(args.model_dir);

    // Load reference from binary file
    Reference ref;
    if (reference_read(args.reference_bin, &ref) != EXIT_SUCCESS) {
        log_error("Failed to read reference scores binary: %s", args.reference_bin);
        return EXIT_FAILURE;
    }

    // Load reference fasta for sequence lookup
    faidx_t *fa_in;
    if ((fa_in = fai_load(args.fasta)) == NULL) return EXIT_FAILURE; // Load reference fasta for sequence lookup

    FILE *variants_fp = open_file_or_log(args.variants, "r");
    if (variants_fp == NULL) return EXIT_FAILURE;

    FILE *output = open_file_or_log(args.output, "w");
    if (output == NULL) return EXIT_FAILURE;

    // Skip first line
    char buffer[MAX_LINE];
    if (fgets(buffer, sizeof(buffer), variants_fp) == NULL) {
        log_error("Could not read from file: %s", args.variants);
        return EXIT_FAILURE;
    }

    // Loop over VCF variants
    Variant variant;
    GeneReference current_gene;
    gene_reference_init(&current_gene);

    while (variant_tsv_read_next(variants_fp, &variant) == EXIT_SUCCESS) {
        float *alt_predictions;
        int num_alt_predictions;
        if (process_variant_row(models, fa_in, &ref, &current_gene, &variant, &alt_predictions, &num_alt_predictions) != EXIT_SUCCESS) {
            continue;
        }

        fprintf(output, "#%s_%c_%li_%li:%s_%li_%s_%s\n", variant.gene, current_gene.strand, current_gene.start, current_gene.end, variant.chr, variant.pos+1, variant.ref, variant.alt);

        log_info("%s\t%li\t%li\t%s\t%c\t%i", variant.chr, current_gene.start, current_gene.end, current_gene.name, current_gene.strand, current_gene.end - current_gene.start);

        write_gene_scores(output, &current_gene, alt_predictions);

        free(alt_predictions);
    }

    if (current_gene.seq.s != NULL) free(current_gene.seq.s);
    free(current_gene.scores);

    destroy_models(models);
}

