#include <htslib/hts.h>
#include <htslib/kstring.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <htslib/faidx.h>
#include <string.h>

#include "../logging/log.h"
#include "../predict.h"
#include "../gene_regions.h"
#include "../gene_reference.h"
#include "../reference.h"
#include "../utils.h"

#define MAX_LINE 8192
#define MAX_FIELD 256

#define INITIAL_REF_SIZE_MALLOC 100000
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

int Variant_tsv_read_next(FILE *fp, Variant *variant) {
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

int main(int argc, char *argv[]) {
    setenv("TF_CPP_MIN_LOG_LEVEL", "2", 1);
    setenv("NVIDIA_TF32_OVERRIDE", "1", 1);
    setenv("TF_CUDNN_USE_AUTOTUNE", "0", 1);

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
    Reference_read(args.reference_bin, &ref);

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

    // Skip first line
    char buffer[MAX_LINE];
    if (fgets(buffer, sizeof(buffer), variants_fp) == NULL) {
        log_error("Could not read from file: %s", args.variants);
        return EXIT_FAILURE;
    }

    // Loop over VCF variants
    Variant variant;
    GeneReference current_gene;
    GeneReference_init(&current_gene);

    while (Variant_tsv_read_next(variants_fp, &variant) == EXIT_SUCCESS) {
        // If gene is different from previous variant, we need to load the reference scores for the current gene
        if (strcmp(variant.gene, current_gene.name) != 0) {
            GeneReference_update(variant.chr, variant.gene, fa_in, &ref, &current_gene);
        }

        fprintf(output, "#%s_%c_%li_%li:%s_%li_%s_%s\n", variant.gene, current_gene.strand, current_gene.start, current_gene.end, variant.chr, variant.pos+1, variant.ref, variant.alt);

        // Replace ref by alt in gene sequence
        const int ref_len = strnlen(variant.ref, MAX_FIELD);
        const int alt_len = strnlen(variant.alt, MAX_FIELD);
        const uint64_t pos_in_gene = variant.pos - current_gene.start;

        kstring_t alt = { 0 };
        create_alt_seq(&current_gene.seq, pos_in_gene, ref_len, alt_len, variant.alt, &(alt.s), &(alt.l));

        // Add BOUNDAR_SIZE'd padding to the gene sequence, so that each position of the gene gets a prediction
        int padded_slen = alt.l + CONTEXT_SIZE;
        char *padded_seq = malloc(padded_slen);
        memset(padded_seq, 'N', BOUNDARY_SIZE); // Prepend with 5000 Ns
        memcpy(padded_seq + BOUNDARY_SIZE, alt.s, alt.l);
        memset(padded_seq + (padded_slen - BOUNDARY_SIZE), 'N', BOUNDARY_SIZE); // Append with 5000 Ns
        free(alt.s);

        float *encoding = malloc(padded_slen * ENCODING_SIZE * sizeof(float));
        memset(encoding, 0, padded_slen * ENCODING_SIZE * sizeof(float));
        int encoding_len = one_hot_encode(padded_seq, padded_slen, (float *) encoding);
        free(padded_seq);

        if (current_gene.strand == NEGATIVE_STRAND) reverse_encoding(encoding, encoding_len);

        int num_alt_predictions;
        float *alt_predictions;
        predict(models, encoding_len, 1, (float *) encoding, &num_alt_predictions, &alt_predictions);
        free(encoding);

        if (current_gene.strand == NEGATIVE_STRAND) reverse_prediction(alt_predictions, num_alt_predictions, NUM_SCORES);

        // Fix predictions order
        if (alt_len != ref_len) {
            align_predictions_alt_to_ref(pos_in_gene, current_gene.seq.l, ref_len, alt_len, &alt_predictions);
        }

        log_info("%s\t%li\t%li\t%s\t%c\t%i", variant.chr, current_gene.start, current_gene.end, current_gene.name, current_gene.strand, current_gene.end - current_gene.start);

        for (int i = 0; i < current_gene.seq.l; i++) {
            const float ref_acceptor = current_gene.scores[i * NUM_SCORES + ACCEPTOR_POS];
            const float ref_donor = current_gene.scores[i * NUM_SCORES + DONOR_POS];

            const float alt_acceptor = alt_predictions[i * NUM_SCORES + ACCEPTOR_POS];
            const float alt_donor = alt_predictions[i * NUM_SCORES + DONOR_POS];

            if (ref_acceptor < SCORE_THRESHOLD && ref_donor < SCORE_THRESHOLD && alt_acceptor < SCORE_THRESHOLD && alt_donor < SCORE_THRESHOLD) continue;

            fprintf(output, "%li\t%f\t%f\t%f\t%f\n", i + current_gene.start + 1, ref_acceptor, ref_donor, alt_acceptor, alt_donor);
        }

        free(alt_predictions);
    }

    if (current_gene.seq.s != NULL) free(current_gene.seq.s);
    free(current_gene.scores);

    destroy_models(models);
}

