#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/kstring.h>
#include <htslib/regidx.h>

#include "../logging/log.h"
#include "../predict.h"
#include "../gene_regions.h"
#include "../gene_reference.h"
#include "../reference.h"
#include "../utils.h"
#include "../variant_input.h"

#define SCORE_THRESHOLD ZERO_EPSILON

#define REQUIRED_ARGS \
    REQUIRED_STRING_ARG(variants, "variants", "VCF or TSV file containing variants to predict for using SpliceAI") \
    REQUIRED_STRING_ARG(reference_bin, "reference_scores", "Binary file containing reference scores") \
    REQUIRED_STRING_ARG(model_dir, "model_dir", "Directory containing SpliceAI models") \
    REQUIRED_STRING_ARG(fasta, "fasta", "Human reference fasta") \
    REQUIRED_STRING_ARG(regions, "regions", "Gene region structure parsed from GFF with gff_to_bed.py") \
    REQUIRED_STRING_ARG(output, "output", "TSV of splice sites found, where REF or ALT scores exceed 0.001.")

#define OPTIONAL_ARGS \
    OPTIONAL_STRING_ARG(input_format, "auto", "--input-format", "vcf|tsv|auto", "Format of the variants file. Detected from the file itself by default")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>

/*
 * Predict over the whole gene with one alternate allele substituted in, leaving
 * *alt_predictions aligned position-for-position with the gene's reference scores.
 */
int process_variant_row(Model *models, faidx_t *fa, const Reference *ref, GeneReference *current_gene, const char *chrom, const char *gene_name, hts_pos_t pos, const char *ref_allele, const char *alt_allele, float **alt_predictions, int *num_alt_predictions) {
    // If gene is different from previous variant, we need to load the reference scores for the current gene
    if (strncmp(gene_name, current_gene->name, FIELD_MAX_LEN) != 0) {
        if (gene_reference_update(chrom, gene_name, fa, ref, current_gene) != EXIT_SUCCESS) {
            log_warn("Failed to find reference for gene %s. Skipping variant %s:%"PRIhts_pos".", gene_name, chrom, pos + 1);
            return EXIT_FAILURE;
        }
    }

    // Replace ref by alt in gene sequence
    const int ref_len = strlen(ref_allele);
    const int alt_len = strlen(alt_allele);
    const uint64_t pos_in_gene = pos - current_gene->start;

    kstring_t alt = { 0 };
    create_alt_seq(&current_gene->seq, pos_in_gene, ref_len, alt_len, alt_allele, &(alt.s), &(alt.l));

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

    // Opened before load_models so a bad format value or unreadable path fails cheaply.
    VariantFormat input_format;
    if (variant_input_format_parse(args.input_format, &input_format) != EXIT_SUCCESS) return EXIT_FAILURE;

    VariantReader *reader;
    if (variant_reader_open(args.variants, input_format, &reader) != EXIT_SUCCESS) return EXIT_FAILURE;

    regidx_t *gene_index = NULL;
    uint64_t regions_digest;
    if (gene_regions_build_regidx(args.regions, &gene_index, &regions_digest) != EXIT_SUCCESS) return EXIT_FAILURE;

    // Load reference from binary file
    Reference ref;
    if (reference_read(args.reference_bin, &ref) != EXIT_SUCCESS) {
        log_error("Failed to read reference scores binary: %s", args.reference_bin);
        return EXIT_FAILURE;
    }

    // Load reference fasta for sequence lookup
    faidx_t *fa_in;
    if ((fa_in = fai_load(args.fasta)) == NULL) return EXIT_FAILURE; // Load reference fasta for sequence lookup

    // Checked before load_models: a mismatch is silently wrong, not loudly broken.
    if (reference_check_inputs(&ref, fasta_digest(fa_in), regions_digest, args.reference_bin) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    FILE *output = open_file_or_log(args.output, "w");
    if (output == NULL) return EXIT_FAILURE;

    // Load SpliceAI tensorflow models
    Model *models = load_models(args.model_dir);

    // Loop initialisations
    regitr_t *itr = regitr_init(gene_index);
    GeneList genes;
    gene_list_init(&genes);
    GeneReference current_gene;
    gene_reference_init(&current_gene);

    int ret = EXIT_SUCCESS;
    VariantRecord record;
    int read_status;
    while ((read_status = variant_reader_next(reader, &record)) == EXIT_SUCCESS) {
        const int record_ref_len = strlen(record.ref);
        if (gene_regions_containing(gene_index, itr, record.chrom, record.pos, record_ref_len, &genes) == 0) {
            log_warn("No gene fully contains %s:%"PRIhts_pos". Skipping variant.", record.chrom, record.pos + 1);
            continue;
        }

        // One score block per (allele, gene) pair: each is an independent prediction.
        for (int i = 0; i < record.n_alt; i++) {
            for (size_t g = 0; g < genes.n; g++) {
                const Gene *gene = &genes.genes[g];

                float *alt_predictions;
                int num_alt_predictions;
                if (process_variant_row(models, fa_in, &ref, &current_gene, record.chrom, gene->name, record.pos, record.ref, record.alt[i], &alt_predictions, &num_alt_predictions) != EXIT_SUCCESS) {
                    continue;
                }

                fprintf(output, "#%s_%c_%li_%li:%s_%"PRIhts_pos"_%s_%s\n", gene->name, current_gene.strand, current_gene.start, current_gene.end, record.chrom, record.pos + 1, record.ref, record.alt[i]);

                log_info("%s\t%li\t%li\t%s\t%c\t%i", record.chrom, current_gene.start, current_gene.end, current_gene.name, current_gene.strand, current_gene.end - current_gene.start);

                write_gene_scores(output, &current_gene, alt_predictions);

                free(alt_predictions);
            }
        }
    }

    if (read_status == EXIT_FAILURE) ret = EXIT_FAILURE;

    gene_reference_destroy(&current_gene);
    gene_list_destroy(&genes);
    regitr_destroy(itr);
    regidx_destroy(gene_index);
    variant_reader_close(reader);
    fclose(output);
    fai_destroy(fa_in);

    destroy_models(models);

    return ret;
}
