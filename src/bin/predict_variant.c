#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/kstring.h>
#include <htslib/regidx.h>
#include <htslib/vcf.h>

#include "../logging/log.h"
#include "../predict.h"
#include "../utils.h"
#include "../reference.h"
#include "../gene_reference.h"
#include "../gene_regions.h"
#include "../variant_input.h"
#include "../variant_output.h"

#define REQUIRED_ARGS \
    REQUIRED_STRING_ARG(variants, "variants", "VCF or TSV file containing variants to predict for using SpliceAI") \
    REQUIRED_STRING_ARG(reference_bin, "reference_scores", "Binary file containing reference scores") \
    REQUIRED_STRING_ARG(model_dir, "model_dir", "Directory containing SpliceAI models") \
    REQUIRED_STRING_ARG(fasta, "fasta", "Human reference fasta") \
    REQUIRED_STRING_ARG(regions, "regions", "Gene region structure parsed from GFF with gff_to_bed.py") \
    REQUIRED_STRING_ARG(output, "output", "Annotated variants, in the same format as the input")

#define OPTIONAL_ARGS \
    OPTIONAL_INT_ARG(window_radius, 500, "--window-radius", "bases", "Bases scored either side of the variant") \
    OPTIONAL_STRING_ARG(input_format, "auto", "--input-format", "vcf|tsv|auto", "Format of the variants file. Detected from the file itself by default") \
    OPTIONAL_STRING_ARG(splice_output, "\0", "--splice-output", "file", "Output TSV with sparse splice predictions per variant")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>

/*
 * Score one alternate allele against one gene, appending the pipe-delimited annotation to
 * *info_str.
 *
 * The gene is identified by name alone; gene_reference_update resolves its start, end and
 * strand from the reference scores binary.
 */
int score_allele_for_gene(Model *models, faidx_t *fa, const Reference *ref, GeneReference *gene_reference, int window_radius, int window_size, const char *chrom, const char *gene_name, hts_pos_t pos, const char *ref_allele, char *alt_allele, kstring_t *info_str) {
    int ref_len = strlen(ref_allele);
    int alt_len = strlen(alt_allele);

    if (ref_len > window_radius || alt_len > window_radius) {
        log_warn("Oversized indel found. Skipping prediction for at %s:%"PRIhts_pos":%s", chrom, pos + 1, gene_name);
        kputc('.', info_str);
        return EXIT_SUCCESS;
    }

    // The name comparison is a cache; gene_reference_update clears the name on failure, so a
    // match means the rest of the struct belongs to this gene.
    if (strncmp(gene_name, gene_reference->name, FIELD_MAX_LEN) != 0) {
        if (gene_reference_update(chrom, gene_name, fa, ref, gene_reference) == EXIT_FAILURE) {
            log_warn("Failed to find reference for gene %s. Skipping prediction for %s:%"PRIhts_pos":%s...", gene_name, chrom, pos + 1, gene_name);
            kputc('.', info_str);
            return EXIT_SUCCESS;
        }
    }

    int num_ref_predictions = window_size * NUM_SCORES;
    float *ref_predictions;
    if (gene_reference_get_score_window(pos, window_radius, gene_reference, &ref_predictions) == EXIT_FAILURE) {
        // Memory issue, already logged by function
        return EXIT_FAILURE;
    }

    // Predict alt
    int num_alt_predictions;
    float *alt_predictions;

    // The model trims BOUNDARY_SIZE (half of CONTEXT_SIZE) of context from each side of its
    // input to produce predictions for the remaining "core" region, so the window built here
    // must reserve BOUNDARY_SIZE (not the full CONTEXT_SIZE) of margin on each side of the
    // window_size-wide region of interest for padded_seq's width (CONTEXT_SIZE + window_size)
    // to hold it.
    hts_pos_t gene_pos = pos - gene_reference->start;

    const int width = CONTEXT_SIZE + window_size;
    char padded_seq[width];
    build_alt_window(&gene_reference->seq, gene_pos, ref_len, alt_allele, alt_len,
                     padded_seq, width, BOUNDARY_SIZE + window_radius);

    if (predict_padded_sequence(models, padded_seq, width, gene_reference->strand, &alt_predictions, &num_alt_predictions) != EXIT_SUCCESS) {
        free(ref_predictions);
        return EXIT_FAILURE;
    }

    // Align
    if (alt_len != ref_len) {
        align_predictions_alt_to_ref(window_radius, window_size, ref_len, alt_len, &alt_predictions);
    }

    // Form scores
    Score score = calculate_delta_scores(alt_allele, (char *) gene_name, ref_predictions, alt_predictions, num_ref_predictions, window_radius);
    free(ref_predictions);
    free(alt_predictions);

    const char *score_fmt = "%s|%s|%.2f|%.2f|%.2f|%.2f|%d|%d|%d|%d";
    int tmp_len = snprintf(NULL, 0, score_fmt, score.alt, score.gene, score.ag, score.al, score.dg, score.dl, score.ag_idx, score.al_idx, score.dg_idx, score.dl_idx);
    char *tmp = malloc(tmp_len + 1);
    if (tmp == NULL) {
        log_fatal("Failed to allocate %d bytes for score string", tmp_len + 1);
        exit(EXIT_FAILURE);
    }
    snprintf(tmp, tmp_len + 1, score_fmt, score.alt, score.gene, score.ag, score.al, score.dg, score.dl, score.ag_idx, score.al_idx, score.dg_idx, score.dl_idx);
    kputs(tmp, info_str);
    free(tmp);

    return EXIT_SUCCESS;
}

/*
 * Annotate one input record and hand it to the writer.
 *
 * annotations is scratch owned by the caller, holding at least record->n_alt entries: one
 * annotation string per alternate allele, each comma-joining that allele's overlapping genes.
 */
int process_variant_record(Model *models, faidx_t *fa, const Reference *ref, GeneReference *gene_reference, int window_radius, int window_size, regidx_t *gene_index, regitr_t *itr, GeneList *genes, const VariantRecord *record, kstring_t *annotations, VariantWriter *writer) {
    for (int i = 0; i < record->n_alt; i++) annotations[i].l = 0;

    // Nothing to say about a variant no gene fully contains: write it through untouched.
    const int ref_len = strlen(record->ref);
    if (gene_regions_containing(gene_index, itr, record->chrom, record->pos, ref_len, genes) == 0) {
        return variant_writer_write(writer, record, annotations);
    }

    for (int i = 0; i < record->n_alt; i++) {
        char *alt_allele = record->alt[i];

        if ('.' == alt_allele[0] || // Deletion
            '*' == alt_allele[0] || // Missing
            '<' == alt_allele[0] // <ID> string
        ) {
            log_warn("Unsupported alternate allele found: %s. Skipping prediction(s) for %s:%"PRIhts_pos, alt_allele, record->chrom, record->pos + 1);
            kputc('.', &annotations[i]);
            continue;
        }

        for (size_t g = 0; g < genes->n; g++) {
            if (annotations[i].l > 0) kputc(',', &annotations[i]);

            if (score_allele_for_gene(models, fa, ref, gene_reference, window_radius, window_size, record->chrom, genes->genes[g].name, record->pos, record->ref, alt_allele, &annotations[i]) != EXIT_SUCCESS) {
                return EXIT_FAILURE;
            }
        }
    }

    return variant_writer_write(writer, record, annotations);
}

int main(int argc, char *argv[]) {
    setenv("TF_CPP_MIN_LOG_LEVEL", "2", 1);

    args_t args = make_default_args();
    if (!parse_args(argc, argv, &args) || args.help) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    const char *variants = args.variants;
    const char *reference_bin = args.reference_bin;
    const char *model_dir = args.model_dir;
    const char *fasta = args.fasta;
    const char *gene_regions = args.regions;
    const char *annotated_variants = args.output;

    const int window_radius = args.window_radius;
    const char *prediction_output = args.splice_output;
    const bool produce_splice_output = prediction_output[0] != '\0';
    (void) produce_splice_output; // --splice-output is parsed but not yet implemented

    // Opened before load_models so a bad format value or unusable path fails cheaply.
    VariantFormat input_format;
    if (variant_input_format_parse(args.input_format, &input_format) != EXIT_SUCCESS) return EXIT_FAILURE;

    VariantReader *reader;
    if (variant_reader_open(variants, input_format, &reader) != EXIT_SUCCESS) return EXIT_FAILURE;

    VariantWriter *writer;
    if (variant_writer_open(annotated_variants, reader, &writer) != EXIT_SUCCESS) return EXIT_FAILURE;

    regidx_t *gene_index = NULL;
    uint64_t regions_digest;
    if (gene_regions_build_regidx(gene_regions, &gene_index, &regions_digest) != EXIT_SUCCESS) return EXIT_FAILURE;

    Reference ref;
    if (reference_read(reference_bin, &ref) != EXIT_SUCCESS) {
        log_error("Failed to read reference scores binary: %s", reference_bin);
        return EXIT_FAILURE;
    }

    faidx_t *fa_in;
    if ((fa_in = fai_load(fasta)) == NULL) return EXIT_FAILURE; // Load reference fasta for sequence lookup

    // Checked before load_models: a mismatch is silently wrong, not loudly broken.
    if (reference_check_inputs(&ref, fasta_digest(fa_in), regions_digest, reference_bin) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    // Load SpliceAI models
    Model *models = load_models(model_dir);

    // Bases scored: the variant plus window_radius either side.
    const int window_size = 2 * window_radius + 1;

    // Loop initialisations
    regitr_t *itr = regitr_init(gene_index);
    GeneList genes;
    gene_list_init(&genes);
    GeneReference gene_reference;
    gene_reference_init(&gene_reference);

    // One annotation string per alternate allele, grown to fit the widest record seen.
    kstring_t *annotations = NULL;
    int m_annotations = 0;

    int ret = EXIT_SUCCESS;
    VariantRecord record;
    int read_status;
    while ((read_status = variant_reader_next(reader, &record)) == EXIT_SUCCESS) {
        if (record.n_alt > m_annotations) {
            kstring_t *grown = realloc(annotations, record.n_alt * sizeof(kstring_t));
            if (grown == NULL) {
                log_fatal("Failed to allocate %zu bytes for annotations", record.n_alt * sizeof(kstring_t));
                exit(EXIT_FAILURE);
            }
            annotations = grown;
            memset(annotations + m_annotations, 0, (record.n_alt - m_annotations) * sizeof(kstring_t));
            m_annotations = record.n_alt;
        }

        if (process_variant_record(models, fa_in, &ref, &gene_reference, window_radius, window_size, gene_index, itr, &genes, &record, annotations, writer) != EXIT_SUCCESS) {
            ret = EXIT_FAILURE;
            break;
        }
    }

    if (read_status == EXIT_FAILURE) ret = EXIT_FAILURE;

    for (int i = 0; i < m_annotations; i++) free(annotations[i].s);
    free(annotations);

    gene_reference_destroy(&gene_reference);
    gene_list_destroy(&genes);
    regitr_destroy(itr);
    regidx_destroy(gene_index);
    variant_writer_close(writer);
    variant_reader_close(reader);
    fai_destroy(fa_in);

    destroy_models(models);

    return ret;
}
