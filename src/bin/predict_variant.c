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
#include "../haplotype.h"
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
    BOOLEAN_ARG(include_unphased, "--include-unphased", "Score heterozygous variants whose phase is unknown, placing them on both haplotypes") \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>

/*
 * The three sequences a variant is scored on, beyond the reference genome itself.
 *
 * ALT is the variant alone. HAP_REF is the copy of the chromosome it sits on with every other
 * co-phased variant applied but not this one, and HAP_ALT is that same copy complete. Comparing
 * REF to ALT says what the variant does in isolation; HAP_REF to HAP_ALT says what it adds to
 * the molecule it is really on; REF to HAP_ALT says what that whole molecule does.
 */
typedef enum { TRACK_ALT, TRACK_HAP_REF, TRACK_HAP_ALT, TRACK_COUNT } Track;

/* "1", "2", or "." for a variant with no genotype and so no copy to name. */
static const char *hap_label(const int hap) {
    if (hap == HAP_1) return "1";
    if (hap == HAP_2) return "2";
    return ".";
}

/*
 * Append one pipe-delimited annotation, comma-separating it from whatever is already there.
 * hap is NULL for the haplotype-independent SpliceAI field, which carries no copy.
 */
static void append_score(kstring_t *out, const Score *score, const char *hap) {
    if (out->l > 0) kputc(',', out);

    ksprintf(out, "%s|%s|", score->alt, score->gene);
    if (hap != NULL) ksprintf(out, "%s|", hap);
    ksprintf(out, "%.2f|%.2f|%.2f|%.2f|%d|%d|%d|%d",
             score->ag, score->al, score->dg, score->dl,
             score->ag_idx, score->al_idx, score->dg_idx, score->dl_idx);
}

/* Every annotation this allele produces, one string per output field. */
typedef struct {
    kstring_t *spliceai;
    kstring_t *spliceai_hap;
    kstring_t *spliceai_tot;
} Annotations;

/*
 * Score one alternate allele against one gene, on every copy of the chromosome that carries it,
 * appending each result to the matching field.
 *
 * The gene is identified by name alone; gene_reference_update resolves its start, end and
 * strand from the reference scores binary.
 */
int score_allele_for_gene(Model *models, faidx_t *fa, const Reference *ref, GeneReference *gene_reference,
                          const HapBuffer *buffer, SeqEditList *edits, int window_radius, int window_size,
                          const char *chrom, const char *gene_name, const HapRecord *record,
                          int alt_index, const Annotations *out) {
    char *alt_allele = record->alt[alt_index];
    const int ref_len = strlen(record->ref);
    const int alt_len = strlen(alt_allele);
    const hts_pos_t pos = record->pos;

    if (ref_len > window_radius || alt_len > window_radius) {
        log_warn("Oversized indel found. Skipping prediction for at %s:%"PRIhts_pos":%s", chrom, pos + 1, gene_name);
        kputc('.', out->spliceai);
        return EXIT_SUCCESS;
    }

    // The name comparison is a cache; gene_reference_update clears the name on failure, so a
    // match means the rest of the struct belongs to this gene.
    if (strncmp(gene_name, gene_reference->name, FIELD_MAX_LEN) != 0) {
        if (gene_reference_update(chrom, gene_name, fa, ref, gene_reference) == EXIT_FAILURE) {
            log_warn("Failed to find reference for gene %s. Skipping prediction for %s:%"PRIhts_pos":%s...", gene_name, chrom, pos + 1, gene_name);
            kputc('.', out->spliceai);
            return EXIT_SUCCESS;
        }
    }

    const int num_scores = window_size * NUM_SCORES;
    float *ref_predictions;
    if (gene_reference_get_score_window(pos, window_radius, gene_reference, &ref_predictions) == EXIT_FAILURE) {
        // Memory issue, already logged by function
        return EXIT_FAILURE;
    }

    /*
     * The copies to report on. Without a genotype there is one pass carrying no copy at all,
     * whose haplotype is the reference: HAP_REF and HAP_ALT then coincide with REF and ALT, and
     * all three comparisons agree.
     */
    int haps[HAP_COUNT];
    int n_haps = 0;
    if (!record->has_gt) {
        haps[n_haps++] = 0;
    } else {
        for (int h = 0; h < HAP_COUNT; h++) {
            if (record->hap_mask[alt_index] & HAP_MASK(h)) haps[n_haps++] = HAP_MASK(h);
        }
    }

    const int64_t gene_pos = pos - (hts_pos_t) gene_reference->start;
    const SeqEdit self = { gene_pos, ref_len, alt_allele, alt_len };

    // Variants beyond this cannot reach any scored position: every one of them depends only on
    // the BOUNDARY_SIZE bases either side of it, and the scored region is window_radius wide.
    const int64_t reach = BOUNDARY_SIZE + window_radius;
    const hts_pos_t lo = pos - reach > (hts_pos_t) gene_reference->start ? pos - reach : (hts_pos_t) gene_reference->start;
    const hts_pos_t hi = pos + reach < (hts_pos_t) gene_reference->end ? pos + reach : (hts_pos_t) gene_reference->end;

    int ret = EXIT_SUCCESS;
    float *scores[TRACK_COUNT] = { NULL, NULL, NULL };

    for (int h = 0; h < n_haps && ret == EXIT_SUCCESS; h++) {
        const int hap = haps[h];

        // HAP_REF is this copy without the variant; adding it back gives HAP_ALT.
        hap_edits_collect(buffer, hap, chrom, (hts_pos_t) gene_reference->start, lo, hi, record, alt_index, edits);
        const bool background_empty = edits->n == 0;

        /*
         * Both windows have to cover the same reference positions as the reference scores, and
         * an edge landing inside a deletion has no reference position to be aligned back to.
         * Widening to the edit's own boundary and then reading the middle out keeps the three
         * arrays the same length.
         */
        int64_t ref_lo = gene_pos - window_radius;
        int64_t ref_hi = gene_pos + window_radius + 1;
        seq_edits_snap(edits->edits, (int) edits->n, &ref_lo, &ref_hi);
        seq_edits_snap(&self, 1, &ref_lo, &ref_hi);
        const int64_t window_offset = (gene_pos - window_radius) - ref_lo;

        if (scores[TRACK_ALT] == NULL) {
            if (gene_reference_predict(models, gene_reference, &self, 1, ref_lo, ref_hi, &scores[TRACK_ALT]) != EXIT_SUCCESS) {
                ret = EXIT_FAILURE;
                break;
            }
        }

        if (background_empty) {
            /*
             * Nothing else on this copy, so its haplotype is the reference genome and its
             * complete form is the variant alone. Reusing the arrays rather than predicting
             * them again is what makes an unphased or isolated variant cost no more than it
             * did before haplotypes existed - and makes the three fields agree exactly.
             */
            scores[TRACK_HAP_REF] = NULL;
            scores[TRACK_HAP_ALT] = NULL;
        } else {
            if (gene_reference_predict(models, gene_reference, edits->edits, (int) edits->n, ref_lo, ref_hi, &scores[TRACK_HAP_REF]) != EXIT_SUCCESS) {
                ret = EXIT_FAILURE;
                break;
            }

            // The same background with the variant put back in, which is the whole copy.
            seq_edit_list_push_sorted(edits, self);
            if (gene_reference_predict(models, gene_reference, edits->edits, (int) edits->n, ref_lo, ref_hi, &scores[TRACK_HAP_ALT]) != EXIT_SUCCESS) {
                ret = EXIT_FAILURE;
                break;
            }
        }

        const float *alt_window = scores[TRACK_ALT] + window_offset * NUM_SCORES;
        const float *hap_ref_window = background_empty ? ref_predictions : scores[TRACK_HAP_REF] + window_offset * NUM_SCORES;
        const float *hap_alt_window = background_empty ? alt_window : scores[TRACK_HAP_ALT] + window_offset * NUM_SCORES;

        // SpliceAI does not depend on the copy, so it is reported once however many carry it.
        if (h == 0) {
            const Score isolated = calculate_delta_scores(alt_allele, (char *) gene_name, ref_predictions, (float *) alt_window, num_scores, window_radius);
            append_score(out->spliceai, &isolated, NULL);
        }

        const Score marginal = calculate_delta_scores(alt_allele, (char *) gene_name, (float *) hap_ref_window, (float *) hap_alt_window, num_scores, window_radius);
        append_score(out->spliceai_hap, &marginal, hap_label(hap));

        const Score total = calculate_delta_scores(alt_allele, (char *) gene_name, ref_predictions, (float *) hap_alt_window, num_scores, window_radius);
        append_score(out->spliceai_tot, &total, hap_label(hap));

        free(scores[TRACK_HAP_REF]);
        free(scores[TRACK_HAP_ALT]);
        scores[TRACK_HAP_REF] = NULL;
        scores[TRACK_HAP_ALT] = NULL;

        // A second copy may snap the window differently, so the ALT window is rebuilt with it.
        if (n_haps > 1) {
            free(scores[TRACK_ALT]);
            scores[TRACK_ALT] = NULL;
        }
    }

    free(scores[TRACK_ALT]);
    free(scores[TRACK_HAP_REF]);
    free(scores[TRACK_HAP_ALT]);
    free(ref_predictions);

    return ret;
}

/*
 * Annotate one input record and hand it to the writer.
 *
 * The annotation arrays are scratch owned by the caller, each holding at least record->n_alt
 * entries: one string per alternate allele, comma-joining that allele's overlapping genes and,
 * for the two haplotype fields, the copies it sits on.
 */
int process_variant_record(Model *models, faidx_t *fa, const Reference *ref, GeneReference *gene_reference,
                           const HapBuffer *buffer, SeqEditList *edits, int window_radius, int window_size,
                           regidx_t *gene_index, regitr_t *itr, GeneList *genes, const HapRecord *record,
                           kstring_t *spliceai, kstring_t *spliceai_hap, kstring_t *spliceai_tot,
                           VariantWriter *writer) {
    for (int i = 0; i < record->n_alt; i++) {
        spliceai[i].l = 0;
        spliceai_hap[i].l = 0;
        spliceai_tot[i].l = 0;
    }

    // Nothing to say about a variant no gene fully contains: write it through untouched.
    const int ref_len = strlen(record->ref);
    if (gene_regions_containing(gene_index, itr, record->chrom, record->pos, ref_len, genes) == 0) {
        return variant_writer_write(writer, record, spliceai, spliceai_hap, spliceai_tot);
    }

    for (int i = 0; i < record->n_alt; i++) {
        char *alt_allele = record->alt[i];

        // A genotype naming only the reference allele says the sample does not carry this one.
        if (record->has_gt && record->hap_mask[i] == 0) continue;

        if ('.' == alt_allele[0] || // Deletion
            '*' == alt_allele[0] || // Missing
            '<' == alt_allele[0] // <ID> string
        ) {
            log_warn("Unsupported alternate allele found: %s. Skipping prediction(s) for %s:%"PRIhts_pos, alt_allele, record->chrom, record->pos + 1);
            kputc('.', &spliceai[i]);
            continue;
        }

        const Annotations out = { &spliceai[i], &spliceai_hap[i], &spliceai_tot[i] };

        for (size_t g = 0; g < genes->n; g++) {
            if (score_allele_for_gene(models, fa, ref, gene_reference, buffer, edits, window_radius, window_size,
                                      record->chrom, genes->genes[g].name, record, i, &out) != EXIT_SUCCESS) {
                return EXIT_FAILURE;
            }
        }
    }

    return variant_writer_write(writer, record, spliceai, spliceai_hap, spliceai_tot);
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

    /*
     * How far apart two variants can be and still share a window. A scored position depends on
     * the BOUNDARY_SIZE bases either side of it, the scored region reaches window_radius from
     * the variant, and an allele may itself be window_radius long before it is refused.
     */
    HapBuffer *buffer;
    if (hap_buffer_open(reader, args.include_unphased, BOUNDARY_SIZE + 2 * window_radius, &buffer) != EXIT_SUCCESS) return EXIT_FAILURE;

    VariantWriter *writer;
    if (variant_writer_open(annotated_variants, reader, &writer) != EXIT_SUCCESS) return EXIT_FAILURE;

    regidx_t *gene_index = NULL;
    uint64_t regions_digest;
    if (gene_regions_build_regidx(gene_regions, &gene_index, &regions_digest, NULL) != EXIT_SUCCESS) return EXIT_FAILURE;

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
    SeqEditList edits;
    seq_edit_list_init(&edits);

    // One string per alternate allele per field, grown to fit the widest record seen.
    kstring_t *spliceai = NULL, *spliceai_hap = NULL, *spliceai_tot = NULL;
    int m_annotations = 0;

    int ret = EXIT_SUCCESS;
    const HapRecord *record;
    int read_status;
    while ((read_status = hap_buffer_next(buffer, &record)) == EXIT_SUCCESS) {
        // Heterozygous and unphased: the alleles are known, the copy each sits on is not, so
        // there is no haplotype to score and nothing truthful to report.
        if (record->drop) {
            log_warn("Skipping %s:%"PRIhts_pos": heterozygous genotype with unknown phase. Pass --include-unphased to score it on both haplotypes.",
                     record->chrom, record->pos + 1);
            continue;
        }

        if (record->n_alt > m_annotations) {
            kstring_t **fields[] = { &spliceai, &spliceai_hap, &spliceai_tot };
            for (size_t f = 0; f < sizeof(fields) / sizeof(fields[0]); f++) {
                kstring_t *grown = realloc(*fields[f], record->n_alt * sizeof(kstring_t));
                if (grown == NULL) {
                    log_fatal("Failed to allocate %zu bytes for annotations", record->n_alt * sizeof(kstring_t));
                    exit(EXIT_FAILURE);
                }
                memset(grown + m_annotations, 0, (record->n_alt - m_annotations) * sizeof(kstring_t));
                *fields[f] = grown;
            }
            m_annotations = record->n_alt;
        }

        if (process_variant_record(models, fa_in, &ref, &gene_reference, buffer, &edits, window_radius, window_size,
                                   gene_index, itr, &genes, record, spliceai, spliceai_hap, spliceai_tot, writer) != EXIT_SUCCESS) {
            ret = EXIT_FAILURE;
            break;
        }
    }

    if (read_status == EXIT_FAILURE) ret = EXIT_FAILURE;

    for (int i = 0; i < m_annotations; i++) {
        free(spliceai[i].s);
        free(spliceai_hap[i].s);
        free(spliceai_tot[i].s);
    }
    free(spliceai);
    free(spliceai_hap);
    free(spliceai_tot);

    seq_edit_list_destroy(&edits);
    gene_reference_destroy(&gene_reference);
    gene_list_destroy(&genes);
    regitr_destroy(itr);
    regidx_destroy(gene_index);
    variant_writer_close(writer);
    hap_buffer_close(buffer);
    variant_reader_close(reader);
    fai_destroy(fa_in);

    destroy_models(models);

    return ret;
}
