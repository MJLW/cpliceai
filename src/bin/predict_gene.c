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
#include "../haplotype.h"
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
    REQUIRED_STRING_ARG(output, "output", "TSV of splice sites found, where any track's scores exceed 0.001.")

#define OPTIONAL_ARGS \
    OPTIONAL_STRING_ARG(input_format, "auto", "--input-format", "vcf|tsv|auto", "Format of the variants file. Detected from the file itself by default")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(include_unphased, "--include-unphased", "Score heterozygous variants whose phase is unknown, placing them on both haplotypes") \
    BOOLEAN_ARG(ref_hapalt_only, "--ref-hapalt-only", "Write only the REF and HAP_ALT score columns, leaving out ALT and HAP_REF") \
    BOOLEAN_ARG(local, "--local", "Ignore genotype/phasing entirely: score every variant on its own against the reference genome, as if run with no GT at all. Nothing is ever dropped, --include-unphased has no additional effect, and only REF and ALT are written (HAP_REF/HAP_ALT would just repeat them) regardless of --ref-hapalt-only") \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>

/*
 * Four sequences are scored at every position of the gene, in this column order.
 *
 * REF is the reference genome and comes free from the reference scores. ALT is the variant
 * alone. HAP_REF is the copy of the chromosome the variant sits on with every other co-phased
 * variant applied but not this one, and HAP_ALT is that same copy complete. Reading ALT against
 * REF says what the variant does in isolation; reading HAP_ALT against HAP_REF says what it
 * adds to the molecule it is really on; reading HAP_ALT against REF says what that whole
 * molecule does.
 *
 * --ref-hapalt-only keeps the first and last of those, which is the pair that answers "what
 * does this sample's copy of the gene look like" without the isolated-variant working.
 *
 * --local keeps just REF and ALT: with no haplotype background ever assembled, HAP_REF and
 * HAP_ALT would always equal them anyway.
 */

/*
 * The complete haplotype of one gene is the same sequence whichever of its variants is being
 * reported on, so it is predicted once and reused. HAP_REF differs per variant - it is the
 * haplotype minus that variant - and cannot be.
 */
typedef struct {
    char   gene[FIELD_MAX_LEN];
    float *scores[HAP_COUNT];
} HapAltCache;

static void hap_alt_cache_init(HapAltCache *cache) {
    cache->gene[0] = '\0';
    for (int h = 0; h < HAP_COUNT; h++) cache->scores[h] = NULL;
}

static void hap_alt_cache_clear(HapAltCache *cache) {
    for (int h = 0; h < HAP_COUNT; h++) {
        free(cache->scores[h]);
        cache->scores[h] = NULL;
    }
    cache->gene[0] = '\0';
}

/*
 * The complete haplotype for one copy of this gene, predicted on first use.
 *
 * Returns NULL, having logged, if prediction fails.
 */
static const float *hap_alt_scores(Model *models, const GeneReference *gene, const char *chrom,
                                   const HapBuffer *buffer, SeqEditList *edits, int hap_index,
                                   HapAltCache *cache) {
    if (strncmp(cache->gene, gene->name, FIELD_MAX_LEN) != 0) {
        hap_alt_cache_clear(cache);
        snprintf(cache->gene, FIELD_MAX_LEN, "%s", gene->name);
    }

    if (cache->scores[hap_index] != NULL) return cache->scores[hap_index];

    hap_edits_collect(buffer, HAP_MASK(hap_index), chrom, (hts_pos_t) gene->start,
                      (hts_pos_t) gene->start, (hts_pos_t) gene->end, NULL, 0, edits);

    if (gene_reference_predict(models, gene, edits->edits, (int) edits->n, 0, (int64_t) gene->seq.l,
                               &cache->scores[hap_index]) != EXIT_SUCCESS) {
        return NULL;
    }

    return cache->scores[hap_index];
}

/*
 * Write one block: every position of the gene where any track crosses the threshold.
 *
 * A position is reported when any of the four says something, not just the reference pair, so a
 * site that only exists on the haplotype is not filtered away before it can be seen.
 */
void write_gene_scores(FILE *output, const GeneReference *gene, const float *alt, const float *hap_ref,
                       const float *hap_alt, bool ref_hapalt_only, bool local) {
    for (size_t i = 0; i < gene->seq.l; i++) {
        const float scores[][2] = {
            { gene->scores[i * NUM_SCORES + ACCEPTOR_POS], gene->scores[i * NUM_SCORES + DONOR_POS] },
            { alt[i * NUM_SCORES + ACCEPTOR_POS],          alt[i * NUM_SCORES + DONOR_POS] },
            { hap_ref[i * NUM_SCORES + ACCEPTOR_POS],      hap_ref[i * NUM_SCORES + DONOR_POS] },
            { hap_alt[i * NUM_SCORES + ACCEPTOR_POS],      hap_alt[i * NUM_SCORES + DONOR_POS] },
        };

        bool any = false;
        for (size_t t = 0; t < sizeof(scores) / sizeof(scores[0]) && !any; t++) {
            any = scores[t][0] >= SCORE_THRESHOLD || scores[t][1] >= SCORE_THRESHOLD;
        }
        if (!any) continue;

        fprintf(output, "%li", i + gene->start + 1);
        if (local) {
            /* HAP_REF and HAP_ALT are always just REF and ALT again under --local (there is no
               haplotype background), so leave them out rather than repeat them. */
            fprintf(output, "\t%f\t%f\t%f\t%f\n", scores[0][0], scores[0][1], scores[1][0], scores[1][1]);
        } else if (ref_hapalt_only) {
            fprintf(output, "\t%f\t%f\t%f\t%f\n", scores[0][0], scores[0][1], scores[3][0], scores[3][1]);
        } else {
            fprintf(output, "\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
                    scores[0][0], scores[0][1], scores[1][0], scores[1][1],
                    scores[2][0], scores[2][1], scores[3][0], scores[3][1]);
        }
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
    int64_t longest_gene;
    if (gene_regions_build_regidx(args.regions, &gene_index, &regions_digest, &longest_gene) != EXIT_SUCCESS) return EXIT_FAILURE;

    /*
     * Every position of the gene is scored, so any variant anywhere in it belongs to the same
     * haplotype and has to still be buffered when its neighbours are reached.
     */
    HapBuffer *buffer;
    if (hap_buffer_open(reader, args.include_unphased, args.local, longest_gene, &buffer) != EXIT_SUCCESS) return EXIT_FAILURE;

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
    SeqEditList edits;
    seq_edit_list_init(&edits);
    HapAltCache hap_alt_cache;
    hap_alt_cache_init(&hap_alt_cache);

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

        const int record_ref_len = strlen(record->ref);
        if (gene_regions_containing(gene_index, itr, record->chrom, record->pos, record_ref_len, &genes) == 0) {
            log_warn("No gene fully contains %s:%"PRIhts_pos". Skipping variant.", record->chrom, record->pos + 1);
            continue;
        }

        // One score block per (allele, gene, haplotype): each is an independent prediction.
        for (int i = 0; i < record->n_alt; i++) {
            // A genotype naming only the reference allele says the sample does not carry this one.
            if (record->has_gt && record->hap_mask[i] == 0) continue;

            for (size_t g = 0; g < genes.n; g++) {
                const Gene *gene = &genes.genes[g];

                if (strncmp(gene->name, current_gene.name, FIELD_MAX_LEN) != 0) {
                    if (gene_reference_update(record->chrom, gene->name, fa_in, &ref, &current_gene) != EXIT_SUCCESS) {
                        log_warn("Failed to find reference for gene %s. Skipping variant %s:%"PRIhts_pos".", gene->name, record->chrom, record->pos + 1);
                        continue;
                    }
                }

                const int64_t gene_len = (int64_t) current_gene.seq.l;
                const SeqEdit self = {
                    record->pos - (hts_pos_t) current_gene.start, record_ref_len,
                    record->alt[i], (int) strlen(record->alt[i]),
                };

                float *alt_predictions;
                if (gene_reference_predict(models, &current_gene, &self, 1, 0, gene_len, &alt_predictions) != EXIT_SUCCESS) {
                    continue;
                }

                /*
                 * The copies to report on. Without a genotype there is one pass carrying no copy
                 * at all, whose haplotype is the reference: HAP_REF and HAP_ALT then coincide
                 * with REF and ALT, and no extra prediction is needed for either.
                 */
                int haps[HAP_COUNT];
                int n_haps = 0;
                if (!record->has_gt) {
                    haps[n_haps++] = -1;
                } else {
                    for (int h = 0; h < HAP_COUNT; h++) {
                        if (record->hap_mask[i] & HAP_MASK(h)) haps[n_haps++] = h;
                    }
                }

                for (int h = 0; h < n_haps; h++) {
                    const int hap_index = haps[h];

                    const float *hap_ref_predictions = current_gene.scores;
                    const float *hap_alt_predictions = alt_predictions;
                    float *hap_ref_owned = NULL;

                    if (hap_index >= 0) {
                        hap_edits_collect(buffer, HAP_MASK(hap_index), record->chrom, (hts_pos_t) current_gene.start,
                                          (hts_pos_t) current_gene.start, (hts_pos_t) current_gene.end, record, i, &edits);

                        // With nothing else on this copy, its haplotype is the reference genome
                        // and its complete form is the variant alone; both are already to hand.
                        if (edits.n > 0) {
                            if (gene_reference_predict(models, &current_gene, edits.edits, (int) edits.n, 0, gene_len, &hap_ref_owned) != EXIT_SUCCESS) {
                                continue;
                            }
                            hap_ref_predictions = hap_ref_owned;

                            hap_alt_predictions = hap_alt_scores(models, &current_gene, record->chrom, buffer, &edits, hap_index, &hap_alt_cache);
                            if (hap_alt_predictions == NULL) {
                                free(hap_ref_owned);
                                continue;
                            }
                        }
                    }

                    fprintf(output, "#%s_%c_%li_%li:%s_%"PRIhts_pos"_%s_%s:HAP%s\n",
                            gene->name, current_gene.strand, current_gene.start, current_gene.end,
                            record->chrom, record->pos + 1, record->ref, record->alt[i],
                            hap_index < 0 ? "." : (hap_index == 0 ? "1" : "2"));

                    log_info("%s\t%li\t%li\t%s\t%c\t%i", record->chrom, current_gene.start, current_gene.end, current_gene.name, current_gene.strand, current_gene.end - current_gene.start);

                    write_gene_scores(output, &current_gene, alt_predictions, hap_ref_predictions,
                                      hap_alt_predictions, args.ref_hapalt_only, args.local);

                    free(hap_ref_owned);
                }

                free(alt_predictions);
            }
        }
    }

    if (read_status == EXIT_FAILURE) ret = EXIT_FAILURE;

    hap_alt_cache_clear(&hap_alt_cache);
    seq_edit_list_destroy(&edits);
    gene_reference_destroy(&current_gene);
    gene_list_destroy(&genes);
    regitr_destroy(itr);
    regidx_destroy(gene_index);
    hap_buffer_close(buffer);
    variant_reader_close(reader);
    fclose(output);
    fai_destroy(fa_in);

    destroy_models(models);

    return ret;
}
